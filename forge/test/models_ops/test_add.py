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


class Add0(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, add_input_0, add_input_1):
        add_output_1 = forge.op.Add("", add_input_0, add_input_1)
        return add_output_1


class Add1(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add1_const_1", shape=(1,), dtype=torch.float32)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add1_const_1"))
        return add_output_1


class Add2(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add2.weight_1", forge.Parameter(*(2048,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add2.weight_1"))
        return add_output_1


class Add3(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add3.weight_1", forge.Parameter(*(1536,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add3.weight_1"))
        return add_output_1


class Add4(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add4.weight_1", forge.Parameter(*(1024,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add4.weight_1"))
        return add_output_1


class Add5(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add5.weight_1",
            forge.Parameter(*(1500, 1024), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add5.weight_1"))
        return add_output_1


class Add6(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add6.weight_1", forge.Parameter(*(4096,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add6.weight_1"))
        return add_output_1


class Add7(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add7.weight_1", forge.Parameter(*(1280,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add7.weight_1"))
        return add_output_1


class Add8(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add8.weight_1",
            forge.Parameter(*(1500, 1280), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add8.weight_1"))
        return add_output_1


class Add9(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add9.weight_1", forge.Parameter(*(5120,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add9.weight_1"))
        return add_output_1


class Add10(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add10.weight_1", forge.Parameter(*(384,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add10.weight_1"))
        return add_output_1


class Add11(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add11.weight_1",
            forge.Parameter(*(1500, 384), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add11.weight_1"))
        return add_output_1


class Add12(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add12.weight_1", forge.Parameter(*(512,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add12.weight_1"))
        return add_output_1


class Add13(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add13.weight_1",
            forge.Parameter(*(1500, 512), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add13.weight_1"))
        return add_output_1


class Add14(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add14.weight_1", forge.Parameter(*(768,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add14.weight_1"))
        return add_output_1


class Add15(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add15.weight_1",
            forge.Parameter(*(1500, 768), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add15.weight_1"))
        return add_output_1


class Add16(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add16.weight_1", forge.Parameter(*(3072,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add16.weight_1"))
        return add_output_1


class Add17(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add17_const_1", shape=(1, 1, 2, 1), dtype=torch.float32)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add17_const_1"))
        return add_output_1


class Add18(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add18_const_1", shape=(2, 1, 7, 7), dtype=torch.float32)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add18_const_1"))
        return add_output_1


class Add19(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add19_const_1", shape=(1, 1, 39, 39), dtype=torch.float32)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add19_const_1"))
        return add_output_1


class Add20(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add20.weight_1", forge.Parameter(*(3129,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add20.weight_1"))
        return add_output_1


class Add21(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add21.weight_1", forge.Parameter(*(16384,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add21.weight_1"))
        return add_output_1


class Add22(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add22.weight_1", forge.Parameter(*(2,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add22.weight_1"))
        return add_output_1


class Add23(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add23.weight_1", forge.Parameter(*(128,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add23.weight_1"))
        return add_output_1


class Add24(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add24.weight_1", forge.Parameter(*(30000,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add24.weight_1"))
        return add_output_1


class Add25(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add25.weight_1", forge.Parameter(*(8192,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add25.weight_1"))
        return add_output_1


class Add26(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add26_const_1", shape=(1, 1, 256, 256), dtype=torch.float32)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add26_const_1"))
        return add_output_1


class Add27(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add27_const_1", shape=(1, 1, 1, 128), dtype=torch.float32)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add27_const_1"))
        return add_output_1


class Add28(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add28_const_1", shape=(1, 1, 1, 384), dtype=torch.float32)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add28_const_1"))
        return add_output_1


class Add29(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add29.weight_1", forge.Parameter(*(9,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add29.weight_1"))
        return add_output_1


class Add30(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add30.weight_1", forge.Parameter(*(30522,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add30.weight_1"))
        return add_output_1


class Add31(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add31.weight_1", forge.Parameter(*(51200,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add31.weight_1"))
        return add_output_1


class Add32(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add32.weight_1", forge.Parameter(*(119547,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add32.weight_1"))
        return add_output_1


class Add33(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add33.weight_1", forge.Parameter(*(28996,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add33.weight_1"))
        return add_output_1


class Add34(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add34.weight_1", forge.Parameter(*(1,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add34.weight_1"))
        return add_output_1


class Add35(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add35_const_1", shape=(1, 1, 6, 6), dtype=torch.float32)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add35_const_1"))
        return add_output_1


class Add36(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add36_const_1", shape=(1, 1, 10, 10), dtype=torch.float32)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add36_const_1"))
        return add_output_1


class Add37(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add37.weight_1", forge.Parameter(*(12288,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add37.weight_1"))
        return add_output_1


class Add38(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add38_const_1", shape=(1, 1, 334, 334), dtype=torch.float32)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add38_const_1"))
        return add_output_1


class Add39(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add39_const_1", shape=(1, 1, 7, 7), dtype=torch.float32)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add39_const_1"))
        return add_output_1


class Add40(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add40.weight_1", forge.Parameter(*(2560,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add40.weight_1"))
        return add_output_1


class Add41(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add41.weight_1", forge.Parameter(*(10240,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add41.weight_1"))
        return add_output_1


class Add42(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add42_const_1", shape=(1, 1, 32, 32), dtype=torch.float32)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add42_const_1"))
        return add_output_1


class Add43(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add43_const_1", shape=(1, 1, 4, 4), dtype=torch.float32)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add43_const_1"))
        return add_output_1


class Add44(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add44_const_0", shape=(1, 1, 256, 256), dtype=torch.float32)

    def forward(self, add_input_1):
        add_output_1 = forge.op.Add("", self.get_constant("add44_const_0"), add_input_1)
        return add_output_1


class Add45(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add45_const_1", shape=(1, 1, 128, 128), dtype=torch.float32)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add45_const_1"))
        return add_output_1


class Add46(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add46_const_1", shape=(1,), dtype=torch.int64)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add46_const_1"))
        return add_output_1


class Add47(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add47_const_1", shape=(1, 1, 12, 12), dtype=torch.float32)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add47_const_1"))
        return add_output_1


class Add48(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add48_const_1", shape=(1, 1, 11, 11), dtype=torch.float32)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add48_const_1"))
        return add_output_1


class Add49(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add49_const_1", shape=(1, 1, 13, 13), dtype=torch.float32)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add49_const_1"))
        return add_output_1


class Add50(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add50_const_1", shape=(1, 1, 5, 5), dtype=torch.float32)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add50_const_1"))
        return add_output_1


class Add51(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add51_const_1", shape=(1, 1, 29, 29), dtype=torch.float32)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add51_const_1"))
        return add_output_1


class Add52(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add52.weight_1", forge.Parameter(*(3584,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add52.weight_1"))
        return add_output_1


class Add53(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add53_const_1", shape=(1, 1, 35, 35), dtype=torch.float32)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add53_const_1"))
        return add_output_1


class Add54(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add54.weight_1", forge.Parameter(*(256,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add54.weight_1"))
        return add_output_1


class Add55(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add55.weight_1", forge.Parameter(*(896,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add55.weight_1"))
        return add_output_1


class Add56(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add56.weight_1", forge.Parameter(*(250002,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add56.weight_1"))
        return add_output_1


class Add57(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add57.weight_1", forge.Parameter(*(3,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add57.weight_1"))
        return add_output_1


class Add58(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add58_const_1", shape=(1, 1, 1, 61), dtype=torch.float32)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add58_const_1"))
        return add_output_1


class Add59(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add59_const_1", shape=(1, 16, 1, 61), dtype=torch.float32)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add59_const_1"))
        return add_output_1


class Add60(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add60_const_1", shape=(1, 8, 1, 61), dtype=torch.float32)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add60_const_1"))
        return add_output_1


class Add61(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add61_const_1", shape=(1, 6, 1, 61), dtype=torch.float32)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add61_const_1"))
        return add_output_1


class Add62(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add62_const_1", shape=(1, 12, 1, 61), dtype=torch.float32)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add62_const_1"))
        return add_output_1


class Add63(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add63.weight_1", forge.Parameter(*(48,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add63.weight_1"))
        return add_output_1


class Add64(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add64.weight_1", forge.Parameter(*(8,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add64.weight_1"))
        return add_output_1


class Add65(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add65.weight_1", forge.Parameter(*(96,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add65.weight_1"))
        return add_output_1


class Add66(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add66.weight_1", forge.Parameter(*(1000,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add66.weight_1"))
        return add_output_1


class Add67(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add67.weight_1", forge.Parameter(*(64,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add67.weight_1"))
        return add_output_1


class Add68(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add68.weight_1", forge.Parameter(*(12,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add68.weight_1"))
        return add_output_1


class Add69(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add69.weight_1", forge.Parameter(*(784,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add69.weight_1"))
        return add_output_1


class Add70(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add70.weight_1",
            forge.Parameter(*(1, 197, 768), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add70.weight_1"))
        return add_output_1


class Add71(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add71.weight_1",
            forge.Parameter(*(1, 197, 192), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add71.weight_1"))
        return add_output_1


class Add72(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add72.weight_1", forge.Parameter(*(192,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add72.weight_1"))
        return add_output_1


class Add73(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add73.weight_1",
            forge.Parameter(*(1, 197, 384), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add73.weight_1"))
        return add_output_1


class Add74(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add74.weight_1", forge.Parameter(*(144,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add74.weight_1"))
        return add_output_1


class Add75(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add75.weight_1", forge.Parameter(*(240,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add75.weight_1"))
        return add_output_1


class Add76(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add76.weight_1", forge.Parameter(*(288,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add76.weight_1"))
        return add_output_1


class Add77(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add77.weight_1", forge.Parameter(*(336,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add77.weight_1"))
        return add_output_1


class Add78(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add78.weight_1", forge.Parameter(*(432,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add78.weight_1"))
        return add_output_1


class Add79(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add79.weight_1", forge.Parameter(*(480,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add79.weight_1"))
        return add_output_1


class Add80(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add80.weight_1", forge.Parameter(*(528,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add80.weight_1"))
        return add_output_1


class Add81(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add81.weight_1", forge.Parameter(*(576,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add81.weight_1"))
        return add_output_1


class Add82(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add82.weight_1", forge.Parameter(*(624,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add82.weight_1"))
        return add_output_1


class Add83(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add83.weight_1", forge.Parameter(*(672,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add83.weight_1"))
        return add_output_1


class Add84(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add84.weight_1", forge.Parameter(*(720,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add84.weight_1"))
        return add_output_1


class Add85(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add85.weight_1", forge.Parameter(*(816,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add85.weight_1"))
        return add_output_1


class Add86(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add86.weight_1", forge.Parameter(*(864,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add86.weight_1"))
        return add_output_1


class Add87(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add87.weight_1", forge.Parameter(*(912,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add87.weight_1"))
        return add_output_1


class Add88(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add88.weight_1", forge.Parameter(*(960,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add88.weight_1"))
        return add_output_1


class Add89(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add89.weight_1", forge.Parameter(*(1008,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add89.weight_1"))
        return add_output_1


class Add90(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add90.weight_1", forge.Parameter(*(1056,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add90.weight_1"))
        return add_output_1


class Add91(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add91.weight_1", forge.Parameter(*(1104,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add91.weight_1"))
        return add_output_1


class Add92(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add92.weight_1", forge.Parameter(*(1152,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add92.weight_1"))
        return add_output_1


class Add93(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add93.weight_1", forge.Parameter(*(1200,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add93.weight_1"))
        return add_output_1


class Add94(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add94.weight_1", forge.Parameter(*(1248,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add94.weight_1"))
        return add_output_1


class Add95(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add95.weight_1", forge.Parameter(*(1296,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add95.weight_1"))
        return add_output_1


class Add96(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add96.weight_1", forge.Parameter(*(1344,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add96.weight_1"))
        return add_output_1


class Add97(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add97.weight_1", forge.Parameter(*(1392,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add97.weight_1"))
        return add_output_1


class Add98(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add98.weight_1", forge.Parameter(*(1440,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add98.weight_1"))
        return add_output_1


class Add99(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add99.weight_1", forge.Parameter(*(1488,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add99.weight_1"))
        return add_output_1


class Add100(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add100.weight_1", forge.Parameter(*(1584,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add100.weight_1"))
        return add_output_1


class Add101(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add101.weight_1", forge.Parameter(*(1632,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add101.weight_1"))
        return add_output_1


class Add102(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add102.weight_1", forge.Parameter(*(1680,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add102.weight_1"))
        return add_output_1


class Add103(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add103.weight_1", forge.Parameter(*(1728,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add103.weight_1"))
        return add_output_1


class Add104(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add104.weight_1", forge.Parameter(*(1776,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add104.weight_1"))
        return add_output_1


class Add105(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add105.weight_1", forge.Parameter(*(1824,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add105.weight_1"))
        return add_output_1


class Add106(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add106.weight_1", forge.Parameter(*(1872,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add106.weight_1"))
        return add_output_1


class Add107(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add107.weight_1", forge.Parameter(*(1920,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add107.weight_1"))
        return add_output_1


class Add108(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add108.weight_1", forge.Parameter(*(1968,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add108.weight_1"))
        return add_output_1


class Add109(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add109.weight_1", forge.Parameter(*(2016,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add109.weight_1"))
        return add_output_1


class Add110(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add110.weight_1", forge.Parameter(*(2064,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add110.weight_1"))
        return add_output_1


class Add111(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add111.weight_1", forge.Parameter(*(2112,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add111.weight_1"))
        return add_output_1


class Add112(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add112.weight_1", forge.Parameter(*(2160,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add112.weight_1"))
        return add_output_1


class Add113(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add113.weight_1", forge.Parameter(*(2208,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add113.weight_1"))
        return add_output_1


class Add114(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add114.weight_1", forge.Parameter(*(160,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add114.weight_1"))
        return add_output_1


class Add115(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add115.weight_1", forge.Parameter(*(224,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add115.weight_1"))
        return add_output_1


class Add116(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add116.weight_1", forge.Parameter(*(320,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add116.weight_1"))
        return add_output_1


class Add117(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add117.weight_1", forge.Parameter(*(352,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add117.weight_1"))
        return add_output_1


class Add118(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add118.weight_1", forge.Parameter(*(416,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add118.weight_1"))
        return add_output_1


class Add119(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add119.weight_1", forge.Parameter(*(448,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add119.weight_1"))
        return add_output_1


class Add120(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add120.weight_1", forge.Parameter(*(544,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add120.weight_1"))
        return add_output_1


class Add121(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add121.weight_1", forge.Parameter(*(608,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add121.weight_1"))
        return add_output_1


class Add122(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add122.weight_1", forge.Parameter(*(640,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
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
            "add127.weight_1", forge.Parameter(*(928,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add127.weight_1"))
        return add_output_1


class Add128(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add128.weight_1", forge.Parameter(*(992,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add128.weight_1"))
        return add_output_1


class Add129(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add129.weight_1", forge.Parameter(*(1088,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add129.weight_1"))
        return add_output_1


class Add130(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add130.weight_1", forge.Parameter(*(1120,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add130.weight_1"))
        return add_output_1


class Add131(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add131.weight_1", forge.Parameter(*(1184,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add131.weight_1"))
        return add_output_1


class Add132(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add132.weight_1", forge.Parameter(*(1216,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add132.weight_1"))
        return add_output_1


class Add133(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add133.weight_1", forge.Parameter(*(1312,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add133.weight_1"))
        return add_output_1


class Add134(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add134.weight_1", forge.Parameter(*(1376,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add134.weight_1"))
        return add_output_1


class Add135(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add135.weight_1", forge.Parameter(*(1408,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add135.weight_1"))
        return add_output_1


class Add136(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add136.weight_1", forge.Parameter(*(1472,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add136.weight_1"))
        return add_output_1


class Add137(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add137.weight_1", forge.Parameter(*(1504,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add137.weight_1"))
        return add_output_1


class Add138(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add138.weight_1", forge.Parameter(*(1568,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add138.weight_1"))
        return add_output_1


class Add139(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add139.weight_1", forge.Parameter(*(1600,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add139.weight_1"))
        return add_output_1


class Add140(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add140.weight_1", forge.Parameter(*(1664,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add140.weight_1"))
        return add_output_1


class Add141(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add141.weight_1", forge.Parameter(*(1696,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add141.weight_1"))
        return add_output_1


class Add142(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add142.weight_1", forge.Parameter(*(1760,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add142.weight_1"))
        return add_output_1


class Add143(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add143.weight_1", forge.Parameter(*(1792,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add143.weight_1"))
        return add_output_1


class Add144(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add144.weight_1", forge.Parameter(*(1856,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add144.weight_1"))
        return add_output_1


class Add145(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add145.weight_1", forge.Parameter(*(1888,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add145.weight_1"))
        return add_output_1


class Add146(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add146.weight_1", forge.Parameter(*(16,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add146.weight_1"))
        return add_output_1


class Add147(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add147.weight_1", forge.Parameter(*(32,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add147.weight_1"))
        return add_output_1


class Add148(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add148.weight_1", forge.Parameter(*(24,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add148.weight_1"))
        return add_output_1


class Add149(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add149.weight_1", forge.Parameter(*(56,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add149.weight_1"))
        return add_output_1


class Add150(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add150.weight_1", forge.Parameter(*(112,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add150.weight_1"))
        return add_output_1


class Add151(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add151.weight_1", forge.Parameter(*(272,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add151.weight_1"))
        return add_output_1


class Add152(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add152.weight_1", forge.Parameter(*(2688,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add152.weight_1"))
        return add_output_1


class Add153(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add153.weight_1", forge.Parameter(*(40,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add153.weight_1"))
        return add_output_1


class Add154(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add154.weight_1", forge.Parameter(*(80,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add154.weight_1"))
        return add_output_1


class Add155(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add155.weight_1", forge.Parameter(*(36,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add155.weight_1"))
        return add_output_1


class Add156(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add156.weight_1", forge.Parameter(*(72,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add156.weight_1"))
        return add_output_1


class Add157(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add157.weight_1", forge.Parameter(*(20,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add157.weight_1"))
        return add_output_1


class Add158(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add158.weight_1", forge.Parameter(*(60,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add158.weight_1"))
        return add_output_1


class Add159(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add159.weight_1", forge.Parameter(*(120,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add159.weight_1"))
        return add_output_1


class Add160(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add160.weight_1", forge.Parameter(*(100,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add160.weight_1"))
        return add_output_1


class Add161(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add161.weight_1", forge.Parameter(*(92,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add161.weight_1"))
        return add_output_1


class Add162(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add162.weight_1", forge.Parameter(*(208,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add162.weight_1"))
        return add_output_1


class Add163(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add163.weight_1", forge.Parameter(*(18,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add163.weight_1"))
        return add_output_1


class Add164(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add164.weight_1", forge.Parameter(*(44,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add164.weight_1"))
        return add_output_1


class Add165(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add165.weight_1", forge.Parameter(*(88,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add165.weight_1"))
        return add_output_1


class Add166(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add166.weight_1", forge.Parameter(*(176,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add166.weight_1"))
        return add_output_1


class Add167(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add167.weight_1", forge.Parameter(*(30,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add167.weight_1"))
        return add_output_1


class Add168(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add168.weight_1", forge.Parameter(*(49,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add168.weight_1"))
        return add_output_1


class Add169(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add169.weight_1", forge.Parameter(*(196,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add169.weight_1"))
        return add_output_1


class Add170(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add170.weight_1", forge.Parameter(*(11221,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add170.weight_1"))
        return add_output_1


class Add171(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add171.weight_1", forge.Parameter(*(21843,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add171.weight_1"))
        return add_output_1


class Add172(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add172.weight_1", forge.Parameter(*(1001,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add172.weight_1"))
        return add_output_1


class Add173(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add173.weight_1", forge.Parameter(*(200,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add173.weight_1"))
        return add_output_1


class Add174(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add174.weight_1", forge.Parameter(*(184,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add174.weight_1"))
        return add_output_1


class Add175(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add175.weight_1", forge.Parameter(*(322,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add175.weight_1"))
        return add_output_1


class Add176(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add176_const_1", shape=(1, 1, 1, 3025), dtype=torch.float32)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add176_const_1"))
        return add_output_1


class Add177(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add177.weight_1", forge.Parameter(*(261,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add177.weight_1"))
        return add_output_1


class Add178(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add178_const_1", shape=(1, 1, 1, 50176), dtype=torch.float32)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add178_const_1"))
        return add_output_1


class Add179(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add179_const_1", shape=(4096,), dtype=torch.float32)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add179_const_1"))
        return add_output_1


class Add180(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add180_const_1", shape=(2,), dtype=torch.float32)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add180_const_1"))
        return add_output_1


class Add181(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add181_const_1", shape=(64, 1, 49, 49), dtype=torch.float32)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add181_const_1"))
        return add_output_1


class Add182(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add182_const_1", shape=(16, 1, 49, 49), dtype=torch.float32)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add182_const_1"))
        return add_output_1


class Add183(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add183_const_1", shape=(4, 1, 49, 49), dtype=torch.float32)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add183_const_1"))
        return add_output_1


class Add184(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add184.weight_1",
            forge.Parameter(*(1, 197, 1024), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add184.weight_1"))
        return add_output_1


class Add185(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add185.weight_1", forge.Parameter(*(728,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add185.weight_1"))
        return add_output_1


class Add186(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add186_const_0", shape=(5880, 2), dtype=torch.float32)

    def forward(self, add_input_1):
        add_output_1 = forge.op.Add("", self.get_constant("add186_const_0"), add_input_1)
        return add_output_1


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (
        Add0,
        [((2, 1, 2048), torch.float32), ((2, 1, 2048), torch.float32)],
        {"model_name": ["pt_stereo_facebook_musicgen_large_music_generation_hf"], "pcc": 0.99},
    ),
    pytest.param(
        (
            Add0,
            [((2, 1, 2048), torch.float32), ((1, 2048), torch.float32)],
            {"model_name": ["pt_stereo_facebook_musicgen_large_music_generation_hf"], "pcc": 0.99},
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Add1,
        [((2, 13, 1), torch.float32)],
        {
            "model_name": [
                "pt_stereo_facebook_musicgen_large_music_generation_hf",
                "pt_stereo_facebook_musicgen_medium_music_generation_hf",
                "pt_stereo_facebook_musicgen_small_music_generation_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 12, 13, 13), torch.float32), ((2, 1, 1, 13), torch.float32)],
        {
            "model_name": [
                "pt_stereo_facebook_musicgen_large_music_generation_hf",
                "pt_stereo_facebook_musicgen_medium_music_generation_hf",
                "pt_stereo_facebook_musicgen_small_music_generation_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((2, 12, 13, 13), torch.float32), ((2, 12, 13, 13), torch.float32)],
        {
            "model_name": [
                "pt_stereo_facebook_musicgen_large_music_generation_hf",
                "pt_stereo_facebook_musicgen_medium_music_generation_hf",
                "pt_stereo_facebook_musicgen_small_music_generation_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((2, 13, 768), torch.float32), ((2, 13, 768), torch.float32)],
        {
            "model_name": [
                "pt_stereo_facebook_musicgen_large_music_generation_hf",
                "pt_stereo_facebook_musicgen_medium_music_generation_hf",
                "pt_stereo_facebook_musicgen_small_music_generation_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add2,
        [((2, 13, 2048), torch.float32)],
        {"model_name": ["pt_stereo_facebook_musicgen_large_music_generation_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((2, 1, 1, 13), torch.float32), ((2, 1, 1, 13), torch.float32)],
        {
            "model_name": [
                "pt_stereo_facebook_musicgen_large_music_generation_hf",
                "pt_stereo_facebook_musicgen_medium_music_generation_hf",
                "pt_stereo_facebook_musicgen_small_music_generation_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((2, 32, 1, 13), torch.float32), ((2, 1, 1, 13), torch.float32)],
        {"model_name": ["pt_stereo_facebook_musicgen_large_music_generation_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((2, 1, 1536), torch.float32), ((2, 1, 1536), torch.float32)],
        {"model_name": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"], "pcc": 0.99},
    ),
    pytest.param(
        (
            Add0,
            [((2, 1, 1536), torch.float32), ((1, 1536), torch.float32)],
            {"model_name": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"], "pcc": 0.99},
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Add3,
        [((2, 13, 1536), torch.float32)],
        {"model_name": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((2, 24, 1, 13), torch.float32), ((2, 1, 1, 13), torch.float32)],
        {"model_name": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((2, 1, 1024), torch.float32), ((2, 1, 1024), torch.float32)],
        {"model_name": ["pt_stereo_facebook_musicgen_small_music_generation_hf"], "pcc": 0.99},
    ),
    pytest.param(
        (
            Add0,
            [((2, 1, 1024), torch.float32), ((1, 1024), torch.float32)],
            {"model_name": ["pt_stereo_facebook_musicgen_small_music_generation_hf"], "pcc": 0.99},
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Add4,
        [((2, 13, 1024), torch.float32)],
        {"model_name": ["pt_stereo_facebook_musicgen_small_music_generation_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((2, 16, 1, 13), torch.float32), ((2, 1, 1, 13), torch.float32)],
        {"model_name": ["pt_stereo_facebook_musicgen_small_music_generation_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1, 1024), torch.float32), ((1, 1, 1024), torch.float32)],
        {
            "model_name": [
                "pt_whisper_openai_whisper_medium_speech_recognition_hf",
                "pt_t5_google_flan_t5_large_text_gen_hf",
                "pt_t5_t5_large_text_gen_hf",
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add4,
        [((1, 1, 1024), torch.float32)],
        {
            "model_name": [
                "pt_whisper_openai_whisper_medium_speech_recognition_hf",
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1024, 3000), torch.float32), ((1024, 1), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_medium_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1024, 1500), torch.float32), ((1024, 1), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_medium_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Add5,
        [((1, 1500, 1024), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_medium_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Add4,
        [((1, 1500, 1024), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_medium_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1500, 1024), torch.float32), ((1, 1500, 1024), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_medium_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Add6,
        [((1, 1500, 4096), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_medium_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Add6,
        [((1, 1, 4096), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_medium_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1, 1280), torch.float32), ((1, 1, 1280), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_large_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Add7,
        [((1, 1, 1280), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_large_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1280, 3000), torch.float32), ((1280, 1), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_large_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1280, 1500), torch.float32), ((1280, 1), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_large_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Add8,
        [((1, 1500, 1280), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_large_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Add7,
        [((1, 1500, 1280), torch.float32)],
        {
            "model_name": [
                "pt_whisper_openai_whisper_large_speech_recognition_hf",
                "pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1500, 1280), torch.float32), ((1, 1500, 1280), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_large_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Add9,
        [((1, 1500, 5120), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_large_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Add9,
        [((1, 1, 5120), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_large_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1, 384), torch.float32), ((1, 1, 384), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Add10,
        [((1, 1, 384), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 384, 3000), torch.float32), ((384, 1), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 384, 1500), torch.float32), ((384, 1), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Add11,
        [((1, 1500, 384), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Add10,
        [((1, 1500, 384), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1500, 384), torch.float32), ((1, 1500, 384), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Add3,
        [((1, 1500, 1536), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Add3,
        [((1, 1, 1536), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1, 512), torch.float32), ((1, 1, 512), torch.float32)],
        {
            "model_name": [
                "pt_whisper_openai_whisper_base_speech_recognition_hf",
                "pt_t5_t5_small_text_gen_hf",
                "pt_t5_google_flan_t5_small_text_gen_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add12,
        [((1, 1, 512), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_base_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 512, 3000), torch.float32), ((512, 1), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_base_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 512, 1500), torch.float32), ((512, 1), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_base_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Add13,
        [((1, 1500, 512), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_base_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Add12,
        [((1, 1500, 512), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_base_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1500, 512), torch.float32), ((1, 1500, 512), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_base_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Add2,
        [((1, 1500, 2048), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_base_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Add2,
        [((1, 1, 2048), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_base_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1, 768), torch.float32), ((1, 1, 768), torch.float32)],
        {
            "model_name": [
                "pt_whisper_openai_whisper_small_speech_recognition_hf",
                "pt_t5_t5_base_text_gen_hf",
                "pt_t5_google_flan_t5_base_text_gen_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add14,
        [((1, 1, 768), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_small_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 768, 3000), torch.float32), ((768, 1), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_small_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 768, 1500), torch.float32), ((768, 1), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_small_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Add15,
        [((1, 1500, 768), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_small_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Add14,
        [((1, 1500, 768), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_small_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1500, 768), torch.float32), ((1, 1500, 768), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_small_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Add16,
        [((1, 1500, 3072), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_small_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Add16,
        [((1, 1, 3072), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_small_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 2, 1280), torch.float32), ((1, 2, 1280), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf"], "pcc": 0.99},
    ),
    (
        Add7,
        [((1, 2, 1280), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf"], "pcc": 0.99},
    ),
    (
        Add17,
        [((1, 20, 2, 2), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf"], "pcc": 0.99},
    ),
    (
        Add9,
        [((1, 2, 5120), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((2, 7, 512), torch.float32), ((1, 7, 512), torch.float32)],
        {"model_name": ["pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text"], "pcc": 0.99},
    ),
    (
        Add12,
        [((2, 7, 512), torch.float32)],
        {"model_name": ["pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text"], "pcc": 0.99},
    ),
    (
        Add18,
        [((2, 8, 7, 7), torch.float32)],
        {"model_name": ["pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text"], "pcc": 0.99},
    ),
    (
        Add0,
        [((2, 1, 7, 7), torch.float32), ((2, 1, 7, 7), torch.float32)],
        {"model_name": ["pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text"], "pcc": 0.99},
    ),
    (
        Add0,
        [((2, 8, 7, 7), torch.float32), ((2, 1, 7, 7), torch.float32)],
        {"model_name": ["pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text"], "pcc": 0.99},
    ),
    (
        Add0,
        [((2, 7, 512), torch.float32), ((2, 7, 512), torch.float32)],
        {"model_name": ["pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text"], "pcc": 0.99},
    ),
    (
        Add2,
        [((2, 7, 2048), torch.float32)],
        {"model_name": ["pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text"], "pcc": 0.99},
    ),
    (
        Add1,
        [((1, 39, 1), torch.float32)],
        {
            "model_name": [
                "pt_deepseek_deepseek_math_7b_instruct_qa_hf",
                "DeepSeekWrapper_decoder",
                "pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf",
                "pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf",
                "pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm_hf",
                "pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 32, 39, 128), torch.float32), ((1, 32, 39, 128), torch.float32)],
        {"model_name": ["pt_deepseek_deepseek_math_7b_instruct_qa_hf", "DeepSeekWrapper_decoder"], "pcc": 0.99},
    ),
    (
        Add19,
        [((1, 32, 39, 39), torch.float32)],
        {"model_name": ["pt_deepseek_deepseek_math_7b_instruct_qa_hf", "DeepSeekWrapper_decoder"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 39, 4096), torch.float32), ((1, 39, 4096), torch.float32)],
        {"model_name": ["pt_deepseek_deepseek_math_7b_instruct_qa_hf", "DeepSeekWrapper_decoder"], "pcc": 0.99},
    ),
    (Add14, [((1, 204, 768), torch.float32)], {"model_name": ["pt_vilt_dandelin_vilt_b32_mlm_mlm_hf"], "pcc": 0.99}),
    (
        Add0,
        [((1, 12, 204, 204), torch.float32), ((1, 1, 1, 204), torch.float32)],
        {"model_name": ["pt_vilt_dandelin_vilt_b32_mlm_mlm_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 204, 768), torch.float32), ((1, 204, 768), torch.float32)],
        {"model_name": ["pt_vilt_dandelin_vilt_b32_mlm_mlm_hf"], "pcc": 0.99},
    ),
    (Add16, [((1, 204, 3072), torch.float32)], {"model_name": ["pt_vilt_dandelin_vilt_b32_mlm_mlm_hf"], "pcc": 0.99}),
    (
        Add14,
        [((1, 768), torch.float32)],
        {
            "model_name": [
                "pt_vilt_dandelin_vilt_b32_mlm_mlm_hf",
                "pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf",
                "pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf",
                "pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf",
                "pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf",
                "pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add14,
        [((1, 201, 768), torch.float32)],
        {"model_name": ["pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 12, 201, 201), torch.float32), ((1, 1, 1, 201), torch.float32)],
        {"model_name": ["pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 201, 768), torch.float32), ((1, 201, 768), torch.float32)],
        {"model_name": ["pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf"], "pcc": 0.99},
    ),
    (
        Add16,
        [((1, 201, 3072), torch.float32)],
        {"model_name": ["pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf"], "pcc": 0.99},
    ),
    (
        Add3,
        [((1, 1536), torch.float32)],
        {"model_name": ["pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf"], "pcc": 0.99},
    ),
    (
        Add20,
        [((1, 3129), torch.float32)],
        {"model_name": ["pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 128, 128), torch.float32), ((1, 128, 128), torch.float32)],
        {
            "model_name": [
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_albert_base_v2_token_cls_hf",
                "pt_albert_xxlarge_v2_token_cls_hf",
                "pt_albert_large_v1_token_cls_hf",
                "pt_albert_large_v2_token_cls_hf",
                "pt_albert_base_v1_token_cls_hf",
                "pt_albert_base_v1_mlm_hf",
                "pt_albert_xlarge_v2_token_cls_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
                "pt_albert_large_v2_mlm_hf",
                "pt_albert_base_v2_mlm_hf",
                "pt_albert_xlarge_v1_token_cls_hf",
                "pt_albert_xlarge_v1_mlm_hf",
                "pt_albert_large_v1_mlm_hf",
                "pt_albert_xxlarge_v1_mlm_hf",
                "pt_albert_xlarge_v2_mlm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add6,
        [((1, 128, 4096), torch.float32)],
        {
            "model_name": [
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_albert_xxlarge_v2_token_cls_hf",
                "pt_albert_large_v1_token_cls_hf",
                "pt_albert_large_v2_token_cls_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
                "pt_albert_large_v2_mlm_hf",
                "pt_albert_large_v1_mlm_hf",
                "pt_albert_xxlarge_v1_mlm_hf",
                "pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 64, 128, 128), torch.float32), ((1, 1, 1, 128), torch.float32)],
        {
            "model_name": [
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_albert_xxlarge_v2_token_cls_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
                "pt_albert_xxlarge_v1_mlm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 128, 4096), torch.float32), ((1, 128, 4096), torch.float32)],
        {
            "model_name": [
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_albert_xxlarge_v2_token_cls_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
                "pt_albert_xxlarge_v1_mlm_hf",
                "pt_mistral_mistralai_mistral_7b_v0_1_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add21,
        [((1, 128, 16384), torch.float32)],
        {
            "model_name": [
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_albert_xxlarge_v2_token_cls_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
                "pt_albert_xxlarge_v1_mlm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add22,
        [((1, 128, 2), torch.float32)],
        {
            "model_name": [
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_albert_base_v2_token_cls_hf",
                "pt_albert_xxlarge_v2_token_cls_hf",
                "pt_albert_large_v1_token_cls_hf",
                "pt_albert_large_v2_token_cls_hf",
                "pt_albert_base_v1_token_cls_hf",
                "pt_albert_xlarge_v2_token_cls_hf",
                "pt_albert_xlarge_v1_token_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add14,
        [((1, 128, 768), torch.float32)],
        {
            "model_name": [
                "pt_albert_base_v2_token_cls_hf",
                "pt_albert_base_v1_token_cls_hf",
                "pt_albert_base_v1_mlm_hf",
                "pt_albert_base_v2_mlm_hf",
                "pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf",
                "pt_bert_bert_base_uncased_mlm_hf",
                "pt_distilbert_distilbert_base_multilingual_cased_mlm_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
                "pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf",
                "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "pt_distilbert_distilbert_base_uncased_mlm_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader",
                "pt_roberta_xlm_roberta_base_mlm_hf",
                "pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 12, 128, 128), torch.float32), ((1, 1, 1, 128), torch.float32)],
        {
            "model_name": [
                "pt_albert_base_v2_token_cls_hf",
                "pt_albert_base_v1_token_cls_hf",
                "pt_albert_base_v1_mlm_hf",
                "pt_albert_base_v2_mlm_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader",
                "pt_roberta_xlm_roberta_base_mlm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 128, 768), torch.float32), ((1, 128, 768), torch.float32)],
        {
            "model_name": [
                "pt_albert_base_v2_token_cls_hf",
                "pt_albert_base_v1_token_cls_hf",
                "pt_albert_base_v1_mlm_hf",
                "pt_albert_base_v2_mlm_hf",
                "pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf",
                "pt_bert_bert_base_uncased_mlm_hf",
                "pt_distilbert_distilbert_base_multilingual_cased_mlm_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
                "pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf",
                "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "pt_distilbert_distilbert_base_uncased_mlm_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader",
                "pt_roberta_xlm_roberta_base_mlm_hf",
                "pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf",
                "pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add16,
        [((1, 128, 3072), torch.float32)],
        {
            "model_name": [
                "pt_albert_base_v2_token_cls_hf",
                "pt_albert_base_v1_token_cls_hf",
                "pt_albert_base_v1_mlm_hf",
                "pt_albert_base_v2_mlm_hf",
                "pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf",
                "pt_bert_bert_base_uncased_mlm_hf",
                "pt_distilbert_distilbert_base_multilingual_cased_mlm_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
                "pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf",
                "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "pt_distilbert_distilbert_base_uncased_mlm_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader",
                "pt_roberta_xlm_roberta_base_mlm_hf",
                "pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add4,
        [((1, 128, 1024), torch.float32)],
        {
            "model_name": [
                "pt_albert_large_v1_token_cls_hf",
                "pt_albert_large_v2_token_cls_hf",
                "pt_albert_large_v2_mlm_hf",
                "pt_albert_large_v1_mlm_hf",
                "pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 16, 128, 128), torch.float32), ((1, 1, 1, 128), torch.float32)],
        {
            "model_name": [
                "pt_albert_large_v1_token_cls_hf",
                "pt_albert_large_v2_token_cls_hf",
                "pt_albert_xlarge_v2_token_cls_hf",
                "pt_albert_large_v2_mlm_hf",
                "pt_albert_xlarge_v1_token_cls_hf",
                "pt_albert_xlarge_v1_mlm_hf",
                "pt_albert_large_v1_mlm_hf",
                "pt_albert_xlarge_v2_mlm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 128, 1024), torch.float32), ((1, 128, 1024), torch.float32)],
        {
            "model_name": [
                "pt_albert_large_v1_token_cls_hf",
                "pt_albert_large_v2_token_cls_hf",
                "pt_albert_large_v2_mlm_hf",
                "pt_albert_large_v1_mlm_hf",
                "pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add23,
        [((1, 128, 128), torch.float32)],
        {
            "model_name": [
                "pt_albert_base_v1_mlm_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
                "pt_albert_large_v2_mlm_hf",
                "pt_albert_base_v2_mlm_hf",
                "pt_albert_xlarge_v1_mlm_hf",
                "pt_albert_large_v1_mlm_hf",
                "pt_albert_xxlarge_v1_mlm_hf",
                "pt_albert_xlarge_v2_mlm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add24,
        [((1, 128, 30000), torch.float32)],
        {
            "model_name": [
                "pt_albert_base_v1_mlm_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
                "pt_albert_large_v2_mlm_hf",
                "pt_albert_base_v2_mlm_hf",
                "pt_albert_xlarge_v1_mlm_hf",
                "pt_albert_large_v1_mlm_hf",
                "pt_albert_xxlarge_v1_mlm_hf",
                "pt_albert_xlarge_v2_mlm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add2,
        [((1, 128, 2048), torch.float32)],
        {
            "model_name": [
                "pt_albert_xlarge_v2_token_cls_hf",
                "pt_albert_xlarge_v1_token_cls_hf",
                "pt_albert_xlarge_v1_mlm_hf",
                "pt_albert_xlarge_v2_mlm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 128, 2048), torch.float32), ((1, 128, 2048), torch.float32)],
        {
            "model_name": [
                "pt_albert_xlarge_v2_token_cls_hf",
                "pt_albert_xlarge_v1_token_cls_hf",
                "pt_albert_xlarge_v1_mlm_hf",
                "pt_albert_xlarge_v2_mlm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add25,
        [((1, 128, 8192), torch.float32)],
        {
            "model_name": [
                "pt_albert_xlarge_v2_token_cls_hf",
                "pt_albert_xlarge_v1_token_cls_hf",
                "pt_albert_xlarge_v1_mlm_hf",
                "pt_albert_xlarge_v2_mlm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 256, 1024), torch.float32), ((1, 256, 1024), torch.float32)],
        {
            "model_name": [
                "pt_bart_facebook_bart_large_mnli_seq_cls_hf",
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_mono_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
                "pt_opt_facebook_opt_350m_clm_hf",
                "pt_xglm_facebook_xglm_564m_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add4,
        [((1, 256, 1024), torch.float32)],
        {
            "model_name": [
                "pt_bart_facebook_bart_large_mnli_seq_cls_hf",
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_mono_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
                "pt_opt_facebook_opt_350m_clm_hf",
                "pt_xglm_facebook_xglm_564m_clm_hf",
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add26,
        [((1, 16, 256, 256), torch.float32)],
        {
            "model_name": [
                "pt_bart_facebook_bart_large_mnli_seq_cls_hf",
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_mono_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
                "pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1, 256, 256), torch.float32), ((1, 1, 256, 256), torch.float32)],
        {
            "model_name": [
                "pt_bart_facebook_bart_large_mnli_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_clm_hf",
                "pt_opt_facebook_opt_1_3b_clm_hf",
                "pt_opt_facebook_opt_125m_clm_hf",
                "pt_opt_facebook_opt_350m_clm_hf",
                "pt_xglm_facebook_xglm_1_7b_clm_hf",
                "pt_xglm_facebook_xglm_564m_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 16, 256, 256), torch.float32), ((1, 1, 256, 256), torch.float32)],
        {
            "model_name": [
                "pt_bart_facebook_bart_large_mnli_seq_cls_hf",
                "pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf",
                "pt_opt_facebook_opt_350m_clm_hf",
                "pt_xglm_facebook_xglm_1_7b_clm_hf",
                "pt_xglm_facebook_xglm_564m_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add6,
        [((1, 256, 4096), torch.float32)],
        {
            "model_name": [
                "pt_bart_facebook_bart_large_mnli_seq_cls_hf",
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_mono_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
                "pt_xglm_facebook_xglm_564m_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add27,
        [((1, 12, 128, 128), torch.float32)],
        {
            "model_name": [
                "pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf",
                "pt_bert_bert_base_uncased_mlm_hf",
                "pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf",
                "pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add22,
        [((1, 2), torch.float32)],
        {
            "model_name": [
                "pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf",
                "pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 384, 1024), torch.float32), ((1, 384, 1024), torch.float32)],
        {"model_name": ["pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf"], "pcc": 0.99},
    ),
    (
        Add4,
        [((1, 384, 1024), torch.float32)],
        {"model_name": ["pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf"], "pcc": 0.99},
    ),
    (
        Add28,
        [((1, 16, 384, 384), torch.float32)],
        {"model_name": ["pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf"], "pcc": 0.99},
    ),
    (
        Add6,
        [((1, 384, 4096), torch.float32)],
        {"model_name": ["pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((384, 1), torch.float32), ((1,), torch.float32)],
        {
            "model_name": [
                "pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf",
                "pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add27,
        [((1, 16, 128, 128), torch.float32)],
        {"model_name": ["pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf"], "pcc": 0.99},
    ),
    (
        Add29,
        [((1, 128, 9), torch.float32)],
        {
            "model_name": [
                "pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf",
                "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add30,
        [((1, 128, 30522), torch.float32)],
        {
            "model_name": ["pt_bert_bert_base_uncased_mlm_hf", "pt_distilbert_distilbert_base_uncased_mlm_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 256, 16, 32), torch.float32), ((1, 256, 16, 32), torch.float32)],
        {
            "model_name": [
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_mono_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add31,
        [((1, 256, 51200), torch.float32)],
        {
            "model_name": [
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_mono_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
                "pt_phi2_microsoft_phi_2_clm_hf",
                "pt_phi2_microsoft_phi_2_pytdml_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 12, 128, 128), torch.float32), ((1, 12, 128, 128), torch.float32)],
        {
            "model_name": [
                "pt_distilbert_distilbert_base_multilingual_cased_mlm_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
                "pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf",
                "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "pt_distilbert_distilbert_base_uncased_mlm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add32,
        [((1, 128, 119547), torch.float32)],
        {"model_name": ["pt_distilbert_distilbert_base_multilingual_cased_mlm_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 384, 768), torch.float32), ((1, 384, 768), torch.float32)],
        {"model_name": ["pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf"], "pcc": 0.99},
    ),
    (
        Add14,
        [((1, 384, 768), torch.float32)],
        {"model_name": ["pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 12, 384, 384), torch.float32), ((1, 12, 384, 384), torch.float32)],
        {"model_name": ["pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf"], "pcc": 0.99},
    ),
    (
        Add16,
        [((1, 384, 3072), torch.float32)],
        {"model_name": ["pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf"], "pcc": 0.99},
    ),
    (
        Add33,
        [((1, 128, 28996), torch.float32)],
        {"model_name": ["pt_distilbert_distilbert_base_cased_mlm_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((128, 1), torch.float32), ((1,), torch.float32)],
        {
            "model_name": [
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add34,
        [((1, 1), torch.float32)],
        {
            "model_name": [
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 71, 6, 64), torch.float32), ((1, 71, 6, 64), torch.float32)],
        {"model_name": ["pt_falcon_tiiuae_falcon_7b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1, 6, 64), torch.float32), ((1, 1, 6, 64), torch.float32)],
        {"model_name": ["pt_falcon_tiiuae_falcon_7b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (
        Add35,
        [((1, 71, 6, 6), torch.float32)],
        {"model_name": ["pt_falcon_tiiuae_falcon_7b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 6, 4544), torch.float32), ((1, 6, 4544), torch.float32)],
        {"model_name": ["pt_falcon_tiiuae_falcon_7b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (
        Add1,
        [((1, 10, 1), torch.float32)],
        {
            "model_name": [
                "pt_falcon3_tiiuae_falcon3_3b_base_clm_hf",
                "pt_falcon3_tiiuae_falcon3_1b_base_clm_hf",
                "pt_falcon3_tiiuae_falcon3_7b_base_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 12, 10, 256), torch.float32), ((1, 12, 10, 256), torch.float32)],
        {
            "model_name": ["pt_falcon3_tiiuae_falcon3_3b_base_clm_hf", "pt_falcon3_tiiuae_falcon3_7b_base_clm_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 4, 10, 256), torch.float32), ((1, 4, 10, 256), torch.float32)],
        {
            "model_name": [
                "pt_falcon3_tiiuae_falcon3_3b_base_clm_hf",
                "pt_falcon3_tiiuae_falcon3_1b_base_clm_hf",
                "pt_falcon3_tiiuae_falcon3_7b_base_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add36,
        [((1, 12, 10, 10), torch.float32)],
        {
            "model_name": ["pt_falcon3_tiiuae_falcon3_3b_base_clm_hf", "pt_falcon3_tiiuae_falcon3_7b_base_clm_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 10, 3072), torch.float32), ((1, 10, 3072), torch.float32)],
        {
            "model_name": ["pt_falcon3_tiiuae_falcon3_3b_base_clm_hf", "pt_falcon3_tiiuae_falcon3_7b_base_clm_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 8, 10, 256), torch.float32), ((1, 8, 10, 256), torch.float32)],
        {"model_name": ["pt_falcon3_tiiuae_falcon3_1b_base_clm_hf"], "pcc": 0.99},
    ),
    (
        Add36,
        [((1, 8, 10, 10), torch.float32)],
        {"model_name": ["pt_falcon3_tiiuae_falcon3_1b_base_clm_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 10, 2048), torch.float32), ((1, 10, 2048), torch.float32)],
        {"model_name": ["pt_falcon3_tiiuae_falcon3_1b_base_clm_hf"], "pcc": 0.99},
    ),
    (Add37, [((1, 334, 12288), torch.float32)], {"model_name": ["pt_fuyu_adept_fuyu_8b_qa_hf"], "pcc": 0.99}),
    (
        Add0,
        [((1, 64, 334, 32), torch.float32), ((1, 64, 334, 32), torch.float32)],
        {"model_name": ["pt_fuyu_adept_fuyu_8b_qa_hf"], "pcc": 0.99},
    ),
    (Add38, [((1, 64, 334, 334), torch.float32)], {"model_name": ["pt_fuyu_adept_fuyu_8b_qa_hf"], "pcc": 0.99}),
    (Add6, [((1, 334, 4096), torch.float32)], {"model_name": ["pt_fuyu_adept_fuyu_8b_qa_hf"], "pcc": 0.99}),
    (
        Add0,
        [((1, 334, 4096), torch.float32), ((1, 334, 4096), torch.float32)],
        {"model_name": ["pt_fuyu_adept_fuyu_8b_qa_hf"], "pcc": 0.99},
    ),
    (Add21, [((1, 334, 16384), torch.float32)], {"model_name": ["pt_fuyu_adept_fuyu_8b_qa_hf"], "pcc": 0.99}),
    (Add1, [((1, 7, 1), torch.float32)], {"model_name": ["pt_gemma_google_gemma_2b_text_gen_hf"], "pcc": 0.99}),
    (
        Add0,
        [((2048,), torch.float32), ((1,), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_2b_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 8, 7, 256), torch.float32), ((1, 8, 7, 256), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_2b_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1, 7, 256), torch.float32), ((1, 1, 7, 256), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_2b_text_gen_hf"], "pcc": 0.99},
    ),
    (Add39, [((1, 8, 7, 7), torch.float32)], {"model_name": ["pt_gemma_google_gemma_2b_text_gen_hf"], "pcc": 0.99}),
    (
        Add0,
        [((1, 7, 2048), torch.float32), ((1, 7, 2048), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_2b_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 256, 768), torch.float32), ((1, 256, 768), torch.float32)],
        {
            "model_name": [
                "pt_gpt2_gpt2_text_gen_hf",
                "pt_gptneo_eleutherai_gpt_neo_125m_clm_hf",
                "pt_opt_facebook_opt_125m_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((256, 768), torch.float32), ((768,), torch.float32)],
        {"model_name": ["pt_gpt2_gpt2_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 12, 256, 256), torch.float32), ((1, 1, 256, 256), torch.float32)],
        {
            "model_name": [
                "pt_gpt2_gpt2_text_gen_hf",
                "pt_gptneo_eleutherai_gpt_neo_125m_clm_hf",
                "pt_opt_facebook_opt_125m_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 12, 256, 256), torch.float32), ((1, 1, 1, 256), torch.float32)],
        {"model_name": ["pt_gpt2_gpt2_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Add14,
        [((256, 768), torch.float32)],
        {"model_name": ["pt_gpt2_gpt2_text_gen_hf", "pt_opt_facebook_opt_125m_clm_hf"], "pcc": 0.99},
    ),
    (
        Add16,
        [((256, 3072), torch.float32)],
        {"model_name": ["pt_gpt2_gpt2_text_gen_hf", "pt_opt_facebook_opt_125m_clm_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 256, 2560), torch.float32), ((1, 256, 2560), torch.float32)],
        {
            "model_name": [
                "pt_gptneo_eleutherai_gpt_neo_2_7b_clm_hf",
                "pt_phi2_microsoft_phi_2_clm_hf",
                "pt_phi2_microsoft_phi_2_pytdml_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 20, 256, 256), torch.float32), ((1, 1, 256, 256), torch.float32)],
        {"model_name": ["pt_gptneo_eleutherai_gpt_neo_2_7b_clm_hf"], "pcc": 0.99},
    ),
    (
        Add26,
        [((1, 20, 256, 256), torch.float32)],
        {"model_name": ["pt_gptneo_eleutherai_gpt_neo_2_7b_clm_hf"], "pcc": 0.99},
    ),
    (
        Add40,
        [((1, 256, 2560), torch.float32)],
        {
            "model_name": [
                "pt_gptneo_eleutherai_gpt_neo_2_7b_clm_hf",
                "pt_phi2_microsoft_phi_2_clm_hf",
                "pt_phi2_microsoft_phi_2_pytdml_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add41,
        [((1, 256, 10240), torch.float32)],
        {
            "model_name": [
                "pt_gptneo_eleutherai_gpt_neo_2_7b_clm_hf",
                "pt_phi2_microsoft_phi_2_clm_hf",
                "pt_phi2_microsoft_phi_2_pytdml_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 256, 2048), torch.float32), ((1, 256, 2048), torch.float32)],
        {
            "model_name": [
                "pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_clm_hf",
                "pt_opt_facebook_opt_1_3b_clm_hf",
                "pt_xglm_facebook_xglm_1_7b_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add2,
        [((1, 256, 2048), torch.float32)],
        {
            "model_name": [
                "pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf",
                "pt_opt_facebook_opt_1_3b_clm_hf",
                "pt_xglm_facebook_xglm_1_7b_clm_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add25,
        [((1, 256, 8192), torch.float32)],
        {"model_name": ["pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf", "pt_xglm_facebook_xglm_1_7b_clm_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 32, 768), torch.float32), ((1, 32, 768), torch.float32)],
        {
            "model_name": [
                "pt_gptneo_eleutherai_gpt_neo_125m_seq_cls_hf",
                "pt_opt_facebook_opt_125m_seq_cls_hf",
                "pt_opt_facebook_opt_125m_qa_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 12, 32, 32), torch.float32), ((1, 1, 32, 32), torch.float32)],
        {
            "model_name": [
                "pt_gptneo_eleutherai_gpt_neo_125m_seq_cls_hf",
                "pt_opt_facebook_opt_125m_seq_cls_hf",
                "pt_opt_facebook_opt_125m_qa_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add42,
        [((1, 12, 32, 32), torch.float32)],
        {"model_name": ["pt_gptneo_eleutherai_gpt_neo_125m_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Add14,
        [((1, 32, 768), torch.float32)],
        {
            "model_name": [
                "pt_gptneo_eleutherai_gpt_neo_125m_seq_cls_hf",
                "pt_opt_facebook_opt_125m_seq_cls_hf",
                "pt_opt_facebook_opt_125m_qa_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add16,
        [((1, 32, 3072), torch.float32)],
        {"model_name": ["pt_gptneo_eleutherai_gpt_neo_125m_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 32, 2560), torch.float32), ((1, 32, 2560), torch.float32)],
        {"model_name": ["pt_gptneo_eleutherai_gpt_neo_2_7b_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 20, 32, 32), torch.float32), ((1, 1, 32, 32), torch.float32)],
        {"model_name": ["pt_gptneo_eleutherai_gpt_neo_2_7b_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Add42,
        [((1, 20, 32, 32), torch.float32)],
        {"model_name": ["pt_gptneo_eleutherai_gpt_neo_2_7b_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Add40,
        [((1, 32, 2560), torch.float32)],
        {"model_name": ["pt_gptneo_eleutherai_gpt_neo_2_7b_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Add41,
        [((1, 32, 10240), torch.float32)],
        {"model_name": ["pt_gptneo_eleutherai_gpt_neo_2_7b_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Add26,
        [((1, 12, 256, 256), torch.float32)],
        {"model_name": ["pt_gptneo_eleutherai_gpt_neo_125m_clm_hf"], "pcc": 0.99},
    ),
    (
        Add14,
        [((1, 256, 768), torch.float32)],
        {
            "model_name": [
                "pt_gptneo_eleutherai_gpt_neo_125m_clm_hf",
                "pt_opt_facebook_opt_125m_clm_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add16,
        [((1, 256, 3072), torch.float32)],
        {"model_name": ["pt_gptneo_eleutherai_gpt_neo_125m_clm_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 32, 2048), torch.float32), ((1, 32, 2048), torch.float32)],
        {
            "model_name": [
                "pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf",
                "pt_opt_facebook_opt_1_3b_seq_cls_hf",
                "pt_opt_facebook_opt_1_3b_qa_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 16, 32, 32), torch.float32), ((1, 1, 32, 32), torch.float32)],
        {
            "model_name": [
                "pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf",
                "pt_opt_facebook_opt_350m_qa_hf",
                "pt_opt_facebook_opt_350m_seq_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add42,
        [((1, 16, 32, 32), torch.float32)],
        {"model_name": ["pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Add2,
        [((1, 32, 2048), torch.float32)],
        {
            "model_name": [
                "pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf",
                "pt_opt_facebook_opt_1_3b_seq_cls_hf",
                "pt_opt_facebook_opt_1_3b_qa_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add25,
        [((1, 32, 8192), torch.float32)],
        {"model_name": ["pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Add1,
        [((1, 4, 1), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 32, 4, 64), torch.float32), ((1, 32, 4, 64), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_seq_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 8, 4, 64), torch.float32), ((1, 8, 4, 64), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_seq_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add43,
        [((1, 32, 4, 4), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 4, 2048), torch.float32), ((1, 4, 2048), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_seq_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add1,
        [((1, 256, 1), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_clm_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 32, 256, 64), torch.float32), ((1, 32, 256, 64), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 8, 256, 64), torch.float32), ((1, 8, 256, 64), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add44,
        [((1, 1, 1, 256), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 32, 256, 256), torch.float32), ((1, 1, 256, 256), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_clm_hf",
                "pt_opt_facebook_opt_1_3b_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 32, 4, 128), torch.float32), ((1, 32, 4, 128), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 8, 4, 128), torch.float32), ((1, 8, 4, 128), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 4, 4096), torch.float32), ((1, 4, 4096), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add1,
        [((1, 128, 1), torch.float32)],
        {"model_name": ["pt_mistral_mistralai_mistral_7b_v0_1_clm_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 32, 128, 128), torch.float32), ((1, 32, 128, 128), torch.float32)],
        {"model_name": ["pt_mistral_mistralai_mistral_7b_v0_1_clm_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 8, 128, 128), torch.float32), ((1, 8, 128, 128), torch.float32)],
        {"model_name": ["pt_mistral_mistralai_mistral_7b_v0_1_clm_hf"], "pcc": 0.99},
    ),
    (
        Add45,
        [((1, 32, 128, 128), torch.float32)],
        {"model_name": ["pt_mistral_mistralai_mistral_7b_v0_1_clm_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 7, 768), torch.float32), ((1, 7, 768), torch.float32)],
        {"model_name": ["pt_nanogpt_financialsupport_nanogpt_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((7, 768), torch.float32), ((768,), torch.float32)],
        {"model_name": ["pt_nanogpt_financialsupport_nanogpt_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 12, 7, 7), torch.float32), ((1, 1, 7, 7), torch.float32)],
        {"model_name": ["pt_nanogpt_financialsupport_nanogpt_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 12, 7, 7), torch.float32), ((1, 1, 1, 7), torch.float32)],
        {"model_name": ["pt_nanogpt_financialsupport_nanogpt_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Add14,
        [((7, 768), torch.float32)],
        {"model_name": ["pt_nanogpt_financialsupport_nanogpt_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Add16,
        [((7, 3072), torch.float32)],
        {"model_name": ["pt_nanogpt_financialsupport_nanogpt_text_gen_hf"], "pcc": 0.99},
    ),
    pytest.param(
        (
            Add46,
            [((1, 256), torch.int64)],
            {
                "model_name": [
                    "pt_opt_facebook_opt_1_3b_clm_hf",
                    "pt_opt_facebook_opt_125m_clm_hf",
                    "pt_opt_facebook_opt_350m_clm_hf",
                ],
                "pcc": 0.99,
            },
        ),
        marks=[
            pytest.mark.xfail(
                reason="TypeError: Dtype mismatch: framework_model.dtype=torch.int64, compiled_model.dtype=torch.int32"
            )
        ],
    ),
    (Add25, [((256, 8192), torch.float32)], {"model_name": ["pt_opt_facebook_opt_1_3b_clm_hf"], "pcc": 0.99}),
    (Add2, [((256, 2048), torch.float32)], {"model_name": ["pt_opt_facebook_opt_1_3b_clm_hf"], "pcc": 0.99}),
    (
        Add0,
        [((256, 2048), torch.float32), ((256, 2048), torch.float32)],
        {"model_name": ["pt_opt_facebook_opt_1_3b_clm_hf"], "pcc": 0.99},
    ),
    pytest.param(
        (
            Add46,
            [((1, 32), torch.int64)],
            {
                "model_name": [
                    "pt_opt_facebook_opt_1_3b_seq_cls_hf",
                    "pt_opt_facebook_opt_1_3b_qa_hf",
                    "pt_opt_facebook_opt_350m_qa_hf",
                    "pt_opt_facebook_opt_125m_seq_cls_hf",
                    "pt_opt_facebook_opt_350m_seq_cls_hf",
                    "pt_opt_facebook_opt_125m_qa_hf",
                ],
                "pcc": 0.99,
            },
        ),
        marks=[
            pytest.mark.xfail(
                reason="TypeError: Dtype mismatch: framework_model.dtype=torch.int64, compiled_model.dtype=torch.int32"
            )
        ],
    ),
    (
        Add0,
        [((1, 1, 32, 32), torch.float32), ((1, 1, 32, 32), torch.float32)],
        {
            "model_name": [
                "pt_opt_facebook_opt_1_3b_seq_cls_hf",
                "pt_opt_facebook_opt_1_3b_qa_hf",
                "pt_opt_facebook_opt_350m_qa_hf",
                "pt_opt_facebook_opt_125m_seq_cls_hf",
                "pt_opt_facebook_opt_350m_seq_cls_hf",
                "pt_opt_facebook_opt_125m_qa_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 32, 32, 32), torch.float32), ((1, 1, 32, 32), torch.float32)],
        {"model_name": ["pt_opt_facebook_opt_1_3b_seq_cls_hf", "pt_opt_facebook_opt_1_3b_qa_hf"], "pcc": 0.99},
    ),
    (
        Add25,
        [((32, 8192), torch.float32)],
        {"model_name": ["pt_opt_facebook_opt_1_3b_seq_cls_hf", "pt_opt_facebook_opt_1_3b_qa_hf"], "pcc": 0.99},
    ),
    (
        Add2,
        [((32, 2048), torch.float32)],
        {"model_name": ["pt_opt_facebook_opt_1_3b_seq_cls_hf", "pt_opt_facebook_opt_1_3b_qa_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((32, 2048), torch.float32), ((32, 2048), torch.float32)],
        {"model_name": ["pt_opt_facebook_opt_1_3b_seq_cls_hf", "pt_opt_facebook_opt_1_3b_qa_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((32, 1), torch.float32), ((1,), torch.float32)],
        {
            "model_name": [
                "pt_opt_facebook_opt_1_3b_qa_hf",
                "pt_opt_facebook_opt_350m_qa_hf",
                "pt_opt_facebook_opt_125m_qa_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 32, 1024), torch.float32), ((1, 32, 1024), torch.float32)],
        {"model_name": ["pt_opt_facebook_opt_350m_qa_hf", "pt_opt_facebook_opt_350m_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Add4,
        [((1, 32, 1024), torch.float32)],
        {"model_name": ["pt_opt_facebook_opt_350m_qa_hf", "pt_opt_facebook_opt_350m_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Add6,
        [((32, 4096), torch.float32)],
        {"model_name": ["pt_opt_facebook_opt_350m_qa_hf", "pt_opt_facebook_opt_350m_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Add4,
        [((32, 1024), torch.float32)],
        {"model_name": ["pt_opt_facebook_opt_350m_qa_hf", "pt_opt_facebook_opt_350m_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((32, 1024), torch.float32), ((32, 1024), torch.float32)],
        {"model_name": ["pt_opt_facebook_opt_350m_qa_hf", "pt_opt_facebook_opt_350m_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Add16,
        [((32, 3072), torch.float32)],
        {"model_name": ["pt_opt_facebook_opt_125m_seq_cls_hf", "pt_opt_facebook_opt_125m_qa_hf"], "pcc": 0.99},
    ),
    (
        Add14,
        [((32, 768), torch.float32)],
        {"model_name": ["pt_opt_facebook_opt_125m_seq_cls_hf", "pt_opt_facebook_opt_125m_qa_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((32, 768), torch.float32), ((32, 768), torch.float32)],
        {"model_name": ["pt_opt_facebook_opt_125m_seq_cls_hf", "pt_opt_facebook_opt_125m_qa_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((256, 768), torch.float32), ((256, 768), torch.float32)],
        {"model_name": ["pt_opt_facebook_opt_125m_clm_hf"], "pcc": 0.99},
    ),
    (Add6, [((256, 4096), torch.float32)], {"model_name": ["pt_opt_facebook_opt_350m_clm_hf"], "pcc": 0.99}),
    (Add4, [((256, 1024), torch.float32)], {"model_name": ["pt_opt_facebook_opt_350m_clm_hf"], "pcc": 0.99}),
    (
        Add0,
        [((256, 1024), torch.float32), ((256, 1024), torch.float32)],
        {"model_name": ["pt_opt_facebook_opt_350m_clm_hf"], "pcc": 0.99},
    ),
    (
        Add40,
        [((1, 12, 2560), torch.float32)],
        {
            "model_name": ["pt_phi2_microsoft_phi_2_pytdml_token_cls_hf", "pt_phi2_microsoft_phi_2_token_cls_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 32, 12, 32), torch.float32), ((1, 32, 12, 32), torch.float32)],
        {
            "model_name": ["pt_phi2_microsoft_phi_2_pytdml_token_cls_hf", "pt_phi2_microsoft_phi_2_token_cls_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Add47,
        [((1, 32, 12, 12), torch.float32)],
        {
            "model_name": ["pt_phi2_microsoft_phi_2_pytdml_token_cls_hf", "pt_phi2_microsoft_phi_2_token_cls_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Add41,
        [((1, 12, 10240), torch.float32)],
        {
            "model_name": ["pt_phi2_microsoft_phi_2_pytdml_token_cls_hf", "pt_phi2_microsoft_phi_2_token_cls_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 12, 2560), torch.float32), ((1, 12, 2560), torch.float32)],
        {
            "model_name": ["pt_phi2_microsoft_phi_2_pytdml_token_cls_hf", "pt_phi2_microsoft_phi_2_token_cls_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Add22,
        [((1, 12, 2), torch.float32)],
        {
            "model_name": ["pt_phi2_microsoft_phi_2_pytdml_token_cls_hf", "pt_phi2_microsoft_phi_2_token_cls_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 32, 256, 32), torch.float32), ((1, 32, 256, 32), torch.float32)],
        {"model_name": ["pt_phi2_microsoft_phi_2_clm_hf", "pt_phi2_microsoft_phi_2_pytdml_clm_hf"], "pcc": 0.99},
    ),
    (
        Add26,
        [((1, 32, 256, 256), torch.float32)],
        {
            "model_name": [
                "pt_phi2_microsoft_phi_2_clm_hf",
                "pt_phi2_microsoft_phi_2_pytdml_clm_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add40,
        [((1, 11, 2560), torch.float32)],
        {
            "model_name": ["pt_phi2_microsoft_phi_2_seq_cls_hf", "pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 32, 11, 32), torch.float32), ((1, 32, 11, 32), torch.float32)],
        {
            "model_name": ["pt_phi2_microsoft_phi_2_seq_cls_hf", "pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Add48,
        [((1, 32, 11, 11), torch.float32)],
        {
            "model_name": ["pt_phi2_microsoft_phi_2_seq_cls_hf", "pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Add41,
        [((1, 11, 10240), torch.float32)],
        {
            "model_name": ["pt_phi2_microsoft_phi_2_seq_cls_hf", "pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 11, 2560), torch.float32), ((1, 11, 2560), torch.float32)],
        {
            "model_name": ["pt_phi2_microsoft_phi_2_seq_cls_hf", "pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 32, 256, 96), torch.float32), ((1, 32, 256, 96), torch.float32)],
        {"model_name": ["pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 256, 3072), torch.float32), ((1, 256, 3072), torch.float32)],
        {"model_name": ["pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf"], "pcc": 0.99},
    ),
    (
        Add1,
        [((1, 13, 1), torch.float32)],
        {"model_name": ["pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 32, 13, 96), torch.float32), ((1, 32, 13, 96), torch.float32)],
        {"model_name": ["pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf"], "pcc": 0.99},
    ),
    (
        Add49,
        [((1, 32, 13, 13), torch.float32)],
        {"model_name": ["pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 13, 3072), torch.float32), ((1, 13, 3072), torch.float32)],
        {"model_name": ["pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf"], "pcc": 0.99},
    ),
    (
        Add22,
        [((1, 13, 2), torch.float32)],
        {"model_name": ["pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf"], "pcc": 0.99},
    ),
    (
        Add1,
        [((1, 5, 1), torch.float32)],
        {"model_name": ["pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 32, 5, 96), torch.float32), ((1, 32, 5, 96), torch.float32)],
        {"model_name": ["pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Add50,
        [((1, 32, 5, 5), torch.float32)],
        {"model_name": ["pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 5, 3072), torch.float32), ((1, 5, 3072), torch.float32)],
        {"model_name": ["pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf"], "pcc": 0.99},
    ),
    (Add1, [((1, 6, 1), torch.float32)], {"model_name": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"], "pcc": 0.99}),
    (Add4, [((1, 6, 1024), torch.float32)], {"model_name": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"], "pcc": 0.99}),
    (
        Add0,
        [((1, 16, 6, 64), torch.float32), ((1, 16, 6, 64), torch.float32)],
        {"model_name": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"], "pcc": 0.99},
    ),
    (Add35, [((1, 16, 6, 6), torch.float32)], {"model_name": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"], "pcc": 0.99}),
    (
        Add0,
        [((1, 6, 1024), torch.float32), ((1, 6, 1024), torch.float32)],
        {"model_name": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Add1,
        [((1, 29, 1), torch.float32)],
        {
            "model_name": [
                "pt_qwen1_5_qwen_qwen1_5_0_5b_chat_clm_hf",
                "pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf",
                "pt_qwen_v2_qwen_qwen2_5_7b_clm_hf",
                "pt_qwen_v2_qwen_qwen2_5_3b_clm_hf",
                "pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (Add4, [((1, 29, 1024), torch.float32)], {"model_name": ["pt_qwen1_5_qwen_qwen1_5_0_5b_chat_clm_hf"], "pcc": 0.99}),
    (
        Add0,
        [((1, 16, 29, 64), torch.float32), ((1, 16, 29, 64), torch.float32)],
        {"model_name": ["pt_qwen1_5_qwen_qwen1_5_0_5b_chat_clm_hf"], "pcc": 0.99},
    ),
    (
        Add51,
        [((1, 16, 29, 29), torch.float32)],
        {"model_name": ["pt_qwen1_5_qwen_qwen1_5_0_5b_chat_clm_hf", "pt_qwen_v2_qwen_qwen2_5_3b_clm_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 29, 1024), torch.float32), ((1, 29, 1024), torch.float32)],
        {"model_name": ["pt_qwen1_5_qwen_qwen1_5_0_5b_chat_clm_hf"], "pcc": 0.99},
    ),
    (
        Add1,
        [((1, 35, 1), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_instruct_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add52,
        [((1, 35, 3584), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_instruct_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 28, 35, 128), torch.float32), ((1, 28, 35, 128), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_instruct_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add12,
        [((1, 35, 512), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_instruct_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 4, 35, 128), torch.float32), ((1, 4, 35, 128), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_instruct_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add53,
        [((1, 28, 35, 35), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_instruct_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 35, 3584), torch.float32), ((1, 35, 3584), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_instruct_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add3,
        [((1, 35, 1536), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 12, 35, 128), torch.float32), ((1, 12, 35, 128), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add54,
        [((1, 35, 256), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 2, 35, 128), torch.float32), ((1, 2, 35, 128), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add53,
        [((1, 12, 35, 35), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 35, 1536), torch.float32), ((1, 35, 1536), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add2,
        [((1, 35, 2048), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 16, 35, 128), torch.float32), ((1, 16, 35, 128), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add53,
        [((1, 16, 35, 35), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 35, 2048), torch.float32), ((1, 35, 2048), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add55,
        [((1, 35, 896), torch.float32)],
        {"model_name": ["pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 14, 35, 64), torch.float32), ((1, 14, 35, 64), torch.float32)],
        {"model_name": ["pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Add23,
        [((1, 35, 128), torch.float32)],
        {"model_name": ["pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 2, 35, 64), torch.float32), ((1, 2, 35, 64), torch.float32)],
        {"model_name": ["pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Add53,
        [((1, 14, 35, 35), torch.float32)],
        {"model_name": ["pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 35, 896), torch.float32), ((1, 35, 896), torch.float32)],
        {"model_name": ["pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf"], "pcc": 0.99},
    ),
    (Add3, [((1, 29, 1536), torch.float32)], {"model_name": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99}),
    (
        Add0,
        [((1, 12, 29, 128), torch.float32), ((1, 12, 29, 128), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Add54,
        [((1, 29, 256), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf", "pt_qwen_v2_qwen_qwen2_5_3b_clm_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 2, 29, 128), torch.float32), ((1, 2, 29, 128), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf", "pt_qwen_v2_qwen_qwen2_5_3b_clm_hf"], "pcc": 0.99},
    ),
    (Add51, [((1, 12, 29, 29), torch.float32)], {"model_name": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99}),
    (
        Add0,
        [((1, 29, 1536), torch.float32), ((1, 29, 1536), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Add3,
        [((1, 39, 1536), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 12, 39, 128), torch.float32), ((1, 12, 39, 128), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (
        Add54,
        [((1, 39, 256), torch.float32)],
        {
            "model_name": [
                "pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf",
                "pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 2, 39, 128), torch.float32), ((1, 2, 39, 128), torch.float32)],
        {
            "model_name": [
                "pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf",
                "pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add19,
        [((1, 12, 39, 39), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 39, 1536), torch.float32), ((1, 39, 1536), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (
        Add52,
        [((1, 39, 3584), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 28, 39, 128), torch.float32), ((1, 28, 39, 128), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (
        Add12,
        [((1, 39, 512), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 4, 39, 128), torch.float32), ((1, 4, 39, 128), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (
        Add19,
        [((1, 28, 39, 39), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 39, 3584), torch.float32), ((1, 39, 3584), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (Add52, [((1, 29, 3584), torch.float32)], {"model_name": ["pt_qwen_v2_qwen_qwen2_5_7b_clm_hf"], "pcc": 0.99}),
    (
        Add0,
        [((1, 28, 29, 128), torch.float32), ((1, 28, 29, 128), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_7b_clm_hf"], "pcc": 0.99},
    ),
    (Add12, [((1, 29, 512), torch.float32)], {"model_name": ["pt_qwen_v2_qwen_qwen2_5_7b_clm_hf"], "pcc": 0.99}),
    (
        Add0,
        [((1, 4, 29, 128), torch.float32), ((1, 4, 29, 128), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_7b_clm_hf"], "pcc": 0.99},
    ),
    (Add51, [((1, 28, 29, 29), torch.float32)], {"model_name": ["pt_qwen_v2_qwen_qwen2_5_7b_clm_hf"], "pcc": 0.99}),
    (
        Add0,
        [((1, 29, 3584), torch.float32), ((1, 29, 3584), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_7b_clm_hf"], "pcc": 0.99},
    ),
    (Add2, [((1, 29, 2048), torch.float32)], {"model_name": ["pt_qwen_v2_qwen_qwen2_5_3b_clm_hf"], "pcc": 0.99}),
    (
        Add0,
        [((1, 16, 29, 128), torch.float32), ((1, 16, 29, 128), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_3b_clm_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 29, 2048), torch.float32), ((1, 29, 2048), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_3b_clm_hf"], "pcc": 0.99},
    ),
    (
        Add2,
        [((1, 39, 2048), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 16, 39, 128), torch.float32), ((1, 16, 39, 128), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (
        Add19,
        [((1, 16, 39, 39), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 39, 2048), torch.float32), ((1, 39, 2048), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (Add55, [((1, 29, 896), torch.float32)], {"model_name": ["pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"], "pcc": 0.99}),
    (
        Add0,
        [((1, 14, 29, 64), torch.float32), ((1, 14, 29, 64), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"], "pcc": 0.99},
    ),
    (Add23, [((1, 29, 128), torch.float32)], {"model_name": ["pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"], "pcc": 0.99}),
    (
        Add0,
        [((1, 2, 29, 64), torch.float32), ((1, 2, 29, 64), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"], "pcc": 0.99},
    ),
    (Add51, [((1, 14, 29, 29), torch.float32)], {"model_name": ["pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"], "pcc": 0.99}),
    (
        Add0,
        [((1, 29, 896), torch.float32), ((1, 29, 896), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Add55,
        [((1, 39, 896), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 14, 39, 64), torch.float32), ((1, 14, 39, 64), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (
        Add23,
        [((1, 39, 128), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 2, 39, 64), torch.float32), ((1, 2, 39, 64), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (
        Add19,
        [((1, 14, 39, 39), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 39, 896), torch.float32), ((1, 39, 896), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf"], "pcc": 0.99},
    ),
    pytest.param(
        (
            Add46,
            [((1, 128), torch.int64)],
            {
                "model_name": [
                    "pt_roberta_xlm_roberta_base_mlm_hf",
                    "pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf",
                ],
                "pcc": 0.99,
            },
        ),
        marks=[
            pytest.mark.xfail(
                reason="TypeError: Dtype mismatch: framework_model.dtype=torch.int64, compiled_model.dtype=torch.int32"
            )
        ],
    ),
    (Add56, [((1, 128, 250002), torch.float32)], {"model_name": ["pt_roberta_xlm_roberta_base_mlm_hf"], "pcc": 0.99}),
    (
        Add57,
        [((1, 3), torch.float32)],
        {
            "model_name": [
                "pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf",
                "pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf",
                "pt_autoencoder_linear_img_enc_github",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 768, 1, 128), torch.float32), ((768, 1, 1), torch.float32)],
        {"model_name": ["pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 768, 128), torch.float32), ((768, 1), torch.float32)],
        {"model_name": ["pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 768, 128), torch.float32), ((1, 768, 128), torch.float32)],
        {"model_name": ["pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 3072, 1, 128), torch.float32), ((3072, 1, 1), torch.float32)],
        {"model_name": ["pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Add1,
        [((1, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_t5_google_flan_t5_large_text_gen_hf",
                "pt_t5_t5_large_text_gen_hf",
                "pt_t5_t5_small_text_gen_hf",
                "pt_t5_google_flan_t5_small_text_gen_hf",
                "pt_t5_t5_base_text_gen_hf",
                "pt_t5_google_flan_t5_base_text_gen_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 16, 1, 1), torch.float32), ((1, 16, 1, 1), torch.float32)],
        {"model_name": ["pt_t5_google_flan_t5_large_text_gen_hf", "pt_t5_t5_large_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Add1,
        [((1, 61, 1), torch.float32)],
        {
            "model_name": [
                "pt_t5_google_flan_t5_large_text_gen_hf",
                "pt_t5_t5_large_text_gen_hf",
                "pt_t5_t5_small_text_gen_hf",
                "pt_t5_google_flan_t5_small_text_gen_hf",
                "pt_t5_t5_base_text_gen_hf",
                "pt_t5_google_flan_t5_base_text_gen_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add58,
        [((1, 16, 61, 61), torch.float32)],
        {"model_name": ["pt_t5_google_flan_t5_large_text_gen_hf", "pt_t5_t5_large_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 16, 61, 61), torch.float32), ((1, 16, 61, 61), torch.float32)],
        {"model_name": ["pt_t5_google_flan_t5_large_text_gen_hf", "pt_t5_t5_large_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 61, 1024), torch.float32), ((1, 61, 1024), torch.float32)],
        {"model_name": ["pt_t5_google_flan_t5_large_text_gen_hf", "pt_t5_t5_large_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Add59,
        [((1, 16, 1, 61), torch.float32)],
        {"model_name": ["pt_t5_google_flan_t5_large_text_gen_hf", "pt_t5_t5_large_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 8, 1, 1), torch.float32), ((1, 8, 1, 1), torch.float32)],
        {"model_name": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99},
    ),
    (Add58, [((1, 8, 61, 61), torch.float32)], {"model_name": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99}),
    (
        Add0,
        [((1, 8, 61, 61), torch.float32), ((1, 8, 61, 61), torch.float32)],
        {"model_name": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 61, 512), torch.float32), ((1, 61, 512), torch.float32)],
        {"model_name": ["pt_t5_t5_small_text_gen_hf", "pt_t5_google_flan_t5_small_text_gen_hf"], "pcc": 0.99},
    ),
    (Add60, [((1, 8, 1, 61), torch.float32)], {"model_name": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99}),
    (
        Add0,
        [((1, 6, 1, 1), torch.float32), ((1, 6, 1, 1), torch.float32)],
        {"model_name": ["pt_t5_google_flan_t5_small_text_gen_hf"], "pcc": 0.99},
    ),
    (Add58, [((1, 6, 61, 61), torch.float32)], {"model_name": ["pt_t5_google_flan_t5_small_text_gen_hf"], "pcc": 0.99}),
    (
        Add0,
        [((1, 6, 61, 61), torch.float32), ((1, 6, 61, 61), torch.float32)],
        {"model_name": ["pt_t5_google_flan_t5_small_text_gen_hf"], "pcc": 0.99},
    ),
    (Add61, [((1, 6, 1, 61), torch.float32)], {"model_name": ["pt_t5_google_flan_t5_small_text_gen_hf"], "pcc": 0.99}),
    (
        Add0,
        [((1, 12, 1, 1), torch.float32), ((1, 12, 1, 1), torch.float32)],
        {"model_name": ["pt_t5_t5_base_text_gen_hf", "pt_t5_google_flan_t5_base_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Add58,
        [((1, 12, 61, 61), torch.float32)],
        {"model_name": ["pt_t5_t5_base_text_gen_hf", "pt_t5_google_flan_t5_base_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 12, 61, 61), torch.float32), ((1, 12, 61, 61), torch.float32)],
        {"model_name": ["pt_t5_t5_base_text_gen_hf", "pt_t5_google_flan_t5_base_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 61, 768), torch.float32), ((1, 61, 768), torch.float32)],
        {"model_name": ["pt_t5_t5_base_text_gen_hf", "pt_t5_google_flan_t5_base_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Add62,
        [((1, 12, 1, 61), torch.float32)],
        {"model_name": ["pt_t5_t5_base_text_gen_hf", "pt_t5_google_flan_t5_base_text_gen_hf"], "pcc": 0.99},
    ),
    (Add2, [((1024, 2048), torch.float32)], {"model_name": ["pt_nbeats_seasionality_basis_clm_hf"], "pcc": 0.99}),
    (Add63, [((1024, 48), torch.float32)], {"model_name": ["pt_nbeats_seasionality_basis_clm_hf"], "pcc": 0.99}),
    (
        Add0,
        [((1024, 24), torch.float32), ((1024, 24), torch.float32)],
        {
            "model_name": [
                "pt_nbeats_seasionality_basis_clm_hf",
                "pt_nbeats_trend_basis_clm_hf",
                "pt_nbeats_generic_basis_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1024, 1), torch.float32), ((1024, 24), torch.float32)],
        {
            "model_name": [
                "pt_nbeats_seasionality_basis_clm_hf",
                "pt_nbeats_trend_basis_clm_hf",
                "pt_nbeats_generic_basis_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1024, 72), torch.float32), ((1024, 72), torch.float32)],
        {"model_name": ["pt_nbeats_seasionality_basis_clm_hf"], "pcc": 0.99},
    ),
    (Add54, [((1024, 256), torch.float32)], {"model_name": ["pt_nbeats_trend_basis_clm_hf"], "pcc": 0.99}),
    (Add64, [((1024, 8), torch.float32)], {"model_name": ["pt_nbeats_trend_basis_clm_hf"], "pcc": 0.99}),
    (Add12, [((1024, 512), torch.float32)], {"model_name": ["pt_nbeats_generic_basis_clm_hf"], "pcc": 0.99}),
    (Add65, [((1024, 96), torch.float32)], {"model_name": ["pt_nbeats_generic_basis_clm_hf"], "pcc": 0.99}),
    (
        Add0,
        [((1, 64, 55, 55), torch.float32), ((64, 1, 1), torch.float32)],
        {"model_name": ["pt_alexnet_alexnet_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 192, 27, 27), torch.float32), ((192, 1, 1), torch.float32)],
        {"model_name": ["pt_alexnet_alexnet_img_cls_torchhub", "pt_rcnn_base_obj_det_torchvision_rect_0"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 384, 13, 13), torch.float32), ((384, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_alexnet_alexnet_img_cls_torchhub",
                "pt_alexnet_base_img_cls_osmr",
                "pt_rcnn_base_obj_det_torchvision_rect_0",
                "pt_yolox_yolox_tiny_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 256, 13, 13), torch.float32), ((256, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_alexnet_alexnet_img_cls_torchhub",
                "pt_alexnet_base_img_cls_osmr",
                "pt_rcnn_base_obj_det_torchvision_rect_0",
                "pt_yolox_yolox_nano_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add6,
        [((1, 4096), torch.float32)],
        {
            "model_name": [
                "pt_alexnet_alexnet_img_cls_torchhub",
                "pt_alexnet_base_img_cls_osmr",
                "pt_vgg_bn_vgg19_obj_det_osmr",
                "pt_vgg_19_obj_det_hf",
                "pt_vgg_vgg19_bn_obj_det_torchhub",
                "pt_vgg_vgg11_obj_det_osmr",
                "pt_vgg_bn_vgg19b_obj_det_osmr",
                "pt_vgg_vgg16_obj_det_osmr",
                "pt_vgg_vgg19_obj_det_osmr",
                "pt_vgg_vgg13_obj_det_osmr",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add66,
        [((1, 1000), torch.float32)],
        {
            "model_name": [
                "pt_alexnet_alexnet_img_cls_torchhub",
                "pt_alexnet_base_img_cls_osmr",
                "pt_deit_facebook_deit_base_patch16_224_img_cls_hf",
                "pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf",
                "pt_deit_facebook_deit_small_patch16_224_img_cls_hf",
                "pt_deit_facebook_deit_base_distilled_patch16_224_img_cls_hf",
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_googlenet_base_img_cls_torchvision",
                "pt_hrnet_hrnet_w18_pose_estimation_timm",
                "pt_hrnet_hrnet_w44_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w30_pose_estimation_osmr",
                "pt_hrnet_hrnet_w32_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnet_w64_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w48_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w44_pose_estimation_osmr",
                "pt_hrnet_hrnet_w40_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w40_pose_estimation_osmr",
                "pt_hrnet_hrnet_w48_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w32_pose_estimation_osmr",
                "pt_inception_v4_img_cls_timm",
                "pt_inception_v4_img_cls_osmr",
                "pt_mlp_mixer_mixer_b32_224_img_cls_timm",
                "pt_mlp_mixer_mixer_s16_224_img_cls_timm",
                "pt_mlp_mixer_mixer_s32_224_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_miil_img_cls_timm",
                "pt_mlp_mixer_mixer_l32_224_img_cls_timm",
                "pt_mlp_mixer_mixer_l16_224_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_img_cls_timm",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_regnet_facebook_regnet_y_040_img_cls_hf",
                "ResNetForImageClassification",
                "ResNet",
                "pt_resnet_50_img_cls_timm",
                "pt_resnext_resnext26_32x4d_img_cls_osmr",
                "pt_resnext_resnext101_64x4d_img_cls_osmr",
                "pt_resnext_resnext50_32x4d_img_cls_osmr",
                "pt_resnext_resnext50_32x4d_img_cls_torchhub",
                "pt_resnext_resnext14_32x4d_img_cls_osmr",
                "pt_resnext_resnext101_32x8d_img_cls_torchhub",
                "pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub",
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_vgg_bn_vgg19_obj_det_osmr",
                "pt_vgg_19_obj_det_hf",
                "pt_vgg_vgg19_bn_obj_det_torchhub",
                "pt_vgg_vgg11_obj_det_osmr",
                "pt_vgg_vgg19_bn_obj_det_timm",
                "pt_vgg_bn_vgg19b_obj_det_osmr",
                "pt_vgg_vgg16_obj_det_osmr",
                "pt_vgg_vgg19_obj_det_osmr",
                "pt_vgg_vgg13_obj_det_osmr",
                "pt_vit_google_vit_large_patch16_224_img_cls_hf",
                "pt_vit_google_vit_base_patch16_224_img_cls_hf",
                "pt_vovnet_ese_vovnet39b_obj_det_torchhub",
                "pt_vovnet_vovnet39_obj_det_osmr",
                "pt_vovnet_vovnet57_obj_det_osmr",
                "pt_vovnet_vovnet27s_obj_det_osmr",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_wideresnet_wide_resnet101_2_img_cls_torchvision",
                "pt_wideresnet_wide_resnet50_2_img_cls_timm",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
                "pt_wideresnet_wide_resnet50_2_img_cls_torchvision",
                "pt_xception_xception_img_cls_timm",
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception65_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 96, 54, 54), torch.float32), ((96, 1, 1), torch.float32)],
        {"model_name": ["pt_alexnet_base_img_cls_osmr"], "pcc": 0.99},
    ),
    (Add1, [((1, 96, 54, 54), torch.float32)], {"model_name": ["pt_alexnet_base_img_cls_osmr"], "pcc": 0.99}),
    (
        Add0,
        [((1, 256, 27, 27), torch.float32), ((256, 1, 1), torch.float32)],
        {"model_name": ["pt_alexnet_base_img_cls_osmr"], "pcc": 0.99},
    ),
    (Add1, [((1, 256, 27, 27), torch.float32)], {"model_name": ["pt_alexnet_base_img_cls_osmr"], "pcc": 0.99}),
    (Add23, [((1, 128), torch.float32)], {"model_name": ["pt_autoencoder_linear_img_enc_github"], "pcc": 0.99}),
    (Add67, [((1, 64), torch.float32)], {"model_name": ["pt_autoencoder_linear_img_enc_github"], "pcc": 0.99}),
    (Add68, [((1, 12), torch.float32)], {"model_name": ["pt_autoencoder_linear_img_enc_github"], "pcc": 0.99}),
    (Add69, [((1, 784), torch.float32)], {"model_name": ["pt_autoencoder_linear_img_enc_github"], "pcc": 0.99}),
    (
        Add0,
        [((1, 16, 28, 28), torch.float32), ((16, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_autoencoder_conv_img_enc_github",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 4, 14, 14), torch.float32), ((4, 1, 1), torch.float32)],
        {"model_name": ["pt_autoencoder_conv_img_enc_github"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 16, 14, 14), torch.float32), ((16, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_autoencoder_conv_img_enc_github",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1, 28, 28), torch.float32), ((1, 1, 1), torch.float32)],
        {"model_name": ["pt_autoencoder_conv_img_enc_github"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 768, 14, 14), torch.float32), ((768, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_deit_facebook_deit_base_patch16_224_img_cls_hf",
                "pt_deit_facebook_deit_base_distilled_patch16_224_img_cls_hf",
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_mlp_mixer_mixer_b16_224_miil_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_miil_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_in21k_img_cls_timm",
                "pt_vit_google_vit_base_patch16_224_img_cls_hf",
                "pt_vovnet_ese_vovnet39b_obj_det_torchhub",
                "pt_vovnet_vovnet39_obj_det_osmr",
                "pt_vovnet_vovnet57_obj_det_osmr",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add70,
        [((1, 197, 768), torch.float32)],
        {
            "model_name": [
                "pt_deit_facebook_deit_base_patch16_224_img_cls_hf",
                "pt_deit_facebook_deit_base_distilled_patch16_224_img_cls_hf",
                "pt_vit_google_vit_base_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add14,
        [((1, 197, 768), torch.float32)],
        {
            "model_name": [
                "pt_deit_facebook_deit_base_patch16_224_img_cls_hf",
                "pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf",
                "pt_deit_facebook_deit_base_distilled_patch16_224_img_cls_hf",
                "pt_vit_google_vit_base_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 197, 768), torch.float32), ((1, 197, 768), torch.float32)],
        {
            "model_name": [
                "pt_deit_facebook_deit_base_patch16_224_img_cls_hf",
                "pt_deit_facebook_deit_base_distilled_patch16_224_img_cls_hf",
                "pt_vit_google_vit_base_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add16,
        [((1, 197, 3072), torch.float32)],
        {
            "model_name": [
                "pt_deit_facebook_deit_base_patch16_224_img_cls_hf",
                "pt_deit_facebook_deit_base_distilled_patch16_224_img_cls_hf",
                "pt_vit_google_vit_base_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 192, 14, 14), torch.float32), ((192, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf",
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_hrnet_hrnetv2_w48_pose_estimation_osmr",
                "pt_hrnet_hrnet_w48_pose_estimation_timm",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_vovnet_ese_vovnet39b_obj_det_torchhub",
                "pt_vovnet_vovnet39_obj_det_osmr",
                "pt_vovnet_vovnet57_obj_det_osmr",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add71,
        [((1, 197, 192), torch.float32)],
        {"model_name": ["pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add72,
        [((1, 197, 192), torch.float32)],
        {"model_name": ["pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 197, 192), torch.float32), ((1, 197, 192), torch.float32)],
        {"model_name": ["pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 384, 14, 14), torch.float32), ((384, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_deit_facebook_deit_small_patch16_224_img_cls_hf",
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_vovnet_vovnet27s_obj_det_osmr",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add73,
        [((1, 197, 384), torch.float32)],
        {"model_name": ["pt_deit_facebook_deit_small_patch16_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add10,
        [((1, 197, 384), torch.float32)],
        {"model_name": ["pt_deit_facebook_deit_small_patch16_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 197, 384), torch.float32), ((1, 197, 384), torch.float32)],
        {"model_name": ["pt_deit_facebook_deit_small_patch16_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add3,
        [((1, 197, 1536), torch.float32)],
        {"model_name": ["pt_deit_facebook_deit_small_patch16_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((96,), torch.float32), ((1,), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_googlenet_base_img_cls_torchvision",
                "pt_hrnet_hrnetv2_w48_pose_estimation_osmr",
                "pt_hrnet_hrnet_w48_pose_estimation_timm",
                "pt_inception_v4_img_cls_timm",
                "pt_inception_v4_img_cls_osmr",
                "pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf",
                "pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                "pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_vovnet_vovnet27s_obj_det_osmr",
                "pt_yolox_yolox_m_obj_det_torchhub",
                "pt_yolox_yolox_tiny_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add65,
        [((96,), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_googlenet_base_img_cls_torchvision",
                "pt_hrnet_hrnetv2_w48_pose_estimation_osmr",
                "pt_hrnet_hrnet_w48_pose_estimation_timm",
                "pt_inception_v4_img_cls_timm",
                "pt_inception_v4_img_cls_osmr",
                "pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf",
                "pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                "pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_vovnet_vovnet27s_obj_det_osmr",
                "pt_yolox_yolox_m_obj_det_torchhub",
                "pt_yolox_yolox_tiny_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 96, 112, 112), torch.float32), ((96, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 96, 56, 56), torch.float32), ((96, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((192,), torch.float32), ((1,), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_googlenet_base_img_cls_torchvision",
                "pt_hrnet_hrnetv2_w48_pose_estimation_osmr",
                "pt_hrnet_hrnet_w48_pose_estimation_timm",
                "pt_inception_v4_img_cls_timm",
                "pt_inception_v4_img_cls_osmr",
                "pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf",
                "pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_regnet_facebook_regnet_y_040_img_cls_hf",
                "pt_vovnet_ese_vovnet39b_obj_det_torchhub",
                "pt_vovnet_vovnet39_obj_det_osmr",
                "pt_vovnet_vovnet57_obj_det_osmr",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_yolox_yolox_m_obj_det_torchhub",
                "pt_yolox_yolox_tiny_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add72,
        [((192,), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_googlenet_base_img_cls_torchvision",
                "pt_hrnet_hrnetv2_w48_pose_estimation_osmr",
                "pt_hrnet_hrnet_w48_pose_estimation_timm",
                "pt_inception_v4_img_cls_timm",
                "pt_inception_v4_img_cls_osmr",
                "pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf",
                "pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_regnet_facebook_regnet_y_040_img_cls_hf",
                "pt_vovnet_ese_vovnet39b_obj_det_torchhub",
                "pt_vovnet_vovnet39_obj_det_osmr",
                "pt_vovnet_vovnet57_obj_det_osmr",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_yolox_yolox_m_obj_det_torchhub",
                "pt_yolox_yolox_tiny_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 192, 56, 56), torch.float32), ((192, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_googlenet_base_img_cls_torchvision",
                "pt_regnet_facebook_regnet_y_040_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((144,), torch.float32), ((1,), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_googlenet_base_img_cls_torchvision",
                "pt_hrnet_hrnet_w18_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
                "pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                "pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add74,
        [((144,), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_googlenet_base_img_cls_torchvision",
                "pt_hrnet_hrnet_w18_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
                "pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                "pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 144, 56, 56), torch.float32), ((144, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((240,), torch.float32), ((1,), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_hrnet_hrnetv2_w30_pose_estimation_osmr",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add75,
        [((240,), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_hrnet_hrnetv2_w30_pose_estimation_osmr",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 240, 56, 56), torch.float32), ((240, 1, 1), torch.float32)],
        {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((288,), torch.float32), ((1,), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_googlenet_base_img_cls_torchvision",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add76,
        [((288,), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_googlenet_base_img_cls_torchvision",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 288, 56, 56), torch.float32), ((288, 1, 1), torch.float32)],
        {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((336,), torch.float32), ((1,), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add77,
        [((336,), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 336, 56, 56), torch.float32), ((336, 1, 1), torch.float32)],
        {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((384,), torch.float32), ((1,), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_googlenet_base_img_cls_torchvision",
                "pt_hrnet_hrnetv2_w48_pose_estimation_osmr",
                "pt_hrnet_hrnet_w48_pose_estimation_timm",
                "pt_inception_v4_img_cls_timm",
                "pt_inception_v4_img_cls_osmr",
                "pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_vovnet_vovnet27s_obj_det_osmr",
                "pt_yolox_yolox_m_obj_det_torchhub",
                "pt_yolox_yolox_tiny_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add10,
        [((384,), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_googlenet_base_img_cls_torchvision",
                "pt_hrnet_hrnetv2_w48_pose_estimation_osmr",
                "pt_hrnet_hrnet_w48_pose_estimation_timm",
                "pt_inception_v4_img_cls_timm",
                "pt_inception_v4_img_cls_osmr",
                "pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_vovnet_vovnet27s_obj_det_osmr",
                "pt_yolox_yolox_m_obj_det_torchhub",
                "pt_yolox_yolox_tiny_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 384, 56, 56), torch.float32), ((384, 1, 1), torch.float32)],
        {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 192, 28, 28), torch.float32), ((192, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_googlenet_base_img_cls_torchvision",
                "pt_hrnet_hrnetv2_w48_pose_estimation_osmr",
                "pt_hrnet_hrnet_w48_pose_estimation_timm",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_regnet_facebook_regnet_y_040_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 240, 28, 28), torch.float32), ((240, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 288, 28, 28), torch.float32), ((288, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_googlenet_base_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 336, 28, 28), torch.float32), ((336, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 384, 28, 28), torch.float32), ((384, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((432,), torch.float32), ((1,), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add78,
        [((432,), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 432, 28, 28), torch.float32), ((432, 1, 1), torch.float32)],
        {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((480,), torch.float32), ((1,), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add79,
        [((480,), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 480, 28, 28), torch.float32), ((480, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((528,), torch.float32), ((1,), torch.float32)],
        {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (Add80, [((528,), torch.float32)], {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99}),
    (
        Add0,
        [((1, 528, 28, 28), torch.float32), ((528, 1, 1), torch.float32)],
        {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((576,), torch.float32), ((1,), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add81,
        [((576,), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 576, 28, 28), torch.float32), ((576, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((624,), torch.float32), ((1,), torch.float32)],
        {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (Add82, [((624,), torch.float32)], {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99}),
    (
        Add0,
        [((1, 624, 28, 28), torch.float32), ((624, 1, 1), torch.float32)],
        {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((672,), torch.float32), ((1,), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add83,
        [((672,), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 672, 28, 28), torch.float32), ((672, 1, 1), torch.float32)],
        {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((720,), torch.float32), ((1,), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add84,
        [((720,), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 720, 28, 28), torch.float32), ((720, 1, 1), torch.float32)],
        {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((768,), torch.float32), ((1,), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_vovnet_ese_vovnet39b_obj_det_torchhub",
                "pt_vovnet_vovnet39_obj_det_osmr",
                "pt_vovnet_vovnet57_obj_det_osmr",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_yolox_yolox_m_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add14,
        [((768,), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_vovnet_ese_vovnet39b_obj_det_torchhub",
                "pt_vovnet_vovnet39_obj_det_osmr",
                "pt_vovnet_vovnet57_obj_det_osmr",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_yolox_yolox_m_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 768, 28, 28), torch.float32), ((768, 1, 1), torch.float32)],
        {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 432, 14, 14), torch.float32), ((432, 1, 1), torch.float32)],
        {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 480, 14, 14), torch.float32), ((480, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 528, 14, 14), torch.float32), ((528, 1, 1), torch.float32)],
        {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 576, 14, 14), torch.float32), ((576, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 624, 14, 14), torch.float32), ((624, 1, 1), torch.float32)],
        {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 672, 14, 14), torch.float32), ((672, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 720, 14, 14), torch.float32), ((720, 1, 1), torch.float32)],
        {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((816,), torch.float32), ((1,), torch.float32)],
        {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (Add85, [((816,), torch.float32)], {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99}),
    (
        Add0,
        [((1, 816, 14, 14), torch.float32), ((816, 1, 1), torch.float32)],
        {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((864,), torch.float32), ((1,), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add86,
        [((864,), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 864, 14, 14), torch.float32), ((864, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((912,), torch.float32), ((1,), torch.float32)],
        {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (Add87, [((912,), torch.float32)], {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99}),
    (
        Add0,
        [((1, 912, 14, 14), torch.float32), ((912, 1, 1), torch.float32)],
        {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((960,), torch.float32), ((1,), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add88,
        [((960,), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 960, 14, 14), torch.float32), ((960, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1008,), torch.float32), ((1,), torch.float32)],
        {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (Add89, [((1008,), torch.float32)], {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99}),
    (
        Add0,
        [((1, 1008, 14, 14), torch.float32), ((1008, 1, 1), torch.float32)],
        {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1056,), torch.float32), ((1,), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add90,
        [((1056,), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1056, 14, 14), torch.float32), ((1056, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1104,), torch.float32), ((1,), torch.float32)],
        {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (Add91, [((1104,), torch.float32)], {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99}),
    (
        Add0,
        [((1, 1104, 14, 14), torch.float32), ((1104, 1, 1), torch.float32)],
        {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1152,), torch.float32), ((1,), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add92,
        [((1152,), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1152, 14, 14), torch.float32), ((1152, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1200,), torch.float32), ((1,), torch.float32)],
        {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (Add93, [((1200,), torch.float32)], {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99}),
    (
        Add0,
        [((1, 1200, 14, 14), torch.float32), ((1200, 1, 1), torch.float32)],
        {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1248,), torch.float32), ((1,), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add94,
        [((1248,), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1248, 14, 14), torch.float32), ((1248, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1296,), torch.float32), ((1,), torch.float32)],
        {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (Add95, [((1296,), torch.float32)], {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99}),
    (
        Add0,
        [((1, 1296, 14, 14), torch.float32), ((1296, 1, 1), torch.float32)],
        {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1344,), torch.float32), ((1,), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add96,
        [((1344,), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1344, 14, 14), torch.float32), ((1344, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1392,), torch.float32), ((1,), torch.float32)],
        {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (Add97, [((1392,), torch.float32)], {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99}),
    (
        Add0,
        [((1, 1392, 14, 14), torch.float32), ((1392, 1, 1), torch.float32)],
        {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1440,), torch.float32), ((1,), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add98,
        [((1440,), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1440, 14, 14), torch.float32), ((1440, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1488,), torch.float32), ((1,), torch.float32)],
        {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (Add99, [((1488,), torch.float32)], {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99}),
    (
        Add0,
        [((1, 1488, 14, 14), torch.float32), ((1488, 1, 1), torch.float32)],
        {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1536,), torch.float32), ((1,), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_xception_xception_img_cls_timm",
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception65_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add3,
        [((1536,), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_xception_xception_img_cls_timm",
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception65_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1536, 14, 14), torch.float32), ((1536, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1584,), torch.float32), ((1,), torch.float32)],
        {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (Add100, [((1584,), torch.float32)], {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99}),
    (
        Add0,
        [((1, 1584, 14, 14), torch.float32), ((1584, 1, 1), torch.float32)],
        {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1632,), torch.float32), ((1,), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add101,
        [((1632,), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1632, 14, 14), torch.float32), ((1632, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1680,), torch.float32), ((1,), torch.float32)],
        {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (Add102, [((1680,), torch.float32)], {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99}),
    (
        Add0,
        [((1, 1680, 14, 14), torch.float32), ((1680, 1, 1), torch.float32)],
        {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1728,), torch.float32), ((1,), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add103,
        [((1728,), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1728, 14, 14), torch.float32), ((1728, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1776,), torch.float32), ((1,), torch.float32)],
        {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (Add104, [((1776,), torch.float32)], {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99}),
    (
        Add0,
        [((1, 1776, 14, 14), torch.float32), ((1776, 1, 1), torch.float32)],
        {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1824,), torch.float32), ((1,), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add105,
        [((1824,), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1824, 14, 14), torch.float32), ((1824, 1, 1), torch.float32)],
        {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1872,), torch.float32), ((1,), torch.float32)],
        {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (Add106, [((1872,), torch.float32)], {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99}),
    (
        Add0,
        [((1, 1872, 14, 14), torch.float32), ((1872, 1, 1), torch.float32)],
        {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1920,), torch.float32), ((1,), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add107,
        [((1920,), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1920, 14, 14), torch.float32), ((1920, 1, 1), torch.float32)],
        {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1968,), torch.float32), ((1,), torch.float32)],
        {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (Add108, [((1968,), torch.float32)], {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99}),
    (
        Add0,
        [((1, 1968, 14, 14), torch.float32), ((1968, 1, 1), torch.float32)],
        {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((2016,), torch.float32), ((1,), torch.float32)],
        {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (Add109, [((2016,), torch.float32)], {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99}),
    (
        Add0,
        [((1, 2016, 14, 14), torch.float32), ((2016, 1, 1), torch.float32)],
        {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((2064,), torch.float32), ((1,), torch.float32)],
        {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (Add110, [((2064,), torch.float32)], {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99}),
    (
        Add0,
        [((1, 2064, 14, 14), torch.float32), ((2064, 1, 1), torch.float32)],
        {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((2112,), torch.float32), ((1,), torch.float32)],
        {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (Add111, [((2112,), torch.float32)], {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99}),
    (
        Add0,
        [((1, 2112, 14, 14), torch.float32), ((2112, 1, 1), torch.float32)],
        {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1056, 7, 7), torch.float32), ((1056, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 192, 7, 7), torch.float32), ((192, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1104, 7, 7), torch.float32), ((1104, 1, 1), torch.float32)],
        {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1152, 7, 7), torch.float32), ((1152, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1200, 7, 7), torch.float32), ((1200, 1, 1), torch.float32)],
        {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1248, 7, 7), torch.float32), ((1248, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1296, 7, 7), torch.float32), ((1296, 1, 1), torch.float32)],
        {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1344, 7, 7), torch.float32), ((1344, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1392, 7, 7), torch.float32), ((1392, 1, 1), torch.float32)],
        {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1440, 7, 7), torch.float32), ((1440, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1488, 7, 7), torch.float32), ((1488, 1, 1), torch.float32)],
        {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1536, 7, 7), torch.float32), ((1536, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1584, 7, 7), torch.float32), ((1584, 1, 1), torch.float32)],
        {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1632, 7, 7), torch.float32), ((1632, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1680, 7, 7), torch.float32), ((1680, 1, 1), torch.float32)],
        {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1728, 7, 7), torch.float32), ((1728, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1776, 7, 7), torch.float32), ((1776, 1, 1), torch.float32)],
        {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1824, 7, 7), torch.float32), ((1824, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1872, 7, 7), torch.float32), ((1872, 1, 1), torch.float32)],
        {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1920, 7, 7), torch.float32), ((1920, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1968, 7, 7), torch.float32), ((1968, 1, 1), torch.float32)],
        {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 2016, 7, 7), torch.float32), ((2016, 1, 1), torch.float32)],
        {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 2064, 7, 7), torch.float32), ((2064, 1, 1), torch.float32)],
        {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 2112, 7, 7), torch.float32), ((2112, 1, 1), torch.float32)],
        {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((2160,), torch.float32), ((1,), torch.float32)],
        {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (Add112, [((2160,), torch.float32)], {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99}),
    (
        Add0,
        [((1, 2160, 7, 7), torch.float32), ((2160, 1, 1), torch.float32)],
        {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((2208,), torch.float32), ((1,), torch.float32)],
        {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (Add113, [((2208,), torch.float32)], {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99}),
    (
        Add0,
        [((1, 2208, 7, 7), torch.float32), ((2208, 1, 1), torch.float32)],
        {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((64,), torch.float32), ((1,), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_dla_dla34_visual_bb_torchvision",
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_dla_dla46x_c_visual_bb_torchvision",
                "pt_dla_dla60x_c_visual_bb_torchvision",
                "pt_dla_dla46_c_visual_bb_torchvision",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_dla_dla169_visual_bb_torchvision",
                "pt_googlenet_base_img_cls_torchvision",
                "pt_hrnet_hrnet_w18_pose_estimation_timm",
                "pt_hrnet_hrnet_w44_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w30_pose_estimation_osmr",
                "pt_hrnet_hrnet_w32_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnet_w64_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w48_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w44_pose_estimation_osmr",
                "pt_hrnet_hrnet_w40_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w40_pose_estimation_osmr",
                "pt_hrnet_hrnet_w48_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w32_pose_estimation_osmr",
                "pt_inception_v4_img_cls_timm",
                "pt_inception_v4_img_cls_osmr",
                "pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf",
                "pt_mobilenet_v1_basic_img_cls_torchvision",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_monodepth2_mono_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodle_base_obj_det_torchvision",
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "ResNetForImageClassification",
                "ResNet",
                "pt_resnet_50_img_cls_timm",
                "pt_resnext_resnext26_32x4d_img_cls_osmr",
                "pt_resnext_resnext101_64x4d_img_cls_osmr",
                "pt_resnext_resnext50_32x4d_img_cls_osmr",
                "pt_resnext_resnext50_32x4d_img_cls_torchhub",
                "pt_resnext_resnext14_32x4d_img_cls_osmr",
                "pt_resnext_resnext101_32x8d_img_cls_torchhub",
                "pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub",
                "pt_retinanet_retinanet_rn101fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn152fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn34fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn18fpn_obj_det_hf",
                "pt_ssd300_resnet50_base_img_cls_torchhub",
                "pt_unet_base_img_seg_torchhub",
                "pt_unet_cityscape_img_seg_osmr",
                "pt_unet_qubvel_img_seg_torchhub",
                "pt_vgg_bn_vgg19_obj_det_osmr",
                "pt_vgg_vgg19_bn_obj_det_torchhub",
                "pt_vgg_vgg19_bn_obj_det_timm",
                "pt_vgg_bn_vgg19b_obj_det_osmr",
                "pt_vovnet_ese_vovnet39b_obj_det_torchhub",
                "pt_vovnet_vovnet39_obj_det_osmr",
                "pt_vovnet_vovnet57_obj_det_osmr",
                "pt_vovnet_vovnet27s_obj_det_osmr",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_wideresnet_wide_resnet101_2_img_cls_torchvision",
                "pt_wideresnet_wide_resnet50_2_img_cls_timm",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
                "pt_wideresnet_wide_resnet50_2_img_cls_torchvision",
                "pt_xception_xception_img_cls_timm",
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception65_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
                "pt_yolox_yolox_l_obj_det_torchhub",
                "pt_yolox_yolox_s_obj_det_torchhub",
                "pt_yolox_yolox_darknet_obj_det_torchhub",
                "pt_yolox_yolox_nano_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add67,
        [((64,), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_dla_dla34_visual_bb_torchvision",
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_dla_dla46x_c_visual_bb_torchvision",
                "pt_dla_dla60x_c_visual_bb_torchvision",
                "pt_dla_dla46_c_visual_bb_torchvision",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_dla_dla169_visual_bb_torchvision",
                "pt_googlenet_base_img_cls_torchvision",
                "pt_hrnet_hrnet_w18_pose_estimation_timm",
                "pt_hrnet_hrnet_w44_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w30_pose_estimation_osmr",
                "pt_hrnet_hrnet_w32_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnet_w64_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w48_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w44_pose_estimation_osmr",
                "pt_hrnet_hrnet_w40_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w40_pose_estimation_osmr",
                "pt_hrnet_hrnet_w48_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w32_pose_estimation_osmr",
                "pt_inception_v4_img_cls_timm",
                "pt_inception_v4_img_cls_osmr",
                "pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf",
                "pt_mobilenet_v1_basic_img_cls_torchvision",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_monodepth2_mono_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodle_base_obj_det_torchvision",
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "ResNetForImageClassification",
                "ResNet",
                "pt_resnet_50_img_cls_timm",
                "pt_resnext_resnext26_32x4d_img_cls_osmr",
                "pt_resnext_resnext101_64x4d_img_cls_osmr",
                "pt_resnext_resnext50_32x4d_img_cls_osmr",
                "pt_resnext_resnext50_32x4d_img_cls_torchhub",
                "pt_resnext_resnext14_32x4d_img_cls_osmr",
                "pt_resnext_resnext101_32x8d_img_cls_torchhub",
                "pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub",
                "pt_retinanet_retinanet_rn101fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn152fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn34fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn18fpn_obj_det_hf",
                "pt_ssd300_resnet50_base_img_cls_torchhub",
                "pt_unet_base_img_seg_torchhub",
                "pt_unet_cityscape_img_seg_osmr",
                "pt_unet_qubvel_img_seg_torchhub",
                "pt_vgg_bn_vgg19_obj_det_osmr",
                "pt_vgg_vgg19_bn_obj_det_torchhub",
                "pt_vgg_vgg19_bn_obj_det_timm",
                "pt_vgg_bn_vgg19b_obj_det_osmr",
                "pt_vovnet_ese_vovnet39b_obj_det_torchhub",
                "pt_vovnet_vovnet39_obj_det_osmr",
                "pt_vovnet_vovnet57_obj_det_osmr",
                "pt_vovnet_vovnet27s_obj_det_osmr",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_wideresnet_wide_resnet101_2_img_cls_torchvision",
                "pt_wideresnet_wide_resnet50_2_img_cls_timm",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
                "pt_wideresnet_wide_resnet50_2_img_cls_torchvision",
                "pt_xception_xception_img_cls_timm",
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception65_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
                "pt_yolox_yolox_l_obj_det_torchhub",
                "pt_yolox_yolox_s_obj_det_torchhub",
                "pt_yolox_yolox_darknet_obj_det_torchhub",
                "pt_yolox_yolox_nano_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 64, 112, 112), torch.float32), ((64, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_dla_dla46x_c_visual_bb_torchvision",
                "pt_dla_dla60x_c_visual_bb_torchvision",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_dla_dla169_visual_bb_torchvision",
                "pt_googlenet_base_img_cls_torchvision",
                "pt_hrnet_hrnet_w18_pose_estimation_timm",
                "pt_hrnet_hrnet_w44_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w30_pose_estimation_osmr",
                "pt_hrnet_hrnet_w32_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnet_w64_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w48_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w44_pose_estimation_osmr",
                "pt_hrnet_hrnet_w40_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w40_pose_estimation_osmr",
                "pt_hrnet_hrnet_w48_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w32_pose_estimation_osmr",
                "pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf",
                "pt_mobilenet_v1_basic_img_cls_torchvision",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "ResNetForImageClassification",
                "ResNet",
                "pt_resnet_50_img_cls_timm",
                "pt_resnext_resnext26_32x4d_img_cls_osmr",
                "pt_resnext_resnext101_64x4d_img_cls_osmr",
                "pt_resnext_resnext50_32x4d_img_cls_osmr",
                "pt_resnext_resnext50_32x4d_img_cls_torchhub",
                "pt_resnext_resnext14_32x4d_img_cls_osmr",
                "pt_resnext_resnext101_32x8d_img_cls_torchhub",
                "pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub",
                "pt_unet_cityscape_img_seg_osmr",
                "pt_unet_qubvel_img_seg_torchhub",
                "pt_vovnet_ese_vovnet39b_obj_det_torchhub",
                "pt_vovnet_vovnet39_obj_det_osmr",
                "pt_vovnet_vovnet57_obj_det_osmr",
                "pt_vovnet_vovnet27s_obj_det_osmr",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_wideresnet_wide_resnet101_2_img_cls_torchvision",
                "pt_wideresnet_wide_resnet50_2_img_cls_timm",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
                "pt_wideresnet_wide_resnet50_2_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 64, 56, 56), torch.float32), ((64, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_dla_dla34_visual_bb_torchvision",
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_dla_dla46x_c_visual_bb_torchvision",
                "pt_dla_dla60x_c_visual_bb_torchvision",
                "pt_dla_dla46_c_visual_bb_torchvision",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_dla_dla169_visual_bb_torchvision",
                "pt_googlenet_base_img_cls_torchvision",
                "pt_hrnet_hrnet_w18_pose_estimation_timm",
                "pt_hrnet_hrnet_w44_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w30_pose_estimation_osmr",
                "pt_hrnet_hrnet_w32_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnet_w64_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w48_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w44_pose_estimation_osmr",
                "pt_hrnet_hrnet_w40_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w40_pose_estimation_osmr",
                "pt_hrnet_hrnet_w48_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w32_pose_estimation_osmr",
                "pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf",
                "pt_mobilenet_v1_basic_img_cls_torchvision",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_monodle_base_obj_det_torchvision",
                "pt_rcnn_base_obj_det_torchvision_rect_0",
                "ResNetForImageClassification",
                "ResNet",
                "pt_resnet_50_img_cls_timm",
                "pt_unet_qubvel_img_seg_torchhub",
                "pt_vovnet_vovnet27s_obj_det_osmr",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((128,), torch.float32), ((1,), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_dla_dla34_visual_bb_torchvision",
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_dla_dla102x_visual_bb_torchvision",
                "pt_dla_dla46x_c_visual_bb_torchvision",
                "pt_dla_dla60x_c_visual_bb_torchvision",
                "pt_dla_dla46_c_visual_bb_torchvision",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_dla_dla60x_visual_bb_torchvision",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_dla_dla169_visual_bb_torchvision",
                "pt_googlenet_base_img_cls_torchvision",
                "pt_hrnet_hrnet_w18_pose_estimation_timm",
                "pt_hrnet_hrnet_w44_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w30_pose_estimation_osmr",
                "pt_hrnet_hrnet_w32_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnet_w64_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w48_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w44_pose_estimation_osmr",
                "pt_hrnet_hrnet_w40_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w40_pose_estimation_osmr",
                "pt_hrnet_hrnet_w48_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w32_pose_estimation_osmr",
                "pt_inception_v4_img_cls_timm",
                "pt_inception_v4_img_cls_osmr",
                "pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf",
                "pt_mobilenet_v1_basic_img_cls_torchvision",
                "pt_monodepth2_mono_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodle_base_obj_det_torchvision",
                "pt_regnet_facebook_regnet_y_040_img_cls_hf",
                "ResNetForImageClassification",
                "ResNet",
                "pt_resnet_50_img_cls_timm",
                "pt_resnext_resnext26_32x4d_img_cls_osmr",
                "pt_resnext_resnext50_32x4d_img_cls_osmr",
                "pt_resnext_resnext50_32x4d_img_cls_torchhub",
                "pt_resnext_resnext14_32x4d_img_cls_osmr",
                "pt_retinanet_retinanet_rn101fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn152fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn34fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn18fpn_obj_det_hf",
                "pt_ssd300_resnet50_base_img_cls_torchhub",
                "pt_unet_base_img_seg_torchhub",
                "pt_unet_cityscape_img_seg_osmr",
                "pt_unet_qubvel_img_seg_torchhub",
                "pt_vgg_bn_vgg19_obj_det_osmr",
                "pt_vgg_vgg19_bn_obj_det_torchhub",
                "pt_vgg_vgg19_bn_obj_det_timm",
                "pt_vgg_bn_vgg19b_obj_det_osmr",
                "pt_vovnet_ese_vovnet39b_obj_det_torchhub",
                "pt_vovnet_vovnet39_obj_det_osmr",
                "pt_vovnet_vovnet57_obj_det_osmr",
                "pt_vovnet_vovnet27s_obj_det_osmr",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_wideresnet_wide_resnet101_2_img_cls_torchvision",
                "pt_wideresnet_wide_resnet50_2_img_cls_timm",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
                "pt_wideresnet_wide_resnet50_2_img_cls_torchvision",
                "pt_xception_xception_img_cls_timm",
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception65_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
                "pt_yolox_yolox_l_obj_det_torchhub",
                "pt_yolox_yolox_s_obj_det_torchhub",
                "pt_yolox_yolox_darknet_obj_det_torchhub",
                "pt_yolox_yolox_nano_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add23,
        [((128,), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_dla_dla34_visual_bb_torchvision",
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_dla_dla102x_visual_bb_torchvision",
                "pt_dla_dla46x_c_visual_bb_torchvision",
                "pt_dla_dla60x_c_visual_bb_torchvision",
                "pt_dla_dla46_c_visual_bb_torchvision",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_dla_dla60x_visual_bb_torchvision",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_dla_dla169_visual_bb_torchvision",
                "pt_googlenet_base_img_cls_torchvision",
                "pt_hrnet_hrnet_w18_pose_estimation_timm",
                "pt_hrnet_hrnet_w44_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w30_pose_estimation_osmr",
                "pt_hrnet_hrnet_w32_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnet_w64_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w48_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w44_pose_estimation_osmr",
                "pt_hrnet_hrnet_w40_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w40_pose_estimation_osmr",
                "pt_hrnet_hrnet_w48_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w32_pose_estimation_osmr",
                "pt_inception_v4_img_cls_timm",
                "pt_inception_v4_img_cls_osmr",
                "pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf",
                "pt_mobilenet_v1_basic_img_cls_torchvision",
                "pt_monodepth2_mono_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodle_base_obj_det_torchvision",
                "pt_regnet_facebook_regnet_y_040_img_cls_hf",
                "ResNetForImageClassification",
                "ResNet",
                "pt_resnet_50_img_cls_timm",
                "pt_resnext_resnext26_32x4d_img_cls_osmr",
                "pt_resnext_resnext50_32x4d_img_cls_osmr",
                "pt_resnext_resnext50_32x4d_img_cls_torchhub",
                "pt_resnext_resnext14_32x4d_img_cls_osmr",
                "pt_retinanet_retinanet_rn101fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn152fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn34fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn18fpn_obj_det_hf",
                "pt_ssd300_resnet50_base_img_cls_torchhub",
                "pt_unet_base_img_seg_torchhub",
                "pt_unet_cityscape_img_seg_osmr",
                "pt_unet_qubvel_img_seg_torchhub",
                "pt_vgg_bn_vgg19_obj_det_osmr",
                "pt_vgg_vgg19_bn_obj_det_torchhub",
                "pt_vgg_vgg19_bn_obj_det_timm",
                "pt_vgg_bn_vgg19b_obj_det_osmr",
                "pt_vovnet_ese_vovnet39b_obj_det_torchhub",
                "pt_vovnet_vovnet39_obj_det_osmr",
                "pt_vovnet_vovnet57_obj_det_osmr",
                "pt_vovnet_vovnet27s_obj_det_osmr",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_wideresnet_wide_resnet101_2_img_cls_torchvision",
                "pt_wideresnet_wide_resnet50_2_img_cls_timm",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
                "pt_wideresnet_wide_resnet50_2_img_cls_torchvision",
                "pt_xception_xception_img_cls_timm",
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception65_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
                "pt_yolox_yolox_l_obj_det_torchhub",
                "pt_yolox_yolox_s_obj_det_torchhub",
                "pt_yolox_yolox_darknet_obj_det_torchhub",
                "pt_yolox_yolox_nano_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 128, 56, 56), torch.float32), ((128, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_dla_dla102x_visual_bb_torchvision",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_dla_dla60x_visual_bb_torchvision",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_dla_dla169_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_pose_estimation_timm",
                "pt_hrnet_hrnet_w44_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w30_pose_estimation_osmr",
                "pt_hrnet_hrnet_w32_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnet_w64_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w48_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w44_pose_estimation_osmr",
                "pt_hrnet_hrnet_w40_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w40_pose_estimation_osmr",
                "pt_hrnet_hrnet_w48_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w32_pose_estimation_osmr",
                "pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf",
                "pt_mobilenet_v1_basic_img_cls_torchvision",
                "pt_regnet_facebook_regnet_y_040_img_cls_hf",
                "ResNetForImageClassification",
                "ResNet",
                "pt_resnet_50_img_cls_timm",
                "pt_resnext_resnext26_32x4d_img_cls_osmr",
                "pt_resnext_resnext50_32x4d_img_cls_osmr",
                "pt_resnext_resnext50_32x4d_img_cls_torchhub",
                "pt_resnext_resnext14_32x4d_img_cls_osmr",
                "pt_unet_cityscape_img_seg_osmr",
                "pt_unet_qubvel_img_seg_torchhub",
                "pt_vovnet_ese_vovnet39b_obj_det_torchhub",
                "pt_vovnet_vovnet39_obj_det_osmr",
                "pt_vovnet_vovnet57_obj_det_osmr",
                "pt_vovnet_vovnet27s_obj_det_osmr",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_wideresnet_wide_resnet101_2_img_cls_torchvision",
                "pt_wideresnet_wide_resnet50_2_img_cls_timm",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
                "pt_wideresnet_wide_resnet50_2_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((160,), torch.float32), ((1,), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_googlenet_base_img_cls_torchvision",
                "pt_hrnet_hrnet_w40_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w40_pose_estimation_osmr",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_vovnet_ese_vovnet39b_obj_det_torchhub",
                "pt_vovnet_vovnet39_obj_det_osmr",
                "pt_vovnet_vovnet57_obj_det_osmr",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_yolox_yolox_x_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add114,
        [((160,), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_googlenet_base_img_cls_torchvision",
                "pt_hrnet_hrnet_w40_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w40_pose_estimation_osmr",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_vovnet_ese_vovnet39b_obj_det_torchhub",
                "pt_vovnet_vovnet39_obj_det_osmr",
                "pt_vovnet_vovnet57_obj_det_osmr",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_yolox_yolox_x_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 160, 56, 56), torch.float32), ((160, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((224,), torch.float32), ((1,), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_googlenet_base_img_cls_torchvision",
                "pt_inception_v4_img_cls_timm",
                "pt_inception_v4_img_cls_osmr",
                "pt_vovnet_ese_vovnet39b_obj_det_torchhub",
                "pt_vovnet_vovnet39_obj_det_osmr",
                "pt_vovnet_vovnet57_obj_det_osmr",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add115,
        [((224,), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_googlenet_base_img_cls_torchvision",
                "pt_inception_v4_img_cls_timm",
                "pt_inception_v4_img_cls_osmr",
                "pt_vovnet_ese_vovnet39b_obj_det_torchhub",
                "pt_vovnet_vovnet39_obj_det_osmr",
                "pt_vovnet_vovnet57_obj_det_osmr",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 224, 56, 56), torch.float32), ((224, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((256,), torch.float32), ((1,), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_dla_dla34_visual_bb_torchvision",
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_dla_dla102x_visual_bb_torchvision",
                "pt_dla_dla46x_c_visual_bb_torchvision",
                "pt_dla_dla60x_c_visual_bb_torchvision",
                "pt_dla_dla46_c_visual_bb_torchvision",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_dla_dla60x_visual_bb_torchvision",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_dla_dla169_visual_bb_torchvision",
                "pt_fpn_base_img_cls_torchvision",
                "pt_googlenet_base_img_cls_torchvision",
                "pt_hrnet_hrnet_w18_pose_estimation_timm",
                "pt_hrnet_hrnet_w44_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w30_pose_estimation_osmr",
                "pt_hrnet_hrnet_w32_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnet_w64_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w48_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w44_pose_estimation_osmr",
                "pt_hrnet_hrnet_w40_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w40_pose_estimation_osmr",
                "pt_hrnet_hrnet_w48_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w32_pose_estimation_osmr",
                "pt_inception_v4_img_cls_timm",
                "pt_inception_v4_img_cls_osmr",
                "pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf",
                "pt_mobilenet_v1_basic_img_cls_torchvision",
                "pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_monodepth2_mono_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodle_base_obj_det_torchvision",
                "ResNetForImageClassification",
                "ResNet",
                "pt_resnet_50_img_cls_timm",
                "pt_resnext_resnext26_32x4d_img_cls_osmr",
                "pt_resnext_resnext101_64x4d_img_cls_osmr",
                "pt_resnext_resnext50_32x4d_img_cls_osmr",
                "pt_resnext_resnext50_32x4d_img_cls_torchhub",
                "pt_resnext_resnext14_32x4d_img_cls_osmr",
                "pt_resnext_resnext101_32x8d_img_cls_torchhub",
                "pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub",
                "pt_retinanet_retinanet_rn101fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn152fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn34fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn18fpn_obj_det_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_ssd300_resnet50_base_img_cls_torchhub",
                "pt_unet_base_img_seg_torchhub",
                "pt_unet_cityscape_img_seg_osmr",
                "pt_unet_qubvel_img_seg_torchhub",
                "pt_vgg_bn_vgg19_obj_det_osmr",
                "pt_vgg_vgg19_bn_obj_det_torchhub",
                "pt_vgg_vgg19_bn_obj_det_timm",
                "pt_vgg_bn_vgg19b_obj_det_osmr",
                "pt_vovnet_ese_vovnet39b_obj_det_torchhub",
                "pt_vovnet_vovnet39_obj_det_osmr",
                "pt_vovnet_vovnet57_obj_det_osmr",
                "pt_vovnet_vovnet27s_obj_det_osmr",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_wideresnet_wide_resnet101_2_img_cls_torchvision",
                "pt_wideresnet_wide_resnet50_2_img_cls_timm",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
                "pt_wideresnet_wide_resnet50_2_img_cls_torchvision",
                "pt_xception_xception_img_cls_timm",
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception65_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
                "pt_yolox_yolox_l_obj_det_torchhub",
                "pt_yolox_yolox_s_obj_det_torchhub",
                "pt_yolox_yolox_darknet_obj_det_torchhub",
                "pt_yolox_yolox_nano_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add54,
        [((256,), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_dla_dla34_visual_bb_torchvision",
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_dla_dla102x_visual_bb_torchvision",
                "pt_dla_dla46x_c_visual_bb_torchvision",
                "pt_dla_dla60x_c_visual_bb_torchvision",
                "pt_dla_dla46_c_visual_bb_torchvision",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_dla_dla60x_visual_bb_torchvision",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_dla_dla169_visual_bb_torchvision",
                "pt_fpn_base_img_cls_torchvision",
                "pt_googlenet_base_img_cls_torchvision",
                "pt_hrnet_hrnet_w18_pose_estimation_timm",
                "pt_hrnet_hrnet_w44_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w30_pose_estimation_osmr",
                "pt_hrnet_hrnet_w32_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnet_w64_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w48_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w44_pose_estimation_osmr",
                "pt_hrnet_hrnet_w40_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w40_pose_estimation_osmr",
                "pt_hrnet_hrnet_w48_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w32_pose_estimation_osmr",
                "pt_inception_v4_img_cls_timm",
                "pt_inception_v4_img_cls_osmr",
                "pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf",
                "pt_mobilenet_v1_basic_img_cls_torchvision",
                "pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_monodepth2_mono_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodle_base_obj_det_torchvision",
                "ResNetForImageClassification",
                "ResNet",
                "pt_resnet_50_img_cls_timm",
                "pt_resnext_resnext26_32x4d_img_cls_osmr",
                "pt_resnext_resnext101_64x4d_img_cls_osmr",
                "pt_resnext_resnext50_32x4d_img_cls_osmr",
                "pt_resnext_resnext50_32x4d_img_cls_torchhub",
                "pt_resnext_resnext14_32x4d_img_cls_osmr",
                "pt_resnext_resnext101_32x8d_img_cls_torchhub",
                "pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub",
                "pt_retinanet_retinanet_rn101fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn152fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn34fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn18fpn_obj_det_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_ssd300_resnet50_base_img_cls_torchhub",
                "pt_unet_base_img_seg_torchhub",
                "pt_unet_cityscape_img_seg_osmr",
                "pt_unet_qubvel_img_seg_torchhub",
                "pt_vgg_bn_vgg19_obj_det_osmr",
                "pt_vgg_vgg19_bn_obj_det_torchhub",
                "pt_vgg_vgg19_bn_obj_det_timm",
                "pt_vgg_bn_vgg19b_obj_det_osmr",
                "pt_vovnet_ese_vovnet39b_obj_det_torchhub",
                "pt_vovnet_vovnet39_obj_det_osmr",
                "pt_vovnet_vovnet57_obj_det_osmr",
                "pt_vovnet_vovnet27s_obj_det_osmr",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_wideresnet_wide_resnet101_2_img_cls_torchvision",
                "pt_wideresnet_wide_resnet50_2_img_cls_timm",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
                "pt_wideresnet_wide_resnet50_2_img_cls_torchvision",
                "pt_xception_xception_img_cls_timm",
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception65_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
                "pt_yolox_yolox_l_obj_det_torchhub",
                "pt_yolox_yolox_s_obj_det_torchhub",
                "pt_yolox_yolox_darknet_obj_det_torchhub",
                "pt_yolox_yolox_nano_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 256, 56, 56), torch.float32), ((256, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_dla_dla102x_visual_bb_torchvision",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_dla_dla60x_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_pose_estimation_timm",
                "pt_hrnet_hrnet_w44_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w30_pose_estimation_osmr",
                "pt_hrnet_hrnet_w32_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_hrnet_hrnet_w64_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w48_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w44_pose_estimation_osmr",
                "pt_hrnet_hrnet_w40_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w40_pose_estimation_osmr",
                "pt_hrnet_hrnet_w48_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w32_pose_estimation_osmr",
                "pt_monodle_base_obj_det_torchvision",
                "ResNetForImageClassification",
                "ResNet",
                "pt_resnet_50_img_cls_timm",
                "pt_resnext_resnext26_32x4d_img_cls_osmr",
                "pt_resnext_resnext101_64x4d_img_cls_osmr",
                "pt_resnext_resnext50_32x4d_img_cls_osmr",
                "pt_resnext_resnext50_32x4d_img_cls_torchhub",
                "pt_resnext_resnext14_32x4d_img_cls_osmr",
                "pt_resnext_resnext101_32x8d_img_cls_torchhub",
                "pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub",
                "pt_unet_cityscape_img_seg_osmr",
                "pt_unet_qubvel_img_seg_torchhub",
                "pt_vgg_bn_vgg19_obj_det_osmr",
                "pt_vgg_19_obj_det_hf",
                "pt_vgg_vgg19_bn_obj_det_torchhub",
                "pt_vgg_vgg11_obj_det_osmr",
                "pt_vgg_vgg19_bn_obj_det_timm",
                "pt_vgg_bn_vgg19b_obj_det_osmr",
                "pt_vgg_vgg16_obj_det_osmr",
                "pt_vgg_vgg19_obj_det_osmr",
                "pt_vgg_vgg13_obj_det_osmr",
                "pt_vovnet_ese_vovnet39b_obj_det_torchhub",
                "pt_vovnet_vovnet39_obj_det_osmr",
                "pt_vovnet_vovnet57_obj_det_osmr",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_wideresnet_wide_resnet101_2_img_cls_torchvision",
                "pt_wideresnet_wide_resnet50_2_img_cls_timm",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
                "pt_wideresnet_wide_resnet50_2_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 128, 28, 28), torch.float32), ((128, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_dla_dla34_visual_bb_torchvision",
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_dla_dla46x_c_visual_bb_torchvision",
                "pt_dla_dla60x_c_visual_bb_torchvision",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_dla_dla169_visual_bb_torchvision",
                "pt_googlenet_base_img_cls_torchvision",
                "pt_hrnet_hrnet_w32_pose_estimation_timm",
                "pt_hrnet_hrnet_w64_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w32_pose_estimation_osmr",
                "pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf",
                "pt_mobilenet_v1_basic_img_cls_torchvision",
                "pt_monodle_base_obj_det_torchvision",
                "ResNetForImageClassification",
                "ResNet",
                "pt_resnet_50_img_cls_timm",
                "pt_unet_qubvel_img_seg_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 160, 28, 28), torch.float32), ((160, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_hrnet_hrnet_w40_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w40_pose_estimation_osmr",
                "pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_vovnet_ese_vovnet39b_obj_det_torchhub",
                "pt_vovnet_vovnet39_obj_det_osmr",
                "pt_vovnet_vovnet57_obj_det_osmr",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 224, 28, 28), torch.float32), ((224, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 256, 28, 28), torch.float32), ((256, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_dla_dla102x_visual_bb_torchvision",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_dla_dla60x_visual_bb_torchvision",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_dla_dla169_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_pose_estimation_timm",
                "pt_hrnet_hrnet_w44_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w30_pose_estimation_osmr",
                "pt_hrnet_hrnet_w32_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnet_w64_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w48_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w44_pose_estimation_osmr",
                "pt_hrnet_hrnet_w40_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w40_pose_estimation_osmr",
                "pt_hrnet_hrnet_w48_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w32_pose_estimation_osmr",
                "pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf",
                "pt_mobilenet_v1_basic_img_cls_torchvision",
                "pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "ResNetForImageClassification",
                "ResNet",
                "pt_resnet_50_img_cls_timm",
                "pt_resnext_resnext26_32x4d_img_cls_osmr",
                "pt_resnext_resnext50_32x4d_img_cls_osmr",
                "pt_resnext_resnext50_32x4d_img_cls_torchhub",
                "pt_resnext_resnext14_32x4d_img_cls_osmr",
                "pt_unet_cityscape_img_seg_osmr",
                "pt_unet_qubvel_img_seg_torchhub",
                "pt_vovnet_vovnet27s_obj_det_osmr",
                "pt_wideresnet_wide_resnet101_2_img_cls_torchvision",
                "pt_wideresnet_wide_resnet50_2_img_cls_timm",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
                "pt_wideresnet_wide_resnet50_2_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((320,), torch.float32), ((1,), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_googlenet_base_img_cls_torchvision",
                "pt_hrnet_hrnet_w40_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w40_pose_estimation_osmr",
                "pt_inception_v4_img_cls_timm",
                "pt_inception_v4_img_cls_osmr",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_yolox_yolox_x_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add116,
        [((320,), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_googlenet_base_img_cls_torchvision",
                "pt_hrnet_hrnet_w40_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w40_pose_estimation_osmr",
                "pt_inception_v4_img_cls_timm",
                "pt_inception_v4_img_cls_osmr",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_yolox_yolox_x_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 320, 28, 28), torch.float32), ((320, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((352,), torch.float32), ((1,), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_hrnet_hrnet_w44_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w44_pose_estimation_osmr",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add117,
        [((352,), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_hrnet_hrnet_w44_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w44_pose_estimation_osmr",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 352, 28, 28), torch.float32), ((352, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((416,), torch.float32), ((1,), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add118,
        [((416,), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 416, 28, 28), torch.float32), ((416, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((448,), torch.float32), ((1,), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_inception_v4_img_cls_timm",
                "pt_inception_v4_img_cls_osmr",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add119,
        [((448,), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_inception_v4_img_cls_timm",
                "pt_inception_v4_img_cls_osmr",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 448, 28, 28), torch.float32), ((448, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((512,), torch.float32), ((1,), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_dla_dla34_visual_bb_torchvision",
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_dla_dla102x_visual_bb_torchvision",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_dla_dla60x_visual_bb_torchvision",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_dla_dla169_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_pose_estimation_timm",
                "pt_hrnet_hrnet_w44_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w30_pose_estimation_osmr",
                "pt_hrnet_hrnet_w32_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnet_w64_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w48_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w44_pose_estimation_osmr",
                "pt_hrnet_hrnet_w40_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w40_pose_estimation_osmr",
                "pt_hrnet_hrnet_w48_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w32_pose_estimation_osmr",
                "pt_inception_v4_img_cls_timm",
                "pt_inception_v4_img_cls_osmr",
                "pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf",
                "pt_mobilenet_v1_basic_img_cls_torchvision",
                "pt_monodepth2_mono_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodle_base_obj_det_torchvision",
                "pt_regnet_facebook_regnet_y_040_img_cls_hf",
                "ResNetForImageClassification",
                "ResNet",
                "pt_resnet_50_img_cls_timm",
                "pt_resnext_resnext26_32x4d_img_cls_osmr",
                "pt_resnext_resnext101_64x4d_img_cls_osmr",
                "pt_resnext_resnext50_32x4d_img_cls_osmr",
                "pt_resnext_resnext50_32x4d_img_cls_torchhub",
                "pt_resnext_resnext14_32x4d_img_cls_osmr",
                "pt_resnext_resnext101_32x8d_img_cls_torchhub",
                "pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub",
                "pt_retinanet_retinanet_rn101fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn152fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn34fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn18fpn_obj_det_hf",
                "pt_ssd300_resnet50_base_img_cls_torchhub",
                "pt_unet_base_img_seg_torchhub",
                "pt_unet_cityscape_img_seg_osmr",
                "pt_unet_qubvel_img_seg_torchhub",
                "pt_vgg_bn_vgg19_obj_det_osmr",
                "pt_vgg_vgg19_bn_obj_det_torchhub",
                "pt_vgg_vgg19_bn_obj_det_timm",
                "pt_vgg_bn_vgg19b_obj_det_osmr",
                "pt_vovnet_ese_vovnet39b_obj_det_torchhub",
                "pt_vovnet_vovnet39_obj_det_osmr",
                "pt_vovnet_vovnet57_obj_det_osmr",
                "pt_vovnet_vovnet27s_obj_det_osmr",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_wideresnet_wide_resnet101_2_img_cls_torchvision",
                "pt_wideresnet_wide_resnet50_2_img_cls_timm",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
                "pt_wideresnet_wide_resnet50_2_img_cls_torchvision",
                "pt_yolox_yolox_l_obj_det_torchhub",
                "pt_yolox_yolox_s_obj_det_torchhub",
                "pt_yolox_yolox_darknet_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add12,
        [((512,), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_dla_dla34_visual_bb_torchvision",
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_dla_dla102x_visual_bb_torchvision",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_dla_dla60x_visual_bb_torchvision",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_dla_dla169_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_pose_estimation_timm",
                "pt_hrnet_hrnet_w44_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w30_pose_estimation_osmr",
                "pt_hrnet_hrnet_w32_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnet_w64_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w48_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w44_pose_estimation_osmr",
                "pt_hrnet_hrnet_w40_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w40_pose_estimation_osmr",
                "pt_hrnet_hrnet_w48_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w32_pose_estimation_osmr",
                "pt_inception_v4_img_cls_timm",
                "pt_inception_v4_img_cls_osmr",
                "pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf",
                "pt_mobilenet_v1_basic_img_cls_torchvision",
                "pt_monodepth2_mono_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodle_base_obj_det_torchvision",
                "pt_regnet_facebook_regnet_y_040_img_cls_hf",
                "ResNetForImageClassification",
                "ResNet",
                "pt_resnet_50_img_cls_timm",
                "pt_resnext_resnext26_32x4d_img_cls_osmr",
                "pt_resnext_resnext101_64x4d_img_cls_osmr",
                "pt_resnext_resnext50_32x4d_img_cls_osmr",
                "pt_resnext_resnext50_32x4d_img_cls_torchhub",
                "pt_resnext_resnext14_32x4d_img_cls_osmr",
                "pt_resnext_resnext101_32x8d_img_cls_torchhub",
                "pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub",
                "pt_retinanet_retinanet_rn101fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn152fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn34fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn18fpn_obj_det_hf",
                "pt_ssd300_resnet50_base_img_cls_torchhub",
                "pt_unet_base_img_seg_torchhub",
                "pt_unet_cityscape_img_seg_osmr",
                "pt_unet_qubvel_img_seg_torchhub",
                "pt_vgg_bn_vgg19_obj_det_osmr",
                "pt_vgg_vgg19_bn_obj_det_torchhub",
                "pt_vgg_vgg19_bn_obj_det_timm",
                "pt_vgg_bn_vgg19b_obj_det_osmr",
                "pt_vovnet_ese_vovnet39b_obj_det_torchhub",
                "pt_vovnet_vovnet39_obj_det_osmr",
                "pt_vovnet_vovnet57_obj_det_osmr",
                "pt_vovnet_vovnet27s_obj_det_osmr",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_wideresnet_wide_resnet101_2_img_cls_torchvision",
                "pt_wideresnet_wide_resnet50_2_img_cls_timm",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
                "pt_wideresnet_wide_resnet50_2_img_cls_torchvision",
                "pt_yolox_yolox_l_obj_det_torchhub",
                "pt_yolox_yolox_s_obj_det_torchhub",
                "pt_yolox_yolox_darknet_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 512, 28, 28), torch.float32), ((512, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_dla_dla102x_visual_bb_torchvision",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_dla_dla60x_visual_bb_torchvision",
                "pt_regnet_facebook_regnet_y_040_img_cls_hf",
                "ResNetForImageClassification",
                "ResNet",
                "pt_resnet_50_img_cls_timm",
                "pt_resnext_resnext26_32x4d_img_cls_osmr",
                "pt_resnext_resnext101_64x4d_img_cls_osmr",
                "pt_resnext_resnext50_32x4d_img_cls_osmr",
                "pt_resnext_resnext50_32x4d_img_cls_torchhub",
                "pt_resnext_resnext14_32x4d_img_cls_osmr",
                "pt_resnext_resnext101_32x8d_img_cls_torchhub",
                "pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub",
                "pt_unet_cityscape_img_seg_osmr",
                "pt_unet_qubvel_img_seg_torchhub",
                "pt_vgg_bn_vgg19_obj_det_osmr",
                "pt_vgg_19_obj_det_hf",
                "pt_vgg_vgg19_bn_obj_det_torchhub",
                "pt_vgg_vgg11_obj_det_osmr",
                "pt_vgg_vgg19_bn_obj_det_timm",
                "pt_vgg_bn_vgg19b_obj_det_osmr",
                "pt_vgg_vgg16_obj_det_osmr",
                "pt_vgg_vgg19_obj_det_osmr",
                "pt_vgg_vgg13_obj_det_osmr",
                "pt_vovnet_ese_vovnet39b_obj_det_torchhub",
                "pt_vovnet_vovnet39_obj_det_osmr",
                "pt_vovnet_vovnet57_obj_det_osmr",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_wideresnet_wide_resnet101_2_img_cls_torchvision",
                "pt_wideresnet_wide_resnet50_2_img_cls_timm",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
                "pt_wideresnet_wide_resnet50_2_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 256, 14, 14), torch.float32), ((256, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_dla_dla34_visual_bb_torchvision",
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_dla_dla46x_c_visual_bb_torchvision",
                "pt_dla_dla60x_c_visual_bb_torchvision",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_dla_dla169_visual_bb_torchvision",
                "pt_googlenet_base_img_cls_torchvision",
                "pt_hrnet_hrnet_w64_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
                "pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf",
                "pt_mobilenet_v1_basic_img_cls_torchvision",
                "pt_monodle_base_obj_det_torchvision",
                "ResNetForImageClassification",
                "ResNet",
                "pt_resnet_50_img_cls_timm",
                "pt_unet_qubvel_img_seg_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 128, 14, 14), torch.float32), ((128, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_dla_dla46x_c_visual_bb_torchvision",
                "pt_dla_dla60x_c_visual_bb_torchvision",
                "pt_dla_dla46_c_visual_bb_torchvision",
                "pt_googlenet_base_img_cls_torchvision",
                "pt_hrnet_hrnet_w18_pose_estimation_timm",
                "pt_hrnet_hrnet_w44_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w30_pose_estimation_osmr",
                "pt_hrnet_hrnet_w32_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnet_w64_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w48_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w44_pose_estimation_osmr",
                "pt_hrnet_hrnet_w40_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w40_pose_estimation_osmr",
                "pt_hrnet_hrnet_w48_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w32_pose_estimation_osmr",
                "pt_monodle_base_obj_det_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 288, 14, 14), torch.float32), ((288, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_googlenet_base_img_cls_torchvision",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 320, 14, 14), torch.float32), ((320, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_googlenet_base_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 352, 14, 14), torch.float32), ((352, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 416, 14, 14), torch.float32), ((416, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 448, 14, 14), torch.float32), ((448, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_googlenet_base_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 512, 14, 14), torch.float32), ((512, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_dla_dla102x_visual_bb_torchvision",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_dla_dla60x_visual_bb_torchvision",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_dla_dla169_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_pose_estimation_timm",
                "pt_hrnet_hrnet_w44_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w30_pose_estimation_osmr",
                "pt_hrnet_hrnet_w32_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnet_w64_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w48_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w44_pose_estimation_osmr",
                "pt_hrnet_hrnet_w40_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w40_pose_estimation_osmr",
                "pt_hrnet_hrnet_w48_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w32_pose_estimation_osmr",
                "pt_mlp_mixer_mixer_s16_224_img_cls_timm",
                "pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf",
                "pt_mobilenet_v1_basic_img_cls_torchvision",
                "pt_regnet_facebook_regnet_y_040_img_cls_hf",
                "ResNetForImageClassification",
                "ResNet",
                "pt_resnet_50_img_cls_timm",
                "pt_resnext_resnext26_32x4d_img_cls_osmr",
                "pt_resnext_resnext50_32x4d_img_cls_osmr",
                "pt_resnext_resnext50_32x4d_img_cls_torchhub",
                "pt_resnext_resnext14_32x4d_img_cls_osmr",
                "pt_unet_cityscape_img_seg_osmr",
                "pt_unet_qubvel_img_seg_torchhub",
                "pt_vgg_bn_vgg19_obj_det_osmr",
                "pt_vgg_19_obj_det_hf",
                "pt_vgg_vgg19_bn_obj_det_torchhub",
                "pt_vgg_vgg11_obj_det_osmr",
                "pt_vgg_vgg19_bn_obj_det_timm",
                "pt_vgg_bn_vgg19b_obj_det_osmr",
                "pt_vgg_vgg16_obj_det_osmr",
                "pt_vgg_vgg19_obj_det_osmr",
                "pt_vgg_vgg13_obj_det_osmr",
                "pt_wideresnet_wide_resnet101_2_img_cls_torchvision",
                "pt_wideresnet_wide_resnet50_2_img_cls_timm",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
                "pt_wideresnet_wide_resnet50_2_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((544,), torch.float32), ((1,), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add120,
        [((544,), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 544, 14, 14), torch.float32), ((544, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((608,), torch.float32), ((1,), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add121,
        [((608,), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 608, 14, 14), torch.float32), ((608, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((640,), torch.float32), ((1,), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_yolox_yolox_x_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add122,
        [((640,), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_yolox_yolox_x_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 640, 14, 14), torch.float32), ((640, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((704,), torch.float32), ((1,), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add123,
        [((704,), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 704, 14, 14), torch.float32), ((704, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((736,), torch.float32), ((1,), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add124,
        [((736,), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 736, 14, 14), torch.float32), ((736, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((800,), torch.float32), ((1,), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add125,
        [((800,), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 800, 14, 14), torch.float32), ((800, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((832,), torch.float32), ((1,), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add126,
        [((832,), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 832, 14, 14), torch.float32), ((832, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((896,), torch.float32), ((1,), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add55,
        [((896,), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 896, 14, 14), torch.float32), ((896, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((928,), torch.float32), ((1,), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add127,
        [((928,), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 928, 14, 14), torch.float32), ((928, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((992,), torch.float32), ((1,), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add128,
        [((992,), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 992, 14, 14), torch.float32), ((992, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1024,), torch.float32), ((1,), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_dla_dla102x_visual_bb_torchvision",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_dla_dla60x_visual_bb_torchvision",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_dla_dla169_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_pose_estimation_timm",
                "pt_hrnet_hrnet_w44_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w30_pose_estimation_osmr",
                "pt_hrnet_hrnet_w32_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnet_w64_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w48_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w44_pose_estimation_osmr",
                "pt_hrnet_hrnet_w40_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w40_pose_estimation_osmr",
                "pt_hrnet_hrnet_w48_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w32_pose_estimation_osmr",
                "pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf",
                "pt_mobilenet_v1_basic_img_cls_torchvision",
                "ResNetForImageClassification",
                "ResNet",
                "pt_resnet_50_img_cls_timm",
                "pt_resnext_resnext26_32x4d_img_cls_osmr",
                "pt_resnext_resnext101_64x4d_img_cls_osmr",
                "pt_resnext_resnext50_32x4d_img_cls_osmr",
                "pt_resnext_resnext50_32x4d_img_cls_torchhub",
                "pt_resnext_resnext14_32x4d_img_cls_osmr",
                "pt_resnext_resnext101_32x8d_img_cls_torchhub",
                "pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub",
                "pt_retinanet_retinanet_rn101fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn152fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
                "pt_ssd300_resnet50_base_img_cls_torchhub",
                "pt_unet_qubvel_img_seg_torchhub",
                "pt_vovnet_ese_vovnet39b_obj_det_torchhub",
                "pt_vovnet_vovnet39_obj_det_osmr",
                "pt_vovnet_vovnet57_obj_det_osmr",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_wideresnet_wide_resnet101_2_img_cls_torchvision",
                "pt_wideresnet_wide_resnet50_2_img_cls_timm",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
                "pt_wideresnet_wide_resnet50_2_img_cls_torchvision",
                "pt_xception_xception_img_cls_timm",
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception65_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
                "pt_yolox_yolox_l_obj_det_torchhub",
                "pt_yolox_yolox_darknet_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add4,
        [((1024,), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_dla_dla102x_visual_bb_torchvision",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_dla_dla60x_visual_bb_torchvision",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_dla_dla169_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_pose_estimation_timm",
                "pt_hrnet_hrnet_w44_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w30_pose_estimation_osmr",
                "pt_hrnet_hrnet_w32_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnet_w64_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w48_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w44_pose_estimation_osmr",
                "pt_hrnet_hrnet_w40_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w40_pose_estimation_osmr",
                "pt_hrnet_hrnet_w48_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w32_pose_estimation_osmr",
                "pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf",
                "pt_mobilenet_v1_basic_img_cls_torchvision",
                "ResNetForImageClassification",
                "ResNet",
                "pt_resnet_50_img_cls_timm",
                "pt_resnext_resnext26_32x4d_img_cls_osmr",
                "pt_resnext_resnext101_64x4d_img_cls_osmr",
                "pt_resnext_resnext50_32x4d_img_cls_osmr",
                "pt_resnext_resnext50_32x4d_img_cls_torchhub",
                "pt_resnext_resnext14_32x4d_img_cls_osmr",
                "pt_resnext_resnext101_32x8d_img_cls_torchhub",
                "pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub",
                "pt_retinanet_retinanet_rn101fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn152fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
                "pt_ssd300_resnet50_base_img_cls_torchhub",
                "pt_unet_qubvel_img_seg_torchhub",
                "pt_vovnet_ese_vovnet39b_obj_det_torchhub",
                "pt_vovnet_vovnet39_obj_det_osmr",
                "pt_vovnet_vovnet57_obj_det_osmr",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_wideresnet_wide_resnet101_2_img_cls_torchvision",
                "pt_wideresnet_wide_resnet50_2_img_cls_timm",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
                "pt_wideresnet_wide_resnet50_2_img_cls_torchvision",
                "pt_xception_xception_img_cls_timm",
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception65_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
                "pt_yolox_yolox_l_obj_det_torchhub",
                "pt_yolox_yolox_darknet_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1024, 14, 14), torch.float32), ((1024, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_dla_dla102x_visual_bb_torchvision",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_dla_dla60x_visual_bb_torchvision",
                "pt_mlp_mixer_mixer_l16_224_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_l16_224_img_cls_timm",
                "ResNetForImageClassification",
                "ResNet",
                "pt_resnet_50_img_cls_timm",
                "pt_resnext_resnext26_32x4d_img_cls_osmr",
                "pt_resnext_resnext101_64x4d_img_cls_osmr",
                "pt_resnext_resnext50_32x4d_img_cls_osmr",
                "pt_resnext_resnext50_32x4d_img_cls_torchhub",
                "pt_resnext_resnext14_32x4d_img_cls_osmr",
                "pt_resnext_resnext101_32x8d_img_cls_torchhub",
                "pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub",
                "pt_unet_qubvel_img_seg_torchhub",
                "pt_vit_google_vit_large_patch16_224_img_cls_hf",
                "pt_wideresnet_wide_resnet101_2_img_cls_torchvision",
                "pt_wideresnet_wide_resnet50_2_img_cls_timm",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
                "pt_wideresnet_wide_resnet50_2_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1088,), torch.float32), ((1,), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_regnet_facebook_regnet_y_040_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add129,
        [((1088,), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_regnet_facebook_regnet_y_040_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1088, 14, 14), torch.float32), ((1088, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_regnet_facebook_regnet_y_040_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1120,), torch.float32), ((1,), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add130,
        [((1120,), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1120, 14, 14), torch.float32), ((1120, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1184,), torch.float32), ((1,), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add131,
        [((1184,), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1184, 14, 14), torch.float32), ((1184, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1216,), torch.float32), ((1,), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add132,
        [((1216,), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1216, 14, 14), torch.float32), ((1216, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1280,), torch.float32), ((1,), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_yolox_yolox_x_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add7,
        [((1280,), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_yolox_yolox_x_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1280, 14, 14), torch.float32), ((1280, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1312,), torch.float32), ((1,), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add133,
        [((1312,), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1312, 14, 14), torch.float32), ((1312, 1, 1), torch.float32)],
        {"model_name": ["pt_densenet_densenet201_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1376,), torch.float32), ((1,), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add134,
        [((1376,), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1376, 14, 14), torch.float32), ((1376, 1, 1), torch.float32)],
        {"model_name": ["pt_densenet_densenet201_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1408,), torch.float32), ((1,), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add135,
        [((1408,), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1408, 14, 14), torch.float32), ((1408, 1, 1), torch.float32)],
        {"model_name": ["pt_densenet_densenet201_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1472,), torch.float32), ((1,), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add136,
        [((1472,), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1472, 14, 14), torch.float32), ((1472, 1, 1), torch.float32)],
        {"model_name": ["pt_densenet_densenet201_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1504,), torch.float32), ((1,), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add137,
        [((1504,), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1504, 14, 14), torch.float32), ((1504, 1, 1), torch.float32)],
        {"model_name": ["pt_densenet_densenet201_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1568,), torch.float32), ((1,), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add138,
        [((1568,), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1568, 14, 14), torch.float32), ((1568, 1, 1), torch.float32)],
        {"model_name": ["pt_densenet_densenet201_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1600,), torch.float32), ((1,), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add139,
        [((1600,), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1600, 14, 14), torch.float32), ((1600, 1, 1), torch.float32)],
        {"model_name": ["pt_densenet_densenet201_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1664,), torch.float32), ((1,), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add140,
        [((1664,), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1664, 14, 14), torch.float32), ((1664, 1, 1), torch.float32)],
        {"model_name": ["pt_densenet_densenet201_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1696,), torch.float32), ((1,), torch.float32)],
        {"model_name": ["pt_densenet_densenet201_img_cls_torchvision"], "pcc": 0.99},
    ),
    (Add141, [((1696,), torch.float32)], {"model_name": ["pt_densenet_densenet201_img_cls_torchvision"], "pcc": 0.99}),
    (
        Add0,
        [((1, 1696, 14, 14), torch.float32), ((1696, 1, 1), torch.float32)],
        {"model_name": ["pt_densenet_densenet201_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1760,), torch.float32), ((1,), torch.float32)],
        {"model_name": ["pt_densenet_densenet201_img_cls_torchvision"], "pcc": 0.99},
    ),
    (Add142, [((1760,), torch.float32)], {"model_name": ["pt_densenet_densenet201_img_cls_torchvision"], "pcc": 0.99}),
    (
        Add0,
        [((1, 1760, 14, 14), torch.float32), ((1760, 1, 1), torch.float32)],
        {"model_name": ["pt_densenet_densenet201_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1792,), torch.float32), ((1,), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add143,
        [((1792,), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1792, 14, 14), torch.float32), ((1792, 1, 1), torch.float32)],
        {"model_name": ["pt_densenet_densenet201_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 896, 7, 7), torch.float32), ((896, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 128, 7, 7), torch.float32), ((128, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_dla_dla46_c_visual_bb_torchvision",
                "pt_googlenet_base_img_cls_torchvision",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 928, 7, 7), torch.float32), ((928, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 960, 7, 7), torch.float32), ((960, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 992, 7, 7), torch.float32), ((992, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1024, 7, 7), torch.float32), ((1024, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_dla_dla102x_visual_bb_torchvision",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_dla_dla60x_visual_bb_torchvision",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_dla_dla169_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_pose_estimation_timm",
                "pt_hrnet_hrnet_w44_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w30_pose_estimation_osmr",
                "pt_hrnet_hrnet_w32_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnet_w64_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w48_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w44_pose_estimation_osmr",
                "pt_hrnet_hrnet_w40_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w40_pose_estimation_osmr",
                "pt_hrnet_hrnet_w48_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w32_pose_estimation_osmr",
                "pt_mlp_mixer_mixer_l32_224_img_cls_timm",
                "pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf",
                "pt_mobilenet_v1_basic_img_cls_torchvision",
                "pt_resnext_resnext26_32x4d_img_cls_osmr",
                "pt_resnext_resnext50_32x4d_img_cls_osmr",
                "pt_resnext_resnext50_32x4d_img_cls_torchhub",
                "pt_resnext_resnext14_32x4d_img_cls_osmr",
                "pt_vovnet_ese_vovnet39b_obj_det_torchhub",
                "pt_vovnet_vovnet39_obj_det_osmr",
                "pt_vovnet_vovnet57_obj_det_osmr",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_wideresnet_wide_resnet101_2_img_cls_torchvision",
                "pt_wideresnet_wide_resnet50_2_img_cls_timm",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
                "pt_wideresnet_wide_resnet50_2_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1088, 7, 7), torch.float32), ((1088, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_regnet_facebook_regnet_y_040_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1120, 7, 7), torch.float32), ((1120, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1184, 7, 7), torch.float32), ((1184, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1216, 7, 7), torch.float32), ((1216, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1280, 7, 7), torch.float32), ((1280, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1312, 7, 7), torch.float32), ((1312, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1376, 7, 7), torch.float32), ((1376, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1408, 7, 7), torch.float32), ((1408, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1472, 7, 7), torch.float32), ((1472, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1504, 7, 7), torch.float32), ((1504, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1568, 7, 7), torch.float32), ((1568, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1600, 7, 7), torch.float32), ((1600, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1664, 7, 7), torch.float32), ((1664, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1696, 7, 7), torch.float32), ((1696, 1, 1), torch.float32)],
        {"model_name": ["pt_densenet_densenet201_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1760, 7, 7), torch.float32), ((1760, 1, 1), torch.float32)],
        {"model_name": ["pt_densenet_densenet201_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1792, 7, 7), torch.float32), ((1792, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1856,), torch.float32), ((1,), torch.float32)],
        {"model_name": ["pt_densenet_densenet201_img_cls_torchvision"], "pcc": 0.99},
    ),
    (Add144, [((1856,), torch.float32)], {"model_name": ["pt_densenet_densenet201_img_cls_torchvision"], "pcc": 0.99}),
    (
        Add0,
        [((1, 1856, 7, 7), torch.float32), ((1856, 1, 1), torch.float32)],
        {"model_name": ["pt_densenet_densenet201_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1888,), torch.float32), ((1,), torch.float32)],
        {"model_name": ["pt_densenet_densenet201_img_cls_torchvision"], "pcc": 0.99},
    ),
    (Add145, [((1888,), torch.float32)], {"model_name": ["pt_densenet_densenet201_img_cls_torchvision"], "pcc": 0.99}),
    (
        Add0,
        [((1, 1888, 7, 7), torch.float32), ((1888, 1, 1), torch.float32)],
        {"model_name": ["pt_densenet_densenet201_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 640, 7, 7), torch.float32), ((640, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 672, 7, 7), torch.float32), ((672, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 704, 7, 7), torch.float32), ((704, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 736, 7, 7), torch.float32), ((736, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 768, 7, 7), torch.float32), ((768, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_mlp_mixer_mixer_b32_224_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 800, 7, 7), torch.float32), ((800, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 832, 7, 7), torch.float32), ((832, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 864, 7, 7), torch.float32), ((864, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 512, 7, 7), torch.float32), ((512, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_dla_dla34_visual_bb_torchvision",
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_dla_dla169_visual_bb_torchvision",
                "pt_hrnet_hrnet_w64_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
                "pt_mlp_mixer_mixer_s32_224_img_cls_timm",
                "pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf",
                "pt_mobilenet_v1_basic_img_cls_torchvision",
                "pt_monodle_base_obj_det_torchvision",
                "ResNetForImageClassification",
                "ResNet",
                "pt_resnet_50_img_cls_timm",
                "pt_unet_qubvel_img_seg_torchhub",
                "pt_vovnet_vovnet27s_obj_det_osmr",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 544, 7, 7), torch.float32), ((544, 1, 1), torch.float32)],
        {"model_name": ["pt_densenet_densenet121_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 576, 7, 7), torch.float32), ((576, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 608, 7, 7), torch.float32), ((608, 1, 1), torch.float32)],
        {"model_name": ["pt_densenet_densenet121_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((16,), torch.float32), ((1,), torch.float32)],
        {
            "model_name": [
                "pt_dla_dla34_visual_bb_torchvision",
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_dla_dla102x_visual_bb_torchvision",
                "pt_dla_dla46x_c_visual_bb_torchvision",
                "pt_dla_dla60x_c_visual_bb_torchvision",
                "pt_dla_dla46_c_visual_bb_torchvision",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_dla_dla60x_visual_bb_torchvision",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_dla_dla169_visual_bb_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_googlenet_base_img_cls_torchvision",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                "pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_monodle_base_obj_det_torchvision",
                "pt_unet_qubvel_img_seg_torchhub",
                "pt_yolox_yolox_nano_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add146,
        [((16,), torch.float32)],
        {
            "model_name": [
                "pt_dla_dla34_visual_bb_torchvision",
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_dla_dla102x_visual_bb_torchvision",
                "pt_dla_dla46x_c_visual_bb_torchvision",
                "pt_dla_dla60x_c_visual_bb_torchvision",
                "pt_dla_dla46_c_visual_bb_torchvision",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_dla_dla60x_visual_bb_torchvision",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_dla_dla169_visual_bb_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_googlenet_base_img_cls_torchvision",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                "pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_monodle_base_obj_det_torchvision",
                "pt_unet_qubvel_img_seg_torchhub",
                "pt_yolox_yolox_nano_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 16, 224, 224), torch.float32), ((16, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_dla_dla34_visual_bb_torchvision",
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_dla_dla102x_visual_bb_torchvision",
                "pt_dla_dla46x_c_visual_bb_torchvision",
                "pt_dla_dla60x_c_visual_bb_torchvision",
                "pt_dla_dla46_c_visual_bb_torchvision",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_dla_dla60x_visual_bb_torchvision",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_dla_dla169_visual_bb_torchvision",
                "pt_monodle_base_obj_det_torchvision",
                "pt_unet_qubvel_img_seg_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((32,), torch.float32), ((1,), torch.float32)],
        {
            "model_name": [
                "pt_dla_dla34_visual_bb_torchvision",
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_dla_dla102x_visual_bb_torchvision",
                "pt_dla_dla46x_c_visual_bb_torchvision",
                "pt_dla_dla60x_c_visual_bb_torchvision",
                "pt_dla_dla46_c_visual_bb_torchvision",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_dla_dla60x_visual_bb_torchvision",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_dla_dla169_visual_bb_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_googlenet_base_img_cls_torchvision",
                "pt_hrnet_hrnet_w18_pose_estimation_timm",
                "pt_hrnet_hrnet_w44_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w30_pose_estimation_osmr",
                "pt_hrnet_hrnet_w32_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnet_w64_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w48_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w44_pose_estimation_osmr",
                "pt_hrnet_hrnet_w40_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w40_pose_estimation_osmr",
                "pt_hrnet_hrnet_w48_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w32_pose_estimation_osmr",
                "pt_inception_v4_img_cls_timm",
                "pt_inception_v4_img_cls_osmr",
                "pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf",
                "pt_mobilenet_v1_basic_img_cls_torchvision",
                "pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_monodle_base_obj_det_torchvision",
                "pt_regnet_facebook_regnet_y_040_img_cls_hf",
                "pt_unet_base_img_seg_torchhub",
                "pt_unet_qubvel_img_seg_torchhub",
                "pt_xception_xception_img_cls_timm",
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception65_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
                "pt_yolox_yolox_s_obj_det_torchhub",
                "pt_yolox_yolox_darknet_obj_det_torchhub",
                "pt_yolox_yolox_nano_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add147,
        [((32,), torch.float32)],
        {
            "model_name": [
                "pt_dla_dla34_visual_bb_torchvision",
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_dla_dla102x_visual_bb_torchvision",
                "pt_dla_dla46x_c_visual_bb_torchvision",
                "pt_dla_dla60x_c_visual_bb_torchvision",
                "pt_dla_dla46_c_visual_bb_torchvision",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_dla_dla60x_visual_bb_torchvision",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_dla_dla169_visual_bb_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_googlenet_base_img_cls_torchvision",
                "pt_hrnet_hrnet_w18_pose_estimation_timm",
                "pt_hrnet_hrnet_w44_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w30_pose_estimation_osmr",
                "pt_hrnet_hrnet_w32_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnet_w64_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w48_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w44_pose_estimation_osmr",
                "pt_hrnet_hrnet_w40_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w40_pose_estimation_osmr",
                "pt_hrnet_hrnet_w48_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w32_pose_estimation_osmr",
                "pt_inception_v4_img_cls_timm",
                "pt_inception_v4_img_cls_osmr",
                "pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf",
                "pt_mobilenet_v1_basic_img_cls_torchvision",
                "pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_monodle_base_obj_det_torchvision",
                "pt_regnet_facebook_regnet_y_040_img_cls_hf",
                "pt_unet_base_img_seg_torchhub",
                "pt_unet_qubvel_img_seg_torchhub",
                "pt_xception_xception_img_cls_timm",
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception65_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
                "pt_yolox_yolox_s_obj_det_torchhub",
                "pt_yolox_yolox_darknet_obj_det_torchhub",
                "pt_yolox_yolox_nano_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 32, 112, 112), torch.float32), ((32, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_dla_dla34_visual_bb_torchvision",
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_dla_dla102x_visual_bb_torchvision",
                "pt_dla_dla46x_c_visual_bb_torchvision",
                "pt_dla_dla60x_c_visual_bb_torchvision",
                "pt_dla_dla46_c_visual_bb_torchvision",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_dla_dla60x_visual_bb_torchvision",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_dla_dla169_visual_bb_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf",
                "pt_mobilenet_v1_basic_img_cls_torchvision",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_monodle_base_obj_det_torchvision",
                "pt_regnet_facebook_regnet_y_040_img_cls_hf",
                "pt_unet_qubvel_img_seg_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 64, 56, 56), torch.float32), ((1, 64, 56, 56), torch.float32)],
        {
            "model_name": [
                "pt_dla_dla34_visual_bb_torchvision",
                "pt_dla_dla46x_c_visual_bb_torchvision",
                "pt_dla_dla60x_c_visual_bb_torchvision",
                "pt_dla_dla46_c_visual_bb_torchvision",
                "pt_hrnet_hrnet_w64_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
                "pt_monodle_base_obj_det_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 128, 28, 28), torch.float32), ((1, 128, 28, 28), torch.float32)],
        {
            "model_name": [
                "pt_dla_dla34_visual_bb_torchvision",
                "pt_hrnet_hrnet_w64_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
                "pt_monodle_base_obj_det_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 256, 14, 14), torch.float32), ((1, 256, 14, 14), torch.float32)],
        {
            "model_name": [
                "pt_dla_dla34_visual_bb_torchvision",
                "pt_hrnet_hrnet_w64_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
                "pt_monodle_base_obj_det_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 512, 7, 7), torch.float32), ((1, 512, 7, 7), torch.float32)],
        {
            "model_name": [
                "pt_dla_dla34_visual_bb_torchvision",
                "pt_hrnet_hrnet_w64_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
                "pt_monodle_base_obj_det_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1000, 1, 1), torch.float32), ((1000, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_dla_dla34_visual_bb_torchvision",
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_dla_dla102x_visual_bb_torchvision",
                "pt_dla_dla46x_c_visual_bb_torchvision",
                "pt_dla_dla60x_c_visual_bb_torchvision",
                "pt_dla_dla46_c_visual_bb_torchvision",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_dla_dla60x_visual_bb_torchvision",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_dla_dla169_visual_bb_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 128, 56, 56), torch.float32), ((1, 128, 56, 56), torch.float32)],
        {
            "model_name": [
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_dla_dla102x_visual_bb_torchvision",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_dla_dla60x_visual_bb_torchvision",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_dla_dla169_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_pose_estimation_timm",
                "pt_hrnet_hrnet_w44_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w30_pose_estimation_osmr",
                "pt_hrnet_hrnet_w32_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnet_w64_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w48_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w44_pose_estimation_osmr",
                "pt_hrnet_hrnet_w40_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w40_pose_estimation_osmr",
                "pt_hrnet_hrnet_w48_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w32_pose_estimation_osmr",
                "pt_regnet_facebook_regnet_y_040_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 256, 28, 28), torch.float32), ((1, 256, 28, 28), torch.float32)],
        {
            "model_name": [
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_dla_dla102x_visual_bb_torchvision",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_dla_dla60x_visual_bb_torchvision",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_dla_dla169_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_pose_estimation_timm",
                "pt_hrnet_hrnet_w44_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w30_pose_estimation_osmr",
                "pt_hrnet_hrnet_w32_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnet_w64_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w48_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w44_pose_estimation_osmr",
                "pt_hrnet_hrnet_w40_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w40_pose_estimation_osmr",
                "pt_hrnet_hrnet_w48_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w32_pose_estimation_osmr",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 512, 14, 14), torch.float32), ((1, 512, 14, 14), torch.float32)],
        {
            "model_name": [
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_dla_dla102x_visual_bb_torchvision",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_dla_dla60x_visual_bb_torchvision",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_dla_dla169_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_pose_estimation_timm",
                "pt_hrnet_hrnet_w44_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w30_pose_estimation_osmr",
                "pt_hrnet_hrnet_w32_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnet_w64_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w48_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w44_pose_estimation_osmr",
                "pt_hrnet_hrnet_w40_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w40_pose_estimation_osmr",
                "pt_hrnet_hrnet_w48_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w32_pose_estimation_osmr",
                "pt_regnet_facebook_regnet_y_040_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1024, 7, 7), torch.float32), ((1, 1024, 7, 7), torch.float32)],
        {
            "model_name": [
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_dla_dla102x_visual_bb_torchvision",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_dla_dla60x_visual_bb_torchvision",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_dla_dla169_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_pose_estimation_timm",
                "pt_hrnet_hrnet_w44_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w30_pose_estimation_osmr",
                "pt_hrnet_hrnet_w32_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnet_w64_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w48_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w44_pose_estimation_osmr",
                "pt_hrnet_hrnet_w40_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w40_pose_estimation_osmr",
                "pt_hrnet_hrnet_w48_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w32_pose_estimation_osmr",
                "pt_vovnet_ese_vovnet39b_obj_det_torchhub",
                "pt_vovnet_vovnet39_obj_det_osmr",
                "pt_vovnet_vovnet57_obj_det_osmr",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 128, 112, 112), torch.float32), ((128, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_dla_dla102x_visual_bb_torchvision",
                "pt_dla_dla60x_visual_bb_torchvision",
                "pt_regnet_facebook_regnet_y_040_img_cls_hf",
                "pt_unet_cityscape_img_seg_osmr",
                "pt_vgg_bn_vgg19_obj_det_osmr",
                "pt_vgg_19_obj_det_hf",
                "pt_vgg_vgg19_bn_obj_det_torchhub",
                "pt_vgg_vgg11_obj_det_osmr",
                "pt_vgg_vgg19_bn_obj_det_timm",
                "pt_vgg_bn_vgg19b_obj_det_osmr",
                "pt_vgg_vgg16_obj_det_osmr",
                "pt_vgg_vgg19_obj_det_osmr",
                "pt_vgg_vgg13_obj_det_osmr",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 64, 28, 28), torch.float32), ((64, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_dla_dla46x_c_visual_bb_torchvision",
                "pt_dla_dla60x_c_visual_bb_torchvision",
                "pt_dla_dla46_c_visual_bb_torchvision",
                "pt_googlenet_base_img_cls_torchvision",
                "pt_hrnet_hrnet_w18_pose_estimation_timm",
                "pt_hrnet_hrnet_w44_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w30_pose_estimation_osmr",
                "pt_hrnet_hrnet_w32_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnet_w64_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w48_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w44_pose_estimation_osmr",
                "pt_hrnet_hrnet_w40_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w40_pose_estimation_osmr",
                "pt_hrnet_hrnet_w48_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w32_pose_estimation_osmr",
                "pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_monodle_base_obj_det_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 64, 28, 28), torch.float32), ((1, 64, 28, 28), torch.float32)],
        {
            "model_name": [
                "pt_dla_dla46x_c_visual_bb_torchvision",
                "pt_dla_dla60x_c_visual_bb_torchvision",
                "pt_dla_dla46_c_visual_bb_torchvision",
                "pt_hrnet_hrnet_w32_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w32_pose_estimation_osmr",
                "pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 128, 14, 14), torch.float32), ((1, 128, 14, 14), torch.float32)],
        {
            "model_name": [
                "pt_dla_dla46x_c_visual_bb_torchvision",
                "pt_dla_dla60x_c_visual_bb_torchvision",
                "pt_dla_dla46_c_visual_bb_torchvision",
                "pt_hrnet_hrnet_w32_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w32_pose_estimation_osmr",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 256, 7, 7), torch.float32), ((256, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_dla_dla46x_c_visual_bb_torchvision",
                "pt_dla_dla60x_c_visual_bb_torchvision",
                "pt_dla_dla46_c_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_pose_estimation_timm",
                "pt_hrnet_hrnet_w44_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w30_pose_estimation_osmr",
                "pt_hrnet_hrnet_w32_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnet_w64_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w48_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w44_pose_estimation_osmr",
                "pt_hrnet_hrnet_w40_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w40_pose_estimation_osmr",
                "pt_hrnet_hrnet_w48_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w32_pose_estimation_osmr",
                "pt_monodle_base_obj_det_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 256, 7, 7), torch.float32), ((1, 256, 7, 7), torch.float32)],
        {
            "model_name": [
                "pt_dla_dla46x_c_visual_bb_torchvision",
                "pt_dla_dla60x_c_visual_bb_torchvision",
                "pt_dla_dla46_c_visual_bb_torchvision",
                "pt_hrnet_hrnet_w32_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w32_pose_estimation_osmr",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 32, 56, 56), torch.float32), ((32, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_dla_dla46_c_visual_bb_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_hrnet_hrnet_w18_pose_estimation_timm",
                "pt_hrnet_hrnet_w44_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w30_pose_estimation_osmr",
                "pt_hrnet_hrnet_w32_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnet_w64_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w48_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w44_pose_estimation_osmr",
                "pt_hrnet_hrnet_w40_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w40_pose_estimation_osmr",
                "pt_hrnet_hrnet_w48_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w32_pose_estimation_osmr",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 32, 28, 28), torch.float32), ((32, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_dla_dla46_c_visual_bb_torchvision",
                "pt_googlenet_base_img_cls_torchvision",
                "pt_hrnet_hrnet_w32_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w32_pose_estimation_osmr",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 64, 14, 14), torch.float32), ((64, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_dla_dla46_c_visual_bb_torchvision",
                "pt_googlenet_base_img_cls_torchvision",
                "pt_hrnet_hrnet_w32_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnet_w64_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w32_pose_estimation_osmr",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 256, 112, 112), torch.float32), ((256, 1, 1), torch.float32)],
        {"model_name": ["pt_dla_dla102x2_visual_bb_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 512, 56, 56), torch.float32), ((512, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_resnext_resnext101_64x4d_img_cls_osmr",
                "pt_resnext_resnext101_32x8d_img_cls_torchhub",
                "pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1024, 28, 28), torch.float32), ((1024, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_resnext_resnext101_64x4d_img_cls_osmr",
                "pt_resnext_resnext101_32x8d_img_cls_torchhub",
                "pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((2048,), torch.float32), ((1,), torch.float32)],
        {
            "model_name": [
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_pose_estimation_timm",
                "pt_hrnet_hrnet_w44_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w30_pose_estimation_osmr",
                "pt_hrnet_hrnet_w32_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnet_w64_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w48_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w44_pose_estimation_osmr",
                "pt_hrnet_hrnet_w40_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w40_pose_estimation_osmr",
                "pt_hrnet_hrnet_w48_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w32_pose_estimation_osmr",
                "ResNetForImageClassification",
                "ResNet",
                "pt_resnet_50_img_cls_timm",
                "pt_resnext_resnext26_32x4d_img_cls_osmr",
                "pt_resnext_resnext101_64x4d_img_cls_osmr",
                "pt_resnext_resnext50_32x4d_img_cls_osmr",
                "pt_resnext_resnext50_32x4d_img_cls_torchhub",
                "pt_resnext_resnext14_32x4d_img_cls_osmr",
                "pt_resnext_resnext101_32x8d_img_cls_torchhub",
                "pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub",
                "pt_retinanet_retinanet_rn101fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn152fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
                "pt_unet_qubvel_img_seg_torchhub",
                "pt_wideresnet_wide_resnet101_2_img_cls_torchvision",
                "pt_wideresnet_wide_resnet50_2_img_cls_timm",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
                "pt_wideresnet_wide_resnet50_2_img_cls_torchvision",
                "pt_xception_xception_img_cls_timm",
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception65_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add2,
        [((2048,), torch.float32)],
        {
            "model_name": [
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_pose_estimation_timm",
                "pt_hrnet_hrnet_w44_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w30_pose_estimation_osmr",
                "pt_hrnet_hrnet_w32_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnet_w64_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w48_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w44_pose_estimation_osmr",
                "pt_hrnet_hrnet_w40_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w40_pose_estimation_osmr",
                "pt_hrnet_hrnet_w48_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w32_pose_estimation_osmr",
                "ResNetForImageClassification",
                "ResNet",
                "pt_resnet_50_img_cls_timm",
                "pt_resnext_resnext26_32x4d_img_cls_osmr",
                "pt_resnext_resnext101_64x4d_img_cls_osmr",
                "pt_resnext_resnext50_32x4d_img_cls_osmr",
                "pt_resnext_resnext50_32x4d_img_cls_torchhub",
                "pt_resnext_resnext14_32x4d_img_cls_osmr",
                "pt_resnext_resnext101_32x8d_img_cls_torchhub",
                "pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub",
                "pt_retinanet_retinanet_rn101fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn152fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
                "pt_unet_qubvel_img_seg_torchhub",
                "pt_wideresnet_wide_resnet101_2_img_cls_torchvision",
                "pt_wideresnet_wide_resnet50_2_img_cls_timm",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
                "pt_wideresnet_wide_resnet50_2_img_cls_torchvision",
                "pt_xception_xception_img_cls_timm",
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception65_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 2048, 14, 14), torch.float32), ((2048, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_resnext_resnext101_64x4d_img_cls_osmr",
                "pt_resnext_resnext101_32x8d_img_cls_torchhub",
                "pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 2048, 7, 7), torch.float32), ((2048, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_pose_estimation_timm",
                "pt_hrnet_hrnet_w44_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w30_pose_estimation_osmr",
                "pt_hrnet_hrnet_w32_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnet_w64_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w48_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w44_pose_estimation_osmr",
                "pt_hrnet_hrnet_w40_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w40_pose_estimation_osmr",
                "pt_hrnet_hrnet_w48_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w32_pose_estimation_osmr",
                "ResNetForImageClassification",
                "ResNet",
                "pt_resnet_50_img_cls_timm",
                "pt_resnext_resnext26_32x4d_img_cls_osmr",
                "pt_resnext_resnext101_64x4d_img_cls_osmr",
                "pt_resnext_resnext50_32x4d_img_cls_osmr",
                "pt_resnext_resnext50_32x4d_img_cls_torchhub",
                "pt_resnext_resnext14_32x4d_img_cls_osmr",
                "pt_resnext_resnext101_32x8d_img_cls_torchhub",
                "pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub",
                "pt_unet_qubvel_img_seg_torchhub",
                "pt_wideresnet_wide_resnet101_2_img_cls_torchvision",
                "pt_wideresnet_wide_resnet50_2_img_cls_timm",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
                "pt_wideresnet_wide_resnet50_2_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((48,), torch.float32), ((1,), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_googlenet_base_img_cls_torchvision",
                "pt_hrnet_hrnetv2_w48_pose_estimation_osmr",
                "pt_hrnet_hrnet_w48_pose_estimation_timm",
                "pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf",
                "pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_yolox_yolox_m_obj_det_torchhub",
                "pt_yolox_yolox_tiny_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add63,
        [((48,), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_googlenet_base_img_cls_torchvision",
                "pt_hrnet_hrnetv2_w48_pose_estimation_osmr",
                "pt_hrnet_hrnet_w48_pose_estimation_timm",
                "pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf",
                "pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_yolox_yolox_m_obj_det_torchhub",
                "pt_yolox_yolox_tiny_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 48, 160, 160), torch.float32), ((48, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_yolo_v5_yolov5m_imgcls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_imgcls_torchhub_640x640",
                "pt_yolox_yolox_m_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 12, 1, 1), torch.float32), ((12, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 48, 1, 1), torch.float32), ((48, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_regnet_facebook_regnet_y_040_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((24,), torch.float32), ((1,), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_googlenet_base_img_cls_torchvision",
                "pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf",
                "pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                "pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_yolox_yolox_tiny_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add148,
        [((24,), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_googlenet_base_img_cls_torchvision",
                "pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf",
                "pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                "pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_yolox_yolox_tiny_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 24, 160, 160), torch.float32), ((24, 1, 1), torch.float32)],
        {"model_name": ["pt_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 6, 1, 1), torch.float32), ((6, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 24, 1, 1), torch.float32), ((24, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 24, 160, 160), torch.float32), ((1, 24, 160, 160), torch.float32)],
        {"model_name": ["pt_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 144, 160, 160), torch.float32), ((144, 1, 1), torch.float32)],
        {"model_name": ["pt_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 144, 80, 80), torch.float32), ((144, 1, 1), torch.float32)],
        {"model_name": ["pt_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 144, 1, 1), torch.float32), ((144, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 32, 80, 80), torch.float32), ((32, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_yolo_v5_yolov5n_imgcls_torchhub_320x320",
                "pt_yolo_v5_yolov5n_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_imgcls_torchhub_320x320",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 192, 80, 80), torch.float32), ((192, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_yolo_v5_yolov5m_imgcls_torchhub_640x640",
                "pt_yolox_yolox_m_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 8, 1, 1), torch.float32), ((8, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_regnet_facebook_regnet_y_040_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 192, 1, 1), torch.float32), ((192, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_regnet_facebook_regnet_y_040_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 32, 80, 80), torch.float32), ((1, 32, 80, 80), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_yolo_v5_yolov5n_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_imgcls_torchhub_320x320",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 192, 40, 40), torch.float32), ((192, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_yolo_v5_yolov5m_imgcls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_imgcls_torchhub_640x640",
                "pt_yolox_yolox_m_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((56,), torch.float32), ((1,), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add149,
        [((56,), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 56, 40, 40), torch.float32), ((56, 1, 1), torch.float32)],
        {"model_name": ["pt_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 336, 40, 40), torch.float32), ((336, 1, 1), torch.float32)],
        {"model_name": ["pt_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 14, 1, 1), torch.float32), ((14, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 336, 1, 1), torch.float32), ((336, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 56, 40, 40), torch.float32), ((1, 56, 40, 40), torch.float32)],
        {"model_name": ["pt_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 336, 20, 20), torch.float32), ((336, 1, 1), torch.float32)],
        {"model_name": ["pt_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((112,), torch.float32), ((1,), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_googlenet_base_img_cls_torchvision",
                "pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_vovnet_vovnet27s_obj_det_osmr",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add150,
        [((112,), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_googlenet_base_img_cls_torchvision",
                "pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_vovnet_vovnet27s_obj_det_osmr",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 112, 20, 20), torch.float32), ((112, 1, 1), torch.float32)],
        {"model_name": ["pt_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 672, 20, 20), torch.float32), ((672, 1, 1), torch.float32)],
        {"model_name": ["pt_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 28, 1, 1), torch.float32), ((28, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 672, 1, 1), torch.float32), ((672, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 112, 20, 20), torch.float32), ((1, 112, 20, 20), torch.float32)],
        {"model_name": ["pt_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 160, 20, 20), torch.float32), ((160, 1, 1), torch.float32)],
        {"model_name": ["pt_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 960, 20, 20), torch.float32), ((960, 1, 1), torch.float32)],
        {"model_name": ["pt_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 40, 1, 1), torch.float32), ((40, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 960, 1, 1), torch.float32), ((960, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 160, 20, 20), torch.float32), ((1, 160, 20, 20), torch.float32)],
        {"model_name": ["pt_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 960, 10, 10), torch.float32), ((960, 1, 1), torch.float32)],
        {"model_name": ["pt_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((272,), torch.float32), ((1,), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add151,
        [((272,), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 272, 10, 10), torch.float32), ((272, 1, 1), torch.float32)],
        {"model_name": ["pt_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1632, 10, 10), torch.float32), ((1632, 1, 1), torch.float32)],
        {"model_name": ["pt_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 68, 1, 1), torch.float32), ((68, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1632, 1, 1), torch.float32), ((1632, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 272, 10, 10), torch.float32), ((1, 272, 10, 10), torch.float32)],
        {"model_name": ["pt_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 448, 10, 10), torch.float32), ((448, 1, 1), torch.float32)],
        {"model_name": ["pt_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((2688,), torch.float32), ((1,), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add152,
        [((2688,), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 2688, 10, 10), torch.float32), ((2688, 1, 1), torch.float32)],
        {"model_name": ["pt_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 112, 1, 1), torch.float32), ((112, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 2688, 1, 1), torch.float32), ((2688, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 448, 10, 10), torch.float32), ((1, 448, 10, 10), torch.float32)],
        {"model_name": ["pt_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1792, 10, 10), torch.float32), ((1792, 1, 1), torch.float32)],
        {"model_name": ["pt_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 32, 1, 1), torch.float32), ((32, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_regnet_facebook_regnet_y_040_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 16, 112, 112), torch.float32), ((16, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 4, 1, 1), torch.float32), ((4, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 96, 1, 1), torch.float32), ((96, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 24, 56, 56), torch.float32), ((24, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_monodle_base_obj_det_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 24, 56, 56), torch.float32), ((1, 24, 56, 56), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 144, 28, 28), torch.float32), ((144, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((40,), torch.float32), ((1,), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_hrnet_hrnet_w40_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w40_pose_estimation_osmr",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add153,
        [((40,), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_hrnet_hrnet_w40_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w40_pose_estimation_osmr",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 40, 28, 28), torch.float32), ((40, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_hrnet_hrnet_w40_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w40_pose_estimation_osmr",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 10, 1, 1), torch.float32), ((10, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 240, 1, 1), torch.float32), ((240, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 40, 28, 28), torch.float32), ((1, 40, 28, 28), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 240, 14, 14), torch.float32), ((240, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((80,), torch.float32), ((1,), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_hrnet_hrnet_w40_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w40_pose_estimation_osmr",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_vovnet_vovnet27s_obj_det_osmr",
                "pt_yolox_yolox_x_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add154,
        [((80,), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_hrnet_hrnet_w40_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w40_pose_estimation_osmr",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_vovnet_vovnet27s_obj_det_osmr",
                "pt_yolox_yolox_x_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 80, 14, 14), torch.float32), ((80, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_hrnet_hrnet_w40_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w40_pose_estimation_osmr",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 20, 1, 1), torch.float32), ((20, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 480, 1, 1), torch.float32), ((480, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 80, 14, 14), torch.float32), ((1, 80, 14, 14), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 112, 14, 14), torch.float32), ((112, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 112, 14, 14), torch.float32), ((1, 112, 14, 14), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1152, 1, 1), torch.float32), ((1152, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 192, 7, 7), torch.float32), ((1, 192, 7, 7), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 320, 7, 7), torch.float32), ((320, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_googlenet_base_img_cls_torchvision",
                "pt_hrnet_hrnet_w40_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w40_pose_estimation_osmr",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 48, 112, 112), torch.float32), ((48, 1, 1), torch.float32)],
        {"model_name": ["pt_efficientnet_efficientnet_b4_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 24, 112, 112), torch.float32), ((24, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 24, 112, 112), torch.float32), ((1, 24, 112, 112), torch.float32)],
        {"model_name": ["pt_efficientnet_efficientnet_b4_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 144, 112, 112), torch.float32), ((144, 1, 1), torch.float32)],
        {"model_name": ["pt_efficientnet_efficientnet_b4_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 32, 56, 56), torch.float32), ((1, 32, 56, 56), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_hrnet_hrnet_w32_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w32_pose_estimation_osmr",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 56, 28, 28), torch.float32), ((56, 1, 1), torch.float32)],
        {"model_name": ["pt_efficientnet_efficientnet_b4_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 56, 28, 28), torch.float32), ((1, 56, 28, 28), torch.float32)],
        {"model_name": ["pt_efficientnet_efficientnet_b4_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 336, 14, 14), torch.float32), ((336, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 160, 14, 14), torch.float32), ((160, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_hrnet_hrnet_w40_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w40_pose_estimation_osmr",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 160, 14, 14), torch.float32), ((1, 160, 14, 14), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_hrnet_hrnet_w40_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w40_pose_estimation_osmr",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 272, 7, 7), torch.float32), ((272, 1, 1), torch.float32)],
        {"model_name": ["pt_efficientnet_efficientnet_b4_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 272, 7, 7), torch.float32), ((1, 272, 7, 7), torch.float32)],
        {"model_name": ["pt_efficientnet_efficientnet_b4_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 448, 7, 7), torch.float32), ((448, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_googlenet_base_img_cls_torchvision",
                "pt_hrnet_hrnet_w64_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 2688, 7, 7), torch.float32), ((2688, 1, 1), torch.float32)],
        {"model_name": ["pt_efficientnet_efficientnet_b4_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 448, 7, 7), torch.float32), ((1, 448, 7, 7), torch.float32)],
        {"model_name": ["pt_efficientnet_efficientnet_b4_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 256, 64, 64), torch.float32), ((256, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_fpn_base_img_cls_torchvision",
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 256, 16, 16), torch.float32), ((256, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_fpn_base_img_cls_torchvision",
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 256, 8, 8), torch.float32), ((256, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_fpn_base_img_cls_torchvision",
                "pt_inception_v4_img_cls_timm",
                "pt_inception_v4_img_cls_osmr",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 256, 16, 16), torch.float32), ((1, 256, 16, 16), torch.float32)],
        {"model_name": ["pt_fpn_base_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 256, 64, 64), torch.float32), ((1, 256, 64, 64), torch.float32)],
        {"model_name": ["pt_fpn_base_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((8,), torch.float32), ((1,), torch.float32)],
        {
            "model_name": [
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add64,
        [((8,), torch.float32)],
        {
            "model_name": [
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 8, 112, 112), torch.float32), ((8, 1, 1), torch.float32)],
        {"model_name": ["pt_ghostnet_ghostnet_100_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 16, 112, 112), torch.float32), ((1, 16, 112, 112), torch.float32)],
        {
            "model_name": [
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 48, 56, 56), torch.float32), ((48, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_hrnet_hrnetv2_w48_pose_estimation_osmr",
                "pt_hrnet_hrnet_w48_pose_estimation_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((12,), torch.float32), ((1,), torch.float32)],
        {"model_name": ["pt_ghostnet_ghostnet_100_img_cls_timm"], "pcc": 0.99},
    ),
    (Add68, [((12,), torch.float32)], {"model_name": ["pt_ghostnet_ghostnet_100_img_cls_timm"], "pcc": 0.99}),
    (
        Add0,
        [((1, 12, 56, 56), torch.float32), ((12, 1, 1), torch.float32)],
        {"model_name": ["pt_ghostnet_ghostnet_100_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 16, 56, 56), torch.float32), ((16, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((36,), torch.float32), ((1,), torch.float32)],
        {
            "model_name": [
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_hrnet_hrnet_w18_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add155,
        [((36,), torch.float32)],
        {
            "model_name": [
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_hrnet_hrnet_w18_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 36, 56, 56), torch.float32), ((36, 1, 1), torch.float32)],
        {"model_name": ["pt_ghostnet_ghostnet_100_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((72,), torch.float32), ((1,), torch.float32)],
        {
            "model_name": [
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_hrnet_hrnet_w18_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add156,
        [((72,), torch.float32)],
        {
            "model_name": [
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_hrnet_hrnet_w18_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 72, 28, 28), torch.float32), ((72, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_hrnet_hrnet_w18_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 72, 1, 1), torch.float32), ((72, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    pytest.param(
        (
            Add1,
            [((1, 72, 1, 1), torch.float32)],
            {
                "model_name": [
                    "pt_ghostnet_ghostnet_100_img_cls_timm",
                    "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
                    "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                ],
                "pcc": 0.99,
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Add0,
        [((20,), torch.float32), ((1,), torch.float32)],
        {"model_name": ["pt_ghostnet_ghostnet_100_img_cls_timm"], "pcc": 0.99},
    ),
    (Add157, [((20,), torch.float32)], {"model_name": ["pt_ghostnet_ghostnet_100_img_cls_timm"], "pcc": 0.99}),
    (
        Add0,
        [((1, 20, 28, 28), torch.float32), ((20, 1, 1), torch.float32)],
        {"model_name": ["pt_ghostnet_ghostnet_100_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 24, 28, 28), torch.float32), ((24, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((60,), torch.float32), ((1,), torch.float32)],
        {
            "model_name": [
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_hrnet_hrnetv2_w30_pose_estimation_osmr",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add158,
        [((60,), torch.float32)],
        {
            "model_name": [
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_hrnet_hrnetv2_w30_pose_estimation_osmr",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 60, 28, 28), torch.float32), ((60, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_hrnet_hrnetv2_w30_pose_estimation_osmr",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 120, 1, 1), torch.float32), ((120, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    pytest.param(
        (
            Add1,
            [((1, 120, 1, 1), torch.float32)],
            {
                "model_name": [
                    "pt_ghostnet_ghostnet_100_img_cls_timm",
                    "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                    "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
                    "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
                    "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                ],
                "pcc": 0.99,
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Add0,
        [((120,), torch.float32), ((1,), torch.float32)],
        {
            "model_name": [
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_hrnet_hrnetv2_w30_pose_estimation_osmr",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add159,
        [((120,), torch.float32)],
        {
            "model_name": [
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_hrnet_hrnetv2_w30_pose_estimation_osmr",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 120, 28, 28), torch.float32), ((120, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_hrnet_hrnetv2_w30_pose_estimation_osmr",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 40, 14, 14), torch.float32), ((40, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_hrnet_hrnet_w40_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w40_pose_estimation_osmr",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((100,), torch.float32), ((1,), torch.float32)],
        {"model_name": ["pt_ghostnet_ghostnet_100_img_cls_timm"], "pcc": 0.99},
    ),
    (Add160, [((100,), torch.float32)], {"model_name": ["pt_ghostnet_ghostnet_100_img_cls_timm"], "pcc": 0.99}),
    (
        Add0,
        [((1, 100, 14, 14), torch.float32), ((100, 1, 1), torch.float32)],
        {"model_name": ["pt_ghostnet_ghostnet_100_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((92,), torch.float32), ((1,), torch.float32)],
        {"model_name": ["pt_ghostnet_ghostnet_100_img_cls_timm"], "pcc": 0.99},
    ),
    (Add161, [((92,), torch.float32)], {"model_name": ["pt_ghostnet_ghostnet_100_img_cls_timm"], "pcc": 0.99}),
    (
        Add0,
        [((1, 92, 14, 14), torch.float32), ((92, 1, 1), torch.float32)],
        {"model_name": ["pt_ghostnet_ghostnet_100_img_cls_timm"], "pcc": 0.99},
    ),
    pytest.param(
        (
            Add1,
            [((1, 480, 1, 1), torch.float32)],
            {
                "model_name": [
                    "pt_ghostnet_ghostnet_100_img_cls_timm",
                    "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
                    "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                ],
                "pcc": 0.99,
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Add0,
        [((1, 56, 14, 14), torch.float32), ((56, 1, 1), torch.float32)],
        {"model_name": ["pt_ghostnet_ghostnet_100_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 168, 1, 1), torch.float32), ((168, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    pytest.param(
        (
            Add1,
            [((1, 672, 1, 1), torch.float32)],
            {
                "model_name": [
                    "pt_ghostnet_ghostnet_100_img_cls_timm",
                    "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
                    "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                ],
                "pcc": 0.99,
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Add0,
        [((1, 80, 7, 7), torch.float32), ((80, 1, 1), torch.float32)],
        {"model_name": ["pt_ghostnet_ghostnet_100_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 112, 7, 7), torch.float32), ((112, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_vovnet_vovnet27s_obj_det_osmr",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 160, 7, 7), torch.float32), ((160, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 160, 7, 7), torch.float32), ((1, 160, 7, 7), torch.float32)],
        {
            "model_name": [
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 480, 7, 7), torch.float32), ((480, 1, 1), torch.float32)],
        {"model_name": ["pt_ghostnet_ghostnet_100_img_cls_timm"], "pcc": 0.99},
    ),
    pytest.param(
        (
            Add1,
            [((1, 960, 1, 1), torch.float32)],
            {
                "model_name": [
                    "pt_ghostnet_ghostnet_100_img_cls_timm",
                    "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
                    "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                ],
                "pcc": 0.99,
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Add0,
        [((1, 1280, 1, 1), torch.float32), ((1280, 1, 1), torch.float32)],
        {
            "model_name": ["pt_ghostnet_ghostnet_100_img_cls_timm", "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm"],
            "pcc": 0.99,
        },
    ),
    (Add1, [((1, 1, 224, 224), torch.float32)], {"model_name": ["pt_googlenet_base_img_cls_torchvision"], "pcc": 0.99}),
    (
        Add0,
        [((1, 176, 28, 28), torch.float32), ((176, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_googlenet_base_img_cls_torchvision",
                "pt_hrnet_hrnet_w44_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w44_pose_estimation_osmr",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 96, 28, 28), torch.float32), ((96, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_googlenet_base_img_cls_torchvision",
                "pt_hrnet_hrnetv2_w48_pose_estimation_osmr",
                "pt_hrnet_hrnet_w48_pose_estimation_timm",
                "pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 304, 14, 14), torch.float32), ((304, 1, 1), torch.float32)],
        {"model_name": ["pt_googlenet_base_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((208,), torch.float32), ((1,), torch.float32)],
        {"model_name": ["pt_googlenet_base_img_cls_torchvision"], "pcc": 0.99},
    ),
    (Add162, [((208,), torch.float32)], {"model_name": ["pt_googlenet_base_img_cls_torchvision"], "pcc": 0.99}),
    (
        Add0,
        [((1, 208, 14, 14), torch.float32), ((208, 1, 1), torch.float32)],
        {"model_name": ["pt_googlenet_base_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 48, 14, 14), torch.float32), ((48, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_googlenet_base_img_cls_torchvision",
                "pt_hrnet_hrnetv2_w48_pose_estimation_osmr",
                "pt_hrnet_hrnet_w48_pose_estimation_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 296, 14, 14), torch.float32), ((296, 1, 1), torch.float32)],
        {"model_name": ["pt_googlenet_base_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 224, 14, 14), torch.float32), ((224, 1, 1), torch.float32)],
        {"model_name": ["pt_googlenet_base_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 280, 14, 14), torch.float32), ((280, 1, 1), torch.float32)],
        {"model_name": ["pt_googlenet_base_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 624, 7, 7), torch.float32), ((624, 1, 1), torch.float32)],
        {"model_name": ["pt_googlenet_base_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 384, 7, 7), torch.float32), ((384, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_googlenet_base_img_cls_torchvision",
                "pt_hrnet_hrnetv2_w48_pose_estimation_osmr",
                "pt_hrnet_hrnet_w48_pose_estimation_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 256, 56, 56), torch.float32), ((1, 256, 56, 56), torch.float32)],
        {
            "model_name": [
                "pt_hrnet_hrnet_w18_pose_estimation_timm",
                "pt_hrnet_hrnet_w44_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w30_pose_estimation_osmr",
                "pt_hrnet_hrnet_w32_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_hrnet_hrnet_w64_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w48_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w44_pose_estimation_osmr",
                "pt_hrnet_hrnet_w40_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w40_pose_estimation_osmr",
                "pt_hrnet_hrnet_w48_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w32_pose_estimation_osmr",
                "ResNetForImageClassification",
                "ResNet",
                "pt_resnet_50_img_cls_timm",
                "pt_resnext_resnext26_32x4d_img_cls_osmr",
                "pt_resnext_resnext101_64x4d_img_cls_osmr",
                "pt_resnext_resnext50_32x4d_img_cls_osmr",
                "pt_resnext_resnext50_32x4d_img_cls_torchhub",
                "pt_resnext_resnext14_32x4d_img_cls_osmr",
                "pt_resnext_resnext101_32x8d_img_cls_torchhub",
                "pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub",
                "pt_unet_qubvel_img_seg_torchhub",
                "pt_wideresnet_wide_resnet101_2_img_cls_torchvision",
                "pt_wideresnet_wide_resnet50_2_img_cls_timm",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
                "pt_wideresnet_wide_resnet50_2_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((18,), torch.float32), ((1,), torch.float32)],
        {
            "model_name": [
                "pt_hrnet_hrnet_w18_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add163,
        [((18,), torch.float32)],
        {
            "model_name": [
                "pt_hrnet_hrnet_w18_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 18, 56, 56), torch.float32), ((18, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_hrnet_hrnet_w18_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 18, 56, 56), torch.float32), ((1, 18, 56, 56), torch.float32)],
        {
            "model_name": [
                "pt_hrnet_hrnet_w18_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 36, 28, 28), torch.float32), ((36, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_hrnet_hrnet_w18_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 36, 28, 28), torch.float32), ((1, 36, 28, 28), torch.float32)],
        {
            "model_name": [
                "pt_hrnet_hrnet_w18_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 18, 28, 28), torch.float32), ((18, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_hrnet_hrnet_w18_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 72, 14, 14), torch.float32), ((72, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_hrnet_hrnet_w18_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 72, 14, 14), torch.float32), ((1, 72, 14, 14), torch.float32)],
        {
            "model_name": [
                "pt_hrnet_hrnet_w18_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 18, 14, 14), torch.float32), ((18, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_hrnet_hrnet_w18_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 36, 14, 14), torch.float32), ((36, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_hrnet_hrnet_w18_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 144, 7, 7), torch.float32), ((144, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_hrnet_hrnet_w18_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 144, 7, 7), torch.float32), ((1, 144, 7, 7), torch.float32)],
        {
            "model_name": [
                "pt_hrnet_hrnet_w18_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 126, 7, 7), torch.float32), ((126, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_hrnet_hrnet_w18_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((44,), torch.float32), ((1,), torch.float32)],
        {
            "model_name": ["pt_hrnet_hrnet_w44_pose_estimation_timm", "pt_hrnet_hrnetv2_w44_pose_estimation_osmr"],
            "pcc": 0.99,
        },
    ),
    (
        Add164,
        [((44,), torch.float32)],
        {
            "model_name": ["pt_hrnet_hrnet_w44_pose_estimation_timm", "pt_hrnet_hrnetv2_w44_pose_estimation_osmr"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 44, 56, 56), torch.float32), ((44, 1, 1), torch.float32)],
        {
            "model_name": ["pt_hrnet_hrnet_w44_pose_estimation_timm", "pt_hrnet_hrnetv2_w44_pose_estimation_osmr"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 44, 56, 56), torch.float32), ((1, 44, 56, 56), torch.float32)],
        {
            "model_name": ["pt_hrnet_hrnet_w44_pose_estimation_timm", "pt_hrnet_hrnetv2_w44_pose_estimation_osmr"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((88,), torch.float32), ((1,), torch.float32)],
        {
            "model_name": [
                "pt_hrnet_hrnet_w44_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w44_pose_estimation_osmr",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add165,
        [((88,), torch.float32)],
        {
            "model_name": [
                "pt_hrnet_hrnet_w44_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w44_pose_estimation_osmr",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 88, 28, 28), torch.float32), ((88, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_hrnet_hrnet_w44_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w44_pose_estimation_osmr",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 88, 28, 28), torch.float32), ((1, 88, 28, 28), torch.float32)],
        {
            "model_name": ["pt_hrnet_hrnet_w44_pose_estimation_timm", "pt_hrnet_hrnetv2_w44_pose_estimation_osmr"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 44, 28, 28), torch.float32), ((44, 1, 1), torch.float32)],
        {
            "model_name": ["pt_hrnet_hrnet_w44_pose_estimation_timm", "pt_hrnet_hrnetv2_w44_pose_estimation_osmr"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((176,), torch.float32), ((1,), torch.float32)],
        {
            "model_name": ["pt_hrnet_hrnet_w44_pose_estimation_timm", "pt_hrnet_hrnetv2_w44_pose_estimation_osmr"],
            "pcc": 0.99,
        },
    ),
    (
        Add166,
        [((176,), torch.float32)],
        {
            "model_name": ["pt_hrnet_hrnet_w44_pose_estimation_timm", "pt_hrnet_hrnetv2_w44_pose_estimation_osmr"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 176, 14, 14), torch.float32), ((176, 1, 1), torch.float32)],
        {
            "model_name": ["pt_hrnet_hrnet_w44_pose_estimation_timm", "pt_hrnet_hrnetv2_w44_pose_estimation_osmr"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 176, 14, 14), torch.float32), ((1, 176, 14, 14), torch.float32)],
        {
            "model_name": ["pt_hrnet_hrnet_w44_pose_estimation_timm", "pt_hrnet_hrnetv2_w44_pose_estimation_osmr"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 44, 14, 14), torch.float32), ((44, 1, 1), torch.float32)],
        {
            "model_name": ["pt_hrnet_hrnet_w44_pose_estimation_timm", "pt_hrnet_hrnetv2_w44_pose_estimation_osmr"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 88, 14, 14), torch.float32), ((88, 1, 1), torch.float32)],
        {
            "model_name": ["pt_hrnet_hrnet_w44_pose_estimation_timm", "pt_hrnet_hrnetv2_w44_pose_estimation_osmr"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 352, 7, 7), torch.float32), ((352, 1, 1), torch.float32)],
        {
            "model_name": ["pt_hrnet_hrnet_w44_pose_estimation_timm", "pt_hrnet_hrnetv2_w44_pose_estimation_osmr"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 352, 7, 7), torch.float32), ((1, 352, 7, 7), torch.float32)],
        {
            "model_name": ["pt_hrnet_hrnet_w44_pose_estimation_timm", "pt_hrnet_hrnetv2_w44_pose_estimation_osmr"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 308, 7, 7), torch.float32), ((308, 1, 1), torch.float32)],
        {
            "model_name": ["pt_hrnet_hrnet_w44_pose_estimation_timm", "pt_hrnet_hrnetv2_w44_pose_estimation_osmr"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((30,), torch.float32), ((1,), torch.float32)],
        {
            "model_name": ["pt_hrnet_hrnetv2_w30_pose_estimation_osmr", "pt_hrnet_hrnet_w30_pose_estimation_timm"],
            "pcc": 0.99,
        },
    ),
    (
        Add167,
        [((30,), torch.float32)],
        {
            "model_name": ["pt_hrnet_hrnetv2_w30_pose_estimation_osmr", "pt_hrnet_hrnet_w30_pose_estimation_timm"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 30, 56, 56), torch.float32), ((30, 1, 1), torch.float32)],
        {
            "model_name": ["pt_hrnet_hrnetv2_w30_pose_estimation_osmr", "pt_hrnet_hrnet_w30_pose_estimation_timm"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 30, 56, 56), torch.float32), ((1, 30, 56, 56), torch.float32)],
        {
            "model_name": ["pt_hrnet_hrnetv2_w30_pose_estimation_osmr", "pt_hrnet_hrnet_w30_pose_estimation_timm"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 60, 28, 28), torch.float32), ((1, 60, 28, 28), torch.float32)],
        {
            "model_name": ["pt_hrnet_hrnetv2_w30_pose_estimation_osmr", "pt_hrnet_hrnet_w30_pose_estimation_timm"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 30, 28, 28), torch.float32), ((30, 1, 1), torch.float32)],
        {
            "model_name": ["pt_hrnet_hrnetv2_w30_pose_estimation_osmr", "pt_hrnet_hrnet_w30_pose_estimation_timm"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 120, 14, 14), torch.float32), ((120, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_hrnet_hrnetv2_w30_pose_estimation_osmr",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 120, 14, 14), torch.float32), ((1, 120, 14, 14), torch.float32)],
        {
            "model_name": ["pt_hrnet_hrnetv2_w30_pose_estimation_osmr", "pt_hrnet_hrnet_w30_pose_estimation_timm"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 30, 14, 14), torch.float32), ((30, 1, 1), torch.float32)],
        {
            "model_name": ["pt_hrnet_hrnetv2_w30_pose_estimation_osmr", "pt_hrnet_hrnet_w30_pose_estimation_timm"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 60, 14, 14), torch.float32), ((60, 1, 1), torch.float32)],
        {
            "model_name": ["pt_hrnet_hrnetv2_w30_pose_estimation_osmr", "pt_hrnet_hrnet_w30_pose_estimation_timm"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 240, 7, 7), torch.float32), ((240, 1, 1), torch.float32)],
        {
            "model_name": ["pt_hrnet_hrnetv2_w30_pose_estimation_osmr", "pt_hrnet_hrnet_w30_pose_estimation_timm"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 240, 7, 7), torch.float32), ((1, 240, 7, 7), torch.float32)],
        {
            "model_name": ["pt_hrnet_hrnetv2_w30_pose_estimation_osmr", "pt_hrnet_hrnet_w30_pose_estimation_timm"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 210, 7, 7), torch.float32), ((210, 1, 1), torch.float32)],
        {
            "model_name": ["pt_hrnet_hrnetv2_w30_pose_estimation_osmr", "pt_hrnet_hrnet_w30_pose_estimation_timm"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 32, 14, 14), torch.float32), ((32, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_hrnet_hrnet_w32_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w32_pose_estimation_osmr",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 224, 7, 7), torch.float32), ((224, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_hrnet_hrnet_w32_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w32_pose_estimation_osmr",
                "pt_vovnet_ese_vovnet39b_obj_det_torchhub",
                "pt_vovnet_vovnet39_obj_det_osmr",
                "pt_vovnet_vovnet57_obj_det_osmr",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 16, 56, 56), torch.float32), ((1, 16, 56, 56), torch.float32)],
        {
            "model_name": [
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 32, 28, 28), torch.float32), ((1, 32, 28, 28), torch.float32)],
        {
            "model_name": [
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 64, 14, 14), torch.float32), ((1, 64, 14, 14), torch.float32)],
        {
            "model_name": [
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 128, 7, 7), torch.float32), ((1, 128, 7, 7), torch.float32)],
        {
            "model_name": [
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 48, 56, 56), torch.float32), ((1, 48, 56, 56), torch.float32)],
        {
            "model_name": ["pt_hrnet_hrnetv2_w48_pose_estimation_osmr", "pt_hrnet_hrnet_w48_pose_estimation_timm"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 96, 28, 28), torch.float32), ((1, 96, 28, 28), torch.float32)],
        {
            "model_name": [
                "pt_hrnet_hrnetv2_w48_pose_estimation_osmr",
                "pt_hrnet_hrnet_w48_pose_estimation_timm",
                "pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 48, 28, 28), torch.float32), ((48, 1, 1), torch.float32)],
        {
            "model_name": ["pt_hrnet_hrnetv2_w48_pose_estimation_osmr", "pt_hrnet_hrnet_w48_pose_estimation_timm"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 192, 14, 14), torch.float32), ((1, 192, 14, 14), torch.float32)],
        {
            "model_name": ["pt_hrnet_hrnetv2_w48_pose_estimation_osmr", "pt_hrnet_hrnet_w48_pose_estimation_timm"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 96, 14, 14), torch.float32), ((96, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_hrnet_hrnetv2_w48_pose_estimation_osmr",
                "pt_hrnet_hrnet_w48_pose_estimation_timm",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_vovnet_vovnet27s_obj_det_osmr",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 384, 7, 7), torch.float32), ((1, 384, 7, 7), torch.float32)],
        {
            "model_name": ["pt_hrnet_hrnetv2_w48_pose_estimation_osmr", "pt_hrnet_hrnet_w48_pose_estimation_timm"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 336, 7, 7), torch.float32), ((336, 1, 1), torch.float32)],
        {
            "model_name": ["pt_hrnet_hrnetv2_w48_pose_estimation_osmr", "pt_hrnet_hrnet_w48_pose_estimation_timm"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 40, 56, 56), torch.float32), ((40, 1, 1), torch.float32)],
        {
            "model_name": ["pt_hrnet_hrnet_w40_pose_estimation_timm", "pt_hrnet_hrnetv2_w40_pose_estimation_osmr"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 40, 56, 56), torch.float32), ((1, 40, 56, 56), torch.float32)],
        {
            "model_name": ["pt_hrnet_hrnet_w40_pose_estimation_timm", "pt_hrnet_hrnetv2_w40_pose_estimation_osmr"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 80, 28, 28), torch.float32), ((80, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_hrnet_hrnet_w40_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w40_pose_estimation_osmr",
                "pt_vovnet_vovnet27s_obj_det_osmr",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 80, 28, 28), torch.float32), ((1, 80, 28, 28), torch.float32)],
        {
            "model_name": ["pt_hrnet_hrnet_w40_pose_estimation_timm", "pt_hrnet_hrnetv2_w40_pose_estimation_osmr"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 320, 7, 7), torch.float32), ((1, 320, 7, 7), torch.float32)],
        {
            "model_name": ["pt_hrnet_hrnet_w40_pose_estimation_timm", "pt_hrnet_hrnetv2_w40_pose_estimation_osmr"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 280, 7, 7), torch.float32), ((280, 1, 1), torch.float32)],
        {
            "model_name": ["pt_hrnet_hrnet_w40_pose_estimation_timm", "pt_hrnet_hrnetv2_w40_pose_estimation_osmr"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 32, 149, 149), torch.float32), ((32, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_inception_v4_img_cls_timm",
                "pt_inception_v4_img_cls_osmr",
                "pt_xception_xception_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 32, 147, 147), torch.float32), ((32, 1, 1), torch.float32)],
        {"model_name": ["pt_inception_v4_img_cls_timm", "pt_inception_v4_img_cls_osmr"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 64, 147, 147), torch.float32), ((64, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_inception_v4_img_cls_timm",
                "pt_inception_v4_img_cls_osmr",
                "pt_xception_xception_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 96, 73, 73), torch.float32), ((96, 1, 1), torch.float32)],
        {"model_name": ["pt_inception_v4_img_cls_timm", "pt_inception_v4_img_cls_osmr"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 64, 73, 73), torch.float32), ((64, 1, 1), torch.float32)],
        {"model_name": ["pt_inception_v4_img_cls_timm", "pt_inception_v4_img_cls_osmr"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 96, 71, 71), torch.float32), ((96, 1, 1), torch.float32)],
        {"model_name": ["pt_inception_v4_img_cls_timm", "pt_inception_v4_img_cls_osmr"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 192, 35, 35), torch.float32), ((192, 1, 1), torch.float32)],
        {"model_name": ["pt_inception_v4_img_cls_timm", "pt_inception_v4_img_cls_osmr"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 224, 35, 35), torch.float32), ((224, 1, 1), torch.float32)],
        {"model_name": ["pt_inception_v4_img_cls_timm", "pt_inception_v4_img_cls_osmr"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 96, 35, 35), torch.float32), ((96, 1, 1), torch.float32)],
        {"model_name": ["pt_inception_v4_img_cls_timm", "pt_inception_v4_img_cls_osmr"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 384, 17, 17), torch.float32), ((384, 1, 1), torch.float32)],
        {"model_name": ["pt_inception_v4_img_cls_timm", "pt_inception_v4_img_cls_osmr"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 256, 17, 17), torch.float32), ((256, 1, 1), torch.float32)],
        {"model_name": ["pt_inception_v4_img_cls_timm", "pt_inception_v4_img_cls_osmr"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 768, 17, 17), torch.float32), ((768, 1, 1), torch.float32)],
        {"model_name": ["pt_inception_v4_img_cls_timm", "pt_inception_v4_img_cls_osmr"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 224, 17, 17), torch.float32), ((224, 1, 1), torch.float32)],
        {"model_name": ["pt_inception_v4_img_cls_timm", "pt_inception_v4_img_cls_osmr"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 192, 17, 17), torch.float32), ((192, 1, 1), torch.float32)],
        {"model_name": ["pt_inception_v4_img_cls_timm", "pt_inception_v4_img_cls_osmr"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 128, 17, 17), torch.float32), ((128, 1, 1), torch.float32)],
        {"model_name": ["pt_inception_v4_img_cls_timm", "pt_inception_v4_img_cls_osmr"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 192, 8, 8), torch.float32), ((192, 1, 1), torch.float32)],
        {"model_name": ["pt_inception_v4_img_cls_timm", "pt_inception_v4_img_cls_osmr"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 320, 17, 17), torch.float32), ((320, 1, 1), torch.float32)],
        {"model_name": ["pt_inception_v4_img_cls_timm", "pt_inception_v4_img_cls_osmr"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 320, 8, 8), torch.float32), ((320, 1, 1), torch.float32)],
        {"model_name": ["pt_inception_v4_img_cls_timm", "pt_inception_v4_img_cls_osmr"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1024, 8, 8), torch.float32), ((1024, 1, 1), torch.float32)],
        {"model_name": ["pt_inception_v4_img_cls_timm", "pt_inception_v4_img_cls_osmr"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 448, 8, 8), torch.float32), ((448, 1, 1), torch.float32)],
        {"model_name": ["pt_inception_v4_img_cls_timm", "pt_inception_v4_img_cls_osmr"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 512, 8, 8), torch.float32), ((512, 1, 1), torch.float32)],
        {"model_name": ["pt_inception_v4_img_cls_timm", "pt_inception_v4_img_cls_osmr"], "pcc": 0.99},
    ),
    (
        Add10,
        [((1, 768, 384), torch.float32)],
        {
            "model_name": [
                "pt_mlp_mixer_mixer_b32_224_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_miil_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_miil_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_in21k_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (Add168, [((1, 768, 49), torch.float32)], {"model_name": ["pt_mlp_mixer_mixer_b32_224_img_cls_timm"], "pcc": 0.99}),
    (
        Add0,
        [((1, 49, 768), torch.float32), ((1, 49, 768), torch.float32)],
        {
            "model_name": [
                "pt_mlp_mixer_mixer_b32_224_img_cls_timm",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add16,
        [((1, 49, 3072), torch.float32)],
        {
            "model_name": [
                "pt_mlp_mixer_mixer_b32_224_img_cls_timm",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add14,
        [((1, 49, 768), torch.float32)],
        {
            "model_name": [
                "pt_mlp_mixer_mixer_b32_224_img_cls_timm",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add54,
        [((1, 512, 256), torch.float32)],
        {
            "model_name": ["pt_mlp_mixer_mixer_s16_224_img_cls_timm", "pt_mlp_mixer_mixer_s32_224_img_cls_timm"],
            "pcc": 0.99,
        },
    ),
    (
        Add169,
        [((1, 512, 196), torch.float32)],
        {"model_name": ["pt_mlp_mixer_mixer_s16_224_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 196, 512), torch.float32), ((1, 196, 512), torch.float32)],
        {"model_name": ["pt_mlp_mixer_mixer_s16_224_img_cls_timm"], "pcc": 0.99},
    ),
    (Add2, [((1, 196, 2048), torch.float32)], {"model_name": ["pt_mlp_mixer_mixer_s16_224_img_cls_timm"], "pcc": 0.99}),
    (Add12, [((1, 196, 512), torch.float32)], {"model_name": ["pt_mlp_mixer_mixer_s16_224_img_cls_timm"], "pcc": 0.99}),
    (Add168, [((1, 512, 49), torch.float32)], {"model_name": ["pt_mlp_mixer_mixer_s32_224_img_cls_timm"], "pcc": 0.99}),
    (
        Add0,
        [((1, 49, 512), torch.float32), ((1, 49, 512), torch.float32)],
        {"model_name": ["pt_mlp_mixer_mixer_s32_224_img_cls_timm"], "pcc": 0.99},
    ),
    (Add2, [((1, 49, 2048), torch.float32)], {"model_name": ["pt_mlp_mixer_mixer_s32_224_img_cls_timm"], "pcc": 0.99}),
    (Add12, [((1, 49, 512), torch.float32)], {"model_name": ["pt_mlp_mixer_mixer_s32_224_img_cls_timm"], "pcc": 0.99}),
    (
        Add169,
        [((1, 768, 196), torch.float32)],
        {
            "model_name": [
                "pt_mlp_mixer_mixer_b16_224_miil_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_miil_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_in21k_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 196, 768), torch.float32), ((1, 196, 768), torch.float32)],
        {
            "model_name": [
                "pt_mlp_mixer_mixer_b16_224_miil_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_miil_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_in21k_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add16,
        [((1, 196, 3072), torch.float32)],
        {
            "model_name": [
                "pt_mlp_mixer_mixer_b16_224_miil_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_miil_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_in21k_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add14,
        [((1, 196, 768), torch.float32)],
        {
            "model_name": [
                "pt_mlp_mixer_mixer_b16_224_miil_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_miil_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_in21k_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add170,
        [((1, 11221), torch.float32)],
        {"model_name": ["pt_mlp_mixer_mixer_b16_224_miil_in21k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add12,
        [((1, 1024, 512), torch.float32)],
        {
            "model_name": [
                "pt_mlp_mixer_mixer_l32_224_img_cls_timm",
                "pt_mlp_mixer_mixer_l16_224_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_l16_224_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add168,
        [((1, 1024, 49), torch.float32)],
        {"model_name": ["pt_mlp_mixer_mixer_l32_224_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 49, 1024), torch.float32), ((1, 49, 1024), torch.float32)],
        {"model_name": ["pt_mlp_mixer_mixer_l32_224_img_cls_timm"], "pcc": 0.99},
    ),
    (Add6, [((1, 49, 4096), torch.float32)], {"model_name": ["pt_mlp_mixer_mixer_l32_224_img_cls_timm"], "pcc": 0.99}),
    (Add4, [((1, 49, 1024), torch.float32)], {"model_name": ["pt_mlp_mixer_mixer_l32_224_img_cls_timm"], "pcc": 0.99}),
    (
        Add169,
        [((1, 1024, 196), torch.float32)],
        {
            "model_name": ["pt_mlp_mixer_mixer_l16_224_in21k_img_cls_timm", "pt_mlp_mixer_mixer_l16_224_img_cls_timm"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 196, 1024), torch.float32), ((1, 196, 1024), torch.float32)],
        {
            "model_name": ["pt_mlp_mixer_mixer_l16_224_in21k_img_cls_timm", "pt_mlp_mixer_mixer_l16_224_img_cls_timm"],
            "pcc": 0.99,
        },
    ),
    (
        Add6,
        [((1, 196, 4096), torch.float32)],
        {
            "model_name": ["pt_mlp_mixer_mixer_l16_224_in21k_img_cls_timm", "pt_mlp_mixer_mixer_l16_224_img_cls_timm"],
            "pcc": 0.99,
        },
    ),
    (
        Add4,
        [((1, 196, 1024), torch.float32)],
        {
            "model_name": ["pt_mlp_mixer_mixer_l16_224_in21k_img_cls_timm", "pt_mlp_mixer_mixer_l16_224_img_cls_timm"],
            "pcc": 0.99,
        },
    ),
    (
        Add171,
        [((1, 21843), torch.float32)],
        {
            "model_name": [
                "pt_mlp_mixer_mixer_l16_224_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_in21k_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 24, 96, 96), torch.float32), ((24, 1, 1), torch.float32)],
        {"model_name": ["pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 48, 96, 96), torch.float32), ((48, 1, 1), torch.float32)],
        {"model_name": ["pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 48, 48, 48), torch.float32), ((48, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf",
                "pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 96, 48, 48), torch.float32), ((96, 1, 1), torch.float32)],
        {"model_name": ["pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 96, 24, 24), torch.float32), ((96, 1, 1), torch.float32)],
        {"model_name": ["pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 192, 24, 24), torch.float32), ((192, 1, 1), torch.float32)],
        {"model_name": ["pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 192, 12, 12), torch.float32), ((192, 1, 1), torch.float32)],
        {"model_name": ["pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 384, 12, 12), torch.float32), ((384, 1, 1), torch.float32)],
        {"model_name": ["pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 384, 6, 6), torch.float32), ((384, 1, 1), torch.float32)],
        {"model_name": ["pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 768, 6, 6), torch.float32), ((768, 1, 1), torch.float32)],
        {"model_name": ["pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add172,
        [((1, 1001), torch.float32)],
        {
            "model_name": [
                "pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf",
                "pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf",
                "pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (Add29, [((1, 9), torch.float32)], {"model_name": ["pt_mobilenet_v1_basic_img_cls_torchvision"], "pcc": 0.99}),
    (
        Add0,
        [((1, 16, 48, 48), torch.float32), ((16, 1, 1), torch.float32)],
        {"model_name": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 8, 48, 48), torch.float32), ((8, 1, 1), torch.float32)],
        {"model_name": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 48, 24, 24), torch.float32), ((48, 1, 1), torch.float32)],
        {"model_name": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 8, 24, 24), torch.float32), ((8, 1, 1), torch.float32)],
        {"model_name": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 8, 24, 24), torch.float32), ((1, 8, 24, 24), torch.float32)],
        {"model_name": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 48, 12, 12), torch.float32), ((48, 1, 1), torch.float32)],
        {"model_name": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 16, 12, 12), torch.float32), ((16, 1, 1), torch.float32)],
        {"model_name": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 96, 12, 12), torch.float32), ((96, 1, 1), torch.float32)],
        {"model_name": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 16, 12, 12), torch.float32), ((1, 16, 12, 12), torch.float32)],
        {"model_name": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 96, 6, 6), torch.float32), ((96, 1, 1), torch.float32)],
        {"model_name": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 24, 6, 6), torch.float32), ((24, 1, 1), torch.float32)],
        {"model_name": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 144, 6, 6), torch.float32), ((144, 1, 1), torch.float32)],
        {"model_name": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 24, 6, 6), torch.float32), ((1, 24, 6, 6), torch.float32)],
        {"model_name": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 32, 6, 6), torch.float32), ((32, 1, 1), torch.float32)],
        {"model_name": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 192, 6, 6), torch.float32), ((192, 1, 1), torch.float32)],
        {"model_name": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 32, 6, 6), torch.float32), ((1, 32, 6, 6), torch.float32)],
        {"model_name": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 192, 3, 3), torch.float32), ((192, 1, 1), torch.float32)],
        {"model_name": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 56, 3, 3), torch.float32), ((56, 1, 1), torch.float32)],
        {"model_name": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 336, 3, 3), torch.float32), ((336, 1, 1), torch.float32)],
        {"model_name": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 56, 3, 3), torch.float32), ((1, 56, 3, 3), torch.float32)],
        {"model_name": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 112, 3, 3), torch.float32), ((112, 1, 1), torch.float32)],
        {"model_name": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1280, 3, 3), torch.float32), ((1280, 1, 1), torch.float32)],
        {"model_name": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 96, 14, 14), torch.float32), ((1, 96, 14, 14), torch.float32)],
        {
            "model_name": [
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 24, 80, 80), torch.float32), ((24, 1, 1), torch.float32)],
        {"model_name": ["pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 16, 80, 80), torch.float32), ((16, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                "pt_yolo_v5_yolov5n_imgcls_torchhub_320x320",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 96, 80, 80), torch.float32), ((96, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                "pt_yolo_v5_yolov5m_imgcls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_imgcls_torchhub_640x640",
                "pt_yolox_yolox_m_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 96, 40, 40), torch.float32), ((96, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                "pt_yolo_v5_yolov5m_imgcls_torchhub_320x320",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 24, 40, 40), torch.float32), ((24, 1, 1), torch.float32)],
        {"model_name": ["pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 144, 40, 40), torch.float32), ((144, 1, 1), torch.float32)],
        {"model_name": ["pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 24, 40, 40), torch.float32), ((1, 24, 40, 40), torch.float32)],
        {"model_name": ["pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 144, 20, 20), torch.float32), ((144, 1, 1), torch.float32)],
        {"model_name": ["pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 24, 20, 20), torch.float32), ((24, 1, 1), torch.float32)],
        {"model_name": ["pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 24, 20, 20), torch.float32), ((1, 24, 20, 20), torch.float32)],
        {"model_name": ["pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 144, 10, 10), torch.float32), ((144, 1, 1), torch.float32)],
        {"model_name": ["pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 48, 10, 10), torch.float32), ((48, 1, 1), torch.float32)],
        {"model_name": ["pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 288, 10, 10), torch.float32), ((288, 1, 1), torch.float32)],
        {"model_name": ["pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 48, 10, 10), torch.float32), ((1, 48, 10, 10), torch.float32)],
        {"model_name": ["pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 72, 10, 10), torch.float32), ((72, 1, 1), torch.float32)],
        {"model_name": ["pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 432, 10, 10), torch.float32), ((432, 1, 1), torch.float32)],
        {"model_name": ["pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 72, 10, 10), torch.float32), ((1, 72, 10, 10), torch.float32)],
        {"model_name": ["pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 432, 5, 5), torch.float32), ((432, 1, 1), torch.float32)],
        {"model_name": ["pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 120, 5, 5), torch.float32), ((120, 1, 1), torch.float32)],
        {"model_name": ["pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 720, 5, 5), torch.float32), ((720, 1, 1), torch.float32)],
        {"model_name": ["pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 120, 5, 5), torch.float32), ((1, 120, 5, 5), torch.float32)],
        {"model_name": ["pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 240, 5, 5), torch.float32), ((240, 1, 1), torch.float32)],
        {"model_name": ["pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1280, 5, 5), torch.float32), ((1280, 1, 1), torch.float32)],
        {"model_name": ["pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 960, 28, 28), torch.float32), ((960, 1, 1), torch.float32)],
        {"model_name": ["pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 160, 28, 28), torch.float32), ((1, 160, 28, 28), torch.float32)],
        {"model_name": ["pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 256, 1, 1), torch.float32), ((256, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_ssd300_resnet50_base_img_cls_torchhub",
                "pt_vovnet_ese_vovnet39b_obj_det_torchhub",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 21, 28, 28), torch.float32), ((21, 1, 1), torch.float32)],
        {"model_name": ["pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add1,
        [((1, 16, 112, 112), torch.float32)],
        {
            "model_name": [
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 16, 1, 1), torch.float32), ((16, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_ssd300_resnet50_base_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    pytest.param(
        (
            Add1,
            [((1, 16, 1, 1), torch.float32)],
            {
                "model_name": [
                    "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                    "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
                ],
                "pcc": 0.99,
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Add0,
        [((1, 72, 56, 56), torch.float32), ((72, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 24, 28, 28), torch.float32), ((1, 24, 28, 28), torch.float32)],
        {
            "model_name": [
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add1,
        [((1, 96, 28, 28), torch.float32)],
        {
            "model_name": [
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add1,
        [((1, 96, 14, 14), torch.float32)],
        {
            "model_name": [
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    pytest.param(
        (
            Add1,
            [((1, 96, 1, 1), torch.float32)],
            {
                "model_name": [
                    "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                    "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
                ],
                "pcc": 0.99,
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Add1,
        [((1, 240, 14, 14), torch.float32)],
        {
            "model_name": [
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 64, 1, 1), torch.float32), ((64, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    pytest.param(
        (
            Add1,
            [((1, 240, 1, 1), torch.float32)],
            {
                "model_name": [
                    "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                    "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
                ],
                "pcc": 0.99,
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Add0,
        [((1, 40, 14, 14), torch.float32), ((1, 40, 14, 14), torch.float32)],
        {
            "model_name": [
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add1,
        [((1, 120, 14, 14), torch.float32)],
        {
            "model_name": [
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 144, 14, 14), torch.float32), ((144, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add1,
        [((1, 144, 14, 14), torch.float32)],
        {
            "model_name": [
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    pytest.param(
        (
            Add1,
            [((1, 144, 1, 1), torch.float32)],
            {
                "model_name": [
                    "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                    "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
                ],
                "pcc": 0.99,
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Add0,
        [((1, 48, 14, 14), torch.float32), ((1, 48, 14, 14), torch.float32)],
        {
            "model_name": [
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add1,
        [((1, 288, 14, 14), torch.float32)],
        {
            "model_name": [
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 288, 7, 7), torch.float32), ((288, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add1,
        [((1, 288, 7, 7), torch.float32)],
        {
            "model_name": [
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 288, 1, 1), torch.float32), ((288, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    pytest.param(
        (
            Add1,
            [((1, 288, 1, 1), torch.float32)],
            {
                "model_name": [
                    "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                    "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
                ],
                "pcc": 0.99,
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Add0,
        [((1, 96, 7, 7), torch.float32), ((96, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add1,
        [((1, 576, 7, 7), torch.float32)],
        {
            "model_name": [
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 576, 1, 1), torch.float32), ((576, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    pytest.param(
        (
            Add1,
            [((1, 576, 1, 1), torch.float32)],
            {
                "model_name": [
                    "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                    "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
                ],
                "pcc": 0.99,
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Add0,
        [((1, 96, 7, 7), torch.float32), ((1, 96, 7, 7), torch.float32)],
        {
            "model_name": [
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add4,
        [((1, 1024), torch.float32)],
        {"model_name": ["pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Add1,
        [((1, 1024), torch.float32)],
        {"model_name": ["pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1024, 1, 1), torch.float32), ((1024, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_vovnet_ese_vovnet39b_obj_det_torchhub",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    pytest.param(
        (
            Add1,
            [((1, 1024, 1, 1), torch.float32)],
            {
                "model_name": [
                    "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
                    "pt_vovnet_ese_vovnet39b_obj_det_torchhub",
                    "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
                    "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                ],
                "pcc": 0.99,
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Add1,
        [((1, 240, 28, 28), torch.float32)],
        {
            "model_name": [
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((200,), torch.float32), ((1,), torch.float32)],
        {
            "model_name": [
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add173,
        [((200,), torch.float32)],
        {
            "model_name": [
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 200, 14, 14), torch.float32), ((200, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add1,
        [((1, 200, 14, 14), torch.float32)],
        {
            "model_name": [
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((184,), torch.float32), ((1,), torch.float32)],
        {
            "model_name": [
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add174,
        [((184,), torch.float32)],
        {
            "model_name": [
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 184, 14, 14), torch.float32), ((184, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add1,
        [((1, 184, 14, 14), torch.float32)],
        {
            "model_name": [
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add1,
        [((1, 480, 14, 14), torch.float32)],
        {
            "model_name": [
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add1,
        [((1, 672, 14, 14), torch.float32)],
        {
            "model_name": [
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add1,
        [((1, 672, 7, 7), torch.float32)],
        {
            "model_name": [
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add1,
        [((1, 960, 7, 7), torch.float32)],
        {
            "model_name": [
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    pytest.param(
        (
            Add1,
            [((1, 1280, 1, 1), torch.float32)],
            {"model_name": ["pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm"], "pcc": 0.99},
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Add7,
        [((1, 1280), torch.float32)],
        {"model_name": ["pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Add1,
        [((1, 1280), torch.float32)],
        {"model_name": ["pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 64, 160, 512), torch.float32), ((64, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_monodepth2_mono_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_stereo_1024x320_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 64, 80, 256), torch.float32), ((64, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_monodepth2_mono_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_stereo_1024x320_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 64, 80, 256), torch.float32), ((1, 64, 80, 256), torch.float32)],
        {
            "model_name": [
                "pt_monodepth2_mono_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_stereo_1024x320_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 128, 40, 128), torch.float32), ((128, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_monodepth2_mono_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_stereo_1024x320_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 128, 40, 128), torch.float32), ((1, 128, 40, 128), torch.float32)],
        {
            "model_name": [
                "pt_monodepth2_mono_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_stereo_1024x320_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 256, 20, 64), torch.float32), ((256, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_monodepth2_mono_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_stereo_1024x320_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 256, 20, 64), torch.float32), ((1, 256, 20, 64), torch.float32)],
        {
            "model_name": [
                "pt_monodepth2_mono_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_stereo_1024x320_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 512, 10, 32), torch.float32), ((512, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_monodepth2_mono_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_stereo_1024x320_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 512, 10, 32), torch.float32), ((1, 512, 10, 32), torch.float32)],
        {
            "model_name": [
                "pt_monodepth2_mono_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_stereo_1024x320_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 256, 10, 32), torch.float32), ((256, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_monodepth2_mono_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_stereo_1024x320_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 128, 20, 64), torch.float32), ((128, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_monodepth2_mono_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_stereo_1024x320_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 64, 40, 128), torch.float32), ((64, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_monodepth2_mono_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_stereo_1024x320_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 32, 80, 256), torch.float32), ((32, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_monodepth2_mono_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_stereo_1024x320_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 32, 160, 512), torch.float32), ((32, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_monodepth2_mono_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_stereo_1024x320_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 16, 160, 512), torch.float32), ((16, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_monodepth2_mono_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_stereo_1024x320_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 16, 320, 1024), torch.float32), ((16, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_monodepth2_mono_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_stereo_1024x320_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1, 320, 1024), torch.float32), ((1, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_monodepth2_mono_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_stereo_1024x320_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 64, 96, 320), torch.float32), ((64, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 64, 48, 160), torch.float32), ((64, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 64, 48, 160), torch.float32), ((1, 64, 48, 160), torch.float32)],
        {
            "model_name": [
                "pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 128, 24, 80), torch.float32), ((128, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 128, 24, 80), torch.float32), ((1, 128, 24, 80), torch.float32)],
        {
            "model_name": [
                "pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 256, 12, 40), torch.float32), ((256, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 256, 12, 40), torch.float32), ((1, 256, 12, 40), torch.float32)],
        {
            "model_name": [
                "pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 512, 6, 20), torch.float32), ((512, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 512, 6, 20), torch.float32), ((1, 512, 6, 20), torch.float32)],
        {
            "model_name": [
                "pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 256, 6, 20), torch.float32), ((256, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 128, 12, 40), torch.float32), ((128, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 64, 24, 80), torch.float32), ((64, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 32, 48, 160), torch.float32), ((32, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 32, 96, 320), torch.float32), ((32, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 16, 96, 320), torch.float32), ((16, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 16, 192, 640), torch.float32), ((16, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1, 192, 640), torch.float32), ((1, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 3, 56, 56), torch.float32), ((3, 1, 1), torch.float32)],
        {"model_name": ["pt_monodle_base_obj_det_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 2, 56, 56), torch.float32), ((2, 1, 1), torch.float32)],
        {"model_name": ["pt_monodle_base_obj_det_torchvision"], "pcc": 0.99},
    ),
    (
        Add175,
        [((1, 512, 322), torch.float32)],
        {"model_name": ["pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add175,
        [((1, 3025, 322), torch.float32)],
        {"model_name": ["pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add176,
        [((1, 1, 512, 3025), torch.float32)],
        {"model_name": ["pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add4,
        [((1, 512, 1024), torch.float32)],
        {
            "model_name": [
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 512, 1024), torch.float32), ((1, 512, 1024), torch.float32)],
        {
            "model_name": [
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add66,
        [((1, 1, 1000), torch.float32)],
        {
            "model_name": [
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add177,
        [((1, 512, 261), torch.float32)],
        {"model_name": ["pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add177,
        [((1, 50176, 261), torch.float32)],
        {"model_name": ["pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add178,
        [((1, 1, 512, 50176), torch.float32)],
        {
            "model_name": [
                "pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add12,
        [((1, 512, 512), torch.float32)],
        {"model_name": ["pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 256, 224, 224), torch.float32), ((256, 1, 1), torch.float32)],
        {"model_name": ["pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add54,
        [((1, 50176, 256), torch.float32)],
        {"model_name": ["pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add12,
        [((1, 50176, 512), torch.float32)],
        {"model_name": ["pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf"], "pcc": 0.99},
    ),
    (Add179, [((1, 4096), torch.float32)], {"model_name": ["pt_rcnn_base_obj_det_torchvision_rect_0"], "pcc": 0.99}),
    (Add180, [((1, 2), torch.float32)], {"model_name": ["pt_rcnn_base_obj_det_torchvision_rect_0"], "pcc": 0.99}),
    (
        Add0,
        [((1, 128, 1, 1), torch.float32), ((128, 1, 1), torch.float32)],
        {"model_name": ["pt_regnet_facebook_regnet_y_040_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 192, 28, 28), torch.float32), ((1, 192, 28, 28), torch.float32)],
        {"model_name": ["pt_regnet_facebook_regnet_y_040_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 512, 1, 1), torch.float32), ((512, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_regnet_facebook_regnet_y_040_img_cls_hf",
                "pt_vovnet_ese_vovnet39b_obj_det_torchhub",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1088, 1, 1), torch.float32), ((1088, 1, 1), torch.float32)],
        {"model_name": ["pt_regnet_facebook_regnet_y_040_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1088, 7, 7), torch.float32), ((1, 1088, 7, 7), torch.float32)],
        {"model_name": ["pt_regnet_facebook_regnet_y_040_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 272, 1, 1), torch.float32), ((272, 1, 1), torch.float32)],
        {"model_name": ["pt_regnet_facebook_regnet_y_040_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 512, 28, 28), torch.float32), ((1, 512, 28, 28), torch.float32)],
        {
            "model_name": [
                "ResNetForImageClassification",
                "ResNet",
                "pt_resnet_50_img_cls_timm",
                "pt_resnext_resnext26_32x4d_img_cls_osmr",
                "pt_resnext_resnext101_64x4d_img_cls_osmr",
                "pt_resnext_resnext50_32x4d_img_cls_osmr",
                "pt_resnext_resnext50_32x4d_img_cls_torchhub",
                "pt_resnext_resnext14_32x4d_img_cls_osmr",
                "pt_resnext_resnext101_32x8d_img_cls_torchhub",
                "pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub",
                "pt_unet_qubvel_img_seg_torchhub",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_wideresnet_wide_resnet101_2_img_cls_torchvision",
                "pt_wideresnet_wide_resnet50_2_img_cls_timm",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
                "pt_wideresnet_wide_resnet50_2_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1024, 14, 14), torch.float32), ((1, 1024, 14, 14), torch.float32)],
        {
            "model_name": [
                "ResNetForImageClassification",
                "ResNet",
                "pt_resnet_50_img_cls_timm",
                "pt_resnext_resnext26_32x4d_img_cls_osmr",
                "pt_resnext_resnext101_64x4d_img_cls_osmr",
                "pt_resnext_resnext50_32x4d_img_cls_osmr",
                "pt_resnext_resnext50_32x4d_img_cls_torchhub",
                "pt_resnext_resnext14_32x4d_img_cls_osmr",
                "pt_resnext_resnext101_32x8d_img_cls_torchhub",
                "pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub",
                "pt_unet_qubvel_img_seg_torchhub",
                "pt_wideresnet_wide_resnet101_2_img_cls_torchvision",
                "pt_wideresnet_wide_resnet50_2_img_cls_timm",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
                "pt_wideresnet_wide_resnet50_2_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 2048, 7, 7), torch.float32), ((1, 2048, 7, 7), torch.float32)],
        {
            "model_name": [
                "ResNetForImageClassification",
                "ResNet",
                "pt_resnet_50_img_cls_timm",
                "pt_resnext_resnext26_32x4d_img_cls_osmr",
                "pt_resnext_resnext101_64x4d_img_cls_osmr",
                "pt_resnext_resnext50_32x4d_img_cls_osmr",
                "pt_resnext_resnext50_32x4d_img_cls_torchhub",
                "pt_resnext_resnext14_32x4d_img_cls_osmr",
                "pt_resnext_resnext101_32x8d_img_cls_torchhub",
                "pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub",
                "pt_unet_qubvel_img_seg_torchhub",
                "pt_wideresnet_wide_resnet101_2_img_cls_torchvision",
                "pt_wideresnet_wide_resnet50_2_img_cls_timm",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
                "pt_wideresnet_wide_resnet50_2_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 64, 240, 320), torch.float32), ((64, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_retinanet_retinanet_rn101fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn152fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn34fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn18fpn_obj_det_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 64, 120, 160), torch.float32), ((64, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_retinanet_retinanet_rn101fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn152fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn34fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn18fpn_obj_det_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 256, 120, 160), torch.float32), ((256, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_retinanet_retinanet_rn101fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn152fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 256, 120, 160), torch.float32), ((1, 256, 120, 160), torch.float32)],
        {
            "model_name": [
                "pt_retinanet_retinanet_rn101fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn152fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 128, 120, 160), torch.float32), ((128, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_retinanet_retinanet_rn101fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn152fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 128, 60, 80), torch.float32), ((128, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_retinanet_retinanet_rn101fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn152fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn34fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn18fpn_obj_det_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 512, 60, 80), torch.float32), ((512, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_retinanet_retinanet_rn101fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn152fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 512, 60, 80), torch.float32), ((1, 512, 60, 80), torch.float32)],
        {
            "model_name": [
                "pt_retinanet_retinanet_rn101fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn152fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 256, 60, 80), torch.float32), ((256, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_retinanet_retinanet_rn101fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn152fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn34fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn18fpn_obj_det_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 256, 30, 40), torch.float32), ((256, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_retinanet_retinanet_rn101fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn152fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn34fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn18fpn_obj_det_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1024, 30, 40), torch.float32), ((1024, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_retinanet_retinanet_rn101fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn152fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1024, 30, 40), torch.float32), ((1, 1024, 30, 40), torch.float32)],
        {
            "model_name": [
                "pt_retinanet_retinanet_rn101fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn152fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 512, 30, 40), torch.float32), ((512, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_retinanet_retinanet_rn101fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn152fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 512, 15, 20), torch.float32), ((512, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_retinanet_retinanet_rn101fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn152fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn34fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn18fpn_obj_det_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 2048, 15, 20), torch.float32), ((2048, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_retinanet_retinanet_rn101fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn152fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 2048, 15, 20), torch.float32), ((1, 2048, 15, 20), torch.float32)],
        {
            "model_name": [
                "pt_retinanet_retinanet_rn101fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn152fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 256, 15, 20), torch.float32), ((256, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_retinanet_retinanet_rn101fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn152fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn34fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn18fpn_obj_det_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 256, 30, 40), torch.float32), ((1, 256, 30, 40), torch.float32)],
        {
            "model_name": [
                "pt_retinanet_retinanet_rn101fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn152fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn34fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn18fpn_obj_det_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 256, 60, 80), torch.float32), ((1, 256, 60, 80), torch.float32)],
        {
            "model_name": [
                "pt_retinanet_retinanet_rn101fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn152fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn34fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn18fpn_obj_det_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 720, 60, 80), torch.float32), ((720, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_retinanet_retinanet_rn101fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn152fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn34fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn18fpn_obj_det_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 720, 30, 40), torch.float32), ((720, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_retinanet_retinanet_rn101fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn152fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn34fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn18fpn_obj_det_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 720, 15, 20), torch.float32), ((720, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_retinanet_retinanet_rn101fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn152fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn34fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn18fpn_obj_det_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 256, 8, 10), torch.float32), ((256, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_retinanet_retinanet_rn101fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn152fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn34fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn18fpn_obj_det_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 720, 8, 10), torch.float32), ((720, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_retinanet_retinanet_rn101fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn152fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn34fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn18fpn_obj_det_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 256, 4, 5), torch.float32), ((256, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_retinanet_retinanet_rn101fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn152fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn34fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn18fpn_obj_det_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 720, 4, 5), torch.float32), ((720, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_retinanet_retinanet_rn101fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn152fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn34fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn18fpn_obj_det_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 36, 60, 80), torch.float32), ((36, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_retinanet_retinanet_rn101fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn152fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn34fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn18fpn_obj_det_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 36, 30, 40), torch.float32), ((36, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_retinanet_retinanet_rn101fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn152fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn34fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn18fpn_obj_det_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 36, 15, 20), torch.float32), ((36, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_retinanet_retinanet_rn101fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn152fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn34fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn18fpn_obj_det_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 36, 8, 10), torch.float32), ((36, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_retinanet_retinanet_rn101fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn152fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn34fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn18fpn_obj_det_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 36, 4, 5), torch.float32), ((36, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_retinanet_retinanet_rn101fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn152fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn34fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn18fpn_obj_det_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 64, 120, 160), torch.float32), ((1, 64, 120, 160), torch.float32)],
        {
            "model_name": ["pt_retinanet_retinanet_rn34fpn_obj_det_hf", "pt_retinanet_retinanet_rn18fpn_obj_det_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 128, 60, 80), torch.float32), ((1, 128, 60, 80), torch.float32)],
        {
            "model_name": ["pt_retinanet_retinanet_rn34fpn_obj_det_hf", "pt_retinanet_retinanet_rn18fpn_obj_det_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 512, 15, 20), torch.float32), ((1, 512, 15, 20), torch.float32)],
        {
            "model_name": ["pt_retinanet_retinanet_rn34fpn_obj_det_hf", "pt_retinanet_retinanet_rn18fpn_obj_det_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 32, 128, 128), torch.float32), ((32, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add147,
        [((1, 16384, 32), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 32, 16, 16), torch.float32), ((32, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add147,
        [((1, 256, 32), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 16384, 32), torch.float32), ((1, 16384, 32), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add23,
        [((1, 16384, 128), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 128, 128, 128), torch.float32), ((128, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 64, 64, 64), torch.float32), ((64, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add67,
        [((1, 4096, 64), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 64, 16, 16), torch.float32), ((64, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add67,
        [((1, 256, 64), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 4096, 64), torch.float32), ((1, 4096, 64), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add54,
        [((1, 4096, 256), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 160, 32, 32), torch.float32), ((160, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add114,
        [((1, 1024, 160), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 160, 16, 16), torch.float32), ((160, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add114,
        [((1, 256, 160), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1024, 160), torch.float32), ((1, 1024, 160), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add122,
        [((1, 1024, 640), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 640, 32, 32), torch.float32), ((640, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add54,
        [((1, 256, 256), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 256, 256), torch.float32), ((1, 256, 256), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1024, 16, 16), torch.float32), ((1024, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 64, 128, 128), torch.float32), ((64, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_unet_base_img_seg_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add67,
        [((1, 16384, 64), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 16384, 64), torch.float32), ((1, 16384, 64), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add54,
        [((1, 16384, 256), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 256, 128, 128), torch.float32), ((256, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 128, 64, 64), torch.float32), ((128, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_unet_base_img_seg_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add23,
        [((1, 4096, 128), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 128, 16, 16), torch.float32), ((128, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add23,
        [((1, 256, 128), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 4096, 128), torch.float32), ((1, 4096, 128), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add12,
        [((1, 4096, 512), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 512, 64, 64), torch.float32), ((512, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 320, 32, 32), torch.float32), ((320, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add116,
        [((1, 1024, 320), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 320, 16, 16), torch.float32), ((320, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add116,
        [((1, 256, 320), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1024, 320), torch.float32), ((1, 1024, 320), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add7,
        [((1, 1024, 1280), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1280, 32, 32), torch.float32), ((1280, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 512, 16, 16), torch.float32), ((512, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_unet_base_img_seg_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add12,
        [((1, 256, 512), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 256, 512), torch.float32), ((1, 256, 512), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 2048, 16, 16), torch.float32), ((2048, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add14,
        [((1, 1024, 768), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add14,
        [((1, 4096, 768), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add14,
        [((1, 16384, 768), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 768, 128, 128), torch.float32), ((768, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 150, 128, 128), torch.float32), ((150, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add54,
        [((1, 1024, 256), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 64, 150, 150), torch.float32), ((64, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_ssd300_resnet50_base_img_cls_torchhub",
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception65_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 64, 75, 75), torch.float32), ((64, 1, 1), torch.float32)],
        {"model_name": ["pt_ssd300_resnet50_base_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 256, 75, 75), torch.float32), ((256, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_ssd300_resnet50_base_img_cls_torchhub",
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception65_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 256, 75, 75), torch.float32), ((1, 256, 75, 75), torch.float32)],
        {
            "model_name": ["pt_ssd300_resnet50_base_img_cls_torchhub", "pt_xception_xception71_img_cls_timm"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 128, 75, 75), torch.float32), ((128, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_ssd300_resnet50_base_img_cls_torchhub",
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception65_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 128, 38, 38), torch.float32), ((128, 1, 1), torch.float32)],
        {"model_name": ["pt_ssd300_resnet50_base_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 512, 38, 38), torch.float32), ((512, 1, 1), torch.float32)],
        {"model_name": ["pt_ssd300_resnet50_base_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 512, 38, 38), torch.float32), ((1, 512, 38, 38), torch.float32)],
        {"model_name": ["pt_ssd300_resnet50_base_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 256, 38, 38), torch.float32), ((256, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_ssd300_resnet50_base_img_cls_torchhub",
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception65_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1024, 38, 38), torch.float32), ((1024, 1, 1), torch.float32)],
        {"model_name": ["pt_ssd300_resnet50_base_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1024, 38, 38), torch.float32), ((1, 1024, 38, 38), torch.float32)],
        {"model_name": ["pt_ssd300_resnet50_base_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 16, 38, 38), torch.float32), ((16, 1, 1), torch.float32)],
        {"model_name": ["pt_ssd300_resnet50_base_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 512, 19, 19), torch.float32), ((512, 1, 1), torch.float32)],
        {"model_name": ["pt_ssd300_resnet50_base_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 24, 19, 19), torch.float32), ((24, 1, 1), torch.float32)],
        {"model_name": ["pt_ssd300_resnet50_base_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 256, 19, 19), torch.float32), ((256, 1, 1), torch.float32)],
        {"model_name": ["pt_ssd300_resnet50_base_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 512, 10, 10), torch.float32), ((512, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_ssd300_resnet50_base_img_cls_torchhub",
                "pt_yolo_v5_yolov5l_imgcls_torchhub_320x320",
                "pt_yolo_v5_yolov5s_imgcls_torchhub_320x320",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 24, 10, 10), torch.float32), ((24, 1, 1), torch.float32)],
        {"model_name": ["pt_ssd300_resnet50_base_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 128, 10, 10), torch.float32), ((128, 1, 1), torch.float32)],
        {
            "model_name": ["pt_ssd300_resnet50_base_img_cls_torchhub", "pt_yolo_v5_yolov5n_imgcls_torchhub_320x320"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 256, 5, 5), torch.float32), ((256, 1, 1), torch.float32)],
        {"model_name": ["pt_ssd300_resnet50_base_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 24, 5, 5), torch.float32), ((24, 1, 1), torch.float32)],
        {"model_name": ["pt_ssd300_resnet50_base_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 128, 5, 5), torch.float32), ((128, 1, 1), torch.float32)],
        {"model_name": ["pt_ssd300_resnet50_base_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 256, 3, 3), torch.float32), ((256, 1, 1), torch.float32)],
        {"model_name": ["pt_ssd300_resnet50_base_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 16, 3, 3), torch.float32), ((16, 1, 1), torch.float32)],
        {"model_name": ["pt_ssd300_resnet50_base_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 128, 3, 3), torch.float32), ((128, 1, 1), torch.float32)],
        {"model_name": ["pt_ssd300_resnet50_base_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 324, 38, 38), torch.float32), ((324, 1, 1), torch.float32)],
        {"model_name": ["pt_ssd300_resnet50_base_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 486, 19, 19), torch.float32), ((486, 1, 1), torch.float32)],
        {"model_name": ["pt_ssd300_resnet50_base_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 486, 10, 10), torch.float32), ((486, 1, 1), torch.float32)],
        {"model_name": ["pt_ssd300_resnet50_base_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 486, 5, 5), torch.float32), ((486, 1, 1), torch.float32)],
        {"model_name": ["pt_ssd300_resnet50_base_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 324, 3, 3), torch.float32), ((324, 1, 1), torch.float32)],
        {"model_name": ["pt_ssd300_resnet50_base_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 324, 1, 1), torch.float32), ((324, 1, 1), torch.float32)],
        {"model_name": ["pt_ssd300_resnet50_base_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Add65,
        [((64, 49, 96), torch.float32)],
        {"model_name": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((64, 3, 49, 49), torch.float32), ((1, 3, 49, 49), torch.float32)],
        {"model_name": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 3136, 96), torch.float32), ((1, 3136, 96), torch.float32)],
        {"model_name": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add10,
        [((1, 3136, 384), torch.float32)],
        {"model_name": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add65,
        [((1, 3136, 96), torch.float32)],
        {"model_name": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add181,
        [((1, 64, 3, 49, 49), torch.float32)],
        {"model_name": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add72,
        [((16, 49, 192), torch.float32)],
        {"model_name": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((16, 6, 49, 49), torch.float32), ((1, 6, 49, 49), torch.float32)],
        {"model_name": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 784, 192), torch.float32), ((1, 784, 192), torch.float32)],
        {"model_name": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add14,
        [((1, 784, 768), torch.float32)],
        {"model_name": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add72,
        [((1, 784, 192), torch.float32)],
        {"model_name": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add182,
        [((1, 16, 6, 49, 49), torch.float32)],
        {"model_name": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add10,
        [((4, 49, 384), torch.float32)],
        {"model_name": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((4, 12, 49, 49), torch.float32), ((1, 12, 49, 49), torch.float32)],
        {"model_name": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 196, 384), torch.float32), ((1, 196, 384), torch.float32)],
        {"model_name": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add3,
        [((1, 196, 1536), torch.float32)],
        {"model_name": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add10,
        [((1, 196, 384), torch.float32)],
        {"model_name": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add183,
        [((1, 4, 12, 49, 49), torch.float32)],
        {"model_name": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 24, 49, 49), torch.float32), ((1, 24, 49, 49), torch.float32)],
        {"model_name": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 32, 256, 256), torch.float32), ((32, 1, 1), torch.float32)],
        {"model_name": ["pt_unet_base_img_seg_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 256, 32, 32), torch.float32), ((256, 1, 1), torch.float32)],
        {"model_name": ["pt_unet_base_img_seg_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1, 256, 256), torch.float32), ((1, 1, 1), torch.float32)],
        {"model_name": ["pt_unet_base_img_seg_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 64, 224, 224), torch.float32), ((64, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_unet_cityscape_img_seg_osmr",
                "pt_vgg_bn_vgg19_obj_det_osmr",
                "pt_vgg_19_obj_det_hf",
                "pt_vgg_vgg19_bn_obj_det_torchhub",
                "pt_vgg_vgg11_obj_det_osmr",
                "pt_vgg_vgg19_bn_obj_det_timm",
                "pt_vgg_bn_vgg19b_obj_det_osmr",
                "pt_vgg_vgg16_obj_det_osmr",
                "pt_vgg_vgg19_obj_det_osmr",
                "pt_vgg_vgg13_obj_det_osmr",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 19, 224, 224), torch.float32), ((19, 1, 1), torch.float32)],
        {"model_name": ["pt_unet_cityscape_img_seg_osmr"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1, 224, 224), torch.float32), ((1, 1, 1), torch.float32)],
        {"model_name": ["pt_unet_qubvel_img_seg_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 4096, 1, 1), torch.float32), ((4096, 1, 1), torch.float32)],
        {"model_name": ["pt_vgg_vgg19_bn_obj_det_timm"], "pcc": 0.99},
    ),
    (
        Add184,
        [((1, 197, 1024), torch.float32)],
        {"model_name": ["pt_vit_google_vit_large_patch16_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add4,
        [((1, 197, 1024), torch.float32)],
        {"model_name": ["pt_vit_google_vit_large_patch16_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 197, 1024), torch.float32), ((1, 197, 1024), torch.float32)],
        {"model_name": ["pt_vit_google_vit_large_patch16_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add6,
        [((1, 197, 4096), torch.float32)],
        {"model_name": ["pt_vit_google_vit_large_patch16_224_img_cls_hf"], "pcc": 0.99},
    ),
    pytest.param(
        (
            Add1,
            [((1, 256, 1, 1), torch.float32)],
            {
                "model_name": [
                    "pt_vovnet_ese_vovnet39b_obj_det_torchhub",
                    "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
                    "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                ],
                "pcc": 0.99,
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Add1,
            [((1, 512, 1, 1), torch.float32)],
            {
                "model_name": [
                    "pt_vovnet_ese_vovnet39b_obj_det_torchhub",
                    "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
                    "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                ],
                "pcc": 0.99,
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Add0,
        [((1, 768, 1, 1), torch.float32), ((768, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_vovnet_ese_vovnet39b_obj_det_torchhub",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    pytest.param(
        (
            Add1,
            [((1, 768, 1, 1), torch.float32)],
            {
                "model_name": [
                    "pt_vovnet_ese_vovnet39b_obj_det_torchhub",
                    "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
                    "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                ],
                "pcc": 0.99,
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Add0,
        [((1, 768, 14, 14), torch.float32), ((1, 768, 14, 14), torch.float32)],
        {
            "model_name": [
                "pt_vovnet_ese_vovnet39b_obj_det_torchhub",
                "pt_vovnet_vovnet39_obj_det_osmr",
                "pt_vovnet_vovnet57_obj_det_osmr",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 128, 147, 147), torch.float32), ((128, 1, 1), torch.float32)],
        {"model_name": ["pt_xception_xception_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 128, 74, 74), torch.float32), ((128, 1, 1), torch.float32)],
        {"model_name": ["pt_xception_xception_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 128, 74, 74), torch.float32), ((1, 128, 74, 74), torch.float32)],
        {"model_name": ["pt_xception_xception_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 256, 74, 74), torch.float32), ((256, 1, 1), torch.float32)],
        {"model_name": ["pt_xception_xception_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 256, 37, 37), torch.float32), ((256, 1, 1), torch.float32)],
        {"model_name": ["pt_xception_xception_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 256, 37, 37), torch.float32), ((1, 256, 37, 37), torch.float32)],
        {"model_name": ["pt_xception_xception_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((728,), torch.float32), ((1,), torch.float32)],
        {
            "model_name": [
                "pt_xception_xception_img_cls_timm",
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception65_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add185,
        [((728,), torch.float32)],
        {
            "model_name": [
                "pt_xception_xception_img_cls_timm",
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception65_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 728, 37, 37), torch.float32), ((728, 1, 1), torch.float32)],
        {"model_name": ["pt_xception_xception_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 728, 19, 19), torch.float32), ((728, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_xception_xception_img_cls_timm",
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception65_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 728, 19, 19), torch.float32), ((1, 728, 19, 19), torch.float32)],
        {
            "model_name": [
                "pt_xception_xception_img_cls_timm",
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception65_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1024, 19, 19), torch.float32), ((1024, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_xception_xception_img_cls_timm",
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception65_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1024, 10, 10), torch.float32), ((1024, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_xception_xception_img_cls_timm",
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception65_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
                "pt_yolo_v5_yolov5l_imgcls_torchhub_320x320",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1024, 10, 10), torch.float32), ((1, 1024, 10, 10), torch.float32)],
        {
            "model_name": [
                "pt_xception_xception_img_cls_timm",
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception65_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1536, 10, 10), torch.float32), ((1536, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_xception_xception_img_cls_timm",
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception65_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 2048, 10, 10), torch.float32), ((2048, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_xception_xception_img_cls_timm",
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception65_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 32, 150, 150), torch.float32), ((32, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception65_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 128, 150, 150), torch.float32), ((128, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception65_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 128, 75, 75), torch.float32), ((1, 128, 75, 75), torch.float32)],
        {
            "model_name": [
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception65_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 256, 38, 38), torch.float32), ((1, 256, 38, 38), torch.float32)],
        {
            "model_name": [
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception65_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 728, 38, 38), torch.float32), ((728, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception65_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 728, 38, 38), torch.float32), ((1, 728, 38, 38), torch.float32)],
        {"model_name": ["pt_xception_xception71_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 32, 640, 640), torch.float32), ((32, 1, 1), torch.float32)],
        {
            "model_name": ["pt_yolo_v5_yolov5s_imgcls_torchhub_1280x1280", "pt_yolox_yolox_darknet_obj_det_torchhub"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 64, 320, 320), torch.float32), ((64, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v5_yolov5s_imgcls_torchhub_1280x1280",
                "pt_yolo_v5_yolov5l_imgcls_torchhub_640x640",
                "pt_yolox_yolox_l_obj_det_torchhub",
                "pt_yolox_yolox_darknet_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 32, 320, 320), torch.float32), ((32, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v5_yolov5s_imgcls_torchhub_1280x1280",
                "pt_yolo_v5_yolov5s_imgcls_torchhub_640x640",
                "pt_yolox_yolox_s_obj_det_torchhub",
                "pt_yolox_yolox_darknet_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 32, 320, 320), torch.float32), ((1, 32, 320, 320), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5s_imgcls_torchhub_1280x1280"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 128, 160, 160), torch.float32), ((128, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v5_yolov5s_imgcls_torchhub_1280x1280",
                "pt_yolo_v5_yolov5l_imgcls_torchhub_640x640",
                "pt_yolox_yolox_l_obj_det_torchhub",
                "pt_yolox_yolox_darknet_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 64, 160, 160), torch.float32), ((64, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v5_yolov5s_imgcls_torchhub_1280x1280",
                "pt_yolo_v5_yolov5s_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5l_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5l_imgcls_torchhub_320x320",
                "pt_yolox_yolox_l_obj_det_torchhub",
                "pt_yolox_yolox_s_obj_det_torchhub",
                "pt_yolox_yolox_darknet_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 64, 160, 160), torch.float32), ((1, 64, 160, 160), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v5_yolov5s_imgcls_torchhub_1280x1280",
                "pt_yolo_v5_yolov5l_imgcls_torchhub_640x640",
                "pt_yolox_yolox_l_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 256, 80, 80), torch.float32), ((256, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v5_yolov5s_imgcls_torchhub_1280x1280",
                "pt_yolo_v5_yolov5l_imgcls_torchhub_640x640",
                "pt_yolox_yolox_l_obj_det_torchhub",
                "pt_yolox_yolox_darknet_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 128, 80, 80), torch.float32), ((128, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v5_yolov5s_imgcls_torchhub_1280x1280",
                "pt_yolo_v5_yolov5s_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5l_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5l_imgcls_torchhub_320x320",
                "pt_yolox_yolox_l_obj_det_torchhub",
                "pt_yolox_yolox_s_obj_det_torchhub",
                "pt_yolox_yolox_darknet_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 128, 80, 80), torch.float32), ((1, 128, 80, 80), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v5_yolov5s_imgcls_torchhub_1280x1280",
                "pt_yolo_v5_yolov5l_imgcls_torchhub_640x640",
                "pt_yolox_yolox_l_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 512, 40, 40), torch.float32), ((512, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v5_yolov5s_imgcls_torchhub_1280x1280",
                "pt_yolo_v5_yolov5l_imgcls_torchhub_640x640",
                "pt_yolox_yolox_l_obj_det_torchhub",
                "pt_yolox_yolox_darknet_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 256, 40, 40), torch.float32), ((256, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v5_yolov5s_imgcls_torchhub_1280x1280",
                "pt_yolo_v5_yolov5s_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5l_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5l_imgcls_torchhub_320x320",
                "pt_yolox_yolox_l_obj_det_torchhub",
                "pt_yolox_yolox_s_obj_det_torchhub",
                "pt_yolox_yolox_darknet_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 256, 40, 40), torch.float32), ((1, 256, 40, 40), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v5_yolov5s_imgcls_torchhub_1280x1280",
                "pt_yolo_v5_yolov5l_imgcls_torchhub_640x640",
                "pt_yolox_yolox_l_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 255, 160, 160), torch.float32), ((255, 1, 1), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5s_imgcls_torchhub_1280x1280"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 255, 160, 160), torch.float32), ((1, 255, 160, 160), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5s_imgcls_torchhub_1280x1280"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 255, 80, 80), torch.float32), ((255, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v5_yolov5s_imgcls_torchhub_1280x1280",
                "pt_yolo_v5_yolov5x_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5n_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5l_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5m_imgcls_torchhub_640x640",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 255, 80, 80), torch.float32), ((1, 255, 80, 80), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v5_yolov5s_imgcls_torchhub_1280x1280",
                "pt_yolo_v5_yolov5x_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5n_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5l_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5m_imgcls_torchhub_640x640",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 255, 40, 40), torch.float32), ((255, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v5_yolov5s_imgcls_torchhub_1280x1280",
                "pt_yolo_v5_yolov5x_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5m_imgcls_torchhub_320x320",
                "pt_yolo_v5_yolov5n_imgcls_torchhub_320x320",
                "pt_yolo_v5_yolov5n_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5l_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5l_imgcls_torchhub_320x320",
                "pt_yolo_v5_yolov5x_imgcls_torchhub_320x320",
                "pt_yolo_v5_yolov5s_imgcls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_imgcls_torchhub_640x640",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 255, 40, 40), torch.float32), ((1, 255, 40, 40), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v5_yolov5s_imgcls_torchhub_1280x1280",
                "pt_yolo_v5_yolov5x_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5m_imgcls_torchhub_320x320",
                "pt_yolo_v5_yolov5n_imgcls_torchhub_320x320",
                "pt_yolo_v5_yolov5n_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5l_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5l_imgcls_torchhub_320x320",
                "pt_yolo_v5_yolov5x_imgcls_torchhub_320x320",
                "pt_yolo_v5_yolov5s_imgcls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_imgcls_torchhub_640x640",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 80, 320, 320), torch.float32), ((80, 1, 1), torch.float32)],
        {
            "model_name": ["pt_yolo_v5_yolov5x_imgcls_torchhub_640x640", "pt_yolox_yolox_x_obj_det_torchhub"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 160, 160, 160), torch.float32), ((160, 1, 1), torch.float32)],
        {
            "model_name": ["pt_yolo_v5_yolov5x_imgcls_torchhub_640x640", "pt_yolox_yolox_x_obj_det_torchhub"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 80, 160, 160), torch.float32), ((80, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v5_yolov5x_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5x_imgcls_torchhub_320x320",
                "pt_yolox_yolox_x_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 80, 160, 160), torch.float32), ((1, 80, 160, 160), torch.float32)],
        {
            "model_name": ["pt_yolo_v5_yolov5x_imgcls_torchhub_640x640", "pt_yolox_yolox_x_obj_det_torchhub"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 320, 80, 80), torch.float32), ((320, 1, 1), torch.float32)],
        {
            "model_name": ["pt_yolo_v5_yolov5x_imgcls_torchhub_640x640", "pt_yolox_yolox_x_obj_det_torchhub"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 160, 80, 80), torch.float32), ((160, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v5_yolov5x_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5x_imgcls_torchhub_320x320",
                "pt_yolox_yolox_x_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 160, 80, 80), torch.float32), ((1, 160, 80, 80), torch.float32)],
        {
            "model_name": ["pt_yolo_v5_yolov5x_imgcls_torchhub_640x640", "pt_yolox_yolox_x_obj_det_torchhub"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 640, 40, 40), torch.float32), ((640, 1, 1), torch.float32)],
        {
            "model_name": ["pt_yolo_v5_yolov5x_imgcls_torchhub_640x640", "pt_yolox_yolox_x_obj_det_torchhub"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 320, 40, 40), torch.float32), ((320, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v5_yolov5x_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5x_imgcls_torchhub_320x320",
                "pt_yolox_yolox_x_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 320, 40, 40), torch.float32), ((1, 320, 40, 40), torch.float32)],
        {
            "model_name": ["pt_yolo_v5_yolov5x_imgcls_torchhub_640x640", "pt_yolox_yolox_x_obj_det_torchhub"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1280, 20, 20), torch.float32), ((1280, 1, 1), torch.float32)],
        {
            "model_name": ["pt_yolo_v5_yolov5x_imgcls_torchhub_640x640", "pt_yolox_yolox_x_obj_det_torchhub"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 640, 20, 20), torch.float32), ((640, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v5_yolov5x_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5x_imgcls_torchhub_320x320",
                "pt_yolox_yolox_x_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 640, 20, 20), torch.float32), ((1, 640, 20, 20), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5x_imgcls_torchhub_640x640"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 255, 20, 20), torch.float32), ((255, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v5_yolov5x_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5m_imgcls_torchhub_320x320",
                "pt_yolo_v5_yolov5n_imgcls_torchhub_320x320",
                "pt_yolo_v5_yolov5n_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5l_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5l_imgcls_torchhub_320x320",
                "pt_yolo_v5_yolov5x_imgcls_torchhub_320x320",
                "pt_yolo_v5_yolov5s_imgcls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_imgcls_torchhub_640x640",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 255, 20, 20), torch.float32), ((1, 255, 20, 20), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v5_yolov5x_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5m_imgcls_torchhub_320x320",
                "pt_yolo_v5_yolov5n_imgcls_torchhub_320x320",
                "pt_yolo_v5_yolov5n_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5l_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5l_imgcls_torchhub_320x320",
                "pt_yolo_v5_yolov5x_imgcls_torchhub_320x320",
                "pt_yolo_v5_yolov5s_imgcls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_imgcls_torchhub_640x640",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 80, 240, 240), torch.float32), ((80, 1, 1), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5x_imgcls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 160, 120, 120), torch.float32), ((160, 1, 1), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5x_imgcls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 80, 120, 120), torch.float32), ((80, 1, 1), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5x_imgcls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 80, 120, 120), torch.float32), ((1, 80, 120, 120), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5x_imgcls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 320, 60, 60), torch.float32), ((320, 1, 1), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5x_imgcls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 160, 60, 60), torch.float32), ((160, 1, 1), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5x_imgcls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 160, 60, 60), torch.float32), ((1, 160, 60, 60), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5x_imgcls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 640, 30, 30), torch.float32), ((640, 1, 1), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5x_imgcls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 320, 30, 30), torch.float32), ((320, 1, 1), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5x_imgcls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 320, 30, 30), torch.float32), ((1, 320, 30, 30), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5x_imgcls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1280, 15, 15), torch.float32), ((1280, 1, 1), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5x_imgcls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 640, 15, 15), torch.float32), ((640, 1, 1), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5x_imgcls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 640, 15, 15), torch.float32), ((1, 640, 15, 15), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5x_imgcls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 255, 60, 60), torch.float32), ((255, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v5_yolov5x_imgcls_torchhub_480x480",
                "pt_yolo_v5_yolov5s_imgcls_torchhub_480x480",
                "pt_yolo_v5_yolov5m_imgcls_torchhub_480x480",
                "pt_yolo_v5_yolov5l_imgcls_torchhub_480x480",
                "pt_yolo_v5_yolov5n_imgcls_torchhub_480x480",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 255, 60, 60), torch.float32), ((1, 255, 60, 60), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v5_yolov5x_imgcls_torchhub_480x480",
                "pt_yolo_v5_yolov5s_imgcls_torchhub_480x480",
                "pt_yolo_v5_yolov5m_imgcls_torchhub_480x480",
                "pt_yolo_v5_yolov5l_imgcls_torchhub_480x480",
                "pt_yolo_v5_yolov5n_imgcls_torchhub_480x480",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 255, 30, 30), torch.float32), ((255, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v5_yolov5x_imgcls_torchhub_480x480",
                "pt_yolo_v5_yolov5s_imgcls_torchhub_480x480",
                "pt_yolo_v5_yolov5m_imgcls_torchhub_480x480",
                "pt_yolo_v5_yolov5l_imgcls_torchhub_480x480",
                "pt_yolo_v5_yolov5n_imgcls_torchhub_480x480",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 255, 30, 30), torch.float32), ((1, 255, 30, 30), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v5_yolov5x_imgcls_torchhub_480x480",
                "pt_yolo_v5_yolov5s_imgcls_torchhub_480x480",
                "pt_yolo_v5_yolov5m_imgcls_torchhub_480x480",
                "pt_yolo_v5_yolov5l_imgcls_torchhub_480x480",
                "pt_yolo_v5_yolov5n_imgcls_torchhub_480x480",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 255, 15, 15), torch.float32), ((255, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v5_yolov5x_imgcls_torchhub_480x480",
                "pt_yolo_v5_yolov5s_imgcls_torchhub_480x480",
                "pt_yolo_v5_yolov5m_imgcls_torchhub_480x480",
                "pt_yolo_v5_yolov5l_imgcls_torchhub_480x480",
                "pt_yolo_v5_yolov5n_imgcls_torchhub_480x480",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 255, 15, 15), torch.float32), ((1, 255, 15, 15), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v5_yolov5x_imgcls_torchhub_480x480",
                "pt_yolo_v5_yolov5s_imgcls_torchhub_480x480",
                "pt_yolo_v5_yolov5m_imgcls_torchhub_480x480",
                "pt_yolo_v5_yolov5l_imgcls_torchhub_480x480",
                "pt_yolo_v5_yolov5n_imgcls_torchhub_480x480",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 32, 240, 240), torch.float32), ((32, 1, 1), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5s_imgcls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 64, 120, 120), torch.float32), ((64, 1, 1), torch.float32)],
        {
            "model_name": ["pt_yolo_v5_yolov5s_imgcls_torchhub_480x480", "pt_yolo_v5_yolov5l_imgcls_torchhub_480x480"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 32, 120, 120), torch.float32), ((32, 1, 1), torch.float32)],
        {
            "model_name": ["pt_yolo_v5_yolov5s_imgcls_torchhub_480x480", "pt_yolo_v5_yolov5n_imgcls_torchhub_480x480"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 32, 120, 120), torch.float32), ((1, 32, 120, 120), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5s_imgcls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 128, 60, 60), torch.float32), ((128, 1, 1), torch.float32)],
        {
            "model_name": ["pt_yolo_v5_yolov5s_imgcls_torchhub_480x480", "pt_yolo_v5_yolov5l_imgcls_torchhub_480x480"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 64, 60, 60), torch.float32), ((64, 1, 1), torch.float32)],
        {
            "model_name": ["pt_yolo_v5_yolov5s_imgcls_torchhub_480x480", "pt_yolo_v5_yolov5n_imgcls_torchhub_480x480"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 64, 60, 60), torch.float32), ((1, 64, 60, 60), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5s_imgcls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 256, 30, 30), torch.float32), ((256, 1, 1), torch.float32)],
        {
            "model_name": ["pt_yolo_v5_yolov5s_imgcls_torchhub_480x480", "pt_yolo_v5_yolov5l_imgcls_torchhub_480x480"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 128, 30, 30), torch.float32), ((128, 1, 1), torch.float32)],
        {
            "model_name": ["pt_yolo_v5_yolov5s_imgcls_torchhub_480x480", "pt_yolo_v5_yolov5n_imgcls_torchhub_480x480"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 128, 30, 30), torch.float32), ((1, 128, 30, 30), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5s_imgcls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 512, 15, 15), torch.float32), ((512, 1, 1), torch.float32)],
        {
            "model_name": ["pt_yolo_v5_yolov5s_imgcls_torchhub_480x480", "pt_yolo_v5_yolov5l_imgcls_torchhub_480x480"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 256, 15, 15), torch.float32), ((256, 1, 1), torch.float32)],
        {
            "model_name": ["pt_yolo_v5_yolov5s_imgcls_torchhub_480x480", "pt_yolo_v5_yolov5n_imgcls_torchhub_480x480"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 256, 15, 15), torch.float32), ((1, 256, 15, 15), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5s_imgcls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 48, 80, 80), torch.float32), ((48, 1, 1), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5m_imgcls_torchhub_320x320"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 48, 80, 80), torch.float32), ((1, 48, 80, 80), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5m_imgcls_torchhub_320x320"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 96, 40, 40), torch.float32), ((1, 96, 40, 40), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5m_imgcls_torchhub_320x320"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 384, 20, 20), torch.float32), ((384, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v5_yolov5m_imgcls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_imgcls_torchhub_640x640",
                "pt_yolox_yolox_m_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 192, 20, 20), torch.float32), ((192, 1, 1), torch.float32)],
        {
            "model_name": ["pt_yolo_v5_yolov5m_imgcls_torchhub_320x320", "pt_yolox_yolox_m_obj_det_torchhub"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 192, 20, 20), torch.float32), ((1, 192, 20, 20), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5m_imgcls_torchhub_320x320"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 768, 10, 10), torch.float32), ((768, 1, 1), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5m_imgcls_torchhub_320x320"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 384, 10, 10), torch.float32), ((384, 1, 1), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5m_imgcls_torchhub_320x320"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 384, 10, 10), torch.float32), ((1, 384, 10, 10), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5m_imgcls_torchhub_320x320"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 255, 10, 10), torch.float32), ((255, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v5_yolov5m_imgcls_torchhub_320x320",
                "pt_yolo_v5_yolov5n_imgcls_torchhub_320x320",
                "pt_yolo_v5_yolov5l_imgcls_torchhub_320x320",
                "pt_yolo_v5_yolov5x_imgcls_torchhub_320x320",
                "pt_yolo_v5_yolov5s_imgcls_torchhub_320x320",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 255, 10, 10), torch.float32), ((1, 255, 10, 10), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v5_yolov5m_imgcls_torchhub_320x320",
                "pt_yolo_v5_yolov5n_imgcls_torchhub_320x320",
                "pt_yolo_v5_yolov5l_imgcls_torchhub_320x320",
                "pt_yolo_v5_yolov5x_imgcls_torchhub_320x320",
                "pt_yolo_v5_yolov5s_imgcls_torchhub_320x320",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 16, 160, 160), torch.float32), ((16, 1, 1), torch.float32)],
        {
            "model_name": ["pt_yolo_v5_yolov5n_imgcls_torchhub_320x320", "pt_yolo_v5_yolov5n_imgcls_torchhub_640x640"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 16, 80, 80), torch.float32), ((1, 16, 80, 80), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5n_imgcls_torchhub_320x320"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 64, 40, 40), torch.float32), ((64, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v5_yolov5n_imgcls_torchhub_320x320",
                "pt_yolo_v5_yolov5n_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_imgcls_torchhub_320x320",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 32, 40, 40), torch.float32), ((32, 1, 1), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5n_imgcls_torchhub_320x320"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 32, 40, 40), torch.float32), ((1, 32, 40, 40), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5n_imgcls_torchhub_320x320"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 128, 20, 20), torch.float32), ((128, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v5_yolov5n_imgcls_torchhub_320x320",
                "pt_yolo_v5_yolov5n_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_imgcls_torchhub_320x320",
                "pt_yolox_yolox_s_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 64, 20, 20), torch.float32), ((64, 1, 1), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5n_imgcls_torchhub_320x320"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 64, 20, 20), torch.float32), ((1, 64, 20, 20), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5n_imgcls_torchhub_320x320"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 256, 10, 10), torch.float32), ((256, 1, 1), torch.float32)],
        {
            "model_name": ["pt_yolo_v5_yolov5n_imgcls_torchhub_320x320", "pt_yolo_v5_yolov5s_imgcls_torchhub_320x320"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 128, 10, 10), torch.float32), ((1, 128, 10, 10), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5n_imgcls_torchhub_320x320"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 16, 320, 320), torch.float32), ((16, 1, 1), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5n_imgcls_torchhub_640x640"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 32, 160, 160), torch.float32), ((32, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v5_yolov5n_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_imgcls_torchhub_320x320",
                "pt_yolox_yolox_s_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 16, 160, 160), torch.float32), ((1, 16, 160, 160), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5n_imgcls_torchhub_640x640"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 64, 80, 80), torch.float32), ((64, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v5_yolov5n_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5l_imgcls_torchhub_320x320",
                "pt_yolo_v5_yolov5s_imgcls_torchhub_320x320",
                "pt_yolox_yolox_s_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 128, 40, 40), torch.float32), ((128, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v5_yolov5n_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5l_imgcls_torchhub_320x320",
                "pt_yolo_v5_yolov5s_imgcls_torchhub_320x320",
                "pt_yolox_yolox_s_obj_det_torchhub",
                "pt_yolox_yolox_darknet_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 64, 40, 40), torch.float32), ((1, 64, 40, 40), torch.float32)],
        {
            "model_name": ["pt_yolo_v5_yolov5n_imgcls_torchhub_640x640", "pt_yolo_v5_yolov5s_imgcls_torchhub_320x320"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 256, 20, 20), torch.float32), ((256, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v5_yolov5n_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5l_imgcls_torchhub_320x320",
                "pt_yolo_v5_yolov5s_imgcls_torchhub_320x320",
                "pt_yolox_yolox_l_obj_det_torchhub",
                "pt_yolox_yolox_s_obj_det_torchhub",
                "pt_yolox_yolox_darknet_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 128, 20, 20), torch.float32), ((1, 128, 20, 20), torch.float32)],
        {
            "model_name": ["pt_yolo_v5_yolov5n_imgcls_torchhub_640x640", "pt_yolo_v5_yolov5s_imgcls_torchhub_320x320"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 32, 160, 160), torch.float32), ((1, 32, 160, 160), torch.float32)],
        {
            "model_name": ["pt_yolo_v5_yolov5s_imgcls_torchhub_640x640", "pt_yolox_yolox_s_obj_det_torchhub"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 64, 80, 80), torch.float32), ((1, 64, 80, 80), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v5_yolov5s_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5l_imgcls_torchhub_320x320",
                "pt_yolox_yolox_s_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 128, 40, 40), torch.float32), ((1, 128, 40, 40), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v5_yolov5s_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5l_imgcls_torchhub_320x320",
                "pt_yolox_yolox_s_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 512, 20, 20), torch.float32), ((512, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v5_yolov5s_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5l_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5l_imgcls_torchhub_320x320",
                "pt_yolox_yolox_l_obj_det_torchhub",
                "pt_yolox_yolox_s_obj_det_torchhub",
                "pt_yolox_yolox_darknet_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 256, 20, 20), torch.float32), ((1, 256, 20, 20), torch.float32)],
        {
            "model_name": ["pt_yolo_v5_yolov5s_imgcls_torchhub_640x640", "pt_yolo_v5_yolov5l_imgcls_torchhub_320x320"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 48, 240, 240), torch.float32), ((48, 1, 1), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5m_imgcls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 96, 120, 120), torch.float32), ((96, 1, 1), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5m_imgcls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 48, 120, 120), torch.float32), ((48, 1, 1), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5m_imgcls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 48, 120, 120), torch.float32), ((1, 48, 120, 120), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5m_imgcls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 192, 60, 60), torch.float32), ((192, 1, 1), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5m_imgcls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 96, 60, 60), torch.float32), ((96, 1, 1), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5m_imgcls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 96, 60, 60), torch.float32), ((1, 96, 60, 60), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5m_imgcls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 384, 30, 30), torch.float32), ((384, 1, 1), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5m_imgcls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 192, 30, 30), torch.float32), ((192, 1, 1), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5m_imgcls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 192, 30, 30), torch.float32), ((1, 192, 30, 30), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5m_imgcls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 768, 15, 15), torch.float32), ((768, 1, 1), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5m_imgcls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 384, 15, 15), torch.float32), ((384, 1, 1), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5m_imgcls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 384, 15, 15), torch.float32), ((1, 384, 15, 15), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5m_imgcls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1024, 20, 20), torch.float32), ((1024, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v5_yolov5l_imgcls_torchhub_640x640",
                "pt_yolox_yolox_l_obj_det_torchhub",
                "pt_yolox_yolox_darknet_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 512, 20, 20), torch.float32), ((1, 512, 20, 20), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5l_imgcls_torchhub_640x640"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 512, 10, 10), torch.float32), ((1, 512, 10, 10), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5l_imgcls_torchhub_320x320"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 80, 80, 80), torch.float32), ((80, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v5_yolov5x_imgcls_torchhub_320x320",
                "pt_yolox_yolox_l_obj_det_torchhub",
                "pt_yolox_yolox_m_obj_det_torchhub",
                "pt_yolox_yolox_s_obj_det_torchhub",
                "pt_yolox_yolox_x_obj_det_torchhub",
                "pt_yolox_yolox_darknet_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 80, 80, 80), torch.float32), ((1, 80, 80, 80), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5x_imgcls_torchhub_320x320"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 160, 40, 40), torch.float32), ((160, 1, 1), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5x_imgcls_torchhub_320x320"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 160, 40, 40), torch.float32), ((1, 160, 40, 40), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5x_imgcls_torchhub_320x320"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 320, 20, 20), torch.float32), ((320, 1, 1), torch.float32)],
        {
            "model_name": ["pt_yolo_v5_yolov5x_imgcls_torchhub_320x320", "pt_yolox_yolox_x_obj_det_torchhub"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 320, 20, 20), torch.float32), ((1, 320, 20, 20), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5x_imgcls_torchhub_320x320"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1280, 10, 10), torch.float32), ((1280, 1, 1), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5x_imgcls_torchhub_320x320"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 640, 10, 10), torch.float32), ((640, 1, 1), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5x_imgcls_torchhub_320x320"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 640, 10, 10), torch.float32), ((1, 640, 10, 10), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5x_imgcls_torchhub_320x320"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 64, 240, 240), torch.float32), ((64, 1, 1), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5l_imgcls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 128, 120, 120), torch.float32), ((128, 1, 1), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5l_imgcls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 64, 120, 120), torch.float32), ((1, 64, 120, 120), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5l_imgcls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 256, 60, 60), torch.float32), ((256, 1, 1), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5l_imgcls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 128, 60, 60), torch.float32), ((1, 128, 60, 60), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5l_imgcls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 512, 30, 30), torch.float32), ((512, 1, 1), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5l_imgcls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 256, 30, 30), torch.float32), ((1, 256, 30, 30), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5l_imgcls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1024, 15, 15), torch.float32), ((1024, 1, 1), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5l_imgcls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 512, 15, 15), torch.float32), ((1, 512, 15, 15), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5l_imgcls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 16, 240, 240), torch.float32), ((16, 1, 1), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5n_imgcls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 16, 120, 120), torch.float32), ((16, 1, 1), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5n_imgcls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 16, 120, 120), torch.float32), ((1, 16, 120, 120), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5n_imgcls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 32, 60, 60), torch.float32), ((32, 1, 1), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5n_imgcls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 32, 60, 60), torch.float32), ((1, 32, 60, 60), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5n_imgcls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 64, 30, 30), torch.float32), ((64, 1, 1), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5n_imgcls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 64, 30, 30), torch.float32), ((1, 64, 30, 30), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5n_imgcls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 128, 15, 15), torch.float32), ((128, 1, 1), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5n_imgcls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 128, 15, 15), torch.float32), ((1, 128, 15, 15), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5n_imgcls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 256, 10, 10), torch.float32), ((1, 256, 10, 10), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5s_imgcls_torchhub_320x320"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 48, 320, 320), torch.float32), ((48, 1, 1), torch.float32)],
        {
            "model_name": ["pt_yolo_v5_yolov5m_imgcls_torchhub_640x640", "pt_yolox_yolox_m_obj_det_torchhub"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 96, 160, 160), torch.float32), ((96, 1, 1), torch.float32)],
        {
            "model_name": ["pt_yolo_v5_yolov5m_imgcls_torchhub_640x640", "pt_yolox_yolox_m_obj_det_torchhub"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 48, 160, 160), torch.float32), ((1, 48, 160, 160), torch.float32)],
        {
            "model_name": ["pt_yolo_v5_yolov5m_imgcls_torchhub_640x640", "pt_yolox_yolox_m_obj_det_torchhub"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 96, 80, 80), torch.float32), ((1, 96, 80, 80), torch.float32)],
        {
            "model_name": ["pt_yolo_v5_yolov5m_imgcls_torchhub_640x640", "pt_yolox_yolox_m_obj_det_torchhub"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 384, 40, 40), torch.float32), ((384, 1, 1), torch.float32)],
        {
            "model_name": ["pt_yolo_v5_yolov5m_imgcls_torchhub_640x640", "pt_yolox_yolox_m_obj_det_torchhub"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 192, 40, 40), torch.float32), ((1, 192, 40, 40), torch.float32)],
        {
            "model_name": ["pt_yolo_v5_yolov5m_imgcls_torchhub_640x640", "pt_yolox_yolox_m_obj_det_torchhub"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 768, 20, 20), torch.float32), ((768, 1, 1), torch.float32)],
        {
            "model_name": ["pt_yolo_v5_yolov5m_imgcls_torchhub_640x640", "pt_yolox_yolox_m_obj_det_torchhub"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 384, 20, 20), torch.float32), ((1, 384, 20, 20), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5m_imgcls_torchhub_640x640"], "pcc": 0.99},
    ),
    (
        Add186,
        [((1, 5880, 2), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v6_yolov6s_obj_det_torchhub",
                "pt_yolo_v6_yolov6l_obj_det_torchhub",
                "pt_yolo_v6_yolov6n_obj_det_torchhub",
                "pt_yolo_v6_yolov6m_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 5880, 2), torch.float32), ((1, 5880, 2), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v6_yolov6s_obj_det_torchhub",
                "pt_yolo_v6_yolov6l_obj_det_torchhub",
                "pt_yolo_v6_yolov6n_obj_det_torchhub",
                "pt_yolo_v6_yolov6m_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 64, 112, 160), torch.float32), ((1, 64, 112, 160), torch.float32)],
        {"model_name": ["pt_yolo_v6_yolov6l_obj_det_torchhub", "pt_yolo_v6_yolov6m_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 128, 56, 80), torch.float32), ((1, 128, 56, 80), torch.float32)],
        {"model_name": ["pt_yolo_v6_yolov6l_obj_det_torchhub", "pt_yolo_v6_yolov6m_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 256, 28, 40), torch.float32), ((1, 256, 28, 40), torch.float32)],
        {"model_name": ["pt_yolo_v6_yolov6l_obj_det_torchhub", "pt_yolo_v6_yolov6m_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 512, 14, 20), torch.float32), ((1, 512, 14, 20), torch.float32)],
        {"model_name": ["pt_yolo_v6_yolov6l_obj_det_torchhub", "pt_yolo_v6_yolov6m_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 128, 28, 40), torch.float32), ((1, 128, 28, 40), torch.float32)],
        {"model_name": ["pt_yolo_v6_yolov6l_obj_det_torchhub", "pt_yolo_v6_yolov6m_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 64, 56, 80), torch.float32), ((1, 64, 56, 80), torch.float32)],
        {"model_name": ["pt_yolo_v6_yolov6l_obj_det_torchhub", "pt_yolo_v6_yolov6m_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 256, 14, 20), torch.float32), ((1, 256, 14, 20), torch.float32)],
        {"model_name": ["pt_yolo_v6_yolov6l_obj_det_torchhub", "pt_yolo_v6_yolov6m_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 4, 80, 80), torch.float32), ((4, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_yolox_yolox_l_obj_det_torchhub",
                "pt_yolox_yolox_m_obj_det_torchhub",
                "pt_yolox_yolox_s_obj_det_torchhub",
                "pt_yolox_yolox_x_obj_det_torchhub",
                "pt_yolox_yolox_darknet_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1, 80, 80), torch.float32), ((1, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_yolox_yolox_l_obj_det_torchhub",
                "pt_yolox_yolox_m_obj_det_torchhub",
                "pt_yolox_yolox_s_obj_det_torchhub",
                "pt_yolox_yolox_x_obj_det_torchhub",
                "pt_yolox_yolox_darknet_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 4, 40, 40), torch.float32), ((4, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_yolox_yolox_l_obj_det_torchhub",
                "pt_yolox_yolox_m_obj_det_torchhub",
                "pt_yolox_yolox_s_obj_det_torchhub",
                "pt_yolox_yolox_x_obj_det_torchhub",
                "pt_yolox_yolox_darknet_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1, 40, 40), torch.float32), ((1, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_yolox_yolox_l_obj_det_torchhub",
                "pt_yolox_yolox_m_obj_det_torchhub",
                "pt_yolox_yolox_s_obj_det_torchhub",
                "pt_yolox_yolox_x_obj_det_torchhub",
                "pt_yolox_yolox_darknet_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 80, 40, 40), torch.float32), ((80, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_yolox_yolox_l_obj_det_torchhub",
                "pt_yolox_yolox_m_obj_det_torchhub",
                "pt_yolox_yolox_s_obj_det_torchhub",
                "pt_yolox_yolox_x_obj_det_torchhub",
                "pt_yolox_yolox_darknet_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 4, 20, 20), torch.float32), ((4, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_yolox_yolox_l_obj_det_torchhub",
                "pt_yolox_yolox_m_obj_det_torchhub",
                "pt_yolox_yolox_s_obj_det_torchhub",
                "pt_yolox_yolox_x_obj_det_torchhub",
                "pt_yolox_yolox_darknet_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1, 20, 20), torch.float32), ((1, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_yolox_yolox_l_obj_det_torchhub",
                "pt_yolox_yolox_m_obj_det_torchhub",
                "pt_yolox_yolox_s_obj_det_torchhub",
                "pt_yolox_yolox_x_obj_det_torchhub",
                "pt_yolox_yolox_darknet_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 80, 20, 20), torch.float32), ((80, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_yolox_yolox_l_obj_det_torchhub",
                "pt_yolox_yolox_m_obj_det_torchhub",
                "pt_yolox_yolox_s_obj_det_torchhub",
                "pt_yolox_yolox_x_obj_det_torchhub",
                "pt_yolox_yolox_darknet_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 24, 208, 208), torch.float32), ((24, 1, 1), torch.float32)],
        {"model_name": ["pt_yolox_yolox_tiny_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 48, 104, 104), torch.float32), ((48, 1, 1), torch.float32)],
        {"model_name": ["pt_yolox_yolox_tiny_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 24, 104, 104), torch.float32), ((24, 1, 1), torch.float32)],
        {"model_name": ["pt_yolox_yolox_tiny_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 24, 104, 104), torch.float32), ((1, 24, 104, 104), torch.float32)],
        {"model_name": ["pt_yolox_yolox_tiny_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 96, 52, 52), torch.float32), ((96, 1, 1), torch.float32)],
        {"model_name": ["pt_yolox_yolox_tiny_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 48, 52, 52), torch.float32), ((48, 1, 1), torch.float32)],
        {"model_name": ["pt_yolox_yolox_tiny_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 48, 52, 52), torch.float32), ((1, 48, 52, 52), torch.float32)],
        {"model_name": ["pt_yolox_yolox_tiny_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 192, 26, 26), torch.float32), ((192, 1, 1), torch.float32)],
        {"model_name": ["pt_yolox_yolox_tiny_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 96, 26, 26), torch.float32), ((96, 1, 1), torch.float32)],
        {"model_name": ["pt_yolox_yolox_tiny_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 96, 26, 26), torch.float32), ((1, 96, 26, 26), torch.float32)],
        {"model_name": ["pt_yolox_yolox_tiny_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 192, 13, 13), torch.float32), ((192, 1, 1), torch.float32)],
        {"model_name": ["pt_yolox_yolox_tiny_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 4, 52, 52), torch.float32), ((4, 1, 1), torch.float32)],
        {"model_name": ["pt_yolox_yolox_tiny_obj_det_torchhub", "pt_yolox_yolox_nano_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1, 52, 52), torch.float32), ((1, 1, 1), torch.float32)],
        {"model_name": ["pt_yolox_yolox_tiny_obj_det_torchhub", "pt_yolox_yolox_nano_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 80, 52, 52), torch.float32), ((80, 1, 1), torch.float32)],
        {"model_name": ["pt_yolox_yolox_tiny_obj_det_torchhub", "pt_yolox_yolox_nano_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 4, 26, 26), torch.float32), ((4, 1, 1), torch.float32)],
        {"model_name": ["pt_yolox_yolox_tiny_obj_det_torchhub", "pt_yolox_yolox_nano_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1, 26, 26), torch.float32), ((1, 1, 1), torch.float32)],
        {"model_name": ["pt_yolox_yolox_tiny_obj_det_torchhub", "pt_yolox_yolox_nano_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 80, 26, 26), torch.float32), ((80, 1, 1), torch.float32)],
        {"model_name": ["pt_yolox_yolox_tiny_obj_det_torchhub", "pt_yolox_yolox_nano_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 96, 13, 13), torch.float32), ((96, 1, 1), torch.float32)],
        {"model_name": ["pt_yolox_yolox_tiny_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 4, 13, 13), torch.float32), ((4, 1, 1), torch.float32)],
        {"model_name": ["pt_yolox_yolox_tiny_obj_det_torchhub", "pt_yolox_yolox_nano_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1, 13, 13), torch.float32), ((1, 1, 1), torch.float32)],
        {"model_name": ["pt_yolox_yolox_tiny_obj_det_torchhub", "pt_yolox_yolox_nano_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 80, 13, 13), torch.float32), ((80, 1, 1), torch.float32)],
        {"model_name": ["pt_yolox_yolox_tiny_obj_det_torchhub", "pt_yolox_yolox_nano_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 64, 320, 320), torch.float32), ((1, 64, 320, 320), torch.float32)],
        {"model_name": ["pt_yolox_yolox_darknet_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 128, 160, 160), torch.float32), ((1, 128, 160, 160), torch.float32)],
        {"model_name": ["pt_yolox_yolox_darknet_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 256, 80, 80), torch.float32), ((1, 256, 80, 80), torch.float32)],
        {"model_name": ["pt_yolox_yolox_darknet_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 512, 40, 40), torch.float32), ((1, 512, 40, 40), torch.float32)],
        {"model_name": ["pt_yolox_yolox_darknet_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1024, 20, 20), torch.float32), ((1, 1024, 20, 20), torch.float32)],
        {"model_name": ["pt_yolox_yolox_darknet_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 16, 208, 208), torch.float32), ((16, 1, 1), torch.float32)],
        {"model_name": ["pt_yolox_yolox_nano_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 16, 104, 104), torch.float32), ((16, 1, 1), torch.float32)],
        {"model_name": ["pt_yolox_yolox_nano_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 32, 104, 104), torch.float32), ((32, 1, 1), torch.float32)],
        {"model_name": ["pt_yolox_yolox_nano_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 16, 104, 104), torch.float32), ((1, 16, 104, 104), torch.float32)],
        {"model_name": ["pt_yolox_yolox_nano_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 32, 52, 52), torch.float32), ((32, 1, 1), torch.float32)],
        {"model_name": ["pt_yolox_yolox_nano_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 64, 52, 52), torch.float32), ((64, 1, 1), torch.float32)],
        {"model_name": ["pt_yolox_yolox_nano_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 32, 52, 52), torch.float32), ((1, 32, 52, 52), torch.float32)],
        {"model_name": ["pt_yolox_yolox_nano_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 64, 26, 26), torch.float32), ((64, 1, 1), torch.float32)],
        {"model_name": ["pt_yolox_yolox_nano_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 128, 26, 26), torch.float32), ((128, 1, 1), torch.float32)],
        {"model_name": ["pt_yolox_yolox_nano_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 64, 26, 26), torch.float32), ((1, 64, 26, 26), torch.float32)],
        {"model_name": ["pt_yolox_yolox_nano_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 128, 13, 13), torch.float32), ((128, 1, 1), torch.float32)],
        {"model_name": ["pt_yolox_yolox_nano_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 64, 13, 13), torch.float32), ((64, 1, 1), torch.float32)],
        {"model_name": ["pt_yolox_yolox_nano_obj_det_torchhub"], "pcc": 0.99},
    ),
]


@pytest.mark.nightly_models_ops
@pytest.mark.parametrize("forge_module_and_shapes_dtypes", forge_modules_and_shapes_dtypes_list, ids=ids_func)
def test_module(forge_module_and_shapes_dtypes, forge_property_recorder):
    forge_property_recorder.record_op_name("Add")

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
