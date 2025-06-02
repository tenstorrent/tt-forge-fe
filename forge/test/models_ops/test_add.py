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
            "add1.weight_1", forge.Parameter(*(1000,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add1.weight_1"))
        return add_output_1


class Add2(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add2.weight_0", forge.Parameter(*(384,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_1):
        add_output_1 = forge.op.Add("", self.get_parameter("add2.weight_0"), add_input_1)
        return add_output_1


class Add3(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add3_const_1", shape=(1, 1, 1, 13), dtype=torch.float32)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add3_const_1"))
        return add_output_1


class Add4(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add4.weight_0", forge.Parameter(*(1536,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_1):
        add_output_1 = forge.op.Add("", self.get_parameter("add4.weight_0"), add_input_1)
        return add_output_1


class Add5(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add5.weight_1", forge.Parameter(*(384,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add5.weight_1"))
        return add_output_1


class Add6(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add6.weight_0", forge.Parameter(*(64,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_1):
        add_output_1 = forge.op.Add("", self.get_parameter("add6.weight_0"), add_input_1)
        return add_output_1


class Add7(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add7.weight_0", forge.Parameter(*(256,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_1):
        add_output_1 = forge.op.Add("", self.get_parameter("add7.weight_0"), add_input_1)
        return add_output_1


class Add8(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add8.weight_0", forge.Parameter(*(128,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_1):
        add_output_1 = forge.op.Add("", self.get_parameter("add8.weight_0"), add_input_1)
        return add_output_1


class Add9(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add9.weight_0", forge.Parameter(*(512,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_1):
        add_output_1 = forge.op.Add("", self.get_parameter("add9.weight_0"), add_input_1)
        return add_output_1


class Add10(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add10.weight_0", forge.Parameter(*(320,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_1):
        add_output_1 = forge.op.Add("", self.get_parameter("add10.weight_0"), add_input_1)
        return add_output_1


class Add11(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add11.weight_0", forge.Parameter(*(1280,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_1):
        add_output_1 = forge.op.Add("", self.get_parameter("add11.weight_0"), add_input_1)
        return add_output_1


class Add12(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add12.weight_0", forge.Parameter(*(2048,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_1):
        add_output_1 = forge.op.Add("", self.get_parameter("add12.weight_0"), add_input_1)
        return add_output_1


class Add13(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add13_const_1", shape=(1,), dtype=torch.float32)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add13_const_1"))
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
            "add15.weight_1", forge.Parameter(*(3072,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add15.weight_1"))
        return add_output_1


class Add16(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add16.weight_1", forge.Parameter(*(2,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add16.weight_1"))
        return add_output_1


class Add17(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add17.weight_1", forge.Parameter(*(64,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add17.weight_1"))
        return add_output_1


class Add18(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add18.weight_1", forge.Parameter(*(256,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add18.weight_1"))
        return add_output_1


class Add19(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add19.weight_1", forge.Parameter(*(128,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add19.weight_1"))
        return add_output_1


class Add20(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add20.weight_1", forge.Parameter(*(512,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add20.weight_1"))
        return add_output_1


class Add21(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add21.weight_1", forge.Parameter(*(1024,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add21.weight_1"))
        return add_output_1


class Add22(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add22.weight_1", forge.Parameter(*(2048,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add22.weight_1"))
        return add_output_1


class Add23(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add23.weight_1", forge.Parameter(*(30000,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add23.weight_1"))
        return add_output_1


class Add24(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add24.weight_1", forge.Parameter(*(4096,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add24.weight_1"))
        return add_output_1


class Add25(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add25_const_1", shape=(1, 1, 1, 128), dtype=torch.float32)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add25_const_1"))
        return add_output_1


class Add26(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add26.weight_1",
            forge.Parameter(*(1, 197, 768), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add26.weight_1"))
        return add_output_1


class Add27(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add27_const_0", shape=(1, 100, 256), dtype=torch.float32)

    def forward(self, add_input_1):
        add_output_1 = forge.op.Add("", self.get_constant("add27_const_0"), add_input_1)
        return add_output_1


class Add28(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add28_const_1", shape=(1, 280, 256), dtype=torch.float32)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add28_const_1"))
        return add_output_1


class Add29(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add29_const_1", shape=(1, 1, 280, 280), dtype=torch.float32)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add29_const_1"))
        return add_output_1


class Add30(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add30_const_1", shape=(1, 1, 100, 280), dtype=torch.float32)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add30_const_1"))
        return add_output_1


class Add31(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add31.weight_1", forge.Parameter(*(92,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add31.weight_1"))
        return add_output_1


class Add32(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add32.weight_1", forge.Parameter(*(16,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add32.weight_1"))
        return add_output_1


class Add33(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add33.weight_1", forge.Parameter(*(32,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add33.weight_1"))
        return add_output_1


class Add34(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add34.weight_1", forge.Parameter(*(24,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add34.weight_1"))
        return add_output_1


class Add35(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add35.weight_1", forge.Parameter(*(144,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add35.weight_1"))
        return add_output_1


class Add36(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add36.weight_1", forge.Parameter(*(192,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add36.weight_1"))
        return add_output_1


class Add37(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add37.weight_1", forge.Parameter(*(48,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add37.weight_1"))
        return add_output_1


class Add38(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add38.weight_1", forge.Parameter(*(288,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add38.weight_1"))
        return add_output_1


class Add39(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add39.weight_1", forge.Parameter(*(96,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add39.weight_1"))
        return add_output_1


class Add40(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add40.weight_1", forge.Parameter(*(576,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add40.weight_1"))
        return add_output_1


class Add41(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add41.weight_1", forge.Parameter(*(136,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add41.weight_1"))
        return add_output_1


class Add42(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add42.weight_1", forge.Parameter(*(816,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add42.weight_1"))
        return add_output_1


class Add43(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add43.weight_1", forge.Parameter(*(232,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add43.weight_1"))
        return add_output_1


class Add44(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add44.weight_1", forge.Parameter(*(1392,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add44.weight_1"))
        return add_output_1


class Add45(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add45.weight_1", forge.Parameter(*(1280,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add45.weight_1"))
        return add_output_1


class Add46(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add46_const_1", shape=(1, 1, 207, 207), dtype=torch.float32)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add46_const_1"))
        return add_output_1


class Add47(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add47.weight_1", forge.Parameter(*(208,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add47.weight_1"))
        return add_output_1


class Add48(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add48.weight_1", forge.Parameter(*(160,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add48.weight_1"))
        return add_output_1


class Add49(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add49.weight_1", forge.Parameter(*(112,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add49.weight_1"))
        return add_output_1


class Add50(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add50.weight_1", forge.Parameter(*(224,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add50.weight_1"))
        return add_output_1


class Add51(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add51.weight_1", forge.Parameter(*(320,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add51.weight_1"))
        return add_output_1


class Add52(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add52.weight_1", forge.Parameter(*(30,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add52.weight_1"))
        return add_output_1


class Add53(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add53.weight_1", forge.Parameter(*(60,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add53.weight_1"))
        return add_output_1


class Add54(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add54.weight_1", forge.Parameter(*(120,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add54.weight_1"))
        return add_output_1


class Add55(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add55.weight_1", forge.Parameter(*(240,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add55.weight_1"))
        return add_output_1


class Add56(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add56.weight_1", forge.Parameter(*(448,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add56.weight_1"))
        return add_output_1


class Add57(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add57.weight_1", forge.Parameter(*(49,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add57.weight_1"))
        return add_output_1


class Add58(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add58.weight_1", forge.Parameter(*(960,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add58.weight_1"))
        return add_output_1


class Add59(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add59.weight_1", forge.Parameter(*(1001,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add59.weight_1"))
        return add_output_1


class Add60(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add60_const_1", shape=(1,), dtype=torch.int64)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add60_const_1"))
        return add_output_1


class Add61(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add61.weight_1", forge.Parameter(*(322,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add61.weight_1"))
        return add_output_1


class Add62(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add62_const_1", shape=(1, 1, 1, 3025), dtype=torch.float32)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add62_const_1"))
        return add_output_1


class Add63(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add63_const_1", shape=(1, 1, 12, 12), dtype=torch.float32)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add63_const_1"))
        return add_output_1


class Add64(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add64.weight_1", forge.Parameter(*(8192,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add64.weight_1"))
        return add_output_1


class Add65(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add65.weight_1", forge.Parameter(*(1536,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add65.weight_1"))
        return add_output_1


class Add66(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add66_const_1", shape=(1, 1, 35, 35), dtype=torch.float32)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add66_const_1"))
        return add_output_1


class Add67(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add67.weight_1", forge.Parameter(*(336,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add67.weight_1"))
        return add_output_1


class Add68(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add68.weight_1", forge.Parameter(*(672,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add68.weight_1"))
        return add_output_1


class Add69(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add69.weight_1", forge.Parameter(*(1344,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add69.weight_1"))
        return add_output_1


class Add70(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add70.weight_1", forge.Parameter(*(2520,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add70.weight_1"))
        return add_output_1


class Add71(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add71.weight_1", forge.Parameter(*(104,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add71.weight_1"))
        return add_output_1


class Add72(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add72.weight_1", forge.Parameter(*(440,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add72.weight_1"))
        return add_output_1


class Add73(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add73_const_1", shape=(64, 1, 64, 64), dtype=torch.float32)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add73_const_1"))
        return add_output_1


class Add74(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add74_const_1", shape=(16, 1, 64, 64), dtype=torch.float32)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add74_const_1"))
        return add_output_1


class Add75(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add75.weight_1", forge.Parameter(*(1152,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add75.weight_1"))
        return add_output_1


class Add76(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add76_const_1", shape=(4, 1, 64, 64), dtype=torch.float32)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add76_const_1"))
        return add_output_1


class Add77(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add77.weight_1", forge.Parameter(*(2304,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add77.weight_1"))
        return add_output_1


class Add78(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add78_const_1", shape=(1, 1, 1, 61), dtype=torch.float32)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add78_const_1"))
        return add_output_1


class Add79(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add79_const_1", shape=(1, 6, 1, 61), dtype=torch.float32)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add79_const_1"))
        return add_output_1


class Add80(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add80.weight_1",
            forge.Parameter(*(1, 197, 1024), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add80.weight_1"))
        return add_output_1


class Add81(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add81.weight_1",
            forge.Parameter(*(1500, 1024), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add81.weight_1"))
        return add_output_1


class Add82(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add82.weight_1", forge.Parameter(*(728,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add82.weight_1"))
        return add_output_1


class Add83(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add83_const_1", shape=(16,), dtype=torch.float32)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add83_const_1"))
        return add_output_1


class Add84(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add84_const_1", shape=(32,), dtype=torch.float32)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add84_const_1"))
        return add_output_1


class Add85(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add85_const_1", shape=(64,), dtype=torch.float32)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add85_const_1"))
        return add_output_1


class Add86(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add86_const_1", shape=(128,), dtype=torch.float32)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add86_const_1"))
        return add_output_1


class Add87(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add87_const_1", shape=(256,), dtype=torch.float32)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add87_const_1"))
        return add_output_1


class Add88(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add88_const_0", shape=(5880, 2), dtype=torch.float32)

    def forward(self, add_input_1):
        add_output_1 = forge.op.Add("", self.get_constant("add88_const_0"), add_input_1)
        return add_output_1


class Add89(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add89.weight_1",
            forge.Parameter(*(1, 64, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add89.weight_1"))
        return add_output_1


class Add90(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add90.weight_1",
            forge.Parameter(*(1, 256, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add90.weight_1"))
        return add_output_1


class Add91(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add91.weight_1",
            forge.Parameter(*(1, 128, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add91.weight_1"))
        return add_output_1


class Add92(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add92.weight_1",
            forge.Parameter(*(1, 512, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add92.weight_1"))
        return add_output_1


class Add93(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add93.weight_1",
            forge.Parameter(*(1, 1024, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add93.weight_1"))
        return add_output_1


class Add94(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add94.weight_1",
            forge.Parameter(*(1, 2048, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add94.weight_1"))
        return add_output_1


class Add95(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add95.weight_0", forge.Parameter(*(92,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_1):
        add_output_1 = forge.op.Add("", self.get_parameter("add95.weight_0"), add_input_1)
        return add_output_1


class Add96(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add96.weight_0", forge.Parameter(*(4,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_1):
        add_output_1 = forge.op.Add("", self.get_parameter("add96.weight_0"), add_input_1)
        return add_output_1


class Add97(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add97.weight_0", forge.Parameter(*(768,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_1):
        add_output_1 = forge.op.Add("", self.get_parameter("add97.weight_0"), add_input_1)
        return add_output_1


class Add98(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add98_const_1", shape=(1, 1, 256, 256), dtype=torch.float32)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add98_const_1"))
        return add_output_1


class Add99(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add99.weight_1", forge.Parameter(*(9,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add99.weight_1"))
        return add_output_1


class Add100(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add100.weight_1", forge.Parameter(*(40,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add100.weight_1"))
        return add_output_1


class Add101(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add101.weight_1", forge.Parameter(*(80,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add101.weight_1"))
        return add_output_1


class Add102(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add102.weight_1", forge.Parameter(*(480,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add102.weight_1"))
        return add_output_1


class Add103(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add103.weight_1", forge.Parameter(*(56,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add103.weight_1"))
        return add_output_1


class Add104(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add104.weight_1", forge.Parameter(*(272,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add104.weight_1"))
        return add_output_1


class Add105(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add105.weight_1", forge.Parameter(*(1632,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add105.weight_1"))
        return add_output_1


class Add106(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add106.weight_1", forge.Parameter(*(8,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add106.weight_1"))
        return add_output_1


class Add107(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add107.weight_1", forge.Parameter(*(12,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add107.weight_1"))
        return add_output_1


class Add108(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add108.weight_1", forge.Parameter(*(36,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add108.weight_1"))
        return add_output_1


class Add109(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add109.weight_1", forge.Parameter(*(72,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add109.weight_1"))
        return add_output_1


class Add110(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add110.weight_1", forge.Parameter(*(20,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add110.weight_1"))
        return add_output_1


class Add111(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add111.weight_1", forge.Parameter(*(100,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add111.weight_1"))
        return add_output_1


class Add112(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add112.weight_1", forge.Parameter(*(196,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add112.weight_1"))
        return add_output_1


class Add113(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add113.weight_1", forge.Parameter(*(200,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add113.weight_1"))
        return add_output_1


class Add114(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add114.weight_1", forge.Parameter(*(184,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add114.weight_1"))
        return add_output_1


class Add115(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add115.weight_1", forge.Parameter(*(261,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add115.weight_1"))
        return add_output_1


class Add116(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add116_const_1", shape=(1, 1, 1, 50176), dtype=torch.float32)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add116_const_1"))
        return add_output_1


class Add117(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add117.weight_1", forge.Parameter(*(432,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add117.weight_1"))
        return add_output_1


class Add118(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add118.weight_1", forge.Parameter(*(1008,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add118.weight_1"))
        return add_output_1


class Add119(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add119.weight_1", forge.Parameter(*(784,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add119.weight_1"))
        return add_output_1


class Add120(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add120.weight_1", forge.Parameter(*(3,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add120.weight_1"))
        return add_output_1


class Add121(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add121_const_1", shape=(1, 12, 1, 61), dtype=torch.float32)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add121_const_1"))
        return add_output_1


class Add122(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add122_const_1", shape=(197, 197), dtype=torch.float32)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add122_const_1"))
        return add_output_1


class Add123(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add123.weight_1",
            forge.Parameter(*(1500, 768), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add123.weight_1"))
        return add_output_1


class Add124(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add124_const_1", shape=(512,), dtype=torch.float32)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add124_const_1"))
        return add_output_1


class Add125(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add125.weight_1", forge.Parameter(*(640,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add125.weight_1"))
        return add_output_1


class Add126(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add126.weight_0", forge.Parameter(*(251,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_1):
        add_output_1 = forge.op.Add("", self.get_parameter("add126.weight_0"), add_input_1)
        return add_output_1


class Add127(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add127_const_0", shape=(1,), dtype=torch.float32)

    def forward(self, add_input_1):
        add_output_1 = forge.op.Add("", self.get_constant("add127_const_0"), add_input_1)
        return add_output_1


class Add128(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add128_const_1", shape=(8, 1), dtype=torch.float32)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add128_const_1"))
        return add_output_1


class Add129(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add129.weight_1",
            forge.Parameter(*(264, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add129.weight_1"))
        return add_output_1


class Add130(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add130.weight_1",
            forge.Parameter(*(128, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add130.weight_1"))
        return add_output_1


class Add131(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add131.weight_1",
            forge.Parameter(*(64, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add131.weight_1"))
        return add_output_1


class Add132(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add132.weight_1",
            forge.Parameter(*(32, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add132.weight_1"))
        return add_output_1


class Add133(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add133.weight_1",
            forge.Parameter(*(16, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add133.weight_1"))
        return add_output_1


class Add134(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add134_const_1", shape=(1, 1, 1, 384), dtype=torch.float32)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add134_const_1"))
        return add_output_1


class Add135(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add135.weight_1",
            forge.Parameter(*(1, 197, 192), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add135.weight_1"))
        return add_output_1


class Add136(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add136_const_0", shape=(1, 1, 256, 256), dtype=torch.float32)

    def forward(self, add_input_1):
        add_output_1 = forge.op.Add("", self.get_constant("add136_const_0"), add_input_1)
        return add_output_1


class Add137(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add137.weight_1", forge.Parameter(*(44,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add137.weight_1"))
        return add_output_1


class Add138(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add138.weight_1", forge.Parameter(*(88,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add138.weight_1"))
        return add_output_1


class Add139(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add139.weight_1", forge.Parameter(*(176,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add139.weight_1"))
        return add_output_1


class Add140(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add140.weight_1", forge.Parameter(*(352,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add140.weight_1"))
        return add_output_1


class Add141(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add141.weight_1", forge.Parameter(*(10,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add141.weight_1"))
        return add_output_1


class Add142(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add142.weight_1", forge.Parameter(*(262,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add142.weight_1"))
        return add_output_1


class Add143(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add143_const_1", shape=(1, 1, 6, 6), dtype=torch.float32)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add143_const_1"))
        return add_output_1


class Add144(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add144.weight_1", forge.Parameter(*(1296,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add144.weight_1"))
        return add_output_1


class Add145(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add145.weight_1", forge.Parameter(*(250002,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add145.weight_1"))
        return add_output_1


class Add146(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add146_const_1", shape=(64, 1, 49, 49), dtype=torch.float32)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add146_const_1"))
        return add_output_1


class Add147(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add147_const_1", shape=(16, 1, 49, 49), dtype=torch.float32)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add147_const_1"))
        return add_output_1


class Add148(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add148_const_1", shape=(4, 1, 49, 49), dtype=torch.float32)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add148_const_1"))
        return add_output_1


class Add149(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add149_const_1", shape=(1, 8, 1, 61), dtype=torch.float32)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add149_const_1"))
        return add_output_1


class Add150(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add150.weight_1",
            forge.Parameter(*(1, 1370, 1280), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add150.weight_1"))
        return add_output_1


class Add151(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add151.weight_1", forge.Parameter(*(3840,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add151.weight_1"))
        return add_output_1


class Add152(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add152_const_1", shape=(1370, 1370), dtype=torch.float32)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add152_const_1"))
        return add_output_1


class Add153(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add153.weight_1", forge.Parameter(*(5120,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add153.weight_1"))
        return add_output_1


class Add154(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add154_const_1", shape=(1, 1, 2, 2), dtype=torch.float32)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add154_const_1"))
        return add_output_1


class Add155(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add155_const_1", shape=(1500, 1280), dtype=torch.float32)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add155_const_1"))
        return add_output_1


class Add156(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add156.weight_1", forge.Parameter(*(4,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add156.weight_1"))
        return add_output_1


class Add157(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add157_const_0", shape=(1, 2, 8400), dtype=torch.float32)

    def forward(self, add_input_1):
        add_output_1 = forge.op.Add("", self.get_constant("add157_const_0"), add_input_1)
        return add_output_1


class Add158(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add158.weight_1", forge.Parameter(*(1,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add158.weight_1"))
        return add_output_1


class Add159(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add159.weight_0", forge.Parameter(*(3072,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_1):
        add_output_1 = forge.op.Add("", self.get_parameter("add159.weight_0"), add_input_1)
        return add_output_1


class Add160(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add160.weight_0", forge.Parameter(*(30522,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_1):
        add_output_1 = forge.op.Add("", self.get_parameter("add160.weight_0"), add_input_1)
        return add_output_1


class Add161(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add161.weight_0", forge.Parameter(*(96,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_1):
        add_output_1 = forge.op.Add("", self.get_parameter("add161.weight_0"), add_input_1)
        return add_output_1


class Add162(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add162.weight_0", forge.Parameter(*(192,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_1):
        add_output_1 = forge.op.Add("", self.get_parameter("add162.weight_0"), add_input_1)
        return add_output_1


class Add163(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add163.weight_1", forge.Parameter(*(416,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add163.weight_1"))
        return add_output_1


class Add164(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add164.weight_1", forge.Parameter(*(544,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add164.weight_1"))
        return add_output_1


class Add165(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add165.weight_1", forge.Parameter(*(608,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add165.weight_1"))
        return add_output_1


class Add166(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add166.weight_1", forge.Parameter(*(704,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add166.weight_1"))
        return add_output_1


class Add167(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add167.weight_1", forge.Parameter(*(736,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add167.weight_1"))
        return add_output_1


class Add168(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add168.weight_1", forge.Parameter(*(800,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add168.weight_1"))
        return add_output_1


class Add169(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add169.weight_1", forge.Parameter(*(832,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add169.weight_1"))
        return add_output_1


class Add170(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add170.weight_1", forge.Parameter(*(864,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add170.weight_1"))
        return add_output_1


class Add171(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add171.weight_1", forge.Parameter(*(896,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add171.weight_1"))
        return add_output_1


class Add172(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add172.weight_1", forge.Parameter(*(928,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add172.weight_1"))
        return add_output_1


class Add173(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add173.weight_1", forge.Parameter(*(992,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add173.weight_1"))
        return add_output_1


class Add174(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add174.weight_1", forge.Parameter(*(28996,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add174.weight_1"))
        return add_output_1


class Add175(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add175.weight_1", forge.Parameter(*(2688,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add175.weight_1"))
        return add_output_1


class Add176(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add176.weight_1", forge.Parameter(*(1792,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add176.weight_1"))
        return add_output_1


class Add177(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add177.weight_1", forge.Parameter(*(2560,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add177.weight_1"))
        return add_output_1


class Add178(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add178.weight_1", forge.Parameter(*(10240,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add178.weight_1"))
        return add_output_1


class Add179(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add179.weight_1", forge.Parameter(*(51200,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add179.weight_1"))
        return add_output_1


class Add180(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add180_const_1", shape=(1, 1, 29, 29), dtype=torch.float32)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add180_const_1"))
        return add_output_1


class Add181(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add181.weight_1", forge.Parameter(*(168,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add181.weight_1"))
        return add_output_1


class Add182(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add182.weight_1", forge.Parameter(*(2016,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add182.weight_1"))
        return add_output_1


class Add183(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add183.weight_1", forge.Parameter(*(720,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add183.weight_1"))
        return add_output_1


class Add184(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add184.weight_1", forge.Parameter(*(1920,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add184.weight_1"))
        return add_output_1


class Add185(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add185_const_1", shape=(1, 1, 2, 1), dtype=torch.float32)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add185_const_1"))
        return add_output_1


class Add186(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add186_const_1", shape=(1, 1, 1, 8), dtype=torch.float32)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add186_const_1"))
        return add_output_1


class Add187(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add187.weight_1", forge.Parameter(*(32000,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add187.weight_1"))
        return add_output_1


class Add188(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add188.weight_1", forge.Parameter(*(97,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add188.weight_1"))
        return add_output_1


class Add189(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add189.weight_1", forge.Parameter(*(16384,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add189.weight_1"))
        return add_output_1


class Add190(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add190.weight_1", forge.Parameter(*(4608,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add190.weight_1"))
        return add_output_1


class Add191(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add191_const_0", shape=(1, 1, 32, 32), dtype=torch.float32)

    def forward(self, add_input_1):
        add_output_1 = forge.op.Add("", self.get_constant("add191_const_0"), add_input_1)
        return add_output_1


class Add192(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add192.weight_1", forge.Parameter(*(6144,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add192.weight_1"))
        return add_output_1


class Add193(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add193.weight_1", forge.Parameter(*(1056,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add193.weight_1"))
        return add_output_1


class Add194(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add194.weight_1", forge.Parameter(*(1088,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add194.weight_1"))
        return add_output_1


class Add195(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add195.weight_1", forge.Parameter(*(1120,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add195.weight_1"))
        return add_output_1


class Add196(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add196.weight_1", forge.Parameter(*(1184,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add196.weight_1"))
        return add_output_1


class Add197(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add197.weight_1", forge.Parameter(*(1216,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add197.weight_1"))
        return add_output_1


class Add198(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add198.weight_1", forge.Parameter(*(1248,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add198.weight_1"))
        return add_output_1


class Add199(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add199.weight_1", forge.Parameter(*(1312,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add199.weight_1"))
        return add_output_1


class Add200(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add200.weight_1", forge.Parameter(*(1376,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add200.weight_1"))
        return add_output_1


class Add201(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add201.weight_1", forge.Parameter(*(1408,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add201.weight_1"))
        return add_output_1


class Add202(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add202.weight_1", forge.Parameter(*(1440,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add202.weight_1"))
        return add_output_1


class Add203(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add203.weight_1", forge.Parameter(*(1472,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add203.weight_1"))
        return add_output_1


class Add204(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add204.weight_1", forge.Parameter(*(1504,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add204.weight_1"))
        return add_output_1


class Add205(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add205.weight_1", forge.Parameter(*(1568,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add205.weight_1"))
        return add_output_1


class Add206(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add206.weight_1", forge.Parameter(*(1600,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add206.weight_1"))
        return add_output_1


class Add207(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add207.weight_1", forge.Parameter(*(1664,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add207.weight_1"))
        return add_output_1


class Add208(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add208_const_1", shape=(1, 1, 39, 39), dtype=torch.float32)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add208_const_1"))
        return add_output_1


class Add209(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add209.weight_1", forge.Parameter(*(696,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add209.weight_1"))
        return add_output_1


class Add210(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add210.weight_1", forge.Parameter(*(3712,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add210.weight_1"))
        return add_output_1


class Add211(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add211.weight_1", forge.Parameter(*(888,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add211.weight_1"))
        return add_output_1


class Add212(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add212.weight_1", forge.Parameter(*(3129,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add212.weight_1"))
        return add_output_1


class Add213(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add213.weight_0", forge.Parameter(*(32,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_1):
        add_output_1 = forge.op.Add("", self.get_parameter("add213.weight_0"), add_input_1)
        return add_output_1


class Add214(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add214.weight_0", forge.Parameter(*(160,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_1):
        add_output_1 = forge.op.Add("", self.get_parameter("add214.weight_0"), add_input_1)
        return add_output_1


class Add215(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add215.weight_0", forge.Parameter(*(640,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_1):
        add_output_1 = forge.op.Add("", self.get_parameter("add215.weight_0"), add_input_1)
        return add_output_1


class Add216(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add216.weight_0", forge.Parameter(*(1024,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_1):
        add_output_1 = forge.op.Add("", self.get_parameter("add216.weight_0"), add_input_1)
        return add_output_1


class Add217(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add217.weight_1", forge.Parameter(*(312,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add217.weight_1"))
        return add_output_1


class Add218(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add218.weight_1", forge.Parameter(*(21128,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add218.weight_1"))
        return add_output_1


class Add219(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add219.weight_1", forge.Parameter(*(360,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add219.weight_1"))
        return add_output_1


class Add220(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add220.weight_1", forge.Parameter(*(6625,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add220.weight_1"))
        return add_output_1


class Add221(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add221.weight_1", forge.Parameter(*(1696,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add221.weight_1"))
        return add_output_1


class Add222(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add222.weight_1", forge.Parameter(*(1728,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add222.weight_1"))
        return add_output_1


class Add223(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add223.weight_1", forge.Parameter(*(1760,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add223.weight_1"))
        return add_output_1


class Add224(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add224.weight_1", forge.Parameter(*(1824,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add224.weight_1"))
        return add_output_1


class Add225(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add225.weight_1", forge.Parameter(*(1856,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add225.weight_1"))
        return add_output_1


class Add226(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add226.weight_1", forge.Parameter(*(1888,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add226.weight_1"))
        return add_output_1


class Add227(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add227_const_1", shape=(1, 1, 522, 522), dtype=torch.float32)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add227_const_1"))
        return add_output_1


class Add228(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add228.weight_1", forge.Parameter(*(18,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add228.weight_1"))
        return add_output_1


class Add229(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add229_const_0", shape=(1, 1, 7, 7), dtype=torch.float32)

    def forward(self, add_input_1):
        add_output_1 = forge.op.Add("", self.get_constant("add229_const_0"), add_input_1)
        return add_output_1


class Add230(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add230_const_1", shape=(1, 1, 588, 588), dtype=torch.float32)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add230_const_1"))
        return add_output_1


class Add231(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add231.weight_1", forge.Parameter(*(528,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add231.weight_1"))
        return add_output_1


class Add232(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add232.weight_1",
            forge.Parameter(*(1, 257, 768), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add232.weight_1"))
        return add_output_1


class Add233(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add233.weight_1", forge.Parameter(*(38,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add233.weight_1"))
        return add_output_1


class Add234(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add234.weight_1", forge.Parameter(*(50257,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add234.weight_1"))
        return add_output_1


class Add235(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add235.weight_1", forge.Parameter(*(30522,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add235.weight_1"))
        return add_output_1


class Add236(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add236.weight_1", forge.Parameter(*(408,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add236.weight_1"))
        return add_output_1


class Add237(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add237.weight_1", forge.Parameter(*(912,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add237.weight_1"))
        return add_output_1


class Add238(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add238.weight_1", forge.Parameter(*(216,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add238.weight_1"))
        return add_output_1


class Add239(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add239.weight_1", forge.Parameter(*(1512,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add239.weight_1"))
        return add_output_1


class Add240(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add240_const_1", shape=(1, 16, 1, 61), dtype=torch.float32)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add240_const_1"))
        return add_output_1


class Add241(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add241.weight_1",
            forge.Parameter(*(1500, 512), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add241.weight_1"))
        return add_output_1


class Add242(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add242_const_1", shape=(48,), dtype=torch.float32)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add242_const_1"))
        return add_output_1


class Add243(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add243_const_1", shape=(96,), dtype=torch.float32)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add243_const_1"))
        return add_output_1


class Add244(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add244_const_1", shape=(192,), dtype=torch.float32)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add244_const_1"))
        return add_output_1


class Add245(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add245_const_1", shape=(384,), dtype=torch.float32)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add245_const_1"))
        return add_output_1


class Add246(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add246_const_1", shape=(768,), dtype=torch.float32)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add246_const_1"))
        return add_output_1


class Add247(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add247.weight_0", forge.Parameter(*(4096,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_1):
        add_output_1 = forge.op.Add("", self.get_parameter("add247.weight_0"), add_input_1)
        return add_output_1


class Add248(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add248_const_1", shape=(2, 1, 7, 7), dtype=torch.float32)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add248_const_1"))
        return add_output_1


class Add249(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add249.weight_1",
            forge.Parameter(*(1, 197, 384), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add249.weight_1"))
        return add_output_1


class Add250(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add250.weight_1", forge.Parameter(*(251,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add250.weight_1"))
        return add_output_1


class Add251(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add251_const_1", shape=(1, 1, 14, 20), dtype=torch.float32)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add251_const_1"))
        return add_output_1


class Add252(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add252.weight_1", forge.Parameter(*(21843,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add252.weight_1"))
        return add_output_1


class Add253(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add253.weight_1", forge.Parameter(*(400,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add253.weight_1"))
        return add_output_1


class Add254(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add254.weight_1",
            forge.Parameter(*(1, 50, 768), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add254.weight_1"))
        return add_output_1


class Add255(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add255_const_1", shape=(50, 50), dtype=torch.float32)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add255_const_1"))
        return add_output_1


class Add256(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add256.weight_1",
            forge.Parameter(*(1500, 384), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add256.weight_1"))
        return add_output_1


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (
        Add0,
        [((1, 16, 224, 224), torch.float32), ((16, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_dla_dla102_visual_bb_torchvision",
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_dla_dla34_in1k_img_cls_timm",
                "onnx_dla_dla102x2_visual_bb_torchvision",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "onnx_dla_dla169_visual_bb_torchvision",
                "pt_dla_dla169_visual_bb_torchvision",
                "onnx_dla_dla34_visual_bb_torchvision",
                "pt_dla_dla34_visual_bb_torchvision",
                "onnx_dla_dla46_c_visual_bb_torchvision",
                "pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels",
                "onnx_dla_dla60_visual_bb_torchvision",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_unet_qubvel_img_seg_torchhub",
                "onnx_dla_dla60x_visual_bb_torchvision",
                "pt_dla_dla60x_visual_bb_torchvision",
                "onnx_dla_dla60x_c_visual_bb_torchvision",
                "pt_dla_dla60x_c_visual_bb_torchvision",
                "onnx_dla_dla102x_visual_bb_torchvision",
                "pt_dla_dla102x_visual_bb_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 32, 112, 112), torch.float32), ((32, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_dla_dla102_visual_bb_torchvision",
                "onnx_efficientnet_efficientnet_lite0_img_cls_timm",
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_dla_dla34_in1k_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_regnet_regnet_x_32gf_img_cls_torchvision",
                "pt_regnet_regnet_y_400mf_img_cls_torchvision",
                "onnx_dla_dla102x2_visual_bb_torchvision",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_regnet_regnet_x_3_2gf_img_cls_torchvision",
                "pt_regnet_regnet_y_800mf_img_cls_torchvision",
                "onnx_dla_dla169_visual_bb_torchvision",
                "pt_dla_dla169_visual_bb_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_regnet_facebook_regnet_y_064_img_cls_hf",
                "pt_regnet_regnet_x_800mf_img_cls_torchvision",
                "onnx_dla_dla34_visual_bb_torchvision",
                "pt_dla_dla34_visual_bb_torchvision",
                "pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_regnet_facebook_regnet_y_080_img_cls_hf",
                "pt_regnet_regnet_x_8gf_img_cls_torchvision",
                "onnx_dla_dla46_c_visual_bb_torchvision",
                "pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels",
                "onnx_dla_dla60_visual_bb_torchvision",
                "onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm",
                "pd_mobilenetv1_basic_img_cls_paddlemodels",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                "pt_regnet_facebook_regnet_y_320_img_cls_hf",
                "pt_regnet_regnet_y_1_6gf_img_cls_torchvision",
                "pt_unet_qubvel_img_seg_torchhub",
                "onnx_dla_dla60x_visual_bb_torchvision",
                "pt_dla_dla60x_visual_bb_torchvision",
                "pt_mobilenetv1_basic_img_cls_torchvision",
                "pt_regnet_regnet_x_16gf_img_cls_torchvision",
                "pt_regnet_regnet_y_32gf_img_cls_torchvision",
                "onnx_dla_dla60x_c_visual_bb_torchvision",
                "pt_dla_dla60x_c_visual_bb_torchvision",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_regnet_regnet_x_1_6gf_img_cls_torchvision",
                "pt_regnet_regnet_y_3_2gf_img_cls_torchvision",
                "onnx_dla_dla102x_visual_bb_torchvision",
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
                "pt_dla_dla102x_visual_bb_torchvision",
                "pt_regnet_facebook_regnet_y_040_img_cls_hf",
                "pt_regnet_regnet_x_400mf_img_cls_torchvision",
                "pt_regnet_regnet_y_8gf_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 64, 112, 112), torch.float32), ((64, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_dla_dla102_visual_bb_torchvision",
                "onnx_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pd_resnet_50_img_cls_paddlemodels",
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_googlenet_base_img_cls_torchvision",
                "pt_hrnet_hrnetv2_w30_pose_estimation_osmr",
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "pt_resnext_resnext101_32x8d_img_cls_torchhub",
                "pt_hrnet_hrnetv2_w32_pose_estimation_osmr",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_regnet_regnet_y_800mf_img_cls_torchvision",
                "pt_resnext_resnext14_32x4d_img_cls_osmr",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "onnx_dla_dla169_visual_bb_torchvision",
                "pt_dla_dla169_visual_bb_torchvision",
                "pt_hrnet_hrnetv2_w44_pose_estimation_osmr",
                "pt_regnet_regnet_x_800mf_img_cls_torchvision",
                "pt_resnet_resnet101_img_cls_torchvision",
                "pt_resnext_resnext50_32x4d_img_cls_osmr",
                "pt_vovnet_ese_vovnet39b_obj_det_torchhub",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_hrnet_hrnetv2_w48_pose_estimation_osmr",
                "pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
                "pt_resnet_50_img_cls_hf",
                "pt_resnet_resnet152_img_cls_torchvision",
                "pt_resnext_resnext50_32x4d_img_cls_torchhub",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "onnx_resnet_50_img_cls_hf",
                "onnx_vovnet_v1_vovnet39_obj_det_torchhub",
                "onnx_dla_dla60_visual_bb_torchvision",
                "onnx_vovnet_vovnet_v1_57_obj_det_torchhub",
                "pd_mobilenetv1_basic_img_cls_paddlemodels",
                "pd_resnet_152_img_cls_paddlemodels",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_resnet_resnet50_img_cls_torchvision",
                "pt_unet_qubvel_img_seg_torchhub",
                "pt_vovnet_vovnet27s_img_cls_osmr",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
                "pd_resnet_18_img_cls_paddlemodels",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_mobilenetv1_basic_img_cls_torchvision",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub",
                "pt_vovnet_vovnet39_img_cls_osmr",
                "pt_vovnet_vovnet_v1_57_obj_det_torchhub",
                "pt_wideresnet_wide_resnet50_2_img_cls_timm",
                "onnx_dla_dla60x_c_visual_bb_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_resnet_34_img_cls_paddlemodels",
                "pt_dla_dla60x_c_visual_bb_torchvision",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
                "pt_resnext_resnext101_64x4d_img_cls_osmr",
                "pt_vovnet_vovnet57_img_cls_osmr",
                "pt_hrnet_hrnetv2_w40_pose_estimation_osmr",
                "pt_resnet_50_img_cls_timm",
                "pt_resnext_resnext26_32x4d_img_cls_osmr",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 64, 56, 56), torch.float32), ((64, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_dla_dla102_visual_bb_torchvision",
                "onnx_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pd_resnet_50_img_cls_paddlemodels",
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_dla_dla34_in1k_img_cls_timm",
                "pt_googlenet_base_img_cls_torchvision",
                "pt_hrnet_hrnetv2_w30_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w32_pose_estimation_osmr",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_rcnn_base_obj_det_torchvision_rect_0",
                "pt_regnet_regnet_y_800mf_img_cls_torchvision",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "onnx_dla_dla169_visual_bb_torchvision",
                "pt_dla_dla169_visual_bb_torchvision",
                "pt_hrnet_hrnetv2_w44_pose_estimation_osmr",
                "pt_regnet_regnet_x_800mf_img_cls_torchvision",
                "pt_resnet_resnet101_img_cls_torchvision",
                "onnx_dla_dla34_visual_bb_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_dla_dla34_visual_bb_torchvision",
                "pt_hrnet_hrnetv2_w48_pose_estimation_osmr",
                "pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
                "pt_resnet_50_img_cls_hf",
                "pt_resnet_resnet152_img_cls_torchvision",
                "onnx_dla_dla46_c_visual_bb_torchvision",
                "onnx_resnet_50_img_cls_hf",
                "onnx_dla_dla60_visual_bb_torchvision",
                "pd_mobilenetv1_basic_img_cls_paddlemodels",
                "pd_resnet_152_img_cls_paddlemodels",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_resnet_resnet50_img_cls_torchvision",
                "pt_unet_qubvel_img_seg_torchhub",
                "pt_vovnet_vovnet27s_img_cls_osmr",
                "pd_resnet_18_img_cls_paddlemodels",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_mobilenetv1_basic_img_cls_torchvision",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "onnx_dla_dla60x_c_visual_bb_torchvision",
                "onnx_efficientnet_efficientnet_b5_img_cls_timm",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_resnet_34_img_cls_paddlemodels",
                "pt_dla_dla60x_c_visual_bb_torchvision",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w40_pose_estimation_osmr",
                "pt_regnet_regnet_x_400mf_img_cls_torchvision",
                "pt_resnet_50_img_cls_timm",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 128, 56, 56), torch.float32), ((128, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_dla_dla102_visual_bb_torchvision",
                "onnx_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pd_resnet_50_img_cls_paddlemodels",
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_hrnet_hrnetv2_w30_pose_estimation_osmr",
                "onnx_dla_dla102x2_visual_bb_torchvision",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_hrnet_hrnetv2_w32_pose_estimation_osmr",
                "pt_resnext_resnext14_32x4d_img_cls_osmr",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "onnx_dla_dla169_visual_bb_torchvision",
                "pt_dla_dla169_visual_bb_torchvision",
                "pt_hrnet_hrnetv2_w44_pose_estimation_osmr",
                "pt_regnet_regnet_x_800mf_img_cls_torchvision",
                "pt_resnet_resnet101_img_cls_torchvision",
                "pt_resnext_resnext50_32x4d_img_cls_osmr",
                "pt_vovnet_ese_vovnet39b_obj_det_torchhub",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_hrnet_hrnetv2_w48_pose_estimation_osmr",
                "pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
                "pt_resnet_50_img_cls_hf",
                "pt_resnet_resnet152_img_cls_torchvision",
                "pt_resnext_resnext50_32x4d_img_cls_torchhub",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "onnx_resnet_50_img_cls_hf",
                "onnx_vovnet_v1_vovnet39_obj_det_torchhub",
                "onnx_dla_dla60_visual_bb_torchvision",
                "onnx_vovnet_vovnet_v1_57_obj_det_torchhub",
                "pd_mobilenetv1_basic_img_cls_paddlemodels",
                "pd_resnet_152_img_cls_paddlemodels",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_resnet_resnet50_img_cls_torchvision",
                "pt_unet_qubvel_img_seg_torchhub",
                "pt_vovnet_vovnet27s_img_cls_osmr",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
                "onnx_dla_dla60x_visual_bb_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_dla_dla60x_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_mobilenetv1_basic_img_cls_torchvision",
                "pt_vovnet_vovnet39_img_cls_osmr",
                "pt_vovnet_vovnet_v1_57_obj_det_torchhub",
                "pt_wideresnet_wide_resnet50_2_img_cls_timm",
                "pd_densenet_121_img_cls_paddlemodels",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
                "pt_vovnet_vovnet57_img_cls_osmr",
                "onnx_dla_dla102x_visual_bb_torchvision",
                "pt_dla_dla102x_visual_bb_torchvision",
                "pt_hrnet_hrnetv2_w40_pose_estimation_osmr",
                "pt_regnet_facebook_regnet_y_040_img_cls_hf",
                "pt_resnet_50_img_cls_timm",
                "pt_resnext_resnext26_32x4d_img_cls_osmr",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 128, 56, 56), torch.float32), ((1, 128, 56, 56), torch.float32)],
        {
            "model_names": [
                "onnx_dla_dla102_visual_bb_torchvision",
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_hrnet_hrnetv2_w30_pose_estimation_osmr",
                "onnx_dla_dla102x2_visual_bb_torchvision",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_hrnet_hrnetv2_w32_pose_estimation_osmr",
                "onnx_dla_dla169_visual_bb_torchvision",
                "pt_dla_dla169_visual_bb_torchvision",
                "pt_hrnet_hrnetv2_w44_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w48_pose_estimation_osmr",
                "onnx_dla_dla60_visual_bb_torchvision",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "onnx_dla_dla60x_visual_bb_torchvision",
                "pt_dla_dla60x_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
                "onnx_dla_dla102x_visual_bb_torchvision",
                "pt_dla_dla102x_visual_bb_torchvision",
                "pt_hrnet_hrnetv2_w40_pose_estimation_osmr",
                "pt_regnet_facebook_regnet_y_040_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 128, 28, 28), torch.float32), ((128, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_dla_dla102_visual_bb_torchvision",
                "pd_resnet_50_img_cls_paddlemodels",
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_dla_dla34_in1k_img_cls_timm",
                "pt_googlenet_base_img_cls_torchvision",
                "pt_hrnet_hrnetv2_w32_pose_estimation_osmr",
                "onnx_dla_dla169_visual_bb_torchvision",
                "pt_dla_dla169_visual_bb_torchvision",
                "pt_regnet_regnet_x_800mf_img_cls_torchvision",
                "pt_resnet_resnet101_img_cls_torchvision",
                "onnx_dla_dla34_visual_bb_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_dla_dla34_visual_bb_torchvision",
                "pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
                "pt_resnet_50_img_cls_hf",
                "pt_resnet_resnet152_img_cls_torchvision",
                "onnx_resnet_50_img_cls_hf",
                "onnx_dla_dla60_visual_bb_torchvision",
                "pd_mobilenetv1_basic_img_cls_paddlemodels",
                "pd_resnet_152_img_cls_paddlemodels",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_resnet_resnet50_img_cls_torchvision",
                "pt_unet_qubvel_img_seg_torchhub",
                "pd_resnet_18_img_cls_paddlemodels",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_mobilenetv1_basic_img_cls_torchvision",
                "onnx_dla_dla60x_c_visual_bb_torchvision",
                "onnx_efficientnet_efficientnet_b5_img_cls_timm",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_resnet_34_img_cls_paddlemodels",
                "pt_dla_dla60x_c_visual_bb_torchvision",
                "pt_resnet_50_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 256, 28, 28), torch.float32), ((256, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_dla_dla102_visual_bb_torchvision",
                "pd_resnet_50_img_cls_paddlemodels",
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_hrnet_hrnetv2_w30_pose_estimation_osmr",
                "onnx_dla_dla102x2_visual_bb_torchvision",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_hrnet_hrnetv2_w32_pose_estimation_osmr",
                "pt_resnext_resnext14_32x4d_img_cls_osmr",
                "onnx_dla_dla169_visual_bb_torchvision",
                "pt_dla_dla169_visual_bb_torchvision",
                "pt_hrnet_hrnetv2_w44_pose_estimation_osmr",
                "pt_resnet_resnet101_img_cls_torchvision",
                "pt_resnext_resnext50_32x4d_img_cls_osmr",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_hrnet_hrnetv2_w48_pose_estimation_osmr",
                "pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_resnet_50_img_cls_hf",
                "pt_resnet_resnet152_img_cls_torchvision",
                "pt_resnext_resnext50_32x4d_img_cls_torchhub",
                "onnx_resnet_50_img_cls_hf",
                "onnx_dla_dla60_visual_bb_torchvision",
                "pd_mobilenetv1_basic_img_cls_paddlemodels",
                "pd_resnet_152_img_cls_paddlemodels",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_resnet_resnet50_img_cls_torchvision",
                "pt_unet_qubvel_img_seg_torchhub",
                "pt_vovnet_vovnet27s_img_cls_osmr",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
                "onnx_dla_dla60x_visual_bb_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_dla_dla60x_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_mobilenetv1_basic_img_cls_torchvision",
                "pt_wideresnet_wide_resnet50_2_img_cls_timm",
                "pd_densenet_121_img_cls_paddlemodels",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
                "onnx_dla_dla102x_visual_bb_torchvision",
                "pt_dla_dla102x_visual_bb_torchvision",
                "pt_hrnet_hrnetv2_w40_pose_estimation_osmr",
                "pt_resnet_50_img_cls_timm",
                "pt_resnext_resnext26_32x4d_img_cls_osmr",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 256, 28, 28), torch.float32), ((1, 256, 28, 28), torch.float32)],
        {
            "model_names": [
                "onnx_dla_dla102_visual_bb_torchvision",
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_hrnet_hrnetv2_w30_pose_estimation_osmr",
                "onnx_dla_dla102x2_visual_bb_torchvision",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_hrnet_hrnetv2_w32_pose_estimation_osmr",
                "onnx_dla_dla169_visual_bb_torchvision",
                "pt_dla_dla169_visual_bb_torchvision",
                "pt_hrnet_hrnetv2_w44_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w48_pose_estimation_osmr",
                "onnx_dla_dla60_visual_bb_torchvision",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "onnx_dla_dla60x_visual_bb_torchvision",
                "pt_dla_dla60x_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
                "onnx_dla_dla102x_visual_bb_torchvision",
                "pt_dla_dla102x_visual_bb_torchvision",
                "pt_hrnet_hrnetv2_w40_pose_estimation_osmr",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 256, 14, 14), torch.float32), ((256, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_dla_dla102_visual_bb_torchvision",
                "pd_resnet_50_img_cls_paddlemodels",
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_dla_dla34_in1k_img_cls_timm",
                "pt_googlenet_base_img_cls_torchvision",
                "onnx_dla_dla169_visual_bb_torchvision",
                "pt_dla_dla169_visual_bb_torchvision",
                "pt_resnet_resnet101_img_cls_torchvision",
                "onnx_dla_dla34_visual_bb_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_dla_dla34_visual_bb_torchvision",
                "pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
                "pt_resnet_50_img_cls_hf",
                "pt_resnet_resnet152_img_cls_torchvision",
                "onnx_resnet_50_img_cls_hf",
                "onnx_dla_dla60_visual_bb_torchvision",
                "pd_mobilenetv1_basic_img_cls_paddlemodels",
                "pd_resnet_152_img_cls_paddlemodels",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_resnet_resnet50_img_cls_torchvision",
                "pt_unet_qubvel_img_seg_torchhub",
                "pd_resnet_18_img_cls_paddlemodels",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_mobilenetv1_basic_img_cls_torchvision",
                "onnx_dla_dla60x_c_visual_bb_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_resnet_34_img_cls_paddlemodels",
                "pt_dla_dla60x_c_visual_bb_torchvision",
                "pt_resnet_50_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 512, 14, 14), torch.float32), ((512, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_dla_dla102_visual_bb_torchvision",
                "pd_resnet_50_img_cls_paddlemodels",
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_hrnet_hrnetv2_w30_pose_estimation_osmr",
                "pt_vgg_19_obj_det_hf",
                "pt_vgg_vgg11_bn_img_cls_torchvision",
                "onnx_dla_dla102x2_visual_bb_torchvision",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_hrnet_hrnetv2_w32_pose_estimation_osmr",
                "pt_mlp_mixer_mixer_s16_224_img_cls_timm",
                "pt_resnext_resnext14_32x4d_img_cls_osmr",
                "pt_vgg_vgg13_img_cls_torchvision",
                "pt_vgg_vgg19_bn_obj_det_timm",
                "onnx_dla_dla169_visual_bb_torchvision",
                "pt_dla_dla169_visual_bb_torchvision",
                "pt_hrnet_hrnetv2_w44_pose_estimation_osmr",
                "pt_resnet_resnet101_img_cls_torchvision",
                "pt_resnext_resnext50_32x4d_img_cls_osmr",
                "pt_vgg_bn_vgg19_obj_det_osmr",
                "pt_vgg_vgg16_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_hrnet_hrnetv2_w48_pose_estimation_osmr",
                "pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
                "pt_resnet_50_img_cls_hf",
                "pt_resnet_resnet152_img_cls_torchvision",
                "pt_resnext_resnext50_32x4d_img_cls_torchhub",
                "pt_vgg_bn_vgg19b_obj_det_osmr",
                "pt_vgg_vgg16_bn_img_cls_torchvision",
                "onnx_resnet_50_img_cls_hf",
                "onnx_dla_dla60_visual_bb_torchvision",
                "pd_mobilenetv1_basic_img_cls_paddlemodels",
                "pd_resnet_152_img_cls_paddlemodels",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_resnet_resnet50_img_cls_torchvision",
                "pt_unet_qubvel_img_seg_torchhub",
                "pt_vgg_vgg16_obj_det_osmr",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
                "onnx_dla_dla60x_visual_bb_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_dla_dla60x_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_mobilenetv1_basic_img_cls_torchvision",
                "pt_vgg_vgg19_obj_det_osmr",
                "pt_wideresnet_wide_resnet50_2_img_cls_timm",
                "onnx_efficientnet_efficientnet_b5_img_cls_timm",
                "pd_densenet_121_img_cls_paddlemodels",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
                "pt_vgg_vgg11_img_cls_torchvision",
                "onnx_dla_dla102x_visual_bb_torchvision",
                "pt_dla_dla102x_visual_bb_torchvision",
                "pt_hrnet_hrnetv2_w40_pose_estimation_osmr",
                "pt_regnet_facebook_regnet_y_040_img_cls_hf",
                "pt_resnet_50_img_cls_timm",
                "pt_resnext_resnext26_32x4d_img_cls_osmr",
                "pt_vgg_vgg13_bn_img_cls_torchvision",
                "pt_vgg_vgg19_bn_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 512, 14, 14), torch.float32), ((1, 512, 14, 14), torch.float32)],
        {
            "model_names": [
                "onnx_dla_dla102_visual_bb_torchvision",
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_hrnet_hrnetv2_w30_pose_estimation_osmr",
                "onnx_dla_dla102x2_visual_bb_torchvision",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_hrnet_hrnetv2_w32_pose_estimation_osmr",
                "onnx_dla_dla169_visual_bb_torchvision",
                "pt_dla_dla169_visual_bb_torchvision",
                "pt_hrnet_hrnetv2_w44_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w48_pose_estimation_osmr",
                "onnx_dla_dla60_visual_bb_torchvision",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "onnx_dla_dla60x_visual_bb_torchvision",
                "pt_dla_dla60x_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "onnx_efficientnet_efficientnet_b5_img_cls_timm",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
                "onnx_dla_dla102x_visual_bb_torchvision",
                "pt_dla_dla102x_visual_bb_torchvision",
                "pt_hrnet_hrnetv2_w40_pose_estimation_osmr",
                "pt_regnet_facebook_regnet_y_040_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 512, 7, 7), torch.float32), ((512, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_dla_dla102_visual_bb_torchvision",
                "pd_resnet_50_img_cls_paddlemodels",
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_dla_dla34_in1k_img_cls_timm",
                "onnx_dla_dla169_visual_bb_torchvision",
                "pt_dla_dla169_visual_bb_torchvision",
                "pt_resnet_resnet101_img_cls_torchvision",
                "onnx_dla_dla34_visual_bb_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_dla_dla34_visual_bb_torchvision",
                "pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
                "pt_resnet_50_img_cls_hf",
                "pt_resnet_resnet152_img_cls_torchvision",
                "onnx_resnet_50_img_cls_hf",
                "onnx_dla_dla60_visual_bb_torchvision",
                "pd_mobilenetv1_basic_img_cls_paddlemodels",
                "pd_resnet_152_img_cls_paddlemodels",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_resnet_resnet50_img_cls_torchvision",
                "pt_unet_qubvel_img_seg_torchhub",
                "pt_vovnet_vovnet27s_img_cls_osmr",
                "pd_resnet_18_img_cls_paddlemodels",
                "pt_mobilenetv1_basic_img_cls_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_resnet_34_img_cls_paddlemodels",
                "pt_mlp_mixer_mixer_s32_224_img_cls_timm",
                "pt_resnet_50_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1024, 7, 7), torch.float32), ((1024, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_dla_dla102_visual_bb_torchvision",
                "onnx_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_hrnet_hrnetv2_w30_pose_estimation_osmr",
                "pt_mlp_mixer_mixer_l32_224_img_cls_timm",
                "onnx_dla_dla102x2_visual_bb_torchvision",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_hrnet_hrnetv2_w32_pose_estimation_osmr",
                "pt_resnext_resnext14_32x4d_img_cls_osmr",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "onnx_dla_dla169_visual_bb_torchvision",
                "pt_dla_dla169_visual_bb_torchvision",
                "pt_hrnet_hrnetv2_w44_pose_estimation_osmr",
                "pt_resnext_resnext50_32x4d_img_cls_osmr",
                "pt_vovnet_ese_vovnet39b_obj_det_torchhub",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_hrnet_hrnetv2_w48_pose_estimation_osmr",
                "pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
                "pt_resnext_resnext50_32x4d_img_cls_torchhub",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "onnx_vovnet_v1_vovnet39_obj_det_torchhub",
                "onnx_dla_dla60_visual_bb_torchvision",
                "onnx_vovnet_vovnet_v1_57_obj_det_torchhub",
                "pd_mobilenetv1_basic_img_cls_paddlemodels",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
                "onnx_dla_dla60x_visual_bb_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_dla_dla60x_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_mobilenetv1_basic_img_cls_torchvision",
                "pt_vovnet_vovnet39_img_cls_osmr",
                "pt_vovnet_vovnet_v1_57_obj_det_torchhub",
                "pt_wideresnet_wide_resnet50_2_img_cls_timm",
                "pd_densenet_121_img_cls_paddlemodels",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
                "pt_vovnet_vovnet57_img_cls_osmr",
                "onnx_dla_dla102x_visual_bb_torchvision",
                "pt_dla_dla102x_visual_bb_torchvision",
                "pt_hrnet_hrnetv2_w40_pose_estimation_osmr",
                "pt_resnext_resnext26_32x4d_img_cls_osmr",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1024, 7, 7), torch.float32), ((1, 1024, 7, 7), torch.float32)],
        {
            "model_names": [
                "onnx_dla_dla102_visual_bb_torchvision",
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_hrnet_hrnetv2_w30_pose_estimation_osmr",
                "onnx_dla_dla102x2_visual_bb_torchvision",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_hrnet_hrnetv2_w32_pose_estimation_osmr",
                "onnx_dla_dla169_visual_bb_torchvision",
                "pt_dla_dla169_visual_bb_torchvision",
                "pt_hrnet_hrnetv2_w44_pose_estimation_osmr",
                "pt_vovnet_ese_vovnet39b_obj_det_torchhub",
                "pt_hrnet_hrnetv2_w48_pose_estimation_osmr",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "onnx_vovnet_v1_vovnet39_obj_det_torchhub",
                "onnx_dla_dla60_visual_bb_torchvision",
                "onnx_vovnet_vovnet_v1_57_obj_det_torchhub",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "onnx_dla_dla60x_visual_bb_torchvision",
                "pt_dla_dla60x_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_vovnet_vovnet39_img_cls_osmr",
                "pt_vovnet_vovnet_v1_57_obj_det_torchhub",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
                "pt_vovnet_vovnet57_img_cls_osmr",
                "onnx_dla_dla102x_visual_bb_torchvision",
                "pt_dla_dla102x_visual_bb_torchvision",
                "pt_hrnet_hrnetv2_w40_pose_estimation_osmr",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1000, 1, 1), torch.float32), ((1000, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_dla_dla102_visual_bb_torchvision",
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_dla_dla34_in1k_img_cls_timm",
                "onnx_dla_dla102x2_visual_bb_torchvision",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "onnx_dla_dla169_visual_bb_torchvision",
                "pt_dla_dla169_visual_bb_torchvision",
                "onnx_dla_dla34_visual_bb_torchvision",
                "pt_dla_dla34_visual_bb_torchvision",
                "onnx_dla_dla46_c_visual_bb_torchvision",
                "onnx_dla_dla60_visual_bb_torchvision",
                "pt_dla_dla60_visual_bb_torchvision",
                "onnx_dla_dla60x_visual_bb_torchvision",
                "pt_dla_dla60x_visual_bb_torchvision",
                "onnx_dla_dla60x_c_visual_bb_torchvision",
                "pt_dla_dla60x_c_visual_bb_torchvision",
                "onnx_dla_dla102x_visual_bb_torchvision",
                "pt_dla_dla102x_visual_bb_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 16, 112, 112), torch.float32), ((16, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_lite0_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "onnx_mobilenetv2_mobilenetv2_050_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 96, 112, 112), torch.float32), ((96, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_lite0_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_regnet_regnet_x_3_2gf_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
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
                "onnx_efficientnet_efficientnet_lite0_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_regnet_regnet_x_3_2gf_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "onnx_mobilenetv2_mobilenetv2_050_img_cls_timm",
                "pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels",
                "onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 24, 56, 56), torch.float32), ((24, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_lite0_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 144, 56, 56), torch.float32), ((144, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_lite0_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_regnet_regnet_y_800mf_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_regnet_facebook_regnet_y_064_img_cls_hf",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_140_img_cls_timm",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
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
                "onnx_efficientnet_efficientnet_lite0_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels",
                "onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 144, 28, 28), torch.float32), ((144, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_lite0_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_regnet_regnet_y_800mf_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 40, 28, 28), torch.float32), ((40, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_lite0_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_hrnet_hrnetv2_w40_pose_estimation_osmr",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 240, 28, 28), torch.float32), ((240, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_lite0_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_regnet_regnet_x_8gf_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 40, 28, 28), torch.float32), ((1, 40, 28, 28), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_lite0_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 240, 14, 14), torch.float32), ((240, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_lite0_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 80, 14, 14), torch.float32), ((80, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_lite0_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_hrnet_hrnetv2_w40_pose_estimation_osmr",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 480, 14, 14), torch.float32), ((480, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_lite0_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pd_densenet_121_img_cls_paddlemodels",
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 80, 14, 14), torch.float32), ((1, 80, 14, 14), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_lite0_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 112, 14, 14), torch.float32), ((112, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_lite0_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 672, 14, 14), torch.float32), ((672, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_lite0_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_regnet_regnet_x_800mf_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pd_densenet_121_img_cls_paddlemodels",
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 112, 14, 14), torch.float32), ((1, 112, 14, 14), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_lite0_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 672, 7, 7), torch.float32), ((672, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_lite0_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_regnet_regnet_x_800mf_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pd_densenet_121_img_cls_paddlemodels",
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 192, 7, 7), torch.float32), ((192, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_lite0_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1152, 7, 7), torch.float32), ((1152, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_lite0_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                "pt_densenet_densenet201_img_cls_torchvision",
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 192, 7, 7), torch.float32), ((1, 192, 7, 7), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_lite0_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 320, 7, 7), torch.float32), ((320, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_lite0_img_cls_timm",
                "pt_googlenet_base_img_cls_torchvision",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
                "pt_hrnet_hrnetv2_w40_pose_estimation_osmr",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1280, 7, 7), torch.float32), ((1280, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_lite0_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "onnx_mobilenetv2_mobilenetv2_050_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add1,
        [((1, 1000), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_lite0_img_cls_timm",
                "onnx_segformer_nvidia_mit_b3_img_cls_hf",
                "onnx_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pd_resnet_50_img_cls_paddlemodels",
                "pt_alexnet_base_img_cls_osmr",
                "pt_deit_facebook_deit_base_distilled_patch16_224_img_cls_hf",
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                "pt_googlenet_base_img_cls_torchvision",
                "pt_hrnet_hrnetv2_w30_pose_estimation_osmr",
                "pt_inception_v4_img_cls_osmr",
                "pt_mlp_mixer_mixer_l32_224_img_cls_timm",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_regnet_regnet_x_32gf_img_cls_torchvision",
                "pt_regnet_regnet_y_400mf_img_cls_torchvision",
                "pt_resnext_resnext101_32x8d_img_cls_torchhub",
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "pt_vgg_19_obj_det_hf",
                "pt_vgg_vgg11_bn_img_cls_torchvision",
                "pt_vit_google_vit_large_patch16_224_img_cls_hf",
                "pt_xception_xception65_img_cls_timm",
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_alexnet_base_img_cls_torchhub",
                "pt_deit_facebook_deit_base_patch16_224_img_cls_hf",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_hrnet_hrnetv2_w32_pose_estimation_osmr",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_mlp_mixer_base_img_cls_github",
                "pt_mlp_mixer_mixer_s16_224_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_regnet_regnet_x_3_2gf_img_cls_torchvision",
                "pt_regnet_regnet_y_800mf_img_cls_torchvision",
                "pt_resnext_resnext14_32x4d_img_cls_osmr",
                "pt_vgg_vgg13_img_cls_torchvision",
                "pt_vgg_vgg19_bn_obj_det_timm",
                "pt_vit_vit_b_16_img_cls_torchvision",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_xception_xception71_tf_in1k_img_cls_timm",
                "tf_resnet_resnet50_img_cls_keras",
                "pt_beit_microsoft_beit_large_patch16_224_img_cls_hf",
                "pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_hrnet_hrnetv2_w44_pose_estimation_osmr",
                "pt_mlp_mixer_mixer_b16_224_img_cls_timm",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv3_ssd_resnet101_img_cls_torchvision",
                "pt_regnet_facebook_regnet_y_064_img_cls_hf",
                "pt_regnet_regnet_x_800mf_img_cls_torchvision",
                "pt_resnet_resnet101_img_cls_torchvision",
                "pt_resnext_resnext50_32x4d_img_cls_osmr",
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_vgg_bn_vgg19_obj_det_osmr",
                "pt_vgg_vgg16_img_cls_torchvision",
                "pt_vit_vit_h_14_img_cls_torchvision",
                "pt_vovnet_ese_vovnet39b_obj_det_torchhub",
                "pt_xception_xception_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_hrnet_hrnetv2_w48_pose_estimation_osmr",
                "pt_mlp_mixer_mixer_b16_224_in21k_img_cls_timm",
                "pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
                "pt_mobilenetv3_ssd_resnet152_img_cls_torchvision",
                "pt_regnet_facebook_regnet_y_080_img_cls_hf",
                "pt_regnet_regnet_x_8gf_img_cls_torchvision",
                "pt_resnet_50_img_cls_hf",
                "pt_resnet_resnet152_img_cls_torchvision",
                "pt_resnext_resnext50_32x4d_img_cls_torchhub",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_vgg_bn_vgg19b_obj_det_osmr",
                "pt_vgg_vgg16_bn_img_cls_torchvision",
                "pt_vit_vit_l_16_img_cls_torchvision",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "onnx_efficientnet_efficientnet_b2a_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_050_img_cls_timm",
                "onnx_resnet_50_img_cls_hf",
                "onnx_vovnet_v1_vovnet39_obj_det_torchhub",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm",
                "onnx_vovnet_vovnet_v1_57_obj_det_torchhub",
                "pd_mobilenetv1_basic_img_cls_paddlemodels",
                "pd_resnet_152_img_cls_paddlemodels",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_mlp_mixer_mixer_b32_224_img_cls_timm",
                "pt_mobilenetv3_ssd_resnet50_img_cls_torchvision",
                "pt_regnet_facebook_regnet_y_320_img_cls_hf",
                "pt_regnet_regnet_y_1_6gf_img_cls_torchvision",
                "pt_resnet_resnet50_img_cls_torchvision",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_vgg_vgg16_obj_det_osmr",
                "pt_vovnet_vovnet27s_img_cls_osmr",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
                "onnx_efficientnet_efficientnet_b4_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_140_img_cls_timm",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_vit_base_google_vit_base_patch16_224_img_cls_hf",
                "pd_resnet_18_img_cls_paddlemodels",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_mlp_mixer_mixer_l16_224_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_regnet_regnet_x_16gf_img_cls_torchvision",
                "pt_regnet_regnet_y_32gf_img_cls_torchvision",
                "pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_vgg_vgg19_obj_det_osmr",
                "pt_vovnet_vovnet39_img_cls_osmr",
                "pt_vovnet_vovnet_v1_57_obj_det_torchhub",
                "pt_wideresnet_wide_resnet50_2_img_cls_timm",
                "onnx_efficientnet_efficientnet_b5_img_cls_timm",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "pd_alexnet_base_img_cls_paddlemodels",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_googlenet_base_img_cls_paddlemodels",
                "pd_resnet_34_img_cls_paddlemodels",
                "pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
                "pt_mlp_mixer_mixer_l16_224_in21k_img_cls_timm",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_regnet_regnet_x_1_6gf_img_cls_torchvision",
                "pt_regnet_regnet_y_3_2gf_img_cls_torchvision",
                "pt_resnext_resnext101_64x4d_img_cls_osmr",
                "pt_swin_swin_v2_b_img_cls_torchvision",
                "pt_vgg_vgg11_img_cls_torchvision",
                "pt_vit_google_vit_base_patch16_224_img_cls_hf",
                "pt_vovnet_vovnet57_img_cls_osmr",
                "pt_xception_xception41_img_cls_timm",
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
                "pt_beit_microsoft_beit_base_patch16_224_img_cls_hf",
                "pt_deit_facebook_deit_small_patch16_224_img_cls_hf",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_hrnet_hrnetv2_w40_pose_estimation_osmr",
                "pt_inception_inception_v4_img_cls_timm",
                "pt_mlp_mixer_mixer_s32_224_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_regnet_facebook_regnet_y_040_img_cls_hf",
                "pt_regnet_regnet_x_400mf_img_cls_torchvision",
                "pt_regnet_regnet_y_8gf_img_cls_torchvision",
                "pt_resnet_50_img_cls_timm",
                "pt_resnext_resnext26_32x4d_img_cls_osmr",
                "pt_swin_swin_v2_t_img_cls_torchvision",
                "pt_vgg_vgg13_bn_img_cls_torchvision",
                "pt_vgg_vgg19_bn_obj_det_torchhub",
                "pt_vit_vit_b_32_img_cls_torchvision",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
                "pt_xception_xception71_img_cls_timm",
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
        Add2,
        [((1, 13, 384), torch.float32)],
        {"model_names": ["onnx_minilm_sentence_transformers_all_minilm_l6_v2_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Add3,
        [((1, 12, 13, 13), torch.float32)],
        {"model_names": ["onnx_minilm_sentence_transformers_all_minilm_l6_v2_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Add4,
        [((1, 13, 1536), torch.float32)],
        {"model_names": ["onnx_minilm_sentence_transformers_all_minilm_l6_v2_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Add5,
        [((1, 384), torch.float32)],
        {"model_names": ["onnx_minilm_sentence_transformers_all_minilm_l6_v2_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 64, 128, 128), torch.float32), ((64, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b3_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "onnx_unet_base_img_seg_torchhub",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_unet_base_img_seg_torchhub",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add6,
        [((1, 16384, 64), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b3_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 64, 16, 16), torch.float32), ((64, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b3_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add6,
        [((1, 256, 64), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b3_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 16384, 64), torch.float32), ((1, 16384, 64), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b3_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add7,
        [((1, 16384, 256), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b3_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 256, 128, 128), torch.float32), ((256, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b3_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pt_vgg19_unet_default_sem_seg_github",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 128, 64, 64), torch.float32), ((128, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b3_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "onnx_unet_base_img_seg_torchhub",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_unet_base_img_seg_torchhub",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_swin_swin_v2_b_img_cls_torchvision",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add8,
        [((1, 4096, 128), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b3_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 128, 16, 16), torch.float32), ((128, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b3_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add8,
        [((1, 256, 128), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b3_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 4096, 128), torch.float32), ((1, 4096, 128), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b3_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add9,
        [((1, 4096, 512), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b3_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 512, 64, 64), torch.float32), ((512, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b3_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_vgg19_unet_default_sem_seg_github",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 320, 32, 32), torch.float32), ((320, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b3_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add10,
        [((1, 1024, 320), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b3_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 320, 16, 16), torch.float32), ((320, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b3_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add10,
        [((1, 256, 320), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b3_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1024, 320), torch.float32), ((1, 1024, 320), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b3_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add11,
        [((1, 1024, 1280), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b3_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1280, 32, 32), torch.float32), ((1280, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b3_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 512, 16, 16), torch.float32), ((512, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b3_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "onnx_unet_base_img_seg_torchhub",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_unet_base_img_seg_torchhub",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add9,
        [((1, 256, 512), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b3_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 256, 512), torch.float32), ((1, 256, 512), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b3_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_mlp_mixer_base_img_cls_github",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add12,
        [((1, 256, 2048), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b3_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 2048, 16, 16), torch.float32), ((2048, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b3_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add7,
        [((1, 256, 256), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add7,
        [((1, 1024, 256), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add7,
        [((1, 4096, 256), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 150, 128, 128), torch.float32), ((150, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 256, 56, 56), torch.float32), ((256, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pd_resnet_50_img_cls_paddlemodels",
                "pt_hrnet_hrnetv2_w30_pose_estimation_osmr",
                "pt_resnext_resnext101_32x8d_img_cls_torchhub",
                "pt_vgg_19_obj_det_hf",
                "pt_vgg_vgg11_bn_img_cls_torchvision",
                "onnx_dla_dla102x2_visual_bb_torchvision",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_hrnet_hrnetv2_w32_pose_estimation_osmr",
                "pt_resnext_resnext14_32x4d_img_cls_osmr",
                "pt_vgg_vgg13_img_cls_torchvision",
                "pt_vgg_vgg19_bn_obj_det_timm",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_hrnet_hrnetv2_w44_pose_estimation_osmr",
                "pt_resnet_resnet101_img_cls_torchvision",
                "pt_resnext_resnext50_32x4d_img_cls_osmr",
                "pt_vgg_bn_vgg19_obj_det_osmr",
                "pt_vgg_vgg16_img_cls_torchvision",
                "pt_vovnet_ese_vovnet39b_obj_det_torchhub",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_hrnet_hrnetv2_w48_pose_estimation_osmr",
                "pt_resnet_50_img_cls_hf",
                "pt_resnet_resnet152_img_cls_torchvision",
                "pt_resnext_resnext50_32x4d_img_cls_torchhub",
                "pt_unet_carvana_base_img_seg_github",
                "pt_vgg_bn_vgg19b_obj_det_osmr",
                "pt_vgg_vgg16_bn_img_cls_torchvision",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "onnx_resnet_50_img_cls_hf",
                "onnx_vovnet_v1_vovnet39_obj_det_torchhub",
                "onnx_vovnet_vovnet_v1_57_obj_det_torchhub",
                "pd_resnet_152_img_cls_paddlemodels",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_resnet_resnet50_img_cls_torchvision",
                "pt_unet_qubvel_img_seg_torchhub",
                "pt_vgg_vgg16_obj_det_osmr",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
                "onnx_dla_dla60x_visual_bb_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_dla_dla60x_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_regnet_regnet_x_16gf_img_cls_torchvision",
                "pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub",
                "pt_vgg_vgg19_obj_det_osmr",
                "pt_vovnet_vovnet39_img_cls_osmr",
                "pt_vovnet_vovnet_v1_57_obj_det_torchhub",
                "pt_wideresnet_wide_resnet50_2_img_cls_timm",
                "pd_densenet_121_img_cls_paddlemodels",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
                "pt_resnext_resnext101_64x4d_img_cls_osmr",
                "pt_vgg_vgg11_img_cls_torchvision",
                "pt_vovnet_vovnet57_img_cls_osmr",
                "onnx_dla_dla102x_visual_bb_torchvision",
                "pt_dla_dla102x_visual_bb_torchvision",
                "pt_hrnet_hrnetv2_w40_pose_estimation_osmr",
                "pt_resnet_50_img_cls_timm",
                "pt_resnext_resnext26_32x4d_img_cls_osmr",
                "pt_vgg_vgg13_bn_img_cls_torchvision",
                "pt_vgg_vgg19_bn_obj_det_torchhub",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 256, 1, 1), torch.float32), ((256, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_vovnet_ese_vovnet39b_obj_det_torchhub",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_ssd300_resnet50_base_img_cls_torchhub",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add13,
        [((1, 256, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_detr_facebook_detr_resnet_50_obj_det_hf",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_vovnet_ese_vovnet39b_obj_det_torchhub",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 160, 28, 28), torch.float32), ((160, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_vovnet_ese_vovnet39b_obj_det_torchhub",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "onnx_vovnet_v1_vovnet39_obj_det_torchhub",
                "onnx_vovnet_vovnet_v1_57_obj_det_torchhub",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_vovnet_vovnet39_img_cls_osmr",
                "pt_vovnet_vovnet_v1_57_obj_det_torchhub",
                "pd_densenet_121_img_cls_paddlemodels",
                "pt_vovnet_vovnet57_img_cls_osmr",
                "pt_hrnet_hrnetv2_w40_pose_estimation_osmr",
                "pt_regnet_regnet_x_400mf_img_cls_torchvision",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 512, 28, 28), torch.float32), ((512, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pd_resnet_50_img_cls_paddlemodels",
                "pt_resnext_resnext101_32x8d_img_cls_torchhub",
                "pt_vgg_19_obj_det_hf",
                "pt_vgg_vgg11_bn_img_cls_torchvision",
                "onnx_dla_dla102x2_visual_bb_torchvision",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_resnext_resnext14_32x4d_img_cls_osmr",
                "pt_vgg_vgg13_img_cls_torchvision",
                "pt_vgg_vgg19_bn_obj_det_timm",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_resnet_resnet101_img_cls_torchvision",
                "pt_resnext_resnext50_32x4d_img_cls_osmr",
                "pt_vgg_bn_vgg19_obj_det_osmr",
                "pt_vgg_vgg16_img_cls_torchvision",
                "pt_vovnet_ese_vovnet39b_obj_det_torchhub",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_resnet_50_img_cls_hf",
                "pt_resnet_resnet152_img_cls_torchvision",
                "pt_resnext_resnext50_32x4d_img_cls_torchhub",
                "pt_unet_carvana_base_img_seg_github",
                "pt_vgg_bn_vgg19b_obj_det_osmr",
                "pt_vgg_vgg16_bn_img_cls_torchvision",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "onnx_resnet_50_img_cls_hf",
                "onnx_vovnet_v1_vovnet39_obj_det_torchhub",
                "onnx_vovnet_vovnet_v1_57_obj_det_torchhub",
                "pd_resnet_152_img_cls_paddlemodels",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_resnet_resnet50_img_cls_torchvision",
                "pt_unet_qubvel_img_seg_torchhub",
                "pt_vgg_vgg16_obj_det_osmr",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
                "onnx_dla_dla60x_visual_bb_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_dla_dla60x_visual_bb_torchvision",
                "pt_regnet_regnet_x_16gf_img_cls_torchvision",
                "pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub",
                "pt_vgg_vgg19_obj_det_osmr",
                "pt_vovnet_vovnet39_img_cls_osmr",
                "pt_vovnet_vovnet_v1_57_obj_det_torchhub",
                "pt_wideresnet_wide_resnet50_2_img_cls_timm",
                "pd_densenet_121_img_cls_paddlemodels",
                "pt_resnext_resnext101_64x4d_img_cls_osmr",
                "pt_vgg_vgg11_img_cls_torchvision",
                "pt_vovnet_vovnet57_img_cls_osmr",
                "onnx_dla_dla102x_visual_bb_torchvision",
                "pt_dla_dla102x_visual_bb_torchvision",
                "pt_regnet_facebook_regnet_y_040_img_cls_hf",
                "pt_resnet_50_img_cls_timm",
                "pt_resnext_resnext26_32x4d_img_cls_osmr",
                "pt_vgg_vgg13_bn_img_cls_torchvision",
                "pt_vgg_vgg19_bn_obj_det_torchhub",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 512, 1, 1), torch.float32), ((512, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_vovnet_ese_vovnet39b_obj_det_torchhub",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_regnet_facebook_regnet_y_040_img_cls_hf",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add13,
        [((1, 512, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_detr_facebook_detr_resnet_50_obj_det_hf",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_vovnet_ese_vovnet39b_obj_det_torchhub",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 192, 14, 14), torch.float32), ((192, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_vovnet_ese_vovnet39b_obj_det_torchhub",
                "pt_hrnet_hrnetv2_w48_pose_estimation_osmr",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "onnx_mobilenetv2_mobilenetv2_050_img_cls_timm",
                "onnx_vovnet_v1_vovnet39_obj_det_torchhub",
                "pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels",
                "onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm",
                "onnx_vovnet_vovnet_v1_57_obj_det_torchhub",
                "pt_vovnet_vovnet39_img_cls_osmr",
                "pt_vovnet_vovnet_v1_57_obj_det_torchhub",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_vovnet_vovnet57_img_cls_osmr",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 768, 14, 14), torch.float32), ((768, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_deit_facebook_deit_base_distilled_patch16_224_img_cls_hf",
                "pt_deit_facebook_deit_base_patch16_224_img_cls_hf",
                "pt_vit_vit_b_16_img_cls_torchvision",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_mlp_mixer_mixer_b16_224_img_cls_timm",
                "pt_vovnet_ese_vovnet39b_obj_det_torchhub",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_mlp_mixer_mixer_b16_224_in21k_img_cls_timm",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "onnx_vovnet_v1_vovnet39_obj_det_torchhub",
                "onnx_vovnet_vovnet_v1_57_obj_det_torchhub",
                "pt_densenet_densenet169_img_cls_torchvision",
                "onnx_vit_base_google_vit_base_patch16_224_img_cls_hf",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_vovnet_vovnet39_img_cls_osmr",
                "pt_vovnet_vovnet_v1_57_obj_det_torchhub",
                "pd_densenet_121_img_cls_paddlemodels",
                "pt_vit_google_vit_base_patch16_224_img_cls_hf",
                "pt_vovnet_vovnet57_img_cls_osmr",
                "pt_beit_microsoft_beit_base_patch16_224_img_cls_hf",
                "pt_mlp_mixer_mixer_b16_224_goog_in21k_img_cls_timm",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 768, 1, 1), torch.float32), ((768, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_vovnet_ese_vovnet39b_obj_det_torchhub",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "onnx_efficientnet_efficientnet_b5_img_cls_timm",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add13,
        [((1, 768, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_vovnet_ese_vovnet39b_obj_det_torchhub",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 224, 7, 7), torch.float32), ((224, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_hrnet_hrnetv2_w32_pose_estimation_osmr",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_vovnet_ese_vovnet39b_obj_det_torchhub",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "onnx_vovnet_v1_vovnet39_obj_det_torchhub",
                "onnx_vovnet_vovnet_v1_57_obj_det_torchhub",
                "onnx_mobilenetv2_mobilenetv2_140_img_cls_timm",
                "pt_vovnet_vovnet39_img_cls_osmr",
                "pt_vovnet_vovnet_v1_57_obj_det_torchhub",
                "pt_vovnet_vovnet57_img_cls_osmr",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1024, 1, 1), torch.float32), ((1024, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_vovnet_ese_vovnet39b_obj_det_torchhub",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add13,
        [((1, 1024, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_detr_facebook_detr_resnet_50_obj_det_hf",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_vovnet_ese_vovnet39b_obj_det_torchhub",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 14, 768), torch.float32), ((1, 14, 768), torch.float32)],
        {
            "model_names": ["pd_bert_bert_base_japanese_qa_padlenlp", "pt_albert_twmkn9_albert_base_v2_squad2_qa_hf"],
            "pcc": 0.99,
        },
    ),
    (Add13, [((1, 14, 1), torch.float32)], {"model_names": ["pd_bert_bert_base_japanese_qa_padlenlp"], "pcc": 0.99}),
    (
        Add14,
        [((1, 14, 768), torch.float32)],
        {
            "model_names": ["pd_bert_bert_base_japanese_qa_padlenlp", "pt_albert_twmkn9_albert_base_v2_squad2_qa_hf"],
            "pcc": 0.99,
        },
    ),
    (Add13, [((1, 1, 1, 14), torch.float32)], {"model_names": ["pd_bert_bert_base_japanese_qa_padlenlp"], "pcc": 0.99}),
    (
        Add0,
        [((1, 12, 14, 14), torch.float32), ((1, 1, 1, 14), torch.float32)],
        {
            "model_names": ["pd_bert_bert_base_japanese_qa_padlenlp", "pt_albert_twmkn9_albert_base_v2_squad2_qa_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Add15,
        [((1, 14, 3072), torch.float32)],
        {
            "model_names": ["pd_bert_bert_base_japanese_qa_padlenlp", "pt_albert_twmkn9_albert_base_v2_squad2_qa_hf"],
            "pcc": 0.99,
        },
    ),
    (Add16, [((1, 14, 2), torch.float32)], {"model_names": ["pd_bert_bert_base_japanese_qa_padlenlp"], "pcc": 0.99}),
    (
        Add0,
        [((64,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": [
                "pd_resnet_50_img_cls_paddlemodels",
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_dla_dla34_in1k_img_cls_timm",
                "pt_googlenet_base_img_cls_torchvision",
                "pt_hrnet_hrnetv2_w30_pose_estimation_osmr",
                "pt_inception_v4_img_cls_osmr",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision",
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "pt_resnext_resnext101_32x8d_img_cls_torchhub",
                "pt_vgg_vgg11_bn_img_cls_torchvision",
                "pt_xception_xception65_img_cls_timm",
                "pt_yolo_v6_yolov6n_obj_det_torchhub",
                "pt_hrnet_hrnetv2_w32_pose_estimation_osmr",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision",
                "pt_regnet_regnet_y_800mf_img_cls_torchvision",
                "pt_resnext_resnext14_32x4d_img_cls_osmr",
                "pt_vgg_vgg19_bn_obj_det_timm",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_xception_xception71_tf_in1k_img_cls_timm",
                "pt_yolo_v6_yolov6s_obj_det_torchhub",
                "tf_resnet_resnet50_img_cls_keras",
                "pt_dla_dla169_visual_bb_torchvision",
                "pt_hrnet_hrnetv2_w44_pose_estimation_osmr",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv3_ssd_resnet101_img_cls_torchvision",
                "pt_monodepth2_mono_1024x320_depth_prediction_torchvision",
                "pt_regnet_regnet_x_800mf_img_cls_torchvision",
                "pt_resnet_resnet101_img_cls_torchvision",
                "pt_resnext_resnext50_32x4d_img_cls_osmr",
                "pt_vgg_bn_vgg19_obj_det_osmr",
                "pt_vovnet_ese_vovnet39b_obj_det_torchhub",
                "pt_xception_xception_img_cls_timm",
                "pt_yolov8_default_obj_det_github",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_dla_dla34_visual_bb_torchvision",
                "pt_hrnet_hrnetv2_w48_pose_estimation_osmr",
                "pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_mobilenetv3_ssd_resnet152_img_cls_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_resnet_50_img_cls_hf",
                "pt_resnet_resnet152_img_cls_torchvision",
                "pt_resnext_resnext50_32x4d_img_cls_torchhub",
                "pt_unet_carvana_base_img_seg_github",
                "pt_vgg_bn_vgg19b_obj_det_osmr",
                "pt_vgg_vgg16_bn_img_cls_torchvision",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_yolov9_default_obj_det_github",
                "pd_mobilenetv1_basic_img_cls_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pd_resnet_152_img_cls_paddlemodels",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_mobilenetv3_ssd_resnet50_img_cls_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
                "pt_resnet_resnet50_img_cls_torchvision",
                "pt_retinanet_retinanet_rn18fpn_obj_det_hf",
                "pt_unet_qubvel_img_seg_torchhub",
                "pt_vovnet_vovnet27s_img_cls_osmr",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
                "pt_yolo_v4_default_obj_det_github",
                "pd_resnet_18_img_cls_paddlemodels",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_mobilenetv1_basic_img_cls_torchvision",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub",
                "pt_retinanet_retinanet_rn34fpn_obj_det_hf",
                "pt_unet_base_img_seg_torchhub",
                "pt_vovnet_vovnet39_img_cls_osmr",
                "pt_vovnet_vovnet_v1_57_obj_det_torchhub",
                "pt_wideresnet_wide_resnet50_2_img_cls_timm",
                "pt_yolox_yolox_nano_obj_det_torchhub",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_resnet_34_img_cls_paddlemodels",
                "pt_dla_dla60x_c_visual_bb_torchvision",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_resnext_resnext101_64x4d_img_cls_osmr",
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
                "pt_vgg19_unet_default_sem_seg_github",
                "pt_vovnet_vovnet57_img_cls_osmr",
                "pt_xception_xception41_img_cls_timm",
                "pt_yolo_v6_yolov6m_obj_det_torchhub",
                "pt_yolox_yolox_s_obj_det_torchhub",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
                "pt_hrnet_hrnetv2_w40_pose_estimation_osmr",
                "pt_inception_inception_v4_img_cls_timm",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_regnet_regnet_x_400mf_img_cls_torchvision",
                "pt_resnet_50_img_cls_timm",
                "pt_resnext_resnext26_32x4d_img_cls_osmr",
                "pt_ssd300_resnet50_base_img_cls_torchhub",
                "pt_vgg_vgg13_bn_img_cls_torchvision",
                "pt_vgg_vgg19_bn_obj_det_torchhub",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
                "pt_xception_xception71_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add17,
        [((64,), torch.float32)],
        {
            "model_names": [
                "pd_resnet_50_img_cls_paddlemodels",
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_dla_dla34_in1k_img_cls_timm",
                "pt_googlenet_base_img_cls_torchvision",
                "pt_hrnet_hrnetv2_w30_pose_estimation_osmr",
                "pt_inception_v4_img_cls_osmr",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision",
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "pt_resnext_resnext101_32x8d_img_cls_torchhub",
                "pt_vgg_vgg11_bn_img_cls_torchvision",
                "pt_xception_xception65_img_cls_timm",
                "pt_hrnet_hrnetv2_w32_pose_estimation_osmr",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision",
                "pt_regnet_regnet_y_800mf_img_cls_torchvision",
                "pt_resnext_resnext14_32x4d_img_cls_osmr",
                "pt_vgg_vgg19_bn_obj_det_timm",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_xception_xception71_tf_in1k_img_cls_timm",
                "tf_resnet_resnet50_img_cls_keras",
                "pt_dla_dla169_visual_bb_torchvision",
                "pt_hrnet_hrnetv2_w44_pose_estimation_osmr",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv3_ssd_resnet101_img_cls_torchvision",
                "pt_monodepth2_mono_1024x320_depth_prediction_torchvision",
                "pt_regnet_regnet_x_800mf_img_cls_torchvision",
                "pt_resnet_resnet101_img_cls_torchvision",
                "pt_resnext_resnext50_32x4d_img_cls_osmr",
                "pt_vgg_bn_vgg19_obj_det_osmr",
                "pt_vovnet_ese_vovnet39b_obj_det_torchhub",
                "pt_xception_xception_img_cls_timm",
                "pt_yolov8_default_obj_det_github",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_dla_dla34_visual_bb_torchvision",
                "pt_hrnet_hrnetv2_w48_pose_estimation_osmr",
                "pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_mobilenetv3_ssd_resnet152_img_cls_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_resnet_50_img_cls_hf",
                "pt_resnet_resnet152_img_cls_torchvision",
                "pt_resnext_resnext50_32x4d_img_cls_torchhub",
                "pt_unet_carvana_base_img_seg_github",
                "pt_vgg_bn_vgg19b_obj_det_osmr",
                "pt_vgg_vgg16_bn_img_cls_torchvision",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_yolov9_default_obj_det_github",
                "pd_mobilenetv1_basic_img_cls_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pd_resnet_152_img_cls_paddlemodels",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_mobilenetv3_ssd_resnet50_img_cls_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
                "pt_resnet_resnet50_img_cls_torchvision",
                "pt_retinanet_retinanet_rn18fpn_obj_det_hf",
                "pt_unet_qubvel_img_seg_torchhub",
                "pt_vovnet_vovnet27s_img_cls_osmr",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
                "pt_yolo_v4_default_obj_det_github",
                "pd_resnet_18_img_cls_paddlemodels",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_mobilenetv1_basic_img_cls_torchvision",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub",
                "pt_retinanet_retinanet_rn34fpn_obj_det_hf",
                "pt_unet_base_img_seg_torchhub",
                "pt_vovnet_vovnet39_img_cls_osmr",
                "pt_vovnet_vovnet_v1_57_obj_det_torchhub",
                "pt_wideresnet_wide_resnet50_2_img_cls_timm",
                "pt_yolox_yolox_nano_obj_det_torchhub",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_resnet_34_img_cls_paddlemodels",
                "pt_dla_dla60x_c_visual_bb_torchvision",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_resnext_resnext101_64x4d_img_cls_osmr",
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
                "pt_vgg19_unet_default_sem_seg_github",
                "pt_vovnet_vovnet57_img_cls_osmr",
                "pt_xception_xception41_img_cls_timm",
                "pt_yolox_yolox_s_obj_det_torchhub",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
                "pt_hrnet_hrnetv2_w40_pose_estimation_osmr",
                "pt_inception_inception_v4_img_cls_timm",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_regnet_regnet_x_400mf_img_cls_torchvision",
                "pt_resnet_50_img_cls_timm",
                "pt_resnext_resnext26_32x4d_img_cls_osmr",
                "pt_ssd300_resnet50_base_img_cls_torchhub",
                "pt_vgg_vgg13_bn_img_cls_torchvision",
                "pt_vgg_vgg19_bn_obj_det_torchhub",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
                "pt_xception_xception71_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((256,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": [
                "pd_resnet_50_img_cls_paddlemodels",
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_dla_dla34_in1k_img_cls_timm",
                "pt_googlenet_base_img_cls_torchvision",
                "pt_hrnet_hrnetv2_w30_pose_estimation_osmr",
                "pt_inception_v4_img_cls_osmr",
                "pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision",
                "pt_resnext_resnext101_32x8d_img_cls_torchhub",
                "pt_vgg_vgg11_bn_img_cls_torchvision",
                "pt_xception_xception65_img_cls_timm",
                "pt_yolo_v6_yolov6n_obj_det_torchhub",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_hrnet_hrnetv2_w32_pose_estimation_osmr",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision",
                "pt_resnext_resnext14_32x4d_img_cls_osmr",
                "pt_vgg_vgg19_bn_obj_det_timm",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_xception_xception71_tf_in1k_img_cls_timm",
                "pt_yolo_v6_yolov6s_obj_det_torchhub",
                "tf_resnet_resnet50_img_cls_keras",
                "pt_dla_dla169_visual_bb_torchvision",
                "pt_hrnet_hrnetv2_w44_pose_estimation_osmr",
                "pt_mobilenetv3_ssd_resnet101_img_cls_torchvision",
                "pt_monodepth2_mono_1024x320_depth_prediction_torchvision",
                "pt_resnet_resnet101_img_cls_torchvision",
                "pt_resnext_resnext50_32x4d_img_cls_osmr",
                "pt_vgg_bn_vgg19_obj_det_osmr",
                "pt_vovnet_ese_vovnet39b_obj_det_torchhub",
                "pt_xception_xception_img_cls_timm",
                "pt_yolov8_default_obj_det_github",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_dla_dla34_visual_bb_torchvision",
                "pt_hrnet_hrnetv2_w48_pose_estimation_osmr",
                "pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_mobilenetv3_ssd_resnet152_img_cls_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_resnet_50_img_cls_hf",
                "pt_resnet_resnet152_img_cls_torchvision",
                "pt_resnext_resnext50_32x4d_img_cls_torchhub",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pt_unet_carvana_base_img_seg_github",
                "pt_vgg_bn_vgg19b_obj_det_osmr",
                "pt_vgg_vgg16_bn_img_cls_torchvision",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_yolov9_default_obj_det_github",
                "pd_mobilenetv1_basic_img_cls_paddlemodels",
                "pd_resnet_152_img_cls_paddlemodels",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_mobilenetv3_ssd_resnet50_img_cls_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
                "pt_resnet_resnet50_img_cls_torchvision",
                "pt_retinanet_retinanet_rn18fpn_obj_det_hf",
                "pt_unet_qubvel_img_seg_torchhub",
                "pt_vovnet_vovnet27s_img_cls_osmr",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
                "pt_yolo_v4_default_obj_det_github",
                "pd_resnet_18_img_cls_paddlemodels",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_dla_dla60x_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_mobilenetv1_basic_img_cls_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_regnet_regnet_x_16gf_img_cls_torchvision",
                "pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub",
                "pt_retinanet_retinanet_rn34fpn_obj_det_hf",
                "pt_unet_base_img_seg_torchhub",
                "pt_vovnet_vovnet39_img_cls_osmr",
                "pt_vovnet_vovnet_v1_57_obj_det_torchhub",
                "pt_wideresnet_wide_resnet50_2_img_cls_timm",
                "pt_yolox_yolox_nano_obj_det_torchhub",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_resnet_34_img_cls_paddlemodels",
                "pt_dla_dla60x_c_visual_bb_torchvision",
                "pt_fpn_base_img_cls_torchvision",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
                "pt_resnext_resnext101_64x4d_img_cls_osmr",
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
                "pt_vgg19_unet_default_sem_seg_github",
                "pt_vovnet_vovnet57_img_cls_osmr",
                "pt_xception_xception41_img_cls_timm",
                "pt_yolo_v6_yolov6m_obj_det_torchhub",
                "pt_yolox_yolox_s_obj_det_torchhub",
                "pt_dla_dla102x_visual_bb_torchvision",
                "pt_hrnet_hrnetv2_w40_pose_estimation_osmr",
                "pt_inception_inception_v4_img_cls_timm",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_resnet_50_img_cls_timm",
                "pt_resnext_resnext26_32x4d_img_cls_osmr",
                "pt_ssd300_resnet50_base_img_cls_torchhub",
                "pt_vgg_vgg13_bn_img_cls_torchvision",
                "pt_vgg_vgg19_bn_obj_det_torchhub",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
                "pt_xception_xception71_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add18,
        [((256,), torch.float32)],
        {
            "model_names": [
                "pd_resnet_50_img_cls_paddlemodels",
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_dla_dla34_in1k_img_cls_timm",
                "pt_googlenet_base_img_cls_torchvision",
                "pt_hrnet_hrnetv2_w30_pose_estimation_osmr",
                "pt_inception_v4_img_cls_osmr",
                "pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision",
                "pt_resnext_resnext101_32x8d_img_cls_torchhub",
                "pt_vgg_vgg11_bn_img_cls_torchvision",
                "pt_xception_xception65_img_cls_timm",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_hrnet_hrnetv2_w32_pose_estimation_osmr",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision",
                "pt_resnext_resnext14_32x4d_img_cls_osmr",
                "pt_vgg_vgg19_bn_obj_det_timm",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_xception_xception71_tf_in1k_img_cls_timm",
                "tf_resnet_resnet50_img_cls_keras",
                "pt_dla_dla169_visual_bb_torchvision",
                "pt_hrnet_hrnetv2_w44_pose_estimation_osmr",
                "pt_mobilenetv3_ssd_resnet101_img_cls_torchvision",
                "pt_monodepth2_mono_1024x320_depth_prediction_torchvision",
                "pt_resnet_resnet101_img_cls_torchvision",
                "pt_resnext_resnext50_32x4d_img_cls_osmr",
                "pt_vgg_bn_vgg19_obj_det_osmr",
                "pt_vovnet_ese_vovnet39b_obj_det_torchhub",
                "pt_xception_xception_img_cls_timm",
                "pt_yolov8_default_obj_det_github",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_dla_dla34_visual_bb_torchvision",
                "pt_hrnet_hrnetv2_w48_pose_estimation_osmr",
                "pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_mobilenetv3_ssd_resnet152_img_cls_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_resnet_50_img_cls_hf",
                "pt_resnet_resnet152_img_cls_torchvision",
                "pt_resnext_resnext50_32x4d_img_cls_torchhub",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pt_unet_carvana_base_img_seg_github",
                "pt_vgg_bn_vgg19b_obj_det_osmr",
                "pt_vgg_vgg16_bn_img_cls_torchvision",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_yolov9_default_obj_det_github",
                "pd_mobilenetv1_basic_img_cls_paddlemodels",
                "pd_resnet_152_img_cls_paddlemodels",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_mobilenetv3_ssd_resnet50_img_cls_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
                "pt_resnet_resnet50_img_cls_torchvision",
                "pt_retinanet_retinanet_rn18fpn_obj_det_hf",
                "pt_unet_qubvel_img_seg_torchhub",
                "pt_vovnet_vovnet27s_img_cls_osmr",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
                "pt_yolo_v4_default_obj_det_github",
                "pd_resnet_18_img_cls_paddlemodels",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_dla_dla60x_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_mobilenetv1_basic_img_cls_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_regnet_regnet_x_16gf_img_cls_torchvision",
                "pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub",
                "pt_retinanet_retinanet_rn34fpn_obj_det_hf",
                "pt_unet_base_img_seg_torchhub",
                "pt_vovnet_vovnet39_img_cls_osmr",
                "pt_vovnet_vovnet_v1_57_obj_det_torchhub",
                "pt_wideresnet_wide_resnet50_2_img_cls_timm",
                "pt_yolox_yolox_nano_obj_det_torchhub",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_resnet_34_img_cls_paddlemodels",
                "pt_dla_dla60x_c_visual_bb_torchvision",
                "pt_fpn_base_img_cls_torchvision",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
                "pt_resnext_resnext101_64x4d_img_cls_osmr",
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
                "pt_vgg19_unet_default_sem_seg_github",
                "pt_vovnet_vovnet57_img_cls_osmr",
                "pt_xception_xception41_img_cls_timm",
                "pt_yolox_yolox_s_obj_det_torchhub",
                "pt_dla_dla102x_visual_bb_torchvision",
                "pt_hrnet_hrnetv2_w40_pose_estimation_osmr",
                "pt_inception_inception_v4_img_cls_timm",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_resnet_50_img_cls_timm",
                "pt_resnext_resnext26_32x4d_img_cls_osmr",
                "pt_ssd300_resnet50_base_img_cls_torchhub",
                "pt_vgg_vgg13_bn_img_cls_torchvision",
                "pt_vgg_vgg19_bn_obj_det_torchhub",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
                "pt_xception_xception71_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 256, 56, 56), torch.float32), ((1, 256, 56, 56), torch.float32)],
        {
            "model_names": [
                "pd_resnet_50_img_cls_paddlemodels",
                "pt_hrnet_hrnetv2_w30_pose_estimation_osmr",
                "pt_resnext_resnext101_32x8d_img_cls_torchhub",
                "pt_hrnet_hrnetv2_w32_pose_estimation_osmr",
                "pt_resnext_resnext14_32x4d_img_cls_osmr",
                "pt_hrnet_hrnetv2_w44_pose_estimation_osmr",
                "pt_resnet_resnet101_img_cls_torchvision",
                "pt_resnext_resnext50_32x4d_img_cls_osmr",
                "pt_hrnet_hrnetv2_w48_pose_estimation_osmr",
                "pt_resnet_50_img_cls_hf",
                "pt_resnet_resnet152_img_cls_torchvision",
                "pt_resnext_resnext50_32x4d_img_cls_torchhub",
                "onnx_resnet_50_img_cls_hf",
                "pd_resnet_152_img_cls_paddlemodels",
                "pt_resnet_resnet50_img_cls_torchvision",
                "pt_unet_qubvel_img_seg_torchhub",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_regnet_regnet_x_16gf_img_cls_torchvision",
                "pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub",
                "pt_wideresnet_wide_resnet50_2_img_cls_timm",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
                "pt_resnext_resnext101_64x4d_img_cls_osmr",
                "pt_hrnet_hrnetv2_w40_pose_estimation_osmr",
                "pt_resnet_50_img_cls_timm",
                "pt_resnext_resnext26_32x4d_img_cls_osmr",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((128,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": [
                "pd_resnet_50_img_cls_paddlemodels",
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_dla_dla34_in1k_img_cls_timm",
                "pt_googlenet_base_img_cls_torchvision",
                "pt_hrnet_hrnetv2_w30_pose_estimation_osmr",
                "pt_inception_v4_img_cls_osmr",
                "pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision",
                "pt_vgg_vgg11_bn_img_cls_torchvision",
                "pt_xception_xception65_img_cls_timm",
                "pt_yolo_v6_yolov6n_obj_det_torchhub",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_hrnet_hrnetv2_w32_pose_estimation_osmr",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision",
                "pt_resnext_resnext14_32x4d_img_cls_osmr",
                "pt_vgg_vgg19_bn_obj_det_timm",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_xception_xception71_tf_in1k_img_cls_timm",
                "pt_yolo_v6_yolov6s_obj_det_torchhub",
                "tf_resnet_resnet50_img_cls_keras",
                "pt_dla_dla169_visual_bb_torchvision",
                "pt_hrnet_hrnetv2_w44_pose_estimation_osmr",
                "pt_mobilenetv3_ssd_resnet101_img_cls_torchvision",
                "pt_monodepth2_mono_1024x320_depth_prediction_torchvision",
                "pt_regnet_regnet_x_800mf_img_cls_torchvision",
                "pt_resnet_resnet101_img_cls_torchvision",
                "pt_resnext_resnext50_32x4d_img_cls_osmr",
                "pt_vgg_bn_vgg19_obj_det_osmr",
                "pt_vovnet_ese_vovnet39b_obj_det_torchhub",
                "pt_xception_xception_img_cls_timm",
                "pt_yolov8_default_obj_det_github",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_dla_dla34_visual_bb_torchvision",
                "pt_hrnet_hrnetv2_w48_pose_estimation_osmr",
                "pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
                "pt_mobilenetv3_ssd_resnet152_img_cls_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_resnet_50_img_cls_hf",
                "pt_resnet_resnet152_img_cls_torchvision",
                "pt_resnext_resnext50_32x4d_img_cls_torchhub",
                "pt_unet_carvana_base_img_seg_github",
                "pt_vgg_bn_vgg19b_obj_det_osmr",
                "pt_vgg_vgg16_bn_img_cls_torchvision",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_yolov9_default_obj_det_github",
                "pd_mobilenetv1_basic_img_cls_paddlemodels",
                "pd_resnet_152_img_cls_paddlemodels",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_mobilenetv3_ssd_resnet50_img_cls_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
                "pt_resnet_resnet50_img_cls_torchvision",
                "pt_retinanet_retinanet_rn18fpn_obj_det_hf",
                "pt_unet_qubvel_img_seg_torchhub",
                "pt_vovnet_vovnet27s_img_cls_osmr",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
                "pt_yolo_v4_default_obj_det_github",
                "pd_resnet_18_img_cls_paddlemodels",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_dla_dla60x_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_mobilenetv1_basic_img_cls_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_retinanet_retinanet_rn34fpn_obj_det_hf",
                "pt_unet_base_img_seg_torchhub",
                "pt_vovnet_vovnet39_img_cls_osmr",
                "pt_vovnet_vovnet_v1_57_obj_det_torchhub",
                "pt_wideresnet_wide_resnet50_2_img_cls_timm",
                "pt_yolox_yolox_nano_obj_det_torchhub",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_resnet_34_img_cls_paddlemodels",
                "pt_dla_dla60x_c_visual_bb_torchvision",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
                "pt_vgg19_unet_default_sem_seg_github",
                "pt_vovnet_vovnet57_img_cls_osmr",
                "pt_xception_xception41_img_cls_timm",
                "pt_yolo_v6_yolov6m_obj_det_torchhub",
                "pt_yolox_yolox_s_obj_det_torchhub",
                "pt_dla_dla102x_visual_bb_torchvision",
                "pt_hrnet_hrnetv2_w40_pose_estimation_osmr",
                "pt_inception_inception_v4_img_cls_timm",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_regnet_facebook_regnet_y_040_img_cls_hf",
                "pt_resnet_50_img_cls_timm",
                "pt_resnext_resnext26_32x4d_img_cls_osmr",
                "pt_ssd300_resnet50_base_img_cls_torchhub",
                "pt_vgg_vgg13_bn_img_cls_torchvision",
                "pt_vgg_vgg19_bn_obj_det_torchhub",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
                "pt_xception_xception71_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add19,
        [((128,), torch.float32)],
        {
            "model_names": [
                "pd_resnet_50_img_cls_paddlemodels",
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_dla_dla34_in1k_img_cls_timm",
                "pt_googlenet_base_img_cls_torchvision",
                "pt_hrnet_hrnetv2_w30_pose_estimation_osmr",
                "pt_inception_v4_img_cls_osmr",
                "pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision",
                "pt_vgg_vgg11_bn_img_cls_torchvision",
                "pt_xception_xception65_img_cls_timm",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_hrnet_hrnetv2_w32_pose_estimation_osmr",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision",
                "pt_resnext_resnext14_32x4d_img_cls_osmr",
                "pt_vgg_vgg19_bn_obj_det_timm",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_xception_xception71_tf_in1k_img_cls_timm",
                "tf_resnet_resnet50_img_cls_keras",
                "pt_dla_dla169_visual_bb_torchvision",
                "pt_hrnet_hrnetv2_w44_pose_estimation_osmr",
                "pt_mobilenetv3_ssd_resnet101_img_cls_torchvision",
                "pt_monodepth2_mono_1024x320_depth_prediction_torchvision",
                "pt_regnet_regnet_x_800mf_img_cls_torchvision",
                "pt_resnet_resnet101_img_cls_torchvision",
                "pt_resnext_resnext50_32x4d_img_cls_osmr",
                "pt_vgg_bn_vgg19_obj_det_osmr",
                "pt_vovnet_ese_vovnet39b_obj_det_torchhub",
                "pt_xception_xception_img_cls_timm",
                "pt_yolov8_default_obj_det_github",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_dla_dla34_visual_bb_torchvision",
                "pt_hrnet_hrnetv2_w48_pose_estimation_osmr",
                "pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
                "pt_mobilenetv3_ssd_resnet152_img_cls_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_resnet_50_img_cls_hf",
                "pt_resnet_resnet152_img_cls_torchvision",
                "pt_resnext_resnext50_32x4d_img_cls_torchhub",
                "pt_unet_carvana_base_img_seg_github",
                "pt_vgg_bn_vgg19b_obj_det_osmr",
                "pt_vgg_vgg16_bn_img_cls_torchvision",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_yolov9_default_obj_det_github",
                "pd_mobilenetv1_basic_img_cls_paddlemodels",
                "pd_resnet_152_img_cls_paddlemodels",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_mobilenetv3_ssd_resnet50_img_cls_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
                "pt_resnet_resnet50_img_cls_torchvision",
                "pt_retinanet_retinanet_rn18fpn_obj_det_hf",
                "pt_unet_qubvel_img_seg_torchhub",
                "pt_vovnet_vovnet27s_img_cls_osmr",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
                "pt_yolo_v4_default_obj_det_github",
                "pd_resnet_18_img_cls_paddlemodels",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_dla_dla60x_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_mobilenetv1_basic_img_cls_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_retinanet_retinanet_rn34fpn_obj_det_hf",
                "pt_unet_base_img_seg_torchhub",
                "pt_vovnet_vovnet39_img_cls_osmr",
                "pt_vovnet_vovnet_v1_57_obj_det_torchhub",
                "pt_wideresnet_wide_resnet50_2_img_cls_timm",
                "pt_yolox_yolox_nano_obj_det_torchhub",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_resnet_34_img_cls_paddlemodels",
                "pt_dla_dla60x_c_visual_bb_torchvision",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
                "pt_vgg19_unet_default_sem_seg_github",
                "pt_vovnet_vovnet57_img_cls_osmr",
                "pt_xception_xception41_img_cls_timm",
                "pt_yolox_yolox_s_obj_det_torchhub",
                "pt_dla_dla102x_visual_bb_torchvision",
                "pt_hrnet_hrnetv2_w40_pose_estimation_osmr",
                "pt_inception_inception_v4_img_cls_timm",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_regnet_facebook_regnet_y_040_img_cls_hf",
                "pt_resnet_50_img_cls_timm",
                "pt_resnext_resnext26_32x4d_img_cls_osmr",
                "pt_ssd300_resnet50_base_img_cls_torchhub",
                "pt_vgg_vgg13_bn_img_cls_torchvision",
                "pt_vgg_vgg19_bn_obj_det_torchhub",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
                "pt_xception_xception71_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((512,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": [
                "pd_resnet_50_img_cls_paddlemodels",
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_dla_dla34_in1k_img_cls_timm",
                "pt_hrnet_hrnetv2_w30_pose_estimation_osmr",
                "pt_inception_v4_img_cls_osmr",
                "pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision",
                "pt_resnext_resnext101_32x8d_img_cls_torchhub",
                "pt_vgg_vgg11_bn_img_cls_torchvision",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_hrnet_hrnetv2_w32_pose_estimation_osmr",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision",
                "pt_resnext_resnext14_32x4d_img_cls_osmr",
                "pt_vgg_vgg19_bn_obj_det_timm",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_yolo_v6_yolov6s_obj_det_torchhub",
                "tf_resnet_resnet50_img_cls_keras",
                "pt_dla_dla169_visual_bb_torchvision",
                "pt_hrnet_hrnetv2_w44_pose_estimation_osmr",
                "pt_mobilenetv3_ssd_resnet101_img_cls_torchvision",
                "pt_monodepth2_mono_1024x320_depth_prediction_torchvision",
                "pt_resnet_resnet101_img_cls_torchvision",
                "pt_resnext_resnext50_32x4d_img_cls_osmr",
                "pt_vgg_bn_vgg19_obj_det_osmr",
                "pt_vovnet_ese_vovnet39b_obj_det_torchhub",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_dla_dla34_visual_bb_torchvision",
                "pt_hrnet_hrnetv2_w48_pose_estimation_osmr",
                "pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
                "pt_mobilenetv3_ssd_resnet152_img_cls_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_resnet_50_img_cls_hf",
                "pt_resnet_resnet152_img_cls_torchvision",
                "pt_resnext_resnext50_32x4d_img_cls_torchhub",
                "pt_unet_carvana_base_img_seg_github",
                "pt_vgg_bn_vgg19b_obj_det_osmr",
                "pt_vgg_vgg16_bn_img_cls_torchvision",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_yolov9_default_obj_det_github",
                "pd_mobilenetv1_basic_img_cls_paddlemodels",
                "pd_resnet_152_img_cls_paddlemodels",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_mobilenetv3_ssd_resnet50_img_cls_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
                "pt_resnet_resnet50_img_cls_torchvision",
                "pt_retinanet_retinanet_rn18fpn_obj_det_hf",
                "pt_unet_qubvel_img_seg_torchhub",
                "pt_vovnet_vovnet27s_img_cls_osmr",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
                "pt_yolo_v4_default_obj_det_github",
                "pd_resnet_18_img_cls_paddlemodels",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_dla_dla60x_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_mobilenetv1_basic_img_cls_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_regnet_regnet_x_16gf_img_cls_torchvision",
                "pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub",
                "pt_retinanet_retinanet_rn34fpn_obj_det_hf",
                "pt_unet_base_img_seg_torchhub",
                "pt_vovnet_vovnet39_img_cls_osmr",
                "pt_vovnet_vovnet_v1_57_obj_det_torchhub",
                "pt_wideresnet_wide_resnet50_2_img_cls_timm",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_resnet_34_img_cls_paddlemodels",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
                "pt_resnext_resnext101_64x4d_img_cls_osmr",
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
                "pt_vgg19_unet_default_sem_seg_github",
                "pt_vovnet_vovnet57_img_cls_osmr",
                "pt_yolo_v6_yolov6m_obj_det_torchhub",
                "pt_yolox_yolox_s_obj_det_torchhub",
                "pt_dla_dla102x_visual_bb_torchvision",
                "pt_hrnet_hrnetv2_w40_pose_estimation_osmr",
                "pt_inception_inception_v4_img_cls_timm",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_regnet_facebook_regnet_y_040_img_cls_hf",
                "pt_resnet_50_img_cls_timm",
                "pt_resnext_resnext26_32x4d_img_cls_osmr",
                "pt_ssd300_resnet50_base_img_cls_torchhub",
                "pt_vgg_vgg13_bn_img_cls_torchvision",
                "pt_vgg_vgg19_bn_obj_det_torchhub",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add20,
        [((512,), torch.float32)],
        {
            "model_names": [
                "pd_resnet_50_img_cls_paddlemodels",
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_dla_dla34_in1k_img_cls_timm",
                "pt_hrnet_hrnetv2_w30_pose_estimation_osmr",
                "pt_inception_v4_img_cls_osmr",
                "pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision",
                "pt_resnext_resnext101_32x8d_img_cls_torchhub",
                "pt_vgg_vgg11_bn_img_cls_torchvision",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_hrnet_hrnetv2_w32_pose_estimation_osmr",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision",
                "pt_resnext_resnext14_32x4d_img_cls_osmr",
                "pt_vgg_vgg19_bn_obj_det_timm",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "tf_resnet_resnet50_img_cls_keras",
                "pt_dla_dla169_visual_bb_torchvision",
                "pt_hrnet_hrnetv2_w44_pose_estimation_osmr",
                "pt_mobilenetv3_ssd_resnet101_img_cls_torchvision",
                "pt_monodepth2_mono_1024x320_depth_prediction_torchvision",
                "pt_resnet_resnet101_img_cls_torchvision",
                "pt_resnext_resnext50_32x4d_img_cls_osmr",
                "pt_vgg_bn_vgg19_obj_det_osmr",
                "pt_vovnet_ese_vovnet39b_obj_det_torchhub",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_dla_dla34_visual_bb_torchvision",
                "pt_hrnet_hrnetv2_w48_pose_estimation_osmr",
                "pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
                "pt_mobilenetv3_ssd_resnet152_img_cls_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_resnet_50_img_cls_hf",
                "pt_resnet_resnet152_img_cls_torchvision",
                "pt_resnext_resnext50_32x4d_img_cls_torchhub",
                "pt_unet_carvana_base_img_seg_github",
                "pt_vgg_bn_vgg19b_obj_det_osmr",
                "pt_vgg_vgg16_bn_img_cls_torchvision",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_yolov9_default_obj_det_github",
                "pd_mobilenetv1_basic_img_cls_paddlemodels",
                "pd_resnet_152_img_cls_paddlemodels",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_mobilenetv3_ssd_resnet50_img_cls_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
                "pt_resnet_resnet50_img_cls_torchvision",
                "pt_retinanet_retinanet_rn18fpn_obj_det_hf",
                "pt_unet_qubvel_img_seg_torchhub",
                "pt_vovnet_vovnet27s_img_cls_osmr",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
                "pt_yolo_v4_default_obj_det_github",
                "pd_resnet_18_img_cls_paddlemodels",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_dla_dla60x_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_mobilenetv1_basic_img_cls_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_regnet_regnet_x_16gf_img_cls_torchvision",
                "pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub",
                "pt_retinanet_retinanet_rn34fpn_obj_det_hf",
                "pt_unet_base_img_seg_torchhub",
                "pt_vovnet_vovnet39_img_cls_osmr",
                "pt_vovnet_vovnet_v1_57_obj_det_torchhub",
                "pt_wideresnet_wide_resnet50_2_img_cls_timm",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_resnet_34_img_cls_paddlemodels",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
                "pt_resnext_resnext101_64x4d_img_cls_osmr",
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
                "pt_vgg19_unet_default_sem_seg_github",
                "pt_vovnet_vovnet57_img_cls_osmr",
                "pt_yolox_yolox_s_obj_det_torchhub",
                "pt_dla_dla102x_visual_bb_torchvision",
                "pt_hrnet_hrnetv2_w40_pose_estimation_osmr",
                "pt_inception_inception_v4_img_cls_timm",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_regnet_facebook_regnet_y_040_img_cls_hf",
                "pt_resnet_50_img_cls_timm",
                "pt_resnext_resnext26_32x4d_img_cls_osmr",
                "pt_ssd300_resnet50_base_img_cls_torchhub",
                "pt_vgg_vgg13_bn_img_cls_torchvision",
                "pt_vgg_vgg19_bn_obj_det_torchhub",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 512, 28, 28), torch.float32), ((1, 512, 28, 28), torch.float32)],
        {
            "model_names": [
                "pd_resnet_50_img_cls_paddlemodels",
                "pt_resnext_resnext101_32x8d_img_cls_torchhub",
                "pt_resnext_resnext14_32x4d_img_cls_osmr",
                "pt_resnet_resnet101_img_cls_torchvision",
                "pt_resnext_resnext50_32x4d_img_cls_osmr",
                "pt_resnet_50_img_cls_hf",
                "pt_resnet_resnet152_img_cls_torchvision",
                "pt_resnext_resnext50_32x4d_img_cls_torchhub",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "onnx_resnet_50_img_cls_hf",
                "pd_resnet_152_img_cls_paddlemodels",
                "pt_resnet_resnet50_img_cls_torchvision",
                "pt_unet_qubvel_img_seg_torchhub",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
                "pt_regnet_regnet_x_16gf_img_cls_torchvision",
                "pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub",
                "pt_wideresnet_wide_resnet50_2_img_cls_timm",
                "pt_resnext_resnext101_64x4d_img_cls_osmr",
                "pt_resnet_50_img_cls_timm",
                "pt_resnext_resnext26_32x4d_img_cls_osmr",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1024,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": [
                "pd_resnet_50_img_cls_paddlemodels",
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_hrnet_hrnetv2_w30_pose_estimation_osmr",
                "pt_resnext_resnext101_32x8d_img_cls_torchhub",
                "pt_xception_xception65_img_cls_timm",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_hrnet_hrnetv2_w32_pose_estimation_osmr",
                "pt_resnext_resnext14_32x4d_img_cls_osmr",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_xception_xception71_tf_in1k_img_cls_timm",
                "tf_resnet_resnet50_img_cls_keras",
                "pt_dla_dla169_visual_bb_torchvision",
                "pt_hrnet_hrnetv2_w44_pose_estimation_osmr",
                "pt_mobilenetv3_ssd_resnet101_img_cls_torchvision",
                "pt_resnet_resnet101_img_cls_torchvision",
                "pt_resnext_resnext50_32x4d_img_cls_osmr",
                "pt_vovnet_ese_vovnet39b_obj_det_torchhub",
                "pt_xception_xception_img_cls_timm",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_hrnet_hrnetv2_w48_pose_estimation_osmr",
                "pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
                "pt_mobilenetv3_ssd_resnet152_img_cls_torchvision",
                "pt_resnet_50_img_cls_hf",
                "pt_resnet_resnet152_img_cls_torchvision",
                "pt_resnext_resnext50_32x4d_img_cls_torchhub",
                "pt_unet_carvana_base_img_seg_github",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pd_mobilenetv1_basic_img_cls_paddlemodels",
                "pd_resnet_152_img_cls_paddlemodels",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_mobilenetv3_ssd_resnet50_img_cls_torchvision",
                "pt_resnet_resnet50_img_cls_torchvision",
                "pt_unet_qubvel_img_seg_torchhub",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
                "pt_yolo_v4_default_obj_det_github",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_dla_dla60x_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_mobilenetv1_basic_img_cls_torchvision",
                "pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub",
                "pt_vovnet_vovnet39_img_cls_osmr",
                "pt_vovnet_vovnet_v1_57_obj_det_torchhub",
                "pt_wideresnet_wide_resnet50_2_img_cls_timm",
                "pd_densenet_121_img_cls_paddlemodels",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
                "pt_resnext_resnext101_64x4d_img_cls_osmr",
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
                "pt_vovnet_vovnet57_img_cls_osmr",
                "pt_xception_xception41_img_cls_timm",
                "pt_dla_dla102x_visual_bb_torchvision",
                "pt_hrnet_hrnetv2_w40_pose_estimation_osmr",
                "pt_resnet_50_img_cls_timm",
                "pt_resnext_resnext26_32x4d_img_cls_osmr",
                "pt_ssd300_resnet50_base_img_cls_torchhub",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
                "pt_xception_xception71_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add21,
        [((1024,), torch.float32)],
        {
            "model_names": [
                "pd_resnet_50_img_cls_paddlemodels",
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_hrnet_hrnetv2_w30_pose_estimation_osmr",
                "pt_resnext_resnext101_32x8d_img_cls_torchhub",
                "pt_xception_xception65_img_cls_timm",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_hrnet_hrnetv2_w32_pose_estimation_osmr",
                "pt_resnext_resnext14_32x4d_img_cls_osmr",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_xception_xception71_tf_in1k_img_cls_timm",
                "tf_resnet_resnet50_img_cls_keras",
                "pt_dla_dla169_visual_bb_torchvision",
                "pt_hrnet_hrnetv2_w44_pose_estimation_osmr",
                "pt_mobilenetv3_ssd_resnet101_img_cls_torchvision",
                "pt_resnet_resnet101_img_cls_torchvision",
                "pt_resnext_resnext50_32x4d_img_cls_osmr",
                "pt_vovnet_ese_vovnet39b_obj_det_torchhub",
                "pt_xception_xception_img_cls_timm",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_hrnet_hrnetv2_w48_pose_estimation_osmr",
                "pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
                "pt_mobilenetv3_ssd_resnet152_img_cls_torchvision",
                "pt_resnet_50_img_cls_hf",
                "pt_resnet_resnet152_img_cls_torchvision",
                "pt_resnext_resnext50_32x4d_img_cls_torchhub",
                "pt_unet_carvana_base_img_seg_github",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pd_mobilenetv1_basic_img_cls_paddlemodels",
                "pd_resnet_152_img_cls_paddlemodels",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_mobilenetv3_ssd_resnet50_img_cls_torchvision",
                "pt_resnet_resnet50_img_cls_torchvision",
                "pt_unet_qubvel_img_seg_torchhub",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
                "pt_yolo_v4_default_obj_det_github",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_dla_dla60x_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_mobilenetv1_basic_img_cls_torchvision",
                "pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub",
                "pt_vovnet_vovnet39_img_cls_osmr",
                "pt_vovnet_vovnet_v1_57_obj_det_torchhub",
                "pt_wideresnet_wide_resnet50_2_img_cls_timm",
                "pd_densenet_121_img_cls_paddlemodels",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
                "pt_resnext_resnext101_64x4d_img_cls_osmr",
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
                "pt_vovnet_vovnet57_img_cls_osmr",
                "pt_xception_xception41_img_cls_timm",
                "pt_dla_dla102x_visual_bb_torchvision",
                "pt_hrnet_hrnetv2_w40_pose_estimation_osmr",
                "pt_resnet_50_img_cls_timm",
                "pt_resnext_resnext26_32x4d_img_cls_osmr",
                "pt_ssd300_resnet50_base_img_cls_torchhub",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
                "pt_xception_xception71_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1024, 14, 14), torch.float32), ((1024, 1, 1), torch.float32)],
        {
            "model_names": [
                "pd_resnet_50_img_cls_paddlemodels",
                "pt_resnext_resnext101_32x8d_img_cls_torchhub",
                "pt_vit_google_vit_large_patch16_224_img_cls_hf",
                "onnx_dla_dla102x2_visual_bb_torchvision",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_resnext_resnext14_32x4d_img_cls_osmr",
                "pt_beit_microsoft_beit_large_patch16_224_img_cls_hf",
                "pt_resnet_resnet101_img_cls_torchvision",
                "pt_resnext_resnext50_32x4d_img_cls_osmr",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_resnet_50_img_cls_hf",
                "pt_resnet_resnet152_img_cls_torchvision",
                "pt_resnext_resnext50_32x4d_img_cls_torchhub",
                "pt_unet_carvana_base_img_seg_github",
                "pt_vit_vit_l_16_img_cls_torchvision",
                "onnx_resnet_50_img_cls_hf",
                "pd_resnet_152_img_cls_paddlemodels",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_resnet_resnet50_img_cls_torchvision",
                "pt_unet_qubvel_img_seg_torchhub",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
                "onnx_dla_dla60x_visual_bb_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_dla_dla60x_visual_bb_torchvision",
                "pt_mlp_mixer_mixer_l16_224_img_cls_timm",
                "pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub",
                "pt_wideresnet_wide_resnet50_2_img_cls_timm",
                "pd_densenet_121_img_cls_paddlemodels",
                "pt_mlp_mixer_mixer_l16_224_in21k_img_cls_timm",
                "pt_resnext_resnext101_64x4d_img_cls_osmr",
                "onnx_dla_dla102x_visual_bb_torchvision",
                "pt_dla_dla102x_visual_bb_torchvision",
                "pt_resnet_50_img_cls_timm",
                "pt_resnext_resnext26_32x4d_img_cls_osmr",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1024, 14, 14), torch.float32), ((1, 1024, 14, 14), torch.float32)],
        {
            "model_names": [
                "pd_resnet_50_img_cls_paddlemodels",
                "pt_resnext_resnext101_32x8d_img_cls_torchhub",
                "pt_resnext_resnext14_32x4d_img_cls_osmr",
                "pt_resnet_resnet101_img_cls_torchvision",
                "pt_resnext_resnext50_32x4d_img_cls_osmr",
                "pt_resnet_50_img_cls_hf",
                "pt_resnet_resnet152_img_cls_torchvision",
                "pt_resnext_resnext50_32x4d_img_cls_torchhub",
                "onnx_resnet_50_img_cls_hf",
                "pd_resnet_152_img_cls_paddlemodels",
                "pt_resnet_resnet50_img_cls_torchvision",
                "pt_unet_qubvel_img_seg_torchhub",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
                "pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub",
                "pt_wideresnet_wide_resnet50_2_img_cls_timm",
                "pt_resnext_resnext101_64x4d_img_cls_osmr",
                "pt_resnet_50_img_cls_timm",
                "pt_resnext_resnext26_32x4d_img_cls_osmr",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((2048,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": [
                "pd_resnet_50_img_cls_paddlemodels",
                "pt_hrnet_hrnetv2_w30_pose_estimation_osmr",
                "pt_resnext_resnext101_32x8d_img_cls_torchhub",
                "pt_xception_xception65_img_cls_timm",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_hrnet_hrnetv2_w32_pose_estimation_osmr",
                "pt_resnext_resnext14_32x4d_img_cls_osmr",
                "pt_xception_xception71_tf_in1k_img_cls_timm",
                "tf_resnet_resnet50_img_cls_keras",
                "pt_hrnet_hrnetv2_w44_pose_estimation_osmr",
                "pt_mobilenetv3_ssd_resnet101_img_cls_torchvision",
                "pt_resnet_resnet101_img_cls_torchvision",
                "pt_resnext_resnext50_32x4d_img_cls_osmr",
                "pt_xception_xception_img_cls_timm",
                "pt_hrnet_hrnetv2_w48_pose_estimation_osmr",
                "pt_mobilenetv3_ssd_resnet152_img_cls_torchvision",
                "pt_resnet_50_img_cls_hf",
                "pt_resnet_resnet152_img_cls_torchvision",
                "pt_resnext_resnext50_32x4d_img_cls_torchhub",
                "pd_resnet_152_img_cls_paddlemodels",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_mobilenetv3_ssd_resnet50_img_cls_torchvision",
                "pt_resnet_resnet50_img_cls_torchvision",
                "pt_unet_qubvel_img_seg_torchhub",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_regnet_regnet_x_16gf_img_cls_torchvision",
                "pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub",
                "pt_wideresnet_wide_resnet50_2_img_cls_timm",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
                "pt_resnext_resnext101_64x4d_img_cls_osmr",
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
                "pt_xception_xception41_img_cls_timm",
                "pt_hrnet_hrnetv2_w40_pose_estimation_osmr",
                "pt_resnet_50_img_cls_timm",
                "pt_resnext_resnext26_32x4d_img_cls_osmr",
                "pt_xception_xception71_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add22,
        [((2048,), torch.float32)],
        {
            "model_names": [
                "pd_resnet_50_img_cls_paddlemodels",
                "pt_hrnet_hrnetv2_w30_pose_estimation_osmr",
                "pt_resnext_resnext101_32x8d_img_cls_torchhub",
                "pt_xception_xception65_img_cls_timm",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_hrnet_hrnetv2_w32_pose_estimation_osmr",
                "pt_resnext_resnext14_32x4d_img_cls_osmr",
                "pt_xception_xception71_tf_in1k_img_cls_timm",
                "tf_resnet_resnet50_img_cls_keras",
                "pt_hrnet_hrnetv2_w44_pose_estimation_osmr",
                "pt_mobilenetv3_ssd_resnet101_img_cls_torchvision",
                "pt_resnet_resnet101_img_cls_torchvision",
                "pt_resnext_resnext50_32x4d_img_cls_osmr",
                "pt_xception_xception_img_cls_timm",
                "pt_hrnet_hrnetv2_w48_pose_estimation_osmr",
                "pt_mobilenetv3_ssd_resnet152_img_cls_torchvision",
                "pt_resnet_50_img_cls_hf",
                "pt_resnet_resnet152_img_cls_torchvision",
                "pt_resnext_resnext50_32x4d_img_cls_torchhub",
                "pd_resnet_152_img_cls_paddlemodels",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_mobilenetv3_ssd_resnet50_img_cls_torchvision",
                "pt_resnet_resnet50_img_cls_torchvision",
                "pt_unet_qubvel_img_seg_torchhub",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_regnet_regnet_x_16gf_img_cls_torchvision",
                "pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub",
                "pt_wideresnet_wide_resnet50_2_img_cls_timm",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
                "pt_resnext_resnext101_64x4d_img_cls_osmr",
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
                "pt_xception_xception41_img_cls_timm",
                "pt_hrnet_hrnetv2_w40_pose_estimation_osmr",
                "pt_resnet_50_img_cls_timm",
                "pt_resnext_resnext26_32x4d_img_cls_osmr",
                "pt_xception_xception71_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 2048, 7, 7), torch.float32), ((2048, 1, 1), torch.float32)],
        {
            "model_names": [
                "pd_resnet_50_img_cls_paddlemodels",
                "pt_hrnet_hrnetv2_w30_pose_estimation_osmr",
                "pt_resnext_resnext101_32x8d_img_cls_torchhub",
                "onnx_dla_dla102x2_visual_bb_torchvision",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_hrnet_hrnetv2_w32_pose_estimation_osmr",
                "pt_resnext_resnext14_32x4d_img_cls_osmr",
                "pt_hrnet_hrnetv2_w44_pose_estimation_osmr",
                "pt_resnet_resnet101_img_cls_torchvision",
                "pt_resnext_resnext50_32x4d_img_cls_osmr",
                "pt_hrnet_hrnetv2_w48_pose_estimation_osmr",
                "pt_resnet_50_img_cls_hf",
                "pt_resnet_resnet152_img_cls_torchvision",
                "pt_resnext_resnext50_32x4d_img_cls_torchhub",
                "onnx_resnet_50_img_cls_hf",
                "pd_resnet_152_img_cls_paddlemodels",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_resnet_resnet50_img_cls_torchvision",
                "pt_unet_qubvel_img_seg_torchhub",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_regnet_regnet_x_16gf_img_cls_torchvision",
                "pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub",
                "pt_wideresnet_wide_resnet50_2_img_cls_timm",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
                "pt_resnext_resnext101_64x4d_img_cls_osmr",
                "pt_hrnet_hrnetv2_w40_pose_estimation_osmr",
                "pt_resnet_50_img_cls_timm",
                "pt_resnext_resnext26_32x4d_img_cls_osmr",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 2048, 7, 7), torch.float32), ((1, 2048, 7, 7), torch.float32)],
        {
            "model_names": [
                "pd_resnet_50_img_cls_paddlemodels",
                "pt_resnext_resnext101_32x8d_img_cls_torchhub",
                "pt_resnext_resnext14_32x4d_img_cls_osmr",
                "pt_resnet_resnet101_img_cls_torchvision",
                "pt_resnext_resnext50_32x4d_img_cls_osmr",
                "pt_resnet_50_img_cls_hf",
                "pt_resnet_resnet152_img_cls_torchvision",
                "pt_resnext_resnext50_32x4d_img_cls_torchhub",
                "onnx_resnet_50_img_cls_hf",
                "pd_resnet_152_img_cls_paddlemodels",
                "pt_resnet_resnet50_img_cls_torchvision",
                "pt_unet_qubvel_img_seg_torchhub",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
                "pt_regnet_regnet_x_16gf_img_cls_torchvision",
                "pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub",
                "pt_wideresnet_wide_resnet50_2_img_cls_timm",
                "pt_resnext_resnext101_64x4d_img_cls_osmr",
                "pt_resnet_50_img_cls_timm",
                "pt_resnext_resnext26_32x4d_img_cls_osmr",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 128, 128), torch.float32), ((1, 128, 128), torch.float32)],
        {
            "model_names": [
                "pt_albert_base_v1_mlm_hf",
                "pt_albert_base_v1_token_cls_hf",
                "pt_albert_base_v2_mlm_hf",
                "pt_albert_large_v1_token_cls_hf",
                "pt_albert_large_v2_mlm_hf",
                "pt_albert_large_v2_token_cls_hf",
                "pt_albert_xlarge_v1_mlm_hf",
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
                "pt_albert_xxlarge_v2_token_cls_hf",
                "pt_albert_base_v2_token_cls_hf",
                "pt_albert_large_v1_mlm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add14,
        [((1, 128, 768), torch.float32)],
        {
            "model_names": [
                "pt_albert_base_v1_mlm_hf",
                "pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf",
                "pt_albert_base_v1_token_cls_hf",
                "pt_albert_base_v2_mlm_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_roberta_xlm_roberta_base_mlm_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
                "pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder",
                "pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf",
                "pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf",
                "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "pt_albert_base_v2_token_cls_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 12, 128, 128), torch.float32), ((1, 1, 1, 128), torch.float32)],
        {
            "model_names": [
                "pt_albert_base_v1_mlm_hf",
                "pt_albert_base_v1_token_cls_hf",
                "pt_albert_base_v2_mlm_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_roberta_xlm_roberta_base_mlm_hf",
                "pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder",
                "pt_albert_base_v2_token_cls_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 128, 768), torch.float32), ((1, 128, 768), torch.float32)],
        {
            "model_names": [
                "pt_albert_base_v1_mlm_hf",
                "pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf",
                "pt_albert_base_v1_token_cls_hf",
                "pt_albert_base_v2_mlm_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder",
                "pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_roberta_xlm_roberta_base_mlm_hf",
                "onnx_bert_bert_base_uncased_mlm_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
                "pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder",
                "pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf",
                "pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf",
                "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "pt_albert_base_v2_token_cls_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add15,
        [((1, 128, 3072), torch.float32)],
        {
            "model_names": [
                "pt_albert_base_v1_mlm_hf",
                "pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf",
                "pt_albert_base_v1_token_cls_hf",
                "pt_albert_base_v2_mlm_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_roberta_xlm_roberta_base_mlm_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
                "pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder",
                "pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf",
                "pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf",
                "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "pt_albert_base_v2_token_cls_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add19,
        [((1, 128, 128), torch.float32)],
        {
            "model_names": [
                "pt_albert_base_v1_mlm_hf",
                "pt_albert_base_v2_mlm_hf",
                "pt_albert_large_v2_mlm_hf",
                "pt_albert_xlarge_v1_mlm_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
                "pt_albert_large_v1_mlm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add23,
        [((1, 128, 30000), torch.float32)],
        {
            "model_names": [
                "pt_albert_base_v1_mlm_hf",
                "pt_albert_base_v2_mlm_hf",
                "pt_albert_large_v2_mlm_hf",
                "pt_albert_xlarge_v1_mlm_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
                "pt_albert_large_v1_mlm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 9, 128), torch.float32), ((1, 9, 128), torch.float32)],
        {"model_names": ["pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Add14,
        [((1, 9, 768), torch.float32)],
        {
            "model_names": [
                "pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf",
                "pd_ernie_1_0_seq_cls_padlenlp",
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
                "pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf",
                "pd_ernie_1_0_seq_cls_padlenlp",
                "pd_ernie_1_0_qa_padlenlp",
                "pd_bert_chinese_roberta_base_mlm_padlenlp",
                "pd_bert_bert_base_uncased_qa_padlenlp",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 9, 768), torch.float32), ((1, 9, 768), torch.float32)],
        {
            "model_names": [
                "pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf",
                "pd_ernie_1_0_seq_cls_padlenlp",
                "pd_ernie_1_0_qa_padlenlp",
                "pd_bert_chinese_roberta_base_mlm_padlenlp",
                "pd_bert_bert_base_uncased_qa_padlenlp",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add15,
        [((1, 9, 3072), torch.float32)],
        {
            "model_names": [
                "pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf",
                "pd_ernie_1_0_seq_cls_padlenlp",
                "pd_ernie_1_0_qa_padlenlp",
                "pd_bert_chinese_roberta_base_mlm_padlenlp",
                "pd_bert_bert_base_uncased_qa_padlenlp",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add14,
        [((1, 768), torch.float32)],
        {
            "model_names": [
                "pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf",
                "pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf",
                "pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf",
                "pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf",
                "onnx_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf",
                "pd_blip_text_salesforce_blip_image_captioning_base_text_enc_padlenlp",
                "pd_ernie_1_0_seq_cls_padlenlp",
                "pd_bert_bert_base_uncased_seq_cls_padlenlp",
                "pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf",
                "pd_bert_bert_base_japanese_seq_cls_padlenlp",
                "pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf",
                "pd_bert_chinese_roberta_base_seq_cls_padlenlp",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add16,
        [((1, 2), torch.float32)],
        {
            "model_names": [
                "pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf",
                "pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf",
                "pt_rcnn_base_obj_det_torchvision_rect_0",
                "pd_ernie_1_0_seq_cls_padlenlp",
                "pd_bert_bert_base_uncased_seq_cls_padlenlp",
                "pd_bert_bert_base_japanese_seq_cls_padlenlp",
                "pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf",
                "pd_bert_chinese_roberta_base_seq_cls_padlenlp",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 64, 55, 55), torch.float32), ((64, 1, 1), torch.float32)],
        {"model_names": ["pt_alexnet_base_img_cls_osmr", "pt_alexnet_base_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 192, 27, 27), torch.float32), ((192, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_alexnet_base_img_cls_osmr",
                "pt_alexnet_base_img_cls_torchhub",
                "pt_rcnn_base_obj_det_torchvision_rect_0",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 384, 13, 13), torch.float32), ((384, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_alexnet_base_img_cls_osmr",
                "pt_yolox_yolox_tiny_obj_det_torchhub",
                "pt_alexnet_base_img_cls_torchhub",
                "pt_rcnn_base_obj_det_torchvision_rect_0",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 256, 13, 13), torch.float32), ((256, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_alexnet_base_img_cls_osmr",
                "pt_alexnet_base_img_cls_torchhub",
                "pt_rcnn_base_obj_det_torchvision_rect_0",
                "pt_yolox_yolox_nano_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add24,
        [((1, 4096), torch.float32)],
        {
            "model_names": [
                "pt_alexnet_base_img_cls_osmr",
                "pt_vgg_19_obj_det_hf",
                "pt_vgg_vgg11_bn_img_cls_torchvision",
                "pt_alexnet_base_img_cls_torchhub",
                "pt_rcnn_base_obj_det_torchvision_rect_0",
                "pt_vgg_vgg13_img_cls_torchvision",
                "pt_vgg_bn_vgg19_obj_det_osmr",
                "pt_vgg_vgg16_img_cls_torchvision",
                "pt_vgg_bn_vgg19b_obj_det_osmr",
                "pt_vgg_vgg16_bn_img_cls_torchvision",
                "pt_vgg_vgg16_obj_det_osmr",
                "pt_vgg_vgg19_obj_det_osmr",
                "pd_alexnet_base_img_cls_paddlemodels",
                "pt_vgg_vgg11_img_cls_torchvision",
                "pt_vgg_vgg13_bn_img_cls_torchvision",
                "pt_vgg_vgg19_bn_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add25,
        [((1, 12, 128, 128), torch.float32)],
        {
            "model_names": [
                "pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf",
                "pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf",
                "onnx_bert_bert_base_uncased_mlm_hf",
                "pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add26,
        [((1, 197, 768), torch.float32)],
        {
            "model_names": [
                "pt_deit_facebook_deit_base_distilled_patch16_224_img_cls_hf",
                "pt_deit_facebook_deit_base_patch16_224_img_cls_hf",
                "pt_vit_vit_b_16_img_cls_torchvision",
                "onnx_vit_base_google_vit_base_patch16_224_img_cls_hf",
                "pt_vit_google_vit_base_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add14,
        [((1, 197, 768), torch.float32)],
        {
            "model_names": [
                "pt_deit_facebook_deit_base_distilled_patch16_224_img_cls_hf",
                "pt_deit_facebook_deit_base_patch16_224_img_cls_hf",
                "pt_vit_vit_b_16_img_cls_torchvision",
                "pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf",
                "pt_vit_google_vit_base_patch16_224_img_cls_hf",
                "pt_beit_microsoft_beit_base_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 197, 768), torch.float32), ((1, 197, 768), torch.float32)],
        {
            "model_names": [
                "pt_deit_facebook_deit_base_distilled_patch16_224_img_cls_hf",
                "pt_deit_facebook_deit_base_patch16_224_img_cls_hf",
                "pt_vit_vit_b_16_img_cls_torchvision",
                "onnx_vit_base_google_vit_base_patch16_224_img_cls_hf",
                "pt_vit_google_vit_base_patch16_224_img_cls_hf",
                "pt_beit_microsoft_beit_base_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add15,
        [((1, 197, 3072), torch.float32)],
        {
            "model_names": [
                "pt_deit_facebook_deit_base_distilled_patch16_224_img_cls_hf",
                "pt_deit_facebook_deit_base_patch16_224_img_cls_hf",
                "pt_vit_vit_b_16_img_cls_torchvision",
                "pt_vit_google_vit_base_patch16_224_img_cls_hf",
                "pt_beit_microsoft_beit_base_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add27,
        [((1, 100, 256), torch.float32)],
        {
            "model_names": [
                "pt_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
                "pt_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add18,
        [((1, 100, 256), torch.float32)],
        {
            "model_names": [
                "pt_detr_facebook_detr_resnet_50_obj_det_hf",
                "pt_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 100, 256), torch.float32), ((1, 100, 256), torch.float32)],
        {
            "model_names": [
                "pt_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
                "pt_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add13,
        [((1, 64, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_detr_facebook_detr_resnet_50_obj_det_hf",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pt_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 64, 214, 320), torch.float32), ((1, 64, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_detr_facebook_detr_resnet_50_obj_det_hf",
                "pt_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 64, 107, 160), torch.float32), ((1, 64, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_detr_facebook_detr_resnet_50_obj_det_hf",
                "pt_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 256, 107, 160), torch.float32), ((1, 256, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_detr_facebook_detr_resnet_50_obj_det_hf",
                "pt_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 256, 107, 160), torch.float32), ((1, 256, 107, 160), torch.float32)],
        {
            "model_names": [
                "pt_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
                "pt_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add13,
        [((1, 128, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_detr_facebook_detr_resnet_50_obj_det_hf",
                "pt_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 128, 107, 160), torch.float32), ((1, 128, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_detr_facebook_detr_resnet_50_obj_det_hf",
                "pt_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 128, 54, 80), torch.float32), ((1, 128, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_detr_facebook_detr_resnet_50_obj_det_hf",
                "pt_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 512, 54, 80), torch.float32), ((1, 512, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_detr_facebook_detr_resnet_50_obj_det_hf",
                "pt_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 512, 54, 80), torch.float32), ((1, 512, 54, 80), torch.float32)],
        {
            "model_names": [
                "pt_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
                "pt_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 256, 54, 80), torch.float32), ((1, 256, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_detr_facebook_detr_resnet_50_obj_det_hf",
                "pt_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 256, 27, 40), torch.float32), ((1, 256, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_detr_facebook_detr_resnet_50_obj_det_hf",
                "pt_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1024, 27, 40), torch.float32), ((1, 1024, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_detr_facebook_detr_resnet_50_obj_det_hf",
                "pt_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1024, 27, 40), torch.float32), ((1, 1024, 27, 40), torch.float32)],
        {
            "model_names": [
                "pt_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
                "pt_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 512, 27, 40), torch.float32), ((1, 512, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_detr_facebook_detr_resnet_50_obj_det_hf",
                "pt_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 512, 14, 20), torch.float32), ((1, 512, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_detr_facebook_detr_resnet_50_obj_det_hf",
                "pt_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add13,
        [((1, 2048, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_detr_facebook_detr_resnet_50_obj_det_hf",
                "pt_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 2048, 14, 20), torch.float32), ((1, 2048, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_detr_facebook_detr_resnet_50_obj_det_hf",
                "pt_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 2048, 14, 20), torch.float32), ((1, 2048, 14, 20), torch.float32)],
        {
            "model_names": [
                "pt_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
                "pt_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 256, 14, 20), torch.float32), ((256, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_detr_facebook_detr_resnet_50_obj_det_hf",
                "pt_yolo_v6_yolov6n_obj_det_torchhub",
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "pt_yolo_v6_yolov6s_obj_det_torchhub",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
                "pt_yolo_v6_yolov6l_obj_det_torchhub",
                "pt_yolo_v6_yolov6m_obj_det_torchhub",
                "pt_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add28,
        [((1, 280, 256), torch.float32)],
        {
            "model_names": [
                "pt_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
                "pt_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add18,
        [((1, 280, 256), torch.float32)],
        {
            "model_names": [
                "pt_detr_facebook_detr_resnet_50_obj_det_hf",
                "pt_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add29,
        [((1, 8, 280, 280), torch.float32)],
        {
            "model_names": [
                "pt_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
                "pt_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 280, 256), torch.float32), ((1, 280, 256), torch.float32)],
        {
            "model_names": [
                "pt_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
                "pt_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add22,
        [((1, 280, 2048), torch.float32)],
        {
            "model_names": [
                "pt_detr_facebook_detr_resnet_50_obj_det_hf",
                "pt_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add30,
        [((1, 8, 100, 280), torch.float32)],
        {
            "model_names": [
                "pt_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
                "pt_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add22,
        [((1, 100, 2048), torch.float32)],
        {
            "model_names": [
                "pt_detr_facebook_detr_resnet_50_obj_det_hf",
                "pt_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add31,
        [((1, 100, 92), torch.float32)],
        {
            "model_names": ["pt_detr_facebook_detr_resnet_50_obj_det_hf", "pt_yolos_hustvl_yolos_tiny_obj_det_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((16,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": [
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_dla_dla34_in1k_img_cls_timm",
                "pt_googlenet_base_img_cls_torchvision",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_yolo_v6_yolov6n_obj_det_torchhub",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_dla_dla169_visual_bb_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_yolov8_default_obj_det_github",
                "TranslatedLayer",
                "pt_dla_dla34_visual_bb_torchvision",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_unet_qubvel_img_seg_torchhub",
                "pd_paddleocr_v4_rec_ch_scene_text_recognition_paddlemodels",
                "pt_dla_dla60x_visual_bb_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_yolox_yolox_nano_obj_det_torchhub",
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
                "pt_dla_dla60x_c_visual_bb_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
                "pt_dla_dla102x_visual_bb_torchvision",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add32,
        [((16,), torch.float32)],
        {
            "model_names": [
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_dla_dla34_in1k_img_cls_timm",
                "pt_googlenet_base_img_cls_torchvision",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_dla_dla169_visual_bb_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_yolov8_default_obj_det_github",
                "TranslatedLayer",
                "pt_dla_dla34_visual_bb_torchvision",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_unet_qubvel_img_seg_torchhub",
                "pd_paddleocr_v4_rec_ch_scene_text_recognition_paddlemodels",
                "pt_dla_dla60x_visual_bb_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_yolox_yolox_nano_obj_det_torchhub",
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
                "pt_dla_dla60x_c_visual_bb_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
                "pt_dla_dla102x_visual_bb_torchvision",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((32,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": [
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_dla_dla34_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                "pt_googlenet_base_img_cls_torchvision",
                "pt_hrnet_hrnetv2_w30_pose_estimation_osmr",
                "pt_inception_v4_img_cls_osmr",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_regnet_regnet_x_32gf_img_cls_torchvision",
                "pt_regnet_regnet_y_400mf_img_cls_torchvision",
                "pt_xception_xception65_img_cls_timm",
                "pt_yolo_v6_yolov6n_obj_det_torchhub",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm",
                "pt_hrnet_hrnetv2_w32_pose_estimation_osmr",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf",
                "pt_regnet_regnet_x_3_2gf_img_cls_torchvision",
                "pt_regnet_regnet_y_800mf_img_cls_torchvision",
                "pt_xception_xception71_tf_in1k_img_cls_timm",
                "pt_yolo_v6_yolov6s_obj_det_torchhub",
                "pt_dla_dla169_visual_bb_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_hrnet_hrnetv2_w44_pose_estimation_osmr",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_regnet_facebook_regnet_y_064_img_cls_hf",
                "pt_regnet_regnet_x_800mf_img_cls_torchvision",
                "pt_xception_xception_img_cls_timm",
                "pt_yolov8_default_obj_det_github",
                "pt_dla_dla34_visual_bb_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_hrnet_hrnetv2_w48_pose_estimation_osmr",
                "pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_regnet_facebook_regnet_y_080_img_cls_hf",
                "pt_regnet_regnet_x_8gf_img_cls_torchvision",
                "pt_yolov9_default_obj_det_github",
                "pd_mobilenetv1_basic_img_cls_paddlemodels",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_regnet_facebook_regnet_y_320_img_cls_hf",
                "pt_regnet_regnet_y_1_6gf_img_cls_torchvision",
                "pt_unet_qubvel_img_seg_torchhub",
                "pt_yolo_v4_default_obj_det_github",
                "pt_dla_dla60x_visual_bb_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm",
                "pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_mobilenetv1_basic_img_cls_torchvision",
                "pt_regnet_regnet_x_16gf_img_cls_torchvision",
                "pt_regnet_regnet_y_32gf_img_cls_torchvision",
                "pt_unet_base_img_seg_torchhub",
                "pt_yolox_yolox_nano_obj_det_torchhub",
                "pt_dla_dla60x_c_visual_bb_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_regnet_regnet_x_1_6gf_img_cls_torchvision",
                "pt_regnet_regnet_y_3_2gf_img_cls_torchvision",
                "pt_xception_xception41_img_cls_timm",
                "pt_yolox_yolox_s_obj_det_torchhub",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
                "pt_dla_dla102x_visual_bb_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_hrnet_hrnetv2_w40_pose_estimation_osmr",
                "pt_inception_inception_v4_img_cls_timm",
                "pt_regnet_facebook_regnet_y_040_img_cls_hf",
                "pt_regnet_regnet_x_400mf_img_cls_torchvision",
                "pt_regnet_regnet_y_8gf_img_cls_torchvision",
                "pt_xception_xception71_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add33,
        [((32,), torch.float32)],
        {
            "model_names": [
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_dla_dla34_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                "pt_googlenet_base_img_cls_torchvision",
                "pt_hrnet_hrnetv2_w30_pose_estimation_osmr",
                "pt_inception_v4_img_cls_osmr",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_regnet_regnet_x_32gf_img_cls_torchvision",
                "pt_regnet_regnet_y_400mf_img_cls_torchvision",
                "pt_xception_xception65_img_cls_timm",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm",
                "pt_hrnet_hrnetv2_w32_pose_estimation_osmr",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf",
                "pt_regnet_regnet_x_3_2gf_img_cls_torchvision",
                "pt_regnet_regnet_y_800mf_img_cls_torchvision",
                "pt_xception_xception71_tf_in1k_img_cls_timm",
                "pt_dla_dla169_visual_bb_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_hrnet_hrnetv2_w44_pose_estimation_osmr",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_regnet_facebook_regnet_y_064_img_cls_hf",
                "pt_regnet_regnet_x_800mf_img_cls_torchvision",
                "pt_xception_xception_img_cls_timm",
                "pt_yolov8_default_obj_det_github",
                "pt_dla_dla34_visual_bb_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_hrnet_hrnetv2_w48_pose_estimation_osmr",
                "pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_regnet_facebook_regnet_y_080_img_cls_hf",
                "pt_regnet_regnet_x_8gf_img_cls_torchvision",
                "pt_yolov9_default_obj_det_github",
                "pd_mobilenetv1_basic_img_cls_paddlemodels",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_regnet_facebook_regnet_y_320_img_cls_hf",
                "pt_regnet_regnet_y_1_6gf_img_cls_torchvision",
                "pt_unet_qubvel_img_seg_torchhub",
                "pt_yolo_v4_default_obj_det_github",
                "pt_dla_dla60x_visual_bb_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm",
                "pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_mobilenetv1_basic_img_cls_torchvision",
                "pt_regnet_regnet_x_16gf_img_cls_torchvision",
                "pt_regnet_regnet_y_32gf_img_cls_torchvision",
                "pt_unet_base_img_seg_torchhub",
                "pt_yolox_yolox_nano_obj_det_torchhub",
                "pt_dla_dla60x_c_visual_bb_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_regnet_regnet_x_1_6gf_img_cls_torchvision",
                "pt_regnet_regnet_y_3_2gf_img_cls_torchvision",
                "pt_xception_xception41_img_cls_timm",
                "pt_yolox_yolox_s_obj_det_torchhub",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
                "pt_dla_dla102x_visual_bb_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_hrnet_hrnetv2_w40_pose_estimation_osmr",
                "pt_inception_inception_v4_img_cls_timm",
                "pt_regnet_facebook_regnet_y_040_img_cls_hf",
                "pt_regnet_regnet_x_400mf_img_cls_torchvision",
                "pt_regnet_regnet_y_8gf_img_cls_torchvision",
                "pt_xception_xception71_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 64, 56, 56), torch.float32), ((1, 64, 56, 56), torch.float32)],
        {
            "model_names": [
                "pt_dla_dla34_in1k_img_cls_timm",
                "pt_regnet_regnet_y_800mf_img_cls_torchvision",
                "pt_regnet_regnet_x_800mf_img_cls_torchvision",
                "onnx_dla_dla34_visual_bb_torchvision",
                "pt_dla_dla34_visual_bb_torchvision",
                "onnx_dla_dla46_c_visual_bb_torchvision",
                "pd_resnet_18_img_cls_paddlemodels",
                "onnx_dla_dla60x_c_visual_bb_torchvision",
                "onnx_efficientnet_efficientnet_b5_img_cls_timm",
                "pd_resnet_34_img_cls_paddlemodels",
                "pt_dla_dla60x_c_visual_bb_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 128, 28, 28), torch.float32), ((1, 128, 28, 28), torch.float32)],
        {
            "model_names": [
                "pt_dla_dla34_in1k_img_cls_timm",
                "pt_regnet_regnet_x_800mf_img_cls_torchvision",
                "onnx_dla_dla34_visual_bb_torchvision",
                "pt_dla_dla34_visual_bb_torchvision",
                "pd_resnet_18_img_cls_paddlemodels",
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
                "pt_dla_dla34_in1k_img_cls_timm",
                "onnx_dla_dla34_visual_bb_torchvision",
                "pt_dla_dla34_visual_bb_torchvision",
                "pd_resnet_18_img_cls_paddlemodels",
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
                "pt_dla_dla34_in1k_img_cls_timm",
                "onnx_dla_dla34_visual_bb_torchvision",
                "pt_dla_dla34_visual_bb_torchvision",
                "pd_resnet_18_img_cls_paddlemodels",
                "pd_resnet_34_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 32, 150, 150), torch.float32), ((32, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                "pt_xception_xception65_img_cls_timm",
                "pt_xception_xception71_tf_in1k_img_cls_timm",
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((24,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": [
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                "pt_googlenet_base_img_cls_torchvision",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_yolox_yolox_tiny_obj_det_torchhub",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "TranslatedLayer",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add34,
        [((24,), torch.float32)],
        {
            "model_names": [
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                "pt_googlenet_base_img_cls_torchvision",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_yolox_yolox_tiny_obj_det_torchhub",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "TranslatedLayer",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 24, 150, 150), torch.float32), ((24, 1, 1), torch.float32)],
        {"model_names": ["pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((144,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": [
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                "pt_googlenet_base_img_cls_torchvision",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf",
                "pt_regnet_regnet_y_800mf_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_regnet_facebook_regnet_y_064_img_cls_hf",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add35,
        [((144,), torch.float32)],
        {
            "model_names": [
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                "pt_googlenet_base_img_cls_torchvision",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf",
                "pt_regnet_regnet_y_800mf_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_regnet_facebook_regnet_y_064_img_cls_hf",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 144, 150, 150), torch.float32), ((144, 1, 1), torch.float32)],
        {"model_names": ["pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 144, 75, 75), torch.float32), ((144, 1, 1), torch.float32)],
        {"model_names": ["pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 32, 75, 75), torch.float32), ((32, 1, 1), torch.float32)],
        {"model_names": ["pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((192,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": [
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                "pt_googlenet_base_img_cls_torchvision",
                "pt_inception_v4_img_cls_osmr",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_yolox_yolox_tiny_obj_det_torchhub",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf",
                "pt_regnet_regnet_x_3_2gf_img_cls_torchvision",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_vovnet_ese_vovnet39b_obj_det_torchhub",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_hrnet_hrnetv2_w48_pose_estimation_osmr",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                "pt_yolox_yolox_m_obj_det_torchhub",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm",
                "pt_vovnet_vovnet39_img_cls_osmr",
                "pt_vovnet_vovnet_v1_57_obj_det_torchhub",
                "pd_densenet_121_img_cls_paddlemodels",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_vovnet_vovnet57_img_cls_osmr",
                "pt_yolo_v6_yolov6m_obj_det_torchhub",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_inception_inception_v4_img_cls_timm",
                "pt_regnet_facebook_regnet_y_040_img_cls_hf",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add36,
        [((192,), torch.float32)],
        {
            "model_names": [
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                "pt_googlenet_base_img_cls_torchvision",
                "pt_inception_v4_img_cls_osmr",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_yolox_yolox_tiny_obj_det_torchhub",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf",
                "pt_regnet_regnet_x_3_2gf_img_cls_torchvision",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_vovnet_ese_vovnet39b_obj_det_torchhub",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_hrnet_hrnetv2_w48_pose_estimation_osmr",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                "pt_yolox_yolox_m_obj_det_torchhub",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm",
                "pt_vovnet_vovnet39_img_cls_osmr",
                "pt_vovnet_vovnet_v1_57_obj_det_torchhub",
                "pd_densenet_121_img_cls_paddlemodels",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_vovnet_vovnet57_img_cls_osmr",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_inception_inception_v4_img_cls_timm",
                "pt_regnet_facebook_regnet_y_040_img_cls_hf",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 192, 75, 75), torch.float32), ((192, 1, 1), torch.float32)],
        {"model_names": ["pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 32, 75, 75), torch.float32), ((1, 32, 75, 75), torch.float32)],
        {"model_names": ["pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 192, 38, 38), torch.float32), ((192, 1, 1), torch.float32)],
        {"model_names": ["pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((48,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": [
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                "pt_googlenet_base_img_cls_torchvision",
                "pt_regnet_regnet_y_400mf_img_cls_torchvision",
                "pt_yolox_yolox_tiny_obj_det_torchhub",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_hrnet_hrnetv2_w48_pose_estimation_osmr",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pt_regnet_regnet_y_1_6gf_img_cls_torchvision",
                "pt_yolox_yolox_m_obj_det_torchhub",
                "pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_yolo_v6_yolov6m_obj_det_torchhub",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add37,
        [((48,), torch.float32)],
        {
            "model_names": [
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                "pt_googlenet_base_img_cls_torchvision",
                "pt_regnet_regnet_y_400mf_img_cls_torchvision",
                "pt_yolox_yolox_tiny_obj_det_torchhub",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_hrnet_hrnetv2_w48_pose_estimation_osmr",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pt_regnet_regnet_y_1_6gf_img_cls_torchvision",
                "pt_yolox_yolox_m_obj_det_torchhub",
                "pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 48, 38, 38), torch.float32), ((48, 1, 1), torch.float32)],
        {"model_names": ["pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((288,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": [
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                "pt_googlenet_base_img_cls_torchvision",
                "pt_regnet_facebook_regnet_y_064_img_cls_hf",
                "pt_regnet_regnet_x_800mf_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
                "pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add38,
        [((288,), torch.float32)],
        {
            "model_names": [
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                "pt_googlenet_base_img_cls_torchvision",
                "pt_regnet_facebook_regnet_y_064_img_cls_hf",
                "pt_regnet_regnet_x_800mf_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
                "pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 288, 38, 38), torch.float32), ((288, 1, 1), torch.float32)],
        {"model_names": ["pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 48, 38, 38), torch.float32), ((1, 48, 38, 38), torch.float32)],
        {"model_names": ["pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 288, 19, 19), torch.float32), ((288, 1, 1), torch.float32)],
        {"model_names": ["pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((96,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": [
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                "pt_googlenet_base_img_cls_torchvision",
                "pt_inception_v4_img_cls_osmr",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_yolox_yolox_tiny_obj_det_torchhub",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf",
                "pt_regnet_regnet_x_3_2gf_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_hrnet_hrnetv2_w48_pose_estimation_osmr",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                "pt_vovnet_vovnet27s_img_cls_osmr",
                "pt_yolox_yolox_m_obj_det_torchhub",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm",
                "pd_densenet_121_img_cls_paddlemodels",
                "pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_yolo_v6_yolov6m_obj_det_torchhub",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
                "pt_inception_inception_v4_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add39,
        [((96,), torch.float32)],
        {
            "model_names": [
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                "pt_googlenet_base_img_cls_torchvision",
                "pt_inception_v4_img_cls_osmr",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_yolox_yolox_tiny_obj_det_torchhub",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf",
                "pt_regnet_regnet_x_3_2gf_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_hrnet_hrnetv2_w48_pose_estimation_osmr",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                "pt_vovnet_vovnet27s_img_cls_osmr",
                "pt_yolox_yolox_m_obj_det_torchhub",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm",
                "pd_densenet_121_img_cls_paddlemodels",
                "pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
                "pt_inception_inception_v4_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 96, 19, 19), torch.float32), ((96, 1, 1), torch.float32)],
        {"model_names": ["pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((576,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": [
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_regnet_facebook_regnet_y_064_img_cls_hf",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_regnet_regnet_y_3_2gf_img_cls_torchvision",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add40,
        [((576,), torch.float32)],
        {
            "model_names": [
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_regnet_facebook_regnet_y_064_img_cls_hf",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_regnet_regnet_y_3_2gf_img_cls_torchvision",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 576, 19, 19), torch.float32), ((576, 1, 1), torch.float32)],
        {"model_names": ["pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 96, 19, 19), torch.float32), ((1, 96, 19, 19), torch.float32)],
        {"model_names": ["pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((136,), torch.float32), ((1,), torch.float32)],
        {"model_names": ["pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add41,
        [((136,), torch.float32)],
        {"model_names": ["pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 136, 19, 19), torch.float32), ((136, 1, 1), torch.float32)],
        {"model_names": ["pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((816,), torch.float32), ((1,), torch.float32)],
        {"model_names": ["pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add42,
        [((816,), torch.float32)],
        {"model_names": ["pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 816, 19, 19), torch.float32), ((816, 1, 1), torch.float32)],
        {"model_names": ["pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 136, 19, 19), torch.float32), ((1, 136, 19, 19), torch.float32)],
        {"model_names": ["pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 816, 10, 10), torch.float32), ((816, 1, 1), torch.float32)],
        {"model_names": ["pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((232,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": [
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                "pt_regnet_facebook_regnet_y_320_img_cls_hf",
                "pt_regnet_regnet_y_32gf_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add43,
        [((232,), torch.float32)],
        {
            "model_names": [
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                "pt_regnet_facebook_regnet_y_320_img_cls_hf",
                "pt_regnet_regnet_y_32gf_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 232, 10, 10), torch.float32), ((232, 1, 1), torch.float32)],
        {"model_names": ["pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1392,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": [
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                "pt_regnet_facebook_regnet_y_320_img_cls_hf",
                "pt_regnet_regnet_y_32gf_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add44,
        [((1392,), torch.float32)],
        {
            "model_names": [
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                "pt_regnet_facebook_regnet_y_320_img_cls_hf",
                "pt_regnet_regnet_y_32gf_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1392, 10, 10), torch.float32), ((1392, 1, 1), torch.float32)],
        {"model_names": ["pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 232, 10, 10), torch.float32), ((1, 232, 10, 10), torch.float32)],
        {"model_names": ["pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((384,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": [
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                "pt_googlenet_base_img_cls_torchvision",
                "pt_inception_v4_img_cls_osmr",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_yolox_yolox_tiny_obj_det_torchhub",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_hrnet_hrnetv2_w48_pose_estimation_osmr",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_vovnet_vovnet27s_img_cls_osmr",
                "pt_yolox_yolox_m_obj_det_torchhub",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_yolo_v6_yolov6m_obj_det_torchhub",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
                "pt_inception_inception_v4_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add5,
        [((384,), torch.float32)],
        {
            "model_names": [
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                "pt_googlenet_base_img_cls_torchvision",
                "pt_inception_v4_img_cls_osmr",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_yolox_yolox_tiny_obj_det_torchhub",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_hrnet_hrnetv2_w48_pose_estimation_osmr",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_vovnet_vovnet27s_img_cls_osmr",
                "pt_yolox_yolox_m_obj_det_torchhub",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
                "pt_inception_inception_v4_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 384, 10, 10), torch.float32), ((384, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_320x320",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1280,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": [
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf",
                "pt_yolox_yolox_x_obj_det_torchhub",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add45,
        [((1280,), torch.float32)],
        {
            "model_names": [
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf",
                "pt_yolox_yolox_x_obj_det_torchhub",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1280, 10, 10), torch.float32), ((1280, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_320x320",
            ],
            "pcc": 0.99,
        },
    ),
    (Add13, [((1, 207, 1), torch.float32)], {"model_names": ["pt_gemma_google_gemma_2_2b_it_qa_hf"], "pcc": 0.99}),
    (
        Add0,
        [((2304,), torch.float32), ((1,), torch.float32)],
        {"model_names": ["pt_gemma_google_gemma_2_2b_it_qa_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 8, 207, 256), torch.float32), ((1, 8, 207, 256), torch.float32)],
        {"model_names": ["pt_gemma_google_gemma_2_2b_it_qa_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 4, 207, 256), torch.float32), ((1, 4, 207, 256), torch.float32)],
        {"model_names": ["pt_gemma_google_gemma_2_2b_it_qa_hf"], "pcc": 0.99},
    ),
    (Add46, [((1, 8, 207, 207), torch.float32)], {"model_names": ["pt_gemma_google_gemma_2_2b_it_qa_hf"], "pcc": 0.99}),
    (
        Add0,
        [((1, 207, 2304), torch.float32), ((1, 207, 2304), torch.float32)],
        {"model_names": ["pt_gemma_google_gemma_2_2b_it_qa_hf"], "pcc": 0.99},
    ),
    (
        Add13,
        [((1, 1, 224, 224), torch.float32)],
        {"model_names": ["pt_googlenet_base_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 192, 56, 56), torch.float32), ((192, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_googlenet_base_img_cls_torchvision",
                "pt_regnet_regnet_x_3_2gf_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "onnx_mobilenetv2_mobilenetv2_140_img_cls_timm",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
                "pt_regnet_facebook_regnet_y_040_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 176, 28, 28), torch.float32), ((176, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_googlenet_base_img_cls_torchvision",
                "pt_hrnet_hrnetv2_w44_pose_estimation_osmr",
                "onnx_efficientnet_efficientnet_b5_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 32, 28, 28), torch.float32), ((32, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_googlenet_base_img_cls_torchvision",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_hrnet_hrnetv2_w32_pose_estimation_osmr",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "onnx_dla_dla46_c_visual_bb_torchvision",
                "onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 288, 28, 28), torch.float32), ((288, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_googlenet_base_img_cls_torchvision",
                "pt_regnet_facebook_regnet_y_064_img_cls_hf",
                "pt_regnet_regnet_x_800mf_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "onnx_mobilenetv2_mobilenetv2_140_img_cls_timm",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 192, 28, 28), torch.float32), ((192, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_googlenet_base_img_cls_torchvision",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_regnet_regnet_x_3_2gf_img_cls_torchvision",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_hrnet_hrnetv2_w48_pose_estimation_osmr",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels",
                "onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm",
                "pt_densenet_densenet169_img_cls_torchvision",
                "onnx_mobilenetv2_mobilenetv2_140_img_cls_timm",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
                "pt_regnet_facebook_regnet_y_040_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 96, 28, 28), torch.float32), ((96, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_googlenet_base_img_cls_torchvision",
                "pt_hrnet_hrnetv2_w48_pose_estimation_osmr",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "onnx_mobilenetv2_mobilenetv2_050_img_cls_timm",
                "pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 64, 28, 28), torch.float32), ((64, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_googlenet_base_img_cls_torchvision",
                "pt_hrnet_hrnetv2_w30_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w32_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w44_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w48_pose_estimation_osmr",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "onnx_dla_dla46_c_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "onnx_dla_dla60x_c_visual_bb_torchvision",
                "pt_dla_dla60x_c_visual_bb_torchvision",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w40_pose_estimation_osmr",
                "pt_regnet_regnet_x_400mf_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 304, 14, 14), torch.float32), ((304, 1, 1), torch.float32)],
        {
            "model_names": ["pt_googlenet_base_img_cls_torchvision", "onnx_efficientnet_efficientnet_b5_img_cls_timm"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((208,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": [
                "pt_googlenet_base_img_cls_torchvision",
                "pt_regnet_regnet_y_400mf_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add47,
        [((208,), torch.float32)],
        {
            "model_names": [
                "pt_googlenet_base_img_cls_torchvision",
                "pt_regnet_regnet_y_400mf_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 208, 14, 14), torch.float32), ((208, 1, 1), torch.float32)],
        {
            "model_names": ["pt_googlenet_base_img_cls_torchvision", "pt_regnet_regnet_y_400mf_img_cls_torchvision"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 48, 14, 14), torch.float32), ((48, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_googlenet_base_img_cls_torchvision",
                "pt_hrnet_hrnetv2_w48_pose_estimation_osmr",
                "onnx_mobilenetv2_mobilenetv2_050_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 64, 14, 14), torch.float32), ((64, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_googlenet_base_img_cls_torchvision",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_hrnet_hrnetv2_w32_pose_estimation_osmr",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "onnx_dla_dla46_c_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((160,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": [
                "pt_googlenet_base_img_cls_torchvision",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_yolox_yolox_x_obj_det_torchhub",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_vovnet_ese_vovnet39b_obj_det_torchhub",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_vovnet_vovnet39_img_cls_osmr",
                "pt_vovnet_vovnet_v1_57_obj_det_torchhub",
                "pd_densenet_121_img_cls_paddlemodels",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_vovnet_vovnet57_img_cls_osmr",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_hrnet_hrnetv2_w40_pose_estimation_osmr",
                "pt_regnet_regnet_x_400mf_img_cls_torchvision",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((112,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": [
                "pt_googlenet_base_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                "pt_vovnet_vovnet27s_img_cls_osmr",
                "pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add48,
        [((160,), torch.float32)],
        {
            "model_names": [
                "pt_googlenet_base_img_cls_torchvision",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_yolox_yolox_x_obj_det_torchhub",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_vovnet_ese_vovnet39b_obj_det_torchhub",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_vovnet_vovnet39_img_cls_osmr",
                "pt_vovnet_vovnet_v1_57_obj_det_torchhub",
                "pd_densenet_121_img_cls_paddlemodels",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_vovnet_vovnet57_img_cls_osmr",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_hrnet_hrnetv2_w40_pose_estimation_osmr",
                "pt_regnet_regnet_x_400mf_img_cls_torchvision",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add49,
        [((112,), torch.float32)],
        {
            "model_names": [
                "pt_googlenet_base_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                "pt_vovnet_vovnet27s_img_cls_osmr",
                "pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 296, 14, 14), torch.float32), ((296, 1, 1), torch.float32)],
        {"model_names": ["pt_googlenet_base_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((224,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": [
                "pt_googlenet_base_img_cls_torchvision",
                "pt_inception_v4_img_cls_osmr",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_vovnet_ese_vovnet39b_obj_det_torchhub",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_vovnet_vovnet39_img_cls_osmr",
                "pt_vovnet_vovnet_v1_57_obj_det_torchhub",
                "pd_densenet_121_img_cls_paddlemodels",
                "pt_vovnet_vovnet57_img_cls_osmr",
                "pt_inception_inception_v4_img_cls_timm",
                "pt_regnet_regnet_y_8gf_img_cls_torchvision",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add50,
        [((224,), torch.float32)],
        {
            "model_names": [
                "pt_googlenet_base_img_cls_torchvision",
                "pt_inception_v4_img_cls_osmr",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_vovnet_ese_vovnet39b_obj_det_torchhub",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_vovnet_vovnet39_img_cls_osmr",
                "pt_vovnet_vovnet_v1_57_obj_det_torchhub",
                "pd_densenet_121_img_cls_paddlemodels",
                "pt_vovnet_vovnet57_img_cls_osmr",
                "pt_inception_inception_v4_img_cls_timm",
                "pt_regnet_regnet_y_8gf_img_cls_torchvision",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 224, 14, 14), torch.float32), ((224, 1, 1), torch.float32)],
        {"model_names": ["pt_googlenet_base_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 280, 14, 14), torch.float32), ((280, 1, 1), torch.float32)],
        {"model_names": ["pt_googlenet_base_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 288, 14, 14), torch.float32), ((288, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_googlenet_base_img_cls_torchvision",
                "pt_regnet_regnet_x_800mf_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "onnx_mobilenetv2_mobilenetv2_050_img_cls_timm",
                "pt_densenet_densenet169_img_cls_torchvision",
                "onnx_mobilenetv2_mobilenetv2_140_img_cls_timm",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 448, 14, 14), torch.float32), ((448, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_googlenet_base_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((320,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": [
                "pt_googlenet_base_img_cls_torchvision",
                "pt_inception_v4_img_cls_osmr",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_regnet_regnet_y_800mf_img_cls_torchvision",
                "pt_yolox_yolox_x_obj_det_torchhub",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm",
                "pd_densenet_121_img_cls_paddlemodels",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
                "pt_hrnet_hrnetv2_w40_pose_estimation_osmr",
                "pt_inception_inception_v4_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add51,
        [((320,), torch.float32)],
        {
            "model_names": [
                "pt_googlenet_base_img_cls_torchvision",
                "pt_inception_v4_img_cls_osmr",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_regnet_regnet_y_800mf_img_cls_torchvision",
                "pt_yolox_yolox_x_obj_det_torchhub",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm",
                "pd_densenet_121_img_cls_paddlemodels",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
                "pt_hrnet_hrnetv2_w40_pose_estimation_osmr",
                "pt_inception_inception_v4_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 320, 14, 14), torch.float32), ((320, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_googlenet_base_img_cls_torchvision",
                "pt_regnet_regnet_y_800mf_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 128, 14, 14), torch.float32), ((128, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_googlenet_base_img_cls_torchvision",
                "pt_hrnet_hrnetv2_w30_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w32_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w44_pose_estimation_osmr",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_hrnet_hrnetv2_w48_pose_estimation_osmr",
                "onnx_dla_dla46_c_visual_bb_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "onnx_dla_dla60x_c_visual_bb_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
                "pt_dla_dla60x_c_visual_bb_torchvision",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w40_pose_estimation_osmr",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 448, 7, 7), torch.float32), ((448, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_googlenet_base_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "onnx_mobilenetv2_mobilenetv2_140_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 128, 7, 7), torch.float32), ((128, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_googlenet_base_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "onnx_dla_dla46_c_visual_bb_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 624, 7, 7), torch.float32), ((624, 1, 1), torch.float32)],
        {
            "model_names": ["pt_googlenet_base_img_cls_torchvision", "onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 384, 7, 7), torch.float32), ((384, 1, 1), torch.float32)],
        {
            "model_names": ["pt_googlenet_base_img_cls_torchvision", "pt_hrnet_hrnetv2_w48_pose_estimation_osmr"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((30,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w30_pose_estimation_osmr", "pt_hrnet_hrnet_w30_pose_estimation_timm"],
            "pcc": 0.99,
        },
    ),
    (
        Add52,
        [((30,), torch.float32)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w30_pose_estimation_osmr", "pt_hrnet_hrnet_w30_pose_estimation_timm"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 30, 56, 56), torch.float32), ((30, 1, 1), torch.float32)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w30_pose_estimation_osmr", "pt_hrnet_hrnet_w30_pose_estimation_timm"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 30, 56, 56), torch.float32), ((1, 30, 56, 56), torch.float32)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w30_pose_estimation_osmr", "pt_hrnet_hrnet_w30_pose_estimation_timm"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((60,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": [
                "pt_hrnet_hrnetv2_w30_pose_estimation_osmr",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pd_paddleocr_v4_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add53,
        [((60,), torch.float32)],
        {
            "model_names": [
                "pt_hrnet_hrnetv2_w30_pose_estimation_osmr",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pd_paddleocr_v4_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 60, 28, 28), torch.float32), ((60, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_hrnet_hrnetv2_w30_pose_estimation_osmr",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 60, 28, 28), torch.float32), ((1, 60, 28, 28), torch.float32)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w30_pose_estimation_osmr", "pt_hrnet_hrnet_w30_pose_estimation_timm"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 30, 28, 28), torch.float32), ((30, 1, 1), torch.float32)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w30_pose_estimation_osmr", "pt_hrnet_hrnet_w30_pose_estimation_timm"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((120,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": [
                "pt_hrnet_hrnetv2_w30_pose_estimation_osmr",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pt_regnet_regnet_y_1_6gf_img_cls_torchvision",
                "pd_paddleocr_v4_rec_ch_scene_text_recognition_paddlemodels",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
                "pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add54,
        [((120,), torch.float32)],
        {
            "model_names": [
                "pt_hrnet_hrnetv2_w30_pose_estimation_osmr",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pt_regnet_regnet_y_1_6gf_img_cls_torchvision",
                "pd_paddleocr_v4_rec_ch_scene_text_recognition_paddlemodels",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
                "pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 120, 14, 14), torch.float32), ((120, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_hrnet_hrnetv2_w30_pose_estimation_osmr",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 120, 14, 14), torch.float32), ((1, 120, 14, 14), torch.float32)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w30_pose_estimation_osmr", "pt_hrnet_hrnet_w30_pose_estimation_timm"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 30, 14, 14), torch.float32), ((30, 1, 1), torch.float32)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w30_pose_estimation_osmr", "pt_hrnet_hrnet_w30_pose_estimation_timm"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 60, 14, 14), torch.float32), ((60, 1, 1), torch.float32)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w30_pose_estimation_osmr", "pt_hrnet_hrnet_w30_pose_estimation_timm"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((240,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": [
                "pt_hrnet_hrnetv2_w30_pose_estimation_osmr",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_regnet_regnet_x_8gf_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add55,
        [((240,), torch.float32)],
        {
            "model_names": [
                "pt_hrnet_hrnetv2_w30_pose_estimation_osmr",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_regnet_regnet_x_8gf_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 240, 7, 7), torch.float32), ((240, 1, 1), torch.float32)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w30_pose_estimation_osmr", "pt_hrnet_hrnet_w30_pose_estimation_timm"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 240, 7, 7), torch.float32), ((1, 240, 7, 7), torch.float32)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w30_pose_estimation_osmr", "pt_hrnet_hrnet_w30_pose_estimation_timm"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 210, 7, 7), torch.float32), ((210, 1, 1), torch.float32)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w30_pose_estimation_osmr", "pt_hrnet_hrnet_w30_pose_estimation_timm"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 120, 28, 28), torch.float32), ((120, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_hrnet_hrnetv2_w30_pose_estimation_osmr",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_regnet_regnet_y_1_6gf_img_cls_torchvision",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 256, 7, 7), torch.float32), ((256, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_hrnet_hrnetv2_w30_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w32_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w44_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w48_pose_estimation_osmr",
                "onnx_dla_dla46_c_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "onnx_dla_dla60x_c_visual_bb_torchvision",
                "pt_dla_dla60x_c_visual_bb_torchvision",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w40_pose_estimation_osmr",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 32, 56, 56), torch.float32), ((32, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_hrnet_hrnetv2_w30_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w32_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w44_pose_estimation_osmr",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_hrnet_hrnetv2_w48_pose_estimation_osmr",
                "onnx_dla_dla46_c_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "onnx_mobilenetv2_mobilenetv2_140_img_cls_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w40_pose_estimation_osmr",
                "pt_regnet_regnet_x_400mf_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 32, 149, 149), torch.float32), ((32, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_inception_v4_img_cls_osmr",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_xception_xception_img_cls_timm",
                "pt_inception_inception_v4_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 32, 147, 147), torch.float32), ((32, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_inception_v4_img_cls_osmr",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_inception_inception_v4_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 64, 147, 147), torch.float32), ((64, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_inception_v4_img_cls_osmr",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_xception_xception_img_cls_timm",
                "pt_inception_inception_v4_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 96, 73, 73), torch.float32), ((96, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_inception_v4_img_cls_osmr",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_inception_inception_v4_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 64, 73, 73), torch.float32), ((64, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_inception_v4_img_cls_osmr",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_inception_inception_v4_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 96, 71, 71), torch.float32), ((96, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_inception_v4_img_cls_osmr",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_inception_inception_v4_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 192, 35, 35), torch.float32), ((192, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_inception_v4_img_cls_osmr",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_inception_inception_v4_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 224, 35, 35), torch.float32), ((224, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_inception_v4_img_cls_osmr",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_inception_inception_v4_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 96, 35, 35), torch.float32), ((96, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_inception_v4_img_cls_osmr",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_inception_inception_v4_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 384, 17, 17), torch.float32), ((384, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_inception_v4_img_cls_osmr",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_inception_inception_v4_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 256, 17, 17), torch.float32), ((256, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_inception_v4_img_cls_osmr",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_inception_inception_v4_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 768, 17, 17), torch.float32), ((768, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_inception_v4_img_cls_osmr",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_inception_inception_v4_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 224, 17, 17), torch.float32), ((224, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_inception_v4_img_cls_osmr",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_inception_inception_v4_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 192, 17, 17), torch.float32), ((192, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_inception_v4_img_cls_osmr",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_inception_inception_v4_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 128, 17, 17), torch.float32), ((128, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_inception_v4_img_cls_osmr",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_inception_inception_v4_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 192, 8, 8), torch.float32), ((192, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_inception_v4_img_cls_osmr",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm",
                "pt_inception_inception_v4_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 320, 17, 17), torch.float32), ((320, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_inception_v4_img_cls_osmr",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_inception_inception_v4_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 320, 8, 8), torch.float32), ((320, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_inception_v4_img_cls_osmr",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm",
                "pt_inception_inception_v4_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1024, 8, 8), torch.float32), ((1024, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_inception_v4_img_cls_osmr",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_inception_inception_v4_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 256, 8, 8), torch.float32), ((256, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_inception_v4_img_cls_osmr",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_fpn_base_img_cls_torchvision",
                "pt_inception_inception_v4_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((448,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": [
                "pt_inception_v4_img_cls_osmr",
                "pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_regnet_facebook_regnet_y_080_img_cls_hf",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_inception_inception_v4_img_cls_timm",
                "pt_regnet_regnet_y_8gf_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add56,
        [((448,), torch.float32)],
        {
            "model_names": [
                "pt_inception_v4_img_cls_osmr",
                "pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_regnet_facebook_regnet_y_080_img_cls_hf",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_inception_inception_v4_img_cls_timm",
                "pt_regnet_regnet_y_8gf_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 448, 8, 8), torch.float32), ((448, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_inception_v4_img_cls_osmr",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_inception_inception_v4_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 512, 8, 8), torch.float32), ((512, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_inception_v4_img_cls_osmr",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_inception_inception_v4_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add20,
        [((1, 1024, 512), torch.float32)],
        {
            "model_names": [
                "pt_mlp_mixer_mixer_l32_224_img_cls_timm",
                "pt_mlp_mixer_mixer_l16_224_img_cls_timm",
                "pt_mlp_mixer_mixer_l16_224_in21k_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add57,
        [((1, 1024, 49), torch.float32)],
        {"model_names": ["pt_mlp_mixer_mixer_l32_224_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 49, 1024), torch.float32), ((1, 49, 1024), torch.float32)],
        {"model_names": ["pt_mlp_mixer_mixer_l32_224_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add24,
        [((1, 49, 4096), torch.float32)],
        {"model_names": ["pt_mlp_mixer_mixer_l32_224_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add21,
        [((1, 49, 1024), torch.float32)],
        {"model_names": ["pt_mlp_mixer_mixer_l32_224_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 32, 28, 28), torch.float32), ((1, 32, 28, 28), torch.float32)],
        {
            "model_names": [
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
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
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_vovnet_vovnet27s_img_cls_osmr",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
                "pt_deit_facebook_deit_small_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 64, 14, 14), torch.float32), ((1, 64, 14, 14), torch.float32)],
        {
            "model_names": [
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 96, 14, 14), torch.float32), ((96, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_hrnet_hrnetv2_w48_pose_estimation_osmr",
                "onnx_mobilenetv2_mobilenetv2_050_img_cls_timm",
                "pt_vovnet_vovnet27s_img_cls_osmr",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 576, 14, 14), torch.float32), ((576, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_regnet_facebook_regnet_y_064_img_cls_hf",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_regnet_regnet_y_3_2gf_img_cls_torchvision",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 96, 14, 14), torch.float32), ((1, 96, 14, 14), torch.float32)],
        {
            "model_names": [
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
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
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 160, 7, 7), torch.float32), ((160, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "onnx_mobilenetv2_mobilenetv2_050_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((960,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": [
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pd_densenet_121_img_cls_paddlemodels",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add58,
        [((960,), torch.float32)],
        {
            "model_names": [
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pd_densenet_121_img_cls_paddlemodels",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 960, 7, 7), torch.float32), ((960, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pd_densenet_121_img_cls_paddlemodels",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 160, 7, 7), torch.float32), ((1, 160, 7, 7), torch.float32)],
        {
            "model_names": [
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add59,
        [((1, 1001), torch.float32)],
        {
            "model_names": [
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 64, 160, 512), torch.float32), ((64, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_1024x320_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 64, 80, 256), torch.float32), ((64, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_1024x320_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 64, 80, 256), torch.float32), ((1, 64, 80, 256), torch.float32)],
        {
            "model_names": [
                "pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_1024x320_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 128, 40, 128), torch.float32), ((128, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_1024x320_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 128, 40, 128), torch.float32), ((1, 128, 40, 128), torch.float32)],
        {
            "model_names": [
                "pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_1024x320_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 256, 20, 64), torch.float32), ((256, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_1024x320_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 256, 20, 64), torch.float32), ((1, 256, 20, 64), torch.float32)],
        {
            "model_names": [
                "pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_1024x320_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 512, 10, 32), torch.float32), ((512, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_1024x320_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 512, 10, 32), torch.float32), ((1, 512, 10, 32), torch.float32)],
        {
            "model_names": [
                "pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_1024x320_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 256, 10, 32), torch.float32), ((256, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_1024x320_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 128, 20, 64), torch.float32), ((128, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_1024x320_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 64, 40, 128), torch.float32), ((64, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_1024x320_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 32, 80, 256), torch.float32), ((32, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_1024x320_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 32, 160, 512), torch.float32), ((32, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_1024x320_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 16, 160, 512), torch.float32), ((16, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_1024x320_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 16, 320, 1024), torch.float32), ((16, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_1024x320_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1, 320, 1024), torch.float32), ((1, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_1024x320_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 7, 768), torch.float32), ((1, 7, 768), torch.float32)],
        {
            "model_names": [
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
                "pt_gpt_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((7, 768), torch.float32), ((768,), torch.float32)],
        {
            "model_names": [
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
                "pt_gpt_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 12, 7, 7), torch.float32), ((1, 1, 7, 7), torch.float32)],
        {
            "model_names": [
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
                "pt_gpt_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 12, 7, 7), torch.float32), ((1, 1, 1, 7), torch.float32)],
        {
            "model_names": [
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
                "pt_gpt_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add14,
        [((7, 768), torch.float32)],
        {
            "model_names": [
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
                "pt_gpt_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add15,
        [((7, 3072), torch.float32)],
        {
            "model_names": [
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
                "pt_gpt_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add60,
        [((1, 32), torch.int64)],
        {
            "model_names": [
                "pt_opt_facebook_opt_350m_qa_hf",
                "pt_opt_facebook_opt_1_3b_seq_cls_hf",
                "pt_opt_facebook_opt_350m_seq_cls_hf",
                "pt_opt_facebook_opt_1_3b_qa_hf",
                "pt_opt_facebook_opt_125m_qa_hf",
                "pt_opt_facebook_opt_125m_seq_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 32, 1024), torch.float32), ((1, 32, 1024), torch.float32)],
        {"model_names": ["pt_opt_facebook_opt_350m_qa_hf", "pt_opt_facebook_opt_350m_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Add21,
        [((1, 32, 1024), torch.float32)],
        {"model_names": ["pt_opt_facebook_opt_350m_qa_hf", "pt_opt_facebook_opt_350m_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1, 32, 32), torch.float32), ((1, 1, 32, 32), torch.float32)],
        {
            "model_names": [
                "pt_opt_facebook_opt_350m_qa_hf",
                "pt_opt_facebook_opt_1_3b_seq_cls_hf",
                "pt_opt_facebook_opt_350m_seq_cls_hf",
                "pt_bloom_bigscience_bloom_1b1_clm_hf",
                "pt_opt_facebook_opt_1_3b_qa_hf",
                "pt_opt_facebook_opt_125m_qa_hf",
                "pt_opt_facebook_opt_125m_seq_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 16, 32, 32), torch.float32), ((1, 1, 32, 32), torch.float32)],
        {
            "model_names": [
                "pt_opt_facebook_opt_350m_qa_hf",
                "pt_opt_facebook_opt_350m_seq_cls_hf",
                "pt_bloom_bigscience_bloom_1b1_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add24,
        [((32, 4096), torch.float32)],
        {"model_names": ["pt_opt_facebook_opt_350m_qa_hf", "pt_opt_facebook_opt_350m_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Add21,
        [((32, 1024), torch.float32)],
        {"model_names": ["pt_opt_facebook_opt_350m_qa_hf", "pt_opt_facebook_opt_350m_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((32, 1024), torch.float32), ((32, 1024), torch.float32)],
        {"model_names": ["pt_opt_facebook_opt_350m_qa_hf", "pt_opt_facebook_opt_350m_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((32, 1), torch.float32), ((1,), torch.float32)],
        {
            "model_names": [
                "pt_opt_facebook_opt_350m_qa_hf",
                "pt_opt_facebook_opt_1_3b_qa_hf",
                "pt_opt_facebook_opt_125m_qa_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add21,
        [((1, 1, 1024), torch.float32)],
        {
            "model_names": [
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "pt_whisper_openai_whisper_medium_speech_recognition_hf",
                "pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add61,
        [((1, 512, 322), torch.float32)],
        {"model_names": ["pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add61,
        [((1, 3025, 322), torch.float32)],
        {"model_names": ["pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add62,
        [((1, 1, 512, 3025), torch.float32)],
        {"model_names": ["pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add21,
        [((1, 512, 1024), torch.float32)],
        {
            "model_names": [
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
            "model_names": [
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1, 1024), torch.float32), ((1, 1, 1024), torch.float32)],
        {
            "model_names": [
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "pt_whisper_openai_whisper_medium_speech_recognition_hf",
                "pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf",
                "pt_t5_google_flan_t5_large_text_gen_hf",
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
                "pt_t5_t5_large_text_gen_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add1,
        [((1, 1, 1000), torch.float32)],
        {
            "model_names": [
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add22,
        [((1, 12, 2048), torch.float32)],
        {
            "model_names": ["pt_phi1_microsoft_phi_1_token_cls_hf", "pt_phi_1_5_microsoft_phi_1_5_token_cls_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 32, 12, 32), torch.float32), ((1, 32, 12, 32), torch.float32)],
        {
            "model_names": ["pt_phi1_microsoft_phi_1_token_cls_hf", "pt_phi_1_5_microsoft_phi_1_5_token_cls_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Add63,
        [((1, 32, 12, 12), torch.float32)],
        {
            "model_names": ["pt_phi1_microsoft_phi_1_token_cls_hf", "pt_phi_1_5_microsoft_phi_1_5_token_cls_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Add64,
        [((1, 12, 8192), torch.float32)],
        {
            "model_names": ["pt_phi1_microsoft_phi_1_token_cls_hf", "pt_phi_1_5_microsoft_phi_1_5_token_cls_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 12, 2048), torch.float32), ((1, 12, 2048), torch.float32)],
        {
            "model_names": ["pt_phi1_microsoft_phi_1_token_cls_hf", "pt_phi_1_5_microsoft_phi_1_5_token_cls_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Add16,
        [((1, 12, 2), torch.float32)],
        {
            "model_names": ["pt_phi1_microsoft_phi_1_token_cls_hf", "pt_phi_1_5_microsoft_phi_1_5_token_cls_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Add13,
        [((1, 35, 1), torch.float32)],
        {
            "model_names": [
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add65,
        [((1, 35, 1536), torch.float32)],
        {"model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 12, 35, 128), torch.float32), ((1, 12, 35, 128), torch.float32)],
        {"model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Add18,
        [((1, 35, 256), torch.float32)],
        {"model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 2, 35, 128), torch.float32), ((1, 2, 35, 128), torch.float32)],
        {"model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Add66,
        [((1, 12, 35, 35), torch.float32)],
        {"model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 35, 1536), torch.float32), ((1, 35, 1536), torch.float32)],
        {"model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((336,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": [
                "pt_regnet_regnet_x_32gf_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_regnet_regnet_y_1_6gf_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add67,
        [((336,), torch.float32)],
        {
            "model_names": [
                "pt_regnet_regnet_x_32gf_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_regnet_regnet_y_1_6gf_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 336, 56, 56), torch.float32), ((336, 1, 1), torch.float32)],
        {"model_names": ["pt_regnet_regnet_x_32gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 336, 112, 112), torch.float32), ((336, 1, 1), torch.float32)],
        {"model_names": ["pt_regnet_regnet_x_32gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 336, 56, 56), torch.float32), ((1, 336, 56, 56), torch.float32)],
        {"model_names": ["pt_regnet_regnet_x_32gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((672,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": [
                "pt_regnet_regnet_x_32gf_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_regnet_regnet_x_800mf_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pd_densenet_121_img_cls_paddlemodels",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add68,
        [((672,), torch.float32)],
        {
            "model_names": [
                "pt_regnet_regnet_x_32gf_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_regnet_regnet_x_800mf_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pd_densenet_121_img_cls_paddlemodels",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 672, 28, 28), torch.float32), ((672, 1, 1), torch.float32)],
        {"model_names": ["pt_regnet_regnet_x_32gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 672, 56, 56), torch.float32), ((672, 1, 1), torch.float32)],
        {"model_names": ["pt_regnet_regnet_x_32gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 672, 28, 28), torch.float32), ((1, 672, 28, 28), torch.float32)],
        {"model_names": ["pt_regnet_regnet_x_32gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1344,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": [
                "pt_regnet_regnet_x_32gf_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add69,
        [((1344,), torch.float32)],
        {
            "model_names": [
                "pt_regnet_regnet_x_32gf_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1344, 14, 14), torch.float32), ((1344, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_regnet_regnet_x_32gf_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1344, 28, 28), torch.float32), ((1344, 1, 1), torch.float32)],
        {"model_names": ["pt_regnet_regnet_x_32gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1344, 14, 14), torch.float32), ((1, 1344, 14, 14), torch.float32)],
        {"model_names": ["pt_regnet_regnet_x_32gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((2520,), torch.float32), ((1,), torch.float32)],
        {"model_names": ["pt_regnet_regnet_x_32gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (Add70, [((2520,), torch.float32)], {"model_names": ["pt_regnet_regnet_x_32gf_img_cls_torchvision"], "pcc": 0.99}),
    (
        Add0,
        [((1, 2520, 7, 7), torch.float32), ((2520, 1, 1), torch.float32)],
        {"model_names": ["pt_regnet_regnet_x_32gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 2520, 14, 14), torch.float32), ((2520, 1, 1), torch.float32)],
        {"model_names": ["pt_regnet_regnet_x_32gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 2520, 7, 7), torch.float32), ((1, 2520, 7, 7), torch.float32)],
        {"model_names": ["pt_regnet_regnet_x_32gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 48, 56, 56), torch.float32), ((48, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_regnet_regnet_y_400mf_img_cls_torchvision",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_hrnet_hrnetv2_w48_pose_estimation_osmr",
                "onnx_mobilenetv2_mobilenetv2_050_img_cls_timm",
                "pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels",
                "pt_regnet_regnet_y_1_6gf_img_cls_torchvision",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 48, 112, 112), torch.float32), ((48, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_regnet_regnet_y_400mf_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "onnx_mobilenetv2_mobilenetv2_050_img_cls_timm",
                "pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels",
                "pt_regnet_regnet_y_1_6gf_img_cls_torchvision",
                "onnx_mobilenetv2_mobilenetv2_140_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 8, 1, 1), torch.float32), ((8, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_regnet_regnet_y_400mf_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_regnet_regnet_y_800mf_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_regnet_facebook_regnet_y_064_img_cls_hf",
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_regnet_facebook_regnet_y_080_img_cls_hf",
                "onnx_efficientnet_efficientnet_b2a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pt_regnet_facebook_regnet_y_320_img_cls_hf",
                "pt_regnet_regnet_y_1_6gf_img_cls_torchvision",
                "onnx_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_regnet_regnet_y_32gf_img_cls_torchvision",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_regnet_regnet_y_3_2gf_img_cls_torchvision",
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_regnet_facebook_regnet_y_040_img_cls_hf",
                "pt_regnet_regnet_y_8gf_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 48, 1, 1), torch.float32), ((48, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_regnet_regnet_y_400mf_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "TranslatedLayer",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pt_regnet_regnet_y_1_6gf_img_cls_torchvision",
                "onnx_efficientnet_efficientnet_b4_img_cls_timm",
                "onnx_efficientnet_efficientnet_b5_img_cls_timm",
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_regnet_facebook_regnet_y_040_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 48, 56, 56), torch.float32), ((1, 48, 56, 56), torch.float32)],
        {
            "model_names": [
                "pt_regnet_regnet_y_400mf_img_cls_torchvision",
                "pt_hrnet_hrnetv2_w48_pose_estimation_osmr",
                "pt_regnet_regnet_y_1_6gf_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((104,), torch.float32), ((1,), torch.float32)],
        {"model_names": ["pt_regnet_regnet_y_400mf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (Add71, [((104,), torch.float32)], {"model_names": ["pt_regnet_regnet_y_400mf_img_cls_torchvision"], "pcc": 0.99}),
    (
        Add0,
        [((1, 104, 28, 28), torch.float32), ((104, 1, 1), torch.float32)],
        {"model_names": ["pt_regnet_regnet_y_400mf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 104, 56, 56), torch.float32), ((104, 1, 1), torch.float32)],
        {"model_names": ["pt_regnet_regnet_y_400mf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 12, 1, 1), torch.float32), ((12, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_regnet_regnet_y_400mf_img_cls_torchvision",
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "onnx_efficientnet_efficientnet_b2a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pt_regnet_regnet_y_1_6gf_img_cls_torchvision",
                "onnx_efficientnet_efficientnet_b4_img_cls_timm",
                "onnx_efficientnet_efficientnet_b5_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 104, 1, 1), torch.float32), ((104, 1, 1), torch.float32)],
        {"model_names": ["pt_regnet_regnet_y_400mf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 104, 28, 28), torch.float32), ((1, 104, 28, 28), torch.float32)],
        {"model_names": ["pt_regnet_regnet_y_400mf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 26, 1, 1), torch.float32), ((26, 1, 1), torch.float32)],
        {"model_names": ["pt_regnet_regnet_y_400mf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 208, 28, 28), torch.float32), ((208, 1, 1), torch.float32)],
        {"model_names": ["pt_regnet_regnet_y_400mf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 208, 1, 1), torch.float32), ((208, 1, 1), torch.float32)],
        {"model_names": ["pt_regnet_regnet_y_400mf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 208, 14, 14), torch.float32), ((1, 208, 14, 14), torch.float32)],
        {"model_names": ["pt_regnet_regnet_y_400mf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 52, 1, 1), torch.float32), ((52, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_regnet_regnet_y_400mf_img_cls_torchvision",
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((440,), torch.float32), ((1,), torch.float32)],
        {"model_names": ["pt_regnet_regnet_y_400mf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (Add72, [((440,), torch.float32)], {"model_names": ["pt_regnet_regnet_y_400mf_img_cls_torchvision"], "pcc": 0.99}),
    (
        Add0,
        [((1, 440, 7, 7), torch.float32), ((440, 1, 1), torch.float32)],
        {"model_names": ["pt_regnet_regnet_y_400mf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 440, 14, 14), torch.float32), ((440, 1, 1), torch.float32)],
        {"model_names": ["pt_regnet_regnet_y_400mf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 440, 1, 1), torch.float32), ((440, 1, 1), torch.float32)],
        {"model_names": ["pt_regnet_regnet_y_400mf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 440, 7, 7), torch.float32), ((1, 440, 7, 7), torch.float32)],
        {"model_names": ["pt_regnet_regnet_y_400mf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 110, 1, 1), torch.float32), ((110, 1, 1), torch.float32)],
        {"model_names": ["pt_regnet_regnet_y_400mf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 512, 56, 56), torch.float32), ((512, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_resnext_resnext101_32x8d_img_cls_torchhub",
                "onnx_dla_dla102x2_visual_bb_torchvision",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_regnet_regnet_x_16gf_img_cls_torchvision",
                "pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub",
                "pt_resnext_resnext101_64x4d_img_cls_osmr",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1024, 28, 28), torch.float32), ((1024, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_resnext_resnext101_32x8d_img_cls_torchhub",
                "onnx_dla_dla102x2_visual_bb_torchvision",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub",
                "pt_resnext_resnext101_64x4d_img_cls_osmr",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 2048, 14, 14), torch.float32), ((2048, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_resnext_resnext101_32x8d_img_cls_torchhub",
                "onnx_dla_dla102x2_visual_bb_torchvision",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_regnet_regnet_x_16gf_img_cls_torchvision",
                "pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub",
                "onnx_efficientnet_efficientnet_b5_img_cls_timm",
                "pt_resnext_resnext101_64x4d_img_cls_osmr",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 96, 64, 64), torch.float32), ((96, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "onnx_efficientnet_efficientnet_b2a_img_cls_timm",
                "pt_swin_swin_v2_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add38,
        [((64, 64, 288), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
        },
    ),
    (
        Add20,
        [((1, 15, 15, 512), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "pt_swin_swin_v2_b_img_cls_torchvision",
                "pt_swin_swin_v2_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((64, 3, 64, 64), torch.float32), ((1, 3, 64, 64), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add39,
        [((64, 64, 96), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 64, 64, 96), torch.float32), ((1, 64, 64, 96), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
        },
    ),
    (
        Add5,
        [((1, 64, 64, 384), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
        },
    ),
    (
        Add39,
        [((1, 64, 64, 96), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
        },
    ),
    (
        Add73,
        [((1, 64, 3, 64, 64), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add40,
        [((16, 64, 576), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((16, 6, 64, 64), torch.float32), ((1, 6, 64, 64), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add36,
        [((16, 64, 192), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 32, 32, 192), torch.float32), ((1, 32, 32, 192), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
        },
    ),
    (
        Add14,
        [((1, 32, 32, 768), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
        },
    ),
    (
        Add36,
        [((1, 32, 32, 192), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
        },
    ),
    (
        Add74,
        [((1, 16, 6, 64, 64), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add75,
        [((4, 64, 1152), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((4, 12, 64, 64), torch.float32), ((1, 12, 64, 64), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add5,
        [((4, 64, 384), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 16, 16, 384), torch.float32), ((1, 16, 16, 384), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
        },
    ),
    (
        Add65,
        [((1, 16, 16, 1536), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
        },
    ),
    (
        Add5,
        [((1, 16, 16, 384), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
        },
    ),
    (
        Add76,
        [((1, 4, 12, 64, 64), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add77,
        [((1, 64, 2304), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 24, 64, 64), torch.float32), ((1, 24, 64, 64), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "onnx_efficientnet_efficientnet_b2a_img_cls_timm",
                "pt_swin_swin_v2_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add14,
        [((1, 64, 768), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 8, 8, 768), torch.float32), ((1, 8, 8, 768), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
        },
    ),
    (
        Add15,
        [((1, 8, 8, 3072), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
        },
    ),
    (
        Add14,
        [((1, 8, 8, 768), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
        },
    ),
    (
        Add13,
        [((1, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_t5_google_flan_t5_small_text_gen_hf",
                "pt_t5_t5_base_text_gen_hf",
                "pt_t5_t5_small_text_gen_hf",
                "pt_t5_google_flan_t5_base_text_gen_hf",
                "pt_t5_google_flan_t5_large_text_gen_hf",
                "pt_t5_t5_large_text_gen_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 6, 1, 1), torch.float32), ((1, 6, 1, 1), torch.float32)],
        {"model_names": ["pt_t5_google_flan_t5_small_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1, 512), torch.float32), ((1, 1, 512), torch.float32)],
        {
            "model_names": [
                "pt_t5_google_flan_t5_small_text_gen_hf",
                "pt_t5_t5_small_text_gen_hf",
                "pt_whisper_openai_whisper_base_speech_recognition_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add13,
        [((1, 61, 1), torch.float32)],
        {
            "model_names": [
                "pt_t5_google_flan_t5_small_text_gen_hf",
                "pt_t5_t5_base_text_gen_hf",
                "pt_t5_t5_small_text_gen_hf",
                "pt_t5_google_flan_t5_base_text_gen_hf",
                "pt_t5_google_flan_t5_large_text_gen_hf",
                "pt_t5_t5_large_text_gen_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add78,
        [((1, 6, 61, 61), torch.float32)],
        {"model_names": ["pt_t5_google_flan_t5_small_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 6, 61, 61), torch.float32), ((1, 6, 61, 61), torch.float32)],
        {"model_names": ["pt_t5_google_flan_t5_small_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 61, 512), torch.float32), ((1, 61, 512), torch.float32)],
        {"model_names": ["pt_t5_google_flan_t5_small_text_gen_hf", "pt_t5_t5_small_text_gen_hf"], "pcc": 0.99},
    ),
    (Add79, [((1, 6, 1, 61), torch.float32)], {"model_names": ["pt_t5_google_flan_t5_small_text_gen_hf"], "pcc": 0.99}),
    (
        Add0,
        [((1, 64, 224, 224), torch.float32), ((64, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_vgg_19_obj_det_hf",
                "pt_vgg_vgg11_bn_img_cls_torchvision",
                "pt_vgg_vgg13_img_cls_torchvision",
                "pt_vgg_vgg19_bn_obj_det_timm",
                "pt_vgg_bn_vgg19_obj_det_osmr",
                "pt_vgg_vgg16_img_cls_torchvision",
                "pt_unet_carvana_base_img_seg_github",
                "pt_vgg_bn_vgg19b_obj_det_osmr",
                "pt_vgg_vgg16_bn_img_cls_torchvision",
                "pt_vgg_vgg16_obj_det_osmr",
                "pt_vgg_vgg19_obj_det_osmr",
                "pt_vgg_vgg11_img_cls_torchvision",
                "pt_vgg_vgg13_bn_img_cls_torchvision",
                "pt_vgg_vgg19_bn_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 128, 112, 112), torch.float32), ((128, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_vgg_19_obj_det_hf",
                "pt_vgg_vgg11_bn_img_cls_torchvision",
                "pt_vgg_vgg13_img_cls_torchvision",
                "pt_vgg_vgg19_bn_obj_det_timm",
                "pt_vgg_bn_vgg19_obj_det_osmr",
                "pt_vgg_vgg16_img_cls_torchvision",
                "pt_unet_carvana_base_img_seg_github",
                "pt_vgg_bn_vgg19b_obj_det_osmr",
                "pt_vgg_vgg16_bn_img_cls_torchvision",
                "pt_vgg_vgg16_obj_det_osmr",
                "onnx_dla_dla60x_visual_bb_torchvision",
                "pt_dla_dla60x_visual_bb_torchvision",
                "pt_vgg_vgg19_obj_det_osmr",
                "pt_vgg_vgg11_img_cls_torchvision",
                "onnx_dla_dla102x_visual_bb_torchvision",
                "pt_dla_dla102x_visual_bb_torchvision",
                "pt_regnet_facebook_regnet_y_040_img_cls_hf",
                "pt_vgg_vgg13_bn_img_cls_torchvision",
                "pt_vgg_vgg19_bn_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add80,
        [((1, 197, 1024), torch.float32)],
        {
            "model_names": ["pt_vit_google_vit_large_patch16_224_img_cls_hf", "pt_vit_vit_l_16_img_cls_torchvision"],
            "pcc": 0.99,
        },
    ),
    (
        Add21,
        [((1, 197, 1024), torch.float32)],
        {
            "model_names": [
                "pt_vit_google_vit_large_patch16_224_img_cls_hf",
                "pt_beit_microsoft_beit_large_patch16_224_img_cls_hf",
                "pt_vit_vit_l_16_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 197, 1024), torch.float32), ((1, 197, 1024), torch.float32)],
        {
            "model_names": [
                "pt_vit_google_vit_large_patch16_224_img_cls_hf",
                "pt_beit_microsoft_beit_large_patch16_224_img_cls_hf",
                "pt_vit_vit_l_16_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add24,
        [((1, 197, 4096), torch.float32)],
        {
            "model_names": [
                "pt_vit_google_vit_large_patch16_224_img_cls_hf",
                "pt_beit_microsoft_beit_large_patch16_224_img_cls_hf",
                "pt_vit_vit_l_16_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1024, 3000), torch.float32), ((1024, 1), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_medium_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1024, 1500), torch.float32), ((1024, 1), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_medium_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Add81,
        [((1, 1500, 1024), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_medium_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Add21,
        [((1, 1500, 1024), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_medium_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1500, 1024), torch.float32), ((1, 1500, 1024), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_medium_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Add24,
        [((1, 1500, 4096), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_medium_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Add24,
        [((1, 1, 4096), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_medium_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 64, 150, 150), torch.float32), ((64, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_xception_xception65_img_cls_timm",
                "pt_xception_xception71_tf_in1k_img_cls_timm",
                "pt_xception_xception41_img_cls_timm",
                "pt_ssd300_resnet50_base_img_cls_torchhub",
                "pt_xception_xception71_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 128, 150, 150), torch.float32), ((128, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_xception_xception65_img_cls_timm",
                "pt_xception_xception71_tf_in1k_img_cls_timm",
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 128, 75, 75), torch.float32), ((128, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_xception_xception65_img_cls_timm",
                "pt_xception_xception71_tf_in1k_img_cls_timm",
                "pt_xception_xception41_img_cls_timm",
                "pt_ssd300_resnet50_base_img_cls_torchhub",
                "pt_xception_xception71_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 128, 75, 75), torch.float32), ((1, 128, 75, 75), torch.float32)],
        {
            "model_names": [
                "pt_xception_xception65_img_cls_timm",
                "pt_xception_xception71_tf_in1k_img_cls_timm",
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 256, 75, 75), torch.float32), ((256, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_xception_xception65_img_cls_timm",
                "pt_xception_xception71_tf_in1k_img_cls_timm",
                "pt_xception_xception41_img_cls_timm",
                "pt_ssd300_resnet50_base_img_cls_torchhub",
                "pt_xception_xception71_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 256, 38, 38), torch.float32), ((256, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_xception_xception65_img_cls_timm",
                "pt_xception_xception71_tf_in1k_img_cls_timm",
                "pt_xception_xception41_img_cls_timm",
                "pt_ssd300_resnet50_base_img_cls_torchhub",
                "pt_xception_xception71_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 256, 38, 38), torch.float32), ((1, 256, 38, 38), torch.float32)],
        {
            "model_names": [
                "pt_xception_xception65_img_cls_timm",
                "pt_xception_xception71_tf_in1k_img_cls_timm",
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((728,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": [
                "pt_xception_xception65_img_cls_timm",
                "pt_xception_xception71_tf_in1k_img_cls_timm",
                "pt_xception_xception_img_cls_timm",
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add82,
        [((728,), torch.float32)],
        {
            "model_names": [
                "pt_xception_xception65_img_cls_timm",
                "pt_xception_xception71_tf_in1k_img_cls_timm",
                "pt_xception_xception_img_cls_timm",
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 728, 38, 38), torch.float32), ((728, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_xception_xception65_img_cls_timm",
                "pt_xception_xception71_tf_in1k_img_cls_timm",
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 728, 19, 19), torch.float32), ((728, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_xception_xception65_img_cls_timm",
                "pt_xception_xception71_tf_in1k_img_cls_timm",
                "pt_xception_xception_img_cls_timm",
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 728, 19, 19), torch.float32), ((1, 728, 19, 19), torch.float32)],
        {
            "model_names": [
                "pt_xception_xception65_img_cls_timm",
                "pt_xception_xception71_tf_in1k_img_cls_timm",
                "pt_xception_xception_img_cls_timm",
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1024, 19, 19), torch.float32), ((1024, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_xception_xception65_img_cls_timm",
                "pt_xception_xception71_tf_in1k_img_cls_timm",
                "pt_xception_xception_img_cls_timm",
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1024, 10, 10), torch.float32), ((1024, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_xception_xception65_img_cls_timm",
                "pt_xception_xception71_tf_in1k_img_cls_timm",
                "pt_xception_xception_img_cls_timm",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_320x320",
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1024, 10, 10), torch.float32), ((1, 1024, 10, 10), torch.float32)],
        {
            "model_names": [
                "pt_xception_xception65_img_cls_timm",
                "pt_xception_xception71_tf_in1k_img_cls_timm",
                "pt_xception_xception_img_cls_timm",
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1536,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": [
                "pt_xception_xception65_img_cls_timm",
                "pt_xception_xception71_tf_in1k_img_cls_timm",
                "pt_xception_xception_img_cls_timm",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add65,
        [((1536,), torch.float32)],
        {
            "model_names": [
                "pt_xception_xception65_img_cls_timm",
                "pt_xception_xception71_tf_in1k_img_cls_timm",
                "pt_xception_xception_img_cls_timm",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1536, 10, 10), torch.float32), ((1536, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_xception_xception65_img_cls_timm",
                "pt_xception_xception71_tf_in1k_img_cls_timm",
                "pt_xception_xception_img_cls_timm",
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 2048, 10, 10), torch.float32), ((2048, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_xception_xception65_img_cls_timm",
                "pt_xception_xception71_tf_in1k_img_cls_timm",
                "pt_mobilenetv3_ssd_resnet101_img_cls_torchvision",
                "pt_xception_xception_img_cls_timm",
                "pt_mobilenetv3_ssd_resnet152_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet50_img_cls_torchvision",
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 32, 160, 160), torch.float32), ((32, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5s_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_640x640",
                "pt_yolov8_default_obj_det_github",
                "onnx_yolov10_default_obj_det_github",
                "pt_yolov9_default_obj_det_github",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_640x640",
                "pt_yolox_yolox_s_obj_det_torchhub",
                "onnx_yolov8_default_obj_det_github",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 64, 80, 80), torch.float32), ((64, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5s_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_640x640",
                "pt_mobilenetv3_ssd_resnet101_img_cls_torchvision",
                "pt_yolov8_default_obj_det_github",
                "onnx_yolov10_default_obj_det_github",
                "pt_mobilenetv3_ssd_resnet152_img_cls_torchvision",
                "pt_yolov9_default_obj_det_github",
                "pt_mobilenetv3_ssd_resnet50_img_cls_torchvision",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_640x640",
                "pt_yolox_yolox_s_obj_det_torchhub",
                "onnx_yolov8_default_obj_det_github",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 32, 80, 80), torch.float32), ((32, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5s_img_cls_torchhub_320x320",
                "pt_yolov8_default_obj_det_github",
                "onnx_yolov10_default_obj_det_github",
                "onnx_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_640x640",
                "onnx_yolov8_default_obj_det_github",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 32, 80, 80), torch.float32), ((1, 32, 80, 80), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5s_img_cls_torchhub_320x320",
                "pt_yolov8_default_obj_det_github",
                "onnx_yolov10_default_obj_det_github",
                "onnx_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_640x640",
                "onnx_yolov8_default_obj_det_github",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 128, 40, 40), torch.float32), ((128, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5s_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_640x640",
                "pt_mobilenetv3_ssd_resnet101_img_cls_torchvision",
                "pt_yolov8_default_obj_det_github",
                "onnx_yolov10_default_obj_det_github",
                "pt_mobilenetv3_ssd_resnet152_img_cls_torchvision",
                "pt_yolov9_default_obj_det_github",
                "pt_mobilenetv3_ssd_resnet50_img_cls_torchvision",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_640x640",
                "pt_yolox_yolox_s_obj_det_torchhub",
                "onnx_yolov8_default_obj_det_github",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 64, 40, 40), torch.float32), ((64, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5s_img_cls_torchhub_320x320",
                "pt_yolov8_default_obj_det_github",
                "onnx_yolov10_default_obj_det_github",
                "pt_yolov9_default_obj_det_github",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_640x640",
                "onnx_yolov8_default_obj_det_github",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 64, 40, 40), torch.float32), ((1, 64, 40, 40), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5s_img_cls_torchhub_320x320",
                "pt_yolov8_default_obj_det_github",
                "onnx_yolov10_default_obj_det_github",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_640x640",
                "onnx_yolov8_default_obj_det_github",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 256, 20, 20), torch.float32), ((256, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5s_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_640x640",
                "pt_mobilenetv3_ssd_resnet101_img_cls_torchvision",
                "pt_yolov8_default_obj_det_github",
                "onnx_yolov10_default_obj_det_github",
                "pt_mobilenetv3_ssd_resnet152_img_cls_torchvision",
                "pt_yolov9_default_obj_det_github",
                "pt_mobilenetv3_ssd_resnet50_img_cls_torchvision",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_640x640",
                "pt_yolox_yolox_s_obj_det_torchhub",
                "onnx_yolov8_default_obj_det_github",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 128, 20, 20), torch.float32), ((128, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5s_img_cls_torchhub_320x320",
                "pt_yolov8_default_obj_det_github",
                "onnx_yolov10_default_obj_det_github",
                "pt_yolov9_default_obj_det_github",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_640x640",
                "pt_yolox_yolox_s_obj_det_torchhub",
                "onnx_yolov8_default_obj_det_github",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 128, 20, 20), torch.float32), ((1, 128, 20, 20), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5s_img_cls_torchhub_320x320",
                "pt_yolov8_default_obj_det_github",
                "onnx_yolov10_default_obj_det_github",
                "pt_yolov9_default_obj_det_github",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_640x640",
                "onnx_yolov8_default_obj_det_github",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 512, 10, 10), torch.float32), ((512, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5s_img_cls_torchhub_320x320",
                "pt_mobilenetv3_ssd_resnet101_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet152_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet50_img_cls_torchvision",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_320x320",
                "pt_ssd300_resnet50_base_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 256, 10, 10), torch.float32), ((256, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5s_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_320x320",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 256, 10, 10), torch.float32), ((1, 256, 10, 10), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5s_img_cls_torchhub_320x320"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 255, 40, 40), torch.float32), ((255, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5s_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_640x640",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 255, 40, 40), torch.float32), ((1, 255, 40, 40), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5s_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_640x640",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 255, 20, 20), torch.float32), ((255, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5s_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_640x640",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 255, 20, 20), torch.float32), ((1, 255, 20, 20), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5s_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_640x640",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 255, 10, 10), torch.float32), ((255, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5s_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_320x320",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 255, 10, 10), torch.float32), ((1, 255, 10, 10), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5s_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_320x320",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 32, 320, 320), torch.float32), ((32, 1, 1), torch.float32)],
        {
            "model_names": ["pt_yolo_v5_yolov5s_img_cls_torchhub_640x640", "pt_yolox_yolox_s_obj_det_torchhub"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 64, 160, 160), torch.float32), ((64, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5s_img_cls_torchhub_640x640",
                "pt_mobilenetv3_ssd_resnet101_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet152_img_cls_torchvision",
                "pt_yolov9_default_obj_det_github",
                "pt_mobilenetv3_ssd_resnet50_img_cls_torchvision",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_640x640",
                "pt_yolox_yolox_s_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 32, 160, 160), torch.float32), ((1, 32, 160, 160), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5s_img_cls_torchhub_640x640",
                "pt_yolov9_default_obj_det_github",
                "pt_yolox_yolox_s_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 128, 80, 80), torch.float32), ((128, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5s_img_cls_torchhub_640x640",
                "pt_mobilenetv3_ssd_resnet101_img_cls_torchvision",
                "onnx_yolov10_default_obj_det_github",
                "pt_mobilenetv3_ssd_resnet152_img_cls_torchvision",
                "pt_yolov9_default_obj_det_github",
                "pt_mobilenetv3_ssd_resnet50_img_cls_torchvision",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_640x640",
                "pt_yolox_yolox_s_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 64, 80, 80), torch.float32), ((1, 64, 80, 80), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5s_img_cls_torchhub_640x640",
                "pt_yolov9_default_obj_det_github",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_320x320",
                "pt_yolox_yolox_s_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 256, 40, 40), torch.float32), ((256, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5s_img_cls_torchhub_640x640",
                "pt_mobilenetv3_ssd_resnet101_img_cls_torchvision",
                "onnx_yolov10_default_obj_det_github",
                "pt_mobilenetv3_ssd_resnet152_img_cls_torchvision",
                "pt_yolov9_default_obj_det_github",
                "pt_mobilenetv3_ssd_resnet50_img_cls_torchvision",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_640x640",
                "pt_yolox_yolox_s_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 128, 40, 40), torch.float32), ((1, 128, 40, 40), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5s_img_cls_torchhub_640x640",
                "pt_yolov9_default_obj_det_github",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_320x320",
                "pt_yolox_yolox_s_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 512, 20, 20), torch.float32), ((512, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5s_img_cls_torchhub_640x640",
                "pt_mobilenetv3_ssd_resnet101_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet152_img_cls_torchvision",
                "pt_yolov9_default_obj_det_github",
                "pt_mobilenetv3_ssd_resnet50_img_cls_torchvision",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_640x640",
                "pt_yolox_yolox_s_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 256, 20, 20), torch.float32), ((1, 256, 20, 20), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5s_img_cls_torchhub_640x640",
                "onnx_yolov10_default_obj_det_github",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_320x320",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 255, 80, 80), torch.float32), ((255, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5s_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_640x640",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 255, 80, 80), torch.float32), ((1, 255, 80, 80), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5s_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_640x640",
            ],
            "pcc": 0.99,
        },
    ),
    (Add83, [((16,), torch.float32)], {"model_names": ["pt_yolo_v6_yolov6n_obj_det_torchhub"], "pcc": 0.99}),
    (
        Add0,
        [((1, 16, 224, 320), torch.float32), ((16, 1, 1), torch.float32)],
        {"model_names": ["pt_yolo_v6_yolov6n_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 16, 224, 320), torch.float32), ((1, 16, 224, 320), torch.float32)],
        {"model_names": ["pt_yolo_v6_yolov6n_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Add84,
        [((32,), torch.float32)],
        {"model_names": ["pt_yolo_v6_yolov6n_obj_det_torchhub", "pt_yolo_v6_yolov6s_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 32, 112, 160), torch.float32), ((32, 1, 1), torch.float32)],
        {"model_names": ["pt_yolo_v6_yolov6n_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 32, 112, 160), torch.float32), ((1, 32, 112, 160), torch.float32)],
        {"model_names": ["pt_yolo_v6_yolov6n_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Add85,
        [((64,), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v6_yolov6n_obj_det_torchhub",
                "pt_yolo_v6_yolov6s_obj_det_torchhub",
                "pt_yolo_v6_yolov6m_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 64, 56, 80), torch.float32), ((64, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v6_yolov6n_obj_det_torchhub",
                "pt_yolo_v6_yolov6s_obj_det_torchhub",
                "pt_yolo_v6_yolov6l_obj_det_torchhub",
                "pt_yolo_v6_yolov6m_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 64, 56, 80), torch.float32), ((1, 64, 56, 80), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v6_yolov6n_obj_det_torchhub",
                "pt_yolo_v6_yolov6s_obj_det_torchhub",
                "pt_yolo_v6_yolov6l_obj_det_torchhub",
                "pt_yolo_v6_yolov6m_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add86,
        [((128,), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v6_yolov6n_obj_det_torchhub",
                "pt_yolo_v6_yolov6s_obj_det_torchhub",
                "pt_yolo_v6_yolov6m_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 128, 28, 40), torch.float32), ((128, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v6_yolov6n_obj_det_torchhub",
                "pt_yolo_v6_yolov6s_obj_det_torchhub",
                "pt_yolo_v6_yolov6l_obj_det_torchhub",
                "pt_yolo_v6_yolov6m_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 128, 28, 40), torch.float32), ((1, 128, 28, 40), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v6_yolov6n_obj_det_torchhub",
                "pt_yolo_v6_yolov6s_obj_det_torchhub",
                "pt_yolo_v6_yolov6l_obj_det_torchhub",
                "pt_yolo_v6_yolov6m_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add87,
        [((256,), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v6_yolov6n_obj_det_torchhub",
                "pt_yolo_v6_yolov6s_obj_det_torchhub",
                "pt_yolo_v6_yolov6m_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 256, 14, 20), torch.float32), ((1, 256, 14, 20), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v6_yolov6n_obj_det_torchhub",
                "pt_yolo_v6_yolov6s_obj_det_torchhub",
                "pt_yolo_v6_yolov6l_obj_det_torchhub",
                "pt_yolo_v6_yolov6m_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 128, 14, 20), torch.float32), ((128, 1, 1), torch.float32)],
        {"model_names": ["pt_yolo_v6_yolov6n_obj_det_torchhub", "pt_yolo_v6_yolov6s_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 64, 14, 20), torch.float32), ((64, 1, 1), torch.float32)],
        {"model_names": ["pt_yolo_v6_yolov6n_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 64, 28, 40), torch.float32), ((64, 1, 1), torch.float32)],
        {"model_names": ["pt_yolo_v6_yolov6n_obj_det_torchhub", "pt_yolo_v6_yolov6s_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 64, 28, 40), torch.float32), ((1, 64, 28, 40), torch.float32)],
        {"model_names": ["pt_yolo_v6_yolov6n_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 32, 28, 40), torch.float32), ((32, 1, 1), torch.float32)],
        {"model_names": ["pt_yolo_v6_yolov6n_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 32, 56, 80), torch.float32), ((32, 1, 1), torch.float32)],
        {"model_names": ["pt_yolo_v6_yolov6n_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 32, 56, 80), torch.float32), ((1, 32, 56, 80), torch.float32)],
        {"model_names": ["pt_yolo_v6_yolov6n_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 4, 56, 80), torch.float32), ((4, 1, 1), torch.float32)],
        {"model_names": ["pt_yolo_v6_yolov6n_obj_det_torchhub", "pt_yolo_v6_yolov6s_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 4, 28, 40), torch.float32), ((4, 1, 1), torch.float32)],
        {"model_names": ["pt_yolo_v6_yolov6n_obj_det_torchhub", "pt_yolo_v6_yolov6s_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 128, 14, 20), torch.float32), ((1, 128, 14, 20), torch.float32)],
        {"model_names": ["pt_yolo_v6_yolov6n_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 4, 14, 20), torch.float32), ((4, 1, 1), torch.float32)],
        {"model_names": ["pt_yolo_v6_yolov6n_obj_det_torchhub", "pt_yolo_v6_yolov6s_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Add88,
        [((1, 5880, 2), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v6_yolov6n_obj_det_torchhub",
                "pt_yolo_v6_yolov6s_obj_det_torchhub",
                "pt_yolo_v6_yolov6l_obj_det_torchhub",
                "pt_yolo_v6_yolov6m_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 5880, 2), torch.float32), ((1, 5880, 2), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v6_yolov6n_obj_det_torchhub",
                "pt_yolo_v6_yolov6s_obj_det_torchhub",
                "pt_yolo_v6_yolov6l_obj_det_torchhub",
                "pt_yolo_v6_yolov6m_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 80, 56, 80), torch.float32), ((80, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v6_yolov6n_obj_det_torchhub",
                "pt_yolo_v6_yolov6s_obj_det_torchhub",
                "pt_yolo_v6_yolov6l_obj_det_torchhub",
                "pt_yolo_v6_yolov6m_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 80, 28, 40), torch.float32), ((80, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v6_yolov6n_obj_det_torchhub",
                "pt_yolo_v6_yolov6s_obj_det_torchhub",
                "pt_yolo_v6_yolov6l_obj_det_torchhub",
                "pt_yolo_v6_yolov6m_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 80, 14, 20), torch.float32), ((80, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v6_yolov6n_obj_det_torchhub",
                "pt_yolo_v6_yolov6s_obj_det_torchhub",
                "pt_yolo_v6_yolov6l_obj_det_torchhub",
                "pt_yolo_v6_yolov6m_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 24, 208, 208), torch.float32), ((24, 1, 1), torch.float32)],
        {"model_names": ["pt_yolox_yolox_tiny_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 48, 104, 104), torch.float32), ((48, 1, 1), torch.float32)],
        {"model_names": ["pt_yolox_yolox_tiny_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 24, 104, 104), torch.float32), ((24, 1, 1), torch.float32)],
        {"model_names": ["pt_yolox_yolox_tiny_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 24, 104, 104), torch.float32), ((1, 24, 104, 104), torch.float32)],
        {"model_names": ["pt_yolox_yolox_tiny_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 96, 52, 52), torch.float32), ((96, 1, 1), torch.float32)],
        {"model_names": ["pt_yolox_yolox_tiny_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 48, 52, 52), torch.float32), ((48, 1, 1), torch.float32)],
        {"model_names": ["pt_yolox_yolox_tiny_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 48, 52, 52), torch.float32), ((1, 48, 52, 52), torch.float32)],
        {"model_names": ["pt_yolox_yolox_tiny_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 192, 26, 26), torch.float32), ((192, 1, 1), torch.float32)],
        {"model_names": ["pt_yolox_yolox_tiny_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 96, 26, 26), torch.float32), ((96, 1, 1), torch.float32)],
        {"model_names": ["pt_yolox_yolox_tiny_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 96, 26, 26), torch.float32), ((1, 96, 26, 26), torch.float32)],
        {"model_names": ["pt_yolox_yolox_tiny_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 192, 13, 13), torch.float32), ((192, 1, 1), torch.float32)],
        {"model_names": ["pt_yolox_yolox_tiny_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 4, 52, 52), torch.float32), ((4, 1, 1), torch.float32)],
        {"model_names": ["pt_yolox_yolox_tiny_obj_det_torchhub", "pt_yolox_yolox_nano_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1, 52, 52), torch.float32), ((1, 1, 1), torch.float32)],
        {"model_names": ["pt_yolox_yolox_tiny_obj_det_torchhub", "pt_yolox_yolox_nano_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 80, 52, 52), torch.float32), ((80, 1, 1), torch.float32)],
        {"model_names": ["pt_yolox_yolox_tiny_obj_det_torchhub", "pt_yolox_yolox_nano_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 4, 26, 26), torch.float32), ((4, 1, 1), torch.float32)],
        {"model_names": ["pt_yolox_yolox_tiny_obj_det_torchhub", "pt_yolox_yolox_nano_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1, 26, 26), torch.float32), ((1, 1, 1), torch.float32)],
        {"model_names": ["pt_yolox_yolox_tiny_obj_det_torchhub", "pt_yolox_yolox_nano_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 80, 26, 26), torch.float32), ((80, 1, 1), torch.float32)],
        {"model_names": ["pt_yolox_yolox_tiny_obj_det_torchhub", "pt_yolox_yolox_nano_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 96, 13, 13), torch.float32), ((96, 1, 1), torch.float32)],
        {"model_names": ["pt_yolox_yolox_tiny_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 4, 13, 13), torch.float32), ((4, 1, 1), torch.float32)],
        {"model_names": ["pt_yolox_yolox_tiny_obj_det_torchhub", "pt_yolox_yolox_nano_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1, 13, 13), torch.float32), ((1, 1, 1), torch.float32)],
        {"model_names": ["pt_yolox_yolox_tiny_obj_det_torchhub", "pt_yolox_yolox_nano_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 80, 13, 13), torch.float32), ((80, 1, 1), torch.float32)],
        {"model_names": ["pt_yolox_yolox_tiny_obj_det_torchhub", "pt_yolox_yolox_nano_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Add7,
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
        Add89,
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
        Add89,
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
        Add90,
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
        Add91,
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
        Add91,
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
        Add92,
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
        Add90,
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
        Add90,
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
        Add93,
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
        Add92,
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
        Add92,
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
        Add94,
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
        Add7,
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
        Add12,
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
        Add12,
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
        Add95,
        [((1, 100, 92), torch.float32)],
        {"model_names": ["onnx_detr_facebook_detr_resnet_50_obj_det_hf"], "pcc": 0.99},
    ),
    (
        Add96,
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
        [((1, 256, 112, 112), torch.float32), ((256, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_dla_dla102x2_visual_bb_torchvision",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_regnet_regnet_x_16gf_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add97,
        [((1, 256, 768), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add97,
        [((1, 1024, 768), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add97,
        [((1, 4096, 768), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add97,
        [((1, 16384, 768), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
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
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 11, 768), torch.float32), ((1, 11, 768), torch.float32)],
        {
            "model_names": [
                "pd_bert_chinese_roberta_base_qa_padlenlp",
                "pd_bert_chinese_roberta_base_seq_cls_padlenlp",
                "pd_roberta_rbt4_ch_clm_padlenlp",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add13,
        [((1, 11, 1), torch.float32)],
        {
            "model_names": [
                "pd_bert_chinese_roberta_base_qa_padlenlp",
                "pd_albert_chinese_tiny_mlm_padlenlp",
                "pd_bert_chinese_roberta_base_seq_cls_padlenlp",
                "pd_roberta_rbt4_ch_clm_padlenlp",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add14,
        [((1, 11, 768), torch.float32)],
        {
            "model_names": [
                "pd_bert_chinese_roberta_base_qa_padlenlp",
                "pd_bert_chinese_roberta_base_seq_cls_padlenlp",
                "pd_roberta_rbt4_ch_clm_padlenlp",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add13,
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
                "pd_albert_chinese_tiny_mlm_padlenlp",
                "pd_bert_chinese_roberta_base_seq_cls_padlenlp",
                "pd_roberta_rbt4_ch_clm_padlenlp",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add15,
        [((1, 11, 3072), torch.float32)],
        {
            "model_names": [
                "pd_bert_chinese_roberta_base_qa_padlenlp",
                "pd_bert_chinese_roberta_base_seq_cls_padlenlp",
                "pd_roberta_rbt4_ch_clm_padlenlp",
            ],
            "pcc": 0.99,
        },
    ),
    (Add16, [((1, 11, 2), torch.float32)], {"model_names": ["pd_bert_chinese_roberta_base_qa_padlenlp"], "pcc": 0.99}),
    (
        Add16,
        [((1, 128, 2), torch.float32)],
        {
            "model_names": [
                "pt_albert_base_v1_token_cls_hf",
                "pt_albert_large_v1_token_cls_hf",
                "pt_albert_large_v2_token_cls_hf",
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_albert_xxlarge_v2_token_cls_hf",
                "pt_albert_base_v2_token_cls_hf",
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
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
                "pt_opt_facebook_opt_350m_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add21,
        [((1, 256, 1024), torch.float32)],
        {
            "model_names": [
                "pt_bart_facebook_bart_large_mnli_seq_cls_hf",
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
                "pt_opt_facebook_opt_350m_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add98,
        [((1, 16, 256, 256), torch.float32)],
        {"model_names": ["pt_bart_facebook_bart_large_mnli_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1, 256, 256), torch.float32), ((1, 1, 256, 256), torch.float32)],
        {
            "model_names": [
                "pt_bart_facebook_bart_large_mnli_seq_cls_hf",
                "pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf",
                "pt_gptneo_eleutherai_gpt_neo_125m_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf",
                "pt_phi2_microsoft_phi_2_clm_hf",
                "pt_xglm_facebook_xglm_1_7b_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
                "pt_opt_facebook_opt_350m_clm_hf",
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
                "pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf",
                "pt_xglm_facebook_xglm_1_7b_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
                "pt_opt_facebook_opt_350m_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add24,
        [((1, 256, 4096), torch.float32)],
        {
            "model_names": [
                "pt_bart_facebook_bart_large_mnli_seq_cls_hf",
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
                "pt_albert_large_v1_token_cls_hf",
                "pt_albert_large_v2_mlm_hf",
                "pt_albert_large_v2_token_cls_hf",
                "pt_albert_large_v1_mlm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add21,
        [((1, 128, 1024), torch.float32)],
        {
            "model_names": [
                "pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf",
                "pt_albert_large_v1_token_cls_hf",
                "pt_albert_large_v2_mlm_hf",
                "pt_albert_large_v2_token_cls_hf",
                "pt_albert_large_v1_mlm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add25,
        [((1, 16, 128, 128), torch.float32)],
        {"model_names": ["pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf"], "pcc": 0.99},
    ),
    (
        Add24,
        [((1, 128, 4096), torch.float32)],
        {
            "model_names": [
                "pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf",
                "pt_albert_large_v1_token_cls_hf",
                "pt_albert_large_v2_mlm_hf",
                "pt_albert_large_v2_token_cls_hf",
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
                "pt_albert_xxlarge_v2_token_cls_hf",
                "pt_albert_large_v1_mlm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add99,
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
        Add0,
        [((1, 32, 1, 1), torch.float32), ((32, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2a_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "onnx_efficientnet_efficientnet_b5_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_regnet_facebook_regnet_y_040_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 4, 1, 1), torch.float32), ((4, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 96, 1, 1), torch.float32), ((96, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "TranslatedLayer",
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2a_img_cls_timm",
                "pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 6, 1, 1), torch.float32), ((6, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "TranslatedLayer",
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "onnx_efficientnet_efficientnet_b2a_img_cls_timm",
                "pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b4_img_cls_timm",
                "onnx_efficientnet_efficientnet_b5_img_cls_timm",
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 144, 1, 1), torch.float32), ((144, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_regnet_regnet_y_800mf_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_regnet_facebook_regnet_y_064_img_cls_hf",
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "onnx_efficientnet_efficientnet_b2a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "onnx_efficientnet_efficientnet_b4_img_cls_timm",
                "onnx_efficientnet_efficientnet_b5_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_regnet_regnet_y_3_2gf_img_cls_torchvision",
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((40,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_hrnet_hrnetv2_w40_pose_estimation_osmr",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add100,
        [((40,), torch.float32)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_hrnet_hrnetv2_w40_pose_estimation_osmr",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 10, 1, 1), torch.float32), ((10, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b5_img_cls_timm",
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 240, 1, 1), torch.float32), ((240, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pd_paddleocr_v4_rec_ch_scene_text_recognition_paddlemodels",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "onnx_efficientnet_efficientnet_b5_img_cls_timm",
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((80,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_yolox_yolox_x_obj_det_torchhub",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_yolov8_default_obj_det_github",
                "pt_regnet_regnet_x_8gf_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                "pt_vovnet_vovnet27s_img_cls_osmr",
                "pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_hrnet_hrnetv2_w40_pose_estimation_osmr",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add101,
        [((80,), torch.float32)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_yolox_yolox_x_obj_det_torchhub",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_yolov8_default_obj_det_github",
                "pt_regnet_regnet_x_8gf_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                "pt_vovnet_vovnet27s_img_cls_osmr",
                "pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_hrnet_hrnetv2_w40_pose_estimation_osmr",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((480,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                "pd_paddleocr_v4_rec_ch_scene_text_recognition_paddlemodels",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add102,
        [((480,), torch.float32)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                "pd_paddleocr_v4_rec_ch_scene_text_recognition_paddlemodels",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 20, 1, 1), torch.float32), ((20, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 480, 1, 1), torch.float32), ((480, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pd_paddleocr_v4_rec_ch_scene_text_recognition_paddlemodels",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 28, 1, 1), torch.float32), ((28, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "onnx_efficientnet_efficientnet_b4_img_cls_timm",
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 672, 1, 1), torch.float32), ((672, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "onnx_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1152,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add75,
        [((1152,), torch.float32)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1152, 1, 1), torch.float32), ((1152, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 32, 190, 190), torch.float32), ((32, 1, 1), torch.float32)],
        {"model_names": ["pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 24, 190, 190), torch.float32), ((24, 1, 1), torch.float32)],
        {"model_names": ["pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 144, 190, 190), torch.float32), ((144, 1, 1), torch.float32)],
        {"model_names": ["pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 144, 95, 95), torch.float32), ((144, 1, 1), torch.float32)],
        {"model_names": ["pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 32, 95, 95), torch.float32), ((32, 1, 1), torch.float32)],
        {"model_names": ["pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 192, 95, 95), torch.float32), ((192, 1, 1), torch.float32)],
        {"model_names": ["pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 32, 95, 95), torch.float32), ((1, 32, 95, 95), torch.float32)],
        {"model_names": ["pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 192, 48, 48), torch.float32), ((192, 1, 1), torch.float32)],
        {"model_names": ["pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((56,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": [
                "pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add103,
        [((56,), torch.float32)],
        {
            "model_names": [
                "pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 56, 48, 48), torch.float32), ((56, 1, 1), torch.float32)],
        {"model_names": ["pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 336, 48, 48), torch.float32), ((336, 1, 1), torch.float32)],
        {"model_names": ["pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 56, 48, 48), torch.float32), ((1, 56, 48, 48), torch.float32)],
        {"model_names": ["pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 336, 24, 24), torch.float32), ((336, 1, 1), torch.float32)],
        {"model_names": ["pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 112, 24, 24), torch.float32), ((112, 1, 1), torch.float32)],
        {"model_names": ["pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 672, 24, 24), torch.float32), ((672, 1, 1), torch.float32)],
        {"model_names": ["pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 112, 24, 24), torch.float32), ((1, 112, 24, 24), torch.float32)],
        {"model_names": ["pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 160, 24, 24), torch.float32), ((160, 1, 1), torch.float32)],
        {"model_names": ["pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 960, 24, 24), torch.float32), ((960, 1, 1), torch.float32)],
        {"model_names": ["pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 160, 24, 24), torch.float32), ((1, 160, 24, 24), torch.float32)],
        {"model_names": ["pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 960, 12, 12), torch.float32), ((960, 1, 1), torch.float32)],
        {"model_names": ["pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((272,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": [
                "pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add104,
        [((272,), torch.float32)],
        {
            "model_names": [
                "pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 272, 12, 12), torch.float32), ((272, 1, 1), torch.float32)],
        {"model_names": ["pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1632,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": [
                "pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add105,
        [((1632,), torch.float32)],
        {
            "model_names": [
                "pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1632, 12, 12), torch.float32), ((1632, 1, 1), torch.float32)],
        {"model_names": ["pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 272, 12, 12), torch.float32), ((1, 272, 12, 12), torch.float32)],
        {"model_names": ["pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 448, 12, 12), torch.float32), ((448, 1, 1), torch.float32)],
        {"model_names": ["pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1280, 12, 12), torch.float32), ((1280, 1, 1), torch.float32)],
        {"model_names": ["pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((8,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add106,
        [((8,), torch.float32)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 8, 112, 112), torch.float32), ((8, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_050_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 16, 112, 112), torch.float32), ((1, 16, 112, 112), torch.float32)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 24, 112, 112), torch.float32), ((24, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels",
                "onnx_mobilenetv2_mobilenetv2_140_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((12,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add107,
        [((12,), torch.float32)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 12, 56, 56), torch.float32), ((12, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 16, 56, 56), torch.float32), ((16, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_050_img_cls_timm",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((36,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add108,
        [((36,), torch.float32)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 36, 56, 56), torch.float32), ((36, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((72,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_regnet_regnet_x_1_6gf_img_cls_torchvision",
                "pt_regnet_regnet_y_3_2gf_img_cls_torchvision",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add109,
        [((72,), torch.float32)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_regnet_regnet_x_1_6gf_img_cls_torchvision",
                "pt_regnet_regnet_y_3_2gf_img_cls_torchvision",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 72, 28, 28), torch.float32), ((72, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 72, 1, 1), torch.float32), ((72, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_regnet_facebook_regnet_y_064_img_cls_hf",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_regnet_regnet_y_3_2gf_img_cls_torchvision",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add13,
        [((1, 72, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((20,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add110,
        [((20,), torch.float32)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 20, 28, 28), torch.float32), ((20, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 24, 28, 28), torch.float32), ((24, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 120, 1, 1), torch.float32), ((120, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pt_regnet_regnet_y_1_6gf_img_cls_torchvision",
                "pd_paddleocr_v4_rec_ch_scene_text_recognition_paddlemodels",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add13,
        [((1, 120, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 40, 14, 14), torch.float32), ((40, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_hrnet_hrnetv2_w40_pose_estimation_osmr",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((100,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add111,
        [((100,), torch.float32)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 100, 14, 14), torch.float32), ((100, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((92,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add31,
        [((92,), torch.float32)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 92, 14, 14), torch.float32), ((92, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add13,
        [((1, 480, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pd_paddleocr_v4_rec_ch_scene_text_recognition_paddlemodels",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 56, 14, 14), torch.float32), ((56, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 336, 14, 14), torch.float32), ((336, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_regnet_regnet_y_1_6gf_img_cls_torchvision",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 168, 1, 1), torch.float32), ((168, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_regnet_facebook_regnet_y_080_img_cls_hf",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add13,
        [((1, 672, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 80, 7, 7), torch.float32), ((80, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_050_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 112, 7, 7), torch.float32), ((112, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_vovnet_vovnet27s_img_cls_osmr",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 480, 7, 7), torch.float32), ((480, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_050_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 960, 1, 1), torch.float32), ((960, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "onnx_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add13,
        [((1, 960, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1280, 1, 1), torch.float32), ((1280, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 32, 56, 56), torch.float32), ((1, 32, 56, 56), torch.float32)],
        {
            "model_names": [
                "pt_hrnet_hrnetv2_w32_pose_estimation_osmr",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "onnx_mobilenetv2_mobilenetv2_140_img_cls_timm",
                "pt_regnet_regnet_x_400mf_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 64, 28, 28), torch.float32), ((1, 64, 28, 28), torch.float32)],
        {
            "model_names": [
                "pt_hrnet_hrnetv2_w32_pose_estimation_osmr",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "onnx_dla_dla46_c_visual_bb_torchvision",
                "onnx_dla_dla60x_c_visual_bb_torchvision",
                "pt_dla_dla60x_c_visual_bb_torchvision",
                "pt_regnet_regnet_x_400mf_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 128, 14, 14), torch.float32), ((1, 128, 14, 14), torch.float32)],
        {
            "model_names": [
                "pt_hrnet_hrnetv2_w32_pose_estimation_osmr",
                "onnx_dla_dla46_c_visual_bb_torchvision",
                "onnx_dla_dla60x_c_visual_bb_torchvision",
                "pt_dla_dla60x_c_visual_bb_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 32, 14, 14), torch.float32), ((32, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_hrnet_hrnetv2_w32_pose_estimation_osmr",
                "onnx_mobilenetv2_mobilenetv2_050_img_cls_timm",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 256, 7, 7), torch.float32), ((1, 256, 7, 7), torch.float32)],
        {
            "model_names": [
                "pt_hrnet_hrnetv2_w32_pose_estimation_osmr",
                "onnx_dla_dla46_c_visual_bb_torchvision",
                "onnx_dla_dla60x_c_visual_bb_torchvision",
                "pt_dla_dla60x_c_visual_bb_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add20,
        [((1, 256, 512), torch.float32)],
        {
            "model_names": [
                "pt_mlp_mixer_base_img_cls_github",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1024, 512), torch.float32), ((1024, 1), torch.float32)],
        {"model_names": ["pt_mlp_mixer_base_img_cls_github"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 256, 512), torch.float32), ((256, 1), torch.float32)],
        {"model_names": ["pt_mlp_mixer_base_img_cls_github"], "pcc": 0.99},
    ),
    (
        Add18,
        [((1, 256, 256), torch.float32)],
        {
            "model_names": [
                "pt_mlp_mixer_base_img_cls_github",
                "pt_perceiverio_deepmind_language_perceiver_mlm_hf",
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add18,
        [((1, 512, 256), torch.float32)],
        {
            "model_names": ["pt_mlp_mixer_mixer_s16_224_img_cls_timm", "pt_mlp_mixer_mixer_s32_224_img_cls_timm"],
            "pcc": 0.99,
        },
    ),
    (
        Add112,
        [((1, 512, 196), torch.float32)],
        {"model_names": ["pt_mlp_mixer_mixer_s16_224_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 196, 512), torch.float32), ((1, 196, 512), torch.float32)],
        {"model_names": ["pt_mlp_mixer_mixer_s16_224_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add22,
        [((1, 196, 2048), torch.float32)],
        {"model_names": ["pt_mlp_mixer_mixer_s16_224_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add20,
        [((1, 196, 512), torch.float32)],
        {"model_names": ["pt_mlp_mixer_mixer_s16_224_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 16, 48, 48), torch.float32), ((16, 1, 1), torch.float32)],
        {"model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 8, 48, 48), torch.float32), ((8, 1, 1), torch.float32)],
        {"model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 48, 48, 48), torch.float32), ((48, 1, 1), torch.float32)],
        {"model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 48, 24, 24), torch.float32), ((48, 1, 1), torch.float32)],
        {"model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 8, 24, 24), torch.float32), ((8, 1, 1), torch.float32)],
        {"model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 8, 24, 24), torch.float32), ((1, 8, 24, 24), torch.float32)],
        {"model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 48, 12, 12), torch.float32), ((48, 1, 1), torch.float32)],
        {"model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 16, 12, 12), torch.float32), ((16, 1, 1), torch.float32)],
        {"model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 96, 12, 12), torch.float32), ((96, 1, 1), torch.float32)],
        {"model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 16, 12, 12), torch.float32), ((1, 16, 12, 12), torch.float32)],
        {"model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 96, 6, 6), torch.float32), ((96, 1, 1), torch.float32)],
        {"model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 24, 6, 6), torch.float32), ((24, 1, 1), torch.float32)],
        {"model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 144, 6, 6), torch.float32), ((144, 1, 1), torch.float32)],
        {"model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 24, 6, 6), torch.float32), ((1, 24, 6, 6), torch.float32)],
        {"model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 32, 6, 6), torch.float32), ((32, 1, 1), torch.float32)],
        {"model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 192, 6, 6), torch.float32), ((192, 1, 1), torch.float32)],
        {"model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 32, 6, 6), torch.float32), ((1, 32, 6, 6), torch.float32)],
        {"model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 192, 3, 3), torch.float32), ((192, 1, 1), torch.float32)],
        {"model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 56, 3, 3), torch.float32), ((56, 1, 1), torch.float32)],
        {"model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 336, 3, 3), torch.float32), ((336, 1, 1), torch.float32)],
        {"model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 56, 3, 3), torch.float32), ((1, 56, 3, 3), torch.float32)],
        {"model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 112, 3, 3), torch.float32), ((112, 1, 1), torch.float32)],
        {"model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1280, 3, 3), torch.float32), ((1280, 1, 1), torch.float32)],
        {"model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add13,
        [((1, 16, 112, 112), torch.float32)],
        {
            "model_names": [
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 72, 56, 56), torch.float32), ((72, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_regnet_regnet_x_1_6gf_img_cls_torchvision",
                "pt_regnet_regnet_y_3_2gf_img_cls_torchvision",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 24, 1, 1), torch.float32), ((24, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "TranslatedLayer",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "onnx_efficientnet_efficientnet_b5_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add13,
        [((1, 240, 28, 28), torch.float32)],
        {
            "model_names": [
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add13,
        [((1, 240, 14, 14), torch.float32)],
        {
            "model_names": [
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((200,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": [
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add113,
        [((200,), torch.float32)],
        {
            "model_names": [
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 200, 14, 14), torch.float32), ((200, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add13,
        [((1, 200, 14, 14), torch.float32)],
        {
            "model_names": [
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((184,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": [
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add114,
        [((184,), torch.float32)],
        {
            "model_names": [
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 184, 14, 14), torch.float32), ((184, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add13,
        [((1, 184, 14, 14), torch.float32)],
        {
            "model_names": [
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add13,
        [((1, 480, 14, 14), torch.float32)],
        {
            "model_names": [
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add13,
        [((1, 672, 14, 14), torch.float32)],
        {
            "model_names": [
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add13,
        [((1, 672, 7, 7), torch.float32)],
        {
            "model_names": [
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add13,
        [((1, 960, 7, 7), torch.float32)],
        {
            "model_names": [
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add45,
        [((1, 1280), torch.float32)],
        {"model_names": ["pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Add13,
        [((1, 1280), torch.float32)],
        {"model_names": ["pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 64, 96, 320), torch.float32), ((64, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 64, 48, 160), torch.float32), ((64, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 64, 48, 160), torch.float32), ((1, 64, 48, 160), torch.float32)],
        {
            "model_names": [
                "pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 128, 24, 80), torch.float32), ((128, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 128, 24, 80), torch.float32), ((1, 128, 24, 80), torch.float32)],
        {
            "model_names": [
                "pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 256, 12, 40), torch.float32), ((256, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 256, 12, 40), torch.float32), ((1, 256, 12, 40), torch.float32)],
        {
            "model_names": [
                "pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 512, 6, 20), torch.float32), ((512, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 512, 6, 20), torch.float32), ((1, 512, 6, 20), torch.float32)],
        {
            "model_names": [
                "pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 256, 6, 20), torch.float32), ((256, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 128, 12, 40), torch.float32), ((128, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 64, 24, 80), torch.float32), ((64, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 32, 48, 160), torch.float32), ((32, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 32, 96, 320), torch.float32), ((32, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 16, 96, 320), torch.float32), ((16, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 16, 192, 640), torch.float32), ((16, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1, 192, 640), torch.float32), ((1, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (Add20, [((1024, 512), torch.float32)], {"model_names": ["pt_nbeats_generic_basis_clm_hf"], "pcc": 0.99}),
    (Add39, [((1024, 96), torch.float32)], {"model_names": ["pt_nbeats_generic_basis_clm_hf"], "pcc": 0.99}),
    (
        Add0,
        [((1024, 1), torch.float32), ((1024, 24), torch.float32)],
        {
            "model_names": [
                "pt_nbeats_generic_basis_clm_hf",
                "pt_nbeats_trend_basis_clm_hf",
                "pt_nbeats_seasionality_basis_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1024, 24), torch.float32), ((1024, 24), torch.float32)],
        {
            "model_names": [
                "pt_nbeats_generic_basis_clm_hf",
                "pt_nbeats_trend_basis_clm_hf",
                "pt_nbeats_seasionality_basis_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 32, 2048), torch.float32), ((1, 32, 2048), torch.float32)],
        {"model_names": ["pt_opt_facebook_opt_1_3b_seq_cls_hf", "pt_opt_facebook_opt_1_3b_qa_hf"], "pcc": 0.99},
    ),
    (
        Add22,
        [((1, 32, 2048), torch.float32)],
        {"model_names": ["pt_opt_facebook_opt_1_3b_seq_cls_hf", "pt_opt_facebook_opt_1_3b_qa_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 32, 32, 32), torch.float32), ((1, 1, 32, 32), torch.float32)],
        {"model_names": ["pt_opt_facebook_opt_1_3b_seq_cls_hf", "pt_opt_facebook_opt_1_3b_qa_hf"], "pcc": 0.99},
    ),
    (
        Add64,
        [((32, 8192), torch.float32)],
        {"model_names": ["pt_opt_facebook_opt_1_3b_seq_cls_hf", "pt_opt_facebook_opt_1_3b_qa_hf"], "pcc": 0.99},
    ),
    (
        Add22,
        [((32, 2048), torch.float32)],
        {"model_names": ["pt_opt_facebook_opt_1_3b_seq_cls_hf", "pt_opt_facebook_opt_1_3b_qa_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((32, 2048), torch.float32), ((32, 2048), torch.float32)],
        {"model_names": ["pt_opt_facebook_opt_1_3b_seq_cls_hf", "pt_opt_facebook_opt_1_3b_qa_hf"], "pcc": 0.99},
    ),
    (
        Add115,
        [((1, 512, 261), torch.float32)],
        {"model_names": ["pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add115,
        [((1, 50176, 261), torch.float32)],
        {"model_names": ["pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add116,
        [((1, 1, 512, 50176), torch.float32)],
        {
            "model_names": [
                "pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 96, 56, 56), torch.float32), ((1, 96, 56, 56), torch.float32)],
        {
            "model_names": [
                "pt_regnet_regnet_x_3_2gf_img_cls_torchvision",
                "pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 192, 28, 28), torch.float32), ((1, 192, 28, 28), torch.float32)],
        {
            "model_names": [
                "pt_regnet_regnet_x_3_2gf_img_cls_torchvision",
                "pt_regnet_facebook_regnet_y_040_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((432,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": [
                "pt_regnet_regnet_x_3_2gf_img_cls_torchvision",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add117,
        [((432,), torch.float32)],
        {
            "model_names": [
                "pt_regnet_regnet_x_3_2gf_img_cls_torchvision",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 432, 14, 14), torch.float32), ((432, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_regnet_regnet_x_3_2gf_img_cls_torchvision",
                "onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 432, 28, 28), torch.float32), ((432, 1, 1), torch.float32)],
        {"model_names": ["pt_regnet_regnet_x_3_2gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 432, 14, 14), torch.float32), ((1, 432, 14, 14), torch.float32)],
        {"model_names": ["pt_regnet_regnet_x_3_2gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1008,), torch.float32), ((1,), torch.float32)],
        {"model_names": ["pt_regnet_regnet_x_3_2gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add118,
        [((1008,), torch.float32)],
        {"model_names": ["pt_regnet_regnet_x_3_2gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1008, 7, 7), torch.float32), ((1008, 1, 1), torch.float32)],
        {"model_names": ["pt_regnet_regnet_x_3_2gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1008, 14, 14), torch.float32), ((1008, 1, 1), torch.float32)],
        {"model_names": ["pt_regnet_regnet_x_3_2gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1008, 7, 7), torch.float32), ((1, 1008, 7, 7), torch.float32)],
        {"model_names": ["pt_regnet_regnet_x_3_2gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 64, 1, 1), torch.float32), ((64, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_regnet_regnet_y_800mf_img_cls_torchvision",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 16, 1, 1), torch.float32), ((16, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_regnet_regnet_y_800mf_img_cls_torchvision",
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2a_img_cls_timm",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "onnx_efficientnet_efficientnet_b5_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_ssd300_resnet50_base_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 144, 28, 28), torch.float32), ((1, 144, 28, 28), torch.float32)],
        {"model_names": ["pt_regnet_regnet_y_800mf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 36, 1, 1), torch.float32), ((36, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_regnet_regnet_y_800mf_img_cls_torchvision",
                "pt_regnet_facebook_regnet_y_064_img_cls_hf",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 320, 28, 28), torch.float32), ((320, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_regnet_regnet_y_800mf_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 320, 1, 1), torch.float32), ((320, 1, 1), torch.float32)],
        {"model_names": ["pt_regnet_regnet_y_800mf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 320, 14, 14), torch.float32), ((1, 320, 14, 14), torch.float32)],
        {"model_names": ["pt_regnet_regnet_y_800mf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 80, 1, 1), torch.float32), ((80, 1, 1), torch.float32)],
        {"model_names": ["pt_regnet_regnet_y_800mf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((784,), torch.float32), ((1,), torch.float32)],
        {"model_names": ["pt_regnet_regnet_y_800mf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (Add119, [((784,), torch.float32)], {"model_names": ["pt_regnet_regnet_y_800mf_img_cls_torchvision"], "pcc": 0.99}),
    (
        Add0,
        [((1, 784, 7, 7), torch.float32), ((784, 1, 1), torch.float32)],
        {"model_names": ["pt_regnet_regnet_y_800mf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 784, 14, 14), torch.float32), ((784, 1, 1), torch.float32)],
        {"model_names": ["pt_regnet_regnet_y_800mf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 784, 1, 1), torch.float32), ((784, 1, 1), torch.float32)],
        {"model_names": ["pt_regnet_regnet_y_800mf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 784, 7, 7), torch.float32), ((1, 784, 7, 7), torch.float32)],
        {"model_names": ["pt_regnet_regnet_y_800mf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 196, 1, 1), torch.float32), ((196, 1, 1), torch.float32)],
        {"model_names": ["pt_regnet_regnet_y_800mf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 768, 1, 128), torch.float32), ((768, 1, 1), torch.float32)],
        {"model_names": ["pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 768, 128), torch.float32), ((768, 1), torch.float32)],
        {"model_names": ["pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 768, 128), torch.float32), ((1, 768, 128), torch.float32)],
        {"model_names": ["pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 3072, 1, 128), torch.float32), ((3072, 1, 1), torch.float32)],
        {"model_names": ["pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Add120,
        [((1, 3), torch.float32)],
        {
            "model_names": [
                "pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf",
                "pt_autoencoder_linear_img_enc_github",
                "pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 12, 1, 1), torch.float32), ((1, 12, 1, 1), torch.float32)],
        {"model_names": ["pt_t5_t5_base_text_gen_hf", "pt_t5_google_flan_t5_base_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1, 768), torch.float32), ((1, 1, 768), torch.float32)],
        {
            "model_names": [
                "pt_t5_t5_base_text_gen_hf",
                "pt_whisper_openai_whisper_small_speech_recognition_hf",
                "pt_t5_google_flan_t5_base_text_gen_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add78,
        [((1, 12, 61, 61), torch.float32)],
        {"model_names": ["pt_t5_t5_base_text_gen_hf", "pt_t5_google_flan_t5_base_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 12, 61, 61), torch.float32), ((1, 12, 61, 61), torch.float32)],
        {"model_names": ["pt_t5_t5_base_text_gen_hf", "pt_t5_google_flan_t5_base_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 61, 768), torch.float32), ((1, 61, 768), torch.float32)],
        {"model_names": ["pt_t5_t5_base_text_gen_hf", "pt_t5_google_flan_t5_base_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Add121,
        [((1, 12, 1, 61), torch.float32)],
        {"model_names": ["pt_t5_t5_base_text_gen_hf", "pt_t5_google_flan_t5_base_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 4096, 1, 1), torch.float32), ((4096, 1, 1), torch.float32)],
        {"model_names": ["pt_vgg_vgg19_bn_obj_det_timm"], "pcc": 0.99},
    ),
    (Add77, [((197, 1, 2304), torch.float32)], {"model_names": ["pt_vit_vit_b_16_img_cls_torchvision"], "pcc": 0.99}),
    (
        Add122,
        [((1, 12, 197, 197), torch.float32)],
        {"model_names": ["pt_vit_vit_b_16_img_cls_torchvision"], "pcc": 0.99},
    ),
    (Add14, [((197, 768), torch.float32)], {"model_names": ["pt_vit_vit_b_16_img_cls_torchvision"], "pcc": 0.99}),
    (
        Add0,
        [((768,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": [
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_vovnet_ese_vovnet39b_obj_det_torchhub",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_yolox_yolox_m_obj_det_torchhub",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_vovnet_vovnet39_img_cls_osmr",
                "pt_vovnet_vovnet_v1_57_obj_det_torchhub",
                "pd_densenet_121_img_cls_paddlemodels",
                "pt_vovnet_vovnet57_img_cls_osmr",
                "pt_yolo_v6_yolov6m_obj_det_torchhub",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add14,
        [((768,), torch.float32)],
        {
            "model_names": [
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_vovnet_ese_vovnet39b_obj_det_torchhub",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_yolox_yolox_m_obj_det_torchhub",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_vovnet_vovnet39_img_cls_osmr",
                "pt_vovnet_vovnet_v1_57_obj_det_torchhub",
                "pd_densenet_121_img_cls_paddlemodels",
                "pt_vovnet_vovnet57_img_cls_osmr",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add14,
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
        Add123,
        [((1, 1500, 768), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_small_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Add14,
        [((1, 1500, 768), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_small_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1500, 768), torch.float32), ((1, 1500, 768), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_small_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Add15,
        [((1, 1500, 3072), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_small_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Add15,
        [((1, 1, 3072), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_small_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 256, 75, 75), torch.float32), ((1, 256, 75, 75), torch.float32)],
        {
            "model_names": [
                "pt_xception_xception71_tf_in1k_img_cls_timm",
                "pt_ssd300_resnet50_base_img_cls_torchhub",
                "pt_xception_xception71_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 728, 38, 38), torch.float32), ((1, 728, 38, 38), torch.float32)],
        {
            "model_names": ["pt_xception_xception71_tf_in1k_img_cls_timm", "pt_xception_xception71_img_cls_timm"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 80, 160, 160), torch.float32), ((80, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5x_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_640x640",
                "pt_yolox_yolox_x_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 160, 80, 80), torch.float32), ((160, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5x_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_640x640",
                "pt_yolox_yolox_x_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 80, 80, 80), torch.float32), ((80, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5x_img_cls_torchhub_320x320",
                "pt_yolox_yolox_x_obj_det_torchhub",
                "pt_yolov8_default_obj_det_github",
                "onnx_yolov10_default_obj_det_github",
                "pt_yolov9_default_obj_det_github",
                "pt_yolox_yolox_m_obj_det_torchhub",
                "pt_yolox_yolox_s_obj_det_torchhub",
                "onnx_yolov8_default_obj_det_github",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 80, 80, 80), torch.float32), ((1, 80, 80, 80), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5x_img_cls_torchhub_320x320"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 320, 40, 40), torch.float32), ((320, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5x_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_640x640",
                "pt_yolox_yolox_x_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 160, 40, 40), torch.float32), ((160, 1, 1), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5x_img_cls_torchhub_320x320"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 160, 40, 40), torch.float32), ((1, 160, 40, 40), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5x_img_cls_torchhub_320x320"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 640, 20, 20), torch.float32), ((640, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5x_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_640x640",
                "pt_yolox_yolox_x_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 320, 20, 20), torch.float32), ((320, 1, 1), torch.float32)],
        {
            "model_names": ["pt_yolo_v5_yolov5x_img_cls_torchhub_320x320", "pt_yolox_yolox_x_obj_det_torchhub"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 320, 20, 20), torch.float32), ((1, 320, 20, 20), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5x_img_cls_torchhub_320x320"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 640, 10, 10), torch.float32), ((640, 1, 1), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5x_img_cls_torchhub_320x320"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 640, 10, 10), torch.float32), ((1, 640, 10, 10), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5x_img_cls_torchhub_320x320"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 80, 320, 320), torch.float32), ((80, 1, 1), torch.float32)],
        {
            "model_names": ["pt_yolo_v5_yolov5x_img_cls_torchhub_640x640", "pt_yolox_yolox_x_obj_det_torchhub"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 160, 160, 160), torch.float32), ((160, 1, 1), torch.float32)],
        {
            "model_names": ["pt_yolo_v5_yolov5x_img_cls_torchhub_640x640", "pt_yolox_yolox_x_obj_det_torchhub"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 80, 160, 160), torch.float32), ((1, 80, 160, 160), torch.float32)],
        {
            "model_names": ["pt_yolo_v5_yolov5x_img_cls_torchhub_640x640", "pt_yolox_yolox_x_obj_det_torchhub"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 320, 80, 80), torch.float32), ((320, 1, 1), torch.float32)],
        {
            "model_names": ["pt_yolo_v5_yolov5x_img_cls_torchhub_640x640", "pt_yolox_yolox_x_obj_det_torchhub"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 160, 80, 80), torch.float32), ((1, 160, 80, 80), torch.float32)],
        {
            "model_names": ["pt_yolo_v5_yolov5x_img_cls_torchhub_640x640", "pt_yolox_yolox_x_obj_det_torchhub"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 640, 40, 40), torch.float32), ((640, 1, 1), torch.float32)],
        {
            "model_names": ["pt_yolo_v5_yolov5x_img_cls_torchhub_640x640", "pt_yolox_yolox_x_obj_det_torchhub"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 320, 40, 40), torch.float32), ((1, 320, 40, 40), torch.float32)],
        {
            "model_names": ["pt_yolo_v5_yolov5x_img_cls_torchhub_640x640", "pt_yolox_yolox_x_obj_det_torchhub"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1280, 20, 20), torch.float32), ((1280, 1, 1), torch.float32)],
        {
            "model_names": ["pt_yolo_v5_yolov5x_img_cls_torchhub_640x640", "pt_yolox_yolox_x_obj_det_torchhub"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 640, 20, 20), torch.float32), ((1, 640, 20, 20), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5x_img_cls_torchhub_640x640"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 32, 224, 320), torch.float32), ((32, 1, 1), torch.float32)],
        {"model_names": ["pt_yolo_v6_yolov6s_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 32, 224, 320), torch.float32), ((1, 32, 224, 320), torch.float32)],
        {"model_names": ["pt_yolo_v6_yolov6s_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 64, 112, 160), torch.float32), ((64, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v6_yolov6s_obj_det_torchhub",
                "pt_yolo_v6_yolov6l_obj_det_torchhub",
                "pt_yolo_v6_yolov6m_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 64, 112, 160), torch.float32), ((1, 64, 112, 160), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v6_yolov6s_obj_det_torchhub",
                "pt_yolo_v6_yolov6l_obj_det_torchhub",
                "pt_yolo_v6_yolov6m_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 128, 56, 80), torch.float32), ((128, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v6_yolov6s_obj_det_torchhub",
                "pt_yolo_v6_yolov6l_obj_det_torchhub",
                "pt_yolo_v6_yolov6m_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 128, 56, 80), torch.float32), ((1, 128, 56, 80), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v6_yolov6s_obj_det_torchhub",
                "pt_yolo_v6_yolov6l_obj_det_torchhub",
                "pt_yolo_v6_yolov6m_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 256, 28, 40), torch.float32), ((256, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v6_yolov6s_obj_det_torchhub",
                "pt_yolo_v6_yolov6l_obj_det_torchhub",
                "pt_yolo_v6_yolov6m_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 256, 28, 40), torch.float32), ((1, 256, 28, 40), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v6_yolov6s_obj_det_torchhub",
                "pt_yolo_v6_yolov6l_obj_det_torchhub",
                "pt_yolo_v6_yolov6m_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add124,
        [((512,), torch.float32)],
        {"model_names": ["pt_yolo_v6_yolov6s_obj_det_torchhub", "pt_yolo_v6_yolov6m_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 512, 14, 20), torch.float32), ((512, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v6_yolov6s_obj_det_torchhub",
                "pt_yolo_v6_yolov6l_obj_det_torchhub",
                "pt_yolo_v6_yolov6m_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 512, 14, 20), torch.float32), ((1, 512, 14, 20), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v6_yolov6s_obj_det_torchhub",
                "pt_yolo_v6_yolov6l_obj_det_torchhub",
                "pt_yolo_v6_yolov6m_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((640,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": [
                "pt_yolox_yolox_x_obj_det_torchhub",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add125,
        [((640,), torch.float32)],
        {
            "model_names": [
                "pt_yolox_yolox_x_obj_det_torchhub",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 4, 80, 80), torch.float32), ((4, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_yolox_yolox_x_obj_det_torchhub",
                "pt_yolox_yolox_m_obj_det_torchhub",
                "pt_yolox_yolox_s_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1, 80, 80), torch.float32), ((1, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_yolox_yolox_x_obj_det_torchhub",
                "pt_yolox_yolox_m_obj_det_torchhub",
                "pt_yolox_yolox_s_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 4, 40, 40), torch.float32), ((4, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_yolox_yolox_x_obj_det_torchhub",
                "pt_yolox_yolox_m_obj_det_torchhub",
                "pt_yolox_yolox_s_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1, 40, 40), torch.float32), ((1, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_yolox_yolox_x_obj_det_torchhub",
                "pt_yolox_yolox_m_obj_det_torchhub",
                "pt_yolox_yolox_s_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 80, 40, 40), torch.float32), ((80, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_yolox_yolox_x_obj_det_torchhub",
                "pt_yolov8_default_obj_det_github",
                "onnx_yolov10_default_obj_det_github",
                "pt_yolov9_default_obj_det_github",
                "pt_yolox_yolox_m_obj_det_torchhub",
                "pt_yolox_yolox_s_obj_det_torchhub",
                "onnx_yolov8_default_obj_det_github",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 4, 20, 20), torch.float32), ((4, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_yolox_yolox_x_obj_det_torchhub",
                "pt_yolox_yolox_m_obj_det_torchhub",
                "pt_yolox_yolox_s_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1, 20, 20), torch.float32), ((1, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_yolox_yolox_x_obj_det_torchhub",
                "pt_yolox_yolox_m_obj_det_torchhub",
                "pt_yolox_yolox_s_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 80, 20, 20), torch.float32), ((80, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_yolox_yolox_x_obj_det_torchhub",
                "pt_yolov8_default_obj_det_github",
                "onnx_yolov10_default_obj_det_github",
                "pt_yolov9_default_obj_det_github",
                "pt_yolox_yolox_m_obj_det_torchhub",
                "pt_yolox_yolox_s_obj_det_torchhub",
                "onnx_yolov8_default_obj_det_github",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 112, 109, 64), torch.float32), ((64,), torch.float32)],
        {"model_names": ["tf_resnet_resnet50_img_cls_keras"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 56, 54, 256), torch.float32), ((256,), torch.float32)],
        {"model_names": ["tf_resnet_resnet50_img_cls_keras"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 56, 54, 64), torch.float32), ((64,), torch.float32)],
        {"model_names": ["tf_resnet_resnet50_img_cls_keras"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 56, 54, 256), torch.float32), ((1, 56, 54, 256), torch.float32)],
        {"model_names": ["tf_resnet_resnet50_img_cls_keras"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 28, 27, 512), torch.float32), ((512,), torch.float32)],
        {"model_names": ["tf_resnet_resnet50_img_cls_keras"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 28, 27, 128), torch.float32), ((128,), torch.float32)],
        {"model_names": ["tf_resnet_resnet50_img_cls_keras"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 28, 27, 512), torch.float32), ((1, 28, 27, 512), torch.float32)],
        {"model_names": ["tf_resnet_resnet50_img_cls_keras"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 14, 14, 1024), torch.float32), ((1024,), torch.float32)],
        {"model_names": ["tf_resnet_resnet50_img_cls_keras"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 14, 14, 256), torch.float32), ((256,), torch.float32)],
        {"model_names": ["tf_resnet_resnet50_img_cls_keras"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 14, 14, 1024), torch.float32), ((1, 14, 14, 1024), torch.float32)],
        {"model_names": ["tf_resnet_resnet50_img_cls_keras", "jax_resnet_50_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 7, 7, 2048), torch.float32), ((2048,), torch.float32)],
        {"model_names": ["tf_resnet_resnet50_img_cls_keras"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 7, 7, 512), torch.float32), ((512,), torch.float32)],
        {"model_names": ["tf_resnet_resnet50_img_cls_keras"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 7, 7, 2048), torch.float32), ((1, 7, 7, 2048), torch.float32)],
        {"model_names": ["tf_resnet_resnet50_img_cls_keras", "jax_resnet_50_img_cls_hf"], "pcc": 0.99},
    ),
    (Add13, [((1, 1, 1, 64), torch.float32)], {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99}),
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
    (Add13, [((1, 1, 1, 256), torch.float32)], {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99}),
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
    (Add13, [((1, 1, 1, 128), torch.float32)], {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99}),
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
    (Add13, [((1, 1, 1, 512), torch.float32)], {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99}),
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
    (Add13, [((1, 1, 1, 1024), torch.float32)], {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99}),
    (
        Add0,
        [((1, 14, 14, 1024), torch.float32), ((1, 1, 1, 1024), torch.float32)],
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
    (Add13, [((1, 1, 1, 2048), torch.float32)], {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99}),
    (
        Add0,
        [((1, 7, 7, 2048), torch.float32), ((1, 1, 1, 2048), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1000), torch.float32), ((1, 1000), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add126,
        [((1, 100, 251), torch.float32)],
        {"model_names": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 32, 107, 160), torch.float32), ((32, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
                "pt_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 64, 54, 80), torch.float32), ((64, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
                "pt_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 128, 27, 40), torch.float32), ((128, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
                "pt_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add127,
        [((1, 100, 8, 14, 20), torch.float32)],
        {"model_names": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((100, 264, 14, 20), torch.float32), ((264, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
                "pt_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add13,
        [((100, 8, 1), torch.float32)],
        {"model_names": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Add128,
        [((100, 8, 9240), torch.float32)],
        {"model_names": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Add129,
        [((100, 264, 14, 20), torch.float32)],
        {"model_names": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((100, 128, 14, 20), torch.float32), ((128, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
                "pt_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add128,
        [((100, 8, 4480), torch.float32)],
        {"model_names": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Add130,
        [((100, 128, 14, 20), torch.float32)],
        {"model_names": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((100, 128, 27, 40), torch.float32), ((100, 128, 27, 40), torch.float32)],
        {
            "model_names": [
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
                "pt_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((100, 64, 27, 40), torch.float32), ((64, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
                "pt_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add128,
        [((100, 8, 8640), torch.float32)],
        {"model_names": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Add131,
        [((100, 64, 27, 40), torch.float32)],
        {"model_names": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((100, 64, 54, 80), torch.float32), ((100, 64, 54, 80), torch.float32)],
        {
            "model_names": [
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
                "pt_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((100, 32, 54, 80), torch.float32), ((32, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
                "pt_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add128,
        [((100, 8, 17280), torch.float32)],
        {"model_names": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Add132,
        [((100, 32, 54, 80), torch.float32)],
        {"model_names": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((100, 32, 107, 160), torch.float32), ((100, 32, 107, 160), torch.float32)],
        {
            "model_names": [
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
                "pt_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((100, 16, 107, 160), torch.float32), ((16, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
                "pt_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add128,
        [((100, 8, 34240), torch.float32)],
        {"model_names": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Add133,
        [((100, 16, 107, 160), torch.float32)],
        {"model_names": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((100, 1, 107, 160), torch.float32), ((1, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
                "pt_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 16, 128, 128), torch.float32), ((1, 1, 1, 128), torch.float32)],
        {
            "model_names": [
                "pt_albert_large_v1_token_cls_hf",
                "pt_albert_large_v2_mlm_hf",
                "pt_albert_large_v2_token_cls_hf",
                "pt_albert_xlarge_v1_mlm_hf",
                "pt_albert_large_v1_mlm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add19,
        [((1, 128), torch.float32)],
        {"model_names": ["pt_autoencoder_linear_img_enc_github", "pt_mnist_base_img_cls_github"], "pcc": 0.99},
    ),
    (Add17, [((1, 64), torch.float32)], {"model_names": ["pt_autoencoder_linear_img_enc_github"], "pcc": 0.99}),
    (Add107, [((1, 12), torch.float32)], {"model_names": ["pt_autoencoder_linear_img_enc_github"], "pcc": 0.99}),
    (Add119, [((1, 784), torch.float32)], {"model_names": ["pt_autoencoder_linear_img_enc_github"], "pcc": 0.99}),
    (
        Add0,
        [((1, 16, 197, 197), torch.float32), ((1, 16, 197, 197), torch.float32)],
        {"model_names": ["pt_beit_microsoft_beit_large_patch16_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 384, 1024), torch.float32), ((1, 384, 1024), torch.float32)],
        {
            "model_names": [
                "pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf",
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                "onnx_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add21,
        [((1, 384, 1024), torch.float32)],
        {
            "model_names": [
                "pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf",
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add134,
        [((1, 16, 384, 384), torch.float32)],
        {
            "model_names": [
                "pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf",
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                "onnx_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add24,
        [((1, 384, 4096), torch.float32)],
        {
            "model_names": [
                "pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf",
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((384, 1), torch.float32), ((1,), torch.float32)],
        {
            "model_names": [
                "pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf",
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                "pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf",
                "onnx_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add135,
        [((1, 197, 192), torch.float32)],
        {"model_names": ["pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add36,
        [((1, 197, 192), torch.float32)],
        {"model_names": ["pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 197, 192), torch.float32), ((1, 197, 192), torch.float32)],
        {"model_names": ["pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 200, 7, 7), torch.float32), ((200, 1, 1), torch.float32)],
        {"model_names": ["pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 184, 7, 7), torch.float32), ((184, 1, 1), torch.float32)],
        {"model_names": ["pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 960, 3, 3), torch.float32), ((960, 1, 1), torch.float32)],
        {"model_names": ["pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 256, 2048), torch.float32), ((1, 256, 2048), torch.float32)],
        {
            "model_names": [
                "pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf",
                "pt_xglm_facebook_xglm_1_7b_clm_hf",
                "pt_phi1_microsoft_phi_1_seq_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add136,
        [((1, 1, 1, 256), torch.float32)],
        {
            "model_names": [
                "pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf",
                "pt_gptneo_eleutherai_gpt_neo_125m_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf",
                "pt_phi2_microsoft_phi_2_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add22,
        [((1, 256, 2048), torch.float32)],
        {
            "model_names": [
                "pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_xglm_facebook_xglm_1_7b_clm_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_phi1_microsoft_phi_1_seq_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add64,
        [((1, 256, 8192), torch.float32)],
        {
            "model_names": [
                "pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf",
                "pt_xglm_facebook_xglm_1_7b_clm_hf",
                "pt_phi1_microsoft_phi_1_seq_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((44,), torch.float32), ((1,), torch.float32)],
        {"model_names": ["pt_hrnet_hrnetv2_w44_pose_estimation_osmr"], "pcc": 0.99},
    ),
    (Add137, [((44,), torch.float32)], {"model_names": ["pt_hrnet_hrnetv2_w44_pose_estimation_osmr"], "pcc": 0.99}),
    (
        Add0,
        [((1, 44, 56, 56), torch.float32), ((44, 1, 1), torch.float32)],
        {"model_names": ["pt_hrnet_hrnetv2_w44_pose_estimation_osmr"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 44, 56, 56), torch.float32), ((1, 44, 56, 56), torch.float32)],
        {"model_names": ["pt_hrnet_hrnetv2_w44_pose_estimation_osmr"], "pcc": 0.99},
    ),
    (
        Add0,
        [((88,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": [
                "pt_hrnet_hrnetv2_w44_pose_estimation_osmr",
                "pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add138,
        [((88,), torch.float32)],
        {
            "model_names": [
                "pt_hrnet_hrnetv2_w44_pose_estimation_osmr",
                "pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 88, 28, 28), torch.float32), ((88, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_hrnet_hrnetv2_w44_pose_estimation_osmr",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 88, 28, 28), torch.float32), ((1, 88, 28, 28), torch.float32)],
        {"model_names": ["pt_hrnet_hrnetv2_w44_pose_estimation_osmr"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 44, 28, 28), torch.float32), ((44, 1, 1), torch.float32)],
        {"model_names": ["pt_hrnet_hrnetv2_w44_pose_estimation_osmr"], "pcc": 0.99},
    ),
    (
        Add0,
        [((176,), torch.float32), ((1,), torch.float32)],
        {"model_names": ["pt_hrnet_hrnetv2_w44_pose_estimation_osmr"], "pcc": 0.99},
    ),
    (Add139, [((176,), torch.float32)], {"model_names": ["pt_hrnet_hrnetv2_w44_pose_estimation_osmr"], "pcc": 0.99}),
    (
        Add0,
        [((1, 176, 14, 14), torch.float32), ((176, 1, 1), torch.float32)],
        {"model_names": ["pt_hrnet_hrnetv2_w44_pose_estimation_osmr"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 176, 14, 14), torch.float32), ((1, 176, 14, 14), torch.float32)],
        {"model_names": ["pt_hrnet_hrnetv2_w44_pose_estimation_osmr"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 44, 14, 14), torch.float32), ((44, 1, 1), torch.float32)],
        {"model_names": ["pt_hrnet_hrnetv2_w44_pose_estimation_osmr"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 88, 14, 14), torch.float32), ((88, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_hrnet_hrnetv2_w44_pose_estimation_osmr",
                "onnx_mobilenetv2_mobilenetv2_140_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((352,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": [
                "pt_hrnet_hrnetv2_w44_pose_estimation_osmr",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
                "pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add140,
        [((352,), torch.float32)],
        {
            "model_names": [
                "pt_hrnet_hrnetv2_w44_pose_estimation_osmr",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
                "pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 352, 7, 7), torch.float32), ((352, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_hrnet_hrnetv2_w44_pose_estimation_osmr",
                "onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 352, 7, 7), torch.float32), ((1, 352, 7, 7), torch.float32)],
        {"model_names": ["pt_hrnet_hrnetv2_w44_pose_estimation_osmr"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 308, 7, 7), torch.float32), ((308, 1, 1), torch.float32)],
        {"model_names": ["pt_hrnet_hrnetv2_w44_pose_estimation_osmr"], "pcc": 0.99},
    ),
    (
        Add5,
        [((1, 768, 384), torch.float32)],
        {
            "model_names": [
                "pt_mlp_mixer_mixer_b16_224_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_b32_224_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_goog_in21k_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add112,
        [((1, 768, 196), torch.float32)],
        {
            "model_names": [
                "pt_mlp_mixer_mixer_b16_224_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_goog_in21k_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 196, 768), torch.float32), ((1, 196, 768), torch.float32)],
        {
            "model_names": [
                "pt_mlp_mixer_mixer_b16_224_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_goog_in21k_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add15,
        [((1, 196, 3072), torch.float32)],
        {
            "model_names": [
                "pt_mlp_mixer_mixer_b16_224_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_goog_in21k_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add14,
        [((1, 196, 768), torch.float32)],
        {
            "model_names": [
                "pt_mlp_mixer_mixer_b16_224_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_goog_in21k_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 32, 26, 26), torch.float32), ((32, 1, 1), torch.float32)],
        {"model_names": ["pt_mnist_base_img_cls_github"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 64, 24, 24), torch.float32), ((64, 1, 1), torch.float32)],
        {"model_names": ["pt_mnist_base_img_cls_github"], "pcc": 0.99},
    ),
    (Add141, [((1, 10), torch.float32)], {"model_names": ["pt_mnist_base_img_cls_github"], "pcc": 0.99}),
    (
        Add0,
        [((1, 256, 80, 80), torch.float32), ((256, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_mobilenetv3_ssd_resnet101_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet152_img_cls_torchvision",
                "pt_yolov9_default_obj_det_github",
                "pt_mobilenetv3_ssd_resnet50_img_cls_torchvision",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_640x640",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 256, 80, 80), torch.float32), ((1, 256, 80, 80), torch.float32)],
        {
            "model_names": [
                "pt_mobilenetv3_ssd_resnet101_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet152_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet50_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 512, 40, 40), torch.float32), ((512, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_mobilenetv3_ssd_resnet101_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet152_img_cls_torchvision",
                "pt_yolov9_default_obj_det_github",
                "pt_mobilenetv3_ssd_resnet50_img_cls_torchvision",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_640x640",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 512, 40, 40), torch.float32), ((1, 512, 40, 40), torch.float32)],
        {
            "model_names": [
                "pt_mobilenetv3_ssd_resnet101_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet152_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet50_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1024, 20, 20), torch.float32), ((1024, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_mobilenetv3_ssd_resnet101_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet152_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet50_img_cls_torchvision",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_640x640",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1024, 20, 20), torch.float32), ((1, 1024, 20, 20), torch.float32)],
        {
            "model_names": [
                "pt_mobilenetv3_ssd_resnet101_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet152_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet50_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 2048, 10, 10), torch.float32), ((1, 2048, 10, 10), torch.float32)],
        {
            "model_names": [
                "pt_mobilenetv3_ssd_resnet101_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet152_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet50_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (Add18, [((1024, 256), torch.float32)], {"model_names": ["pt_nbeats_trend_basis_clm_hf"], "pcc": 0.99}),
    (Add106, [((1024, 8), torch.float32)], {"model_names": ["pt_nbeats_trend_basis_clm_hf"], "pcc": 0.99}),
    (
        Add18,
        [((1, 2048, 256), torch.float32)],
        {"model_names": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 2048, 768), torch.float32), ((2048, 768), torch.float32)],
        {"model_names": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 8, 256, 2048), torch.float32), ((1, 1, 1, 2048), torch.float32)],
        {"model_names": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"], "pcc": 0.99},
    ),
    (
        Add45,
        [((1, 2048, 1280), torch.float32)],
        {"model_names": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"], "pcc": 0.99},
    ),
    (
        Add45,
        [((1, 256, 1280), torch.float32)],
        {"model_names": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 256, 1280), torch.float32), ((1, 256, 1280), torch.float32)],
        {"model_names": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"], "pcc": 0.99},
    ),
    (
        Add14,
        [((1, 256, 768), torch.float32)],
        {
            "model_names": [
                "pt_perceiverio_deepmind_language_perceiver_mlm_hf",
                "pt_gptneo_eleutherai_gpt_neo_125m_clm_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add14,
        [((1, 2048, 768), torch.float32)],
        {"model_names": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 2048, 768), torch.float32), ((1, 2048, 768), torch.float32)],
        {"model_names": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"], "pcc": 0.99},
    ),
    (
        Add142,
        [((2048, 262), torch.float32)],
        {"model_names": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"], "pcc": 0.99},
    ),
    (
        Add13,
        [((1, 6, 1), torch.float32)],
        {
            "model_names": [
                "pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf",
                "pt_mamba_state_spaces_mamba_370m_hf_clm_hf",
                "pt_mamba_state_spaces_mamba_790m_hf_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (Add21, [((1, 6, 1024), torch.float32)], {"model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"], "pcc": 0.99}),
    (
        Add0,
        [((1, 16, 6, 64), torch.float32), ((1, 16, 6, 64), torch.float32)],
        {"model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"], "pcc": 0.99},
    ),
    (Add143, [((1, 16, 6, 6), torch.float32)], {"model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"], "pcc": 0.99}),
    (
        Add0,
        [((1, 6, 1024), torch.float32), ((1, 6, 1024), torch.float32)],
        {
            "model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf", "pt_mamba_state_spaces_mamba_370m_hf_clm_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 144, 112, 112), torch.float32), ((144, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_regnet_facebook_regnet_y_064_img_cls_hf",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "onnx_mobilenetv2_mobilenetv2_140_img_cls_timm",
                "onnx_efficientnet_efficientnet_b5_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 144, 56, 56), torch.float32), ((1, 144, 56, 56), torch.float32)],
        {"model_names": ["pt_regnet_facebook_regnet_y_064_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 288, 56, 56), torch.float32), ((288, 1, 1), torch.float32)],
        {"model_names": ["pt_regnet_facebook_regnet_y_064_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 288, 1, 1), torch.float32), ((288, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_regnet_facebook_regnet_y_064_img_cls_hf",
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 288, 28, 28), torch.float32), ((1, 288, 28, 28), torch.float32)],
        {"model_names": ["pt_regnet_facebook_regnet_y_064_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 576, 28, 28), torch.float32), ((576, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_regnet_facebook_regnet_y_064_img_cls_hf",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_regnet_regnet_y_3_2gf_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 576, 1, 1), torch.float32), ((576, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_regnet_facebook_regnet_y_064_img_cls_hf",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_regnet_regnet_y_3_2gf_img_cls_torchvision",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 576, 14, 14), torch.float32), ((1, 576, 14, 14), torch.float32)],
        {
            "model_names": [
                "pt_regnet_facebook_regnet_y_064_img_cls_hf",
                "pt_regnet_regnet_y_3_2gf_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1296,), torch.float32), ((1,), torch.float32)],
        {"model_names": ["pt_regnet_facebook_regnet_y_064_img_cls_hf"], "pcc": 0.99},
    ),
    (Add144, [((1296,), torch.float32)], {"model_names": ["pt_regnet_facebook_regnet_y_064_img_cls_hf"], "pcc": 0.99}),
    (
        Add0,
        [((1, 1296, 14, 14), torch.float32), ((1296, 1, 1), torch.float32)],
        {"model_names": ["pt_regnet_facebook_regnet_y_064_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1296, 7, 7), torch.float32), ((1296, 1, 1), torch.float32)],
        {"model_names": ["pt_regnet_facebook_regnet_y_064_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1296, 1, 1), torch.float32), ((1296, 1, 1), torch.float32)],
        {"model_names": ["pt_regnet_facebook_regnet_y_064_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1296, 7, 7), torch.float32), ((1, 1296, 7, 7), torch.float32)],
        {"model_names": ["pt_regnet_facebook_regnet_y_064_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 324, 1, 1), torch.float32), ((324, 1, 1), torch.float32)],
        {
            "model_names": ["pt_regnet_facebook_regnet_y_064_img_cls_hf", "pt_ssd300_resnet50_base_img_cls_torchhub"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 288, 14, 14), torch.float32), ((1, 288, 14, 14), torch.float32)],
        {"model_names": ["pt_regnet_regnet_x_800mf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 672, 7, 7), torch.float32), ((1, 672, 7, 7), torch.float32)],
        {"model_names": ["pt_regnet_regnet_x_800mf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add60,
        [((1, 128), torch.int64)],
        {
            "model_names": [
                "pt_roberta_xlm_roberta_base_mlm_hf",
                "pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (Add145, [((1, 128, 250002), torch.float32)], {"model_names": ["pt_roberta_xlm_roberta_base_mlm_hf"], "pcc": 0.99}),
    (
        Add0,
        [((1, 32, 128, 128), torch.float32), ((32, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_efficientnet_efficientnet_b2a_img_cls_timm",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add33,
        [((1, 16384, 32), torch.float32)],
        {
            "model_names": [
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
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add33,
        [((1, 256, 32), torch.float32)],
        {
            "model_names": [
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
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add19,
        [((1, 16384, 128), torch.float32)],
        {
            "model_names": [
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
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
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
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add17,
        [((1, 4096, 64), torch.float32)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add17,
        [((1, 256, 64), torch.float32)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 4096, 64), torch.float32), ((1, 4096, 64), torch.float32)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add18,
        [((1, 4096, 256), torch.float32)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 256, 64, 64), torch.float32), ((256, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pt_fpn_base_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 160, 32, 32), torch.float32), ((160, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add48,
        [((1, 1024, 160), torch.float32)],
        {
            "model_names": [
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
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add48,
        [((1, 256, 160), torch.float32)],
        {
            "model_names": [
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
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add125,
        [((1, 1024, 640), torch.float32)],
        {
            "model_names": [
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
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
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
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pt_fpn_base_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 256, 256), torch.float32), ((1, 256, 256), torch.float32)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
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
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add39,
        [((64, 49, 96), torch.float32)],
        {
            "model_names": [
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((64, 3, 49, 49), torch.float32), ((1, 3, 49, 49), torch.float32)],
        {
            "model_names": [
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 3136, 96), torch.float32), ((1, 3136, 96), torch.float32)],
        {"model_names": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add5,
        [((1, 3136, 384), torch.float32)],
        {"model_names": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add39,
        [((1, 3136, 96), torch.float32)],
        {"model_names": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add146,
        [((1, 64, 3, 49, 49), torch.float32)],
        {
            "model_names": [
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add36,
        [((16, 49, 192), torch.float32)],
        {
            "model_names": [
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((16, 6, 49, 49), torch.float32), ((1, 6, 49, 49), torch.float32)],
        {
            "model_names": [
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 784, 192), torch.float32), ((1, 784, 192), torch.float32)],
        {"model_names": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add14,
        [((1, 784, 768), torch.float32)],
        {"model_names": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add36,
        [((1, 784, 192), torch.float32)],
        {"model_names": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add147,
        [((1, 16, 6, 49, 49), torch.float32)],
        {
            "model_names": [
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add5,
        [((4, 49, 384), torch.float32)],
        {
            "model_names": [
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((4, 12, 49, 49), torch.float32), ((1, 12, 49, 49), torch.float32)],
        {
            "model_names": [
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 196, 384), torch.float32), ((1, 196, 384), torch.float32)],
        {"model_names": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add65,
        [((1, 196, 1536), torch.float32)],
        {"model_names": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add5,
        [((1, 196, 384), torch.float32)],
        {"model_names": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add148,
        [((1, 4, 12, 49, 49), torch.float32)],
        {
            "model_names": [
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add14,
        [((1, 49, 768), torch.float32)],
        {
            "model_names": [
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_mlp_mixer_mixer_b32_224_img_cls_timm",
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 24, 49, 49), torch.float32), ((1, 24, 49, 49), torch.float32)],
        {
            "model_names": [
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 49, 768), torch.float32), ((1, 49, 768), torch.float32)],
        {
            "model_names": [
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_mlp_mixer_mixer_b32_224_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add15,
        [((1, 49, 3072), torch.float32)],
        {
            "model_names": [
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_mlp_mixer_mixer_b32_224_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 8, 1, 1), torch.float32), ((1, 8, 1, 1), torch.float32)],
        {"model_names": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99},
    ),
    (Add78, [((1, 8, 61, 61), torch.float32)], {"model_names": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99}),
    (
        Add0,
        [((1, 8, 61, 61), torch.float32), ((1, 8, 61, 61), torch.float32)],
        {"model_names": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99},
    ),
    (Add149, [((1, 8, 1, 61), torch.float32)], {"model_names": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99}),
    (
        Add0,
        [((1, 1280, 37, 37), torch.float32), ((1280, 1, 1), torch.float32)],
        {"model_names": ["pt_vit_vit_h_14_img_cls_torchvision"], "pcc": 0.99},
    ),
    (Add150, [((1, 1370, 1280), torch.float32)], {"model_names": ["pt_vit_vit_h_14_img_cls_torchvision"], "pcc": 0.99}),
    (Add151, [((1370, 1, 3840), torch.float32)], {"model_names": ["pt_vit_vit_h_14_img_cls_torchvision"], "pcc": 0.99}),
    (
        Add152,
        [((1, 16, 1370, 1370), torch.float32)],
        {"model_names": ["pt_vit_vit_h_14_img_cls_torchvision"], "pcc": 0.99},
    ),
    (Add45, [((1370, 1280), torch.float32)], {"model_names": ["pt_vit_vit_h_14_img_cls_torchvision"], "pcc": 0.99}),
    (
        Add0,
        [((1, 1370, 1280), torch.float32), ((1, 1370, 1280), torch.float32)],
        {"model_names": ["pt_vit_vit_h_14_img_cls_torchvision"], "pcc": 0.99},
    ),
    (Add153, [((1, 1370, 5120), torch.float32)], {"model_names": ["pt_vit_vit_h_14_img_cls_torchvision"], "pcc": 0.99}),
    (Add45, [((1, 1370, 1280), torch.float32)], {"model_names": ["pt_vit_vit_h_14_img_cls_torchvision"], "pcc": 0.99}),
    (
        Add0,
        [((1, 768, 14, 14), torch.float32), ((1, 768, 14, 14), torch.float32)],
        {
            "model_names": [
                "pt_vovnet_ese_vovnet39b_obj_det_torchhub",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "onnx_vovnet_v1_vovnet39_obj_det_torchhub",
                "onnx_vovnet_vovnet_v1_57_obj_det_torchhub",
                "pt_vovnet_vovnet39_img_cls_osmr",
                "pt_vovnet_vovnet_v1_57_obj_det_torchhub",
                "pt_vovnet_vovnet57_img_cls_osmr",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 2, 1280), torch.float32), ((1, 2, 1280), torch.float32)],
        {
            "model_names": [
                "pt_whisper_openai_whisper_large_v3_clm_hf",
                "pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add45,
        [((1, 2, 1280), torch.float32)],
        {
            "model_names": [
                "pt_whisper_openai_whisper_large_v3_clm_hf",
                "pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add154,
        [((1, 20, 2, 2), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_large_v3_clm_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1280, 3000), torch.float32), ((1280, 1), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_large_v3_clm_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1280, 1500), torch.float32), ((1280, 1), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_large_v3_clm_hf"], "pcc": 0.99},
    ),
    (
        Add155,
        [((1, 1500, 1280), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_large_v3_clm_hf"], "pcc": 0.99},
    ),
    (
        Add45,
        [((1, 1500, 1280), torch.float32)],
        {
            "model_names": [
                "pt_whisper_openai_whisper_large_v3_clm_hf",
                "pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1500, 1280), torch.float32), ((1, 1500, 1280), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_large_v3_clm_hf"], "pcc": 0.99},
    ),
    (
        Add153,
        [((1, 1500, 5120), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_large_v3_clm_hf"], "pcc": 0.99},
    ),
    (
        Add153,
        [((1, 2, 5120), torch.float32)],
        {
            "model_names": [
                "pt_whisper_openai_whisper_large_v3_clm_hf",
                "pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 128, 147, 147), torch.float32), ((128, 1, 1), torch.float32)],
        {"model_names": ["pt_xception_xception_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 128, 74, 74), torch.float32), ((128, 1, 1), torch.float32)],
        {"model_names": ["pt_xception_xception_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 128, 74, 74), torch.float32), ((1, 128, 74, 74), torch.float32)],
        {"model_names": ["pt_xception_xception_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 256, 74, 74), torch.float32), ((256, 1, 1), torch.float32)],
        {"model_names": ["pt_xception_xception_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 256, 37, 37), torch.float32), ((256, 1, 1), torch.float32)],
        {"model_names": ["pt_xception_xception_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 256, 37, 37), torch.float32), ((1, 256, 37, 37), torch.float32)],
        {"model_names": ["pt_xception_xception_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 728, 37, 37), torch.float32), ((728, 1, 1), torch.float32)],
        {"model_names": ["pt_xception_xception_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 48, 240, 240), torch.float32), ((48, 1, 1), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5m_img_cls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 96, 120, 120), torch.float32), ((96, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5m_img_cls_torchhub_480x480",
                "pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 48, 120, 120), torch.float32), ((48, 1, 1), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5m_img_cls_torchhub_480x480", "TranslatedLayer"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 48, 120, 120), torch.float32), ((1, 48, 120, 120), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5m_img_cls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 192, 60, 60), torch.float32), ((192, 1, 1), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5m_img_cls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 96, 60, 60), torch.float32), ((96, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5m_img_cls_torchhub_480x480",
                "TranslatedLayer",
                "pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 96, 60, 60), torch.float32), ((1, 96, 60, 60), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5m_img_cls_torchhub_480x480", "TranslatedLayer"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 384, 30, 30), torch.float32), ((384, 1, 1), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5m_img_cls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 192, 30, 30), torch.float32), ((192, 1, 1), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5m_img_cls_torchhub_480x480", "TranslatedLayer"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 192, 30, 30), torch.float32), ((1, 192, 30, 30), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5m_img_cls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 768, 15, 15), torch.float32), ((768, 1, 1), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5m_img_cls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 384, 15, 15), torch.float32), ((384, 1, 1), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5m_img_cls_torchhub_480x480", "TranslatedLayer"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 384, 15, 15), torch.float32), ((1, 384, 15, 15), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5m_img_cls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 255, 60, 60), torch.float32), ((255, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5m_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_480x480",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 255, 60, 60), torch.float32), ((1, 255, 60, 60), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5m_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_480x480",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 255, 30, 30), torch.float32), ((255, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5m_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_480x480",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 255, 30, 30), torch.float32), ((1, 255, 30, 30), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5m_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_480x480",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 255, 15, 15), torch.float32), ((255, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5m_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_480x480",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 255, 15, 15), torch.float32), ((1, 255, 15, 15), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5m_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_480x480",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 192, 32, 42), torch.float32), ((192, 1, 1), torch.float32)],
        {"model_names": ["pt_yolos_hustvl_yolos_tiny_obj_det_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1445, 192), torch.float32), ((1, 1445, 192), torch.float32)],
        {"model_names": ["pt_yolos_hustvl_yolos_tiny_obj_det_hf"], "pcc": 0.99},
    ),
    (Add36, [((1, 1445, 192), torch.float32)], {"model_names": ["pt_yolos_hustvl_yolos_tiny_obj_det_hf"], "pcc": 0.99}),
    (Add14, [((1, 1445, 768), torch.float32)], {"model_names": ["pt_yolos_hustvl_yolos_tiny_obj_det_hf"], "pcc": 0.99}),
    (Add36, [((1, 100, 192), torch.float32)], {"model_names": ["pt_yolos_hustvl_yolos_tiny_obj_det_hf"], "pcc": 0.99}),
    (
        Add156,
        [((1, 100, 4), torch.float32)],
        {
            "model_names": [
                "pt_yolos_hustvl_yolos_tiny_obj_det_hf",
                "pt_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 16, 320, 320), torch.float32), ((16, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_yolov8_default_obj_det_github",
                "onnx_yolov10_default_obj_det_github",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_640x640",
                "onnx_yolov8_default_obj_det_github",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 16, 160, 160), torch.float32), ((16, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_yolov8_default_obj_det_github",
                "onnx_yolov10_default_obj_det_github",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_640x640",
                "onnx_yolov8_default_obj_det_github",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 16, 160, 160), torch.float32), ((1, 16, 160, 160), torch.float32)],
        {
            "model_names": [
                "pt_yolov8_default_obj_det_github",
                "onnx_yolov10_default_obj_det_github",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_640x640",
                "onnx_yolov8_default_obj_det_github",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 64, 20, 20), torch.float32), ((64, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_yolov8_default_obj_det_github",
                "onnx_yolov10_default_obj_det_github",
                "pt_yolov9_default_obj_det_github",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_320x320",
                "onnx_yolov8_default_obj_det_github",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add157,
        [((1, 2, 8400), torch.float32)],
        {
            "model_names": [
                "pt_yolov8_default_obj_det_github",
                "onnx_yolov10_default_obj_det_github",
                "pt_yolov9_default_obj_det_github",
                "onnx_yolov8_default_obj_det_github",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 2, 8400), torch.float32), ((1, 2, 8400), torch.float32)],
        {
            "model_names": [
                "pt_yolov8_default_obj_det_github",
                "onnx_yolov10_default_obj_det_github",
                "pt_yolov9_default_obj_det_github",
                "onnx_yolov8_default_obj_det_github",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 16, 240, 240), torch.float32), ((16, 1, 1), torch.float32)],
        {"model_names": ["TranslatedLayer", "pt_yolo_v5_yolov5n_img_cls_torchhub_480x480"], "pcc": 0.99},
    ),
    (Add158, [((1, 16, 240, 240), torch.float32)], {"model_names": ["TranslatedLayer"], "pcc": 0.99}),
    (Add13, [((1, 16, 240, 240), torch.float32)], {"model_names": ["TranslatedLayer"], "pcc": 0.99}),
    (
        Add0,
        [((1, 32, 240, 240), torch.float32), ((32, 1, 1), torch.float32)],
        {"model_names": ["TranslatedLayer"], "pcc": 0.99},
    ),
    (Add158, [((1, 32, 240, 240), torch.float32)], {"model_names": ["TranslatedLayer"], "pcc": 0.99}),
    (Add13, [((1, 32, 240, 240), torch.float32)], {"model_names": ["TranslatedLayer"], "pcc": 0.99}),
    (
        Add0,
        [((1, 32, 120, 120), torch.float32), ((32, 1, 1), torch.float32)],
        {
            "model_names": [
                "TranslatedLayer",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_480x480",
                "pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (Add158, [((1, 32, 120, 120), torch.float32)], {"model_names": ["TranslatedLayer"], "pcc": 0.99}),
    (Add158, [((1, 48, 120, 120), torch.float32)], {"model_names": ["TranslatedLayer"], "pcc": 0.99}),
    (Add13, [((1, 48, 120, 120), torch.float32)], {"model_names": ["TranslatedLayer"], "pcc": 0.99}),
    (
        Add0,
        [((1, 48, 60, 60), torch.float32), ((48, 1, 1), torch.float32)],
        {"model_names": ["TranslatedLayer"], "pcc": 0.99},
    ),
    (Add158, [((1, 48, 60, 60), torch.float32)], {"model_names": ["TranslatedLayer"], "pcc": 0.99}),
    (Add158, [((1, 96, 60, 60), torch.float32)], {"model_names": ["TranslatedLayer"], "pcc": 0.99}),
    (Add13, [((1, 96, 60, 60), torch.float32)], {"model_names": ["TranslatedLayer"], "pcc": 0.99}),
    (
        Add0,
        [((1, 96, 30, 30), torch.float32), ((96, 1, 1), torch.float32)],
        {"model_names": ["TranslatedLayer"], "pcc": 0.99},
    ),
    (Add158, [((1, 96, 30, 30), torch.float32)], {"model_names": ["TranslatedLayer"], "pcc": 0.99}),
    (Add158, [((1, 192, 30, 30), torch.float32)], {"model_names": ["TranslatedLayer"], "pcc": 0.99}),
    (Add13, [((1, 192, 30, 30), torch.float32)], {"model_names": ["TranslatedLayer"], "pcc": 0.99}),
    (
        Add0,
        [((1, 192, 15, 15), torch.float32), ((192, 1, 1), torch.float32)],
        {"model_names": ["TranslatedLayer"], "pcc": 0.99},
    ),
    (Add158, [((1, 192, 15, 15), torch.float32)], {"model_names": ["TranslatedLayer"], "pcc": 0.99}),
    (
        Add0,
        [((1, 192, 1, 1), torch.float32), ((192, 1, 1), torch.float32)],
        {
            "model_names": [
                "TranslatedLayer",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_regnet_facebook_regnet_y_040_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add13,
        [((1, 192, 1, 1), torch.float32)],
        {"model_names": ["TranslatedLayer", "pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"], "pcc": 0.99},
    ),
    (Add158, [((1, 384, 15, 15), torch.float32)], {"model_names": ["TranslatedLayer"], "pcc": 0.99}),
    (Add13, [((1, 384, 15, 15), torch.float32)], {"model_names": ["TranslatedLayer"], "pcc": 0.99}),
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
        Add13,
        [((1, 384, 1, 1), torch.float32)],
        {"model_names": ["TranslatedLayer", "pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 360, 15, 15), torch.float32), ((360, 1, 1), torch.float32)],
        {"model_names": ["TranslatedLayer"], "pcc": 0.99},
    ),
    (
        Add13,
        [((1, 96, 1, 1), torch.float32)],
        {
            "model_names": [
                "TranslatedLayer",
                "pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 96, 15, 15), torch.float32), ((1, 96, 15, 15), torch.float32)],
        {"model_names": ["TranslatedLayer"], "pcc": 0.99},
    ),
    (
        Add13,
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
        [((1, 24, 60, 60), torch.float32), ((1, 24, 60, 60), torch.float32)],
        {
            "model_names": ["TranslatedLayer", "pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm"],
            "pcc": 0.99,
        },
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
    (Add97, [((1, 128, 768), torch.float32)], {"model_names": ["onnx_bert_bert_base_uncased_mlm_hf"], "pcc": 0.99}),
    (Add159, [((1, 128, 3072), torch.float32)], {"model_names": ["onnx_bert_bert_base_uncased_mlm_hf"], "pcc": 0.99}),
    (Add160, [((1, 128, 30522), torch.float32)], {"model_names": ["onnx_bert_bert_base_uncased_mlm_hf"], "pcc": 0.99}),
    (
        Add0,
        [((1, 16, 128, 128), torch.float32), ((16, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 16, 128, 128), torch.float32), ((1, 16, 128, 128), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 96, 128, 128), torch.float32), ((96, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 24, 64, 64), torch.float32), ((24, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 144, 64, 64), torch.float32), ((144, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 144, 32, 32), torch.float32), ((144, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 48, 32, 32), torch.float32), ((48, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 288, 32, 32), torch.float32), ((288, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 48, 32, 32), torch.float32), ((1, 48, 32, 32), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 288, 16, 16), torch.float32), ((288, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 88, 16, 16), torch.float32), ((88, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 528, 16, 16), torch.float32), ((528, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 22, 1, 1), torch.float32), ((22, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 528, 1, 1), torch.float32), ((528, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 88, 16, 16), torch.float32), ((1, 88, 16, 16), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 120, 16, 16), torch.float32), ((120, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 720, 16, 16), torch.float32), ((720, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 30, 1, 1), torch.float32), ((30, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2a_img_cls_timm",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pt_regnet_regnet_y_1_6gf_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 720, 1, 1), torch.float32), ((720, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 120, 16, 16), torch.float32), ((1, 120, 16, 16), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 720, 8, 8), torch.float32), ((720, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 208, 8, 8), torch.float32), ((208, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1248, 8, 8), torch.float32), ((1248, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1248, 1, 1), torch.float32), ((1248, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 208, 8, 8), torch.float32), ((1, 208, 8, 8), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 352, 8, 8), torch.float32), ((352, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 2112, 8, 8), torch.float32), ((2112, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 88, 1, 1), torch.float32), ((88, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 2112, 1, 1), torch.float32), ((2112, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 352, 8, 8), torch.float32), ((1, 352, 8, 8), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1408, 8, 8), torch.float32), ((1408, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add161,
        [((64, 64, 96), torch.float32)],
        {"model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add9,
        [((1, 15, 15, 512), torch.float32)],
        {"model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 4096, 96), torch.float32), ((1, 4096, 96), torch.float32)],
        {"model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add2,
        [((1, 4096, 384), torch.float32)],
        {"model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add161,
        [((1, 4096, 96), torch.float32)],
        {"model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add162,
        [((16, 64, 192), torch.float32)],
        {"model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1024, 192), torch.float32), ((1, 1024, 192), torch.float32)],
        {"model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add162,
        [((1, 1024, 192), torch.float32)],
        {"model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add2,
        [((4, 64, 384), torch.float32)],
        {"model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 256, 384), torch.float32), ((1, 256, 384), torch.float32)],
        {"model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add4,
        [((1, 256, 1536), torch.float32)],
        {"model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add2,
        [((1, 256, 384), torch.float32)],
        {"model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add97,
        [((1, 64, 768), torch.float32)],
        {"model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 64, 768), torch.float32), ((1, 64, 768), torch.float32)],
        {"model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add159,
        [((1, 64, 3072), torch.float32)],
        {"model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"], "pcc": 0.99},
    ),
    (Add22, [((1, 128, 2048), torch.float32)], {"model_names": ["pt_albert_xlarge_v1_mlm_hf"], "pcc": 0.99}),
    (
        Add0,
        [((1, 128, 2048), torch.float32), ((1, 128, 2048), torch.float32)],
        {"model_names": ["pt_albert_xlarge_v1_mlm_hf"], "pcc": 0.99},
    ),
    (Add64, [((1, 128, 8192), torch.float32)], {"model_names": ["pt_albert_xlarge_v1_mlm_hf"], "pcc": 0.99}),
    (
        Add0,
        [((1, 160, 56, 56), torch.float32), ((160, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 224, 56, 56), torch.float32), ((224, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
                "pt_regnet_regnet_y_8gf_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 224, 28, 28), torch.float32), ((224, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 352, 28, 28), torch.float32), ((352, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 384, 28, 28), torch.float32), ((384, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "onnx_efficientnet_efficientnet_b5_img_cls_timm",
                "pd_densenet_121_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((416,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": [
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add163,
        [((416,), torch.float32)],
        {
            "model_names": [
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 416, 28, 28), torch.float32), ((416, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 448, 28, 28), torch.float32), ((448, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_regnet_facebook_regnet_y_080_img_cls_hf",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
                "pt_regnet_regnet_y_8gf_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 480, 28, 28), torch.float32), ((480, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 352, 14, 14), torch.float32), ((352, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 416, 14, 14), torch.float32), ((416, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((544,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": [
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add164,
        [((544,), torch.float32)],
        {
            "model_names": [
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 544, 14, 14), torch.float32), ((544, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((608,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": [
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add165,
        [((608,), torch.float32)],
        {
            "model_names": [
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 608, 14, 14), torch.float32), ((608, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 640, 14, 14), torch.float32), ((640, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((704,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": [
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add166,
        [((704,), torch.float32)],
        {
            "model_names": [
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 704, 14, 14), torch.float32), ((704, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((736,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": [
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add167,
        [((736,), torch.float32)],
        {
            "model_names": [
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 736, 14, 14), torch.float32), ((736, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((800,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": [
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add168,
        [((800,), torch.float32)],
        {
            "model_names": [
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 800, 14, 14), torch.float32), ((800, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((832,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": [
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add169,
        [((832,), torch.float32)],
        {
            "model_names": [
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 832, 14, 14), torch.float32), ((832, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((864,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": [
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add170,
        [((864,), torch.float32)],
        {
            "model_names": [
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 864, 14, 14), torch.float32), ((864, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((896,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": [
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_regnet_facebook_regnet_y_080_img_cls_hf",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_regnet_regnet_x_16gf_img_cls_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
                "pt_regnet_regnet_y_8gf_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add171,
        [((896,), torch.float32)],
        {
            "model_names": [
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_regnet_facebook_regnet_y_080_img_cls_hf",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_regnet_regnet_x_16gf_img_cls_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
                "pt_regnet_regnet_y_8gf_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 896, 14, 14), torch.float32), ((896, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_regnet_facebook_regnet_y_080_img_cls_hf",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_regnet_regnet_x_16gf_img_cls_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
                "pt_regnet_regnet_y_8gf_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((928,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": [
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add172,
        [((928,), torch.float32)],
        {
            "model_names": [
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 928, 14, 14), torch.float32), ((928, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 960, 14, 14), torch.float32), ((960, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((992,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": [
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add173,
        [((992,), torch.float32)],
        {
            "model_names": [
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 992, 14, 14), torch.float32), ((992, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 544, 7, 7), torch.float32), ((544, 1, 1), torch.float32)],
        {
            "model_names": ["pt_densenet_densenet121_img_cls_torchvision", "pd_densenet_121_img_cls_paddlemodels"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 608, 7, 7), torch.float32), ((608, 1, 1), torch.float32)],
        {
            "model_names": ["pt_densenet_densenet121_img_cls_torchvision", "pd_densenet_121_img_cls_paddlemodels"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 640, 7, 7), torch.float32), ((640, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 704, 7, 7), torch.float32), ((704, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 736, 7, 7), torch.float32), ((736, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 768, 7, 7), torch.float32), ((768, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_mlp_mixer_mixer_b32_224_img_cls_timm",
                "pd_densenet_121_img_cls_paddlemodels",
                "pt_vit_vit_b_32_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 800, 7, 7), torch.float32), ((800, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 832, 7, 7), torch.float32), ((832, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 864, 7, 7), torch.float32), ((864, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 896, 7, 7), torch.float32), ((896, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 928, 7, 7), torch.float32), ((928, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 992, 7, 7), torch.float32), ((992, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 12, 128, 128), torch.float32), ((1, 12, 128, 128), torch.float32)],
        {
            "model_names": [
                "pt_distilbert_distilbert_base_cased_mlm_hf",
                "pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf",
                "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add174,
        [((1, 128, 28996), torch.float32)],
        {"model_names": ["pt_distilbert_distilbert_base_cased_mlm_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 24, 112, 112), torch.float32), ((1, 24, 112, 112), torch.float32)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 56, 28, 28), torch.float32), ((56, 1, 1), torch.float32)],
        {"model_names": ["pt_efficientnet_efficientnet_b4_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 336, 28, 28), torch.float32), ((336, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_regnet_regnet_y_1_6gf_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 14, 1, 1), torch.float32), ((14, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "onnx_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 336, 1, 1), torch.float32), ((336, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_regnet_regnet_y_1_6gf_img_cls_torchvision",
                "onnx_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 56, 28, 28), torch.float32), ((1, 56, 28, 28), torch.float32)],
        {"model_names": ["pt_efficientnet_efficientnet_b4_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 160, 14, 14), torch.float32), ((160, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_hrnet_hrnetv2_w40_pose_estimation_osmr",
                "pt_regnet_regnet_x_400mf_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 40, 1, 1), torch.float32), ((40, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 160, 14, 14), torch.float32), ((1, 160, 14, 14), torch.float32)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_hrnet_hrnetv2_w40_pose_estimation_osmr",
                "pt_regnet_regnet_x_400mf_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 272, 7, 7), torch.float32), ((272, 1, 1), torch.float32)],
        {"model_names": ["pt_efficientnet_efficientnet_b4_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1632, 7, 7), torch.float32), ((1632, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 68, 1, 1), torch.float32), ((68, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "onnx_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1632, 1, 1), torch.float32), ((1632, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "onnx_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 272, 7, 7), torch.float32), ((1, 272, 7, 7), torch.float32)],
        {"model_names": ["pt_efficientnet_efficientnet_b4_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((2688,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add175,
        [((2688,), torch.float32)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 2688, 7, 7), torch.float32), ((2688, 1, 1), torch.float32)],
        {"model_names": ["pt_efficientnet_efficientnet_b4_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 112, 1, 1), torch.float32), ((112, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_regnet_facebook_regnet_y_080_img_cls_hf",
                "onnx_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_regnet_regnet_y_8gf_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 2688, 1, 1), torch.float32), ((2688, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "onnx_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 448, 7, 7), torch.float32), ((1, 448, 7, 7), torch.float32)],
        {"model_names": ["pt_efficientnet_efficientnet_b4_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1792,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add176,
        [((1792,), torch.float32)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1792, 7, 7), torch.float32), ((1792, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "onnx_mobilenetv2_mobilenetv2_140_img_cls_timm",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 256, 768), torch.float32), ((1, 256, 768), torch.float32)],
        {"model_names": ["pt_gptneo_eleutherai_gpt_neo_125m_clm_hf", "pt_gpt_gpt2_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 12, 256, 256), torch.float32), ((1, 1, 256, 256), torch.float32)],
        {"model_names": ["pt_gptneo_eleutherai_gpt_neo_125m_clm_hf", "pt_gpt_gpt2_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Add15,
        [((1, 256, 3072), torch.float32)],
        {"model_names": ["pt_gptneo_eleutherai_gpt_neo_125m_clm_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 96, 28, 28), torch.float32), ((1, 96, 28, 28), torch.float32)],
        {
            "model_names": [
                "pt_hrnet_hrnetv2_w48_pose_estimation_osmr",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 48, 28, 28), torch.float32), ((48, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_hrnet_hrnetv2_w48_pose_estimation_osmr",
                "onnx_mobilenetv2_mobilenetv2_140_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 192, 14, 14), torch.float32), ((1, 192, 14, 14), torch.float32)],
        {"model_names": ["pt_hrnet_hrnetv2_w48_pose_estimation_osmr"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 384, 7, 7), torch.float32), ((1, 384, 7, 7), torch.float32)],
        {"model_names": ["pt_hrnet_hrnetv2_w48_pose_estimation_osmr"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 336, 7, 7), torch.float32), ((336, 1, 1), torch.float32)],
        {"model_names": ["pt_hrnet_hrnetv2_w48_pose_estimation_osmr"], "pcc": 0.99},
    ),
    (
        Add13,
        [((1, 256, 1), torch.float32)],
        {"model_names": ["pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 32, 256, 64), torch.float32), ((1, 32, 256, 64), torch.float32)],
        {"model_names": ["pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 8, 256, 64), torch.float32), ((1, 8, 256, 64), torch.float32)],
        {"model_names": ["pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 32, 256, 256), torch.float32), ((1, 1, 256, 256), torch.float32)],
        {
            "model_names": ["pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf", "pt_phi2_microsoft_phi_2_clm_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 960, 28, 28), torch.float32), ((960, 1, 1), torch.float32)],
        {"model_names": ["pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 160, 28, 28), torch.float32), ((1, 160, 28, 28), torch.float32)],
        {"model_names": ["pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 21, 28, 28), torch.float32), ((21, 1, 1), torch.float32)],
        {"model_names": ["pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf"], "pcc": 0.99},
    ),
    (Add177, [((1, 256, 2560), torch.float32)], {"model_names": ["pt_phi2_microsoft_phi_2_clm_hf"], "pcc": 0.99}),
    (
        Add0,
        [((1, 32, 256, 32), torch.float32), ((1, 32, 256, 32), torch.float32)],
        {"model_names": ["pt_phi2_microsoft_phi_2_clm_hf", "pt_phi1_microsoft_phi_1_seq_cls_hf"], "pcc": 0.99},
    ),
    (Add178, [((1, 256, 10240), torch.float32)], {"model_names": ["pt_phi2_microsoft_phi_2_clm_hf"], "pcc": 0.99}),
    (
        Add0,
        [((1, 256, 2560), torch.float32), ((1, 256, 2560), torch.float32)],
        {"model_names": ["pt_phi2_microsoft_phi_2_clm_hf"], "pcc": 0.99},
    ),
    (
        Add179,
        [((1, 256, 51200), torch.float32)],
        {
            "model_names": ["pt_phi2_microsoft_phi_2_clm_hf", "pt_codegen_salesforce_codegen_350m_nl_clm_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Add13,
        [((1, 29, 1), torch.float32)],
        {
            "model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_chat_clm_hf", "pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Add21,
        [((1, 29, 1024), torch.float32)],
        {"model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_chat_clm_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 16, 29, 64), torch.float32), ((1, 16, 29, 64), torch.float32)],
        {"model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_chat_clm_hf"], "pcc": 0.99},
    ),
    (
        Add180,
        [((1, 16, 29, 29), torch.float32)],
        {"model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_chat_clm_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 29, 1024), torch.float32), ((1, 29, 1024), torch.float32)],
        {"model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_chat_clm_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((168,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": [
                "pt_regnet_facebook_regnet_y_080_img_cls_hf",
                "pt_regnet_regnet_x_1_6gf_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add181,
        [((168,), torch.float32)],
        {
            "model_names": [
                "pt_regnet_facebook_regnet_y_080_img_cls_hf",
                "pt_regnet_regnet_x_1_6gf_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 168, 112, 112), torch.float32), ((168, 1, 1), torch.float32)],
        {"model_names": ["pt_regnet_facebook_regnet_y_080_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 168, 56, 56), torch.float32), ((168, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_regnet_facebook_regnet_y_080_img_cls_hf",
                "pt_regnet_regnet_x_1_6gf_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 168, 56, 56), torch.float32), ((1, 168, 56, 56), torch.float32)],
        {"model_names": ["pt_regnet_facebook_regnet_y_080_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 42, 1, 1), torch.float32), ((42, 1, 1), torch.float32)],
        {"model_names": ["pt_regnet_facebook_regnet_y_080_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 448, 56, 56), torch.float32), ((448, 1, 1), torch.float32)],
        {
            "model_names": ["pt_regnet_facebook_regnet_y_080_img_cls_hf", "pt_regnet_regnet_y_8gf_img_cls_torchvision"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 448, 1, 1), torch.float32), ((448, 1, 1), torch.float32)],
        {
            "model_names": ["pt_regnet_facebook_regnet_y_080_img_cls_hf", "pt_regnet_regnet_y_8gf_img_cls_torchvision"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 448, 28, 28), torch.float32), ((1, 448, 28, 28), torch.float32)],
        {
            "model_names": ["pt_regnet_facebook_regnet_y_080_img_cls_hf", "pt_regnet_regnet_y_8gf_img_cls_torchvision"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 896, 28, 28), torch.float32), ((896, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_regnet_facebook_regnet_y_080_img_cls_hf",
                "pt_regnet_regnet_x_16gf_img_cls_torchvision",
                "pt_regnet_regnet_y_8gf_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 896, 1, 1), torch.float32), ((896, 1, 1), torch.float32)],
        {
            "model_names": ["pt_regnet_facebook_regnet_y_080_img_cls_hf", "pt_regnet_regnet_y_8gf_img_cls_torchvision"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 896, 14, 14), torch.float32), ((1, 896, 14, 14), torch.float32)],
        {
            "model_names": [
                "pt_regnet_facebook_regnet_y_080_img_cls_hf",
                "pt_regnet_regnet_x_16gf_img_cls_torchvision",
                "pt_regnet_regnet_y_8gf_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 224, 1, 1), torch.float32), ((224, 1, 1), torch.float32)],
        {
            "model_names": ["pt_regnet_facebook_regnet_y_080_img_cls_hf", "pt_regnet_regnet_y_8gf_img_cls_torchvision"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((2016,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": ["pt_regnet_facebook_regnet_y_080_img_cls_hf", "pt_regnet_regnet_y_8gf_img_cls_torchvision"],
            "pcc": 0.99,
        },
    ),
    (
        Add182,
        [((2016,), torch.float32)],
        {
            "model_names": ["pt_regnet_facebook_regnet_y_080_img_cls_hf", "pt_regnet_regnet_y_8gf_img_cls_torchvision"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 2016, 14, 14), torch.float32), ((2016, 1, 1), torch.float32)],
        {
            "model_names": ["pt_regnet_facebook_regnet_y_080_img_cls_hf", "pt_regnet_regnet_y_8gf_img_cls_torchvision"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 2016, 7, 7), torch.float32), ((2016, 1, 1), torch.float32)],
        {
            "model_names": ["pt_regnet_facebook_regnet_y_080_img_cls_hf", "pt_regnet_regnet_y_8gf_img_cls_torchvision"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 2016, 1, 1), torch.float32), ((2016, 1, 1), torch.float32)],
        {
            "model_names": ["pt_regnet_facebook_regnet_y_080_img_cls_hf", "pt_regnet_regnet_y_8gf_img_cls_torchvision"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 2016, 7, 7), torch.float32), ((1, 2016, 7, 7), torch.float32)],
        {
            "model_names": ["pt_regnet_facebook_regnet_y_080_img_cls_hf", "pt_regnet_regnet_y_8gf_img_cls_torchvision"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 80, 56, 56), torch.float32), ((80, 1, 1), torch.float32)],
        {"model_names": ["pt_regnet_regnet_x_8gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 80, 112, 112), torch.float32), ((80, 1, 1), torch.float32)],
        {"model_names": ["pt_regnet_regnet_x_8gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 80, 56, 56), torch.float32), ((1, 80, 56, 56), torch.float32)],
        {"model_names": ["pt_regnet_regnet_x_8gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 240, 56, 56), torch.float32), ((240, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_regnet_regnet_x_8gf_img_cls_torchvision",
                "onnx_efficientnet_efficientnet_b5_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 240, 28, 28), torch.float32), ((1, 240, 28, 28), torch.float32)],
        {"model_names": ["pt_regnet_regnet_x_8gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((720,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": [
                "pt_regnet_regnet_x_8gf_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add183,
        [((720,), torch.float32)],
        {
            "model_names": [
                "pt_regnet_regnet_x_8gf_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 720, 14, 14), torch.float32), ((720, 1, 1), torch.float32)],
        {"model_names": ["pt_regnet_regnet_x_8gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 720, 28, 28), torch.float32), ((720, 1, 1), torch.float32)],
        {"model_names": ["pt_regnet_regnet_x_8gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 720, 14, 14), torch.float32), ((1, 720, 14, 14), torch.float32)],
        {"model_names": ["pt_regnet_regnet_x_8gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1920,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": [
                "pt_regnet_regnet_x_8gf_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add184,
        [((1920,), torch.float32)],
        {
            "model_names": [
                "pt_regnet_regnet_x_8gf_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1920, 7, 7), torch.float32), ((1920, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_regnet_regnet_x_8gf_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1920, 14, 14), torch.float32), ((1920, 1, 1), torch.float32)],
        {"model_names": ["pt_regnet_regnet_x_8gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1920, 7, 7), torch.float32), ((1, 1920, 7, 7), torch.float32)],
        {"model_names": ["pt_regnet_regnet_x_8gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add17,
        [((1, 16384, 64), torch.float32)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add18,
        [((1, 16384, 256), torch.float32)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add19,
        [((1, 4096, 128), torch.float32)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add19,
        [((1, 256, 128), torch.float32)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add20,
        [((1, 4096, 512), torch.float32)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add51,
        [((1, 1024, 320), torch.float32)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add51,
        [((1, 256, 320), torch.float32)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add45,
        [((1, 1024, 1280), torch.float32)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add18,
        [((1, 1024, 256), torch.float32)],
        {"model_names": ["pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf"], "pcc": 0.99},
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
        Add13,
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
        Add65,
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
        [((1, 1, 224, 224), torch.float32), ((1, 1, 1), torch.float32)],
        {"model_names": ["pt_unet_carvana_base_img_seg_github", "pt_unet_qubvel_img_seg_torchhub"], "pcc": 0.99},
    ),
    (Add15, [((197, 1, 3072), torch.float32)], {"model_names": ["pt_vit_vit_l_16_img_cls_torchvision"], "pcc": 0.99}),
    (
        Add122,
        [((1, 16, 197, 197), torch.float32)],
        {"model_names": ["pt_vit_vit_l_16_img_cls_torchvision"], "pcc": 0.99},
    ),
    (Add21, [((197, 1024), torch.float32)], {"model_names": ["pt_vit_vit_l_16_img_cls_torchvision"], "pcc": 0.99}),
    (
        Add185,
        [((1, 20, 2, 2), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 16, 120, 120), torch.float32), ((16, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5n_img_cls_torchhub_480x480",
                "pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 16, 120, 120), torch.float32), ((1, 16, 120, 120), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5n_img_cls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 64, 60, 60), torch.float32), ((64, 1, 1), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5n_img_cls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 32, 60, 60), torch.float32), ((32, 1, 1), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5n_img_cls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 32, 60, 60), torch.float32), ((1, 32, 60, 60), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5n_img_cls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 128, 30, 30), torch.float32), ((128, 1, 1), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5n_img_cls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 64, 30, 30), torch.float32), ((64, 1, 1), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5n_img_cls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 64, 30, 30), torch.float32), ((1, 64, 30, 30), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5n_img_cls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 256, 15, 15), torch.float32), ((256, 1, 1), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5n_img_cls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 128, 15, 15), torch.float32), ((128, 1, 1), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5n_img_cls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 128, 15, 15), torch.float32), ((1, 128, 15, 15), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5n_img_cls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 64, 320, 320), torch.float32), ((64, 1, 1), torch.float32)],
        {
            "model_names": ["pt_yolov9_default_obj_det_github", "pt_yolo_v5_yolov5l_img_cls_torchhub_640x640"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 128, 160, 160), torch.float32), ((128, 1, 1), torch.float32)],
        {
            "model_names": ["pt_yolov9_default_obj_det_github", "pt_yolo_v5_yolov5l_img_cls_torchhub_640x640"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 256, 160, 160), torch.float32), ((256, 1, 1), torch.float32)],
        {"model_names": ["pt_yolov9_default_obj_det_github"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 512, 80, 80), torch.float32), ((512, 1, 1), torch.float32)],
        {"model_names": ["pt_yolov9_default_obj_det_github"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 6, 768), torch.float32), ((1, 6, 768), torch.float32)],
        {
            "model_names": ["onnx_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Add97,
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
            "model_names": ["onnx_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Add159,
        [((1, 6, 3072), torch.float32)],
        {
            "model_names": ["onnx_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 16, 56, 56), torch.float32), ((1, 16, 56, 56), torch.float32)],
        {
            "model_names": [
                "onnx_mobilenetv2_mobilenetv2_050_img_cls_timm",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 16, 28, 28), torch.float32), ((16, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_mobilenetv2_mobilenetv2_050_img_cls_timm",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_autoencoder_conv_img_enc_github",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 16, 28, 28), torch.float32), ((1, 16, 28, 28), torch.float32)],
        {"model_names": ["onnx_mobilenetv2_mobilenetv2_050_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 32, 14, 14), torch.float32), ((1, 32, 14, 14), torch.float32)],
        {"model_names": ["onnx_mobilenetv2_mobilenetv2_050_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 48, 14, 14), torch.float32), ((1, 48, 14, 14), torch.float32)],
        {
            "model_names": [
                "onnx_mobilenetv2_mobilenetv2_050_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 288, 7, 7), torch.float32), ((288, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_mobilenetv2_mobilenetv2_050_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 80, 7, 7), torch.float32), ((1, 80, 7, 7), torch.float32)],
        {"model_names": ["onnx_mobilenetv2_mobilenetv2_050_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 32, 256, 256), torch.float32), ((32, 1, 1), torch.float32)],
        {"model_names": ["onnx_unet_base_img_seg_torchhub", "pt_unet_base_img_seg_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 256, 32, 32), torch.float32), ((256, 1, 1), torch.float32)],
        {"model_names": ["onnx_unet_base_img_seg_torchhub", "pt_unet_base_img_seg_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1, 256, 256), torch.float32), ((1, 1, 1), torch.float32)],
        {"model_names": ["onnx_unet_base_img_seg_torchhub", "pt_unet_base_img_seg_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 8, 768), torch.float32), ((1, 8, 768), torch.float32)],
        {
            "model_names": [
                "pd_blip_text_salesforce_blip_image_captioning_base_text_enc_padlenlp",
                "pd_bert_bert_base_uncased_seq_cls_padlenlp",
                "pd_chineseclip_text_ofa_sys_chinese_clip_vit_base_patch16_text_enc_padlenlp",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add13,
        [((1, 8, 1), torch.float32)],
        {
            "model_names": [
                "pd_blip_text_salesforce_blip_image_captioning_base_text_enc_padlenlp",
                "pd_bert_bert_base_uncased_seq_cls_padlenlp",
                "pd_chineseclip_text_ofa_sys_chinese_clip_vit_base_patch16_text_enc_padlenlp",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add14,
        [((1, 8, 768), torch.float32)],
        {
            "model_names": [
                "pd_blip_text_salesforce_blip_image_captioning_base_text_enc_padlenlp",
                "pd_bert_bert_base_uncased_seq_cls_padlenlp",
                "pd_chineseclip_text_ofa_sys_chinese_clip_vit_base_patch16_text_enc_padlenlp",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add186,
        [((1, 12, 8, 8), torch.float32)],
        {
            "model_names": [
                "pd_blip_text_salesforce_blip_image_captioning_base_text_enc_padlenlp",
                "pd_chineseclip_text_ofa_sys_chinese_clip_vit_base_patch16_text_enc_padlenlp",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add15,
        [((1, 8, 3072), torch.float32)],
        {
            "model_names": [
                "pd_blip_text_salesforce_blip_image_captioning_base_text_enc_padlenlp",
                "pd_bert_bert_base_uncased_seq_cls_padlenlp",
                "pd_chineseclip_text_ofa_sys_chinese_clip_vit_base_patch16_text_enc_padlenlp",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add13,
        [((1, 9, 1), torch.float32)],
        {
            "model_names": [
                "pd_ernie_1_0_seq_cls_padlenlp",
                "pd_ernie_1_0_qa_padlenlp",
                "pd_bert_chinese_roberta_base_mlm_padlenlp",
                "pd_bert_bert_base_uncased_qa_padlenlp",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add13,
        [((1, 1, 1, 9), torch.float32)],
        {
            "model_names": [
                "pd_ernie_1_0_seq_cls_padlenlp",
                "pd_ernie_1_0_qa_padlenlp",
                "pd_bert_chinese_roberta_base_mlm_padlenlp",
                "pd_bert_bert_base_uncased_qa_padlenlp",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add158,
        [((1, 16, 224, 224), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add13,
        [((1, 16, 224, 224), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 32, 224, 224), torch.float32), ((32, 1, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add158,
        [((1, 32, 224, 224), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add13,
        [((1, 32, 224, 224), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add158,
        [((1, 32, 112, 112), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add158,
        [((1, 48, 112, 112), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add13,
        [((1, 48, 112, 112), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add158,
        [((1, 48, 56, 56), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add158,
        [((1, 96, 56, 56), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add13,
        [((1, 96, 56, 56), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add158,
        [((1, 96, 28, 28), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add158,
        [((1, 192, 28, 28), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add13,
        [((1, 192, 28, 28), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add158,
        [((1, 192, 14, 14), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add158,
        [((1, 384, 14, 14), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add13,
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
        [((1, 24, 28, 28), torch.float32), ((1, 24, 28, 28), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 18, 56, 56), torch.float32), ((18, 1, 1), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
            ],
            "pcc": 0.99,
        },
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
        Add0,
        [((1, 40, 144, 144), torch.float32), ((40, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b3a_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 24, 144, 144), torch.float32), ((24, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b3a_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 24, 144, 144), torch.float32), ((1, 24, 144, 144), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b3a_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 144, 144, 144), torch.float32), ((144, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b3a_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 144, 72, 72), torch.float32), ((144, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b3a_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 32, 72, 72), torch.float32), ((32, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b3a_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 192, 72, 72), torch.float32), ((192, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b3a_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 32, 72, 72), torch.float32), ((1, 32, 72, 72), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b3a_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 192, 36, 36), torch.float32), ((192, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b3a_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 48, 36, 36), torch.float32), ((48, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b3a_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 288, 36, 36), torch.float32), ((288, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b3a_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 48, 36, 36), torch.float32), ((1, 48, 36, 36), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b3a_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 288, 18, 18), torch.float32), ((288, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b3a_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 96, 18, 18), torch.float32), ((96, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b3a_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 576, 18, 18), torch.float32), ((576, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b3a_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 96, 18, 18), torch.float32), ((1, 96, 18, 18), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b3a_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 136, 18, 18), torch.float32), ((136, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b3a_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 816, 18, 18), torch.float32), ((816, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b3a_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 34, 1, 1), torch.float32), ((34, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b3a_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 816, 1, 1), torch.float32), ((816, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b3a_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 136, 18, 18), torch.float32), ((1, 136, 18, 18), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b3a_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 816, 9, 9), torch.float32), ((816, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b3a_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 232, 9, 9), torch.float32), ((232, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b3a_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1392, 9, 9), torch.float32), ((1392, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b3a_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 58, 1, 1), torch.float32), ((58, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "pt_regnet_facebook_regnet_y_320_img_cls_hf",
                "pt_regnet_regnet_y_32gf_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1392, 1, 1), torch.float32), ((1392, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "pt_regnet_facebook_regnet_y_320_img_cls_hf",
                "pt_regnet_regnet_y_32gf_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 232, 9, 9), torch.float32), ((1, 232, 9, 9), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b3a_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 384, 9, 9), torch.float32), ((384, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b3a_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 2304, 9, 9), torch.float32), ((2304, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b3a_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 2304, 1, 1), torch.float32), ((2304, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b3a_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 384, 9, 9), torch.float32), ((1, 384, 9, 9), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b3a_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1536, 9, 9), torch.float32), ((1536, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b3a_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 72, 14, 14), torch.float32), ((72, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 72, 14, 14), torch.float32), ((1, 72, 14, 14), torch.float32)],
        {
            "model_names": [
                "onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
            ],
            "pcc": 0.99,
        },
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
        [((1, 176, 7, 7), torch.float32), ((176, 1, 1), torch.float32)],
        {"model_names": ["onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1056, 7, 7), torch.float32), ((1056, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 176, 7, 7), torch.float32), ((1, 176, 7, 7), torch.float32)],
        {"model_names": ["onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 10, 768), torch.float32), ((1, 10, 768), torch.float32)],
        {"model_names": ["pd_bert_bert_base_japanese_mlm_padlenlp"], "pcc": 0.99},
    ),
    (Add13, [((1, 10, 1), torch.float32)], {"model_names": ["pd_bert_bert_base_japanese_mlm_padlenlp"], "pcc": 0.99}),
    (Add14, [((1, 10, 768), torch.float32)], {"model_names": ["pd_bert_bert_base_japanese_mlm_padlenlp"], "pcc": 0.99}),
    (
        Add13,
        [((1, 1, 1, 10), torch.float32)],
        {"model_names": ["pd_bert_bert_base_japanese_mlm_padlenlp"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 12, 10, 10), torch.float32), ((1, 1, 1, 10), torch.float32)],
        {"model_names": ["pd_bert_bert_base_japanese_mlm_padlenlp"], "pcc": 0.99},
    ),
    (
        Add15,
        [((1, 10, 3072), torch.float32)],
        {"model_names": ["pd_bert_bert_base_japanese_mlm_padlenlp"], "pcc": 0.99},
    ),
    (
        Add187,
        [((1, 10, 32000), torch.float32)],
        {"model_names": ["pd_bert_bert_base_japanese_mlm_padlenlp"], "pcc": 0.99},
    ),
    (
        Add13,
        [((1, 1, 1, 8), torch.float32)],
        {"model_names": ["pd_bert_bert_base_uncased_seq_cls_padlenlp"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 12, 8, 8), torch.float32), ((1, 1, 1, 8), torch.float32)],
        {"model_names": ["pd_bert_bert_base_uncased_seq_cls_padlenlp"], "pcc": 0.99},
    ),
    (
        Add16,
        [((1, 9, 2), torch.float32)],
        {"model_names": ["pd_ernie_1_0_qa_padlenlp", "pd_bert_bert_base_uncased_qa_padlenlp"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 8, 16, 50), torch.float32), ((8, 1, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add13,
        [((1, 8, 16, 50), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 2, 1, 1), torch.float32), ((2, 1, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add13,
        [((1, 8, 1, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 40, 16, 50), torch.float32), ((40, 1, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 40, 8, 50), torch.float32), ((40, 1, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 16, 8, 50), torch.float32), ((16, 1, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 48, 8, 50), torch.float32), ((48, 1, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 16, 8, 50), torch.float32), ((1, 16, 8, 50), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add13,
        [((1, 48, 8, 50), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 48, 4, 50), torch.float32), ((48, 1, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add13,
        [((1, 48, 4, 50), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add13,
        [((1, 48, 1, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 24, 4, 50), torch.float32), ((24, 1, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 120, 4, 50), torch.float32), ((120, 1, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add13,
        [((1, 120, 4, 50), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 24, 4, 50), torch.float32), ((1, 24, 4, 50), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 64, 4, 50), torch.float32), ((64, 1, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add13,
        [((1, 64, 4, 50), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 72, 4, 50), torch.float32), ((72, 1, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add13,
        [((1, 72, 4, 50), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 18, 1, 1), torch.float32), ((18, 1, 1), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pt_regnet_regnet_y_3_2gf_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 144, 4, 50), torch.float32), ((144, 1, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add13,
        [((1, 144, 4, 50), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 144, 2, 50), torch.float32), ((144, 1, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add13,
        [((1, 144, 2, 50), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add13,
        [((1, 144, 1, 1), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 48, 2, 50), torch.float32), ((48, 1, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 288, 2, 50), torch.float32), ((288, 1, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add13,
        [((1, 288, 2, 50), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add13,
        [((1, 288, 1, 1), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 48, 2, 50), torch.float32), ((1, 48, 2, 50), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 192), torch.float32), ((1, 192), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add36,
        [((1, 192), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 48), torch.float32), ((1, 48), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add188,
        [((1, 25, 97), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 64, 128, 128), torch.float32), ((1, 1, 1, 128), torch.float32)],
        {
            "model_names": [
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
                "pt_albert_xxlarge_v2_token_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 128, 4096), torch.float32), ((1, 128, 4096), torch.float32)],
        {
            "model_names": [
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
                "pt_albert_xxlarge_v2_token_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add189,
        [((1, 128, 16384), torch.float32)],
        {
            "model_names": [
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
                "pt_albert_xxlarge_v2_token_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (Add190, [((1, 32, 4608), torch.float32)], {"model_names": ["pt_bloom_bigscience_bloom_1b1_clm_hf"], "pcc": 0.99}),
    (
        Add0,
        [((16, 32, 32), torch.float32), ((16, 1, 32), torch.float32)],
        {"model_names": ["pt_bloom_bigscience_bloom_1b1_clm_hf"], "pcc": 0.99},
    ),
    (Add191, [((1, 1, 1, 32), torch.float32)], {"model_names": ["pt_bloom_bigscience_bloom_1b1_clm_hf"], "pcc": 0.99}),
    (Add65, [((1, 32, 1536), torch.float32)], {"model_names": ["pt_bloom_bigscience_bloom_1b1_clm_hf"], "pcc": 0.99}),
    (
        Add0,
        [((1, 32, 1536), torch.float32), ((1, 32, 1536), torch.float32)],
        {"model_names": ["pt_bloom_bigscience_bloom_1b1_clm_hf"], "pcc": 0.99},
    ),
    (Add192, [((1, 32, 6144), torch.float32)], {"model_names": ["pt_bloom_bigscience_bloom_1b1_clm_hf"], "pcc": 0.99}),
    (Add13, [((1, 32, 6144), torch.float32)], {"model_names": ["pt_bloom_bigscience_bloom_1b1_clm_hf"], "pcc": 0.99}),
    (
        Add0,
        [((1, 256, 16, 32), torch.float32), ((1, 256, 16, 32), torch.float32)],
        {"model_names": ["pt_codegen_salesforce_codegen_350m_nl_clm_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1056,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": [
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add193,
        [((1056,), torch.float32)],
        {
            "model_names": [
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1056, 14, 14), torch.float32), ((1056, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "onnx_efficientnet_efficientnet_b5_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1088,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": [
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_regnet_facebook_regnet_y_040_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add194,
        [((1088,), torch.float32)],
        {
            "model_names": [
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_regnet_facebook_regnet_y_040_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1088, 14, 14), torch.float32), ((1088, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_regnet_facebook_regnet_y_040_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1120,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": [
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add195,
        [((1120,), torch.float32)],
        {
            "model_names": [
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1120, 14, 14), torch.float32), ((1120, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1152, 14, 14), torch.float32), ((1152, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1184,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": [
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add196,
        [((1184,), torch.float32)],
        {
            "model_names": [
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1184, 14, 14), torch.float32), ((1184, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1216,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": [
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add197,
        [((1216,), torch.float32)],
        {
            "model_names": [
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1216, 14, 14), torch.float32), ((1216, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1248,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": [
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add198,
        [((1248,), torch.float32)],
        {
            "model_names": [
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1248, 14, 14), torch.float32), ((1248, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1280, 14, 14), torch.float32), ((1280, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1088, 7, 7), torch.float32), ((1088, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_regnet_facebook_regnet_y_040_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1120, 7, 7), torch.float32), ((1120, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1184, 7, 7), torch.float32), ((1184, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1216, 7, 7), torch.float32), ((1216, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1248, 7, 7), torch.float32), ((1248, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1312,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": [
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add199,
        [((1312,), torch.float32)],
        {
            "model_names": [
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1312, 7, 7), torch.float32), ((1312, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1344, 7, 7), torch.float32), ((1344, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_densenet_densenet169_img_cls_torchvision",
                "onnx_mobilenetv2_mobilenetv2_140_img_cls_timm",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1376,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": [
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add200,
        [((1376,), torch.float32)],
        {
            "model_names": [
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1376, 7, 7), torch.float32), ((1376, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1408,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": [
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add201,
        [((1408,), torch.float32)],
        {
            "model_names": [
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1408, 7, 7), torch.float32), ((1408, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1440,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": [
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add202,
        [((1440,), torch.float32)],
        {
            "model_names": [
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1440, 7, 7), torch.float32), ((1440, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1472,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": [
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add203,
        [((1472,), torch.float32)],
        {
            "model_names": [
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1472, 7, 7), torch.float32), ((1472, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1504,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": [
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add204,
        [((1504,), torch.float32)],
        {
            "model_names": [
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1504, 7, 7), torch.float32), ((1504, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1536, 7, 7), torch.float32), ((1536, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1568,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": [
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add205,
        [((1568,), torch.float32)],
        {
            "model_names": [
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1568, 7, 7), torch.float32), ((1568, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1600,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": [
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add206,
        [((1600,), torch.float32)],
        {
            "model_names": [
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1600, 7, 7), torch.float32), ((1600, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1664,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": [
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add207,
        [((1664,), torch.float32)],
        {
            "model_names": [
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1664, 7, 7), torch.float32), ((1664, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 384, 768), torch.float32), ((1, 384, 768), torch.float32)],
        {"model_names": ["pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf"], "pcc": 0.99},
    ),
    (
        Add14,
        [((1, 384, 768), torch.float32)],
        {"model_names": ["pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 12, 384, 384), torch.float32), ((1, 12, 384, 384), torch.float32)],
        {"model_names": ["pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf"], "pcc": 0.99},
    ),
    (
        Add15,
        [((1, 384, 3072), torch.float32)],
        {"model_names": ["pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 16, 14, 14), torch.float32), ((16, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_autoencoder_conv_img_enc_github",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 128, 7, 7), torch.float32), ((1, 128, 7, 7), torch.float32)],
        {
            "model_names": [
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 2048, 1, 9), torch.float32), ((2048, 1, 1), torch.float32)],
        {"model_names": ["pt_mamba_state_spaces_mamba_370m_hf_clm_hf"], "pcc": 0.99},
    ),
    (
        Add22,
        [((1, 6, 2048), torch.float32)],
        {"model_names": ["pt_mamba_state_spaces_mamba_370m_hf_clm_hf"], "pcc": 0.99},
    ),
    (
        Add13,
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
    (Add57, [((1, 768, 49), torch.float32)], {"model_names": ["pt_mlp_mixer_mixer_b32_224_img_cls_timm"], "pcc": 0.99}),
    (Add60, [((1, 256), torch.int64)], {"model_names": ["pt_opt_facebook_opt_350m_clm_hf"], "pcc": 0.99}),
    (Add24, [((256, 4096), torch.float32)], {"model_names": ["pt_opt_facebook_opt_350m_clm_hf"], "pcc": 0.99}),
    (Add21, [((256, 1024), torch.float32)], {"model_names": ["pt_opt_facebook_opt_350m_clm_hf"], "pcc": 0.99}),
    (
        Add0,
        [((256, 1024), torch.float32), ((256, 1024), torch.float32)],
        {"model_names": ["pt_opt_facebook_opt_350m_clm_hf"], "pcc": 0.99},
    ),
    (
        Add13,
        [((1, 39, 1), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (
        Add65,
        [((1, 39, 1536), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 12, 39, 128), torch.float32), ((1, 12, 39, 128), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (
        Add18,
        [((1, 39, 256), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 2, 39, 128), torch.float32), ((1, 2, 39, 128), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (
        Add208,
        [((1, 12, 39, 39), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 39, 1536), torch.float32), ((1, 39, 1536), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 232, 112, 112), torch.float32), ((232, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_regnet_facebook_regnet_y_320_img_cls_hf",
                "pt_regnet_regnet_y_32gf_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 232, 56, 56), torch.float32), ((232, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_regnet_facebook_regnet_y_320_img_cls_hf",
                "pt_regnet_regnet_y_32gf_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 232, 1, 1), torch.float32), ((232, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_regnet_facebook_regnet_y_320_img_cls_hf",
                "pt_regnet_regnet_y_32gf_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 232, 56, 56), torch.float32), ((1, 232, 56, 56), torch.float32)],
        {
            "model_names": [
                "pt_regnet_facebook_regnet_y_320_img_cls_hf",
                "pt_regnet_regnet_y_32gf_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((696,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": [
                "pt_regnet_facebook_regnet_y_320_img_cls_hf",
                "pt_regnet_regnet_y_32gf_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add209,
        [((696,), torch.float32)],
        {
            "model_names": [
                "pt_regnet_facebook_regnet_y_320_img_cls_hf",
                "pt_regnet_regnet_y_32gf_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 696, 56, 56), torch.float32), ((696, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_regnet_facebook_regnet_y_320_img_cls_hf",
                "pt_regnet_regnet_y_32gf_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 696, 28, 28), torch.float32), ((696, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_regnet_facebook_regnet_y_320_img_cls_hf",
                "pt_regnet_regnet_y_32gf_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 696, 1, 1), torch.float32), ((696, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_regnet_facebook_regnet_y_320_img_cls_hf",
                "pt_regnet_regnet_y_32gf_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 696, 28, 28), torch.float32), ((1, 696, 28, 28), torch.float32)],
        {
            "model_names": [
                "pt_regnet_facebook_regnet_y_320_img_cls_hf",
                "pt_regnet_regnet_y_32gf_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 174, 1, 1), torch.float32), ((174, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_regnet_facebook_regnet_y_320_img_cls_hf",
                "pt_regnet_regnet_y_32gf_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1392, 28, 28), torch.float32), ((1392, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_regnet_facebook_regnet_y_320_img_cls_hf",
                "pt_regnet_regnet_y_32gf_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1392, 14, 14), torch.float32), ((1392, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_regnet_facebook_regnet_y_320_img_cls_hf",
                "pt_regnet_regnet_y_32gf_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1392, 14, 14), torch.float32), ((1, 1392, 14, 14), torch.float32)],
        {
            "model_names": [
                "pt_regnet_facebook_regnet_y_320_img_cls_hf",
                "pt_regnet_regnet_y_32gf_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 348, 1, 1), torch.float32), ((348, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_regnet_facebook_regnet_y_320_img_cls_hf",
                "pt_regnet_regnet_y_32gf_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((3712,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": [
                "pt_regnet_facebook_regnet_y_320_img_cls_hf",
                "pt_regnet_regnet_y_32gf_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add210,
        [((3712,), torch.float32)],
        {
            "model_names": [
                "pt_regnet_facebook_regnet_y_320_img_cls_hf",
                "pt_regnet_regnet_y_32gf_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 3712, 14, 14), torch.float32), ((3712, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_regnet_facebook_regnet_y_320_img_cls_hf",
                "pt_regnet_regnet_y_32gf_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 3712, 7, 7), torch.float32), ((3712, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_regnet_facebook_regnet_y_320_img_cls_hf",
                "pt_regnet_regnet_y_32gf_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 3712, 1, 1), torch.float32), ((3712, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_regnet_facebook_regnet_y_320_img_cls_hf",
                "pt_regnet_regnet_y_32gf_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 3712, 7, 7), torch.float32), ((1, 3712, 7, 7), torch.float32)],
        {
            "model_names": [
                "pt_regnet_facebook_regnet_y_320_img_cls_hf",
                "pt_regnet_regnet_y_32gf_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 120, 56, 56), torch.float32), ((120, 1, 1), torch.float32)],
        {"model_names": ["pt_regnet_regnet_y_1_6gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 120, 28, 28), torch.float32), ((1, 120, 28, 28), torch.float32)],
        {"model_names": ["pt_regnet_regnet_y_1_6gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 336, 14, 14), torch.float32), ((1, 336, 14, 14), torch.float32)],
        {"model_names": ["pt_regnet_regnet_y_1_6gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 84, 1, 1), torch.float32), ((84, 1, 1), torch.float32)],
        {"model_names": ["pt_regnet_regnet_y_1_6gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((888,), torch.float32), ((1,), torch.float32)],
        {"model_names": ["pt_regnet_regnet_y_1_6gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (Add211, [((888,), torch.float32)], {"model_names": ["pt_regnet_regnet_y_1_6gf_img_cls_torchvision"], "pcc": 0.99}),
    (
        Add0,
        [((1, 888, 7, 7), torch.float32), ((888, 1, 1), torch.float32)],
        {"model_names": ["pt_regnet_regnet_y_1_6gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 888, 14, 14), torch.float32), ((888, 1, 1), torch.float32)],
        {"model_names": ["pt_regnet_regnet_y_1_6gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 888, 1, 1), torch.float32), ((888, 1, 1), torch.float32)],
        {"model_names": ["pt_regnet_regnet_y_1_6gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 888, 7, 7), torch.float32), ((1, 888, 7, 7), torch.float32)],
        {"model_names": ["pt_regnet_regnet_y_1_6gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 222, 1, 1), torch.float32), ((222, 1, 1), torch.float32)],
        {"model_names": ["pt_regnet_regnet_y_1_6gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 64, 240, 320), torch.float32), ((64, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_retinanet_retinanet_rn18fpn_obj_det_hf",
                "pt_yolo_v4_default_obj_det_github",
                "pt_retinanet_retinanet_rn34fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 64, 120, 160), torch.float32), ((64, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_retinanet_retinanet_rn18fpn_obj_det_hf",
                "pt_yolo_v4_default_obj_det_github",
                "pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf",
                "pt_retinanet_retinanet_rn34fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 64, 120, 160), torch.float32), ((1, 64, 120, 160), torch.float32)],
        {
            "model_names": [
                "pt_retinanet_retinanet_rn18fpn_obj_det_hf",
                "pt_yolo_v4_default_obj_det_github",
                "pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf",
                "pt_retinanet_retinanet_rn34fpn_obj_det_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 128, 60, 80), torch.float32), ((128, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_retinanet_retinanet_rn18fpn_obj_det_hf",
                "pt_yolo_v4_default_obj_det_github",
                "pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf",
                "pt_retinanet_retinanet_rn34fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 128, 60, 80), torch.float32), ((1, 128, 60, 80), torch.float32)],
        {
            "model_names": [
                "pt_retinanet_retinanet_rn18fpn_obj_det_hf",
                "pt_yolo_v4_default_obj_det_github",
                "pt_retinanet_retinanet_rn34fpn_obj_det_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 256, 30, 40), torch.float32), ((256, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_retinanet_retinanet_rn18fpn_obj_det_hf",
                "pt_yolo_v4_default_obj_det_github",
                "pt_retinanet_retinanet_rn34fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 256, 30, 40), torch.float32), ((1, 256, 30, 40), torch.float32)],
        {
            "model_names": [
                "pt_retinanet_retinanet_rn18fpn_obj_det_hf",
                "pt_yolo_v4_default_obj_det_github",
                "pt_retinanet_retinanet_rn34fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 512, 15, 20), torch.float32), ((512, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_retinanet_retinanet_rn18fpn_obj_det_hf",
                "pt_yolo_v4_default_obj_det_github",
                "pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf",
                "pt_retinanet_retinanet_rn34fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 512, 15, 20), torch.float32), ((1, 512, 15, 20), torch.float32)],
        {
            "model_names": [
                "pt_retinanet_retinanet_rn18fpn_obj_det_hf",
                "pt_yolo_v4_default_obj_det_github",
                "pt_retinanet_retinanet_rn34fpn_obj_det_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 256, 15, 20), torch.float32), ((256, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_retinanet_retinanet_rn18fpn_obj_det_hf",
                "pt_yolo_v4_default_obj_det_github",
                "pt_retinanet_retinanet_rn34fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 256, 60, 80), torch.float32), ((256, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_retinanet_retinanet_rn18fpn_obj_det_hf",
                "pt_yolo_v4_default_obj_det_github",
                "pt_retinanet_retinanet_rn34fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 256, 60, 80), torch.float32), ((1, 256, 60, 80), torch.float32)],
        {
            "model_names": [
                "pt_retinanet_retinanet_rn18fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn34fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 720, 60, 80), torch.float32), ((720, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_retinanet_retinanet_rn18fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn34fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 720, 30, 40), torch.float32), ((720, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_retinanet_retinanet_rn18fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn34fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 720, 15, 20), torch.float32), ((720, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_retinanet_retinanet_rn18fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn34fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 256, 8, 10), torch.float32), ((256, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_retinanet_retinanet_rn18fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn34fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 720, 8, 10), torch.float32), ((720, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_retinanet_retinanet_rn18fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn34fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 256, 4, 5), torch.float32), ((256, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_retinanet_retinanet_rn18fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn34fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 720, 4, 5), torch.float32), ((720, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_retinanet_retinanet_rn18fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn34fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 36, 60, 80), torch.float32), ((36, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_retinanet_retinanet_rn18fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn34fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 36, 30, 40), torch.float32), ((36, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_retinanet_retinanet_rn18fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn34fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 36, 15, 20), torch.float32), ((36, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_retinanet_retinanet_rn18fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn34fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 36, 8, 10), torch.float32), ((36, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_retinanet_retinanet_rn18fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn34fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 36, 4, 5), torch.float32), ((36, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_retinanet_retinanet_rn18fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn34fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add14,
        [((1, 1024, 768), torch.float32)],
        {
            "model_names": [
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
            "model_names": [
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
            "model_names": [
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add38,
        [((64, 49, 288), torch.float32)],
        {"model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 56, 56, 96), torch.float32), ((1, 56, 56, 96), torch.float32)],
        {"model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add5,
        [((1, 56, 56, 384), torch.float32)],
        {"model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add39,
        [((1, 56, 56, 96), torch.float32)],
        {"model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add40,
        [((16, 49, 576), torch.float32)],
        {"model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 28, 28, 192), torch.float32), ((1, 28, 28, 192), torch.float32)],
        {"model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add14,
        [((1, 28, 28, 768), torch.float32)],
        {"model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add36,
        [((1, 28, 28, 192), torch.float32)],
        {"model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add75,
        [((4, 49, 1152), torch.float32)],
        {"model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 14, 14, 384), torch.float32), ((1, 14, 14, 384), torch.float32)],
        {"model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add65,
        [((1, 14, 14, 1536), torch.float32)],
        {"model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add5,
        [((1, 14, 14, 384), torch.float32)],
        {"model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add77,
        [((1, 49, 2304), torch.float32)],
        {"model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 7, 7, 768), torch.float32), ((1, 7, 7, 768), torch.float32)],
        {"model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add15,
        [((1, 7, 7, 3072), torch.float32)],
        {"model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add14,
        [((1, 7, 7, 768), torch.float32)],
        {"model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add14,
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
        Add15,
        [((1, 201, 3072), torch.float32)],
        {"model_names": ["pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf"], "pcc": 0.99},
    ),
    (
        Add65,
        [((1, 1536), torch.float32)],
        {"model_names": ["pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf"], "pcc": 0.99},
    ),
    (
        Add212,
        [((1, 3129), torch.float32)],
        {"model_names": ["pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 80, 28, 28), torch.float32), ((80, 1, 1), torch.float32)],
        {"model_names": ["pt_vovnet_vovnet27s_img_cls_osmr", "pt_hrnet_hrnetv2_w40_pose_estimation_osmr"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 32, 480, 640), torch.float32), ((32, 1, 1), torch.float32)],
        {"model_names": ["pt_yolo_v4_default_obj_det_github"], "pcc": 0.99},
    ),
    (Add13, [((1, 32, 480, 640), torch.float32)], {"model_names": ["pt_yolo_v4_default_obj_det_github"], "pcc": 0.99}),
    (Add13, [((1, 64, 240, 320), torch.float32)], {"model_names": ["pt_yolo_v4_default_obj_det_github"], "pcc": 0.99}),
    (
        Add0,
        [((1, 32, 240, 320), torch.float32), ((32, 1, 1), torch.float32)],
        {"model_names": ["pt_yolo_v4_default_obj_det_github"], "pcc": 0.99},
    ),
    (Add13, [((1, 32, 240, 320), torch.float32)], {"model_names": ["pt_yolo_v4_default_obj_det_github"], "pcc": 0.99}),
    (
        Add0,
        [((1, 64, 240, 320), torch.float32), ((1, 64, 240, 320), torch.float32)],
        {"model_names": ["pt_yolo_v4_default_obj_det_github"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 128, 120, 160), torch.float32), ((128, 1, 1), torch.float32)],
        {
            "model_names": ["pt_yolo_v4_default_obj_det_github", "pt_retinanet_retinanet_rn50fpn_obj_det_hf"],
            "pcc": 0.99,
        },
    ),
    (Add13, [((1, 128, 120, 160), torch.float32)], {"model_names": ["pt_yolo_v4_default_obj_det_github"], "pcc": 0.99}),
    (Add13, [((1, 64, 120, 160), torch.float32)], {"model_names": ["pt_yolo_v4_default_obj_det_github"], "pcc": 0.99}),
    (Add13, [((1, 256, 60, 80), torch.float32)], {"model_names": ["pt_yolo_v4_default_obj_det_github"], "pcc": 0.99}),
    (Add13, [((1, 128, 60, 80), torch.float32)], {"model_names": ["pt_yolo_v4_default_obj_det_github"], "pcc": 0.99}),
    (
        Add0,
        [((1, 512, 30, 40), torch.float32), ((512, 1, 1), torch.float32)],
        {
            "model_names": ["pt_yolo_v4_default_obj_det_github", "pt_retinanet_retinanet_rn50fpn_obj_det_hf"],
            "pcc": 0.99,
        },
    ),
    (Add13, [((1, 512, 30, 40), torch.float32)], {"model_names": ["pt_yolo_v4_default_obj_det_github"], "pcc": 0.99}),
    (Add13, [((1, 256, 30, 40), torch.float32)], {"model_names": ["pt_yolo_v4_default_obj_det_github"], "pcc": 0.99}),
    (
        Add0,
        [((1, 1024, 15, 20), torch.float32), ((1024, 1, 1), torch.float32)],
        {"model_names": ["pt_yolo_v4_default_obj_det_github"], "pcc": 0.99},
    ),
    (Add13, [((1, 1024, 15, 20), torch.float32)], {"model_names": ["pt_yolo_v4_default_obj_det_github"], "pcc": 0.99}),
    (Add13, [((1, 512, 15, 20), torch.float32)], {"model_names": ["pt_yolo_v4_default_obj_det_github"], "pcc": 0.99}),
    (
        Add0,
        [((1, 128, 30, 40), torch.float32), ((128, 1, 1), torch.float32)],
        {"model_names": ["pt_yolo_v4_default_obj_det_github"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 255, 60, 80), torch.float32), ((255, 1, 1), torch.float32)],
        {"model_names": ["pt_yolo_v4_default_obj_det_github"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 255, 30, 40), torch.float32), ((255, 1, 1), torch.float32)],
        {"model_names": ["pt_yolo_v4_default_obj_det_github"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 255, 15, 20), torch.float32), ((255, 1, 1), torch.float32)],
        {"model_names": ["pt_yolo_v4_default_obj_det_github"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 512, 10, 10), torch.float32), ((1, 512, 10, 10), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5l_img_cls_torchhub_320x320"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 64, 160, 160), torch.float32), ((1, 64, 160, 160), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5l_img_cls_torchhub_640x640"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 128, 80, 80), torch.float32), ((1, 128, 80, 80), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5l_img_cls_torchhub_640x640"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 256, 40, 40), torch.float32), ((1, 256, 40, 40), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5l_img_cls_torchhub_640x640"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 512, 20, 20), torch.float32), ((1, 512, 20, 20), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5l_img_cls_torchhub_640x640"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 48, 320, 320), torch.float32), ((48, 1, 1), torch.float32)],
        {
            "model_names": ["pt_yolox_yolox_m_obj_det_torchhub", "pt_yolo_v5_yolov5m_img_cls_torchhub_640x640"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 96, 160, 160), torch.float32), ((96, 1, 1), torch.float32)],
        {
            "model_names": ["pt_yolox_yolox_m_obj_det_torchhub", "pt_yolo_v5_yolov5m_img_cls_torchhub_640x640"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 48, 160, 160), torch.float32), ((48, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_yolox_yolox_m_obj_det_torchhub",
                "onnx_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_640x640",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 48, 160, 160), torch.float32), ((1, 48, 160, 160), torch.float32)],
        {
            "model_names": ["pt_yolox_yolox_m_obj_det_torchhub", "pt_yolo_v5_yolov5m_img_cls_torchhub_640x640"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 192, 80, 80), torch.float32), ((192, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_yolox_yolox_m_obj_det_torchhub",
                "onnx_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_640x640",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 96, 80, 80), torch.float32), ((96, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_yolox_yolox_m_obj_det_torchhub",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_640x640",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 96, 80, 80), torch.float32), ((1, 96, 80, 80), torch.float32)],
        {
            "model_names": ["pt_yolox_yolox_m_obj_det_torchhub", "pt_yolo_v5_yolov5m_img_cls_torchhub_640x640"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 384, 40, 40), torch.float32), ((384, 1, 1), torch.float32)],
        {
            "model_names": ["pt_yolox_yolox_m_obj_det_torchhub", "pt_yolo_v5_yolov5m_img_cls_torchhub_640x640"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 192, 40, 40), torch.float32), ((192, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_yolox_yolox_m_obj_det_torchhub",
                "onnx_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_640x640",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 192, 40, 40), torch.float32), ((1, 192, 40, 40), torch.float32)],
        {
            "model_names": ["pt_yolox_yolox_m_obj_det_torchhub", "pt_yolo_v5_yolov5m_img_cls_torchhub_640x640"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 768, 20, 20), torch.float32), ((768, 1, 1), torch.float32)],
        {
            "model_names": ["pt_yolox_yolox_m_obj_det_torchhub", "pt_yolo_v5_yolov5m_img_cls_torchhub_640x640"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 384, 20, 20), torch.float32), ((384, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_yolox_yolox_m_obj_det_torchhub",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_640x640",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 192, 20, 20), torch.float32), ((192, 1, 1), torch.float32)],
        {
            "model_names": ["pt_yolox_yolox_m_obj_det_torchhub", "pt_yolo_v5_yolov5m_img_cls_torchhub_320x320"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 24, 160, 160), torch.float32), ((24, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 24, 160, 160), torch.float32), ((1, 24, 160, 160), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 144, 160, 160), torch.float32), ((144, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 144, 80, 80), torch.float32), ((144, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 56, 40, 40), torch.float32), ((56, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 336, 40, 40), torch.float32), ((336, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 56, 40, 40), torch.float32), ((1, 56, 40, 40), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 336, 20, 20), torch.float32), ((336, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 112, 20, 20), torch.float32), ((112, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 672, 20, 20), torch.float32), ((672, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 112, 20, 20), torch.float32), ((1, 112, 20, 20), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 160, 20, 20), torch.float32), ((160, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 960, 20, 20), torch.float32), ((960, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 160, 20, 20), torch.float32), ((1, 160, 20, 20), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 960, 10, 10), torch.float32), ((960, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 272, 10, 10), torch.float32), ((272, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1632, 10, 10), torch.float32), ((1632, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 272, 10, 10), torch.float32), ((1, 272, 10, 10), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 448, 10, 10), torch.float32), ((448, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 2688, 10, 10), torch.float32), ((2688, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 448, 10, 10), torch.float32), ((1, 448, 10, 10), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1792, 10, 10), torch.float32), ((1792, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 48, 28, 28), torch.float32), ((1, 48, 28, 28), torch.float32)],
        {"model_names": ["onnx_mobilenetv2_mobilenetv2_140_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 528, 14, 14), torch.float32), ((528, 1, 1), torch.float32)],
        {"model_names": ["onnx_mobilenetv2_mobilenetv2_140_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 88, 14, 14), torch.float32), ((1, 88, 14, 14), torch.float32)],
        {"model_names": ["onnx_mobilenetv2_mobilenetv2_140_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 136, 14, 14), torch.float32), ((136, 1, 1), torch.float32)],
        {"model_names": ["onnx_mobilenetv2_mobilenetv2_140_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 816, 14, 14), torch.float32), ((816, 1, 1), torch.float32)],
        {"model_names": ["onnx_mobilenetv2_mobilenetv2_140_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 136, 14, 14), torch.float32), ((1, 136, 14, 14), torch.float32)],
        {"model_names": ["onnx_mobilenetv2_mobilenetv2_140_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 816, 7, 7), torch.float32), ((816, 1, 1), torch.float32)],
        {"model_names": ["onnx_mobilenetv2_mobilenetv2_140_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 224, 7, 7), torch.float32), ((1, 224, 7, 7), torch.float32)],
        {"model_names": ["onnx_mobilenetv2_mobilenetv2_140_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add213,
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
        Add213,
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
        Add8,
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
        Add6,
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
        Add214,
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
        Add214,
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
        Add215,
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
        Add216,
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
        Add97,
        [((1, 197, 768), torch.float32)],
        {"model_names": ["onnx_vit_base_google_vit_base_patch16_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add159,
        [((1, 197, 3072), torch.float32)],
        {"model_names": ["onnx_vit_base_google_vit_base_patch16_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 11, 128), torch.float32), ((1, 11, 128), torch.float32)],
        {"model_names": ["pd_albert_chinese_tiny_mlm_padlenlp"], "pcc": 0.99},
    ),
    (Add19, [((1, 11, 128), torch.float32)], {"model_names": ["pd_albert_chinese_tiny_mlm_padlenlp"], "pcc": 0.99}),
    (Add217, [((1, 11, 312), torch.float32)], {"model_names": ["pd_albert_chinese_tiny_mlm_padlenlp"], "pcc": 0.99}),
    (
        Add0,
        [((1, 11, 312), torch.float32), ((1, 11, 312), torch.float32)],
        {"model_names": ["pd_albert_chinese_tiny_mlm_padlenlp"], "pcc": 0.99},
    ),
    (Add198, [((1, 11, 1248), torch.float32)], {"model_names": ["pd_albert_chinese_tiny_mlm_padlenlp"], "pcc": 0.99}),
    (
        Add218,
        [((1, 11, 21128), torch.float32)],
        {"model_names": ["pd_albert_chinese_tiny_mlm_padlenlp", "pd_roberta_rbt4_ch_clm_padlenlp"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 15, 768), torch.float32), ((1, 15, 768), torch.float32)],
        {"model_names": ["pd_bert_bert_base_japanese_seq_cls_padlenlp"], "pcc": 0.99},
    ),
    (
        Add13,
        [((1, 15, 1), torch.float32)],
        {"model_names": ["pd_bert_bert_base_japanese_seq_cls_padlenlp"], "pcc": 0.99},
    ),
    (
        Add14,
        [((1, 15, 768), torch.float32)],
        {"model_names": ["pd_bert_bert_base_japanese_seq_cls_padlenlp"], "pcc": 0.99},
    ),
    (
        Add13,
        [((1, 1, 1, 15), torch.float32)],
        {"model_names": ["pd_bert_bert_base_japanese_seq_cls_padlenlp"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 12, 15, 15), torch.float32), ((1, 1, 1, 15), torch.float32)],
        {"model_names": ["pd_bert_bert_base_japanese_seq_cls_padlenlp"], "pcc": 0.99},
    ),
    (
        Add15,
        [((1, 15, 3072), torch.float32)],
        {"model_names": ["pd_bert_bert_base_japanese_seq_cls_padlenlp"], "pcc": 0.99},
    ),
    (
        Add218,
        [((1, 9, 21128), torch.float32)],
        {"model_names": ["pd_bert_chinese_roberta_base_mlm_padlenlp"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 16, 16, 50), torch.float32), ((16, 1, 1), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v4_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add158,
        [((1, 16, 16, 50), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v4_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add13,
        [((1, 16, 16, 50), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v4_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 32, 16, 50), torch.float32), ((32, 1, 1), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v4_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add158,
        [((1, 32, 16, 50), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v4_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add13,
        [((1, 32, 16, 50), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v4_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 64, 16, 50), torch.float32), ((64, 1, 1), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v4_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add158,
        [((1, 64, 16, 50), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v4_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add13,
        [((1, 64, 16, 50), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v4_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 64, 8, 50), torch.float32), ((64, 1, 1), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v4_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add158,
        [((1, 64, 8, 50), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v4_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add13,
        [((1, 64, 8, 50), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v4_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 128, 8, 50), torch.float32), ((128, 1, 1), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v4_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add158,
        [((1, 128, 8, 50), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v4_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add13,
        [((1, 128, 8, 50), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v4_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 128, 8, 25), torch.float32), ((128, 1, 1), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v4_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add158,
        [((1, 128, 8, 25), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v4_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add13,
        [((1, 128, 8, 25), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v4_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 240, 8, 25), torch.float32), ((240, 1, 1), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v4_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add158,
        [((1, 240, 8, 25), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v4_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add13,
        [((1, 240, 8, 25), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v4_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 240, 4, 25), torch.float32), ((240, 1, 1), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v4_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add158,
        [((1, 240, 4, 25), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v4_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add13,
        [((1, 240, 4, 25), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v4_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 60, 1, 1), torch.float32), ((60, 1, 1), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v4_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add13,
        [((1, 240, 1, 1), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v4_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 480, 4, 25), torch.float32), ((480, 1, 1), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v4_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add158,
        [((1, 480, 4, 25), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v4_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add13,
        [((1, 480, 4, 25), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v4_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 480, 2, 25), torch.float32), ((480, 1, 1), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v4_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add158,
        [((1, 480, 2, 25), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v4_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add13,
        [((1, 480, 2, 25), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v4_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 60, 1, 12), torch.float32), ((60, 1, 1), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v4_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 120, 1, 12), torch.float32), ((120, 1, 1), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v4_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add13,
        [((1, 12, 1), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v4_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add54,
        [((1, 12, 120), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v4_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add219,
        [((1, 12, 360), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v4_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 12, 120), torch.float32), ((1, 12, 120), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v4_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add55,
        [((1, 12, 240), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v4_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 480, 1, 12), torch.float32), ((480, 1, 1), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v4_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add220,
        [((1, 12, 6625), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 4, 14, 14), torch.float32), ((4, 1, 1), torch.float32)],
        {"model_names": ["pt_autoencoder_conv_img_enc_github"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1, 28, 28), torch.float32), ((1, 1, 1), torch.float32)],
        {"model_names": ["pt_autoencoder_conv_img_enc_github"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1312, 14, 14), torch.float32), ((1312, 1, 1), torch.float32)],
        {"model_names": ["pt_densenet_densenet201_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1376, 14, 14), torch.float32), ((1376, 1, 1), torch.float32)],
        {"model_names": ["pt_densenet_densenet201_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1408, 14, 14), torch.float32), ((1408, 1, 1), torch.float32)],
        {"model_names": ["pt_densenet_densenet201_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1440, 14, 14), torch.float32), ((1440, 1, 1), torch.float32)],
        {"model_names": ["pt_densenet_densenet201_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1472, 14, 14), torch.float32), ((1472, 1, 1), torch.float32)],
        {"model_names": ["pt_densenet_densenet201_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1504, 14, 14), torch.float32), ((1504, 1, 1), torch.float32)],
        {"model_names": ["pt_densenet_densenet201_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1536, 14, 14), torch.float32), ((1536, 1, 1), torch.float32)],
        {"model_names": ["pt_densenet_densenet201_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1568, 14, 14), torch.float32), ((1568, 1, 1), torch.float32)],
        {"model_names": ["pt_densenet_densenet201_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1600, 14, 14), torch.float32), ((1600, 1, 1), torch.float32)],
        {"model_names": ["pt_densenet_densenet201_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1632, 14, 14), torch.float32), ((1632, 1, 1), torch.float32)],
        {"model_names": ["pt_densenet_densenet201_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1664, 14, 14), torch.float32), ((1664, 1, 1), torch.float32)],
        {"model_names": ["pt_densenet_densenet201_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1696,), torch.float32), ((1,), torch.float32)],
        {"model_names": ["pt_densenet_densenet201_img_cls_torchvision"], "pcc": 0.99},
    ),
    (Add221, [((1696,), torch.float32)], {"model_names": ["pt_densenet_densenet201_img_cls_torchvision"], "pcc": 0.99}),
    (
        Add0,
        [((1, 1696, 14, 14), torch.float32), ((1696, 1, 1), torch.float32)],
        {"model_names": ["pt_densenet_densenet201_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1728,), torch.float32), ((1,), torch.float32)],
        {"model_names": ["pt_densenet_densenet201_img_cls_torchvision"], "pcc": 0.99},
    ),
    (Add222, [((1728,), torch.float32)], {"model_names": ["pt_densenet_densenet201_img_cls_torchvision"], "pcc": 0.99}),
    (
        Add0,
        [((1, 1728, 14, 14), torch.float32), ((1728, 1, 1), torch.float32)],
        {"model_names": ["pt_densenet_densenet201_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1760,), torch.float32), ((1,), torch.float32)],
        {"model_names": ["pt_densenet_densenet201_img_cls_torchvision"], "pcc": 0.99},
    ),
    (Add223, [((1760,), torch.float32)], {"model_names": ["pt_densenet_densenet201_img_cls_torchvision"], "pcc": 0.99}),
    (
        Add0,
        [((1, 1760, 14, 14), torch.float32), ((1760, 1, 1), torch.float32)],
        {"model_names": ["pt_densenet_densenet201_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1792, 14, 14), torch.float32), ((1792, 1, 1), torch.float32)],
        {"model_names": ["pt_densenet_densenet201_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1696, 7, 7), torch.float32), ((1696, 1, 1), torch.float32)],
        {"model_names": ["pt_densenet_densenet201_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1728, 7, 7), torch.float32), ((1728, 1, 1), torch.float32)],
        {"model_names": ["pt_densenet_densenet201_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1760, 7, 7), torch.float32), ((1760, 1, 1), torch.float32)],
        {"model_names": ["pt_densenet_densenet201_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1824,), torch.float32), ((1,), torch.float32)],
        {"model_names": ["pt_densenet_densenet201_img_cls_torchvision"], "pcc": 0.99},
    ),
    (Add224, [((1824,), torch.float32)], {"model_names": ["pt_densenet_densenet201_img_cls_torchvision"], "pcc": 0.99}),
    (
        Add0,
        [((1, 1824, 7, 7), torch.float32), ((1824, 1, 1), torch.float32)],
        {"model_names": ["pt_densenet_densenet201_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1856,), torch.float32), ((1,), torch.float32)],
        {"model_names": ["pt_densenet_densenet201_img_cls_torchvision"], "pcc": 0.99},
    ),
    (Add225, [((1856,), torch.float32)], {"model_names": ["pt_densenet_densenet201_img_cls_torchvision"], "pcc": 0.99}),
    (
        Add0,
        [((1, 1856, 7, 7), torch.float32), ((1856, 1, 1), torch.float32)],
        {"model_names": ["pt_densenet_densenet201_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1888,), torch.float32), ((1,), torch.float32)],
        {"model_names": ["pt_densenet_densenet201_img_cls_torchvision"], "pcc": 0.99},
    ),
    (Add226, [((1888,), torch.float32)], {"model_names": ["pt_densenet_densenet201_img_cls_torchvision"], "pcc": 0.99}),
    (
        Add0,
        [((1, 1888, 7, 7), torch.float32), ((1888, 1, 1), torch.float32)],
        {"model_names": ["pt_densenet_densenet201_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 24, 60, 60), torch.float32), ((24, 1, 1), torch.float32)],
        {"model_names": ["pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 144, 60, 60), torch.float32), ((144, 1, 1), torch.float32)],
        {"model_names": ["pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 144, 30, 30), torch.float32), ((144, 1, 1), torch.float32)],
        {"model_names": ["pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 40, 30, 30), torch.float32), ((40, 1, 1), torch.float32)],
        {"model_names": ["pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 240, 30, 30), torch.float32), ((240, 1, 1), torch.float32)],
        {"model_names": ["pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 40, 30, 30), torch.float32), ((1, 40, 30, 30), torch.float32)],
        {"model_names": ["pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 240, 15, 15), torch.float32), ((240, 1, 1), torch.float32)],
        {"model_names": ["pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 80, 15, 15), torch.float32), ((80, 1, 1), torch.float32)],
        {"model_names": ["pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 480, 15, 15), torch.float32), ((480, 1, 1), torch.float32)],
        {"model_names": ["pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 80, 15, 15), torch.float32), ((1, 80, 15, 15), torch.float32)],
        {"model_names": ["pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 112, 15, 15), torch.float32), ((112, 1, 1), torch.float32)],
        {"model_names": ["pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 672, 15, 15), torch.float32), ((672, 1, 1), torch.float32)],
        {"model_names": ["pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 112, 15, 15), torch.float32), ((1, 112, 15, 15), torch.float32)],
        {"model_names": ["pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 672, 8, 8), torch.float32), ((672, 1, 1), torch.float32)],
        {"model_names": ["pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1152, 8, 8), torch.float32), ((1152, 1, 1), torch.float32)],
        {"model_names": ["pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 192, 8, 8), torch.float32), ((1, 192, 8, 8), torch.float32)],
        {"model_names": ["pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1280, 8, 8), torch.float32), ((1280, 1, 1), torch.float32)],
        {"model_names": ["pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (Add13, [((1, 522, 1), torch.float32)], {"model_names": ["pt_falcon3_tiiuae_falcon3_1b_base_clm_hf"], "pcc": 0.99}),
    (
        Add0,
        [((1, 8, 522, 256), torch.float32), ((1, 8, 522, 256), torch.float32)],
        {"model_names": ["pt_falcon3_tiiuae_falcon3_1b_base_clm_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 4, 522, 256), torch.float32), ((1, 4, 522, 256), torch.float32)],
        {"model_names": ["pt_falcon3_tiiuae_falcon3_1b_base_clm_hf"], "pcc": 0.99},
    ),
    (
        Add227,
        [((1, 8, 522, 522), torch.float32)],
        {"model_names": ["pt_falcon3_tiiuae_falcon3_1b_base_clm_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 522, 2048), torch.float32), ((1, 522, 2048), torch.float32)],
        {"model_names": ["pt_falcon3_tiiuae_falcon3_1b_base_clm_hf"], "pcc": 0.99},
    ),
    (
        Add17,
        [((1, 19200, 64), torch.float32)],
        {"model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 64, 15, 20), torch.float32), ((64, 1, 1), torch.float32)],
        {"model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"], "pcc": 0.99},
    ),
    (
        Add17,
        [((1, 300, 64), torch.float32)],
        {"model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 19200, 64), torch.float32), ((1, 19200, 64), torch.float32)],
        {"model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"], "pcc": 0.99},
    ),
    (
        Add18,
        [((1, 19200, 256), torch.float32)],
        {"model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 256, 120, 160), torch.float32), ((256, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf",
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add19,
        [((1, 4800, 128), torch.float32)],
        {"model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 128, 15, 20), torch.float32), ((128, 1, 1), torch.float32)],
        {"model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"], "pcc": 0.99},
    ),
    (
        Add19,
        [((1, 300, 128), torch.float32)],
        {"model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 4800, 128), torch.float32), ((1, 4800, 128), torch.float32)],
        {"model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"], "pcc": 0.99},
    ),
    (
        Add20,
        [((1, 4800, 512), torch.float32)],
        {"model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 512, 60, 80), torch.float32), ((512, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf",
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 64, 60, 80), torch.float32), ((64, 1, 1), torch.float32)],
        {"model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 320, 30, 40), torch.float32), ((320, 1, 1), torch.float32)],
        {"model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"], "pcc": 0.99},
    ),
    (
        Add51,
        [((1, 1200, 320), torch.float32)],
        {"model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 320, 15, 20), torch.float32), ((320, 1, 1), torch.float32)],
        {"model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"], "pcc": 0.99},
    ),
    (
        Add51,
        [((1, 300, 320), torch.float32)],
        {"model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1200, 320), torch.float32), ((1, 1200, 320), torch.float32)],
        {"model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"], "pcc": 0.99},
    ),
    (
        Add45,
        [((1, 1200, 1280), torch.float32)],
        {"model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1280, 30, 40), torch.float32), ((1280, 1, 1), torch.float32)],
        {"model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 64, 30, 40), torch.float32), ((64, 1, 1), torch.float32)],
        {"model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"], "pcc": 0.99},
    ),
    (
        Add20,
        [((1, 300, 512), torch.float32)],
        {"model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 300, 512), torch.float32), ((1, 300, 512), torch.float32)],
        {"model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"], "pcc": 0.99},
    ),
    (
        Add22,
        [((1, 300, 2048), torch.float32)],
        {"model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 2048, 15, 20), torch.float32), ((2048, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf",
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 32, 30, 40), torch.float32), ((32, 1, 1), torch.float32)],
        {"model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 2, 30, 40), torch.float32), ((2, 1, 1), torch.float32)],
        {"model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 64, 30, 40), torch.float32), ((1, 64, 30, 40), torch.float32)],
        {"model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 32, 60, 80), torch.float32), ((32, 1, 1), torch.float32)],
        {"model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 2, 60, 80), torch.float32), ((2, 1, 1), torch.float32)],
        {"model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 64, 60, 80), torch.float32), ((1, 64, 60, 80), torch.float32)],
        {"model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 32, 120, 160), torch.float32), ((32, 1, 1), torch.float32)],
        {"model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 2, 120, 160), torch.float32), ((2, 1, 1), torch.float32)],
        {"model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 64, 480, 640), torch.float32), ((64, 1, 1), torch.float32)],
        {"model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1, 480, 640), torch.float32), ((1, 1, 1), torch.float32)],
        {"model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((18,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": [
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add228,
        [((18,), torch.float32)],
        {
            "model_names": [
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 18, 56, 56), torch.float32), ((1, 18, 56, 56), torch.float32)],
        {
            "model_names": [
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 36, 28, 28), torch.float32), ((36, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 36, 28, 28), torch.float32), ((1, 36, 28, 28), torch.float32)],
        {
            "model_names": [
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 18, 28, 28), torch.float32), ((18, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 18, 14, 14), torch.float32), ((18, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 36, 14, 14), torch.float32), ((36, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 144, 7, 7), torch.float32), ((144, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 144, 7, 7), torch.float32), ((1, 144, 7, 7), torch.float32)],
        {
            "model_names": [
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 126, 7, 7), torch.float32), ((126, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 3072, 1, 9), torch.float32), ((3072, 1, 1), torch.float32)],
        {"model_names": ["pt_mamba_state_spaces_mamba_790m_hf_clm_hf"], "pcc": 0.99},
    ),
    (
        Add15,
        [((1, 6, 3072), torch.float32)],
        {"model_names": ["pt_mamba_state_spaces_mamba_790m_hf_clm_hf"], "pcc": 0.99},
    ),
    (
        Add13,
        [((1, 6, 3072), torch.float32)],
        {"model_names": ["pt_mamba_state_spaces_mamba_790m_hf_clm_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 3072, 16), torch.float32), ((1, 3072, 16), torch.float32)],
        {"model_names": ["pt_mamba_state_spaces_mamba_790m_hf_clm_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 3072, 6), torch.float32), ((1, 3072, 6), torch.float32)],
        {"model_names": ["pt_mamba_state_spaces_mamba_790m_hf_clm_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 6, 1536), torch.float32), ((1, 6, 1536), torch.float32)],
        {"model_names": ["pt_mamba_state_spaces_mamba_790m_hf_clm_hf"], "pcc": 0.99},
    ),
    (
        Add112,
        [((1, 1024, 196), torch.float32)],
        {
            "model_names": ["pt_mlp_mixer_mixer_l16_224_img_cls_timm", "pt_mlp_mixer_mixer_l16_224_in21k_img_cls_timm"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 196, 1024), torch.float32), ((1, 196, 1024), torch.float32)],
        {
            "model_names": ["pt_mlp_mixer_mixer_l16_224_img_cls_timm", "pt_mlp_mixer_mixer_l16_224_in21k_img_cls_timm"],
            "pcc": 0.99,
        },
    ),
    (
        Add24,
        [((1, 196, 4096), torch.float32)],
        {
            "model_names": ["pt_mlp_mixer_mixer_l16_224_img_cls_timm", "pt_mlp_mixer_mixer_l16_224_in21k_img_cls_timm"],
            "pcc": 0.99,
        },
    ),
    (
        Add21,
        [((1, 196, 1024), torch.float32)],
        {
            "model_names": ["pt_mlp_mixer_mixer_l16_224_img_cls_timm", "pt_mlp_mixer_mixer_l16_224_in21k_img_cls_timm"],
            "pcc": 0.99,
        },
    ),
    (Add99, [((1, 9), torch.float32)], {"model_names": ["pt_mobilenetv1_basic_img_cls_torchvision"], "pcc": 0.99}),
    (
        Add13,
        [((1, 1280, 1, 1), torch.float32)],
        {"model_names": ["pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm"], "pcc": 0.99},
    ),
    (Add22, [((1, 7, 2048), torch.float32)], {"model_names": ["pt_phi1_microsoft_phi_1_clm_hf"], "pcc": 0.99}),
    (
        Add0,
        [((1, 32, 7, 32), torch.float32), ((1, 32, 7, 32), torch.float32)],
        {"model_names": ["pt_phi1_microsoft_phi_1_clm_hf"], "pcc": 0.99},
    ),
    (Add229, [((1, 1, 1, 7), torch.float32)], {"model_names": ["pt_phi1_microsoft_phi_1_clm_hf"], "pcc": 0.99}),
    (
        Add0,
        [((1, 1, 7, 7), torch.float32), ((1, 1, 7, 7), torch.float32)],
        {"model_names": ["pt_phi1_microsoft_phi_1_clm_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 32, 7, 7), torch.float32), ((1, 1, 7, 7), torch.float32)],
        {"model_names": ["pt_phi1_microsoft_phi_1_clm_hf"], "pcc": 0.99},
    ),
    (Add64, [((1, 7, 8192), torch.float32)], {"model_names": ["pt_phi1_microsoft_phi_1_clm_hf"], "pcc": 0.99}),
    (
        Add0,
        [((1, 7, 2048), torch.float32), ((1, 7, 2048), torch.float32)],
        {"model_names": ["pt_phi1_microsoft_phi_1_clm_hf"], "pcc": 0.99},
    ),
    (Add179, [((1, 7, 51200), torch.float32)], {"model_names": ["pt_phi1_microsoft_phi_1_clm_hf"], "pcc": 0.99}),
    (Add65, [((1, 29, 1536), torch.float32)], {"model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99}),
    (
        Add0,
        [((1, 12, 29, 128), torch.float32), ((1, 12, 29, 128), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99},
    ),
    (Add18, [((1, 29, 256), torch.float32)], {"model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99}),
    (
        Add0,
        [((1, 2, 29, 128), torch.float32), ((1, 2, 29, 128), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99},
    ),
    (Add180, [((1, 12, 29, 29), torch.float32)], {"model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99}),
    (
        Add0,
        [((1, 29, 1536), torch.float32), ((1, 29, 1536), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 48, 80, 80), torch.float32), ((48, 1, 1), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5m_img_cls_torchhub_320x320"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 48, 80, 80), torch.float32), ((1, 48, 80, 80), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5m_img_cls_torchhub_320x320"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 96, 40, 40), torch.float32), ((96, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5m_img_cls_torchhub_320x320",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 96, 40, 40), torch.float32), ((1, 96, 40, 40), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5m_img_cls_torchhub_320x320"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 192, 20, 20), torch.float32), ((1, 192, 20, 20), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5m_img_cls_torchhub_320x320"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 768, 10, 10), torch.float32), ((768, 1, 1), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5m_img_cls_torchhub_320x320"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 384, 10, 10), torch.float32), ((1, 384, 10, 10), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5m_img_cls_torchhub_320x320"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 384, 20, 20), torch.float32), ((1, 384, 20, 20), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5m_img_cls_torchhub_640x640"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 64, 224, 320), torch.float32), ((64, 1, 1), torch.float32)],
        {"model_names": ["pt_yolo_v6_yolov6l_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 128, 112, 160), torch.float32), ((128, 1, 1), torch.float32)],
        {"model_names": ["pt_yolo_v6_yolov6l_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 256, 56, 80), torch.float32), ((256, 1, 1), torch.float32)],
        {"model_names": ["pt_yolo_v6_yolov6l_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 512, 28, 40), torch.float32), ((512, 1, 1), torch.float32)],
        {"model_names": ["pt_yolo_v6_yolov6l_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1024, 14, 20), torch.float32), ((1024, 1, 1), torch.float32)],
        {"model_names": ["pt_yolo_v6_yolov6l_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 68, 56, 80), torch.float32), ((68, 1, 1), torch.float32)],
        {"model_names": ["pt_yolo_v6_yolov6l_obj_det_torchhub", "pt_yolo_v6_yolov6m_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 68, 28, 40), torch.float32), ((68, 1, 1), torch.float32)],
        {"model_names": ["pt_yolo_v6_yolov6l_obj_det_torchhub", "pt_yolo_v6_yolov6m_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 68, 14, 20), torch.float32), ((68, 1, 1), torch.float32)],
        {"model_names": ["pt_yolo_v6_yolov6l_obj_det_torchhub", "pt_yolo_v6_yolov6m_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 16, 208, 208), torch.float32), ((16, 1, 1), torch.float32)],
        {"model_names": ["pt_yolox_yolox_nano_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 16, 104, 104), torch.float32), ((16, 1, 1), torch.float32)],
        {"model_names": ["pt_yolox_yolox_nano_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 32, 104, 104), torch.float32), ((32, 1, 1), torch.float32)],
        {"model_names": ["pt_yolox_yolox_nano_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 16, 104, 104), torch.float32), ((1, 16, 104, 104), torch.float32)],
        {"model_names": ["pt_yolox_yolox_nano_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 32, 52, 52), torch.float32), ((32, 1, 1), torch.float32)],
        {"model_names": ["pt_yolox_yolox_nano_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 64, 52, 52), torch.float32), ((64, 1, 1), torch.float32)],
        {"model_names": ["pt_yolox_yolox_nano_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 32, 52, 52), torch.float32), ((1, 32, 52, 52), torch.float32)],
        {"model_names": ["pt_yolox_yolox_nano_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 64, 26, 26), torch.float32), ((64, 1, 1), torch.float32)],
        {"model_names": ["pt_yolox_yolox_nano_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 128, 26, 26), torch.float32), ((128, 1, 1), torch.float32)],
        {"model_names": ["pt_yolox_yolox_nano_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 64, 26, 26), torch.float32), ((1, 64, 26, 26), torch.float32)],
        {"model_names": ["pt_yolox_yolox_nano_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 128, 13, 13), torch.float32), ((128, 1, 1), torch.float32)],
        {"model_names": ["pt_yolox_yolox_nano_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 64, 13, 13), torch.float32), ((64, 1, 1), torch.float32)],
        {"model_names": ["pt_yolox_yolox_nano_obj_det_torchhub"], "pcc": 0.99},
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
        [((1, 384, 56, 56), torch.float32), ((384, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 768, 28, 28), torch.float32), ((768, 1, 1), torch.float32)],
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
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b5_img_cls_timm",
                "pt_regnet_facebook_regnet_y_040_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 3072, 1, 1), torch.float32), ((3072, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add21,
        [((1, 1024), torch.float32)],
        {
            "model_names": [
                "pd_googlenet_base_img_cls_paddlemodels",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add188,
        [((1, 12, 97), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 14, 128), torch.float32), ((1, 14, 128), torch.float32)],
        {"model_names": ["pt_albert_twmkn9_albert_base_v2_squad2_qa_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((14, 1), torch.float32), ((1,), torch.float32)],
        {"model_names": ["pt_albert_twmkn9_albert_base_v2_squad2_qa_hf"], "pcc": 0.99},
    ),
    (
        Add13,
        [((1, 588, 1), torch.float32)],
        {"model_names": ["pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 16, 588, 128), torch.float32), ((1, 16, 588, 128), torch.float32)],
        {"model_names": ["pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf"], "pcc": 0.99},
    ),
    (
        Add230,
        [((1, 16, 588, 588), torch.float32)],
        {"model_names": ["pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 588, 2048), torch.float32), ((1, 588, 2048), torch.float32)],
        {"model_names": ["pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 32, 130, 130), torch.float32), ((32, 1, 1), torch.float32)],
        {"model_names": ["pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 16, 130, 130), torch.float32), ((16, 1, 1), torch.float32)],
        {"model_names": ["pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 96, 130, 130), torch.float32), ((96, 1, 1), torch.float32)],
        {"model_names": ["pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 96, 65, 65), torch.float32), ((96, 1, 1), torch.float32)],
        {"model_names": ["pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 24, 65, 65), torch.float32), ((24, 1, 1), torch.float32)],
        {"model_names": ["pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 144, 65, 65), torch.float32), ((144, 1, 1), torch.float32)],
        {"model_names": ["pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 24, 65, 65), torch.float32), ((1, 24, 65, 65), torch.float32)],
        {"model_names": ["pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 144, 33, 33), torch.float32), ((144, 1, 1), torch.float32)],
        {"model_names": ["pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 48, 33, 33), torch.float32), ((48, 1, 1), torch.float32)],
        {"model_names": ["pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 288, 33, 33), torch.float32), ((288, 1, 1), torch.float32)],
        {"model_names": ["pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 48, 33, 33), torch.float32), ((1, 48, 33, 33), torch.float32)],
        {"model_names": ["pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 288, 17, 17), torch.float32), ((288, 1, 1), torch.float32)],
        {"model_names": ["pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 88, 17, 17), torch.float32), ((88, 1, 1), torch.float32)],
        {"model_names": ["pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((528,), torch.float32), ((1,), torch.float32)],
        {"model_names": ["pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add231,
        [((528,), torch.float32)],
        {"model_names": ["pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 528, 17, 17), torch.float32), ((528, 1, 1), torch.float32)],
        {"model_names": ["pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 88, 17, 17), torch.float32), ((1, 88, 17, 17), torch.float32)],
        {"model_names": ["pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 120, 17, 17), torch.float32), ((120, 1, 1), torch.float32)],
        {"model_names": ["pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 720, 17, 17), torch.float32), ((720, 1, 1), torch.float32)],
        {"model_names": ["pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 120, 17, 17), torch.float32), ((1, 120, 17, 17), torch.float32)],
        {"model_names": ["pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 720, 9, 9), torch.float32), ((720, 1, 1), torch.float32)],
        {"model_names": ["pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 208, 9, 9), torch.float32), ((208, 1, 1), torch.float32)],
        {"model_names": ["pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1248, 9, 9), torch.float32), ((1248, 1, 1), torch.float32)],
        {"model_names": ["pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 208, 9, 9), torch.float32), ((1, 208, 9, 9), torch.float32)],
        {"model_names": ["pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 352, 9, 9), torch.float32), ((352, 1, 1), torch.float32)],
        {"model_names": ["pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1280, 9, 9), torch.float32), ((1280, 1, 1), torch.float32)],
        {"model_names": ["pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 256, 16, 16), torch.float32), ((1, 256, 16, 16), torch.float32)],
        {"model_names": ["pt_fpn_base_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 256, 64, 64), torch.float32), ((1, 256, 64, 64), torch.float32)],
        {"model_names": ["pt_fpn_base_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 768, 8, 32), torch.float32), ((768, 1, 1), torch.float32)],
        {"model_names": ["pt_mgp_alibaba_damo_mgp_str_base_scene_text_recognition_hf"], "pcc": 0.99},
    ),
    (
        Add232,
        [((1, 257, 768), torch.float32)],
        {"model_names": ["pt_mgp_alibaba_damo_mgp_str_base_scene_text_recognition_hf"], "pcc": 0.99},
    ),
    (
        Add77,
        [((1, 257, 2304), torch.float32)],
        {"model_names": ["pt_mgp_alibaba_damo_mgp_str_base_scene_text_recognition_hf"], "pcc": 0.99},
    ),
    (
        Add14,
        [((1, 257, 768), torch.float32)],
        {"model_names": ["pt_mgp_alibaba_damo_mgp_str_base_scene_text_recognition_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 257, 768), torch.float32), ((1, 257, 768), torch.float32)],
        {"model_names": ["pt_mgp_alibaba_damo_mgp_str_base_scene_text_recognition_hf"], "pcc": 0.99},
    ),
    (
        Add15,
        [((1, 257, 3072), torch.float32)],
        {"model_names": ["pt_mgp_alibaba_damo_mgp_str_base_scene_text_recognition_hf"], "pcc": 0.99},
    ),
    (
        Add233,
        [((1, 27, 38), torch.float32)],
        {"model_names": ["pt_mgp_alibaba_damo_mgp_str_base_scene_text_recognition_hf"], "pcc": 0.99},
    ),
    (
        Add234,
        [((1, 27, 50257), torch.float32)],
        {"model_names": ["pt_mgp_alibaba_damo_mgp_str_base_scene_text_recognition_hf"], "pcc": 0.99},
    ),
    (
        Add235,
        [((1, 27, 30522), torch.float32)],
        {"model_names": ["pt_mgp_alibaba_damo_mgp_str_base_scene_text_recognition_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 24, 80, 80), torch.float32), ((24, 1, 1), torch.float32)],
        {"model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 16, 80, 80), torch.float32), ((16, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_320x320",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 24, 40, 40), torch.float32), ((24, 1, 1), torch.float32)],
        {"model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 144, 40, 40), torch.float32), ((144, 1, 1), torch.float32)],
        {"model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 24, 40, 40), torch.float32), ((1, 24, 40, 40), torch.float32)],
        {"model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 144, 20, 20), torch.float32), ((144, 1, 1), torch.float32)],
        {"model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 24, 20, 20), torch.float32), ((24, 1, 1), torch.float32)],
        {"model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 24, 20, 20), torch.float32), ((1, 24, 20, 20), torch.float32)],
        {"model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 144, 10, 10), torch.float32), ((144, 1, 1), torch.float32)],
        {"model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 48, 10, 10), torch.float32), ((48, 1, 1), torch.float32)],
        {"model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 288, 10, 10), torch.float32), ((288, 1, 1), torch.float32)],
        {"model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 48, 10, 10), torch.float32), ((1, 48, 10, 10), torch.float32)],
        {"model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 72, 10, 10), torch.float32), ((72, 1, 1), torch.float32)],
        {"model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 432, 10, 10), torch.float32), ((432, 1, 1), torch.float32)],
        {"model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 72, 10, 10), torch.float32), ((1, 72, 10, 10), torch.float32)],
        {"model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 432, 5, 5), torch.float32), ((432, 1, 1), torch.float32)],
        {"model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 120, 5, 5), torch.float32), ((120, 1, 1), torch.float32)],
        {"model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 720, 5, 5), torch.float32), ((720, 1, 1), torch.float32)],
        {"model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 120, 5, 5), torch.float32), ((1, 120, 5, 5), torch.float32)],
        {"model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 240, 5, 5), torch.float32), ((240, 1, 1), torch.float32)],
        {"model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1280, 5, 5), torch.float32), ((1280, 1, 1), torch.float32)],
        {"model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add13,
        [((1, 16, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add13,
        [((1, 96, 28, 28), torch.float32)],
        {
            "model_names": [
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add13,
        [((1, 96, 14, 14), torch.float32)],
        {
            "model_names": [
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 40, 14, 14), torch.float32), ((1, 40, 14, 14), torch.float32)],
        {
            "model_names": [
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add13,
        [((1, 120, 14, 14), torch.float32)],
        {
            "model_names": [
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 144, 14, 14), torch.float32), ((144, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add13,
        [((1, 144, 14, 14), torch.float32)],
        {
            "model_names": [
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add13,
        [((1, 288, 14, 14), torch.float32)],
        {
            "model_names": [
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add13,
        [((1, 288, 7, 7), torch.float32)],
        {
            "model_names": [
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 96, 7, 7), torch.float32), ((96, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add13,
        [((1, 576, 7, 7), torch.float32)],
        {
            "model_names": [
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add13,
        [((1, 576, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 96, 7, 7), torch.float32), ((1, 96, 7, 7), torch.float32)],
        {
            "model_names": [
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 32, 768), torch.float32), ((1, 32, 768), torch.float32)],
        {"model_names": ["pt_opt_facebook_opt_125m_qa_hf", "pt_opt_facebook_opt_125m_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Add14,
        [((1, 32, 768), torch.float32)],
        {"model_names": ["pt_opt_facebook_opt_125m_qa_hf", "pt_opt_facebook_opt_125m_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 12, 32, 32), torch.float32), ((1, 1, 32, 32), torch.float32)],
        {"model_names": ["pt_opt_facebook_opt_125m_qa_hf", "pt_opt_facebook_opt_125m_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Add15,
        [((32, 3072), torch.float32)],
        {"model_names": ["pt_opt_facebook_opt_125m_qa_hf", "pt_opt_facebook_opt_125m_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Add14,
        [((32, 768), torch.float32)],
        {"model_names": ["pt_opt_facebook_opt_125m_qa_hf", "pt_opt_facebook_opt_125m_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((32, 768), torch.float32), ((32, 768), torch.float32)],
        {"model_names": ["pt_opt_facebook_opt_125m_qa_hf", "pt_opt_facebook_opt_125m_seq_cls_hf"], "pcc": 0.99},
    ),
    (Add98, [((1, 32, 256, 256), torch.float32)], {"model_names": ["pt_phi1_microsoft_phi_1_seq_cls_hf"], "pcc": 0.99}),
    (
        Add171,
        [((1, 35, 896), torch.float32)],
        {"model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 14, 35, 64), torch.float32), ((1, 14, 35, 64), torch.float32)],
        {"model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Add19,
        [((1, 35, 128), torch.float32)],
        {"model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 2, 35, 64), torch.float32), ((1, 2, 35, 64), torch.float32)],
        {"model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Add66,
        [((1, 14, 35, 35), torch.float32)],
        {"model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 35, 896), torch.float32), ((1, 35, 896), torch.float32)],
        {"model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 72, 112, 112), torch.float32), ((72, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_regnet_regnet_x_1_6gf_img_cls_torchvision",
                "pt_regnet_regnet_y_3_2gf_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 72, 56, 56), torch.float32), ((1, 72, 56, 56), torch.float32)],
        {
            "model_names": [
                "pt_regnet_regnet_x_1_6gf_img_cls_torchvision",
                "pt_regnet_regnet_y_3_2gf_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 168, 28, 28), torch.float32), ((168, 1, 1), torch.float32)],
        {"model_names": ["pt_regnet_regnet_x_1_6gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 168, 28, 28), torch.float32), ((1, 168, 28, 28), torch.float32)],
        {"model_names": ["pt_regnet_regnet_x_1_6gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((408,), torch.float32), ((1,), torch.float32)],
        {"model_names": ["pt_regnet_regnet_x_1_6gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (Add236, [((408,), torch.float32)], {"model_names": ["pt_regnet_regnet_x_1_6gf_img_cls_torchvision"], "pcc": 0.99}),
    (
        Add0,
        [((1, 408, 14, 14), torch.float32), ((408, 1, 1), torch.float32)],
        {"model_names": ["pt_regnet_regnet_x_1_6gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 408, 28, 28), torch.float32), ((408, 1, 1), torch.float32)],
        {"model_names": ["pt_regnet_regnet_x_1_6gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 408, 14, 14), torch.float32), ((1, 408, 14, 14), torch.float32)],
        {"model_names": ["pt_regnet_regnet_x_1_6gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((912,), torch.float32), ((1,), torch.float32)],
        {"model_names": ["pt_regnet_regnet_x_1_6gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (Add237, [((912,), torch.float32)], {"model_names": ["pt_regnet_regnet_x_1_6gf_img_cls_torchvision"], "pcc": 0.99}),
    (
        Add0,
        [((1, 912, 7, 7), torch.float32), ((912, 1, 1), torch.float32)],
        {"model_names": ["pt_regnet_regnet_x_1_6gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 912, 14, 14), torch.float32), ((912, 1, 1), torch.float32)],
        {"model_names": ["pt_regnet_regnet_x_1_6gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 912, 7, 7), torch.float32), ((1, 912, 7, 7), torch.float32)],
        {"model_names": ["pt_regnet_regnet_x_1_6gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((216,), torch.float32), ((1,), torch.float32)],
        {"model_names": ["pt_regnet_regnet_y_3_2gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (Add238, [((216,), torch.float32)], {"model_names": ["pt_regnet_regnet_y_3_2gf_img_cls_torchvision"], "pcc": 0.99}),
    (
        Add0,
        [((1, 216, 28, 28), torch.float32), ((216, 1, 1), torch.float32)],
        {"model_names": ["pt_regnet_regnet_y_3_2gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 216, 56, 56), torch.float32), ((216, 1, 1), torch.float32)],
        {"model_names": ["pt_regnet_regnet_y_3_2gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 216, 1, 1), torch.float32), ((216, 1, 1), torch.float32)],
        {"model_names": ["pt_regnet_regnet_y_3_2gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 216, 28, 28), torch.float32), ((1, 216, 28, 28), torch.float32)],
        {"model_names": ["pt_regnet_regnet_y_3_2gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 54, 1, 1), torch.float32), ((54, 1, 1), torch.float32)],
        {"model_names": ["pt_regnet_regnet_y_3_2gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1512,), torch.float32), ((1,), torch.float32)],
        {"model_names": ["pt_regnet_regnet_y_3_2gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add239,
        [((1512,), torch.float32)],
        {"model_names": ["pt_regnet_regnet_y_3_2gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1512, 7, 7), torch.float32), ((1512, 1, 1), torch.float32)],
        {"model_names": ["pt_regnet_regnet_y_3_2gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1512, 14, 14), torch.float32), ((1512, 1, 1), torch.float32)],
        {"model_names": ["pt_regnet_regnet_y_3_2gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1512, 1, 1), torch.float32), ((1512, 1, 1), torch.float32)],
        {"model_names": ["pt_regnet_regnet_y_3_2gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1512, 7, 7), torch.float32), ((1, 1512, 7, 7), torch.float32)],
        {"model_names": ["pt_regnet_regnet_y_3_2gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 256, 120, 160), torch.float32), ((1, 256, 120, 160), torch.float32)],
        {"model_names": ["pt_retinanet_retinanet_rn50fpn_obj_det_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 512, 60, 80), torch.float32), ((1, 512, 60, 80), torch.float32)],
        {"model_names": ["pt_retinanet_retinanet_rn50fpn_obj_det_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1024, 30, 40), torch.float32), ((1024, 1, 1), torch.float32)],
        {"model_names": ["pt_retinanet_retinanet_rn50fpn_obj_det_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1024, 30, 40), torch.float32), ((1, 1024, 30, 40), torch.float32)],
        {"model_names": ["pt_retinanet_retinanet_rn50fpn_obj_det_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 2048, 15, 20), torch.float32), ((1, 2048, 15, 20), torch.float32)],
        {"model_names": ["pt_retinanet_retinanet_rn50fpn_obj_det_hf"], "pcc": 0.99},
    ),
    (Add5, [((64, 64, 384), torch.float32)], {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99}),
    (
        Add0,
        [((64, 4, 64, 64), torch.float32), ((1, 4, 64, 64), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99},
    ),
    (Add19, [((64, 64, 128), torch.float32)], {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99}),
    (
        Add0,
        [((1, 64, 64, 128), torch.float32), ((1, 64, 64, 128), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add20,
        [((1, 64, 64, 512), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add19,
        [((1, 64, 64, 128), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add73,
        [((1, 64, 4, 64, 64), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99},
    ),
    (Add14, [((16, 64, 768), torch.float32)], {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99}),
    (
        Add0,
        [((16, 8, 64, 64), torch.float32), ((1, 8, 64, 64), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99},
    ),
    (Add18, [((16, 64, 256), torch.float32)], {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99}),
    (
        Add0,
        [((1, 32, 32, 256), torch.float32), ((1, 32, 32, 256), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add21,
        [((1, 32, 32, 1024), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add18,
        [((1, 32, 32, 256), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add74,
        [((1, 16, 8, 64, 64), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99},
    ),
    (Add65, [((4, 64, 1536), torch.float32)], {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99}),
    (
        Add0,
        [((4, 16, 64, 64), torch.float32), ((1, 16, 64, 64), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99},
    ),
    (Add20, [((4, 64, 512), torch.float32)], {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99}),
    (
        Add0,
        [((1, 16, 16, 512), torch.float32), ((1, 16, 16, 512), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add22,
        [((1, 16, 16, 2048), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add20,
        [((1, 16, 16, 512), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add76,
        [((1, 4, 16, 64, 64), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99},
    ),
    (Add15, [((1, 64, 3072), torch.float32)], {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99}),
    (
        Add0,
        [((1, 32, 64, 64), torch.float32), ((1, 32, 64, 64), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99},
    ),
    (Add21, [((1, 64, 1024), torch.float32)], {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99}),
    (
        Add0,
        [((1, 8, 8, 1024), torch.float32), ((1, 8, 8, 1024), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add24,
        [((1, 8, 8, 4096), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add21,
        [((1, 8, 8, 1024), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 16, 1, 1), torch.float32), ((1, 16, 1, 1), torch.float32)],
        {"model_names": ["pt_t5_google_flan_t5_large_text_gen_hf", "pt_t5_t5_large_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Add78,
        [((1, 16, 61, 61), torch.float32)],
        {"model_names": ["pt_t5_google_flan_t5_large_text_gen_hf", "pt_t5_t5_large_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 16, 61, 61), torch.float32), ((1, 16, 61, 61), torch.float32)],
        {"model_names": ["pt_t5_google_flan_t5_large_text_gen_hf", "pt_t5_t5_large_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 61, 1024), torch.float32), ((1, 61, 1024), torch.float32)],
        {"model_names": ["pt_t5_google_flan_t5_large_text_gen_hf", "pt_t5_t5_large_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Add240,
        [((1, 16, 1, 61), torch.float32)],
        {"model_names": ["pt_t5_google_flan_t5_large_text_gen_hf", "pt_t5_t5_large_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 64, 512, 512), torch.float32), ((64, 1, 1), torch.float32)],
        {"model_names": ["pt_vgg19_unet_default_sem_seg_github"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 128, 256, 256), torch.float32), ((128, 1, 1), torch.float32)],
        {"model_names": ["pt_vgg19_unet_default_sem_seg_github"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 512, 32, 32), torch.float32), ((512, 1, 1), torch.float32)],
        {"model_names": ["pt_vgg19_unet_default_sem_seg_github"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1, 512, 512), torch.float32), ((1, 1, 1), torch.float32)],
        {"model_names": ["pt_vgg19_unet_default_sem_seg_github"], "pcc": 0.99},
    ),
    (
        Add20,
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
        Add241,
        [((1, 1500, 512), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_base_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Add20,
        [((1, 1500, 512), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_base_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1500, 512), torch.float32), ((1, 1500, 512), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_base_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Add22,
        [((1, 1500, 2048), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_base_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Add22,
        [((1, 1, 2048), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_base_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 16, 80, 80), torch.float32), ((1, 16, 80, 80), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5n_img_cls_torchhub_320x320"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 32, 40, 40), torch.float32), ((32, 1, 1), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5n_img_cls_torchhub_320x320"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 32, 40, 40), torch.float32), ((1, 32, 40, 40), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5n_img_cls_torchhub_320x320"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 64, 20, 20), torch.float32), ((1, 64, 20, 20), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5n_img_cls_torchhub_320x320"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 128, 10, 10), torch.float32), ((128, 1, 1), torch.float32)],
        {
            "model_names": ["pt_yolo_v5_yolov5n_img_cls_torchhub_320x320", "pt_ssd300_resnet50_base_img_cls_torchhub"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 128, 10, 10), torch.float32), ((1, 128, 10, 10), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5n_img_cls_torchhub_320x320"], "pcc": 0.99},
    ),
    (Add242, [((48,), torch.float32)], {"model_names": ["pt_yolo_v6_yolov6m_obj_det_torchhub"], "pcc": 0.99}),
    (
        Add0,
        [((1, 48, 224, 320), torch.float32), ((48, 1, 1), torch.float32)],
        {"model_names": ["pt_yolo_v6_yolov6m_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 48, 224, 320), torch.float32), ((1, 48, 224, 320), torch.float32)],
        {"model_names": ["pt_yolo_v6_yolov6m_obj_det_torchhub"], "pcc": 0.99},
    ),
    (Add243, [((96,), torch.float32)], {"model_names": ["pt_yolo_v6_yolov6m_obj_det_torchhub"], "pcc": 0.99}),
    (
        Add0,
        [((1, 96, 112, 160), torch.float32), ((96, 1, 1), torch.float32)],
        {"model_names": ["pt_yolo_v6_yolov6m_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 96, 112, 160), torch.float32), ((1, 96, 112, 160), torch.float32)],
        {"model_names": ["pt_yolo_v6_yolov6m_obj_det_torchhub"], "pcc": 0.99},
    ),
    (Add244, [((192,), torch.float32)], {"model_names": ["pt_yolo_v6_yolov6m_obj_det_torchhub"], "pcc": 0.99}),
    (
        Add0,
        [((1, 192, 56, 80), torch.float32), ((192, 1, 1), torch.float32)],
        {"model_names": ["pt_yolo_v6_yolov6m_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 192, 56, 80), torch.float32), ((1, 192, 56, 80), torch.float32)],
        {"model_names": ["pt_yolo_v6_yolov6m_obj_det_torchhub"], "pcc": 0.99},
    ),
    (Add245, [((384,), torch.float32)], {"model_names": ["pt_yolo_v6_yolov6m_obj_det_torchhub"], "pcc": 0.99}),
    (
        Add0,
        [((1, 384, 28, 40), torch.float32), ((384, 1, 1), torch.float32)],
        {"model_names": ["pt_yolo_v6_yolov6m_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 384, 28, 40), torch.float32), ((1, 384, 28, 40), torch.float32)],
        {"model_names": ["pt_yolo_v6_yolov6m_obj_det_torchhub"], "pcc": 0.99},
    ),
    (Add246, [((768,), torch.float32)], {"model_names": ["pt_yolo_v6_yolov6m_obj_det_torchhub"], "pcc": 0.99}),
    (
        Add0,
        [((1, 768, 14, 20), torch.float32), ((768, 1, 1), torch.float32)],
        {"model_names": ["pt_yolo_v6_yolov6m_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 768, 14, 20), torch.float32), ((1, 768, 14, 20), torch.float32)],
        {"model_names": ["pt_yolo_v6_yolov6m_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 384, 14, 20), torch.float32), ((384, 1, 1), torch.float32)],
        {"model_names": ["pt_yolo_v6_yolov6m_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 192, 14, 20), torch.float32), ((192, 1, 1), torch.float32)],
        {"model_names": ["pt_yolo_v6_yolov6m_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 192, 28, 40), torch.float32), ((192, 1, 1), torch.float32)],
        {"model_names": ["pt_yolo_v6_yolov6m_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 96, 28, 40), torch.float32), ((96, 1, 1), torch.float32)],
        {"model_names": ["pt_yolo_v6_yolov6m_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 96, 56, 80), torch.float32), ((96, 1, 1), torch.float32)],
        {"model_names": ["pt_yolo_v6_yolov6m_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Add216,
        [((1, 384, 1024), torch.float32)],
        {"model_names": ["onnx_bert_phiyodr_bert_large_finetuned_squad2_qa_hf"], "pcc": 0.99},
    ),
    (
        Add247,
        [((1, 384, 4096), torch.float32)],
        {"model_names": ["onnx_bert_phiyodr_bert_large_finetuned_squad2_qa_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 12, 197, 197), torch.float32), ((1, 12, 197, 197), torch.float32)],
        {"model_names": ["pt_beit_microsoft_beit_base_patch16_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((2, 7, 512), torch.float32), ((1, 7, 512), torch.float32)],
        {"model_names": ["pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text"], "pcc": 0.99},
    ),
    (
        Add20,
        [((2, 7, 512), torch.float32)],
        {"model_names": ["pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text"], "pcc": 0.99},
    ),
    (
        Add248,
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
        Add22,
        [((2, 7, 2048), torch.float32)],
        {"model_names": ["pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text"], "pcc": 0.99},
    ),
    (
        Add249,
        [((1, 197, 384), torch.float32)],
        {"model_names": ["pt_deit_facebook_deit_small_patch16_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add5,
        [((1, 197, 384), torch.float32)],
        {"model_names": ["pt_deit_facebook_deit_small_patch16_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 197, 384), torch.float32), ((1, 197, 384), torch.float32)],
        {"model_names": ["pt_deit_facebook_deit_small_patch16_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add65,
        [((1, 197, 1536), torch.float32)],
        {"model_names": ["pt_deit_facebook_deit_small_patch16_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add250,
        [((1, 100, 251), torch.float32)],
        {"model_names": ["pt_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Add251,
        [((1, 100, 8, 14, 20), torch.float32)],
        {"model_names": ["pt_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Add13,
        [((100, 8, 1, 1), torch.float32)],
        {"model_names": ["pt_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((256, 768), torch.float32), ((768,), torch.float32)],
        {"model_names": ["pt_gpt_gpt2_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 12, 256, 256), torch.float32), ((1, 1, 1, 256), torch.float32)],
        {"model_names": ["pt_gpt_gpt2_text_gen_hf"], "pcc": 0.99},
    ),
    (Add14, [((256, 768), torch.float32)], {"model_names": ["pt_gpt_gpt2_text_gen_hf"], "pcc": 0.99}),
    (Add15, [((256, 3072), torch.float32)], {"model_names": ["pt_gpt_gpt2_text_gen_hf"], "pcc": 0.99}),
    (
        Add0,
        [((1, 40, 56, 56), torch.float32), ((40, 1, 1), torch.float32)],
        {"model_names": ["pt_hrnet_hrnetv2_w40_pose_estimation_osmr"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 40, 56, 56), torch.float32), ((1, 40, 56, 56), torch.float32)],
        {"model_names": ["pt_hrnet_hrnetv2_w40_pose_estimation_osmr"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 80, 28, 28), torch.float32), ((1, 80, 28, 28), torch.float32)],
        {"model_names": ["pt_hrnet_hrnetv2_w40_pose_estimation_osmr"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 320, 7, 7), torch.float32), ((1, 320, 7, 7), torch.float32)],
        {"model_names": ["pt_hrnet_hrnetv2_w40_pose_estimation_osmr"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 280, 7, 7), torch.float32), ((280, 1, 1), torch.float32)],
        {"model_names": ["pt_hrnet_hrnetv2_w40_pose_estimation_osmr"], "pcc": 0.99},
    ),
    (
        Add252,
        [((1, 21843), torch.float32)],
        {"model_names": ["pt_mlp_mixer_mixer_b16_224_goog_in21k_img_cls_timm"], "pcc": 0.99},
    ),
    (Add57, [((1, 512, 49), torch.float32)], {"model_names": ["pt_mlp_mixer_mixer_s32_224_img_cls_timm"], "pcc": 0.99}),
    (
        Add0,
        [((1, 49, 512), torch.float32), ((1, 49, 512), torch.float32)],
        {"model_names": ["pt_mlp_mixer_mixer_s32_224_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add22,
        [((1, 49, 2048), torch.float32)],
        {"model_names": ["pt_mlp_mixer_mixer_s32_224_img_cls_timm"], "pcc": 0.99},
    ),
    (Add20, [((1, 49, 512), torch.float32)], {"model_names": ["pt_mlp_mixer_mixer_s32_224_img_cls_timm"], "pcc": 0.99}),
    (
        Add13,
        [((1, 1024), torch.float32)],
        {"model_names": ["pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"], "pcc": 0.99},
    ),
    (Add22, [((1024, 2048), torch.float32)], {"model_names": ["pt_nbeats_seasionality_basis_clm_hf"], "pcc": 0.99}),
    (Add37, [((1024, 48), torch.float32)], {"model_names": ["pt_nbeats_seasionality_basis_clm_hf"], "pcc": 0.99}),
    (
        Add0,
        [((1024, 72), torch.float32), ((1024, 72), torch.float32)],
        {"model_names": ["pt_nbeats_seasionality_basis_clm_hf"], "pcc": 0.99},
    ),
    (
        Add20,
        [((1, 512, 512), torch.float32)],
        {"model_names": ["pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 256, 224, 224), torch.float32), ((256, 1, 1), torch.float32)],
        {"model_names": ["pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add18,
        [((1, 50176, 256), torch.float32)],
        {"model_names": ["pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add20,
        [((1, 50176, 512), torch.float32)],
        {"model_names": ["pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1088, 1, 1), torch.float32), ((1088, 1, 1), torch.float32)],
        {"model_names": ["pt_regnet_facebook_regnet_y_040_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1088, 7, 7), torch.float32), ((1, 1088, 7, 7), torch.float32)],
        {"model_names": ["pt_regnet_facebook_regnet_y_040_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 272, 1, 1), torch.float32), ((272, 1, 1), torch.float32)],
        {"model_names": ["pt_regnet_facebook_regnet_y_040_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((400,), torch.float32), ((1,), torch.float32)],
        {"model_names": ["pt_regnet_regnet_x_400mf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (Add253, [((400,), torch.float32)], {"model_names": ["pt_regnet_regnet_x_400mf_img_cls_torchvision"], "pcc": 0.99}),
    (
        Add0,
        [((1, 400, 7, 7), torch.float32), ((400, 1, 1), torch.float32)],
        {"model_names": ["pt_regnet_regnet_x_400mf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 400, 14, 14), torch.float32), ((400, 1, 1), torch.float32)],
        {"model_names": ["pt_regnet_regnet_x_400mf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 400, 7, 7), torch.float32), ((1, 400, 7, 7), torch.float32)],
        {"model_names": ["pt_regnet_regnet_x_400mf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 224, 112, 112), torch.float32), ((224, 1, 1), torch.float32)],
        {"model_names": ["pt_regnet_regnet_y_8gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 224, 56, 56), torch.float32), ((1, 224, 56, 56), torch.float32)],
        {"model_names": ["pt_regnet_regnet_y_8gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 56, 1, 1), torch.float32), ((56, 1, 1), torch.float32)],
        {"model_names": ["pt_regnet_regnet_y_8gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 64, 75, 75), torch.float32), ((64, 1, 1), torch.float32)],
        {"model_names": ["pt_ssd300_resnet50_base_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 128, 38, 38), torch.float32), ((128, 1, 1), torch.float32)],
        {"model_names": ["pt_ssd300_resnet50_base_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 512, 38, 38), torch.float32), ((512, 1, 1), torch.float32)],
        {"model_names": ["pt_ssd300_resnet50_base_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 512, 38, 38), torch.float32), ((1, 512, 38, 38), torch.float32)],
        {"model_names": ["pt_ssd300_resnet50_base_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1024, 38, 38), torch.float32), ((1024, 1, 1), torch.float32)],
        {"model_names": ["pt_ssd300_resnet50_base_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1024, 38, 38), torch.float32), ((1, 1024, 38, 38), torch.float32)],
        {"model_names": ["pt_ssd300_resnet50_base_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 16, 38, 38), torch.float32), ((16, 1, 1), torch.float32)],
        {"model_names": ["pt_ssd300_resnet50_base_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 512, 19, 19), torch.float32), ((512, 1, 1), torch.float32)],
        {"model_names": ["pt_ssd300_resnet50_base_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 24, 19, 19), torch.float32), ((24, 1, 1), torch.float32)],
        {"model_names": ["pt_ssd300_resnet50_base_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 256, 19, 19), torch.float32), ((256, 1, 1), torch.float32)],
        {"model_names": ["pt_ssd300_resnet50_base_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 24, 10, 10), torch.float32), ((24, 1, 1), torch.float32)],
        {"model_names": ["pt_ssd300_resnet50_base_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 256, 5, 5), torch.float32), ((256, 1, 1), torch.float32)],
        {"model_names": ["pt_ssd300_resnet50_base_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 24, 5, 5), torch.float32), ((24, 1, 1), torch.float32)],
        {"model_names": ["pt_ssd300_resnet50_base_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 128, 5, 5), torch.float32), ((128, 1, 1), torch.float32)],
        {"model_names": ["pt_ssd300_resnet50_base_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 256, 3, 3), torch.float32), ((256, 1, 1), torch.float32)],
        {"model_names": ["pt_ssd300_resnet50_base_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 16, 3, 3), torch.float32), ((16, 1, 1), torch.float32)],
        {"model_names": ["pt_ssd300_resnet50_base_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 128, 3, 3), torch.float32), ((128, 1, 1), torch.float32)],
        {"model_names": ["pt_ssd300_resnet50_base_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 324, 38, 38), torch.float32), ((324, 1, 1), torch.float32)],
        {"model_names": ["pt_ssd300_resnet50_base_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 486, 19, 19), torch.float32), ((486, 1, 1), torch.float32)],
        {"model_names": ["pt_ssd300_resnet50_base_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 486, 10, 10), torch.float32), ((486, 1, 1), torch.float32)],
        {"model_names": ["pt_ssd300_resnet50_base_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 486, 5, 5), torch.float32), ((486, 1, 1), torch.float32)],
        {"model_names": ["pt_ssd300_resnet50_base_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 324, 3, 3), torch.float32), ((324, 1, 1), torch.float32)],
        {"model_names": ["pt_ssd300_resnet50_base_img_cls_torchhub"], "pcc": 0.99},
    ),
    (Add254, [((1, 50, 768), torch.float32)], {"model_names": ["pt_vit_vit_b_32_img_cls_torchvision"], "pcc": 0.99}),
    (Add77, [((50, 1, 2304), torch.float32)], {"model_names": ["pt_vit_vit_b_32_img_cls_torchvision"], "pcc": 0.99}),
    (Add255, [((1, 12, 50, 50), torch.float32)], {"model_names": ["pt_vit_vit_b_32_img_cls_torchvision"], "pcc": 0.99}),
    (Add14, [((50, 768), torch.float32)], {"model_names": ["pt_vit_vit_b_32_img_cls_torchvision"], "pcc": 0.99}),
    (
        Add0,
        [((1, 50, 768), torch.float32), ((1, 50, 768), torch.float32)],
        {"model_names": ["pt_vit_vit_b_32_img_cls_torchvision"], "pcc": 0.99},
    ),
    (Add15, [((1, 50, 3072), torch.float32)], {"model_names": ["pt_vit_vit_b_32_img_cls_torchvision"], "pcc": 0.99}),
    (Add14, [((1, 50, 768), torch.float32)], {"model_names": ["pt_vit_vit_b_32_img_cls_torchvision"], "pcc": 0.99}),
    (
        Add0,
        [((1, 1, 384), torch.float32), ((1, 1, 384), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Add5,
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
        Add256,
        [((1, 1500, 384), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Add5,
        [((1, 1500, 384), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1500, 384), torch.float32), ((1, 1500, 384), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Add65,
        [((1, 1500, 1536), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Add65,
        [((1, 1, 1536), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 64, 240, 240), torch.float32), ((64, 1, 1), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5l_img_cls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 128, 120, 120), torch.float32), ((128, 1, 1), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5l_img_cls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 64, 120, 120), torch.float32), ((64, 1, 1), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5l_img_cls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 64, 120, 120), torch.float32), ((1, 64, 120, 120), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5l_img_cls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 256, 60, 60), torch.float32), ((256, 1, 1), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5l_img_cls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 128, 60, 60), torch.float32), ((128, 1, 1), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5l_img_cls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 128, 60, 60), torch.float32), ((1, 128, 60, 60), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5l_img_cls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 512, 30, 30), torch.float32), ((512, 1, 1), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5l_img_cls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 256, 30, 30), torch.float32), ((256, 1, 1), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5l_img_cls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 256, 30, 30), torch.float32), ((1, 256, 30, 30), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5l_img_cls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1024, 15, 15), torch.float32), ((1024, 1, 1), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5l_img_cls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 512, 15, 15), torch.float32), ((512, 1, 1), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5l_img_cls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 512, 15, 15), torch.float32), ((1, 512, 15, 15), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5l_img_cls_torchhub_480x480"], "pcc": 0.99},
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
