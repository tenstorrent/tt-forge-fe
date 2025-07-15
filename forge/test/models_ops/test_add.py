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
        self.add_constant("add1_const_1", shape=(1,), dtype=torch.float32)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add1_const_1"))
        return add_output_1


class Add2(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add2.weight_1", forge.Parameter(*(128,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add2.weight_1"))
        return add_output_1


class Add3(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add3.weight_1", forge.Parameter(*(312,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add3.weight_1"))
        return add_output_1


class Add4(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add4.weight_1", forge.Parameter(*(1248,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add4.weight_1"))
        return add_output_1


class Add5(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add5.weight_1", forge.Parameter(*(21128,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add5.weight_1"))
        return add_output_1


class Add6(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add6.weight_1", forge.Parameter(*(64,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add6.weight_1"))
        return add_output_1


class Add7(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add7.weight_1", forge.Parameter(*(256,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add7.weight_1"))
        return add_output_1


class Add8(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add8.weight_1", forge.Parameter(*(512,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add8.weight_1"))
        return add_output_1


class Add9(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add9.weight_1", forge.Parameter(*(1024,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add9.weight_1"))
        return add_output_1


class Add10(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add10.weight_1", forge.Parameter(*(2048,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add10.weight_1"))
        return add_output_1


class Add11(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add11.weight_1", forge.Parameter(*(1000,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add11.weight_1"))
        return add_output_1


class Add12(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add12_const_0", shape=(128, 128), dtype=torch.float32)

    def forward(self, add_input_1):
        add_output_1 = forge.op.Add("", self.get_constant("add12_const_0"), add_input_1)
        return add_output_1


class Add13(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add13.weight_1", forge.Parameter(*(4096,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add13.weight_1"))
        return add_output_1


class Add14(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add14.weight_1", forge.Parameter(*(2,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add14.weight_1"))
        return add_output_1


class Add15(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add15.weight_1", forge.Parameter(*(16,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add15.weight_1"))
        return add_output_1


class Add16(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add16.weight_1", forge.Parameter(*(32,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add16.weight_1"))
        return add_output_1


class Add17(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add17.weight_1", forge.Parameter(*(64,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add17.weight_1"))
        return add_output_1


class Add18(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add18.weight_1", forge.Parameter(*(128,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add18.weight_1"))
        return add_output_1


class Add19(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add19.weight_1", forge.Parameter(*(256,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add19.weight_1"))
        return add_output_1


class Add20(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add20.weight_1", forge.Parameter(*(512,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add20.weight_1"))
        return add_output_1


class Add21(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add21.weight_1", forge.Parameter(*(1024,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add21.weight_1"))
        return add_output_1


class Add22(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add22.weight_1", forge.Parameter(*(768,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add22.weight_1"))
        return add_output_1


class Add23(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add23.weight_1", forge.Parameter(*(3072,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add23.weight_1"))
        return add_output_1


class Add24(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add24.weight_1", forge.Parameter(*(1,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add24.weight_1"))
        return add_output_1


class Add25(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add25.weight_1", forge.Parameter(*(40,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add25.weight_1"))
        return add_output_1


class Add26(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add26.weight_1", forge.Parameter(*(24,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add26.weight_1"))
        return add_output_1


class Add27(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add27.weight_1", forge.Parameter(*(144,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add27.weight_1"))
        return add_output_1


class Add28(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add28.weight_1", forge.Parameter(*(192,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add28.weight_1"))
        return add_output_1


class Add29(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add29.weight_1", forge.Parameter(*(48,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add29.weight_1"))
        return add_output_1


class Add30(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add30.weight_1", forge.Parameter(*(288,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add30.weight_1"))
        return add_output_1


class Add31(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add31.weight_1", forge.Parameter(*(96,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add31.weight_1"))
        return add_output_1


class Add32(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add32.weight_1", forge.Parameter(*(576,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add32.weight_1"))
        return add_output_1


class Add33(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add33.weight_1", forge.Parameter(*(136,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add33.weight_1"))
        return add_output_1


class Add34(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add34.weight_1", forge.Parameter(*(816,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add34.weight_1"))
        return add_output_1


class Add35(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add35.weight_1", forge.Parameter(*(232,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add35.weight_1"))
        return add_output_1


class Add36(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add36.weight_1", forge.Parameter(*(1392,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add36.weight_1"))
        return add_output_1


class Add37(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add37.weight_1", forge.Parameter(*(384,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add37.weight_1"))
        return add_output_1


class Add38(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add38.weight_1", forge.Parameter(*(2304,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add38.weight_1"))
        return add_output_1


class Add39(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add39.weight_1", forge.Parameter(*(1536,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add39.weight_1"))
        return add_output_1


class Add40(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add40.weight_1", forge.Parameter(*(1000,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add40.weight_1"))
        return add_output_1


class Add41(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add41.weight_1", forge.Parameter(*(56,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add41.weight_1"))
        return add_output_1


class Add42(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add42.weight_1", forge.Parameter(*(336,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add42.weight_1"))
        return add_output_1


class Add43(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add43.weight_1", forge.Parameter(*(112,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add43.weight_1"))
        return add_output_1


class Add44(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add44.weight_1", forge.Parameter(*(672,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add44.weight_1"))
        return add_output_1


class Add45(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add45.weight_1", forge.Parameter(*(160,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add45.weight_1"))
        return add_output_1


class Add46(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add46.weight_1", forge.Parameter(*(960,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add46.weight_1"))
        return add_output_1


class Add47(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add47.weight_1", forge.Parameter(*(272,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add47.weight_1"))
        return add_output_1


class Add48(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add48.weight_1", forge.Parameter(*(1632,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add48.weight_1"))
        return add_output_1


class Add49(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add49.weight_1", forge.Parameter(*(448,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add49.weight_1"))
        return add_output_1


class Add50(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add50.weight_1", forge.Parameter(*(2688,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add50.weight_1"))
        return add_output_1


class Add51(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add51.weight_1", forge.Parameter(*(1792,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add51.weight_1"))
        return add_output_1


class Add52(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add52.weight_1", forge.Parameter(*(240,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add52.weight_1"))
        return add_output_1


class Add53(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add53.weight_1", forge.Parameter(*(72,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add53.weight_1"))
        return add_output_1


class Add54(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add54.weight_1", forge.Parameter(*(432,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add54.weight_1"))
        return add_output_1


class Add55(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add55.weight_1", forge.Parameter(*(864,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add55.weight_1"))
        return add_output_1


class Add56(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add56.weight_1", forge.Parameter(*(200,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add56.weight_1"))
        return add_output_1


class Add57(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add57.weight_1", forge.Parameter(*(1200,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add57.weight_1"))
        return add_output_1


class Add58(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add58.weight_1", forge.Parameter(*(344,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add58.weight_1"))
        return add_output_1


class Add59(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add59.weight_1", forge.Parameter(*(2064,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add59.weight_1"))
        return add_output_1


class Add60(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add60.weight_1", forge.Parameter(*(3456,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add60.weight_1"))
        return add_output_1


class Add61(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add61.weight_1", forge.Parameter(*(30,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add61.weight_1"))
        return add_output_1


class Add62(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add62.weight_1", forge.Parameter(*(60,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add62.weight_1"))
        return add_output_1


class Add63(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add63.weight_1", forge.Parameter(*(120,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add63.weight_1"))
        return add_output_1


class Add64(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add64.weight_1", forge.Parameter(*(2048,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add64.weight_1"))
        return add_output_1


class Add65(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add65_const_1", shape=(1, 1, 4, 4), dtype=torch.float32)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add65_const_1"))
        return add_output_1


class Add66(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add66.weight_1", forge.Parameter(*(196,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add66.weight_1"))
        return add_output_1


class Add67(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add67.weight_1", forge.Parameter(*(1001,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add67.weight_1"))
        return add_output_1


class Add68(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add68_const_0", shape=(256, 256), dtype=torch.float32)

    def forward(self, add_input_1):
        add_output_1 = forge.op.Add("", self.get_constant("add68_const_0"), add_input_1)
        return add_output_1


class Add69(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add69.weight_1", forge.Parameter(*(8192,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add69.weight_1"))
        return add_output_1


class Add70(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add70.weight_1", forge.Parameter(*(51200,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add70.weight_1"))
        return add_output_1


class Add71(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add71_const_1", shape=(1, 1, 5, 5), dtype=torch.float32)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add71_const_1"))
        return add_output_1


class Add72(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add72.weight_1", forge.Parameter(*(528,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add72.weight_1"))
        return add_output_1


class Add73(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add73.weight_1", forge.Parameter(*(1056,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add73.weight_1"))
        return add_output_1


class Add74(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add74.weight_1", forge.Parameter(*(2904,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add74.weight_1"))
        return add_output_1


class Add75(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add75.weight_1", forge.Parameter(*(7392,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add75.weight_1"))
        return add_output_1


class Add76(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add76.weight_1", forge.Parameter(*(224,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add76.weight_1"))
        return add_output_1


class Add77(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add77.weight_1", forge.Parameter(*(896,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add77.weight_1"))
        return add_output_1


class Add78(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add78.weight_1", forge.Parameter(*(2016,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add78.weight_1"))
        return add_output_1


class Add79(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add79.weight_1", forge.Parameter(*(640,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add79.weight_1"))
        return add_output_1


class Add80(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add80_const_1", shape=(64, 1, 49, 49), dtype=torch.bfloat16)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add80_const_1"))
        return add_output_1


class Add81(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add81.weight_1", forge.Parameter(*(768,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add81.weight_1"))
        return add_output_1


class Add82(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add82_const_1", shape=(16, 1, 49, 49), dtype=torch.bfloat16)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add82_const_1"))
        return add_output_1


class Add83(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add83.weight_1", forge.Parameter(*(1152,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add83.weight_1"))
        return add_output_1


class Add84(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add84_const_1", shape=(4, 1, 49, 49), dtype=torch.bfloat16)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add84_const_1"))
        return add_output_1


class Add85(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add85.weight_1", forge.Parameter(*(3072,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add85.weight_1"))
        return add_output_1


class Add86(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add86.weight_1", forge.Parameter(*(4096,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add86.weight_1"))
        return add_output_1


class Add87(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add87.weight_1",
            forge.Parameter(*(1, 197, 768), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add87.weight_1"))
        return add_output_1


class Add88(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add88_const_1", shape=(197, 197), dtype=torch.bfloat16)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add88_const_1"))
        return add_output_1


class Add89(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add89.weight_1",
            forge.Parameter(*(1, 50, 1024), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add89.weight_1"))
        return add_output_1


class Add90(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add90_const_1", shape=(50, 50), dtype=torch.bfloat16)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add90_const_1"))
        return add_output_1


class Add91(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add91_const_1", shape=(1,), dtype=torch.bfloat16)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add91_const_1"))
        return add_output_1


class Add92(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add92.weight_1", forge.Parameter(*(80,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add92.weight_1"))
        return add_output_1


class Add93(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add93.weight_1", forge.Parameter(*(728,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add93.weight_1"))
        return add_output_1


class Add94(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add94.weight_1", forge.Parameter(*(320,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add94.weight_1"))
        return add_output_1


class Add95(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add95_const_0", shape=(1, 2, 8400), dtype=torch.bfloat16)

    def forward(self, add_input_1):
        add_output_1 = forge.op.Add("", self.get_constant("add95_const_0"), add_input_1)
        return add_output_1


class Add96(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add96.weight_0", forge.Parameter(*(64,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_1):
        add_output_1 = forge.op.Add("", self.get_parameter("add96.weight_0"), add_input_1)
        return add_output_1


class Add97(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add97.weight_0", forge.Parameter(*(256,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_1):
        add_output_1 = forge.op.Add("", self.get_parameter("add97.weight_0"), add_input_1)
        return add_output_1


class Add98(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add98.weight_0", forge.Parameter(*(128,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_1):
        add_output_1 = forge.op.Add("", self.get_parameter("add98.weight_0"), add_input_1)
        return add_output_1


class Add99(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add99.weight_0", forge.Parameter(*(512,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_1):
        add_output_1 = forge.op.Add("", self.get_parameter("add99.weight_0"), add_input_1)
        return add_output_1


class Add100(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add100.weight_0", forge.Parameter(*(320,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_1):
        add_output_1 = forge.op.Add("", self.get_parameter("add100.weight_0"), add_input_1)
        return add_output_1


class Add101(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add101.weight_0", forge.Parameter(*(1280,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_1):
        add_output_1 = forge.op.Add("", self.get_parameter("add101.weight_0"), add_input_1)
        return add_output_1


class Add102(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add102.weight_0", forge.Parameter(*(2048,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_1):
        add_output_1 = forge.op.Add("", self.get_parameter("add102.weight_0"), add_input_1)
        return add_output_1


class Add103(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add103.weight_0", forge.Parameter(*(768,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_1):
        add_output_1 = forge.op.Add("", self.get_parameter("add103.weight_0"), add_input_1)
        return add_output_1


class Add104(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add104_const_0", shape=(197, 197), dtype=torch.bfloat16)

    def forward(self, add_input_1):
        add_output_1 = forge.op.Add("", self.get_constant("add104_const_0"), add_input_1)
        return add_output_1


class Add105(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add105.weight_1", forge.Parameter(*(352,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add105.weight_1"))
        return add_output_1


class Add106(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add106.weight_1", forge.Parameter(*(416,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add106.weight_1"))
        return add_output_1


class Add107(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add107.weight_1", forge.Parameter(*(480,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add107.weight_1"))
        return add_output_1


class Add108(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add108.weight_1", forge.Parameter(*(544,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add108.weight_1"))
        return add_output_1


class Add109(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add109.weight_1", forge.Parameter(*(608,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add109.weight_1"))
        return add_output_1


class Add110(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add110.weight_1", forge.Parameter(*(704,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add110.weight_1"))
        return add_output_1


class Add111(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add111.weight_1", forge.Parameter(*(736,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add111.weight_1"))
        return add_output_1


class Add112(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add112.weight_1", forge.Parameter(*(800,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add112.weight_1"))
        return add_output_1


class Add113(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add113.weight_1", forge.Parameter(*(832,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add113.weight_1"))
        return add_output_1


class Add114(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add114.weight_1", forge.Parameter(*(928,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add114.weight_1"))
        return add_output_1


class Add115(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add115.weight_1", forge.Parameter(*(992,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add115.weight_1"))
        return add_output_1


class Add116(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add116.weight_1", forge.Parameter(*(18,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add116.weight_1"))
        return add_output_1


class Add117(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add117.weight_1", forge.Parameter(*(1344,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add117.weight_1"))
        return add_output_1


class Add118(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add118.weight_1", forge.Parameter(*(3840,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add118.weight_1"))
        return add_output_1


class Add119(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add119.weight_1", forge.Parameter(*(2560,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add119.weight_1"))
        return add_output_1


class Add120(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add120.weight_1", forge.Parameter(*(8,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add120.weight_1"))
        return add_output_1


class Add121(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add121.weight_1", forge.Parameter(*(12,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add121.weight_1"))
        return add_output_1


class Add122(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add122.weight_1", forge.Parameter(*(36,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add122.weight_1"))
        return add_output_1


class Add123(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add123.weight_1", forge.Parameter(*(20,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add123.weight_1"))
        return add_output_1


class Add124(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add124.weight_1", forge.Parameter(*(100,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add124.weight_1"))
        return add_output_1


class Add125(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add125.weight_1", forge.Parameter(*(92,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add125.weight_1"))
        return add_output_1


class Add126(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add126.weight_1", forge.Parameter(*(49,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add126.weight_1"))
        return add_output_1


class Add127(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add127.weight_1", forge.Parameter(*(720,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add127.weight_1"))
        return add_output_1


class Add128(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add128.weight_1", forge.Parameter(*(1280,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add128.weight_1"))
        return add_output_1


class Add129(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add129.weight_1", forge.Parameter(*(184,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add129.weight_1"))
        return add_output_1


class Add130(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add130.weight_1", forge.Parameter(*(2560,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add130.weight_1"))
        return add_output_1


class Add131(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add131_const_1", shape=(1, 1, 11, 11), dtype=torch.float32)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add131_const_1"))
        return add_output_1


class Add132(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add132.weight_1", forge.Parameter(*(10240,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add132.weight_1"))
        return add_output_1


class Add133(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add133_const_0", shape=(6, 6), dtype=torch.float32)

    def forward(self, add_input_1):
        add_output_1 = forge.op.Add("", self.get_constant("add133_const_0"), add_input_1)
        return add_output_1


class Add134(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add134.weight_1", forge.Parameter(*(1536,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add134.weight_1"))
        return add_output_1


class Add135(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add135_const_0", shape=(35, 35), dtype=torch.float32)

    def forward(self, add_input_1):
        add_output_1 = forge.op.Add("", self.get_constant("add135_const_0"), add_input_1)
        return add_output_1


class Add136(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add136_const_0", shape=(29, 29), dtype=torch.float32)

    def forward(self, add_input_1):
        add_output_1 = forge.op.Add("", self.get_constant("add136_const_0"), add_input_1)
        return add_output_1


class Add137(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add137.weight_1", forge.Parameter(*(1232,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add137.weight_1"))
        return add_output_1


class Add138(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add138.weight_1", forge.Parameter(*(3024,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add138.weight_1"))
        return add_output_1


class Add139(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add139.weight_1", forge.Parameter(*(696,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add139.weight_1"))
        return add_output_1


class Add140(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add140.weight_1", forge.Parameter(*(3712,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add140.weight_1"))
        return add_output_1


class Add141(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add141.weight_1", forge.Parameter(*(2520,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add141.weight_1"))
        return add_output_1


class Add142(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add142.weight_1", forge.Parameter(*(1008,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add142.weight_1"))
        return add_output_1


class Add143(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add143.weight_1", forge.Parameter(*(288,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add143.weight_1"))
        return add_output_1


class Add144(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add144.weight_1", forge.Parameter(*(96,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add144.weight_1"))
        return add_output_1


class Add145(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add145.weight_1", forge.Parameter(*(384,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add145.weight_1"))
        return add_output_1


class Add146(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add146_const_1", shape=(64, 1, 64, 64), dtype=torch.float32)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add146_const_1"))
        return add_output_1


class Add147(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add147.weight_1", forge.Parameter(*(576,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add147.weight_1"))
        return add_output_1


class Add148(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add148.weight_1", forge.Parameter(*(192,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add148.weight_1"))
        return add_output_1


class Add149(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add149_const_1", shape=(16, 1, 64, 64), dtype=torch.float32)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add149_const_1"))
        return add_output_1


class Add150(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add150.weight_1", forge.Parameter(*(1152,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add150.weight_1"))
        return add_output_1


class Add151(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add151_const_1", shape=(4, 1, 64, 64), dtype=torch.float32)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add151_const_1"))
        return add_output_1


class Add152(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add152.weight_1", forge.Parameter(*(2304,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add152.weight_1"))
        return add_output_1


class Add153(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add153_const_1", shape=(1, 1, 513, 513), dtype=torch.float32)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add153_const_1"))
        return add_output_1


class Add154(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add154_const_1", shape=(1, 1, 1, 61), dtype=torch.float32)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add154_const_1"))
        return add_output_1


class Add155(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add155_const_1", shape=(1, 12, 513, 61), dtype=torch.float32)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add155_const_1"))
        return add_output_1


class Add156(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add156.weight_1",
            forge.Parameter(*(1, 197, 192), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add156.weight_1"))
        return add_output_1


class Add157(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add157.weight_1", forge.Parameter(*(176,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add157.weight_1"))
        return add_output_1


class Add158(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add158.weight_1", forge.Parameter(*(304,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add158.weight_1"))
        return add_output_1


class Add159(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add159.weight_1", forge.Parameter(*(1824,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add159.weight_1"))
        return add_output_1


class Add160(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add160.weight_1", forge.Parameter(*(9,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add160.weight_1"))
        return add_output_1


class Add161(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add161.weight_1", forge.Parameter(*(88,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add161.weight_1"))
        return add_output_1


class Add162(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add162.weight_1", forge.Parameter(*(896,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add162.weight_1"))
        return add_output_1


class Add163(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add163_const_1", shape=(1, 1, 1, 128), dtype=torch.float32)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add163_const_1"))
        return add_output_1


class Add164(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add164.weight_1", forge.Parameter(*(3,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add164.weight_1"))
        return add_output_1


class Add165(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add165.weight_1",
            forge.Parameter(*(1, 197, 1024), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add165.weight_1"))
        return add_output_1


class Add166(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add166_const_0", shape=(5880, 2), dtype=torch.bfloat16)

    def forward(self, add_input_1):
        add_output_1 = forge.op.Add("", self.get_constant("add166_const_0"), add_input_1)
        return add_output_1


class Add167(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add167.weight_1", forge.Parameter(*(28996,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add167.weight_1"))
        return add_output_1


class Add168(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add168.weight_1", forge.Parameter(*(30522,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add168.weight_1"))
        return add_output_1


class Add169(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add169.weight_1",
            forge.Parameter(*(1, 197, 192), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add169.weight_1"))
        return add_output_1


class Add170(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add170.weight_0", forge.Parameter(*(192,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_1):
        add_output_1 = forge.op.Add("", self.get_parameter("add170.weight_0"), add_input_1)
        return add_output_1


class Add171(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add171_const_0", shape=(1, 2, 8400), dtype=torch.float32)

    def forward(self, add_input_1):
        add_output_1 = forge.op.Add("", self.get_constant("add171_const_0"), add_input_1)
        return add_output_1


class Add172(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add172.weight_1", forge.Parameter(*(8,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add172.weight_1"))
        return add_output_1


class Add173(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add173.weight_1", forge.Parameter(*(40,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add173.weight_1"))
        return add_output_1


class Add174(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add174.weight_1", forge.Parameter(*(16,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add174.weight_1"))
        return add_output_1


class Add175(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add175.weight_1", forge.Parameter(*(48,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add175.weight_1"))
        return add_output_1


class Add176(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add176.weight_1", forge.Parameter(*(24,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add176.weight_1"))
        return add_output_1


class Add177(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add177.weight_1", forge.Parameter(*(120,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add177.weight_1"))
        return add_output_1


class Add178(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add178.weight_1", forge.Parameter(*(72,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add178.weight_1"))
        return add_output_1


class Add179(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add179.weight_1", forge.Parameter(*(144,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add179.weight_1"))
        return add_output_1


class Add180(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add180.weight_1", forge.Parameter(*(6625,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add180.weight_1"))
        return add_output_1


class Add181(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add181.weight_1", forge.Parameter(*(12,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add181.weight_1"))
        return add_output_1


class Add182(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add182.weight_1", forge.Parameter(*(784,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add182.weight_1"))
        return add_output_1


class Add183(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add183_const_1", shape=(1, 1, 128, 128), dtype=torch.float32)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add183_const_1"))
        return add_output_1


class Add184(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add184.weight_1", forge.Parameter(*(9,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add184.weight_1"))
        return add_output_1


class Add185(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add185.weight_1", forge.Parameter(*(1088,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add185.weight_1"))
        return add_output_1


class Add186(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add186.weight_1", forge.Parameter(*(1120,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add186.weight_1"))
        return add_output_1


class Add187(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add187.weight_1", forge.Parameter(*(1184,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add187.weight_1"))
        return add_output_1


class Add188(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add188.weight_1", forge.Parameter(*(1216,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add188.weight_1"))
        return add_output_1


class Add189(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add189.weight_1", forge.Parameter(*(1248,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add189.weight_1"))
        return add_output_1


class Add190(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add190.weight_1", forge.Parameter(*(1312,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add190.weight_1"))
        return add_output_1


class Add191(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add191.weight_1", forge.Parameter(*(1376,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add191.weight_1"))
        return add_output_1


class Add192(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add192.weight_1", forge.Parameter(*(1408,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add192.weight_1"))
        return add_output_1


class Add193(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add193.weight_1", forge.Parameter(*(1440,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add193.weight_1"))
        return add_output_1


class Add194(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add194.weight_1", forge.Parameter(*(1472,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add194.weight_1"))
        return add_output_1


class Add195(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add195.weight_1", forge.Parameter(*(1504,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add195.weight_1"))
        return add_output_1


class Add196(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add196.weight_1", forge.Parameter(*(1568,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add196.weight_1"))
        return add_output_1


class Add197(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add197.weight_1", forge.Parameter(*(1600,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add197.weight_1"))
        return add_output_1


class Add198(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add198.weight_1", forge.Parameter(*(1664,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add198.weight_1"))
        return add_output_1


class Add199(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add199.weight_1", forge.Parameter(*(1696,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add199.weight_1"))
        return add_output_1


class Add200(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add200.weight_1", forge.Parameter(*(1728,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add200.weight_1"))
        return add_output_1


class Add201(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add201.weight_1", forge.Parameter(*(1760,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add201.weight_1"))
        return add_output_1


class Add202(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add202.weight_1", forge.Parameter(*(1856,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add202.weight_1"))
        return add_output_1


class Add203(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add203.weight_1", forge.Parameter(*(1888,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add203.weight_1"))
        return add_output_1


class Add204(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add204.weight_1", forge.Parameter(*(1920,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add204.weight_1"))
        return add_output_1


class Add205(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add205.weight_1", forge.Parameter(*(208,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add205.weight_1"))
        return add_output_1


class Add206(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add206.weight_1", forge.Parameter(*(2112,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add206.weight_1"))
        return add_output_1


class Add207(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add207_const_1", shape=(1, 1, 256, 256), dtype=torch.float32)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add207_const_1"))
        return add_output_1


class Add208(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add208_const_1", shape=(1,), dtype=torch.int64)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add208_const_1"))
        return add_output_1


class Add209(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add209_const_1", shape=(1, 1, 32, 32), dtype=torch.float32)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add209_const_1"))
        return add_output_1


class Add210(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add210_const_1", shape=(1, 1, 12, 12), dtype=torch.float32)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add210_const_1"))
        return add_output_1


class Add211(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add211_const_1", shape=(1, 8, 513, 61), dtype=torch.float32)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add211_const_1"))
        return add_output_1


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (
        Add0,
        [((1, 11, 128), torch.float32), ((1, 11, 128), torch.float32)],
        {"model_names": ["pd_albert_chinese_tiny_mlm_padlenlp"], "pcc": 0.99},
    ),
    (Add1, [((1, 11, 1), torch.float32)], {"model_names": ["pd_albert_chinese_tiny_mlm_padlenlp"], "pcc": 0.99}),
    (Add2, [((1, 11, 128), torch.float32)], {"model_names": ["pd_albert_chinese_tiny_mlm_padlenlp"], "pcc": 0.99}),
    (Add3, [((1, 11, 312), torch.float32)], {"model_names": ["pd_albert_chinese_tiny_mlm_padlenlp"], "pcc": 0.99}),
    (Add1, [((1, 1, 1, 11), torch.float32)], {"model_names": ["pd_albert_chinese_tiny_mlm_padlenlp"], "pcc": 0.99}),
    (
        Add0,
        [((1, 12, 11, 11), torch.float32), ((1, 1, 1, 11), torch.float32)],
        {"model_names": ["pd_albert_chinese_tiny_mlm_padlenlp"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 11, 312), torch.float32), ((1, 11, 312), torch.float32)],
        {"model_names": ["pd_albert_chinese_tiny_mlm_padlenlp"], "pcc": 0.99},
    ),
    (Add4, [((1, 11, 1248), torch.float32)], {"model_names": ["pd_albert_chinese_tiny_mlm_padlenlp"], "pcc": 0.99}),
    (Add5, [((1, 11, 21128), torch.float32)], {"model_names": ["pd_albert_chinese_tiny_mlm_padlenlp"], "pcc": 0.99}),
    (
        Add0,
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
        Add6,
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
        Add0,
        [((1, 64, 112, 112), torch.float32), ((64, 1, 1), torch.float32)],
        {
            "model_names": [
                "pd_resnet_101_img_cls_paddlemodels",
                "pd_resnet_152_img_cls_paddlemodels",
                "onnx_vovnet_vovnet27s_obj_det_osmr",
                "onnx_vovnet_vovnet_v1_57_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 64, 56, 56), torch.float32), ((64, 1, 1), torch.float32)],
        {
            "model_names": [
                "pd_resnet_101_img_cls_paddlemodels",
                "pd_resnet_152_img_cls_paddlemodels",
                "onnx_vovnet_vovnet27s_obj_det_osmr",
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
                "tf_resnet_resnet50_img_cls_keras",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add7,
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
        Add0,
        [((1, 256, 56, 56), torch.float32), ((256, 1, 1), torch.float32)],
        {
            "model_names": [
                "pd_resnet_101_img_cls_paddlemodels",
                "pd_resnet_152_img_cls_paddlemodels",
                "onnx_vovnet_vovnet_v1_57_obj_det_torchhub",
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
                "tf_resnet_resnet50_img_cls_keras",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add2,
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
        Add0,
        [((1, 128, 56, 56), torch.float32), ((128, 1, 1), torch.float32)],
        {
            "model_names": [
                "pd_resnet_101_img_cls_paddlemodels",
                "pd_resnet_152_img_cls_paddlemodels",
                "onnx_vovnet_vovnet27s_obj_det_osmr",
                "onnx_vovnet_vovnet_v1_57_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 128, 28, 28), torch.float32), ((128, 1, 1), torch.float32)],
        {"model_names": ["pd_resnet_101_img_cls_paddlemodels", "pd_resnet_152_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
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
        Add8,
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
        Add0,
        [((1, 512, 28, 28), torch.float32), ((512, 1, 1), torch.float32)],
        {
            "model_names": [
                "pd_resnet_101_img_cls_paddlemodels",
                "pd_resnet_152_img_cls_paddlemodels",
                "onnx_vovnet_vovnet_v1_57_obj_det_torchhub",
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
        [((1, 256, 28, 28), torch.float32), ((256, 1, 1), torch.float32)],
        {
            "model_names": [
                "pd_resnet_101_img_cls_paddlemodels",
                "pd_resnet_152_img_cls_paddlemodels",
                "onnx_vovnet_vovnet27s_obj_det_osmr",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 256, 14, 14), torch.float32), ((256, 1, 1), torch.float32)],
        {"model_names": ["pd_resnet_101_img_cls_paddlemodels", "pd_resnet_152_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
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
        Add9,
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
        Add0,
        [((1, 1024, 14, 14), torch.float32), ((1024, 1, 1), torch.float32)],
        {"model_names": ["pd_resnet_101_img_cls_paddlemodels", "pd_resnet_152_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1024, 14, 14), torch.float32), ((1, 1024, 14, 14), torch.float32)],
        {"model_names": ["pd_resnet_101_img_cls_paddlemodels", "pd_resnet_152_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 512, 14, 14), torch.float32), ((512, 1, 1), torch.float32)],
        {"model_names": ["pd_resnet_101_img_cls_paddlemodels", "pd_resnet_152_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 512, 7, 7), torch.float32), ((512, 1, 1), torch.float32)],
        {
            "model_names": [
                "pd_resnet_101_img_cls_paddlemodels",
                "pd_resnet_152_img_cls_paddlemodels",
                "onnx_vovnet_vovnet27s_obj_det_osmr",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
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
        Add10,
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
        Add0,
        [((1, 2048, 7, 7), torch.float32), ((2048, 1, 1), torch.float32)],
        {"model_names": ["pd_resnet_101_img_cls_paddlemodels", "pd_resnet_152_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 2048, 7, 7), torch.float32), ((1, 2048, 7, 7), torch.float32)],
        {"model_names": ["pd_resnet_101_img_cls_paddlemodels", "pd_resnet_152_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add11,
        [((1, 1000), torch.float32)],
        {
            "model_names": [
                "pd_resnet_101_img_cls_paddlemodels",
                "pd_resnet_152_img_cls_paddlemodels",
                "pt_swin_swin_v2_t_img_cls_torchvision",
                "onnx_xception_xception71_tf_in1k_img_cls_timm",
                "tf_resnet_resnet50_img_cls_keras",
                "onnx_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "onnx_vovnet_vovnet27s_obj_det_osmr",
                "onnx_vovnet_vovnet_v1_57_obj_det_torchhub",
                "onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm",
                "onnx_deit_facebook_deit_tiny_patch16_224_img_cls_hf",
                "pd_alexnet_base_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 128, 128), torch.float32), ((1, 128, 128), torch.float32)],
        {"model_names": ["pt_albert_large_v2_token_cls_hf", "pt_albert_xlarge_v2_token_cls_hf"], "pcc": 0.99},
    ),
    (
        Add9,
        [((1, 128, 1024), torch.float32)],
        {
            "model_names": [
                "pt_albert_large_v2_token_cls_hf",
                "pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
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
        Add12,
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
        Add0,
        [((1, 16, 128, 128), torch.float32), ((1, 1, 128, 128), torch.float32)],
        {"model_names": ["pt_albert_large_v2_token_cls_hf", "pt_albert_xlarge_v2_token_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 128, 1024), torch.float32), ((1, 128, 1024), torch.float32)],
        {
            "model_names": [
                "pt_albert_large_v2_token_cls_hf",
                "pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add13,
        [((1, 128, 4096), torch.float32)],
        {
            "model_names": [
                "pt_albert_large_v2_token_cls_hf",
                "pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add14,
        [((1, 128, 2), torch.float32)],
        {"model_names": ["pt_albert_large_v2_token_cls_hf", "pt_albert_xlarge_v2_token_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
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
        Add15,
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
        Add0,
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
        Add0,
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
        Add16,
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
        Add0,
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
        Add0,
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
        Add17,
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
        Add0,
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
        Add0,
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
        Add0,
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
        Add18,
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
        Add0,
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
        Add0,
        [((1, 128, 56, 56), torch.bfloat16), ((1, 128, 56, 56), torch.bfloat16)],
        {
            "model_names": [
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
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
        Add0,
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
        Add19,
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
        Add0,
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
        Add0,
        [((1, 256, 28, 28), torch.bfloat16), ((1, 256, 28, 28), torch.bfloat16)],
        {
            "model_names": [
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
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
        Add0,
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
        Add20,
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
        Add0,
        [((1, 512, 14, 14), torch.bfloat16), ((512, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_mlp_mixer_mixer_s16_224_img_cls_timm",
                "pt_mobilenetv1_google_mobilenet_v1_1_0_224_img_cls_hf",
                "pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
                "pt_resnet_resnet152_img_cls_torchvision",
                "pt_vgg_19_obj_det_hf",
                "pt_vgg_vgg16_img_cls_torchvision",
                "pt_vgg_vgg16_obj_det_osmr",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_resnext_resnext50_32x4d_img_cls_osmr",
                "pt_vgg_bn_vgg19b_obj_det_osmr",
                "pt_vgg_vgg11_img_cls_torchvision",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_mobilenetv1_basic_img_cls_torchvision",
                "pt_resnet_50_img_cls_timm",
                "pt_vgg_vgg13_img_cls_torchvision",
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
        Add0,
        [((1, 512, 14, 14), torch.bfloat16), ((1, 512, 14, 14), torch.bfloat16)],
        {
            "model_names": [
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
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
        Add0,
        [((1, 512, 7, 7), torch.bfloat16), ((512, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_mobilenetv1_google_mobilenet_v1_1_0_224_img_cls_hf",
                "pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
                "pt_resnet_resnet152_img_cls_torchvision",
                "pt_vovnet_vovnet27s_img_cls_osmr",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_mlp_mixer_mixer_s32_224_img_cls_timm",
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
        Add0,
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
        Add21,
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
        Add0,
        [((1, 1024, 7, 7), torch.bfloat16), ((1024, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_mobilenetv1_google_mobilenet_v1_1_0_224_img_cls_hf",
                "pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
                "pt_vit_vit_l_32_img_cls_torchvision",
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
        Add0,
        [((1, 1024, 7, 7), torch.bfloat16), ((1, 1024, 7, 7), torch.bfloat16)],
        {
            "model_names": [
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_vovnet_vovnet39_img_cls_osmr",
                "pt_vovnet_v1_vovnet39_obj_det_torchhub",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 1000, 1, 1), torch.bfloat16), ((1000, 1, 1), torch.bfloat16)],
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
        Add0,
        [((1, 128, 768), torch.float32), ((1, 128, 768), torch.float32)],
        {
            "model_names": [
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add22,
        [((1, 128, 768), torch.float32)],
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
        Add0,
        [((1, 12, 128, 128), torch.float32), ((1, 1, 128, 128), torch.float32)],
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
        Add23,
        [((1, 128, 3072), torch.float32)],
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
        Add0,
        [((128, 1), torch.float32), ((1,), torch.float32)],
        {"model_names": ["pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader"], "pcc": 0.99},
    ),
    (
        Add24,
        [((1, 1), torch.float32)],
        {"model_names": ["pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader"], "pcc": 0.99},
    ),
    (
        Add0,
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
        Add25,
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
        Add0,
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
        Add0,
        [((1, 10, 1, 1), torch.bfloat16), ((10, 1, 1), torch.bfloat16)],
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
        Add0,
        [((1, 40, 1, 1), torch.bfloat16), ((40, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
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
        Add26,
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
        Add0,
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
        Add0,
        [((1, 6, 1, 1), torch.bfloat16), ((6, 1, 1), torch.bfloat16)],
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
        Add0,
        [((1, 24, 1, 1), torch.bfloat16), ((24, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
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
        Add0,
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
        Add0,
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
        Add27,
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
        Add0,
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
        Add0,
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
        Add0,
        [((1, 144, 1, 1), torch.bfloat16), ((144, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
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
        Add0,
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
        Add28,
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
        Add0,
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
        Add0,
        [((1, 8, 1, 1), torch.bfloat16), ((8, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                "pt_regnet_regnet_y_128gf_img_cls_torchvision",
                "pt_regnet_regnet_y_8gf_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
                "pt_regnet_facebook_regnet_y_160_img_cls_hf",
                "pt_regnet_facebook_regnet_y_320_img_cls_hf",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 192, 1, 1), torch.bfloat16), ((192, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 32, 56, 56), torch.bfloat16), ((1, 32, 56, 56), torch.bfloat16)],
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
        Add0,
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
        Add0,
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
        Add29,
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
        Add0,
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
        Add0,
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
        Add30,
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
        Add0,
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
        Add0,
        [((1, 12, 1, 1), torch.bfloat16), ((12, 1, 1), torch.bfloat16)],
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
        Add0,
        [((1, 288, 1, 1), torch.bfloat16), ((288, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 48, 28, 28), torch.bfloat16), ((1, 48, 28, 28), torch.bfloat16)],
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
        Add0,
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
        Add0,
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
        Add31,
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
        Add0,
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
        Add0,
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
        Add32,
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
        Add0,
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
        Add0,
        [((1, 576, 1, 1), torch.bfloat16), ((576, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 96, 14, 14), torch.bfloat16), ((1, 96, 14, 14), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_mobilenetv2_basic_img_cls_torchhub",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
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
        Add33,
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
        Add0,
        [((1, 136, 14, 14), torch.bfloat16), ((136, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b3_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
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
        Add34,
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
        Add0,
        [((1, 816, 14, 14), torch.bfloat16), ((816, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b3_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 34, 1, 1), torch.bfloat16), ((34, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b3_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 816, 1, 1), torch.bfloat16), ((816, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b3_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 136, 14, 14), torch.bfloat16), ((1, 136, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b3_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 816, 7, 7), torch.bfloat16), ((816, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b3_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
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
        Add35,
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
        Add0,
        [((1, 232, 7, 7), torch.bfloat16), ((232, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b3_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
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
        Add36,
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
        Add0,
        [((1, 1392, 7, 7), torch.bfloat16), ((1392, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b3_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 58, 1, 1), torch.bfloat16), ((58, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_regnet_facebook_regnet_y_320_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 1392, 1, 1), torch.bfloat16), ((1392, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_regnet_facebook_regnet_y_320_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 232, 7, 7), torch.bfloat16), ((1, 232, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b3_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
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
        Add37,
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
        Add0,
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
        Add0,
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
        Add38,
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
        Add0,
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
        Add0,
        [((1, 96, 1, 1), torch.bfloat16), ((96, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 2304, 1, 1), torch.bfloat16), ((2304, 1, 1), torch.bfloat16)],
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
        Add0,
        [((1, 384, 7, 7), torch.bfloat16), ((1, 384, 7, 7), torch.bfloat16)],
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
        Add0,
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
        Add39,
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
        Add0,
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
        Add40,
        [((1, 1000), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_mlp_mixer_base_img_cls_github",
                "pt_mlp_mixer_mixer_s16_224_img_cls_timm",
                "pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
                "pt_mobilenetv3_ssd_resnet34_img_cls_torchvision",
                "pt_regnet_regnet_y_128gf_img_cls_torchvision",
                "pt_regnet_regnet_y_8gf_img_cls_torchvision",
                "pt_resnet_resnet152_img_cls_torchvision",
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_vgg_19_obj_det_hf",
                "pt_vgg_vgg16_img_cls_torchvision",
                "pt_vgg_vgg16_obj_det_osmr",
                "pt_vit_google_vit_base_patch16_224_img_cls_hf",
                "pt_vit_vit_l_32_img_cls_torchvision",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_vovnet_vovnet27s_img_cls_osmr",
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
                "pt_beit_microsoft_beit_large_patch16_224_img_cls_hf",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_mlp_mixer_mixer_b16_224_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_s32_224_img_cls_timm",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_ssd_resnet101_img_cls_torchvision",
                "pt_regnet_facebook_regnet_y_160_img_cls_hf",
                "pt_regnet_facebook_regnet_y_320_img_cls_hf",
                "pt_regnet_regnet_x_32gf_img_cls_torchvision",
                "pt_regnet_regnet_x_3_2gf_img_cls_torchvision",
                "pt_resnext_resnext101_32x8d_img_cls_torchhub",
                "pt_resnext_resnext101_64x4d_img_cls_osmr",
                "pt_resnext_resnext50_32x4d_img_cls_osmr",
                "pt_vgg_bn_vgg19b_obj_det_osmr",
                "pt_vgg_vgg11_img_cls_torchvision",
                "pt_vovnet_vovnet39_img_cls_osmr",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
                "pt_beit_microsoft_beit_base_patch16_224_img_cls_hf",
                "pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_mlp_mixer_mixer_l16_224_in21k_img_cls_timm",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_mobilenetv3_ssd_resnet152_img_cls_torchvision",
                "pt_resnet_50_img_cls_timm",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_vgg_vgg13_img_cls_torchvision",
                "pt_vit_vit_b_16_img_cls_torchvision",
                "pt_vit_vit_l_16_img_cls_torchvision",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_vovnet_v1_vovnet39_obj_det_torchhub",
                "pt_xception_xception_img_cls_timm",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
                "pt_inception_inception_v4_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_miil_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
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
        Add0,
        [((1, 48, 1, 1), torch.bfloat16), ((48, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
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
        Add41,
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
        Add0,
        [((1, 56, 28, 28), torch.bfloat16), ((56, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b4_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
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
        Add42,
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
        Add0,
        [((1, 336, 28, 28), torch.bfloat16), ((336, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b4_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 14, 1, 1), torch.bfloat16), ((14, 1, 1), torch.bfloat16)],
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
        Add0,
        [((1, 336, 1, 1), torch.bfloat16), ((336, 1, 1), torch.bfloat16)],
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
        Add0,
        [((1, 56, 28, 28), torch.bfloat16), ((1, 56, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b4_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
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
        Add0,
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
        Add43,
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
        Add0,
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
        Add0,
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
        Add44,
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
        Add0,
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
        Add0,
        [((1, 28, 1, 1), torch.bfloat16), ((28, 1, 1), torch.bfloat16)],
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
        Add0,
        [((1, 672, 1, 1), torch.bfloat16), ((672, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 112, 14, 14), torch.bfloat16), ((1, 112, 14, 14), torch.bfloat16)],
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
        Add0,
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
        Add45,
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
        Add0,
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
        Add0,
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
        Add46,
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
        Add0,
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
        Add0,
        [((1, 960, 1, 1), torch.bfloat16), ((960, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 160, 14, 14), torch.bfloat16), ((1, 160, 14, 14), torch.bfloat16)],
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
        Add0,
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
        Add0,
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
        Add47,
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
        Add0,
        [((1, 272, 7, 7), torch.bfloat16), ((272, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b4_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
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
        Add48,
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
        Add0,
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
        Add0,
        [((1, 68, 1, 1), torch.bfloat16), ((68, 1, 1), torch.bfloat16)],
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
        Add0,
        [((1, 1632, 1, 1), torch.bfloat16), ((1632, 1, 1), torch.bfloat16)],
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
        Add0,
        [((1, 272, 7, 7), torch.bfloat16), ((1, 272, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b4_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
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
        Add49,
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
        Add0,
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
        Add0,
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
        Add50,
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
        Add0,
        [((1, 2688, 7, 7), torch.bfloat16), ((2688, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b4_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 112, 1, 1), torch.bfloat16), ((112, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_regnet_regnet_y_8gf_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_regnet_facebook_regnet_y_160_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 2688, 1, 1), torch.bfloat16), ((2688, 1, 1), torch.bfloat16)],
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
        Add0,
        [((1, 448, 7, 7), torch.bfloat16), ((1, 448, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b4_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
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
        Add51,
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
        Add0,
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
        Add0,
        [((1, 56, 112, 112), torch.bfloat16), ((56, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b6_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 56, 1, 1), torch.bfloat16), ((56, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                "pt_regnet_regnet_y_8gf_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
                "pt_regnet_facebook_regnet_y_160_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 32, 1, 1), torch.bfloat16), ((32, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 32, 112, 112), torch.bfloat16), ((1, 32, 112, 112), torch.bfloat16)],
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
        Add0,
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
        Add0,
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
        Add0,
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
        Add52,
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
        Add0,
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
        Add0,
        [((1, 240, 1, 1), torch.bfloat16), ((240, 1, 1), torch.bfloat16)],
        {
            "model_names": [
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
        Add0,
        [((1, 40, 56, 56), torch.bfloat16), ((1, 40, 56, 56), torch.bfloat16)],
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
        Add0,
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
        Add0,
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
        Add53,
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
        Add0,
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
        Add0,
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
        Add54,
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
        Add0,
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
        Add0,
        [((1, 18, 1, 1), torch.bfloat16), ((18, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b6_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 432, 1, 1), torch.bfloat16), ((432, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b6_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 72, 28, 28), torch.bfloat16), ((1, 72, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b6_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
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
        Add0,
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
        Add0,
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
        Add55,
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
        Add0,
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
        Add0,
        [((1, 36, 1, 1), torch.bfloat16), ((36, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b6_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 864, 1, 1), torch.bfloat16), ((864, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b6_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 144, 14, 14), torch.bfloat16), ((1, 144, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b6_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
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
        Add56,
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
        Add0,
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
        Add0,
        [((1200,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b6_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add57,
        [((1200,), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b6_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 1200, 14, 14), torch.bfloat16), ((1200, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b6_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 50, 1, 1), torch.bfloat16), ((50, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b6_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 1200, 1, 1), torch.bfloat16), ((1200, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b6_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 200, 14, 14), torch.bfloat16), ((1, 200, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b6_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 1200, 7, 7), torch.bfloat16), ((1200, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b6_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((344,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b6_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add58,
        [((344,), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b6_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 344, 7, 7), torch.bfloat16), ((344, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b6_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((2064,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b6_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add59,
        [((2064,), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b6_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 2064, 7, 7), torch.bfloat16), ((2064, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b6_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 86, 1, 1), torch.bfloat16), ((86, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b6_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 2064, 1, 1), torch.bfloat16), ((2064, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b6_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 344, 7, 7), torch.bfloat16), ((1, 344, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b6_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
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
        Add0,
        [((3456,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b6_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add60,
        [((3456,), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b6_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 3456, 7, 7), torch.bfloat16), ((3456, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b6_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 3456, 1, 1), torch.bfloat16), ((3456, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b6_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 576, 7, 7), torch.bfloat16), ((1, 576, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b6_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 256, 56, 56), torch.bfloat16), ((256, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_resnet_resnet152_img_cls_torchvision",
                "pt_vgg_19_obj_det_hf",
                "pt_vgg_vgg16_img_cls_torchvision",
                "pt_vgg_vgg16_obj_det_osmr",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_resnext_resnext101_32x8d_img_cls_torchhub",
                "pt_resnext_resnext101_64x4d_img_cls_osmr",
                "pt_resnext_resnext50_32x4d_img_cls_osmr",
                "pt_vgg_bn_vgg19b_obj_det_osmr",
                "pt_vgg_vgg11_img_cls_torchvision",
                "pt_vovnet_vovnet39_img_cls_osmr",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_resnet_50_img_cls_timm",
                "pt_vgg_vgg13_img_cls_torchvision",
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
        Add0,
        [((1, 256, 56, 56), torch.bfloat16), ((1, 256, 56, 56), torch.bfloat16)],
        {
            "model_names": [
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_resnet_resnet152_img_cls_torchvision",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_resnext_resnext101_32x8d_img_cls_torchhub",
                "pt_resnext_resnext101_64x4d_img_cls_osmr",
                "pt_resnext_resnext50_32x4d_img_cls_osmr",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
                "pt_resnet_50_img_cls_timm",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((30,), torch.bfloat16), ((1,), torch.bfloat16)],
        {"model_names": ["pt_hrnet_hrnet_w30_pose_estimation_timm"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add61,
        [((30,), torch.bfloat16)],
        {"model_names": ["pt_hrnet_hrnet_w30_pose_estimation_timm"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 30, 56, 56), torch.bfloat16), ((30, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_hrnet_hrnet_w30_pose_estimation_timm"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 30, 56, 56), torch.bfloat16), ((1, 30, 56, 56), torch.bfloat16)],
        {"model_names": ["pt_hrnet_hrnet_w30_pose_estimation_timm"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((60,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w30_pose_estimation_timm", "pt_ghostnet_ghostnet_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add62,
        [((60,), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w30_pose_estimation_timm", "pt_ghostnet_ghostnet_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 60, 28, 28), torch.bfloat16), ((60, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w30_pose_estimation_timm", "pt_ghostnet_ghostnet_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 60, 28, 28), torch.bfloat16), ((1, 60, 28, 28), torch.bfloat16)],
        {"model_names": ["pt_hrnet_hrnet_w30_pose_estimation_timm"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 30, 28, 28), torch.bfloat16), ((30, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_hrnet_hrnet_w30_pose_estimation_timm"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
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
        Add63,
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
        Add0,
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
        Add0,
        [((1, 120, 14, 14), torch.bfloat16), ((1, 120, 14, 14), torch.bfloat16)],
        {
            "model_names": [
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 30, 14, 14), torch.bfloat16), ((30, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_hrnet_hrnet_w30_pose_estimation_timm"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 60, 14, 14), torch.bfloat16), ((60, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_hrnet_hrnet_w30_pose_estimation_timm"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 240, 7, 7), torch.bfloat16), ((240, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_hrnet_hrnet_w30_pose_estimation_timm"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 240, 7, 7), torch.bfloat16), ((1, 240, 7, 7), torch.bfloat16)],
        {"model_names": ["pt_hrnet_hrnet_w30_pose_estimation_timm"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 210, 7, 7), torch.bfloat16), ((210, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_hrnet_hrnet_w30_pose_estimation_timm"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
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
        Add0,
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
        Add0,
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
        Add0,
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
        Add0,
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
        Add0,
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
        Add64,
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
        Add1,
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
        Add65,
        [((1, 32, 4, 4), torch.float32)],
        {"model_names": ["pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 4, 2048), torch.float32), ((1, 4, 2048), torch.float32)],
        {"model_names": ["pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Add20,
        [((1, 256, 512), torch.bfloat16)],
        {
            "model_names": [
                "pt_mlp_mixer_base_img_cls_github",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 1024, 512), torch.bfloat16), ((1024, 1), torch.bfloat16)],
        {"model_names": ["pt_mlp_mixer_base_img_cls_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 256, 512), torch.bfloat16), ((256, 1), torch.bfloat16)],
        {"model_names": ["pt_mlp_mixer_base_img_cls_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 256, 512), torch.bfloat16), ((1, 256, 512), torch.bfloat16)],
        {
            "model_names": [
                "pt_mlp_mixer_base_img_cls_github",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add19,
        [((1, 256, 256), torch.bfloat16)],
        {
            "model_names": ["pt_mlp_mixer_base_img_cls_github", "pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add19,
        [((1, 512, 256), torch.bfloat16)],
        {
            "model_names": ["pt_mlp_mixer_mixer_s16_224_img_cls_timm", "pt_mlp_mixer_mixer_s32_224_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add66,
        [((1, 512, 196), torch.bfloat16)],
        {"model_names": ["pt_mlp_mixer_mixer_s16_224_img_cls_timm"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 196, 512), torch.bfloat16), ((1, 196, 512), torch.bfloat16)],
        {"model_names": ["pt_mlp_mixer_mixer_s16_224_img_cls_timm"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add64,
        [((1, 196, 2048), torch.bfloat16)],
        {"model_names": ["pt_mlp_mixer_mixer_s16_224_img_cls_timm"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add20,
        [((1, 196, 512), torch.bfloat16)],
        {"model_names": ["pt_mlp_mixer_mixer_s16_224_img_cls_timm"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add67,
        [((1, 1001), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv1_google_mobilenet_v1_1_0_224_img_cls_hf",
                "pt_mobilenetv1_google_mobilenet_v1_0_75_192_img_cls_hf",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
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
        Add0,
        [((1, 64, 80, 80), torch.bfloat16), ((64, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv3_ssd_resnet34_img_cls_torchvision",
                "pt_yolov8_yolov8x_obj_det_github",
                "pt_mobilenetv3_ssd_resnet101_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet152_img_cls_torchvision",
                "pt_yolov9_default_obj_det_github",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 64, 80, 80), torch.bfloat16), ((1, 64, 80, 80), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_ssd_resnet34_img_cls_torchvision", "pt_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
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
        Add0,
        [((1, 128, 40, 40), torch.bfloat16), ((1, 128, 40, 40), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_ssd_resnet34_img_cls_torchvision", "pt_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
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
        Add0,
        [((1, 256, 20, 20), torch.bfloat16), ((1, 256, 20, 20), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_ssd_resnet34_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
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
        Add0,
        [((1, 512, 10, 10), torch.bfloat16), ((1, 512, 10, 10), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_ssd_resnet34_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (Add10, [((1, 256, 2048), torch.float32)], {"model_names": ["pt_phi1_microsoft_phi_1_clm_hf"], "pcc": 0.99}),
    (
        Add0,
        [((1, 32, 256, 32), torch.float32), ((1, 32, 256, 32), torch.float32)],
        {"model_names": ["pt_phi1_microsoft_phi_1_clm_hf"], "pcc": 0.99},
    ),
    (
        Add0,
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
    (Add68, [((1, 1, 256, 256), torch.float32)], {"model_names": ["pt_phi1_microsoft_phi_1_clm_hf"], "pcc": 0.99}),
    (
        Add0,
        [((1, 32, 256, 256), torch.float32), ((1, 1, 256, 256), torch.float32)],
        {"model_names": ["pt_phi1_microsoft_phi_1_clm_hf"], "pcc": 0.99},
    ),
    (Add69, [((1, 256, 8192), torch.float32)], {"model_names": ["pt_phi1_microsoft_phi_1_clm_hf"], "pcc": 0.99}),
    (
        Add0,
        [((1, 256, 2048), torch.float32), ((1, 256, 2048), torch.float32)],
        {"model_names": ["pt_phi1_microsoft_phi_1_clm_hf"], "pcc": 0.99},
    ),
    (
        Add70,
        [((1, 256, 51200), torch.float32)],
        {
            "model_names": ["pt_phi1_microsoft_phi_1_clm_hf", "pt_codegen_salesforce_codegen_350m_nl_clm_hf"],
            "pcc": 0.99,
        },
    ),
    (Add10, [((1, 5, 2048), torch.float32)], {"model_names": ["pt_phi_1_5_microsoft_phi_1_5_seq_cls_hf"], "pcc": 0.99}),
    (
        Add0,
        [((1, 32, 5, 32), torch.float32), ((1, 32, 5, 32), torch.float32)],
        {"model_names": ["pt_phi_1_5_microsoft_phi_1_5_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Add71,
        [((1, 32, 5, 5), torch.float32)],
        {"model_names": ["pt_phi_1_5_microsoft_phi_1_5_seq_cls_hf"], "pcc": 0.99},
    ),
    (Add69, [((1, 5, 8192), torch.float32)], {"model_names": ["pt_phi_1_5_microsoft_phi_1_5_seq_cls_hf"], "pcc": 0.99}),
    (
        Add0,
        [((1, 5, 2048), torch.float32), ((1, 5, 2048), torch.float32)],
        {"model_names": ["pt_phi_1_5_microsoft_phi_1_5_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 32, 192, 192), torch.bfloat16), ((32, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_128gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
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
        Add72,
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
        Add0,
        [((1, 528, 96, 96), torch.bfloat16), ((528, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_128gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 528, 192, 192), torch.bfloat16), ((528, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_128gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 528, 1, 1), torch.bfloat16), ((528, 1, 1), torch.bfloat16)],
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
        Add0,
        [((1, 528, 96, 96), torch.bfloat16), ((1, 528, 96, 96), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_128gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 132, 1, 1), torch.bfloat16), ((132, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_128gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
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
        Add73,
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
        Add0,
        [((1, 1056, 48, 48), torch.bfloat16), ((1056, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_128gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 1056, 96, 96), torch.bfloat16), ((1056, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_128gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 1056, 1, 1), torch.bfloat16), ((1056, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_regnet_regnet_y_128gf_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 1056, 48, 48), torch.bfloat16), ((1, 1056, 48, 48), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_128gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 264, 1, 1), torch.bfloat16), ((264, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_128gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((2904,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_128gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add74,
        [((2904,), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_128gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 2904, 24, 24), torch.bfloat16), ((2904, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_128gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 2904, 48, 48), torch.bfloat16), ((2904, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_128gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 2904, 1, 1), torch.bfloat16), ((2904, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_128gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 2904, 24, 24), torch.bfloat16), ((1, 2904, 24, 24), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_128gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 726, 1, 1), torch.bfloat16), ((726, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_128gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((7392,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_128gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add75,
        [((7392,), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_128gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 7392, 12, 12), torch.bfloat16), ((7392, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_128gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 7392, 24, 24), torch.bfloat16), ((7392, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_128gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 7392, 1, 1), torch.bfloat16), ((7392, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_128gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 7392, 12, 12), torch.bfloat16), ((1, 7392, 12, 12), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_128gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
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
        Add76,
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
        Add0,
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
        Add0,
        [((1, 224, 112, 112), torch.bfloat16), ((224, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_8gf_img_cls_torchvision", "pt_regnet_facebook_regnet_y_160_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 224, 1, 1), torch.bfloat16), ((224, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_8gf_img_cls_torchvision", "pt_regnet_facebook_regnet_y_160_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 224, 56, 56), torch.bfloat16), ((1, 224, 56, 56), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_8gf_img_cls_torchvision", "pt_regnet_facebook_regnet_y_160_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
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
        Add0,
        [((1, 448, 56, 56), torch.bfloat16), ((448, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_8gf_img_cls_torchvision", "pt_regnet_facebook_regnet_y_160_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 448, 1, 1), torch.bfloat16), ((448, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_8gf_img_cls_torchvision", "pt_regnet_facebook_regnet_y_160_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 448, 28, 28), torch.bfloat16), ((1, 448, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_8gf_img_cls_torchvision", "pt_regnet_facebook_regnet_y_160_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
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
        Add77,
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
        Add0,
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
        Add0,
        [((1, 896, 28, 28), torch.bfloat16), ((896, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_8gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 896, 1, 1), torch.bfloat16), ((896, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_8gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 896, 14, 14), torch.bfloat16), ((1, 896, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_8gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((2016,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_8gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add78,
        [((2016,), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_8gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 2016, 7, 7), torch.bfloat16), ((2016, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_8gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 2016, 14, 14), torch.bfloat16), ((2016, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_8gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 2016, 1, 1), torch.bfloat16), ((2016, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_8gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 2016, 7, 7), torch.bfloat16), ((1, 2016, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_8gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 512, 28, 28), torch.bfloat16), ((512, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_resnet_resnet152_img_cls_torchvision",
                "pt_vgg_19_obj_det_hf",
                "pt_vgg_vgg16_img_cls_torchvision",
                "pt_vgg_vgg16_obj_det_osmr",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_resnext_resnext101_32x8d_img_cls_torchhub",
                "pt_resnext_resnext101_64x4d_img_cls_osmr",
                "pt_resnext_resnext50_32x4d_img_cls_osmr",
                "pt_vgg_bn_vgg19b_obj_det_osmr",
                "pt_vgg_vgg11_img_cls_torchvision",
                "pt_vovnet_vovnet39_img_cls_osmr",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_resnet_50_img_cls_timm",
                "pt_vgg_vgg13_img_cls_torchvision",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_vovnet_v1_vovnet39_obj_det_torchhub",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 512, 28, 28), torch.bfloat16), ((1, 512, 28, 28), torch.bfloat16)],
        {
            "model_names": [
                "pt_resnet_resnet152_img_cls_torchvision",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_resnext_resnext101_32x8d_img_cls_torchhub",
                "pt_resnext_resnext101_64x4d_img_cls_osmr",
                "pt_resnext_resnext50_32x4d_img_cls_osmr",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
                "pt_resnet_50_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 1024, 14, 14), torch.bfloat16), ((1024, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_resnet_resnet152_img_cls_torchvision",
                "pt_beit_microsoft_beit_large_patch16_224_img_cls_hf",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_resnext_resnext101_32x8d_img_cls_torchhub",
                "pt_resnext_resnext101_64x4d_img_cls_osmr",
                "pt_resnext_resnext50_32x4d_img_cls_osmr",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_mlp_mixer_mixer_l16_224_in21k_img_cls_timm",
                "pt_resnet_50_img_cls_timm",
                "pt_vit_vit_l_16_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 1024, 14, 14), torch.bfloat16), ((1, 1024, 14, 14), torch.bfloat16)],
        {
            "model_names": [
                "pt_resnet_resnet152_img_cls_torchvision",
                "pt_resnext_resnext101_32x8d_img_cls_torchhub",
                "pt_resnext_resnext101_64x4d_img_cls_osmr",
                "pt_resnext_resnext50_32x4d_img_cls_osmr",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
                "pt_resnet_50_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 2048, 7, 7), torch.bfloat16), ((1, 2048, 7, 7), torch.bfloat16)],
        {
            "model_names": [
                "pt_resnet_resnet152_img_cls_torchvision",
                "pt_resnext_resnext101_32x8d_img_cls_torchhub",
                "pt_resnext_resnext101_64x4d_img_cls_osmr",
                "pt_resnext_resnext50_32x4d_img_cls_osmr",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
                "pt_resnet_50_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 64, 240, 320), torch.bfloat16), ((64, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_retinanet_retinanet_rn50fpn_obj_det_hf", "pt_yolo_v4_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 64, 120, 160), torch.bfloat16), ((64, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_retinanet_retinanet_rn50fpn_obj_det_hf", "pt_yolo_v4_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 256, 120, 160), torch.bfloat16), ((256, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_retinanet_retinanet_rn50fpn_obj_det_hf"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 256, 120, 160), torch.bfloat16), ((1, 256, 120, 160), torch.bfloat16)],
        {"model_names": ["pt_retinanet_retinanet_rn50fpn_obj_det_hf"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 128, 120, 160), torch.bfloat16), ((128, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_retinanet_retinanet_rn50fpn_obj_det_hf", "pt_yolo_v4_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 128, 60, 80), torch.bfloat16), ((128, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_retinanet_retinanet_rn50fpn_obj_det_hf", "pt_yolo_v4_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 512, 60, 80), torch.bfloat16), ((512, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_retinanet_retinanet_rn50fpn_obj_det_hf"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 512, 60, 80), torch.bfloat16), ((1, 512, 60, 80), torch.bfloat16)],
        {"model_names": ["pt_retinanet_retinanet_rn50fpn_obj_det_hf"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 256, 60, 80), torch.bfloat16), ((256, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_retinanet_retinanet_rn50fpn_obj_det_hf", "pt_yolo_v4_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 256, 30, 40), torch.bfloat16), ((256, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_retinanet_retinanet_rn50fpn_obj_det_hf", "pt_yolo_v4_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 1024, 30, 40), torch.bfloat16), ((1024, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_retinanet_retinanet_rn50fpn_obj_det_hf"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 1024, 30, 40), torch.bfloat16), ((1, 1024, 30, 40), torch.bfloat16)],
        {"model_names": ["pt_retinanet_retinanet_rn50fpn_obj_det_hf"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 512, 30, 40), torch.bfloat16), ((512, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_retinanet_retinanet_rn50fpn_obj_det_hf", "pt_yolo_v4_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 512, 15, 20), torch.bfloat16), ((512, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_retinanet_retinanet_rn50fpn_obj_det_hf", "pt_yolo_v4_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 2048, 15, 20), torch.bfloat16), ((2048, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_retinanet_retinanet_rn50fpn_obj_det_hf"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 2048, 15, 20), torch.bfloat16), ((1, 2048, 15, 20), torch.bfloat16)],
        {"model_names": ["pt_retinanet_retinanet_rn50fpn_obj_det_hf"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 256, 15, 20), torch.bfloat16), ((256, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_retinanet_retinanet_rn50fpn_obj_det_hf", "pt_yolo_v4_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 256, 30, 40), torch.bfloat16), ((1, 256, 30, 40), torch.bfloat16)],
        {
            "model_names": ["pt_retinanet_retinanet_rn50fpn_obj_det_hf", "pt_yolo_v4_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 256, 60, 80), torch.bfloat16), ((1, 256, 60, 80), torch.bfloat16)],
        {"model_names": ["pt_retinanet_retinanet_rn50fpn_obj_det_hf"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 720, 60, 80), torch.bfloat16), ((720, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_retinanet_retinanet_rn50fpn_obj_det_hf"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 720, 30, 40), torch.bfloat16), ((720, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_retinanet_retinanet_rn50fpn_obj_det_hf"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 720, 15, 20), torch.bfloat16), ((720, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_retinanet_retinanet_rn50fpn_obj_det_hf"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 256, 8, 10), torch.bfloat16), ((256, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_retinanet_retinanet_rn50fpn_obj_det_hf"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 720, 8, 10), torch.bfloat16), ((720, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_retinanet_retinanet_rn50fpn_obj_det_hf"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 256, 4, 5), torch.bfloat16), ((256, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_retinanet_retinanet_rn50fpn_obj_det_hf"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 720, 4, 5), torch.bfloat16), ((720, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_retinanet_retinanet_rn50fpn_obj_det_hf"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 36, 60, 80), torch.bfloat16), ((36, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_retinanet_retinanet_rn50fpn_obj_det_hf"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 36, 30, 40), torch.bfloat16), ((36, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_retinanet_retinanet_rn50fpn_obj_det_hf"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 36, 15, 20), torch.bfloat16), ((36, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_retinanet_retinanet_rn50fpn_obj_det_hf"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 36, 8, 10), torch.bfloat16), ((36, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_retinanet_retinanet_rn50fpn_obj_det_hf"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 36, 4, 5), torch.bfloat16), ((36, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_retinanet_retinanet_rn50fpn_obj_det_hf"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 32, 128, 128), torch.bfloat16), ((32, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add16,
        [((1, 16384, 32), torch.bfloat16)],
        {"model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 32, 16, 16), torch.bfloat16), ((32, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add16,
        [((1, 256, 32), torch.bfloat16)],
        {"model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 16384, 32), torch.bfloat16), ((1, 16384, 32), torch.bfloat16)],
        {"model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add18,
        [((1, 16384, 128), torch.bfloat16)],
        {"model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 128, 128, 128), torch.bfloat16), ((128, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf", "pt_yolo_v3_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 64, 64, 64), torch.bfloat16), ((64, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add17,
        [((1, 4096, 64), torch.bfloat16)],
        {"model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 64, 16, 16), torch.bfloat16), ((64, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add17,
        [((1, 256, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 4096, 64), torch.bfloat16), ((1, 4096, 64), torch.bfloat16)],
        {"model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add19,
        [((1, 4096, 256), torch.bfloat16)],
        {"model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 256, 64, 64), torch.bfloat16), ((256, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_fpn_base_img_cls_torchvision",
                "pt_yolo_v3_default_obj_det_github",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 160, 32, 32), torch.bfloat16), ((160, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add45,
        [((1, 1024, 160), torch.bfloat16)],
        {"model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 160, 16, 16), torch.bfloat16), ((160, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add45,
        [((1, 256, 160), torch.bfloat16)],
        {"model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 1024, 160), torch.bfloat16), ((1, 1024, 160), torch.bfloat16)],
        {"model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add79,
        [((1, 1024, 640), torch.bfloat16)],
        {"model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 640, 32, 32), torch.bfloat16), ((640, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 256, 16, 16), torch.bfloat16), ((256, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_fpn_base_img_cls_torchvision",
                "pt_yolo_v3_default_obj_det_github",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 256, 256), torch.bfloat16), ((1, 256, 256), torch.bfloat16)],
        {"model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add21,
        [((1, 256, 1024), torch.bfloat16)],
        {"model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 1024, 16, 16), torch.bfloat16), ((1024, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf", "pt_yolo_v3_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 96, 56, 56), torch.bfloat16), ((96, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_s_img_cls_torchvision",
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
        Add30,
        [((64, 49, 288), torch.bfloat16)],
        {"model_names": ["pt_swin_swin_s_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((64, 3, 49, 49), torch.bfloat16), ((1, 3, 49, 49), torch.bfloat16)],
        {"model_names": ["pt_swin_swin_s_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add31,
        [((64, 49, 96), torch.bfloat16)],
        {"model_names": ["pt_swin_swin_s_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 56, 56, 96), torch.bfloat16), ((1, 56, 56, 96), torch.bfloat16)],
        {"model_names": ["pt_swin_swin_s_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add37,
        [((1, 56, 56, 384), torch.bfloat16)],
        {"model_names": ["pt_swin_swin_s_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add31,
        [((1, 56, 56, 96), torch.bfloat16)],
        {"model_names": ["pt_swin_swin_s_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add80,
        [((1, 64, 3, 49, 49), torch.bfloat16)],
        {"model_names": ["pt_swin_swin_s_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add32,
        [((16, 49, 576), torch.bfloat16)],
        {"model_names": ["pt_swin_swin_s_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((16, 6, 49, 49), torch.bfloat16), ((1, 6, 49, 49), torch.bfloat16)],
        {"model_names": ["pt_swin_swin_s_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add28,
        [((16, 49, 192), torch.bfloat16)],
        {"model_names": ["pt_swin_swin_s_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 28, 28, 192), torch.bfloat16), ((1, 28, 28, 192), torch.bfloat16)],
        {"model_names": ["pt_swin_swin_s_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add81,
        [((1, 28, 28, 768), torch.bfloat16)],
        {"model_names": ["pt_swin_swin_s_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add28,
        [((1, 28, 28, 192), torch.bfloat16)],
        {"model_names": ["pt_swin_swin_s_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add82,
        [((1, 16, 6, 49, 49), torch.bfloat16)],
        {"model_names": ["pt_swin_swin_s_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add83,
        [((4, 49, 1152), torch.bfloat16)],
        {"model_names": ["pt_swin_swin_s_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((4, 12, 49, 49), torch.bfloat16), ((1, 12, 49, 49), torch.bfloat16)],
        {"model_names": ["pt_swin_swin_s_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add37,
        [((4, 49, 384), torch.bfloat16)],
        {"model_names": ["pt_swin_swin_s_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 14, 14, 384), torch.bfloat16), ((1, 14, 14, 384), torch.bfloat16)],
        {"model_names": ["pt_swin_swin_s_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add39,
        [((1, 14, 14, 1536), torch.bfloat16)],
        {"model_names": ["pt_swin_swin_s_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add37,
        [((1, 14, 14, 384), torch.bfloat16)],
        {"model_names": ["pt_swin_swin_s_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add84,
        [((1, 4, 12, 49, 49), torch.bfloat16)],
        {"model_names": ["pt_swin_swin_s_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add38,
        [((1, 49, 2304), torch.bfloat16)],
        {"model_names": ["pt_swin_swin_s_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 24, 49, 49), torch.bfloat16), ((1, 24, 49, 49), torch.bfloat16)],
        {"model_names": ["pt_swin_swin_s_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add81,
        [((1, 49, 768), torch.bfloat16)],
        {"model_names": ["pt_swin_swin_s_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 7, 7, 768), torch.bfloat16), ((1, 7, 7, 768), torch.bfloat16)],
        {"model_names": ["pt_swin_swin_s_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add85,
        [((1, 7, 7, 3072), torch.bfloat16)],
        {"model_names": ["pt_swin_swin_s_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add81,
        [((1, 7, 7, 768), torch.bfloat16)],
        {"model_names": ["pt_swin_swin_s_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 64, 224, 224), torch.bfloat16), ((64, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_vgg_19_obj_det_hf",
                "pt_vgg_vgg16_img_cls_torchvision",
                "pt_vgg_vgg16_obj_det_osmr",
                "pt_vgg_bn_vgg19b_obj_det_osmr",
                "pt_vgg_vgg11_img_cls_torchvision",
                "pt_vgg_vgg13_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 128, 112, 112), torch.bfloat16), ((128, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_vgg_19_obj_det_hf",
                "pt_vgg_vgg16_img_cls_torchvision",
                "pt_vgg_vgg16_obj_det_osmr",
                "pt_vgg_bn_vgg19b_obj_det_osmr",
                "pt_vgg_vgg11_img_cls_torchvision",
                "pt_vgg_vgg13_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add86,
        [((1, 4096), torch.bfloat16)],
        {
            "model_names": [
                "pt_vgg_19_obj_det_hf",
                "pt_vgg_vgg16_img_cls_torchvision",
                "pt_vgg_vgg16_obj_det_osmr",
                "pt_vgg_bn_vgg19b_obj_det_osmr",
                "pt_vgg_vgg11_img_cls_torchvision",
                "pt_vgg_vgg13_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 768, 14, 14), torch.bfloat16), ((768, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_vit_google_vit_base_patch16_224_img_cls_hf",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_mlp_mixer_mixer_b16_224_in21k_img_cls_timm",
                "pt_vovnet_vovnet39_img_cls_osmr",
                "pt_beit_microsoft_beit_base_patch16_224_img_cls_hf",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_vit_vit_b_16_img_cls_torchvision",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_vovnet_v1_vovnet39_obj_det_torchhub",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_mlp_mixer_mixer_b16_224_miil_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add87,
        [((1, 197, 768), torch.bfloat16)],
        {
            "model_names": ["pt_vit_google_vit_base_patch16_224_img_cls_hf", "pt_vit_vit_b_16_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add81,
        [((1, 197, 768), torch.bfloat16)],
        {
            "model_names": [
                "pt_vit_google_vit_base_patch16_224_img_cls_hf",
                "pt_beit_microsoft_beit_base_patch16_224_img_cls_hf",
                "pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf",
                "pt_vit_vit_b_16_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add88,
        [((1, 12, 197, 197), torch.bfloat16)],
        {
            "model_names": ["pt_vit_google_vit_base_patch16_224_img_cls_hf", "pt_vit_vit_b_16_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 197, 768), torch.bfloat16), ((1, 197, 768), torch.bfloat16)],
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
        Add85,
        [((1, 197, 3072), torch.bfloat16)],
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
        Add89,
        [((1, 50, 1024), torch.bfloat16)],
        {"model_names": ["pt_vit_vit_l_32_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add85,
        [((50, 1, 3072), torch.bfloat16)],
        {"model_names": ["pt_vit_vit_l_32_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add90,
        [((1, 16, 50, 50), torch.bfloat16)],
        {"model_names": ["pt_vit_vit_l_32_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add21,
        [((50, 1024), torch.bfloat16)],
        {"model_names": ["pt_vit_vit_l_32_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 50, 1024), torch.bfloat16), ((1, 50, 1024), torch.bfloat16)],
        {"model_names": ["pt_vit_vit_l_32_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add86,
        [((1, 50, 4096), torch.bfloat16)],
        {"model_names": ["pt_vit_vit_l_32_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add21,
        [((1, 50, 1024), torch.bfloat16)],
        {"model_names": ["pt_vit_vit_l_32_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 256, 1, 1), torch.bfloat16), ((256, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add91,
        [((1, 256, 1, 1), torch.bfloat16)],
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
        Add0,
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
        Add0,
        [((1, 512, 1, 1), torch.bfloat16), ((512, 1, 1), torch.bfloat16)],
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
        Add91,
        [((1, 512, 1, 1), torch.bfloat16)],
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
        Add0,
        [((1, 192, 14, 14), torch.bfloat16), ((192, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_vovnet_vovnet39_img_cls_osmr",
                "pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_vovnet_v1_vovnet39_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
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
        Add81,
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
        Add0,
        [((1, 768, 14, 14), torch.bfloat16), ((1, 768, 14, 14), torch.bfloat16)],
        {
            "model_names": [
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_vovnet_vovnet39_img_cls_osmr",
                "pt_vovnet_v1_vovnet39_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 768, 1, 1), torch.bfloat16), ((768, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add91,
        [((1, 768, 1, 1), torch.bfloat16)],
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
        Add0,
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
        Add0,
        [((1, 1024, 1, 1), torch.bfloat16), ((1024, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add91,
        [((1, 1024, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
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
        Add92,
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
        Add0,
        [((1, 80, 28, 28), torch.bfloat16), ((80, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_vovnet_vovnet27s_img_cls_osmr", "pt_efficientnet_efficientnet_b7_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
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
        Add0,
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
        Add0,
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
        Add0,
        [((1, 64, 150, 150), torch.bfloat16), ((64, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_xception_xception41_img_cls_timm", "pt_xception_xception71_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 128, 150, 150), torch.bfloat16), ((128, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_xception_xception41_img_cls_timm", "pt_xception_xception71_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 128, 75, 75), torch.bfloat16), ((128, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_xception_xception41_img_cls_timm", "pt_xception_xception71_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 128, 75, 75), torch.bfloat16), ((1, 128, 75, 75), torch.bfloat16)],
        {
            "model_names": ["pt_xception_xception41_img_cls_timm", "pt_xception_xception71_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 256, 75, 75), torch.bfloat16), ((256, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_xception_xception41_img_cls_timm", "pt_xception_xception71_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 256, 38, 38), torch.bfloat16), ((256, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_xception_xception41_img_cls_timm", "pt_xception_xception71_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 256, 38, 38), torch.bfloat16), ((1, 256, 38, 38), torch.bfloat16)],
        {
            "model_names": ["pt_xception_xception41_img_cls_timm", "pt_xception_xception71_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
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
        Add93,
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
        Add0,
        [((1, 728, 38, 38), torch.bfloat16), ((728, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_xception_xception41_img_cls_timm", "pt_xception_xception71_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
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
        Add0,
        [((1, 728, 19, 19), torch.bfloat16), ((1, 728, 19, 19), torch.bfloat16)],
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
        Add0,
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
        Add0,
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
        Add0,
        [((1, 1024, 10, 10), torch.bfloat16), ((1, 1024, 10, 10), torch.bfloat16)],
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
        Add0,
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
        Add0,
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
        Add0,
        [((1, 256, 75, 75), torch.bfloat16), ((1, 256, 75, 75), torch.bfloat16)],
        {"model_names": ["pt_xception_xception71_img_cls_timm"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 728, 38, 38), torch.bfloat16), ((1, 728, 38, 38), torch.bfloat16)],
        {"model_names": ["pt_xception_xception71_img_cls_timm"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 64, 320, 320), torch.float32), ((64, 1, 1), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5l_img_cls_torchhub_640x640"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 128, 160, 160), torch.float32), ((128, 1, 1), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5l_img_cls_torchhub_640x640"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 64, 160, 160), torch.float32), ((64, 1, 1), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5l_img_cls_torchhub_640x640"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 64, 160, 160), torch.float32), ((1, 64, 160, 160), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5l_img_cls_torchhub_640x640"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 256, 80, 80), torch.float32), ((256, 1, 1), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5l_img_cls_torchhub_640x640"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 128, 80, 80), torch.float32), ((128, 1, 1), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5l_img_cls_torchhub_640x640"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 128, 80, 80), torch.float32), ((1, 128, 80, 80), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5l_img_cls_torchhub_640x640"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 512, 40, 40), torch.float32), ((512, 1, 1), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5l_img_cls_torchhub_640x640"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 256, 40, 40), torch.float32), ((256, 1, 1), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5l_img_cls_torchhub_640x640"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 256, 40, 40), torch.float32), ((1, 256, 40, 40), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5l_img_cls_torchhub_640x640"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1024, 20, 20), torch.float32), ((1024, 1, 1), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5l_img_cls_torchhub_640x640"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 512, 20, 20), torch.float32), ((512, 1, 1), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5l_img_cls_torchhub_640x640"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 512, 20, 20), torch.float32), ((1, 512, 20, 20), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5l_img_cls_torchhub_640x640"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 255, 80, 80), torch.float32), ((255, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5l_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_640x640",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
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
        Add0,
        [((1, 255, 40, 40), torch.float32), ((255, 1, 1), torch.float32)],
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
        Add0,
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
        Add0,
        [((1, 255, 20, 20), torch.float32), ((255, 1, 1), torch.float32)],
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
        Add0,
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
        Add0,
        [((1, 16, 240, 240), torch.float32), ((16, 1, 1), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5n_img_cls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 32, 120, 120), torch.float32), ((32, 1, 1), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5n_img_cls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 16, 120, 120), torch.float32), ((16, 1, 1), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5n_img_cls_torchhub_480x480"], "pcc": 0.99},
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
        [((1, 255, 60, 60), torch.float32), ((255, 1, 1), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5n_img_cls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 255, 60, 60), torch.float32), ((1, 255, 60, 60), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5n_img_cls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 255, 30, 30), torch.float32), ((255, 1, 1), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5n_img_cls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 255, 30, 30), torch.float32), ((1, 255, 30, 30), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5n_img_cls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 255, 15, 15), torch.float32), ((255, 1, 1), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5n_img_cls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 255, 15, 15), torch.float32), ((1, 255, 15, 15), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5n_img_cls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 80, 320, 320), torch.bfloat16), ((80, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_yolov8_yolov8x_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 160, 160, 160), torch.bfloat16), ((160, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_yolov8_yolov8x_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 80, 160, 160), torch.bfloat16), ((80, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_yolov8_yolov8x_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 80, 160, 160), torch.bfloat16), ((1, 80, 160, 160), torch.bfloat16)],
        {"model_names": ["pt_yolov8_yolov8x_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
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
        Add94,
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
        Add0,
        [((1, 320, 80, 80), torch.bfloat16), ((320, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_yolov8_yolov8x_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 160, 80, 80), torch.bfloat16), ((160, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_yolov8_yolov8x_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 160, 80, 80), torch.bfloat16), ((1, 160, 80, 80), torch.bfloat16)],
        {"model_names": ["pt_yolov8_yolov8x_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
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
        Add79,
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
        Add0,
        [((1, 640, 40, 40), torch.bfloat16), ((640, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_yolov8_yolov8x_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 320, 40, 40), torch.bfloat16), ((320, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_yolov8_yolov8x_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 320, 40, 40), torch.bfloat16), ((1, 320, 40, 40), torch.bfloat16)],
        {"model_names": ["pt_yolov8_yolov8x_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 640, 20, 20), torch.bfloat16), ((640, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_yolov8_yolov8x_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 320, 20, 20), torch.bfloat16), ((320, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_yolov8_yolov8x_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 320, 20, 20), torch.bfloat16), ((1, 320, 20, 20), torch.bfloat16)],
        {"model_names": ["pt_yolov8_yolov8x_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 80, 80, 80), torch.bfloat16), ((80, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolov8_yolov8x_obj_det_github",
                "pt_yolox_yolox_l_obj_det_torchhub",
                "pt_yolox_yolox_m_obj_det_torchhub",
                "pt_yolov9_default_obj_det_github",
                "pt_yolox_yolox_darknet_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 80, 40, 40), torch.bfloat16), ((80, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolov8_yolov8x_obj_det_github",
                "pt_yolox_yolox_l_obj_det_torchhub",
                "pt_yolox_yolox_m_obj_det_torchhub",
                "pt_yolov9_default_obj_det_github",
                "pt_yolox_yolox_darknet_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 64, 40, 40), torch.bfloat16), ((64, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_yolov8_yolov8x_obj_det_github", "pt_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 80, 20, 20), torch.bfloat16), ((80, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolov8_yolov8x_obj_det_github",
                "pt_yolox_yolox_l_obj_det_torchhub",
                "pt_yolox_yolox_m_obj_det_torchhub",
                "pt_yolov9_default_obj_det_github",
                "pt_yolox_yolox_darknet_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 64, 20, 20), torch.bfloat16), ((64, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_yolov8_yolov8x_obj_det_github", "pt_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add95,
        [((1, 2, 8400), torch.bfloat16)],
        {
            "model_names": ["pt_yolov8_yolov8x_obj_det_github", "pt_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 2, 8400), torch.bfloat16), ((1, 2, 8400), torch.bfloat16)],
        {
            "model_names": ["pt_yolov8_yolov8x_obj_det_github", "pt_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
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
        Add0,
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
        Add0,
        [((1, 64, 160, 160), torch.bfloat16), ((1, 64, 160, 160), torch.bfloat16)],
        {"model_names": ["pt_yolox_yolox_l_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
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
        Add0,
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
        Add0,
        [((1, 128, 80, 80), torch.bfloat16), ((1, 128, 80, 80), torch.bfloat16)],
        {"model_names": ["pt_yolox_yolox_l_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
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
        Add0,
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
        Add0,
        [((1, 256, 40, 40), torch.bfloat16), ((1, 256, 40, 40), torch.bfloat16)],
        {"model_names": ["pt_yolox_yolox_l_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
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
        Add0,
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
        Add0,
        [((1, 4, 80, 80), torch.bfloat16), ((4, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolox_yolox_l_obj_det_torchhub",
                "pt_yolox_yolox_m_obj_det_torchhub",
                "pt_yolox_yolox_darknet_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 1, 80, 80), torch.bfloat16), ((1, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolox_yolox_l_obj_det_torchhub",
                "pt_yolox_yolox_m_obj_det_torchhub",
                "pt_yolox_yolox_darknet_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 4, 40, 40), torch.bfloat16), ((4, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolox_yolox_l_obj_det_torchhub",
                "pt_yolox_yolox_m_obj_det_torchhub",
                "pt_yolox_yolox_darknet_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 1, 40, 40), torch.bfloat16), ((1, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolox_yolox_l_obj_det_torchhub",
                "pt_yolox_yolox_m_obj_det_torchhub",
                "pt_yolox_yolox_darknet_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 4, 20, 20), torch.bfloat16), ((4, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolox_yolox_l_obj_det_torchhub",
                "pt_yolox_yolox_m_obj_det_torchhub",
                "pt_yolox_yolox_darknet_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 1, 20, 20), torch.bfloat16), ((1, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolox_yolox_l_obj_det_torchhub",
                "pt_yolox_yolox_m_obj_det_torchhub",
                "pt_yolox_yolox_darknet_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 48, 320, 320), torch.bfloat16), ((48, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_yolox_yolox_m_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 96, 160, 160), torch.bfloat16), ((96, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_yolox_yolox_m_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 48, 160, 160), torch.bfloat16), ((48, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_yolox_yolox_m_obj_det_torchhub", "pt_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 48, 160, 160), torch.bfloat16), ((1, 48, 160, 160), torch.bfloat16)],
        {"model_names": ["pt_yolox_yolox_m_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 192, 80, 80), torch.bfloat16), ((192, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_yolox_yolox_m_obj_det_torchhub", "pt_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
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
        Add0,
        [((1, 96, 80, 80), torch.bfloat16), ((1, 96, 80, 80), torch.bfloat16)],
        {"model_names": ["pt_yolox_yolox_m_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 384, 40, 40), torch.bfloat16), ((384, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_yolox_yolox_m_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 192, 40, 40), torch.bfloat16), ((192, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_yolox_yolox_m_obj_det_torchhub", "pt_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 192, 40, 40), torch.bfloat16), ((1, 192, 40, 40), torch.bfloat16)],
        {"model_names": ["pt_yolox_yolox_m_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 768, 20, 20), torch.bfloat16), ((768, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_yolox_yolox_m_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 384, 20, 20), torch.bfloat16), ((384, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_yolox_yolox_m_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 192, 20, 20), torch.bfloat16), ((192, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_yolox_yolox_m_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 64, 128, 128), torch.float32), ((64, 1, 1), torch.float32)],
        {"model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Add96,
        [((1, 16384, 64), torch.float32)],
        {"model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 64, 16, 16), torch.float32), ((64, 1, 1), torch.float32)],
        {"model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Add96,
        [((1, 256, 64), torch.float32)],
        {"model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 16384, 64), torch.float32), ((1, 16384, 64), torch.float32)],
        {"model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Add97,
        [((1, 16384, 256), torch.float32)],
        {"model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 256, 128, 128), torch.float32), ((256, 1, 1), torch.float32)],
        {"model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Add1,
        [((1, 16384, 256), torch.float32)],
        {"model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 128, 64, 64), torch.float32), ((128, 1, 1), torch.float32)],
        {"model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Add98,
        [((1, 4096, 128), torch.float32)],
        {"model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 128, 16, 16), torch.float32), ((128, 1, 1), torch.float32)],
        {"model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Add98,
        [((1, 256, 128), torch.float32)],
        {"model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 4096, 128), torch.float32), ((1, 4096, 128), torch.float32)],
        {"model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Add99,
        [((1, 4096, 512), torch.float32)],
        {"model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 512, 64, 64), torch.float32), ((512, 1, 1), torch.float32)],
        {"model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Add1,
        [((1, 4096, 512), torch.float32)],
        {"model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 320, 32, 32), torch.float32), ((320, 1, 1), torch.float32)],
        {"model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Add100,
        [((1, 1024, 320), torch.float32)],
        {"model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 320, 16, 16), torch.float32), ((320, 1, 1), torch.float32)],
        {"model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Add100,
        [((1, 256, 320), torch.float32)],
        {"model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1024, 320), torch.float32), ((1, 1024, 320), torch.float32)],
        {"model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Add101,
        [((1, 1024, 1280), torch.float32)],
        {"model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1280, 32, 32), torch.float32), ((1280, 1, 1), torch.float32)],
        {"model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Add1,
        [((1, 1024, 1280), torch.float32)],
        {"model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 512, 16, 16), torch.float32), ((512, 1, 1), torch.float32)],
        {"model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Add99,
        [((1, 256, 512), torch.float32)],
        {"model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 256, 512), torch.float32), ((1, 256, 512), torch.float32)],
        {"model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Add102,
        [((1, 256, 2048), torch.float32)],
        {"model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 2048, 16, 16), torch.float32), ((2048, 1, 1), torch.float32)],
        {"model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Add1,
        [((1, 256, 2048), torch.float32)],
        {"model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Add103,
        [((1, 256, 768), torch.float32)],
        {"model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Add103,
        [((1, 1024, 768), torch.float32)],
        {"model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Add103,
        [((1, 4096, 768), torch.float32)],
        {"model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Add103,
        [((1, 16384, 768), torch.float32)],
        {"model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 768, 128, 128), torch.float32), ((768, 1, 1), torch.float32)],
        {"model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 150, 128, 128), torch.float32), ((150, 1, 1), torch.float32)],
        {"model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Add0,
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
        Add1,
        [((1, 9, 1), torch.float32)],
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
        Add22,
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
        Add0,
        [((1, 12, 9, 9), torch.float32), ((1, 1, 1, 9), torch.float32)],
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
        Add23,
        [((1, 9, 3072), torch.float32)],
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
        Add22,
        [((1, 768), torch.float32)],
        {
            "model_names": [
                "pd_roberta_rbt4_ch_seq_cls_padlenlp",
                "pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (Add14, [((1, 2), torch.float32)], {"model_names": ["pd_roberta_rbt4_ch_seq_cls_padlenlp"], "pcc": 0.99}),
    (
        Add21,
        [((1, 197, 1024), torch.bfloat16)],
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
        Add104,
        [((1, 16, 197, 197), torch.bfloat16)],
        {
            "model_names": ["pt_beit_microsoft_beit_large_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 16, 197, 197), torch.bfloat16), ((1, 16, 197, 197), torch.bfloat16)],
        {
            "model_names": ["pt_beit_microsoft_beit_large_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 197, 1024), torch.bfloat16), ((1, 197, 1024), torch.bfloat16)],
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
        Add86,
        [((1, 197, 4096), torch.bfloat16)],
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
        Add0,
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
        Add0,
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
        Add0,
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
        Add0,
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
        Add105,
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
        Add0,
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
        Add0,
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
        Add0,
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
        Add106,
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
        Add0,
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
        Add0,
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
        Add107,
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
        Add0,
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
        Add0,
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
        Add0,
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
        Add0,
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
        Add0,
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
        Add0,
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
        Add0,
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
        Add108,
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
        Add0,
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
        Add0,
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
        Add109,
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
        Add0,
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
        Add0,
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
        Add0,
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
        Add110,
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
        Add0,
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
        Add0,
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
        Add111,
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
        Add0,
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
        Add0,
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
        Add112,
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
        Add0,
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
        Add0,
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
        Add113,
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
        Add0,
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
        Add0,
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
        Add114,
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
        Add0,
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
        Add0,
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
        Add115,
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
        Add0,
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
        Add0,
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
        Add0,
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
        Add0,
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
        Add0,
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
        Add0,
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
        Add0,
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
        Add0,
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
        Add0,
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
        Add0,
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
        Add0,
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
        Add0,
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
        Add0,
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
        Add0,
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
        Add0,
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
        Add116,
        [((1, 18), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet121_hf_xray_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 256, 112, 112), torch.bfloat16), ((256, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_dla_dla102x2_visual_bb_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
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
        Add0,
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
        Add0,
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
        Add0,
        [((1, 64, 56, 56), torch.bfloat16), ((1, 64, 56, 56), torch.bfloat16)],
        {
            "model_names": [
                "pt_dla_dla46_c_visual_bb_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
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
        Add0,
        [((1, 64, 28, 28), torch.bfloat16), ((1, 64, 28, 28), torch.bfloat16)],
        {
            "model_names": [
                "pt_dla_dla46_c_visual_bb_torchvision",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
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
        Add0,
        [((1, 128, 14, 14), torch.bfloat16), ((1, 128, 14, 14), torch.bfloat16)],
        {
            "model_names": [
                "pt_dla_dla46_c_visual_bb_torchvision",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 256, 7, 7), torch.bfloat16), ((1, 256, 7, 7), torch.bfloat16)],
        {"model_names": ["pt_dla_dla46_c_visual_bb_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 24, 160, 160), torch.bfloat16), ((24, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 24, 160, 160), torch.bfloat16), ((1, 24, 160, 160), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 144, 160, 160), torch.bfloat16), ((144, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 144, 80, 80), torch.bfloat16), ((144, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 32, 80, 80), torch.bfloat16), ((32, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 32, 80, 80), torch.bfloat16), ((1, 32, 80, 80), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 56, 40, 40), torch.bfloat16), ((56, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 336, 40, 40), torch.bfloat16), ((336, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 56, 40, 40), torch.bfloat16), ((1, 56, 40, 40), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 336, 20, 20), torch.bfloat16), ((336, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 112, 20, 20), torch.bfloat16), ((112, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 672, 20, 20), torch.bfloat16), ((672, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 112, 20, 20), torch.bfloat16), ((1, 112, 20, 20), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 160, 20, 20), torch.bfloat16), ((160, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 960, 20, 20), torch.bfloat16), ((960, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 160, 20, 20), torch.bfloat16), ((1, 160, 20, 20), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 960, 10, 10), torch.bfloat16), ((960, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 272, 10, 10), torch.bfloat16), ((272, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 1632, 10, 10), torch.bfloat16), ((1632, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 272, 10, 10), torch.bfloat16), ((1, 272, 10, 10), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 448, 10, 10), torch.bfloat16), ((448, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 2688, 10, 10), torch.bfloat16), ((2688, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 448, 10, 10), torch.bfloat16), ((1, 448, 10, 10), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 1792, 10, 10), torch.bfloat16), ((1792, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 16, 1, 1), torch.bfloat16), ((16, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 64, 1, 1), torch.bfloat16), ((64, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
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
        Add0,
        [((1, 288, 56, 56), torch.bfloat16), ((288, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b7_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 48, 56, 56), torch.bfloat16), ((1, 48, 56, 56), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b7_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 20, 1, 1), torch.bfloat16), ((20, 1, 1), torch.bfloat16)],
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
        Add0,
        [((1, 480, 1, 1), torch.bfloat16), ((480, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 80, 28, 28), torch.bfloat16), ((1, 80, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b7_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 224, 14, 14), torch.bfloat16), ((224, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b7_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
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
        Add117,
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
        Add0,
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
        Add0,
        [((1, 1344, 1, 1), torch.bfloat16), ((1344, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b7_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 224, 14, 14), torch.bfloat16), ((1, 224, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b7_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
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
        Add0,
        [((3840,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b7_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add118,
        [((3840,), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b7_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 3840, 7, 7), torch.bfloat16), ((3840, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b7_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 160, 1, 1), torch.bfloat16), ((160, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b7_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 3840, 1, 1), torch.bfloat16), ((3840, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b7_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 640, 7, 7), torch.bfloat16), ((1, 640, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b7_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((2560,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b7_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add119,
        [((2560,), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b7_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 2560, 7, 7), torch.bfloat16), ((2560, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b7_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
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
        Add0,
        [((1, 256, 16, 16), torch.bfloat16), ((1, 256, 16, 16), torch.bfloat16)],
        {"model_names": ["pt_fpn_base_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 256, 64, 64), torch.bfloat16), ((1, 256, 64, 64), torch.bfloat16)],
        {
            "model_names": ["pt_fpn_base_img_cls_torchvision", "pt_yolo_v3_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
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
        Add0,
        [((8,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": ["pt_ghostnet_ghostnet_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add120,
        [((8,), torch.bfloat16)],
        {
            "model_names": ["pt_ghostnet_ghostnet_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 8, 112, 112), torch.bfloat16), ((8, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_ghostnet_ghostnet_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 16, 112, 112), torch.bfloat16), ((1, 16, 112, 112), torch.bfloat16)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((12,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": ["pt_ghostnet_ghostnet_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add121,
        [((12,), torch.bfloat16)],
        {
            "model_names": ["pt_ghostnet_ghostnet_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 12, 56, 56), torch.bfloat16), ((12, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_ghostnet_ghostnet_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
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
        Add0,
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
        Add0,
        [((1, 24, 56, 56), torch.bfloat16), ((1, 24, 56, 56), torch.bfloat16)],
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
        Add0,
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
        Add122,
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
        Add0,
        [((1, 36, 56, 56), torch.bfloat16), ((36, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_ghostnet_ghostnet_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 72, 1, 1), torch.bfloat16), ((72, 1, 1), torch.bfloat16)],
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
        Add91,
        [((1, 72, 1, 1), torch.bfloat16)],
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
        Add0,
        [((20,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": ["pt_ghostnet_ghostnet_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add123,
        [((20,), torch.bfloat16)],
        {
            "model_names": ["pt_ghostnet_ghostnet_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 20, 28, 28), torch.bfloat16), ((20, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_ghostnet_ghostnet_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
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
        Add0,
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
        Add0,
        [((1, 40, 28, 28), torch.bfloat16), ((1, 40, 28, 28), torch.bfloat16)],
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
        Add0,
        [((1, 120, 1, 1), torch.bfloat16), ((120, 1, 1), torch.bfloat16)],
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
        Add91,
        [((1, 120, 1, 1), torch.bfloat16)],
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
        Add0,
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
        Add0,
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
        Add0,
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
        Add0,
        [((1, 80, 14, 14), torch.bfloat16), ((1, 80, 14, 14), torch.bfloat16)],
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
        Add0,
        [((100,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": ["pt_ghostnet_ghostnet_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add124,
        [((100,), torch.bfloat16)],
        {
            "model_names": ["pt_ghostnet_ghostnet_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 100, 14, 14), torch.bfloat16), ((100, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_ghostnet_ghostnet_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((92,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": ["pt_ghostnet_ghostnet_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add125,
        [((92,), torch.bfloat16)],
        {
            "model_names": ["pt_ghostnet_ghostnet_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 92, 14, 14), torch.bfloat16), ((92, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_ghostnet_ghostnet_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add91,
        [((1, 480, 1, 1), torch.bfloat16)],
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
        Add0,
        [((1, 56, 14, 14), torch.bfloat16), ((56, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_ghostnet_ghostnet_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 168, 1, 1), torch.bfloat16), ((168, 1, 1), torch.bfloat16)],
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
        Add91,
        [((1, 672, 1, 1), torch.bfloat16)],
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
        Add0,
        [((1, 80, 7, 7), torch.bfloat16), ((80, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_ghostnet_ghostnet_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
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
        Add0,
        [((1, 160, 7, 7), torch.bfloat16), ((1, 160, 7, 7), torch.bfloat16)],
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
        Add0,
        [((1, 480, 7, 7), torch.bfloat16), ((480, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_ghostnet_ghostnet_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add91,
        [((1, 960, 1, 1), torch.bfloat16)],
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
        Add0,
        [((1, 1280, 1, 1), torch.bfloat16), ((1280, 1, 1), torch.bfloat16)],
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
        Add0,
        [((1, 16, 56, 56), torch.bfloat16), ((1, 16, 56, 56), torch.bfloat16)],
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
        Add0,
        [((1, 32, 28, 28), torch.bfloat16), ((1, 32, 28, 28), torch.bfloat16)],
        {
            "model_names": [
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
        Add0,
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
        Add0,
        [((1, 64, 14, 14), torch.bfloat16), ((1, 64, 14, 14), torch.bfloat16)],
        {
            "model_names": [
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
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
        Add0,
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
        Add0,
        [((1, 128, 7, 7), torch.bfloat16), ((1, 128, 7, 7), torch.bfloat16)],
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
        Add0,
        [((18,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add116,
        [((18,), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 18, 56, 56), torch.bfloat16), ((18, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 18, 56, 56), torch.bfloat16), ((1, 18, 56, 56), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 36, 28, 28), torch.bfloat16), ((36, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 36, 28, 28), torch.bfloat16), ((1, 36, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 18, 28, 28), torch.bfloat16), ((18, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 72, 14, 14), torch.bfloat16), ((72, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 72, 14, 14), torch.bfloat16), ((1, 72, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 18, 14, 14), torch.bfloat16), ((18, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 36, 14, 14), torch.bfloat16), ((36, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 144, 7, 7), torch.bfloat16), ((144, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 144, 7, 7), torch.bfloat16), ((1, 144, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 126, 7, 7), torch.bfloat16), ((126, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add37,
        [((1, 768, 384), torch.bfloat16)],
        {
            "model_names": [
                "pt_mlp_mixer_mixer_b16_224_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_miil_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add66,
        [((1, 768, 196), torch.bfloat16)],
        {
            "model_names": [
                "pt_mlp_mixer_mixer_b16_224_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_miil_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 196, 768), torch.bfloat16), ((1, 196, 768), torch.bfloat16)],
        {
            "model_names": [
                "pt_mlp_mixer_mixer_b16_224_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_miil_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add85,
        [((1, 196, 3072), torch.bfloat16)],
        {
            "model_names": [
                "pt_mlp_mixer_mixer_b16_224_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_miil_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add81,
        [((1, 196, 768), torch.bfloat16)],
        {
            "model_names": [
                "pt_mlp_mixer_mixer_b16_224_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_miil_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add126,
        [((1, 512, 49), torch.bfloat16)],
        {"model_names": ["pt_mlp_mixer_mixer_s32_224_img_cls_timm"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 49, 512), torch.bfloat16), ((1, 49, 512), torch.bfloat16)],
        {"model_names": ["pt_mlp_mixer_mixer_s32_224_img_cls_timm"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add64,
        [((1, 49, 2048), torch.bfloat16)],
        {"model_names": ["pt_mlp_mixer_mixer_s32_224_img_cls_timm"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add20,
        [((1, 49, 512), torch.bfloat16)],
        {"model_names": ["pt_mlp_mixer_mixer_s32_224_img_cls_timm"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 24, 96, 96), torch.bfloat16), ((24, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv1_google_mobilenet_v1_0_75_192_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 48, 96, 96), torch.bfloat16), ((48, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv1_google_mobilenet_v1_0_75_192_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 48, 48, 48), torch.bfloat16), ((48, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv1_google_mobilenet_v1_0_75_192_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 96, 48, 48), torch.bfloat16), ((96, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv1_google_mobilenet_v1_0_75_192_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 96, 24, 24), torch.bfloat16), ((96, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv1_google_mobilenet_v1_0_75_192_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 192, 24, 24), torch.bfloat16), ((192, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv1_google_mobilenet_v1_0_75_192_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 192, 12, 12), torch.bfloat16), ((192, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv1_google_mobilenet_v1_0_75_192_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 384, 12, 12), torch.bfloat16), ((384, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv1_google_mobilenet_v1_0_75_192_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 384, 6, 6), torch.bfloat16), ((384, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv1_google_mobilenet_v1_0_75_192_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 768, 6, 6), torch.bfloat16), ((768, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv1_google_mobilenet_v1_0_75_192_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 24, 80, 80), torch.bfloat16), ((24, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 16, 80, 80), torch.bfloat16), ((16, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 96, 40, 40), torch.bfloat16), ((96, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 24, 40, 40), torch.bfloat16), ((24, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 144, 40, 40), torch.bfloat16), ((144, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 24, 40, 40), torch.bfloat16), ((1, 24, 40, 40), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 144, 20, 20), torch.bfloat16), ((144, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 24, 20, 20), torch.bfloat16), ((24, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 24, 20, 20), torch.bfloat16), ((1, 24, 20, 20), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 144, 10, 10), torch.bfloat16), ((144, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 48, 10, 10), torch.bfloat16), ((48, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 288, 10, 10), torch.bfloat16), ((288, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 48, 10, 10), torch.bfloat16), ((1, 48, 10, 10), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 72, 10, 10), torch.bfloat16), ((72, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 432, 10, 10), torch.bfloat16), ((432, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 72, 10, 10), torch.bfloat16), ((1, 72, 10, 10), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 432, 5, 5), torch.bfloat16), ((432, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 120, 5, 5), torch.bfloat16), ((120, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
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
        Add127,
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
        Add0,
        [((1, 720, 5, 5), torch.bfloat16), ((720, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 120, 5, 5), torch.bfloat16), ((1, 120, 5, 5), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 240, 5, 5), torch.bfloat16), ((240, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
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
        Add128,
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
        Add0,
        [((1, 1280, 5, 5), torch.bfloat16), ((1280, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
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
        Add0,
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
        Add0,
        [((1, 320, 7, 7), torch.bfloat16), ((320, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_mobilenet_v2_img_cls_torchvision", "pt_mobilenetv2_basic_img_cls_torchhub"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
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
        Add91,
        [((1, 16, 112, 112), torch.bfloat16)],
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
        Add0,
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
        Add91,
        [((1, 240, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add91,
        [((1, 240, 14, 14), torch.bfloat16)],
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
        Add91,
        [((1, 200, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((184,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add129,
        [((184,), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 184, 14, 14), torch.bfloat16), ((184, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add91,
        [((1, 184, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add91,
        [((1, 480, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add91,
        [((1, 672, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add91,
        [((1, 672, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add91,
        [((1, 960, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add91,
        [((1, 1280, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 256, 80, 80), torch.bfloat16), ((1, 256, 80, 80), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv3_ssd_resnet101_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet152_img_cls_torchvision",
                "pt_yolox_yolox_darknet_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 512, 40, 40), torch.bfloat16), ((1, 512, 40, 40), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv3_ssd_resnet101_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet152_img_cls_torchvision",
                "pt_yolox_yolox_darknet_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 1024, 20, 20), torch.bfloat16), ((1, 1024, 20, 20), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv3_ssd_resnet101_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet152_img_cls_torchvision",
                "pt_yolox_yolox_darknet_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 2048, 10, 10), torch.bfloat16), ((1, 2048, 10, 10), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv3_ssd_resnet101_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet152_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
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
        Add0,
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
        Add0,
        [((1, 64, 48, 160), torch.bfloat16), ((1, 64, 48, 160), torch.bfloat16)],
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
        Add0,
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
        Add0,
        [((1, 128, 24, 80), torch.bfloat16), ((1, 128, 24, 80), torch.bfloat16)],
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
        Add0,
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
        Add0,
        [((1, 256, 12, 40), torch.bfloat16), ((1, 256, 12, 40), torch.bfloat16)],
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
        Add0,
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
        Add0,
        [((1, 512, 6, 20), torch.bfloat16), ((1, 512, 6, 20), torch.bfloat16)],
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
        Add0,
        [((1, 256, 6, 20), torch.bfloat16), ((256, 1, 1), torch.bfloat16)],
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
        Add0,
        [((1, 128, 12, 40), torch.bfloat16), ((128, 1, 1), torch.bfloat16)],
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
        Add0,
        [((1, 64, 24, 80), torch.bfloat16), ((64, 1, 1), torch.bfloat16)],
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
        Add0,
        [((1, 32, 48, 160), torch.bfloat16), ((32, 1, 1), torch.bfloat16)],
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
        Add0,
        [((1, 32, 96, 320), torch.bfloat16), ((32, 1, 1), torch.bfloat16)],
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
        Add0,
        [((1, 16, 96, 320), torch.bfloat16), ((16, 1, 1), torch.bfloat16)],
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
        Add0,
        [((1, 16, 192, 640), torch.bfloat16), ((16, 1, 1), torch.bfloat16)],
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
        Add0,
        [((1, 1, 192, 640), torch.bfloat16), ((1, 1, 1), torch.bfloat16)],
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
        Add130,
        [((1, 11, 2560), torch.float32)],
        {"model_names": ["pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 32, 11, 32), torch.float32), ((1, 32, 11, 32), torch.float32)],
        {"model_names": ["pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Add131,
        [((1, 32, 11, 11), torch.float32)],
        {"model_names": ["pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Add132,
        [((1, 11, 10240), torch.float32)],
        {"model_names": ["pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 11, 2560), torch.float32), ((1, 11, 2560), torch.float32)],
        {"model_names": ["pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf"], "pcc": 0.99},
    ),
    (Add1, [((1, 6, 1), torch.float32)], {"model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"], "pcc": 0.99}),
    (Add9, [((1, 6, 1024), torch.float32)], {"model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"], "pcc": 0.99}),
    (
        Add0,
        [((1, 16, 6, 64), torch.float32), ((1, 16, 6, 64), torch.float32)],
        {"model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1, 6, 6), torch.float32), ((1, 1, 6, 6), torch.float32)],
        {"model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"], "pcc": 0.99},
    ),
    (Add133, [((1, 1, 6, 6), torch.float32)], {"model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"], "pcc": 0.99}),
    (
        Add0,
        [((1, 16, 6, 6), torch.float32), ((1, 1, 6, 6), torch.float32)],
        {"model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 6, 1024), torch.float32), ((1, 6, 1024), torch.float32)],
        {"model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Add1,
        [((1, 35, 1), torch.float32)],
        {"model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Add134,
        [((1, 35, 1536), torch.float32)],
        {"model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 12, 35, 128), torch.float32), ((1, 12, 35, 128), torch.float32)],
        {"model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Add7,
        [((1, 35, 256), torch.float32)],
        {"model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 2, 35, 128), torch.float32), ((1, 2, 35, 128), torch.float32)],
        {"model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1, 35, 35), torch.float32), ((1, 1, 35, 35), torch.float32)],
        {"model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Add135,
        [((1, 1, 35, 35), torch.float32)],
        {"model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 12, 35, 35), torch.float32), ((1, 1, 35, 35), torch.float32)],
        {"model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 35, 1536), torch.float32), ((1, 35, 1536), torch.float32)],
        {"model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Add1,
        [((1, 29, 1), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf", "pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"], "pcc": 0.99},
    ),
    (Add134, [((1, 29, 1536), torch.float32)], {"model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99}),
    (
        Add0,
        [((1, 12, 29, 128), torch.float32), ((1, 12, 29, 128), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99},
    ),
    (Add7, [((1, 29, 256), torch.float32)], {"model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99}),
    (
        Add0,
        [((1, 2, 29, 128), torch.float32), ((1, 2, 29, 128), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1, 29, 29), torch.float32), ((1, 1, 29, 29), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf", "pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Add136,
        [((1, 1, 29, 29), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf", "pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 12, 29, 29), torch.float32), ((1, 1, 29, 29), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 29, 1536), torch.float32), ((1, 29, 1536), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1232,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_facebook_regnet_y_160_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add137,
        [((1232,), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_facebook_regnet_y_160_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 1232, 28, 28), torch.bfloat16), ((1232, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_facebook_regnet_y_160_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 1232, 14, 14), torch.bfloat16), ((1232, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_facebook_regnet_y_160_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 1232, 1, 1), torch.bfloat16), ((1232, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_facebook_regnet_y_160_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 1232, 14, 14), torch.bfloat16), ((1, 1232, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_facebook_regnet_y_160_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 308, 1, 1), torch.bfloat16), ((308, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_facebook_regnet_y_160_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((3024,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_facebook_regnet_y_160_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add138,
        [((3024,), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_facebook_regnet_y_160_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 3024, 14, 14), torch.bfloat16), ((3024, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_facebook_regnet_y_160_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 3024, 7, 7), torch.bfloat16), ((3024, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_facebook_regnet_y_160_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 3024, 1, 1), torch.bfloat16), ((3024, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_facebook_regnet_y_160_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 3024, 7, 7), torch.bfloat16), ((1, 3024, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_facebook_regnet_y_160_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 232, 112, 112), torch.bfloat16), ((232, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_facebook_regnet_y_320_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 232, 56, 56), torch.bfloat16), ((232, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_facebook_regnet_y_320_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 232, 1, 1), torch.bfloat16), ((232, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_facebook_regnet_y_320_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 232, 56, 56), torch.bfloat16), ((1, 232, 56, 56), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_facebook_regnet_y_320_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((696,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_facebook_regnet_y_320_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add139,
        [((696,), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_facebook_regnet_y_320_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 696, 56, 56), torch.bfloat16), ((696, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_facebook_regnet_y_320_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 696, 28, 28), torch.bfloat16), ((696, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_facebook_regnet_y_320_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 696, 1, 1), torch.bfloat16), ((696, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_facebook_regnet_y_320_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 696, 28, 28), torch.bfloat16), ((1, 696, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_facebook_regnet_y_320_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 174, 1, 1), torch.bfloat16), ((174, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_facebook_regnet_y_320_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 1392, 28, 28), torch.bfloat16), ((1392, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_facebook_regnet_y_320_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 1392, 14, 14), torch.bfloat16), ((1392, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_facebook_regnet_y_320_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 1392, 14, 14), torch.bfloat16), ((1, 1392, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_facebook_regnet_y_320_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 348, 1, 1), torch.bfloat16), ((348, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_facebook_regnet_y_320_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((3712,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_facebook_regnet_y_320_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add140,
        [((3712,), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_facebook_regnet_y_320_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 3712, 14, 14), torch.bfloat16), ((3712, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_facebook_regnet_y_320_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 3712, 7, 7), torch.bfloat16), ((3712, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_facebook_regnet_y_320_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 3712, 1, 1), torch.bfloat16), ((3712, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_facebook_regnet_y_320_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 3712, 7, 7), torch.bfloat16), ((1, 3712, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_facebook_regnet_y_320_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 336, 56, 56), torch.bfloat16), ((336, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_x_32gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 336, 112, 112), torch.bfloat16), ((336, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_x_32gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 336, 56, 56), torch.bfloat16), ((1, 336, 56, 56), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_x_32gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 672, 28, 28), torch.bfloat16), ((672, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_x_32gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 672, 56, 56), torch.bfloat16), ((672, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_x_32gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 672, 28, 28), torch.bfloat16), ((1, 672, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_x_32gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 1344, 28, 28), torch.bfloat16), ((1344, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_x_32gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 1344, 14, 14), torch.bfloat16), ((1, 1344, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_x_32gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((2520,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_x_32gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add141,
        [((2520,), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_x_32gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 2520, 7, 7), torch.bfloat16), ((2520, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_x_32gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 2520, 14, 14), torch.bfloat16), ((2520, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_x_32gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 2520, 7, 7), torch.bfloat16), ((1, 2520, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_x_32gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 96, 56, 56), torch.bfloat16), ((1, 96, 56, 56), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_x_3_2gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 192, 28, 28), torch.bfloat16), ((1, 192, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_x_3_2gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 432, 14, 14), torch.bfloat16), ((1, 432, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_x_3_2gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1008,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_x_3_2gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add142,
        [((1008,), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_x_3_2gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 1008, 7, 7), torch.bfloat16), ((1008, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_x_3_2gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 1008, 14, 14), torch.bfloat16), ((1008, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_x_3_2gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 1008, 7, 7), torch.bfloat16), ((1, 1008, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_x_3_2gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 96, 64, 64), torch.float32), ((96, 1, 1), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99},
    ),
    (Add143, [((64, 64, 288), torch.float32)], {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99}),
    (
        Add8,
        [((1, 15, 15, 512), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((64, 3, 64, 64), torch.float32), ((1, 3, 64, 64), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99},
    ),
    (Add144, [((64, 64, 96), torch.float32)], {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99}),
    (
        Add0,
        [((1, 64, 64, 96), torch.float32), ((1, 64, 64, 96), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add145,
        [((1, 64, 64, 384), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add144,
        [((1, 64, 64, 96), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add146,
        [((1, 64, 3, 64, 64), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99},
    ),
    (Add147, [((16, 64, 576), torch.float32)], {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99}),
    (
        Add0,
        [((16, 6, 64, 64), torch.float32), ((1, 6, 64, 64), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99},
    ),
    (Add148, [((16, 64, 192), torch.float32)], {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99}),
    (
        Add0,
        [((1, 32, 32, 192), torch.float32), ((1, 32, 32, 192), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add22,
        [((1, 32, 32, 768), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add148,
        [((1, 32, 32, 192), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add149,
        [((1, 16, 6, 64, 64), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99},
    ),
    (Add150, [((4, 64, 1152), torch.float32)], {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99}),
    (
        Add0,
        [((4, 12, 64, 64), torch.float32), ((1, 12, 64, 64), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99},
    ),
    (Add145, [((4, 64, 384), torch.float32)], {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99}),
    (
        Add0,
        [((1, 16, 16, 384), torch.float32), ((1, 16, 16, 384), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add134,
        [((1, 16, 16, 1536), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add145,
        [((1, 16, 16, 384), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add151,
        [((1, 4, 12, 64, 64), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99},
    ),
    (Add152, [((1, 64, 2304), torch.float32)], {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99}),
    (
        Add0,
        [((1, 24, 64, 64), torch.float32), ((1, 24, 64, 64), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99},
    ),
    (Add22, [((1, 64, 768), torch.float32)], {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99}),
    (
        Add0,
        [((1, 8, 8, 768), torch.float32), ((1, 8, 8, 768), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Add23,
        [((1, 8, 8, 3072), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99},
    ),
    (Add22, [((1, 8, 8, 768), torch.float32)], {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99}),
    (
        Add1,
        [((1, 513, 1), torch.float32)],
        {"model_names": ["pt_t5_t5_base_text_gen_hf", "pt_t5_t5_small_text_gen_hf"], "pcc": 0.99},
    ),
    (Add153, [((1, 12, 513, 513), torch.float32)], {"model_names": ["pt_t5_t5_base_text_gen_hf"], "pcc": 0.99}),
    (
        Add0,
        [((1, 12, 513, 513), torch.float32), ((1, 12, 513, 513), torch.float32)],
        {"model_names": ["pt_t5_t5_base_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 513, 768), torch.float32), ((1, 513, 768), torch.float32)],
        {"model_names": ["pt_t5_t5_base_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Add1,
        [((1, 61, 1), torch.float32)],
        {"model_names": ["pt_t5_t5_base_text_gen_hf", "pt_t5_t5_small_text_gen_hf"], "pcc": 0.99},
    ),
    (Add154, [((1, 12, 61, 61), torch.float32)], {"model_names": ["pt_t5_t5_base_text_gen_hf"], "pcc": 0.99}),
    (
        Add0,
        [((1, 12, 61, 61), torch.float32), ((1, 12, 61, 61), torch.float32)],
        {"model_names": ["pt_t5_t5_base_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 61, 768), torch.float32), ((1, 61, 768), torch.float32)],
        {"model_names": ["pt_t5_t5_base_text_gen_hf"], "pcc": 0.99},
    ),
    (Add155, [((1, 12, 513, 61), torch.float32)], {"model_names": ["pt_t5_t5_base_text_gen_hf"], "pcc": 0.99}),
    (
        Add0,
        [((1, 32, 480, 640), torch.bfloat16), ((32, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_yolo_v4_default_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add91,
        [((1, 32, 480, 640), torch.bfloat16)],
        {"model_names": ["pt_yolo_v4_default_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add91,
        [((1, 64, 240, 320), torch.bfloat16)],
        {"model_names": ["pt_yolo_v4_default_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 32, 240, 320), torch.bfloat16), ((32, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_yolo_v4_default_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add91,
        [((1, 32, 240, 320), torch.bfloat16)],
        {"model_names": ["pt_yolo_v4_default_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 64, 240, 320), torch.bfloat16), ((1, 64, 240, 320), torch.bfloat16)],
        {"model_names": ["pt_yolo_v4_default_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add91,
        [((1, 128, 120, 160), torch.bfloat16)],
        {"model_names": ["pt_yolo_v4_default_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add91,
        [((1, 64, 120, 160), torch.bfloat16)],
        {"model_names": ["pt_yolo_v4_default_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 64, 120, 160), torch.bfloat16), ((1, 64, 120, 160), torch.bfloat16)],
        {"model_names": ["pt_yolo_v4_default_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add91,
        [((1, 256, 60, 80), torch.bfloat16)],
        {"model_names": ["pt_yolo_v4_default_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add91,
        [((1, 128, 60, 80), torch.bfloat16)],
        {"model_names": ["pt_yolo_v4_default_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 128, 60, 80), torch.bfloat16), ((1, 128, 60, 80), torch.bfloat16)],
        {"model_names": ["pt_yolo_v4_default_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add91,
        [((1, 512, 30, 40), torch.bfloat16)],
        {"model_names": ["pt_yolo_v4_default_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add91,
        [((1, 256, 30, 40), torch.bfloat16)],
        {"model_names": ["pt_yolo_v4_default_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 1024, 15, 20), torch.bfloat16), ((1024, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_yolo_v4_default_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add91,
        [((1, 1024, 15, 20), torch.bfloat16)],
        {"model_names": ["pt_yolo_v4_default_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add91,
        [((1, 512, 15, 20), torch.bfloat16)],
        {"model_names": ["pt_yolo_v4_default_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 512, 15, 20), torch.bfloat16), ((1, 512, 15, 20), torch.bfloat16)],
        {"model_names": ["pt_yolo_v4_default_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 128, 30, 40), torch.bfloat16), ((128, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_yolo_v4_default_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 255, 60, 80), torch.bfloat16), ((255, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_yolo_v4_default_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 255, 30, 40), torch.bfloat16), ((255, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_yolo_v4_default_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 255, 15, 20), torch.bfloat16), ((255, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_yolo_v4_default_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 80, 160, 160), torch.float32), ((80, 1, 1), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5x_img_cls_torchhub_320x320"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 160, 80, 80), torch.float32), ((160, 1, 1), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5x_img_cls_torchhub_320x320"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 80, 80, 80), torch.float32), ((80, 1, 1), torch.float32)],
        {
            "model_names": ["pt_yolo_v5_yolov5x_img_cls_torchhub_320x320", "onnx_yolov8_default_obj_det_github"],
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
        {"model_names": ["pt_yolo_v5_yolov5x_img_cls_torchhub_320x320"], "pcc": 0.99},
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
        {"model_names": ["pt_yolo_v5_yolov5x_img_cls_torchhub_320x320"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 320, 20, 20), torch.float32), ((320, 1, 1), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5x_img_cls_torchhub_320x320"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 320, 20, 20), torch.float32), ((1, 320, 20, 20), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5x_img_cls_torchhub_320x320"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1280, 10, 10), torch.float32), ((1280, 1, 1), torch.float32)],
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
        [((1, 255, 10, 10), torch.float32), ((255, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5x_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_320x320",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
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
        Add0,
        [((1, 32, 150, 150), torch.float32), ((32, 1, 1), torch.float32)],
        {"model_names": ["onnx_xception_xception71_tf_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 64, 150, 150), torch.float32), ((64, 1, 1), torch.float32)],
        {"model_names": ["onnx_xception_xception71_tf_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 128, 150, 150), torch.float32), ((128, 1, 1), torch.float32)],
        {"model_names": ["onnx_xception_xception71_tf_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 128, 75, 75), torch.float32), ((128, 1, 1), torch.float32)],
        {"model_names": ["onnx_xception_xception71_tf_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 128, 75, 75), torch.float32), ((1, 128, 75, 75), torch.float32)],
        {"model_names": ["onnx_xception_xception71_tf_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 256, 75, 75), torch.float32), ((256, 1, 1), torch.float32)],
        {"model_names": ["onnx_xception_xception71_tf_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 256, 75, 75), torch.float32), ((1, 256, 75, 75), torch.float32)],
        {"model_names": ["onnx_xception_xception71_tf_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 256, 38, 38), torch.float32), ((256, 1, 1), torch.float32)],
        {"model_names": ["onnx_xception_xception71_tf_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 256, 38, 38), torch.float32), ((1, 256, 38, 38), torch.float32)],
        {"model_names": ["onnx_xception_xception71_tf_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 728, 38, 38), torch.float32), ((728, 1, 1), torch.float32)],
        {"model_names": ["onnx_xception_xception71_tf_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 728, 38, 38), torch.float32), ((1, 728, 38, 38), torch.float32)],
        {"model_names": ["onnx_xception_xception71_tf_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 728, 19, 19), torch.float32), ((728, 1, 1), torch.float32)],
        {"model_names": ["onnx_xception_xception71_tf_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 728, 19, 19), torch.float32), ((1, 728, 19, 19), torch.float32)],
        {"model_names": ["onnx_xception_xception71_tf_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1024, 19, 19), torch.float32), ((1024, 1, 1), torch.float32)],
        {"model_names": ["onnx_xception_xception71_tf_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1024, 10, 10), torch.float32), ((1024, 1, 1), torch.float32)],
        {"model_names": ["onnx_xception_xception71_tf_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1024, 10, 10), torch.float32), ((1, 1024, 10, 10), torch.float32)],
        {"model_names": ["onnx_xception_xception71_tf_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1536, 10, 10), torch.float32), ((1536, 1, 1), torch.float32)],
        {"model_names": ["onnx_xception_xception71_tf_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 2048, 10, 10), torch.float32), ((2048, 1, 1), torch.float32)],
        {"model_names": ["onnx_xception_xception71_tf_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (Add10, [((1, 128, 2048), torch.float32)], {"model_names": ["pt_albert_xlarge_v2_token_cls_hf"], "pcc": 0.99}),
    (
        Add0,
        [((1, 128, 2048), torch.float32), ((1, 128, 2048), torch.float32)],
        {"model_names": ["pt_albert_xlarge_v2_token_cls_hf"], "pcc": 0.99},
    ),
    (Add69, [((1, 128, 8192), torch.float32)], {"model_names": ["pt_albert_xlarge_v2_token_cls_hf"], "pcc": 0.99}),
    (
        Add104,
        [((1, 12, 197, 197), torch.bfloat16)],
        {
            "model_names": ["pt_beit_microsoft_beit_base_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 12, 197, 197), torch.bfloat16), ((1, 12, 197, 197), torch.bfloat16)],
        {
            "model_names": ["pt_beit_microsoft_beit_base_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add156,
        [((1, 197, 192), torch.bfloat16)],
        {
            "model_names": ["pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add28,
        [((1, 197, 192), torch.bfloat16)],
        {
            "model_names": ["pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add88,
        [((1, 3, 197, 197), torch.bfloat16)],
        {
            "model_names": ["pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 197, 192), torch.bfloat16), ((1, 197, 192), torch.bfloat16)],
        {
            "model_names": ["pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 384, 1, 1), torch.bfloat16), ((384, 1, 1), torch.bfloat16)],
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
        Add0,
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
        Add157,
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
        Add0,
        [((1, 176, 14, 14), torch.bfloat16), ((176, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b5_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
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
        Add0,
        [((1, 44, 1, 1), torch.bfloat16), ((44, 1, 1), torch.bfloat16)],
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
        Add0,
        [((1, 176, 14, 14), torch.bfloat16), ((1, 176, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b5_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
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
        Add0,
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
        Add158,
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
        Add0,
        [((1, 304, 7, 7), torch.bfloat16), ((304, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b5_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
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
        Add159,
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
        Add0,
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
        Add0,
        [((1, 76, 1, 1), torch.bfloat16), ((76, 1, 1), torch.bfloat16)],
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
        Add0,
        [((1, 1824, 1, 1), torch.bfloat16), ((1824, 1, 1), torch.bfloat16)],
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
        Add0,
        [((1, 304, 7, 7), torch.bfloat16), ((1, 304, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b5_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
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
        Add85,
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
        Add0,
        [((1, 3072, 7, 7), torch.bfloat16), ((3072, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b5_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 128, 1, 1), torch.bfloat16), ((128, 1, 1), torch.bfloat16)],
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
        Add0,
        [((1, 3072, 1, 1), torch.bfloat16), ((3072, 1, 1), torch.bfloat16)],
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
        Add0,
        [((1, 512, 7, 7), torch.bfloat16), ((1, 512, 7, 7), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 24, 150, 150), torch.bfloat16), ((24, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 144, 150, 150), torch.bfloat16), ((144, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 144, 75, 75), torch.bfloat16), ((144, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 32, 75, 75), torch.bfloat16), ((32, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 192, 75, 75), torch.bfloat16), ((192, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 32, 75, 75), torch.bfloat16), ((1, 32, 75, 75), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 192, 38, 38), torch.bfloat16), ((192, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 48, 38, 38), torch.bfloat16), ((48, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 288, 38, 38), torch.bfloat16), ((288, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 48, 38, 38), torch.bfloat16), ((1, 48, 38, 38), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 288, 19, 19), torch.bfloat16), ((288, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 96, 19, 19), torch.bfloat16), ((96, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 576, 19, 19), torch.bfloat16), ((576, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 96, 19, 19), torch.bfloat16), ((1, 96, 19, 19), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 136, 19, 19), torch.bfloat16), ((136, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 816, 19, 19), torch.bfloat16), ((816, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 136, 19, 19), torch.bfloat16), ((1, 136, 19, 19), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 816, 10, 10), torch.bfloat16), ((816, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 232, 10, 10), torch.bfloat16), ((232, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 1392, 10, 10), torch.bfloat16), ((1392, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 232, 10, 10), torch.bfloat16), ((1, 232, 10, 10), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 384, 10, 10), torch.bfloat16), ((384, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 1280, 10, 10), torch.bfloat16), ((1280, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
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
        Add0,
        [((1, 32, 147, 147), torch.bfloat16), ((32, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_inception_inception_v4_tf_in1k_img_cls_timm", "pt_inception_inception_v4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
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
        Add0,
        [((1, 96, 73, 73), torch.bfloat16), ((96, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_inception_inception_v4_tf_in1k_img_cls_timm", "pt_inception_inception_v4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 64, 73, 73), torch.bfloat16), ((64, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_inception_inception_v4_tf_in1k_img_cls_timm", "pt_inception_inception_v4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 96, 71, 71), torch.bfloat16), ((96, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_inception_inception_v4_tf_in1k_img_cls_timm", "pt_inception_inception_v4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 192, 35, 35), torch.bfloat16), ((192, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_inception_inception_v4_tf_in1k_img_cls_timm", "pt_inception_inception_v4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 224, 35, 35), torch.bfloat16), ((224, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_inception_inception_v4_tf_in1k_img_cls_timm", "pt_inception_inception_v4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 96, 35, 35), torch.bfloat16), ((96, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_inception_inception_v4_tf_in1k_img_cls_timm", "pt_inception_inception_v4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 384, 17, 17), torch.bfloat16), ((384, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_inception_inception_v4_tf_in1k_img_cls_timm", "pt_inception_inception_v4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 256, 17, 17), torch.bfloat16), ((256, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_inception_inception_v4_tf_in1k_img_cls_timm", "pt_inception_inception_v4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 768, 17, 17), torch.bfloat16), ((768, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_inception_inception_v4_tf_in1k_img_cls_timm", "pt_inception_inception_v4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 224, 17, 17), torch.bfloat16), ((224, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_inception_inception_v4_tf_in1k_img_cls_timm", "pt_inception_inception_v4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 192, 17, 17), torch.bfloat16), ((192, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_inception_inception_v4_tf_in1k_img_cls_timm", "pt_inception_inception_v4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 128, 17, 17), torch.bfloat16), ((128, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_inception_inception_v4_tf_in1k_img_cls_timm", "pt_inception_inception_v4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 192, 8, 8), torch.bfloat16), ((192, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_inception_inception_v4_tf_in1k_img_cls_timm", "pt_inception_inception_v4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 320, 17, 17), torch.bfloat16), ((320, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_inception_inception_v4_tf_in1k_img_cls_timm", "pt_inception_inception_v4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 320, 8, 8), torch.bfloat16), ((320, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_inception_inception_v4_tf_in1k_img_cls_timm", "pt_inception_inception_v4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 1024, 8, 8), torch.bfloat16), ((1024, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_inception_inception_v4_tf_in1k_img_cls_timm", "pt_inception_inception_v4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 448, 8, 8), torch.bfloat16), ((448, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_inception_inception_v4_tf_in1k_img_cls_timm", "pt_inception_inception_v4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 512, 8, 8), torch.bfloat16), ((512, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_inception_inception_v4_tf_in1k_img_cls_timm", "pt_inception_inception_v4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add20,
        [((1, 1024, 512), torch.bfloat16)],
        {
            "model_names": ["pt_mlp_mixer_mixer_l16_224_in21k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add66,
        [((1, 1024, 196), torch.bfloat16)],
        {
            "model_names": ["pt_mlp_mixer_mixer_l16_224_in21k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 196, 1024), torch.bfloat16), ((1, 196, 1024), torch.bfloat16)],
        {
            "model_names": ["pt_mlp_mixer_mixer_l16_224_in21k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add86,
        [((1, 196, 4096), torch.bfloat16)],
        {
            "model_names": ["pt_mlp_mixer_mixer_l16_224_in21k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add21,
        [((1, 196, 1024), torch.bfloat16)],
        {
            "model_names": ["pt_mlp_mixer_mixer_l16_224_in21k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add160,
        [((1, 9), torch.bfloat16)],
        {"model_names": ["pt_mobilenetv1_basic_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
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
        Add0,
        [((1, 576, 28, 28), torch.bfloat16), ((576, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 96, 28, 28), torch.bfloat16), ((1, 96, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 960, 28, 28), torch.bfloat16), ((960, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 160, 28, 28), torch.bfloat16), ((1, 160, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 21, 28, 28), torch.bfloat16), ((21, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add91,
        [((1, 16, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
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
        Add161,
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
        Add0,
        [((1, 88, 28, 28), torch.bfloat16), ((88, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 24, 28, 28), torch.bfloat16), ((1, 24, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add91,
        [((1, 96, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add91,
        [((1, 96, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add91,
        [((1, 96, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add91,
        [((1, 240, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 40, 14, 14), torch.bfloat16), ((1, 40, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add91,
        [((1, 120, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 48, 14, 14), torch.bfloat16), ((48, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add91,
        [((1, 144, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add91,
        [((1, 144, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 48, 14, 14), torch.bfloat16), ((1, 48, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add91,
        [((1, 288, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 288, 7, 7), torch.bfloat16), ((288, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add91,
        [((1, 288, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add91,
        [((1, 288, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 96, 7, 7), torch.bfloat16), ((96, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add91,
        [((1, 576, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add91,
        [((1, 576, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 96, 7, 7), torch.bfloat16), ((1, 96, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (Add162, [((1, 29, 896), torch.float32)], {"model_names": ["pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"], "pcc": 0.99}),
    (
        Add0,
        [((1, 14, 29, 64), torch.float32), ((1, 14, 29, 64), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"], "pcc": 0.99},
    ),
    (Add2, [((1, 29, 128), torch.float32)], {"model_names": ["pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"], "pcc": 0.99}),
    (
        Add0,
        [((1, 2, 29, 64), torch.float32), ((1, 2, 29, 64), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 14, 29, 29), torch.float32), ((1, 1, 29, 29), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 29, 896), torch.float32), ((1, 29, 896), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 64, 128, 128), torch.bfloat16), ((64, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_yolo_v3_default_obj_det_github",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add17,
        [((1, 16384, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 16384, 64), torch.bfloat16), ((1, 16384, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add19,
        [((1, 16384, 256), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 256, 128, 128), torch.bfloat16), ((256, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 128, 64, 64), torch.bfloat16), ((128, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_yolo_v3_default_obj_det_github",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add18,
        [((1, 4096, 128), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 128, 16, 16), torch.bfloat16), ((128, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add18,
        [((1, 256, 128), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 4096, 128), torch.bfloat16), ((1, 4096, 128), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add20,
        [((1, 4096, 512), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 512, 64, 64), torch.bfloat16), ((512, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 320, 32, 32), torch.bfloat16), ((320, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add94,
        [((1, 1024, 320), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 320, 16, 16), torch.bfloat16), ((320, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add94,
        [((1, 256, 320), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 1024, 320), torch.bfloat16), ((1, 1024, 320), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add128,
        [((1, 1024, 1280), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 1280, 32, 32), torch.bfloat16), ((1280, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 512, 16, 16), torch.bfloat16), ((512, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_yolo_v3_default_obj_det_github",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add64,
        [((1, 256, 2048), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 2048, 16, 16), torch.bfloat16), ((2048, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add81,
        [((1, 256, 768), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add81,
        [((1, 1024, 768), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add81,
        [((1, 4096, 768), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add81,
        [((1, 16384, 768), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 768, 128, 128), torch.bfloat16), ((768, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 150, 128, 128), torch.bfloat16), ((150, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 768, 1, 128), torch.float32), ((768, 1, 1), torch.float32)],
        {"model_names": ["pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Add163,
        [((1, 12, 128, 128), torch.float32)],
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
        Add164,
        [((1, 3), torch.float32)],
        {
            "model_names": [
                "pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf",
                "pt_autoencoder_linear_img_enc_github",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add38,
        [((197, 1, 2304), torch.bfloat16)],
        {"model_names": ["pt_vit_vit_b_16_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add81,
        [((197, 768), torch.bfloat16)],
        {"model_names": ["pt_vit_vit_b_16_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add165,
        [((1, 197, 1024), torch.bfloat16)],
        {"model_names": ["pt_vit_vit_l_16_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add85,
        [((197, 1, 3072), torch.bfloat16)],
        {"model_names": ["pt_vit_vit_l_16_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add88,
        [((1, 16, 197, 197), torch.bfloat16)],
        {"model_names": ["pt_vit_vit_l_16_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add21,
        [((197, 1024), torch.bfloat16)],
        {"model_names": ["pt_vit_vit_l_16_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 128, 147, 147), torch.bfloat16), ((128, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_xception_xception_img_cls_timm"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 128, 74, 74), torch.bfloat16), ((128, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_xception_xception_img_cls_timm"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 128, 74, 74), torch.bfloat16), ((1, 128, 74, 74), torch.bfloat16)],
        {"model_names": ["pt_xception_xception_img_cls_timm"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 256, 74, 74), torch.bfloat16), ((256, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_xception_xception_img_cls_timm"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 256, 37, 37), torch.bfloat16), ((256, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_xception_xception_img_cls_timm"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 256, 37, 37), torch.bfloat16), ((1, 256, 37, 37), torch.bfloat16)],
        {"model_names": ["pt_xception_xception_img_cls_timm"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 728, 37, 37), torch.bfloat16), ((728, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_xception_xception_img_cls_timm"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 256, 1024), torch.float32), ((1, 256, 1024), torch.float32)],
        {
            "model_names": ["pt_xglm_facebook_xglm_564m_clm_hf", "pt_codegen_salesforce_codegen_350m_nl_clm_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Add9,
        [((1, 256, 1024), torch.float32)],
        {
            "model_names": ["pt_xglm_facebook_xglm_564m_clm_hf", "pt_codegen_salesforce_codegen_350m_nl_clm_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 16, 256, 256), torch.float32), ((1, 1, 256, 256), torch.float32)],
        {
            "model_names": ["pt_xglm_facebook_xglm_564m_clm_hf", "pt_codegen_salesforce_codegen_350m_nl_clm_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Add13,
        [((1, 256, 4096), torch.float32)],
        {
            "model_names": ["pt_xglm_facebook_xglm_564m_clm_hf", "pt_codegen_salesforce_codegen_350m_nl_clm_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 32, 512, 512), torch.bfloat16), ((32, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_yolo_v3_default_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 64, 256, 256), torch.bfloat16), ((64, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_yolo_v3_default_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 32, 256, 256), torch.bfloat16), ((32, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_yolo_v3_default_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 64, 256, 256), torch.bfloat16), ((1, 64, 256, 256), torch.bfloat16)],
        {"model_names": ["pt_yolo_v3_default_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 128, 128, 128), torch.bfloat16), ((1, 128, 128, 128), torch.bfloat16)],
        {"model_names": ["pt_yolo_v3_default_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 512, 32, 32), torch.bfloat16), ((512, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_yolo_v3_default_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 256, 32, 32), torch.bfloat16), ((256, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_yolo_v3_default_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 512, 32, 32), torch.bfloat16), ((1, 512, 32, 32), torch.bfloat16)],
        {"model_names": ["pt_yolo_v3_default_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 1024, 16, 16), torch.bfloat16), ((1, 1024, 16, 16), torch.bfloat16)],
        {"model_names": ["pt_yolo_v3_default_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 255, 16, 16), torch.bfloat16), ((255, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_yolo_v3_default_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 255, 32, 32), torch.bfloat16), ((255, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_yolo_v3_default_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 128, 32, 32), torch.bfloat16), ((128, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_yolo_v3_default_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 255, 64, 64), torch.bfloat16), ((255, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_yolo_v3_default_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 48, 320, 320), torch.float32), ((48, 1, 1), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5m_img_cls_torchhub_640x640"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 96, 160, 160), torch.float32), ((96, 1, 1), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5m_img_cls_torchhub_640x640"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 48, 160, 160), torch.float32), ((48, 1, 1), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5m_img_cls_torchhub_640x640"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 48, 160, 160), torch.float32), ((1, 48, 160, 160), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5m_img_cls_torchhub_640x640"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 192, 80, 80), torch.float32), ((192, 1, 1), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5m_img_cls_torchhub_640x640"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 96, 80, 80), torch.float32), ((96, 1, 1), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5m_img_cls_torchhub_640x640"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 96, 80, 80), torch.float32), ((1, 96, 80, 80), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5m_img_cls_torchhub_640x640"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 384, 40, 40), torch.float32), ((384, 1, 1), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5m_img_cls_torchhub_640x640"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 192, 40, 40), torch.float32), ((192, 1, 1), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5m_img_cls_torchhub_640x640"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 192, 40, 40), torch.float32), ((1, 192, 40, 40), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5m_img_cls_torchhub_640x640"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 768, 20, 20), torch.float32), ((768, 1, 1), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5m_img_cls_torchhub_640x640"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 384, 20, 20), torch.float32), ((384, 1, 1), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5m_img_cls_torchhub_640x640"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 384, 20, 20), torch.float32), ((1, 384, 20, 20), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5m_img_cls_torchhub_640x640"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 32, 160, 160), torch.float32), ((32, 1, 1), torch.float32)],
        {
            "model_names": ["pt_yolo_v5_yolov5s_img_cls_torchhub_320x320", "onnx_yolov8_default_obj_det_github"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 64, 80, 80), torch.float32), ((64, 1, 1), torch.float32)],
        {
            "model_names": ["pt_yolo_v5_yolov5s_img_cls_torchhub_320x320", "onnx_yolov8_default_obj_det_github"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 32, 80, 80), torch.float32), ((32, 1, 1), torch.float32)],
        {
            "model_names": ["pt_yolo_v5_yolov5s_img_cls_torchhub_320x320", "onnx_yolov8_default_obj_det_github"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 32, 80, 80), torch.float32), ((1, 32, 80, 80), torch.float32)],
        {
            "model_names": ["pt_yolo_v5_yolov5s_img_cls_torchhub_320x320", "onnx_yolov8_default_obj_det_github"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 128, 40, 40), torch.float32), ((128, 1, 1), torch.float32)],
        {
            "model_names": ["pt_yolo_v5_yolov5s_img_cls_torchhub_320x320", "onnx_yolov8_default_obj_det_github"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 64, 40, 40), torch.float32), ((64, 1, 1), torch.float32)],
        {
            "model_names": ["pt_yolo_v5_yolov5s_img_cls_torchhub_320x320", "onnx_yolov8_default_obj_det_github"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 64, 40, 40), torch.float32), ((1, 64, 40, 40), torch.float32)],
        {
            "model_names": ["pt_yolo_v5_yolov5s_img_cls_torchhub_320x320", "onnx_yolov8_default_obj_det_github"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 256, 20, 20), torch.float32), ((256, 1, 1), torch.float32)],
        {
            "model_names": ["pt_yolo_v5_yolov5s_img_cls_torchhub_320x320", "onnx_yolov8_default_obj_det_github"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 128, 20, 20), torch.float32), ((128, 1, 1), torch.float32)],
        {
            "model_names": ["pt_yolo_v5_yolov5s_img_cls_torchhub_320x320", "onnx_yolov8_default_obj_det_github"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 128, 20, 20), torch.float32), ((1, 128, 20, 20), torch.float32)],
        {
            "model_names": ["pt_yolo_v5_yolov5s_img_cls_torchhub_320x320", "onnx_yolov8_default_obj_det_github"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 512, 10, 10), torch.float32), ((512, 1, 1), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5s_img_cls_torchhub_320x320"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 256, 10, 10), torch.float32), ((256, 1, 1), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5s_img_cls_torchhub_320x320"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 256, 10, 10), torch.float32), ((1, 256, 10, 10), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5s_img_cls_torchhub_320x320"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 64, 224, 320), torch.bfloat16), ((64, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_yolo_v6_yolov6l_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 128, 112, 160), torch.bfloat16), ((128, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_yolo_v6_yolov6l_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 64, 112, 160), torch.bfloat16), ((64, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_yolo_v6_yolov6l_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 64, 112, 160), torch.bfloat16), ((1, 64, 112, 160), torch.bfloat16)],
        {"model_names": ["pt_yolo_v6_yolov6l_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 256, 56, 80), torch.bfloat16), ((256, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_yolo_v6_yolov6l_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 128, 56, 80), torch.bfloat16), ((128, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_yolo_v6_yolov6l_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 128, 56, 80), torch.bfloat16), ((1, 128, 56, 80), torch.bfloat16)],
        {"model_names": ["pt_yolo_v6_yolov6l_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 512, 28, 40), torch.bfloat16), ((512, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_yolo_v6_yolov6l_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 256, 28, 40), torch.bfloat16), ((256, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_yolo_v6_yolov6l_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 256, 28, 40), torch.bfloat16), ((1, 256, 28, 40), torch.bfloat16)],
        {"model_names": ["pt_yolo_v6_yolov6l_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 1024, 14, 20), torch.bfloat16), ((1024, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_yolo_v6_yolov6l_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 512, 14, 20), torch.bfloat16), ((512, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_yolo_v6_yolov6l_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 512, 14, 20), torch.bfloat16), ((1, 512, 14, 20), torch.bfloat16)],
        {"model_names": ["pt_yolo_v6_yolov6l_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 256, 14, 20), torch.bfloat16), ((256, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_yolo_v6_yolov6l_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 128, 28, 40), torch.bfloat16), ((128, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_yolo_v6_yolov6l_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 128, 28, 40), torch.bfloat16), ((1, 128, 28, 40), torch.bfloat16)],
        {"model_names": ["pt_yolo_v6_yolov6l_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 64, 56, 80), torch.bfloat16), ((64, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_yolo_v6_yolov6l_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 64, 56, 80), torch.bfloat16), ((1, 64, 56, 80), torch.bfloat16)],
        {"model_names": ["pt_yolo_v6_yolov6l_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 68, 56, 80), torch.bfloat16), ((68, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_yolo_v6_yolov6l_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 68, 28, 40), torch.bfloat16), ((68, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_yolo_v6_yolov6l_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 256, 14, 20), torch.bfloat16), ((1, 256, 14, 20), torch.bfloat16)],
        {"model_names": ["pt_yolo_v6_yolov6l_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 68, 14, 20), torch.bfloat16), ((68, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_yolo_v6_yolov6l_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add166,
        [((1, 5880, 2), torch.bfloat16)],
        {"model_names": ["pt_yolo_v6_yolov6l_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 5880, 2), torch.bfloat16), ((1, 5880, 2), torch.bfloat16)],
        {"model_names": ["pt_yolo_v6_yolov6l_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 80, 56, 80), torch.bfloat16), ((80, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_yolo_v6_yolov6l_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 80, 28, 40), torch.bfloat16), ((80, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_yolo_v6_yolov6l_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 80, 14, 20), torch.bfloat16), ((80, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_yolo_v6_yolov6l_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 32, 160, 160), torch.bfloat16), ((32, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_yolov9_default_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 32, 160, 160), torch.bfloat16), ((1, 32, 160, 160), torch.bfloat16)],
        {"model_names": ["pt_yolov9_default_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 256, 160, 160), torch.bfloat16), ((256, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_yolov9_default_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 512, 80, 80), torch.bfloat16), ((512, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_yolov9_default_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 128, 20, 20), torch.bfloat16), ((128, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_yolov9_default_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 128, 20, 20), torch.bfloat16), ((1, 128, 20, 20), torch.bfloat16)],
        {"model_names": ["pt_yolov9_default_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 32, 640, 640), torch.bfloat16), ((32, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_yolox_yolox_darknet_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 32, 320, 320), torch.bfloat16), ((32, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_yolox_yolox_darknet_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 64, 320, 320), torch.bfloat16), ((1, 64, 320, 320), torch.bfloat16)],
        {"model_names": ["pt_yolox_yolox_darknet_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Add0,
        [((1, 128, 160, 160), torch.bfloat16), ((1, 128, 160, 160), torch.bfloat16)],
        {"model_names": ["pt_yolox_yolox_darknet_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
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
        {"model_names": ["tf_resnet_resnet50_img_cls_keras"], "pcc": 0.99},
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
        {"model_names": ["tf_resnet_resnet50_img_cls_keras"], "pcc": 0.99},
    ),
    (
        Add1,
        [((1, 1, 1, 9), torch.float32)],
        {
            "model_names": ["pd_bert_chinese_roberta_base_mlm_padlenlp", "pd_bert_bert_base_uncased_mlm_padlenlp"],
            "pcc": 0.99,
        },
    ),
    (
        Add5,
        [((1, 9, 21128), torch.float32)],
        {"model_names": ["pd_bert_chinese_roberta_base_mlm_padlenlp"], "pcc": 0.99},
    ),
    (
        Add167,
        [((1, 128, 28996), torch.float32)],
        {"model_names": ["pt_distilbert_distilbert_base_cased_mlm_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 16, 112, 112), torch.float32), ((16, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add1,
        [((1, 16, 112, 112), torch.float32)],
        {"model_names": ["onnx_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 16, 56, 56), torch.float32), ((16, 1, 1), torch.float32)],
        {"model_names": ["onnx_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 8, 1, 1), torch.float32), ((8, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 16, 1, 1), torch.float32), ((16, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add1,
        [((1, 16, 1, 1), torch.float32)],
        {"model_names": ["onnx_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 72, 56, 56), torch.float32), ((72, 1, 1), torch.float32)],
        {"model_names": ["onnx_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 72, 28, 28), torch.float32), ((72, 1, 1), torch.float32)],
        {"model_names": ["onnx_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 24, 28, 28), torch.float32), ((24, 1, 1), torch.float32)],
        {"model_names": ["onnx_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 88, 28, 28), torch.float32), ((88, 1, 1), torch.float32)],
        {"model_names": ["onnx_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 24, 28, 28), torch.float32), ((1, 24, 28, 28), torch.float32)],
        {"model_names": ["onnx_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 96, 28, 28), torch.float32), ((96, 1, 1), torch.float32)],
        {"model_names": ["onnx_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Add1,
        [((1, 96, 28, 28), torch.float32)],
        {"model_names": ["onnx_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 96, 14, 14), torch.float32), ((96, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "onnx_vovnet_vovnet27s_obj_det_osmr",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add1,
        [((1, 96, 14, 14), torch.float32)],
        {"model_names": ["onnx_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 24, 1, 1), torch.float32), ((24, 1, 1), torch.float32)],
        {"model_names": ["onnx_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 96, 1, 1), torch.float32), ((96, 1, 1), torch.float32)],
        {"model_names": ["onnx_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Add1,
        [((1, 96, 1, 1), torch.float32)],
        {"model_names": ["onnx_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 40, 14, 14), torch.float32), ((40, 1, 1), torch.float32)],
        {"model_names": ["onnx_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 240, 14, 14), torch.float32), ((240, 1, 1), torch.float32)],
        {"model_names": ["onnx_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Add1,
        [((1, 240, 14, 14), torch.float32)],
        {"model_names": ["onnx_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 64, 1, 1), torch.float32), ((64, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 240, 1, 1), torch.float32), ((240, 1, 1), torch.float32)],
        {"model_names": ["onnx_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Add1,
        [((1, 240, 1, 1), torch.float32)],
        {"model_names": ["onnx_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 40, 14, 14), torch.float32), ((1, 40, 14, 14), torch.float32)],
        {"model_names": ["onnx_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 120, 14, 14), torch.float32), ((120, 1, 1), torch.float32)],
        {"model_names": ["onnx_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Add1,
        [((1, 120, 14, 14), torch.float32)],
        {"model_names": ["onnx_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 32, 1, 1), torch.float32), ((32, 1, 1), torch.float32)],
        {"model_names": ["onnx_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 120, 1, 1), torch.float32), ((120, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add1,
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
        Add0,
        [((1, 48, 14, 14), torch.float32), ((48, 1, 1), torch.float32)],
        {"model_names": ["onnx_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 144, 14, 14), torch.float32), ((144, 1, 1), torch.float32)],
        {"model_names": ["onnx_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Add1,
        [((1, 144, 14, 14), torch.float32)],
        {"model_names": ["onnx_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 40, 1, 1), torch.float32), ((40, 1, 1), torch.float32)],
        {"model_names": ["onnx_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 144, 1, 1), torch.float32), ((144, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add1,
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
        Add0,
        [((1, 48, 14, 14), torch.float32), ((1, 48, 14, 14), torch.float32)],
        {"model_names": ["onnx_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 288, 14, 14), torch.float32), ((288, 1, 1), torch.float32)],
        {"model_names": ["onnx_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Add1,
        [((1, 288, 14, 14), torch.float32)],
        {"model_names": ["onnx_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 288, 7, 7), torch.float32), ((288, 1, 1), torch.float32)],
        {"model_names": ["onnx_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Add1,
        [((1, 288, 7, 7), torch.float32)],
        {"model_names": ["onnx_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 72, 1, 1), torch.float32), ((72, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 288, 1, 1), torch.float32), ((288, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add1,
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
        Add0,
        [((1, 96, 7, 7), torch.float32), ((96, 1, 1), torch.float32)],
        {"model_names": ["onnx_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 576, 7, 7), torch.float32), ((576, 1, 1), torch.float32)],
        {"model_names": ["onnx_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Add1,
        [((1, 576, 7, 7), torch.float32)],
        {"model_names": ["onnx_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 576, 1, 1), torch.float32), ((576, 1, 1), torch.float32)],
        {"model_names": ["onnx_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Add1,
        [((1, 576, 1, 1), torch.float32)],
        {"model_names": ["onnx_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 96, 7, 7), torch.float32), ((1, 96, 7, 7), torch.float32)],
        {"model_names": ["onnx_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Add9,
        [((1, 1024), torch.float32)],
        {"model_names": ["onnx_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Add1,
        [((1, 1024), torch.float32)],
        {"model_names": ["onnx_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 80, 28, 28), torch.float32), ((80, 1, 1), torch.float32)],
        {"model_names": ["onnx_vovnet_vovnet27s_obj_det_osmr"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 384, 14, 14), torch.float32), ((384, 1, 1), torch.float32)],
        {"model_names": ["onnx_vovnet_vovnet27s_obj_det_osmr"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 112, 7, 7), torch.float32), ((112, 1, 1), torch.float32)],
        {"model_names": ["onnx_vovnet_vovnet27s_obj_det_osmr"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 160, 28, 28), torch.float32), ((160, 1, 1), torch.float32)],
        {"model_names": ["onnx_vovnet_vovnet_v1_57_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 192, 14, 14), torch.float32), ((192, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_vovnet_vovnet_v1_57_obj_det_torchhub",
                "onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm",
                "onnx_deit_facebook_deit_tiny_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 768, 14, 14), torch.float32), ((768, 1, 1), torch.float32)],
        {"model_names": ["onnx_vovnet_vovnet_v1_57_obj_det_torchhub"], "pcc": 0.99},
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
        [((1, 1024, 7, 7), torch.float32), ((1024, 1, 1), torch.float32)],
        {"model_names": ["onnx_vovnet_vovnet_v1_57_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1024, 7, 7), torch.float32), ((1, 1024, 7, 7), torch.float32)],
        {"model_names": ["onnx_vovnet_vovnet_v1_57_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 32, 112, 112), torch.float32), ((32, 1, 1), torch.float32)],
        {"model_names": ["onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 96, 112, 112), torch.float32), ((96, 1, 1), torch.float32)],
        {"model_names": ["onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 96, 56, 56), torch.float32), ((96, 1, 1), torch.float32)],
        {"model_names": ["onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 24, 56, 56), torch.float32), ((24, 1, 1), torch.float32)],
        {"model_names": ["onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 144, 56, 56), torch.float32), ((144, 1, 1), torch.float32)],
        {"model_names": ["onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 24, 56, 56), torch.float32), ((1, 24, 56, 56), torch.float32)],
        {"model_names": ["onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 144, 28, 28), torch.float32), ((144, 1, 1), torch.float32)],
        {"model_names": ["onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 32, 28, 28), torch.float32), ((32, 1, 1), torch.float32)],
        {"model_names": ["onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 192, 28, 28), torch.float32), ((192, 1, 1), torch.float32)],
        {"model_names": ["onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 32, 28, 28), torch.float32), ((1, 32, 28, 28), torch.float32)],
        {"model_names": ["onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm"], "pcc": 0.99},
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
        Add0,
        [((1, 1280, 7, 7), torch.float32), ((1280, 1, 1), torch.float32)],
        {"model_names": ["onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add168,
        [((1, 9, 30522), torch.float32)],
        {"model_names": ["pd_bert_bert_base_uncased_mlm_padlenlp"], "pcc": 0.99},
    ),
    (
        Add169,
        [((1, 197, 192), torch.float32)],
        {"model_names": ["onnx_deit_facebook_deit_tiny_patch16_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add170,
        [((1, 197, 192), torch.float32)],
        {"model_names": ["onnx_deit_facebook_deit_tiny_patch16_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 197, 192), torch.float32), ((1, 197, 192), torch.float32)],
        {"model_names": ["onnx_deit_facebook_deit_tiny_patch16_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add103,
        [((1, 197, 768), torch.float32)],
        {"model_names": ["onnx_deit_facebook_deit_tiny_patch16_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add1,
        [((1, 197, 768), torch.float32)],
        {"model_names": ["onnx_deit_facebook_deit_tiny_patch16_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 16, 320, 320), torch.float32), ((16, 1, 1), torch.float32)],
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
    (Add171, [((1, 2, 8400), torch.float32)], {"model_names": ["onnx_yolov8_default_obj_det_github"], "pcc": 0.99}),
    (
        Add0,
        [((1, 2, 8400), torch.float32), ((1, 2, 8400), torch.float32)],
        {"model_names": ["onnx_yolov8_default_obj_det_github"], "pcc": 0.99},
    ),
    (Add13, [((1, 4096), torch.float32)], {"model_names": ["pd_alexnet_base_img_cls_paddlemodels"], "pcc": 0.99}),
    (
        Add0,
        [((8,), torch.float32), ((1,), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add172,
        [((8,), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 8, 16, 50), torch.float32), ((8, 1, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add1,
        [((1, 8, 16, 50), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 2, 1, 1), torch.float32), ((2, 1, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add1,
        [((1, 8, 1, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((40,), torch.float32), ((1,), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add173,
        [((40,), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 40, 16, 50), torch.float32), ((40, 1, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 40, 8, 50), torch.float32), ((40, 1, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((16,), torch.float32), ((1,), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add174,
        [((16,), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 16, 8, 50), torch.float32), ((16, 1, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((48,), torch.float32), ((1,), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add175,
        [((48,), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 48, 8, 50), torch.float32), ((48, 1, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 16, 8, 50), torch.float32), ((1, 16, 8, 50), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add1,
        [((1, 48, 8, 50), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 48, 4, 50), torch.float32), ((48, 1, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add1,
        [((1, 48, 4, 50), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 12, 1, 1), torch.float32), ((12, 1, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 48, 1, 1), torch.float32), ((48, 1, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add1,
        [((1, 48, 1, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((24,), torch.float32), ((1,), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add176,
        [((24,), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 24, 4, 50), torch.float32), ((24, 1, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((120,), torch.float32), ((1,), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add177,
        [((120,), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 120, 4, 50), torch.float32), ((120, 1, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add1,
        [((1, 120, 4, 50), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 30, 1, 1), torch.float32), ((30, 1, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 24, 4, 50), torch.float32), ((1, 24, 4, 50), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 64, 4, 50), torch.float32), ((64, 1, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add1,
        [((1, 64, 4, 50), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add1,
        [((1, 64, 1, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((72,), torch.float32), ((1,), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add178,
        [((72,), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 72, 4, 50), torch.float32), ((72, 1, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add1,
        [((1, 72, 4, 50), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 18, 1, 1), torch.float32), ((18, 1, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add1,
        [((1, 72, 1, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((144,), torch.float32), ((1,), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add179,
        [((144,), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 144, 4, 50), torch.float32), ((144, 1, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add1,
        [((1, 144, 4, 50), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 144, 2, 50), torch.float32), ((144, 1, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add1,
        [((1, 144, 2, 50), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 36, 1, 1), torch.float32), ((36, 1, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 48, 2, 50), torch.float32), ((48, 1, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((288,), torch.float32), ((1,), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add143,
        [((288,), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 288, 2, 50), torch.float32), ((288, 1, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add1,
        [((1, 288, 2, 50), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 48, 2, 50), torch.float32), ((1, 48, 2, 50), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 192), torch.float32), ((1, 192), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add148,
        [((1, 192), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 48), torch.float32), ((1, 48), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add180,
        [((1, 25, 6625), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (Add2, [((1, 128), torch.float32)], {"model_names": ["pt_autoencoder_linear_img_enc_github"], "pcc": 0.99}),
    (Add6, [((1, 64), torch.float32)], {"model_names": ["pt_autoencoder_linear_img_enc_github"], "pcc": 0.99}),
    (Add181, [((1, 12), torch.float32)], {"model_names": ["pt_autoencoder_linear_img_enc_github"], "pcc": 0.99}),
    (Add182, [((1, 784), torch.float32)], {"model_names": ["pt_autoencoder_linear_img_enc_github"], "pcc": 0.99}),
    (
        Add183,
        [((1, 16, 128, 128), torch.float32)],
        {"model_names": ["pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf"], "pcc": 0.99},
    ),
    (
        Add184,
        [((1, 128, 9), torch.float32)],
        {"model_names": ["pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 256, 16, 32), torch.float32), ((1, 256, 16, 32), torch.float32)],
        {"model_names": ["pt_codegen_salesforce_codegen_350m_nl_clm_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1088,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add185,
        [((1088,), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 1088, 14, 14), torch.bfloat16), ((1088, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1120,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add186,
        [((1120,), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 1120, 14, 14), torch.bfloat16), ((1120, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1152,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add83,
        [((1152,), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 1152, 14, 14), torch.bfloat16), ((1152, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1184,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add187,
        [((1184,), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 1184, 14, 14), torch.bfloat16), ((1184, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1216,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add188,
        [((1216,), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 1216, 14, 14), torch.bfloat16), ((1216, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
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
        Add189,
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
        Add0,
        [((1, 1248, 14, 14), torch.bfloat16), ((1248, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 1280, 14, 14), torch.bfloat16), ((1280, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1312,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add190,
        [((1312,), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 1312, 14, 14), torch.bfloat16), ((1312, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1376,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add191,
        [((1376,), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 1376, 14, 14), torch.bfloat16), ((1376, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
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
        Add192,
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
        Add0,
        [((1, 1408, 14, 14), torch.bfloat16), ((1408, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1440,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add193,
        [((1440,), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 1440, 14, 14), torch.bfloat16), ((1440, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1472,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add194,
        [((1472,), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 1472, 14, 14), torch.bfloat16), ((1472, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1504,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add195,
        [((1504,), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 1504, 14, 14), torch.bfloat16), ((1504, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 1536, 14, 14), torch.bfloat16), ((1536, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1568,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add196,
        [((1568,), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 1568, 14, 14), torch.bfloat16), ((1568, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1600,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add197,
        [((1600,), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 1600, 14, 14), torch.bfloat16), ((1600, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 1632, 14, 14), torch.bfloat16), ((1632, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1664,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add198,
        [((1664,), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 1664, 14, 14), torch.bfloat16), ((1664, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1696,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add199,
        [((1696,), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 1696, 14, 14), torch.bfloat16), ((1696, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1728,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add200,
        [((1728,), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 1728, 14, 14), torch.bfloat16), ((1728, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1760,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add201,
        [((1760,), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 1760, 14, 14), torch.bfloat16), ((1760, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 1792, 14, 14), torch.bfloat16), ((1792, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 1088, 7, 7), torch.bfloat16), ((1088, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 1120, 7, 7), torch.bfloat16), ((1120, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 1152, 7, 7), torch.bfloat16), ((1152, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 1184, 7, 7), torch.bfloat16), ((1184, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 1216, 7, 7), torch.bfloat16), ((1216, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
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
        Add0,
        [((1, 1312, 7, 7), torch.bfloat16), ((1312, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 1376, 7, 7), torch.bfloat16), ((1376, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
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
        Add0,
        [((1, 1440, 7, 7), torch.bfloat16), ((1440, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 1472, 7, 7), torch.bfloat16), ((1472, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 1504, 7, 7), torch.bfloat16), ((1504, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 1568, 7, 7), torch.bfloat16), ((1568, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 1600, 7, 7), torch.bfloat16), ((1600, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 1664, 7, 7), torch.bfloat16), ((1664, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 1696, 7, 7), torch.bfloat16), ((1696, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 1728, 7, 7), torch.bfloat16), ((1728, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 1760, 7, 7), torch.bfloat16), ((1760, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1856,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add202,
        [((1856,), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 1856, 7, 7), torch.bfloat16), ((1856, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1888,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add203,
        [((1888,), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 1888, 7, 7), torch.bfloat16), ((1888, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1920,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add204,
        [((1920,), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 1920, 7, 7), torch.bfloat16), ((1920, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 4, 1, 1), torch.bfloat16), ((4, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b2_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 88, 14, 14), torch.bfloat16), ((88, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b2_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 528, 14, 14), torch.bfloat16), ((528, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b2_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 22, 1, 1), torch.bfloat16), ((22, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b2_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 88, 14, 14), torch.bfloat16), ((1, 88, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b2_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 720, 14, 14), torch.bfloat16), ((720, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b2_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 30, 1, 1), torch.bfloat16), ((30, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b2_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 720, 1, 1), torch.bfloat16), ((720, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b2_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 720, 7, 7), torch.bfloat16), ((720, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b2_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((208,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b2_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add205,
        [((208,), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b2_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 208, 7, 7), torch.bfloat16), ((208, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b2_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 52, 1, 1), torch.bfloat16), ((52, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b2_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 1248, 1, 1), torch.bfloat16), ((1248, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b2_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 208, 7, 7), torch.bfloat16), ((1, 208, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b2_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 352, 7, 7), torch.bfloat16), ((352, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b2_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((2112,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b2_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add206,
        [((2112,), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b2_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 2112, 7, 7), torch.bfloat16), ((2112, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b2_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 88, 1, 1), torch.bfloat16), ((88, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b2_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 2112, 1, 1), torch.bfloat16), ((2112, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b2_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 352, 7, 7), torch.bfloat16), ((1, 352, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b2_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 48, 224, 224), torch.bfloat16), ((48, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 24, 224, 224), torch.bfloat16), ((24, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 24, 224, 224), torch.bfloat16), ((1, 24, 224, 224), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 144, 224, 224), torch.bfloat16), ((144, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 240, 112, 112), torch.bfloat16), ((240, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 40, 112, 112), torch.bfloat16), ((1, 40, 112, 112), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 384, 56, 56), torch.bfloat16), ((384, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 768, 28, 28), torch.bfloat16), ((768, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 128, 28, 28), torch.bfloat16), ((1, 128, 28, 28), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 176, 28, 28), torch.bfloat16), ((176, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 1056, 28, 28), torch.bfloat16), ((1056, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 176, 28, 28), torch.bfloat16), ((1, 176, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 304, 14, 14), torch.bfloat16), ((304, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 1824, 14, 14), torch.bfloat16), ((1824, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 304, 14, 14), torch.bfloat16), ((1, 304, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 3072, 14, 14), torch.bfloat16), ((3072, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 256, 768), torch.float32), ((1, 256, 768), torch.float32)],
        {"model_names": ["pt_gpt_gpt2_text_gen_hf", "pt_opt_facebook_opt_125m_clm_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((256, 768), torch.float32), ((768,), torch.float32)],
        {"model_names": ["pt_gpt_gpt2_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Add207,
        [((1, 12, 256, 256), torch.float32)],
        {"model_names": ["pt_gpt_gpt2_text_gen_hf", "pt_opt_facebook_opt_125m_clm_hf"], "pcc": 0.99},
    ),
    (
        Add22,
        [((256, 768), torch.float32)],
        {"model_names": ["pt_gpt_gpt2_text_gen_hf", "pt_opt_facebook_opt_125m_clm_hf"], "pcc": 0.99},
    ),
    (
        Add23,
        [((256, 3072), torch.float32)],
        {"model_names": ["pt_gpt_gpt2_text_gen_hf", "pt_opt_facebook_opt_125m_clm_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 256, 14, 14), torch.bfloat16), ((1, 256, 14, 14), torch.bfloat16)],
        {"model_names": ["pt_hrnet_hrnetv2_w64_pose_estimation_osmr"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (Add208, [((1, 256), torch.int64)], {"model_names": ["pt_opt_facebook_opt_125m_clm_hf"], "pcc": 0.99}),
    (Add22, [((1, 256, 768), torch.float32)], {"model_names": ["pt_opt_facebook_opt_125m_clm_hf"], "pcc": 0.99}),
    (
        Add0,
        [((256, 768), torch.float32), ((256, 768), torch.float32)],
        {"model_names": ["pt_opt_facebook_opt_125m_clm_hf"], "pcc": 0.99},
    ),
    (Add208, [((1, 32), torch.int64)], {"model_names": ["pt_opt_facebook_opt_125m_seq_cls_hf"], "pcc": 0.99}),
    (
        Add0,
        [((1, 32, 768), torch.float32), ((1, 32, 768), torch.float32)],
        {"model_names": ["pt_opt_facebook_opt_125m_seq_cls_hf"], "pcc": 0.99},
    ),
    (Add22, [((1, 32, 768), torch.float32)], {"model_names": ["pt_opt_facebook_opt_125m_seq_cls_hf"], "pcc": 0.99}),
    (Add209, [((1, 12, 32, 32), torch.float32)], {"model_names": ["pt_opt_facebook_opt_125m_seq_cls_hf"], "pcc": 0.99}),
    (Add23, [((32, 3072), torch.float32)], {"model_names": ["pt_opt_facebook_opt_125m_seq_cls_hf"], "pcc": 0.99}),
    (Add22, [((32, 768), torch.float32)], {"model_names": ["pt_opt_facebook_opt_125m_seq_cls_hf"], "pcc": 0.99}),
    (
        Add0,
        [((32, 768), torch.float32), ((32, 768), torch.float32)],
        {"model_names": ["pt_opt_facebook_opt_125m_seq_cls_hf"], "pcc": 0.99},
    ),
    (Add10, [((1, 12, 2048), torch.float32)], {"model_names": ["pt_phi1_microsoft_phi_1_token_cls_hf"], "pcc": 0.99}),
    (
        Add0,
        [((1, 32, 12, 32), torch.float32), ((1, 32, 12, 32), torch.float32)],
        {"model_names": ["pt_phi1_microsoft_phi_1_token_cls_hf"], "pcc": 0.99},
    ),
    (
        Add210,
        [((1, 32, 12, 12), torch.float32)],
        {"model_names": ["pt_phi1_microsoft_phi_1_token_cls_hf"], "pcc": 0.99},
    ),
    (Add69, [((1, 12, 8192), torch.float32)], {"model_names": ["pt_phi1_microsoft_phi_1_token_cls_hf"], "pcc": 0.99}),
    (
        Add0,
        [((1, 12, 2048), torch.float32), ((1, 12, 2048), torch.float32)],
        {"model_names": ["pt_phi1_microsoft_phi_1_token_cls_hf"], "pcc": 0.99},
    ),
    (Add14, [((1, 12, 2), torch.float32)], {"model_names": ["pt_phi1_microsoft_phi_1_token_cls_hf"], "pcc": 0.99}),
    (Add153, [((1, 8, 513, 513), torch.float32)], {"model_names": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99}),
    (
        Add0,
        [((1, 8, 513, 513), torch.float32), ((1, 8, 513, 513), torch.float32)],
        {"model_names": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 513, 512), torch.float32), ((1, 513, 512), torch.float32)],
        {"model_names": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99},
    ),
    (Add154, [((1, 8, 61, 61), torch.float32)], {"model_names": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99}),
    (
        Add0,
        [((1, 8, 61, 61), torch.float32), ((1, 8, 61, 61), torch.float32)],
        {"model_names": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 61, 512), torch.float32), ((1, 61, 512), torch.float32)],
        {"model_names": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99},
    ),
    (Add211, [((1, 8, 513, 61), torch.float32)], {"model_names": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99}),
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
