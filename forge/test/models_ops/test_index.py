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


class Index0(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=0, stop=128, stride=1)
        return index_output_1


class Index1(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=1, stride=1)
        return index_output_1


class Index2(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=1, stop=2, stride=1)
        return index_output_1


class Index3(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=0, stop=1, stride=1)
        return index_output_1


class Index4(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=1, stop=2, stride=1)
        return index_output_1


class Index5(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=30, stride=1)
        return index_output_1


class Index6(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=30, stop=90, stride=1)
        return index_output_1


class Index7(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=90, stop=210, stride=1)
        return index_output_1


class Index8(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=120, stride=1)
        return index_output_1


class Index9(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=120, stop=180, stride=1)
        return index_output_1


class Index10(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=180, stop=210, stride=1)
        return index_output_1


class Index11(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=60, stride=1)
        return index_output_1


class Index12(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=60, stop=90, stride=1)
        return index_output_1


class Index13(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=90, stop=120, stride=1)
        return index_output_1


class Index14(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=30, stop=60, stride=1)
        return index_output_1


class Index15(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=60, stop=120, stride=1)
        return index_output_1


class Index16(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=32, stop=64, stride=1)
        return index_output_1


class Index17(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=0, stop=32, stride=1)
        return index_output_1


class Index18(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=3, stop=4, stride=1)
        return index_output_1


class Index19(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=16, stop=32, stride=1)
        return index_output_1


class Index20(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=0, stop=16, stride=1)
        return index_output_1


class Index21(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=4, stop=5, stride=1)
        return index_output_1


class Index22(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-5, start=0, stop=1, stride=1)
        return index_output_1


class Index23(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-5, start=1, stop=2, stride=1)
        return index_output_1


class Index24(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-5, start=2, stop=3, stride=1)
        return index_output_1


class Index25(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=3, stop=56, stride=1)
        return index_output_1


class Index26(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=3, stride=1)
        return index_output_1


class Index27(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=3, stop=56, stride=1)
        return index_output_1


class Index28(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=3, stride=1)
        return index_output_1


class Index29(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=53, stop=56, stride=1)
        return index_output_1


class Index30(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=53, stride=1)
        return index_output_1


class Index31(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=53, stop=56, stride=1)
        return index_output_1


class Index32(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=53, stride=1)
        return index_output_1


class Index33(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=56, stride=2)
        return index_output_1


class Index34(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=1, stop=56, stride=2)
        return index_output_1


class Index35(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=56, stride=2)
        return index_output_1


class Index36(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=1, stop=56, stride=2)
        return index_output_1


class Index37(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=3, stop=28, stride=1)
        return index_output_1


class Index38(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=3, stop=28, stride=1)
        return index_output_1


class Index39(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=25, stop=28, stride=1)
        return index_output_1


class Index40(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=25, stride=1)
        return index_output_1


class Index41(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=25, stop=28, stride=1)
        return index_output_1


class Index42(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=25, stride=1)
        return index_output_1


class Index43(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=28, stride=2)
        return index_output_1


class Index44(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=1, stop=28, stride=2)
        return index_output_1


class Index45(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=28, stride=2)
        return index_output_1


class Index46(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=1, stop=28, stride=2)
        return index_output_1


class Index47(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=3, stop=14, stride=1)
        return index_output_1


class Index48(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=3, stop=14, stride=1)
        return index_output_1


class Index49(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=11, stop=14, stride=1)
        return index_output_1


class Index50(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=11, stride=1)
        return index_output_1


class Index51(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=11, stop=14, stride=1)
        return index_output_1


class Index52(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=11, stride=1)
        return index_output_1


class Index53(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=14, stride=2)
        return index_output_1


class Index54(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=1, stop=14, stride=2)
        return index_output_1


class Index55(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=14, stride=2)
        return index_output_1


class Index56(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=1, stop=14, stride=2)
        return index_output_1


class Index57(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=1, stop=197, stride=1)
        return index_output_1


class Index58(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=0, stop=1, stride=1)
        return index_output_1


class Index59(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=1, stop=2, stride=1)
        return index_output_1


class Index60(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=2, stop=3, stride=1)
        return index_output_1


class Index61(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=80, stride=1)
        return index_output_1


class Index62(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=80, stop=160, stride=1)
        return index_output_1


class Index63(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=160, stride=1)
        return index_output_1


class Index64(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=160, stop=320, stride=1)
        return index_output_1


class Index65(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=320, stride=1)
        return index_output_1


class Index66(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=320, stop=640, stride=1)
        return index_output_1


class Index67(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=64, stride=1)
        return index_output_1


class Index68(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=64, stop=144, stride=1)
        return index_output_1


class Index69(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=2, stride=1)
        return index_output_1


class Index70(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=2, stop=4, stride=1)
        return index_output_1


class Index71(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=640, stride=2)
        return index_output_1


class Index72(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=1, stop=640, stride=2)
        return index_output_1


class Index73(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=0, stop=640, stride=2)
        return index_output_1


class Index74(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=1, stop=640, stride=2)
        return index_output_1


class Index75(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=729, stride=1)
        return index_output_1


class Index76(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=729, stop=732, stride=1)
        return index_output_1


class Index77(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=16, stride=1)
        return index_output_1


class Index78(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=16, stop=32, stride=1)
        return index_output_1


class Index79(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=32, stop=64, stride=1)
        return index_output_1


class Index80(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=64, stride=1)
        return index_output_1


class Index81(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=64, stop=96, stride=1)
        return index_output_1


class Index82(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=96, stop=112, stride=1)
        return index_output_1


class Index83(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=18, stride=1)
        return index_output_1


class Index84(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=18, stop=54, stride=1)
        return index_output_1


class Index85(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=54, stop=126, stride=1)
        return index_output_1


class Index86(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=72, stride=1)
        return index_output_1


class Index87(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=72, stop=108, stride=1)
        return index_output_1


class Index88(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=108, stop=126, stride=1)
        return index_output_1


class Index89(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=36, stride=1)
        return index_output_1


class Index90(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=36, stop=54, stride=1)
        return index_output_1


class Index91(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=54, stop=72, stride=1)
        return index_output_1


class Index92(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=18, stop=36, stride=1)
        return index_output_1


class Index93(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=36, stop=72, stride=1)
        return index_output_1


class Index94(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=32, stop=80, stride=1)
        return index_output_1


class Index95(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=10, stop=11, stride=1)
        return index_output_1


class Index96(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=64, stop=128, stride=1)
        return index_output_1


class Index97(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=0, stop=64, stride=1)
        return index_output_1


class Index98(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=4, stop=64, stride=1)
        return index_output_1


class Index99(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=4, stride=1)
        return index_output_1


class Index100(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=4, stop=64, stride=1)
        return index_output_1


class Index101(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=4, stride=1)
        return index_output_1


class Index102(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=60, stop=64, stride=1)
        return index_output_1


class Index103(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=60, stop=64, stride=1)
        return index_output_1


class Index104(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=60, stride=1)
        return index_output_1


class Index105(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=64, stride=2)
        return index_output_1


class Index106(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=1, stop=64, stride=2)
        return index_output_1


class Index107(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=64, stride=2)
        return index_output_1


class Index108(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=1, stop=64, stride=2)
        return index_output_1


class Index109(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=4, stop=32, stride=1)
        return index_output_1


class Index110(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=4, stop=32, stride=1)
        return index_output_1


class Index111(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=28, stop=32, stride=1)
        return index_output_1


class Index112(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=28, stride=1)
        return index_output_1


class Index113(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=28, stop=32, stride=1)
        return index_output_1


class Index114(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=28, stride=1)
        return index_output_1


class Index115(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=32, stride=2)
        return index_output_1


class Index116(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=1, stop=32, stride=2)
        return index_output_1


class Index117(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=32, stride=2)
        return index_output_1


class Index118(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=1, stop=32, stride=2)
        return index_output_1


class Index119(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=4, stop=16, stride=1)
        return index_output_1


class Index120(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=4, stop=16, stride=1)
        return index_output_1


class Index121(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=12, stop=16, stride=1)
        return index_output_1


class Index122(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=12, stride=1)
        return index_output_1


class Index123(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=12, stop=16, stride=1)
        return index_output_1


class Index124(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=12, stride=1)
        return index_output_1


class Index125(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=16, stride=2)
        return index_output_1


class Index126(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=1, stop=16, stride=2)
        return index_output_1


class Index127(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=16, stride=2)
        return index_output_1


class Index128(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=1, stop=16, stride=2)
        return index_output_1


class Index129(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=96, stride=1)
        return index_output_1


class Index130(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=96, stop=160, stride=1)
        return index_output_1


class Index131(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=160, stop=224, stride=1)
        return index_output_1


class Index132(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=384, stride=1)
        return index_output_1


class Index133(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=384, stop=576, stride=1)
        return index_output_1


class Index134(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=576, stop=768, stride=1)
        return index_output_1


class Index135(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=256, stride=1)
        return index_output_1


class Index136(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=256, stop=640, stride=1)
        return index_output_1


class Index137(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=640, stop=1024, stride=1)
        return index_output_1


class Index138(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=2, stop=258, stride=1)
        return index_output_1


class Index139(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=0, stop=2, stride=1)
        return index_output_1


class Index140(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=2, stop=4, stride=1)
        return index_output_1


class Index141(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=64, stop=128, stride=1)
        return index_output_1


class Index142(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=128, stride=1)
        return index_output_1


class Index143(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=128, stop=256, stride=1)
        return index_output_1


class Index144(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=256, stop=512, stride=1)
        return index_output_1


class Index145(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=32, stride=1)
        return index_output_1


class Index146(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=1, stride=1)
        return index_output_1


class Index147(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=1, stop=2, stride=1)
        return index_output_1


class Index148(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=2, stop=3, stride=1)
        return index_output_1


class Index149(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=3, stop=4, stride=1)
        return index_output_1


class Index150(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=4, stop=5, stride=1)
        return index_output_1


class Index151(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=5, stop=6, stride=1)
        return index_output_1


class Index152(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=6, stop=7, stride=1)
        return index_output_1


class Index153(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=7, stop=8, stride=1)
        return index_output_1


class Index154(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=8, stop=9, stride=1)
        return index_output_1


class Index155(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=9, stop=10, stride=1)
        return index_output_1


class Index156(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=10, stop=11, stride=1)
        return index_output_1


class Index157(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=11, stop=12, stride=1)
        return index_output_1


class Index158(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=12, stop=13, stride=1)
        return index_output_1


class Index159(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=13, stop=14, stride=1)
        return index_output_1


class Index160(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=14, stop=15, stride=1)
        return index_output_1


class Index161(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=15, stop=16, stride=1)
        return index_output_1


class Index162(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=16, stop=17, stride=1)
        return index_output_1


class Index163(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=17, stop=18, stride=1)
        return index_output_1


class Index164(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=18, stop=19, stride=1)
        return index_output_1


class Index165(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=19, stop=20, stride=1)
        return index_output_1


class Index166(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=20, stop=21, stride=1)
        return index_output_1


class Index167(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=21, stop=22, stride=1)
        return index_output_1


class Index168(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=22, stop=23, stride=1)
        return index_output_1


class Index169(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=23, stop=24, stride=1)
        return index_output_1


class Index170(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=24, stop=25, stride=1)
        return index_output_1


class Index171(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=144, stop=192, stride=1)
        return index_output_1


class Index172(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=48, stop=96, stride=1)
        return index_output_1


class Index173(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=0, stop=48, stride=1)
        return index_output_1


class Index174(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=96, stop=144, stride=1)
        return index_output_1


class Index175(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=256, stride=1)
        return index_output_1


class Index176(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=768, stop=1024, stride=1)
        return index_output_1


class Index177(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=1536, stop=1792, stride=1)
        return index_output_1


class Index178(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=2304, stop=2560, stride=1)
        return index_output_1


class Index179(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=512, stop=768, stride=1)
        return index_output_1


class Index180(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=1280, stop=1536, stride=1)
        return index_output_1


class Index181(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=2048, stop=2304, stride=1)
        return index_output_1


class Index182(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=2816, stop=3072, stride=1)
        return index_output_1


class Index183(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=256, stop=512, stride=1)
        return index_output_1


class Index184(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=1024, stop=1280, stride=1)
        return index_output_1


class Index185(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=1792, stop=2048, stride=1)
        return index_output_1


class Index186(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=2560, stop=2816, stride=1)
        return index_output_1


class Index187(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=1, stop=32, stride=2)
        return index_output_1


class Index188(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=0, stop=32, stride=2)
        return index_output_1


class Index189(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=768, stride=1)
        return index_output_1


class Index190(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=768, stop=1536, stride=1)
        return index_output_1


class Index191(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=1536, stop=2304, stride=1)
        return index_output_1


class Index192(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=0, stop=768, stride=1)
        return index_output_1


class Index193(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=768, stop=1536, stride=1)
        return index_output_1


class Index194(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=1536, stop=2304, stride=1)
        return index_output_1


class Index195(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=64, stop=192, stride=1)
        return index_output_1


class Index196(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=192, stop=448, stride=1)
        return index_output_1


class Index197(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=256, stop=384, stride=1)
        return index_output_1


class Index198(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=384, stop=448, stride=1)
        return index_output_1


class Index199(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=128, stop=192, stride=1)
        return index_output_1


class Index200(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=192, stop=256, stride=1)
        return index_output_1


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (
        Index0,
        [((1, 512), torch.int64)],
        {
            "model_names": [
                "pt_albert_large_v2_token_cls_hf",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_albert_xlarge_v2_token_cls_hf",
                "pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
                "pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index1,
        [((2, 768), torch.float32)],
        {
            "model_names": ["pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index2,
        [((2, 768), torch.float32)],
        {
            "model_names": ["pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index3,
        [((2,), torch.float32)],
        {
            "model_names": ["pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index4,
        [((2,), torch.float32)],
        {
            "model_names": ["pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index1,
        [((1, 128, 768), torch.float32)],
        {
            "model_names": [
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index5,
        [((1, 210, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w30_pose_estimation_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "0", "stop": "30", "stride": "1"},
        },
    ),
    (
        Index6,
        [((1, 210, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w30_pose_estimation_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "30", "stop": "90", "stride": "1"},
        },
    ),
    (
        Index7,
        [((1, 210, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w30_pose_estimation_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "90", "stop": "210", "stride": "1"},
        },
    ),
    (
        Index8,
        [((1, 210, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w30_pose_estimation_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "0", "stop": "120", "stride": "1"},
        },
    ),
    (
        Index9,
        [((1, 210, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w30_pose_estimation_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "120", "stop": "180", "stride": "1"},
        },
    ),
    (
        Index10,
        [((1, 210, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w30_pose_estimation_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "180", "stop": "210", "stride": "1"},
        },
    ),
    (
        Index11,
        [((1, 120, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w30_pose_estimation_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "0", "stop": "60", "stride": "1"},
        },
    ),
    (
        Index12,
        [((1, 120, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w30_pose_estimation_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "60", "stop": "90", "stride": "1"},
        },
    ),
    (
        Index13,
        [((1, 120, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w30_pose_estimation_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "90", "stop": "120", "stride": "1"},
        },
    ),
    (
        Index5,
        [((1, 120, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w30_pose_estimation_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "0", "stop": "30", "stride": "1"},
        },
    ),
    (
        Index14,
        [((1, 120, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w30_pose_estimation_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "30", "stop": "60", "stride": "1"},
        },
    ),
    (
        Index15,
        [((1, 120, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w30_pose_estimation_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "60", "stop": "120", "stride": "1"},
        },
    ),
    (
        Index16,
        [((1, 32, 4, 64), torch.float32)],
        {
            "model_names": ["pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "32", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index17,
        [((1, 32, 4, 64), torch.float32)],
        {
            "model_names": ["pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index16,
        [((1, 8, 4, 64), torch.float32)],
        {
            "model_names": ["pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "32", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index17,
        [((1, 8, 4, 64), torch.float32)],
        {
            "model_names": ["pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index18,
        [((1, 4, 2), torch.float32)],
        {
            "model_names": ["pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "3", "stop": "4", "stride": "1"},
        },
    ),
    (
        Index17,
        [((1, 32, 256, 64), torch.float32)],
        {
            "model_names": ["pt_phi1_microsoft_phi_1_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index16,
        [((1, 32, 256, 64), torch.float32)],
        {
            "model_names": ["pt_phi1_microsoft_phi_1_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "32", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index19,
        [((1, 32, 256, 32), torch.float32)],
        {
            "model_names": ["pt_phi1_microsoft_phi_1_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "16", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index20,
        [((1, 32, 256, 32), torch.float32)],
        {
            "model_names": ["pt_phi1_microsoft_phi_1_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "16", "stride": "1"},
        },
    ),
    (
        Index17,
        [((1, 32, 5, 64), torch.float32)],
        {
            "model_names": ["pt_phi_1_5_microsoft_phi_1_5_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index16,
        [((1, 32, 5, 64), torch.float32)],
        {
            "model_names": ["pt_phi_1_5_microsoft_phi_1_5_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "32", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index19,
        [((1, 32, 5, 32), torch.float32)],
        {
            "model_names": ["pt_phi_1_5_microsoft_phi_1_5_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "16", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index20,
        [((1, 32, 5, 32), torch.float32)],
        {
            "model_names": ["pt_phi_1_5_microsoft_phi_1_5_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "16", "stride": "1"},
        },
    ),
    (
        Index21,
        [((1, 5, 2), torch.float32)],
        {
            "model_names": ["pt_phi_1_5_microsoft_phi_1_5_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "4", "stop": "5", "stride": "1"},
        },
    ),
    (
        Index22,
        [((3, 64, 3, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-5", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index23,
        [((3, 64, 3, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-5", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index24,
        [((3, 64, 3, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-5", "start": "2", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index25,
        [((1, 56, 56, 96), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "3", "stop": "56", "stride": "1"},
        },
    ),
    (
        Index26,
        [((1, 56, 56, 96), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "0", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index27,
        [((1, 56, 56, 96), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "start": "3", "stop": "56", "stride": "1"},
        },
    ),
    (
        Index28,
        [((1, 56, 56, 96), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "start": "0", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index29,
        [((1, 56, 56, 96), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "53", "stop": "56", "stride": "1"},
        },
    ),
    (
        Index30,
        [((1, 56, 56, 96), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "0", "stop": "53", "stride": "1"},
        },
    ),
    (
        Index31,
        [((1, 56, 56, 96), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "start": "53", "stop": "56", "stride": "1"},
        },
    ),
    (
        Index32,
        [((1, 56, 56, 96), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "start": "0", "stop": "53", "stride": "1"},
        },
    ),
    (
        Index33,
        [((1, 56, 56, 96), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "0", "stop": "56", "stride": "2"},
        },
    ),
    (
        Index34,
        [((1, 56, 56, 96), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "1", "stop": "56", "stride": "2"},
        },
    ),
    (
        Index35,
        [((1, 28, 56, 96), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "start": "0", "stop": "56", "stride": "2"},
        },
    ),
    (
        Index36,
        [((1, 28, 56, 96), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "start": "1", "stop": "56", "stride": "2"},
        },
    ),
    (
        Index22,
        [((3, 16, 6, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-5", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index23,
        [((3, 16, 6, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-5", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index24,
        [((3, 16, 6, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-5", "start": "2", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index37,
        [((1, 28, 28, 192), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "3", "stop": "28", "stride": "1"},
        },
    ),
    (
        Index26,
        [((1, 28, 28, 192), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "0", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index38,
        [((1, 28, 28, 192), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "start": "3", "stop": "28", "stride": "1"},
        },
    ),
    (
        Index28,
        [((1, 28, 28, 192), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "start": "0", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index39,
        [((1, 28, 28, 192), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "25", "stop": "28", "stride": "1"},
        },
    ),
    (
        Index40,
        [((1, 28, 28, 192), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "0", "stop": "25", "stride": "1"},
        },
    ),
    (
        Index41,
        [((1, 28, 28, 192), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "start": "25", "stop": "28", "stride": "1"},
        },
    ),
    (
        Index42,
        [((1, 28, 28, 192), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "start": "0", "stop": "25", "stride": "1"},
        },
    ),
    (
        Index43,
        [((1, 28, 28, 192), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "0", "stop": "28", "stride": "2"},
        },
    ),
    (
        Index44,
        [((1, 28, 28, 192), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "1", "stop": "28", "stride": "2"},
        },
    ),
    (
        Index45,
        [((1, 14, 28, 192), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "start": "0", "stop": "28", "stride": "2"},
        },
    ),
    (
        Index46,
        [((1, 14, 28, 192), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "start": "1", "stop": "28", "stride": "2"},
        },
    ),
    (
        Index22,
        [((3, 4, 12, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-5", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index23,
        [((3, 4, 12, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-5", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index24,
        [((3, 4, 12, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-5", "start": "2", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index47,
        [((1, 14, 14, 384), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "3", "stop": "14", "stride": "1"},
        },
    ),
    (
        Index26,
        [((1, 14, 14, 384), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "0", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index48,
        [((1, 14, 14, 384), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "start": "3", "stop": "14", "stride": "1"},
        },
    ),
    (
        Index28,
        [((1, 14, 14, 384), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "start": "0", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index49,
        [((1, 14, 14, 384), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "11", "stop": "14", "stride": "1"},
        },
    ),
    (
        Index50,
        [((1, 14, 14, 384), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "0", "stop": "11", "stride": "1"},
        },
    ),
    (
        Index51,
        [((1, 14, 14, 384), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "start": "11", "stop": "14", "stride": "1"},
        },
    ),
    (
        Index52,
        [((1, 14, 14, 384), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "start": "0", "stop": "11", "stride": "1"},
        },
    ),
    (
        Index53,
        [((1, 14, 14, 384), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "0", "stop": "14", "stride": "2"},
        },
    ),
    (
        Index54,
        [((1, 14, 14, 384), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "1", "stop": "14", "stride": "2"},
        },
    ),
    (
        Index55,
        [((1, 7, 14, 384), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "start": "0", "stop": "14", "stride": "2"},
        },
    ),
    (
        Index56,
        [((1, 7, 14, 384), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "start": "1", "stop": "14", "stride": "2"},
        },
    ),
    (
        Index22,
        [((3, 1, 24, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-5", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index23,
        [((3, 1, 24, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-5", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index24,
        [((3, 1, 24, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-5", "start": "2", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index1,
        [((1, 197, 768), torch.bfloat16)],
        {
            "model_names": ["pt_vit_google_vit_base_patch16_224_img_cls_hf", "pt_vit_vit_b_16_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index57,
        [((1, 197, 768), torch.bfloat16)],
        {
            "model_names": ["pt_beit_microsoft_beit_base_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "start": "1", "stop": "197", "stride": "1"},
        },
    ),
    (
        Index58,
        [((3, 50, 1, 1024), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_l_32_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-4", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index59,
        [((3, 50, 1, 1024), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_l_32_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-4", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index60,
        [((3, 50, 1, 1024), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_l_32_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-4", "start": "2", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index1,
        [((1, 50, 1024), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_l_32_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index61,
        [((1, 160, 160, 160), torch.bfloat16)],
        {
            "model_names": ["pt_yolov8_yolov8x_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "0", "stop": "80", "stride": "1"},
        },
    ),
    (
        Index62,
        [((1, 160, 160, 160), torch.bfloat16)],
        {
            "model_names": ["pt_yolov8_yolov8x_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "80", "stop": "160", "stride": "1"},
        },
    ),
    (
        Index63,
        [((1, 320, 80, 80), torch.bfloat16)],
        {
            "model_names": ["pt_yolov8_yolov8x_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "0", "stop": "160", "stride": "1"},
        },
    ),
    (
        Index64,
        [((1, 320, 80, 80), torch.bfloat16)],
        {
            "model_names": ["pt_yolov8_yolov8x_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "160", "stop": "320", "stride": "1"},
        },
    ),
    (
        Index65,
        [((1, 640, 40, 40), torch.bfloat16)],
        {
            "model_names": ["pt_yolov8_yolov8x_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "0", "stop": "320", "stride": "1"},
        },
    ),
    (
        Index66,
        [((1, 640, 40, 40), torch.bfloat16)],
        {
            "model_names": ["pt_yolov8_yolov8x_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "320", "stop": "640", "stride": "1"},
        },
    ),
    (
        Index65,
        [((1, 640, 20, 20), torch.bfloat16)],
        {
            "model_names": ["pt_yolov8_yolov8x_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "0", "stop": "320", "stride": "1"},
        },
    ),
    (
        Index66,
        [((1, 640, 20, 20), torch.bfloat16)],
        {
            "model_names": ["pt_yolov8_yolov8x_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "320", "stop": "640", "stride": "1"},
        },
    ),
    (
        Index67,
        [((1, 144, 8400), torch.bfloat16)],
        {
            "model_names": ["pt_yolov8_yolov8x_obj_det_github", "pt_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index68,
        [((1, 144, 8400), torch.bfloat16)],
        {
            "model_names": ["pt_yolov8_yolov8x_obj_det_github", "pt_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "start": "64", "stop": "144", "stride": "1"},
        },
    ),
    (
        Index69,
        [((1, 4, 8400), torch.bfloat16)],
        {
            "model_names": ["pt_yolov8_yolov8x_obj_det_github", "pt_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "start": "0", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index70,
        [((1, 4, 8400), torch.bfloat16)],
        {
            "model_names": ["pt_yolov8_yolov8x_obj_det_github", "pt_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "start": "2", "stop": "4", "stride": "1"},
        },
    ),
    (
        Index71,
        [((1, 3, 640, 640), torch.bfloat16)],
        {
            "model_names": ["pt_yolox_yolox_l_obj_det_torchhub", "pt_yolox_yolox_m_obj_det_torchhub"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "start": "0", "stop": "640", "stride": "2"},
        },
    ),
    (
        Index72,
        [((1, 3, 640, 640), torch.bfloat16)],
        {
            "model_names": ["pt_yolox_yolox_l_obj_det_torchhub", "pt_yolox_yolox_m_obj_det_torchhub"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "start": "1", "stop": "640", "stride": "2"},
        },
    ),
    (
        Index73,
        [((1, 3, 320, 640), torch.bfloat16)],
        {
            "model_names": ["pt_yolox_yolox_l_obj_det_torchhub", "pt_yolox_yolox_m_obj_det_torchhub"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "start": "0", "stop": "640", "stride": "2"},
        },
    ),
    (
        Index74,
        [((1, 3, 320, 640), torch.bfloat16)],
        {
            "model_names": ["pt_yolox_yolox_l_obj_det_torchhub", "pt_yolox_yolox_m_obj_det_torchhub"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "start": "1", "stop": "640", "stride": "2"},
        },
    ),
    (
        Index1,
        [((1, 9, 768), torch.float32)],
        {
            "model_names": ["pd_roberta_rbt4_ch_seq_cls_padlenlp"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index75,
        [((732, 16), torch.bfloat16)],
        {
            "model_names": ["pt_beit_microsoft_beit_large_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "start": "0", "stop": "729", "stride": "1"},
        },
    ),
    (
        Index76,
        [((732, 16), torch.bfloat16)],
        {
            "model_names": ["pt_beit_microsoft_beit_large_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "start": "729", "stop": "732", "stride": "1"},
        },
    ),
    (
        Index57,
        [((1, 197, 1024), torch.bfloat16)],
        {
            "model_names": ["pt_beit_microsoft_beit_large_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "start": "1", "stop": "197", "stride": "1"},
        },
    ),
    (
        Index1,
        [((1, 197, 1024), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_l_16_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index77,
        [((1, 64, 28, 28), torch.bfloat16)],
        {
            "model_names": [
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "0", "stop": "16", "stride": "1"},
        },
    ),
    (
        Index78,
        [((1, 64, 28, 28), torch.bfloat16)],
        {
            "model_names": [
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "16", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index79,
        [((1, 64, 28, 28), torch.bfloat16)],
        {
            "model_names": [
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "32", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index80,
        [((1, 112, 7, 7), torch.bfloat16)],
        {
            "model_names": [
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index81,
        [((1, 112, 7, 7), torch.bfloat16)],
        {
            "model_names": [
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "64", "stop": "96", "stride": "1"},
        },
    ),
    (
        Index82,
        [((1, 112, 7, 7), torch.bfloat16)],
        {
            "model_names": [
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "96", "stop": "112", "stride": "1"},
        },
    ),
    (
        Index83,
        [((1, 126, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "0", "stop": "18", "stride": "1"},
        },
    ),
    (
        Index84,
        [((1, 126, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "18", "stop": "54", "stride": "1"},
        },
    ),
    (
        Index85,
        [((1, 126, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "54", "stop": "126", "stride": "1"},
        },
    ),
    (
        Index86,
        [((1, 126, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "0", "stop": "72", "stride": "1"},
        },
    ),
    (
        Index87,
        [((1, 126, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "72", "stop": "108", "stride": "1"},
        },
    ),
    (
        Index88,
        [((1, 126, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "108", "stop": "126", "stride": "1"},
        },
    ),
    (
        Index83,
        [((1, 72, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "0", "stop": "18", "stride": "1"},
        },
    ),
    (
        Index89,
        [((1, 72, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "0", "stop": "36", "stride": "1"},
        },
    ),
    (
        Index90,
        [((1, 72, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "36", "stop": "54", "stride": "1"},
        },
    ),
    (
        Index91,
        [((1, 72, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "54", "stop": "72", "stride": "1"},
        },
    ),
    (
        Index92,
        [((1, 72, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "18", "stop": "36", "stride": "1"},
        },
    ),
    (
        Index93,
        [((1, 72, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "36", "stop": "72", "stride": "1"},
        },
    ),
    (
        Index17,
        [((1, 32, 11, 80), torch.float32)],
        {
            "model_names": ["pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index94,
        [((1, 32, 11, 80), torch.float32)],
        {
            "model_names": ["pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "32", "stop": "80", "stride": "1"},
        },
    ),
    (
        Index19,
        [((1, 32, 11, 32), torch.float32)],
        {
            "model_names": ["pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "16", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index20,
        [((1, 32, 11, 32), torch.float32)],
        {
            "model_names": ["pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "16", "stride": "1"},
        },
    ),
    (
        Index95,
        [((1, 11, 2), torch.float32)],
        {
            "model_names": ["pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "10", "stop": "11", "stride": "1"},
        },
    ),
    (
        Index16,
        [((1, 16, 6, 64), torch.float32)],
        {
            "model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "32", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index17,
        [((1, 16, 6, 64), torch.float32)],
        {
            "model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index96,
        [((1, 12, 35, 128), torch.float32)],
        {
            "model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "64", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index97,
        [((1, 12, 35, 128), torch.float32)],
        {
            "model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index96,
        [((1, 2, 35, 128), torch.float32)],
        {
            "model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "64", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index97,
        [((1, 2, 35, 128), torch.float32)],
        {
            "model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index96,
        [((1, 12, 29, 128), torch.float32)],
        {
            "model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "64", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index97,
        [((1, 12, 29, 128), torch.float32)],
        {
            "model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index96,
        [((1, 2, 29, 128), torch.float32)],
        {
            "model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "64", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index97,
        [((1, 2, 29, 128), torch.float32)],
        {
            "model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index22,
        [((3, 64, 3, 64, 32), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index23,
        [((3, 64, 3, 64, 32), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index24,
        [((3, 64, 3, 64, 32), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "2", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index98,
        [((1, 64, 64, 96), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "4", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index99,
        [((1, 64, 64, 96), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "4", "stride": "1"},
        },
    ),
    (
        Index100,
        [((1, 64, 64, 96), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "4", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index101,
        [((1, 64, 64, 96), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "4", "stride": "1"},
        },
    ),
    (
        Index102,
        [((1, 64, 64, 96), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "60", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index11,
        [((1, 64, 64, 96), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "60", "stride": "1"},
        },
    ),
    (
        Index103,
        [((1, 64, 64, 96), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "60", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index104,
        [((1, 64, 64, 96), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "60", "stride": "1"},
        },
    ),
    (
        Index105,
        [((1, 64, 64, 96), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "64", "stride": "2"},
        },
    ),
    (
        Index106,
        [((1, 64, 64, 96), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "1", "stop": "64", "stride": "2"},
        },
    ),
    (
        Index107,
        [((1, 32, 64, 96), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "64", "stride": "2"},
        },
    ),
    (
        Index108,
        [((1, 32, 64, 96), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "1", "stop": "64", "stride": "2"},
        },
    ),
    (
        Index22,
        [((3, 16, 6, 64, 32), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index23,
        [((3, 16, 6, 64, 32), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index24,
        [((3, 16, 6, 64, 32), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "2", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index109,
        [((1, 32, 32, 192), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "4", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index99,
        [((1, 32, 32, 192), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "4", "stride": "1"},
        },
    ),
    (
        Index110,
        [((1, 32, 32, 192), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "4", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index101,
        [((1, 32, 32, 192), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "4", "stride": "1"},
        },
    ),
    (
        Index111,
        [((1, 32, 32, 192), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "28", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index112,
        [((1, 32, 32, 192), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "28", "stride": "1"},
        },
    ),
    (
        Index113,
        [((1, 32, 32, 192), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "28", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index114,
        [((1, 32, 32, 192), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "28", "stride": "1"},
        },
    ),
    (
        Index115,
        [((1, 32, 32, 192), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "32", "stride": "2"},
        },
    ),
    (
        Index116,
        [((1, 32, 32, 192), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "1", "stop": "32", "stride": "2"},
        },
    ),
    (
        Index117,
        [((1, 16, 32, 192), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "32", "stride": "2"},
        },
    ),
    (
        Index118,
        [((1, 16, 32, 192), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "1", "stop": "32", "stride": "2"},
        },
    ),
    (
        Index22,
        [((3, 4, 12, 64, 32), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index23,
        [((3, 4, 12, 64, 32), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index24,
        [((3, 4, 12, 64, 32), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "2", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index119,
        [((1, 16, 16, 384), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "4", "stop": "16", "stride": "1"},
        },
    ),
    (
        Index99,
        [((1, 16, 16, 384), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "4", "stride": "1"},
        },
    ),
    (
        Index120,
        [((1, 16, 16, 384), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "4", "stop": "16", "stride": "1"},
        },
    ),
    (
        Index101,
        [((1, 16, 16, 384), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "4", "stride": "1"},
        },
    ),
    (
        Index121,
        [((1, 16, 16, 384), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "12", "stop": "16", "stride": "1"},
        },
    ),
    (
        Index122,
        [((1, 16, 16, 384), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "12", "stride": "1"},
        },
    ),
    (
        Index123,
        [((1, 16, 16, 384), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "12", "stop": "16", "stride": "1"},
        },
    ),
    (
        Index124,
        [((1, 16, 16, 384), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "12", "stride": "1"},
        },
    ),
    (
        Index125,
        [((1, 16, 16, 384), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "16", "stride": "2"},
        },
    ),
    (
        Index126,
        [((1, 16, 16, 384), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "1", "stop": "16", "stride": "2"},
        },
    ),
    (
        Index127,
        [((1, 8, 16, 384), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "16", "stride": "2"},
        },
    ),
    (
        Index128,
        [((1, 8, 16, 384), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "1", "stop": "16", "stride": "2"},
        },
    ),
    (
        Index22,
        [((3, 1, 24, 64, 32), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index23,
        [((3, 1, 24, 64, 32), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index24,
        [((3, 1, 24, 64, 32), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "2", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index75,
        [((732, 12), torch.bfloat16)],
        {
            "model_names": ["pt_beit_microsoft_beit_base_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "start": "0", "stop": "729", "stride": "1"},
        },
    ),
    (
        Index76,
        [((732, 12), torch.bfloat16)],
        {
            "model_names": ["pt_beit_microsoft_beit_base_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "start": "729", "stop": "732", "stride": "1"},
        },
    ),
    (
        Index1,
        [((1, 197, 192), torch.bfloat16)],
        {
            "model_names": ["pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index129,
        [((1, 224, 35, 35), torch.bfloat16)],
        {
            "model_names": ["pt_inception_inception_v4_tf_in1k_img_cls_timm", "pt_inception_inception_v4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "0", "stop": "96", "stride": "1"},
        },
    ),
    (
        Index130,
        [((1, 224, 35, 35), torch.bfloat16)],
        {
            "model_names": ["pt_inception_inception_v4_tf_in1k_img_cls_timm", "pt_inception_inception_v4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "96", "stop": "160", "stride": "1"},
        },
    ),
    (
        Index131,
        [((1, 224, 35, 35), torch.bfloat16)],
        {
            "model_names": ["pt_inception_inception_v4_tf_in1k_img_cls_timm", "pt_inception_inception_v4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "160", "stop": "224", "stride": "1"},
        },
    ),
    (
        Index132,
        [((1, 768, 17, 17), torch.bfloat16)],
        {
            "model_names": ["pt_inception_inception_v4_tf_in1k_img_cls_timm", "pt_inception_inception_v4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "0", "stop": "384", "stride": "1"},
        },
    ),
    (
        Index133,
        [((1, 768, 17, 17), torch.bfloat16)],
        {
            "model_names": ["pt_inception_inception_v4_tf_in1k_img_cls_timm", "pt_inception_inception_v4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "384", "stop": "576", "stride": "1"},
        },
    ),
    (
        Index134,
        [((1, 768, 17, 17), torch.bfloat16)],
        {
            "model_names": ["pt_inception_inception_v4_tf_in1k_img_cls_timm", "pt_inception_inception_v4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "576", "stop": "768", "stride": "1"},
        },
    ),
    (
        Index135,
        [((1, 1024, 8, 8), torch.bfloat16)],
        {
            "model_names": ["pt_inception_inception_v4_tf_in1k_img_cls_timm", "pt_inception_inception_v4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "0", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index136,
        [((1, 1024, 8, 8), torch.bfloat16)],
        {
            "model_names": ["pt_inception_inception_v4_tf_in1k_img_cls_timm", "pt_inception_inception_v4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "256", "stop": "640", "stride": "1"},
        },
    ),
    (
        Index137,
        [((1, 1024, 8, 8), torch.bfloat16)],
        {
            "model_names": ["pt_inception_inception_v4_tf_in1k_img_cls_timm", "pt_inception_inception_v4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "640", "stop": "1024", "stride": "1"},
        },
    ),
    (
        Index16,
        [((1, 14, 29, 64), torch.float32)],
        {
            "model_names": ["pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "32", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index17,
        [((1, 14, 29, 64), torch.float32)],
        {
            "model_names": ["pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index16,
        [((1, 2, 29, 64), torch.float32)],
        {
            "model_names": ["pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "32", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index17,
        [((1, 2, 29, 64), torch.float32)],
        {
            "model_names": ["pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index58,
        [((3, 197, 1, 768), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_b_16_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-4", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index59,
        [((3, 197, 1, 768), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_b_16_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-4", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index60,
        [((3, 197, 1, 768), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_b_16_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-4", "start": "2", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index58,
        [((3, 197, 1, 1024), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_l_16_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-4", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index59,
        [((3, 197, 1, 1024), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_l_16_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-4", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index60,
        [((3, 197, 1, 1024), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_l_16_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-4", "start": "2", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index138,
        [((2050, 1024), torch.float32)],
        {
            "model_names": ["pt_xglm_facebook_xglm_564m_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "2", "stop": "258", "stride": "1"},
        },
    ),
    (
        Index139,
        [((1, 5880, 4), torch.bfloat16)],
        {
            "model_names": ["pt_yolo_v6_yolov6l_obj_det_torchhub"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "start": "0", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index140,
        [((1, 5880, 4), torch.bfloat16)],
        {
            "model_names": ["pt_yolo_v6_yolov6l_obj_det_torchhub"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "start": "2", "stop": "4", "stride": "1"},
        },
    ),
    (
        Index80,
        [((1, 128, 160, 160), torch.bfloat16)],
        {
            "model_names": ["pt_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index141,
        [((1, 128, 160, 160), torch.bfloat16)],
        {
            "model_names": ["pt_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "64", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index142,
        [((1, 256, 159, 159), torch.bfloat16)],
        {
            "model_names": ["pt_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "0", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index143,
        [((1, 256, 159, 159), torch.bfloat16)],
        {
            "model_names": ["pt_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "128", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index142,
        [((1, 256, 80, 80), torch.bfloat16)],
        {
            "model_names": ["pt_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "0", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index143,
        [((1, 256, 80, 80), torch.bfloat16)],
        {
            "model_names": ["pt_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "128", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index135,
        [((1, 512, 79, 79), torch.bfloat16)],
        {
            "model_names": ["pt_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "0", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index144,
        [((1, 512, 79, 79), torch.bfloat16)],
        {
            "model_names": ["pt_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "256", "stop": "512", "stride": "1"},
        },
    ),
    (
        Index135,
        [((1, 512, 40, 40), torch.bfloat16)],
        {
            "model_names": ["pt_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "0", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index144,
        [((1, 512, 40, 40), torch.bfloat16)],
        {
            "model_names": ["pt_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "256", "stop": "512", "stride": "1"},
        },
    ),
    (
        Index135,
        [((1, 512, 39, 39), torch.bfloat16)],
        {
            "model_names": ["pt_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "0", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index144,
        [((1, 512, 39, 39), torch.bfloat16)],
        {
            "model_names": ["pt_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "256", "stop": "512", "stride": "1"},
        },
    ),
    (
        Index135,
        [((1, 512, 20, 20), torch.bfloat16)],
        {
            "model_names": ["pt_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "0", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index144,
        [((1, 512, 20, 20), torch.bfloat16)],
        {
            "model_names": ["pt_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "256", "stop": "512", "stride": "1"},
        },
    ),
    (
        Index142,
        [((1, 256, 79, 79), torch.bfloat16)],
        {
            "model_names": ["pt_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "0", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index143,
        [((1, 256, 79, 79), torch.bfloat16)],
        {
            "model_names": ["pt_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "128", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index1,
        [((1, 197, 192), torch.float32)],
        {
            "model_names": ["onnx_deit_facebook_deit_tiny_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index77,
        [((1, 32, 160, 160), torch.float32)],
        {
            "model_names": ["onnx_yolov8_default_obj_det_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "16", "stride": "1"},
        },
    ),
    (
        Index78,
        [((1, 32, 160, 160), torch.float32)],
        {
            "model_names": ["onnx_yolov8_default_obj_det_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "16", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index145,
        [((1, 64, 80, 80), torch.float32)],
        {
            "model_names": ["onnx_yolov8_default_obj_det_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index79,
        [((1, 64, 80, 80), torch.float32)],
        {
            "model_names": ["onnx_yolov8_default_obj_det_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "32", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index80,
        [((1, 128, 40, 40), torch.float32)],
        {
            "model_names": ["onnx_yolov8_default_obj_det_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index141,
        [((1, 128, 40, 40), torch.float32)],
        {
            "model_names": ["onnx_yolov8_default_obj_det_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "64", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index142,
        [((1, 256, 20, 20), torch.float32)],
        {
            "model_names": ["onnx_yolov8_default_obj_det_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index143,
        [((1, 256, 20, 20), torch.float32)],
        {
            "model_names": ["onnx_yolov8_default_obj_det_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "128", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index67,
        [((1, 144, 8400), torch.float32)],
        {
            "model_names": ["onnx_yolov8_default_obj_det_github"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index68,
        [((1, 144, 8400), torch.float32)],
        {
            "model_names": ["onnx_yolov8_default_obj_det_github"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "64", "stop": "144", "stride": "1"},
        },
    ),
    (
        Index69,
        [((1, 4, 8400), torch.float32)],
        {
            "model_names": ["onnx_yolov8_default_obj_det_github"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index70,
        [((1, 4, 8400), torch.float32)],
        {
            "model_names": ["onnx_yolov8_default_obj_det_github"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "2", "stop": "4", "stride": "1"},
        },
    ),
    (
        Index146,
        [((25, 1, 288), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index147,
        [((25, 1, 288), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index148,
        [((25, 1, 288), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "2", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index149,
        [((25, 1, 288), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "3", "stop": "4", "stride": "1"},
        },
    ),
    (
        Index150,
        [((25, 1, 288), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "4", "stop": "5", "stride": "1"},
        },
    ),
    (
        Index151,
        [((25, 1, 288), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "5", "stop": "6", "stride": "1"},
        },
    ),
    (
        Index152,
        [((25, 1, 288), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "6", "stop": "7", "stride": "1"},
        },
    ),
    (
        Index153,
        [((25, 1, 288), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "7", "stop": "8", "stride": "1"},
        },
    ),
    (
        Index154,
        [((25, 1, 288), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "8", "stop": "9", "stride": "1"},
        },
    ),
    (
        Index155,
        [((25, 1, 288), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "9", "stop": "10", "stride": "1"},
        },
    ),
    (
        Index156,
        [((25, 1, 288), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "10", "stop": "11", "stride": "1"},
        },
    ),
    (
        Index157,
        [((25, 1, 288), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "11", "stop": "12", "stride": "1"},
        },
    ),
    (
        Index158,
        [((25, 1, 288), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "12", "stop": "13", "stride": "1"},
        },
    ),
    (
        Index159,
        [((25, 1, 288), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "13", "stop": "14", "stride": "1"},
        },
    ),
    (
        Index160,
        [((25, 1, 288), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "14", "stop": "15", "stride": "1"},
        },
    ),
    (
        Index161,
        [((25, 1, 288), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "15", "stop": "16", "stride": "1"},
        },
    ),
    (
        Index162,
        [((25, 1, 288), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "16", "stop": "17", "stride": "1"},
        },
    ),
    (
        Index163,
        [((25, 1, 288), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "17", "stop": "18", "stride": "1"},
        },
    ),
    (
        Index164,
        [((25, 1, 288), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "18", "stop": "19", "stride": "1"},
        },
    ),
    (
        Index165,
        [((25, 1, 288), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "19", "stop": "20", "stride": "1"},
        },
    ),
    (
        Index166,
        [((25, 1, 288), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "20", "stop": "21", "stride": "1"},
        },
    ),
    (
        Index167,
        [((25, 1, 288), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "21", "stop": "22", "stride": "1"},
        },
    ),
    (
        Index168,
        [((25, 1, 288), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "22", "stop": "23", "stride": "1"},
        },
    ),
    (
        Index169,
        [((25, 1, 288), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "23", "stop": "24", "stride": "1"},
        },
    ),
    (
        Index170,
        [((25, 1, 288), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "24", "stop": "25", "stride": "1"},
        },
    ),
    (
        Index171,
        [((1, 192), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "144", "stop": "192", "stride": "1"},
        },
    ),
    (
        Index172,
        [((1, 192), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "48", "stop": "96", "stride": "1"},
        },
    ),
    (
        Index173,
        [((1, 192), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "48", "stride": "1"},
        },
    ),
    (
        Index174,
        [((1, 192), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "96", "stop": "144", "stride": "1"},
        },
    ),
    (
        Index146,
        [((25, 1, 96), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index147,
        [((25, 1, 96), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index148,
        [((25, 1, 96), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "2", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index149,
        [((25, 1, 96), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "3", "stop": "4", "stride": "1"},
        },
    ),
    (
        Index150,
        [((25, 1, 96), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "4", "stop": "5", "stride": "1"},
        },
    ),
    (
        Index151,
        [((25, 1, 96), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "5", "stop": "6", "stride": "1"},
        },
    ),
    (
        Index152,
        [((25, 1, 96), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "6", "stop": "7", "stride": "1"},
        },
    ),
    (
        Index153,
        [((25, 1, 96), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "7", "stop": "8", "stride": "1"},
        },
    ),
    (
        Index154,
        [((25, 1, 96), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "8", "stop": "9", "stride": "1"},
        },
    ),
    (
        Index155,
        [((25, 1, 96), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "9", "stop": "10", "stride": "1"},
        },
    ),
    (
        Index156,
        [((25, 1, 96), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "10", "stop": "11", "stride": "1"},
        },
    ),
    (
        Index157,
        [((25, 1, 96), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "11", "stop": "12", "stride": "1"},
        },
    ),
    (
        Index158,
        [((25, 1, 96), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "12", "stop": "13", "stride": "1"},
        },
    ),
    (
        Index159,
        [((25, 1, 96), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "13", "stop": "14", "stride": "1"},
        },
    ),
    (
        Index160,
        [((25, 1, 96), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "14", "stop": "15", "stride": "1"},
        },
    ),
    (
        Index161,
        [((25, 1, 96), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "15", "stop": "16", "stride": "1"},
        },
    ),
    (
        Index162,
        [((25, 1, 96), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "16", "stop": "17", "stride": "1"},
        },
    ),
    (
        Index163,
        [((25, 1, 96), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "17", "stop": "18", "stride": "1"},
        },
    ),
    (
        Index164,
        [((25, 1, 96), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "18", "stop": "19", "stride": "1"},
        },
    ),
    (
        Index165,
        [((25, 1, 96), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "19", "stop": "20", "stride": "1"},
        },
    ),
    (
        Index166,
        [((25, 1, 96), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "20", "stop": "21", "stride": "1"},
        },
    ),
    (
        Index167,
        [((25, 1, 96), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "21", "stop": "22", "stride": "1"},
        },
    ),
    (
        Index168,
        [((25, 1, 96), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "22", "stop": "23", "stride": "1"},
        },
    ),
    (
        Index169,
        [((25, 1, 96), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "23", "stop": "24", "stride": "1"},
        },
    ),
    (
        Index170,
        [((25, 1, 96), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "24", "stop": "25", "stride": "1"},
        },
    ),
    (
        Index175,
        [((3072, 1024), torch.float32)],
        {
            "model_names": ["pt_codegen_salesforce_codegen_350m_nl_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index176,
        [((3072, 1024), torch.float32)],
        {
            "model_names": ["pt_codegen_salesforce_codegen_350m_nl_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "768", "stop": "1024", "stride": "1"},
        },
    ),
    (
        Index177,
        [((3072, 1024), torch.float32)],
        {
            "model_names": ["pt_codegen_salesforce_codegen_350m_nl_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "1536", "stop": "1792", "stride": "1"},
        },
    ),
    (
        Index178,
        [((3072, 1024), torch.float32)],
        {
            "model_names": ["pt_codegen_salesforce_codegen_350m_nl_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "2304", "stop": "2560", "stride": "1"},
        },
    ),
    (
        Index179,
        [((3072, 1024), torch.float32)],
        {
            "model_names": ["pt_codegen_salesforce_codegen_350m_nl_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "512", "stop": "768", "stride": "1"},
        },
    ),
    (
        Index180,
        [((3072, 1024), torch.float32)],
        {
            "model_names": ["pt_codegen_salesforce_codegen_350m_nl_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "1280", "stop": "1536", "stride": "1"},
        },
    ),
    (
        Index181,
        [((3072, 1024), torch.float32)],
        {
            "model_names": ["pt_codegen_salesforce_codegen_350m_nl_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "2048", "stop": "2304", "stride": "1"},
        },
    ),
    (
        Index182,
        [((3072, 1024), torch.float32)],
        {
            "model_names": ["pt_codegen_salesforce_codegen_350m_nl_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "2816", "stop": "3072", "stride": "1"},
        },
    ),
    (
        Index183,
        [((3072, 1024), torch.float32)],
        {
            "model_names": ["pt_codegen_salesforce_codegen_350m_nl_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "256", "stop": "512", "stride": "1"},
        },
    ),
    (
        Index184,
        [((3072, 1024), torch.float32)],
        {
            "model_names": ["pt_codegen_salesforce_codegen_350m_nl_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "1024", "stop": "1280", "stride": "1"},
        },
    ),
    (
        Index185,
        [((3072, 1024), torch.float32)],
        {
            "model_names": ["pt_codegen_salesforce_codegen_350m_nl_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "1792", "stop": "2048", "stride": "1"},
        },
    ),
    (
        Index186,
        [((3072, 1024), torch.float32)],
        {
            "model_names": ["pt_codegen_salesforce_codegen_350m_nl_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "2560", "stop": "2816", "stride": "1"},
        },
    ),
    (
        Index17,
        [((1, 256, 16, 64), torch.float32)],
        {
            "model_names": ["pt_codegen_salesforce_codegen_350m_nl_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index16,
        [((1, 256, 16, 64), torch.float32)],
        {
            "model_names": ["pt_codegen_salesforce_codegen_350m_nl_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "32", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index187,
        [((1, 256, 16, 32), torch.float32)],
        {
            "model_names": ["pt_codegen_salesforce_codegen_350m_nl_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "1", "stop": "32", "stride": "2"},
        },
    ),
    (
        Index188,
        [((1, 256, 16, 32), torch.float32)],
        {
            "model_names": ["pt_codegen_salesforce_codegen_350m_nl_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "32", "stride": "2"},
        },
    ),
    (
        Index189,
        [((2304, 768), torch.float32)],
        {
            "model_names": ["pt_gpt_gpt2_text_gen_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "768", "stride": "1"},
        },
    ),
    (
        Index190,
        [((2304, 768), torch.float32)],
        {
            "model_names": ["pt_gpt_gpt2_text_gen_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "768", "stop": "1536", "stride": "1"},
        },
    ),
    (
        Index191,
        [((2304, 768), torch.float32)],
        {
            "model_names": ["pt_gpt_gpt2_text_gen_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "1536", "stop": "2304", "stride": "1"},
        },
    ),
    (
        Index192,
        [((2304,), torch.float32)],
        {
            "model_names": ["pt_gpt_gpt2_text_gen_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "768", "stride": "1"},
        },
    ),
    (
        Index193,
        [((2304,), torch.float32)],
        {
            "model_names": ["pt_gpt_gpt2_text_gen_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "768", "stop": "1536", "stride": "1"},
        },
    ),
    (
        Index194,
        [((2304,), torch.float32)],
        {
            "model_names": ["pt_gpt_gpt2_text_gen_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "1536", "stop": "2304", "stride": "1"},
        },
    ),
    (
        Index80,
        [((1, 448, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w64_pose_estimation_osmr"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index195,
        [((1, 448, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w64_pose_estimation_osmr"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "64", "stop": "192", "stride": "1"},
        },
    ),
    (
        Index196,
        [((1, 448, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w64_pose_estimation_osmr"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "192", "stop": "448", "stride": "1"},
        },
    ),
    (
        Index135,
        [((1, 448, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w64_pose_estimation_osmr"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "0", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index197,
        [((1, 448, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w64_pose_estimation_osmr"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "256", "stop": "384", "stride": "1"},
        },
    ),
    (
        Index198,
        [((1, 448, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w64_pose_estimation_osmr"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "384", "stop": "448", "stride": "1"},
        },
    ),
    (
        Index142,
        [((1, 256, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w64_pose_estimation_osmr"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "0", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index199,
        [((1, 256, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w64_pose_estimation_osmr"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "128", "stop": "192", "stride": "1"},
        },
    ),
    (
        Index200,
        [((1, 256, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w64_pose_estimation_osmr"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "192", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index80,
        [((1, 256, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w64_pose_estimation_osmr"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index141,
        [((1, 256, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w64_pose_estimation_osmr"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "64", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index143,
        [((1, 256, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w64_pose_estimation_osmr"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "128", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index17,
        [((1, 32, 12, 64), torch.float32)],
        {
            "model_names": ["pt_phi1_microsoft_phi_1_token_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index16,
        [((1, 32, 12, 64), torch.float32)],
        {
            "model_names": ["pt_phi1_microsoft_phi_1_token_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "32", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index19,
        [((1, 32, 12, 32), torch.float32)],
        {
            "model_names": ["pt_phi1_microsoft_phi_1_token_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "16", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index20,
        [((1, 32, 12, 32), torch.float32)],
        {
            "model_names": ["pt_phi1_microsoft_phi_1_token_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "16", "stride": "1"},
        },
    ),
]


@pytest.mark.nightly_models_ops
@pytest.mark.parametrize("forge_module_and_shapes_dtypes", forge_modules_and_shapes_dtypes_list, ids=ids_func)
def test_module(forge_module_and_shapes_dtypes):

    record_forge_op_name("Index")

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
