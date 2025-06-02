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
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=1, stride=1)
        return index_output_1


class Index1(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=1, stride=1)
        return index_output_1


class Index2(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=1, stop=2, stride=1)
        return index_output_1


class Index3(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=0, stop=128, stride=1)
        return index_output_1


class Index4(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=0, stop=9, stride=1)
        return index_output_1


class Index5(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=0, stop=384, stride=1)
        return index_output_1


class Index6(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=0, stop=8, stride=1)
        return index_output_1


class Index7(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=0, stop=14, stride=1)
        return index_output_1


class Index8(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=1, stop=197, stride=1)
        return index_output_1


class Index9(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=128, stop=256, stride=1)
        return index_output_1


class Index10(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=2, stop=3, stride=1)
        return index_output_1


class Index11(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=64, stride=1)
        return index_output_1


class Index12(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=64, stop=160, stride=1)
        return index_output_1


class Index13(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=160, stop=176, stride=1)
        return index_output_1


class Index14(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=88, stride=1)
        return index_output_1


class Index15(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=88, stop=132, stride=1)
        return index_output_1


class Index16(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=132, stop=176, stride=1)
        return index_output_1


class Index17(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=44, stride=1)
        return index_output_1


class Index18(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=44, stop=88, stride=1)
        return index_output_1


class Index19(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=88, stop=176, stride=1)
        return index_output_1


class Index20(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=128, stride=1)
        return index_output_1


class Index21(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=128, stop=256, stride=1)
        return index_output_1


class Index22(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=256, stop=288, stride=1)
        return index_output_1


class Index23(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=192, stride=1)
        return index_output_1


class Index24(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=192, stop=288, stride=1)
        return index_output_1


class Index25(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=288, stop=304, stride=1)
        return index_output_1


class Index26(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=160, stride=1)
        return index_output_1


class Index27(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=160, stop=272, stride=1)
        return index_output_1


class Index28(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=272, stop=296, stride=1)
        return index_output_1


class Index29(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=256, stop=280, stride=1)
        return index_output_1


class Index30(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=112, stride=1)
        return index_output_1


class Index31(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=112, stop=256, stride=1)
        return index_output_1


class Index32(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=256, stride=1)
        return index_output_1


class Index33(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=256, stop=416, stride=1)
        return index_output_1


class Index34(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=416, stop=448, stride=1)
        return index_output_1


class Index35(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=384, stride=1)
        return index_output_1


class Index36(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=384, stop=576, stride=1)
        return index_output_1


class Index37(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=576, stop=624, stride=1)
        return index_output_1


class Index38(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=30, stride=1)
        return index_output_1


class Index39(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=30, stop=90, stride=1)
        return index_output_1


class Index40(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=90, stop=210, stride=1)
        return index_output_1


class Index41(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=120, stride=1)
        return index_output_1


class Index42(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=120, stop=180, stride=1)
        return index_output_1


class Index43(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=180, stop=210, stride=1)
        return index_output_1


class Index44(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=60, stride=1)
        return index_output_1


class Index45(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=60, stop=90, stride=1)
        return index_output_1


class Index46(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=90, stop=120, stride=1)
        return index_output_1


class Index47(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=30, stop=60, stride=1)
        return index_output_1


class Index48(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=60, stop=120, stride=1)
        return index_output_1


class Index49(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=96, stride=1)
        return index_output_1


class Index50(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=96, stop=160, stride=1)
        return index_output_1


class Index51(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=160, stop=224, stride=1)
        return index_output_1


class Index52(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=576, stop=768, stride=1)
        return index_output_1


class Index53(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=256, stop=640, stride=1)
        return index_output_1


class Index54(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=640, stop=1024, stride=1)
        return index_output_1


class Index55(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=768, stride=1)
        return index_output_1


class Index56(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=768, stop=1536, stride=1)
        return index_output_1


class Index57(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=1536, stop=2304, stride=1)
        return index_output_1


class Index58(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=0, stop=768, stride=1)
        return index_output_1


class Index59(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=768, stop=1536, stride=1)
        return index_output_1


class Index60(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=1536, stop=2304, stride=1)
        return index_output_1


class Index61(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=7, stride=1)
        return index_output_1


class Index62(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=256, stride=1)
        return index_output_1


class Index63(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=0, stop=7, stride=1)
        return index_output_1


class Index64(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=1, stop=2, stride=1)
        return index_output_1


class Index65(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=0, stop=1, stride=1)
        return index_output_1


class Index66(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=1, stop=2, stride=1)
        return index_output_1


class Index67(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=0, stop=32, stride=1)
        return index_output_1


class Index68(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=32, stop=64, stride=1)
        return index_output_1


class Index69(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=16, stop=32, stride=1)
        return index_output_1


class Index70(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=0, stop=16, stride=1)
        return index_output_1


class Index71(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=64, stop=128, stride=1)
        return index_output_1


class Index72(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=0, stop=64, stride=1)
        return index_output_1


class Index73(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-5, start=0, stop=1, stride=1)
        return index_output_1


class Index74(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-5, start=1, stop=2, stride=1)
        return index_output_1


class Index75(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-5, start=2, stop=3, stride=1)
        return index_output_1


class Index76(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=4, stop=64, stride=1)
        return index_output_1


class Index77(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=4, stride=1)
        return index_output_1


class Index78(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=4, stop=64, stride=1)
        return index_output_1


class Index79(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=4, stride=1)
        return index_output_1


class Index80(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=60, stop=64, stride=1)
        return index_output_1


class Index81(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=60, stop=64, stride=1)
        return index_output_1


class Index82(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=60, stride=1)
        return index_output_1


class Index83(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=64, stride=2)
        return index_output_1


class Index84(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=1, stop=64, stride=2)
        return index_output_1


class Index85(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=-4, stop=64, stride=1)
        return index_output_1


class Index86(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=-4, stride=1)
        return index_output_1


class Index87(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=-4, stop=64, stride=1)
        return index_output_1


class Index88(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=-4, stride=1)
        return index_output_1


class Index89(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=64, stride=2)
        return index_output_1


class Index90(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=1, stop=64, stride=2)
        return index_output_1


class Index91(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=4, stop=32, stride=1)
        return index_output_1


class Index92(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=4, stop=32, stride=1)
        return index_output_1


class Index93(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=28, stop=32, stride=1)
        return index_output_1


class Index94(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=28, stride=1)
        return index_output_1


class Index95(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=28, stop=32, stride=1)
        return index_output_1


class Index96(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=28, stride=1)
        return index_output_1


class Index97(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=32, stride=2)
        return index_output_1


class Index98(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=1, stop=32, stride=2)
        return index_output_1


class Index99(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=-4, stop=32, stride=1)
        return index_output_1


class Index100(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=-4, stop=32, stride=1)
        return index_output_1


class Index101(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=32, stride=2)
        return index_output_1


class Index102(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=1, stop=32, stride=2)
        return index_output_1


class Index103(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=4, stop=16, stride=1)
        return index_output_1


class Index104(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=4, stop=16, stride=1)
        return index_output_1


class Index105(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=12, stop=16, stride=1)
        return index_output_1


class Index106(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=12, stride=1)
        return index_output_1


class Index107(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=12, stop=16, stride=1)
        return index_output_1


class Index108(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=12, stride=1)
        return index_output_1


class Index109(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=16, stride=2)
        return index_output_1


class Index110(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=1, stop=16, stride=2)
        return index_output_1


class Index111(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=-4, stop=16, stride=1)
        return index_output_1


class Index112(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=-4, stop=16, stride=1)
        return index_output_1


class Index113(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=16, stride=2)
        return index_output_1


class Index114(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=1, stop=16, stride=2)
        return index_output_1


class Index115(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=0, stop=2, stride=1)
        return index_output_1


class Index116(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=2, stop=4, stride=1)
        return index_output_1


class Index117(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=416, stride=2)
        return index_output_1


class Index118(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=1, stop=416, stride=2)
        return index_output_1


class Index119(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=0, stop=416, stride=2)
        return index_output_1


class Index120(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=1, stop=416, stride=2)
        return index_output_1


class Index121(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=32, stride=1)
        return index_output_1


class Index122(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=32, stop=96, stride=1)
        return index_output_1


class Index123(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=96, stop=224, stride=1)
        return index_output_1


class Index124(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=128, stop=192, stride=1)
        return index_output_1


class Index125(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=192, stop=224, stride=1)
        return index_output_1


class Index126(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=64, stop=96, stride=1)
        return index_output_1


class Index127(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=96, stop=128, stride=1)
        return index_output_1


class Index128(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=32, stop=64, stride=1)
        return index_output_1


class Index129(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=64, stop=128, stride=1)
        return index_output_1


class Index130(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=-1, stop=72, stride=1)
        return index_output_1


class Index131(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=-24, stop=96, stride=1)
        return index_output_1


class Index132(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=0, stop=72, stride=1)
        return index_output_1


class Index133(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=0, stop=1, stride=1)
        return index_output_1


class Index134(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=1, stop=2, stride=1)
        return index_output_1


class Index135(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=2, stop=3, stride=1)
        return index_output_1


class Index136(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=640, stride=2)
        return index_output_1


class Index137(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=1, stop=640, stride=2)
        return index_output_1


class Index138(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=0, stop=640, stride=2)
        return index_output_1


class Index139(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=1, stop=640, stride=2)
        return index_output_1


class Index140(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=729, stride=1)
        return index_output_1


class Index141(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=729, stop=732, stride=1)
        return index_output_1


class Index142(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=0, stop=256, stride=1)
        return index_output_1


class Index143(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=44, stop=132, stride=1)
        return index_output_1


class Index144(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=132, stop=308, stride=1)
        return index_output_1


class Index145(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=176, stride=1)
        return index_output_1


class Index146(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=176, stop=264, stride=1)
        return index_output_1


class Index147(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=264, stop=308, stride=1)
        return index_output_1


class Index148(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=0, stop=4, stride=1)
        return index_output_1


class Index149(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=4, stop=8, stride=1)
        return index_output_1


class Index150(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=3, stop=56, stride=1)
        return index_output_1


class Index151(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=3, stride=1)
        return index_output_1


class Index152(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=3, stop=56, stride=1)
        return index_output_1


class Index153(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=3, stride=1)
        return index_output_1


class Index154(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=53, stop=56, stride=1)
        return index_output_1


class Index155(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=53, stride=1)
        return index_output_1


class Index156(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=53, stop=56, stride=1)
        return index_output_1


class Index157(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=53, stride=1)
        return index_output_1


class Index158(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=56, stride=2)
        return index_output_1


class Index159(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=1, stop=56, stride=2)
        return index_output_1


class Index160(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=56, stride=2)
        return index_output_1


class Index161(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=1, stop=56, stride=2)
        return index_output_1


class Index162(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=3, stop=28, stride=1)
        return index_output_1


class Index163(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=3, stop=28, stride=1)
        return index_output_1


class Index164(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=25, stop=28, stride=1)
        return index_output_1


class Index165(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=25, stride=1)
        return index_output_1


class Index166(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=25, stop=28, stride=1)
        return index_output_1


class Index167(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=25, stride=1)
        return index_output_1


class Index168(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=28, stride=2)
        return index_output_1


class Index169(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=1, stop=28, stride=2)
        return index_output_1


class Index170(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=28, stride=2)
        return index_output_1


class Index171(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=1, stop=28, stride=2)
        return index_output_1


class Index172(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=3, stop=14, stride=1)
        return index_output_1


class Index173(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=3, stop=14, stride=1)
        return index_output_1


class Index174(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=11, stop=14, stride=1)
        return index_output_1


class Index175(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=11, stride=1)
        return index_output_1


class Index176(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=11, stop=14, stride=1)
        return index_output_1


class Index177(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=11, stride=1)
        return index_output_1


class Index178(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=14, stride=2)
        return index_output_1


class Index179(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=1, stop=14, stride=2)
        return index_output_1


class Index180(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=14, stride=2)
        return index_output_1


class Index181(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=1, stop=14, stride=2)
        return index_output_1


class Index182(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=1, stop=-100, stride=1)
        return index_output_1


class Index183(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=-100, stop=4251, stride=1)
        return index_output_1


class Index184(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=-100, stop=1445, stride=1)
        return index_output_1


class Index185(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=16, stride=1)
        return index_output_1


class Index186(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=16, stop=32, stride=1)
        return index_output_1


class Index187(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=64, stride=1)
        return index_output_1


class Index188(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=64, stop=144, stride=1)
        return index_output_1


class Index189(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=2, stride=1)
        return index_output_1


class Index190(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=2, stop=4, stride=1)
        return index_output_1


class Index191(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=64, stop=128, stride=1)
        return index_output_1


class Index192(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=32, stride=1)
        return index_output_1


class Index193(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=32, stop=64, stride=1)
        return index_output_1


class Index194(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=48, stride=1)
        return index_output_1


class Index195(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=48, stop=144, stride=1)
        return index_output_1


class Index196(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=144, stop=336, stride=1)
        return index_output_1


class Index197(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=288, stop=336, stride=1)
        return index_output_1


class Index198(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=96, stop=144, stride=1)
        return index_output_1


class Index199(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=144, stop=192, stride=1)
        return index_output_1


class Index200(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=48, stop=96, stride=1)
        return index_output_1


class Index201(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=96, stop=192, stride=1)
        return index_output_1


class Index202(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=32, stop=80, stride=1)
        return index_output_1


class Index203(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=2, stop=3, stride=1)
        return index_output_1


class Index204(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=3, stop=4, stride=1)
        return index_output_1


class Index205(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=2, stop=258, stride=1)
        return index_output_1


class Index206(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=256, stop=512, stride=1)
        return index_output_1


class Index207(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=3, stop=4, stride=1)
        return index_output_1


class Index208(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=4, stop=5, stride=1)
        return index_output_1


class Index209(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=5, stop=6, stride=1)
        return index_output_1


class Index210(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=6, stop=7, stride=1)
        return index_output_1


class Index211(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=7, stop=8, stride=1)
        return index_output_1


class Index212(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=8, stop=9, stride=1)
        return index_output_1


class Index213(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=9, stop=10, stride=1)
        return index_output_1


class Index214(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=10, stop=11, stride=1)
        return index_output_1


class Index215(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=11, stop=12, stride=1)
        return index_output_1


class Index216(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=12, stop=13, stride=1)
        return index_output_1


class Index217(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=13, stop=14, stride=1)
        return index_output_1


class Index218(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=14, stop=15, stride=1)
        return index_output_1


class Index219(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=15, stop=16, stride=1)
        return index_output_1


class Index220(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=16, stop=17, stride=1)
        return index_output_1


class Index221(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=17, stop=18, stride=1)
        return index_output_1


class Index222(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=18, stop=19, stride=1)
        return index_output_1


class Index223(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=19, stop=20, stride=1)
        return index_output_1


class Index224(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=20, stop=21, stride=1)
        return index_output_1


class Index225(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=21, stop=22, stride=1)
        return index_output_1


class Index226(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=22, stop=23, stride=1)
        return index_output_1


class Index227(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=23, stop=24, stride=1)
        return index_output_1


class Index228(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=24, stop=25, stride=1)
        return index_output_1


class Index229(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=144, stop=192, stride=1)
        return index_output_1


class Index230(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=48, stop=96, stride=1)
        return index_output_1


class Index231(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=0, stop=48, stride=1)
        return index_output_1


class Index232(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=96, stop=144, stride=1)
        return index_output_1


class Index233(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=768, stop=1024, stride=1)
        return index_output_1


class Index234(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=1536, stop=1792, stride=1)
        return index_output_1


class Index235(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=2304, stop=2560, stride=1)
        return index_output_1


class Index236(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=512, stop=768, stride=1)
        return index_output_1


class Index237(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=1280, stop=1536, stride=1)
        return index_output_1


class Index238(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=2048, stop=2304, stride=1)
        return index_output_1


class Index239(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=2816, stop=3072, stride=1)
        return index_output_1


class Index240(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=256, stop=512, stride=1)
        return index_output_1


class Index241(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=1024, stop=1280, stride=1)
        return index_output_1


class Index242(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=1792, stop=2048, stride=1)
        return index_output_1


class Index243(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=2560, stop=2816, stride=1)
        return index_output_1


class Index244(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=1, stop=32, stride=2)
        return index_output_1


class Index245(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=0, stop=32, stride=2)
        return index_output_1


class Index246(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=96, stop=112, stride=1)
        return index_output_1


class Index247(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=2048, stride=1)
        return index_output_1


class Index248(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=2048, stop=4096, stride=1)
        return index_output_1


class Index249(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=0, stop=6, stride=1)
        return index_output_1


class Index250(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=64, stop=80, stride=1)
        return index_output_1


class Index251(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=80, stop=96, stride=1)
        return index_output_1


class Index252(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=4, stop=5, stride=1)
        return index_output_1


class Index253(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=5, stop=6, stride=1)
        return index_output_1


class Index254(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=18, stride=1)
        return index_output_1


class Index255(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=18, stop=54, stride=1)
        return index_output_1


class Index256(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=54, stop=126, stride=1)
        return index_output_1


class Index257(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=72, stride=1)
        return index_output_1


class Index258(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=72, stop=108, stride=1)
        return index_output_1


class Index259(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=108, stop=126, stride=1)
        return index_output_1


class Index260(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=36, stride=1)
        return index_output_1


class Index261(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=36, stop=54, stride=1)
        return index_output_1


class Index262(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=54, stop=72, stride=1)
        return index_output_1


class Index263(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=18, stop=36, stride=1)
        return index_output_1


class Index264(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=36, stop=72, stride=1)
        return index_output_1


class Index265(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=3072, stride=1)
        return index_output_1


class Index266(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=3072, stop=6144, stride=1)
        return index_output_1


class Index267(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=96, stride=1)
        return index_output_1


class Index268(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=96, stop=112, stride=1)
        return index_output_1


class Index269(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=112, stop=128, stride=1)
        return index_output_1


class Index270(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=255, stop=256, stride=1)
        return index_output_1


class Index271(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=40, stride=1)
        return index_output_1


class Index272(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=40, stop=120, stride=1)
        return index_output_1


class Index273(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=120, stop=280, stride=1)
        return index_output_1


class Index274(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=160, stop=240, stride=1)
        return index_output_1


class Index275(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=240, stop=280, stride=1)
        return index_output_1


class Index276(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=80, stride=1)
        return index_output_1


class Index277(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=80, stop=120, stride=1)
        return index_output_1


class Index278(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=120, stop=160, stride=1)
        return index_output_1


class Index279(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=40, stop=80, stride=1)
        return index_output_1


class Index280(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=80, stop=160, stride=1)
        return index_output_1


class Index281(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=12, stop=24, stride=1)
        return index_output_1


class Index282(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=0, stop=12, stride=1)
        return index_output_1


class Index283(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=36, stop=48, stride=1)
        return index_output_1


class Index284(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=24, stop=36, stride=1)
        return index_output_1


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (
        Index0,
        [((1, 13, 384), torch.float32)],
        {
            "model_names": ["onnx_minilm_sentence_transformers_all_minilm_l6_v2_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index1,
        [((2, 1, 14), torch.float32)],
        {
            "model_names": ["pd_bert_bert_base_japanese_qa_padlenlp"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index2,
        [((2, 1, 14), torch.float32)],
        {
            "model_names": ["pd_bert_bert_base_japanese_qa_padlenlp"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index3,
        [((1, 512), torch.int64)],
        {
            "model_names": [
                "pt_albert_base_v1_mlm_hf",
                "pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf",
                "pt_albert_base_v1_token_cls_hf",
                "pt_albert_base_v2_mlm_hf",
                "pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder",
                "pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf",
                "pt_albert_large_v1_token_cls_hf",
                "pt_albert_large_v2_mlm_hf",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_albert_large_v2_token_cls_hf",
                "pt_albert_xlarge_v1_mlm_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
                "pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder",
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
                "pt_albert_xxlarge_v2_token_cls_hf",
                "pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf",
                "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "pt_albert_base_v2_token_cls_hf",
                "pt_albert_large_v1_mlm_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index4,
        [((1, 512), torch.int64)],
        {
            "model_names": ["pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "9", "stride": "1"},
        },
    ),
    (
        Index5,
        [((1, 512), torch.int64)],
        {
            "model_names": [
                "pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf",
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                "pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "384", "stride": "1"},
        },
    ),
    (
        Index6,
        [((1, 512), torch.int64)],
        {
            "model_names": ["pd_blip_text_salesforce_blip_image_captioning_base_text_enc_padlenlp"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "8", "stride": "1"},
        },
    ),
    (
        Index7,
        [((1, 512), torch.int64)],
        {
            "model_names": ["pt_albert_twmkn9_albert_base_v2_squad2_qa_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "14", "stride": "1"},
        },
    ),
    (
        Index0,
        [((1, 9, 768), torch.float32)],
        {
            "model_names": ["pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf", "pd_ernie_1_0_seq_cls_padlenlp"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index0,
        [((1, 128, 768), torch.float32)],
        {
            "model_names": [
                "pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder",
                "pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder",
                "pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf",
                "pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index0,
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
            "args": {"dim": "-2", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index8,
        [((1, 197, 768), torch.float32)],
        {
            "model_names": ["pt_beit_microsoft_beit_base_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "1", "stop": "197", "stride": "1"},
        },
    ),
    (
        Index9,
        [((1, 8, 207, 256), torch.float32)],
        {
            "model_names": ["pt_gemma_google_gemma_2_2b_it_qa_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "128", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index3,
        [((1, 8, 207, 256), torch.float32)],
        {
            "model_names": ["pt_gemma_google_gemma_2_2b_it_qa_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index9,
        [((1, 4, 207, 256), torch.float32)],
        {
            "model_names": ["pt_gemma_google_gemma_2_2b_it_qa_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "128", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index3,
        [((1, 4, 207, 256), torch.float32)],
        {
            "model_names": ["pt_gemma_google_gemma_2_2b_it_qa_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index1,
        [((1, 3, 224, 224), torch.float32)],
        {
            "model_names": ["pt_googlenet_base_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index2,
        [((1, 3, 224, 224), torch.float32)],
        {
            "model_names": ["pt_googlenet_base_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index10,
        [((1, 3, 224, 224), torch.float32)],
        {
            "model_names": ["pt_googlenet_base_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "2", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index11,
        [((1, 176, 28, 28), torch.float32)],
        {
            "model_names": ["pt_googlenet_base_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index12,
        [((1, 176, 28, 28), torch.float32)],
        {
            "model_names": ["pt_googlenet_base_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "64", "stop": "160", "stride": "1"},
        },
    ),
    (
        Index13,
        [((1, 176, 28, 28), torch.float32)],
        {
            "model_names": ["pt_googlenet_base_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "160", "stop": "176", "stride": "1"},
        },
    ),
    (
        Index14,
        [((1, 176, 28, 28), torch.float32)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w44_pose_estimation_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "88", "stride": "1"},
        },
    ),
    (
        Index15,
        [((1, 176, 28, 28), torch.float32)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w44_pose_estimation_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "88", "stop": "132", "stride": "1"},
        },
    ),
    (
        Index16,
        [((1, 176, 28, 28), torch.float32)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w44_pose_estimation_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "132", "stop": "176", "stride": "1"},
        },
    ),
    (
        Index17,
        [((1, 176, 28, 28), torch.float32)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w44_pose_estimation_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "44", "stride": "1"},
        },
    ),
    (
        Index18,
        [((1, 176, 28, 28), torch.float32)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w44_pose_estimation_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "44", "stop": "88", "stride": "1"},
        },
    ),
    (
        Index19,
        [((1, 176, 28, 28), torch.float32)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w44_pose_estimation_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "88", "stop": "176", "stride": "1"},
        },
    ),
    (
        Index20,
        [((1, 288, 28, 28), torch.float32)],
        {
            "model_names": ["pt_googlenet_base_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index21,
        [((1, 288, 28, 28), torch.float32)],
        {
            "model_names": ["pt_googlenet_base_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "128", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index22,
        [((1, 288, 28, 28), torch.float32)],
        {
            "model_names": ["pt_googlenet_base_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "256", "stop": "288", "stride": "1"},
        },
    ),
    (
        Index23,
        [((1, 304, 14, 14), torch.float32)],
        {
            "model_names": ["pt_googlenet_base_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "192", "stride": "1"},
        },
    ),
    (
        Index24,
        [((1, 304, 14, 14), torch.float32)],
        {
            "model_names": ["pt_googlenet_base_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "192", "stop": "288", "stride": "1"},
        },
    ),
    (
        Index25,
        [((1, 304, 14, 14), torch.float32)],
        {
            "model_names": ["pt_googlenet_base_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "288", "stop": "304", "stride": "1"},
        },
    ),
    (
        Index26,
        [((1, 296, 14, 14), torch.float32)],
        {
            "model_names": ["pt_googlenet_base_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "160", "stride": "1"},
        },
    ),
    (
        Index27,
        [((1, 296, 14, 14), torch.float32)],
        {
            "model_names": ["pt_googlenet_base_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "160", "stop": "272", "stride": "1"},
        },
    ),
    (
        Index28,
        [((1, 296, 14, 14), torch.float32)],
        {
            "model_names": ["pt_googlenet_base_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "272", "stop": "296", "stride": "1"},
        },
    ),
    (
        Index20,
        [((1, 280, 14, 14), torch.float32)],
        {
            "model_names": ["pt_googlenet_base_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index21,
        [((1, 280, 14, 14), torch.float32)],
        {
            "model_names": ["pt_googlenet_base_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "128", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index29,
        [((1, 280, 14, 14), torch.float32)],
        {
            "model_names": ["pt_googlenet_base_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "256", "stop": "280", "stride": "1"},
        },
    ),
    (
        Index30,
        [((1, 288, 14, 14), torch.float32)],
        {
            "model_names": ["pt_googlenet_base_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "112", "stride": "1"},
        },
    ),
    (
        Index31,
        [((1, 288, 14, 14), torch.float32)],
        {
            "model_names": ["pt_googlenet_base_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "112", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index22,
        [((1, 288, 14, 14), torch.float32)],
        {
            "model_names": ["pt_googlenet_base_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "256", "stop": "288", "stride": "1"},
        },
    ),
    (
        Index32,
        [((1, 448, 14, 14), torch.float32)],
        {
            "model_names": ["pt_googlenet_base_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index33,
        [((1, 448, 14, 14), torch.float32)],
        {
            "model_names": ["pt_googlenet_base_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "256", "stop": "416", "stride": "1"},
        },
    ),
    (
        Index34,
        [((1, 448, 14, 14), torch.float32)],
        {
            "model_names": ["pt_googlenet_base_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "416", "stop": "448", "stride": "1"},
        },
    ),
    (
        Index32,
        [((1, 448, 7, 7), torch.float32)],
        {
            "model_names": ["pt_googlenet_base_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index33,
        [((1, 448, 7, 7), torch.float32)],
        {
            "model_names": ["pt_googlenet_base_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "256", "stop": "416", "stride": "1"},
        },
    ),
    (
        Index34,
        [((1, 448, 7, 7), torch.float32)],
        {
            "model_names": ["pt_googlenet_base_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "416", "stop": "448", "stride": "1"},
        },
    ),
    (
        Index35,
        [((1, 624, 7, 7), torch.float32)],
        {
            "model_names": ["pt_googlenet_base_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "384", "stride": "1"},
        },
    ),
    (
        Index36,
        [((1, 624, 7, 7), torch.float32)],
        {
            "model_names": ["pt_googlenet_base_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "384", "stop": "576", "stride": "1"},
        },
    ),
    (
        Index37,
        [((1, 624, 7, 7), torch.float32)],
        {
            "model_names": ["pt_googlenet_base_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "576", "stop": "624", "stride": "1"},
        },
    ),
    (
        Index38,
        [((1, 210, 7, 7), torch.float32)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w30_pose_estimation_osmr", "pt_hrnet_hrnet_w30_pose_estimation_timm"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "30", "stride": "1"},
        },
    ),
    (
        Index39,
        [((1, 210, 7, 7), torch.float32)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w30_pose_estimation_osmr", "pt_hrnet_hrnet_w30_pose_estimation_timm"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "30", "stop": "90", "stride": "1"},
        },
    ),
    (
        Index40,
        [((1, 210, 7, 7), torch.float32)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w30_pose_estimation_osmr", "pt_hrnet_hrnet_w30_pose_estimation_timm"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "90", "stop": "210", "stride": "1"},
        },
    ),
    (
        Index41,
        [((1, 210, 7, 7), torch.float32)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w30_pose_estimation_osmr", "pt_hrnet_hrnet_w30_pose_estimation_timm"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "120", "stride": "1"},
        },
    ),
    (
        Index42,
        [((1, 210, 7, 7), torch.float32)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w30_pose_estimation_osmr", "pt_hrnet_hrnet_w30_pose_estimation_timm"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "120", "stop": "180", "stride": "1"},
        },
    ),
    (
        Index43,
        [((1, 210, 7, 7), torch.float32)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w30_pose_estimation_osmr", "pt_hrnet_hrnet_w30_pose_estimation_timm"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "180", "stop": "210", "stride": "1"},
        },
    ),
    (
        Index44,
        [((1, 120, 28, 28), torch.float32)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w30_pose_estimation_osmr", "pt_hrnet_hrnet_w30_pose_estimation_timm"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "60", "stride": "1"},
        },
    ),
    (
        Index45,
        [((1, 120, 28, 28), torch.float32)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w30_pose_estimation_osmr", "pt_hrnet_hrnet_w30_pose_estimation_timm"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "60", "stop": "90", "stride": "1"},
        },
    ),
    (
        Index46,
        [((1, 120, 28, 28), torch.float32)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w30_pose_estimation_osmr", "pt_hrnet_hrnet_w30_pose_estimation_timm"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "90", "stop": "120", "stride": "1"},
        },
    ),
    (
        Index38,
        [((1, 120, 28, 28), torch.float32)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w30_pose_estimation_osmr", "pt_hrnet_hrnet_w30_pose_estimation_timm"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "30", "stride": "1"},
        },
    ),
    (
        Index47,
        [((1, 120, 28, 28), torch.float32)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w30_pose_estimation_osmr", "pt_hrnet_hrnet_w30_pose_estimation_timm"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "30", "stop": "60", "stride": "1"},
        },
    ),
    (
        Index48,
        [((1, 120, 28, 28), torch.float32)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w30_pose_estimation_osmr", "pt_hrnet_hrnet_w30_pose_estimation_timm"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "60", "stop": "120", "stride": "1"},
        },
    ),
    (
        Index49,
        [((1, 224, 35, 35), torch.float32)],
        {
            "model_names": [
                "pt_inception_v4_img_cls_osmr",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_inception_inception_v4_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "96", "stride": "1"},
        },
    ),
    (
        Index50,
        [((1, 224, 35, 35), torch.float32)],
        {
            "model_names": [
                "pt_inception_v4_img_cls_osmr",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_inception_inception_v4_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "96", "stop": "160", "stride": "1"},
        },
    ),
    (
        Index51,
        [((1, 224, 35, 35), torch.float32)],
        {
            "model_names": [
                "pt_inception_v4_img_cls_osmr",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_inception_inception_v4_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "160", "stop": "224", "stride": "1"},
        },
    ),
    (
        Index35,
        [((1, 768, 17, 17), torch.float32)],
        {
            "model_names": [
                "pt_inception_v4_img_cls_osmr",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_inception_inception_v4_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "384", "stride": "1"},
        },
    ),
    (
        Index36,
        [((1, 768, 17, 17), torch.float32)],
        {
            "model_names": [
                "pt_inception_v4_img_cls_osmr",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_inception_inception_v4_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "384", "stop": "576", "stride": "1"},
        },
    ),
    (
        Index52,
        [((1, 768, 17, 17), torch.float32)],
        {
            "model_names": [
                "pt_inception_v4_img_cls_osmr",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_inception_inception_v4_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "576", "stop": "768", "stride": "1"},
        },
    ),
    (
        Index32,
        [((1, 1024, 8, 8), torch.float32)],
        {
            "model_names": [
                "pt_inception_v4_img_cls_osmr",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_inception_inception_v4_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index53,
        [((1, 1024, 8, 8), torch.float32)],
        {
            "model_names": [
                "pt_inception_v4_img_cls_osmr",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_inception_inception_v4_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "256", "stop": "640", "stride": "1"},
        },
    ),
    (
        Index54,
        [((1, 1024, 8, 8), torch.float32)],
        {
            "model_names": [
                "pt_inception_v4_img_cls_osmr",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_inception_inception_v4_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "640", "stop": "1024", "stride": "1"},
        },
    ),
    (
        Index55,
        [((2304, 768), torch.float32)],
        {
            "model_names": [
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
                "pt_gpt_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
                "pt_gpt_gpt2_text_gen_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "768", "stride": "1"},
        },
    ),
    (
        Index56,
        [((2304, 768), torch.float32)],
        {
            "model_names": [
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
                "pt_gpt_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
                "pt_gpt_gpt2_text_gen_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "768", "stop": "1536", "stride": "1"},
        },
    ),
    (
        Index57,
        [((2304, 768), torch.float32)],
        {
            "model_names": [
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
                "pt_gpt_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
                "pt_gpt_gpt2_text_gen_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "1536", "stop": "2304", "stride": "1"},
        },
    ),
    (
        Index58,
        [((2304,), torch.float32)],
        {
            "model_names": [
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
                "pt_gpt_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
                "pt_gpt_gpt2_text_gen_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "768", "stride": "1"},
        },
    ),
    (
        Index59,
        [((2304,), torch.float32)],
        {
            "model_names": [
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
                "pt_gpt_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
                "pt_gpt_gpt2_text_gen_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "768", "stop": "1536", "stride": "1"},
        },
    ),
    (
        Index60,
        [((2304,), torch.float32)],
        {
            "model_names": [
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
                "pt_gpt_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
                "pt_gpt_gpt2_text_gen_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "1536", "stop": "2304", "stride": "1"},
        },
    ),
    (
        Index61,
        [((1, 1, 1024, 1024), torch.bool)],
        {
            "model_names": [
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
                "pt_gpt_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "7", "stride": "1"},
        },
    ),
    (
        Index62,
        [((1, 1, 1024, 1024), torch.bool)],
        {
            "model_names": ["pt_gpt_gpt2_text_gen_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index63,
        [((1, 1, 7, 1024), torch.bool)],
        {
            "model_names": [
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
                "pt_gpt_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "7", "stride": "1"},
        },
    ),
    (
        Index0,
        [((2, 512), torch.float32)],
        {
            "model_names": ["pt_opt_facebook_opt_350m_qa_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index64,
        [((2, 512), torch.float32)],
        {
            "model_names": ["pt_opt_facebook_opt_350m_qa_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index65,
        [((2,), torch.float32)],
        {
            "model_names": [
                "pt_opt_facebook_opt_350m_qa_hf",
                "pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf",
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                "pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf",
                "pt_opt_facebook_opt_1_3b_qa_hf",
                "pt_albert_twmkn9_albert_base_v2_squad2_qa_hf",
                "pt_opt_facebook_opt_125m_qa_hf",
                "onnx_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index66,
        [((2,), torch.float32)],
        {
            "model_names": [
                "pt_opt_facebook_opt_350m_qa_hf",
                "pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf",
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                "pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf",
                "pt_opt_facebook_opt_1_3b_qa_hf",
                "pt_albert_twmkn9_albert_base_v2_squad2_qa_hf",
                "pt_opt_facebook_opt_125m_qa_hf",
                "onnx_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index67,
        [((1, 32, 12, 64), torch.float32)],
        {
            "model_names": ["pt_phi1_microsoft_phi_1_token_cls_hf", "pt_phi_1_5_microsoft_phi_1_5_token_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index68,
        [((1, 32, 12, 64), torch.float32)],
        {
            "model_names": ["pt_phi1_microsoft_phi_1_token_cls_hf", "pt_phi_1_5_microsoft_phi_1_5_token_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "32", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index69,
        [((1, 32, 12, 32), torch.float32)],
        {
            "model_names": ["pt_phi1_microsoft_phi_1_token_cls_hf", "pt_phi_1_5_microsoft_phi_1_5_token_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "16", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index70,
        [((1, 32, 12, 32), torch.float32)],
        {
            "model_names": ["pt_phi1_microsoft_phi_1_token_cls_hf", "pt_phi_1_5_microsoft_phi_1_5_token_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "16", "stride": "1"},
        },
    ),
    (
        Index71,
        [((1, 12, 35, 128), torch.float32)],
        {
            "model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "64", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index72,
        [((1, 12, 35, 128), torch.float32)],
        {
            "model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index71,
        [((1, 2, 35, 128), torch.float32)],
        {
            "model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "64", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index72,
        [((1, 2, 35, 128), torch.float32)],
        {
            "model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index73,
        [((3, 64, 3, 64, 32), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index74,
        [((3, 64, 3, 64, 32), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index75,
        [((3, 64, 3, 64, 32), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "2", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index76,
        [((1, 64, 64, 96), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "4", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index77,
        [((1, 64, 64, 96), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "4", "stride": "1"},
        },
    ),
    (
        Index78,
        [((1, 64, 64, 96), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "4", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index79,
        [((1, 64, 64, 96), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "4", "stride": "1"},
        },
    ),
    (
        Index80,
        [((1, 64, 64, 96), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "60", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index44,
        [((1, 64, 64, 96), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "60", "stride": "1"},
        },
    ),
    (
        Index81,
        [((1, 64, 64, 96), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "60", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index82,
        [((1, 64, 64, 96), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "60", "stride": "1"},
        },
    ),
    (
        Index83,
        [((1, 64, 64, 96), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "64", "stride": "2"},
        },
    ),
    (
        Index84,
        [((1, 64, 64, 96), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "1", "stop": "64", "stride": "2"},
        },
    ),
    (
        Index85,
        [((1, 64, 64, 96), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "-4", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index86,
        [((1, 64, 64, 96), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "-4", "stride": "1"},
        },
    ),
    (
        Index87,
        [((1, 64, 64, 96), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "-4", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index88,
        [((1, 64, 64, 96), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "-4", "stride": "1"},
        },
    ),
    (
        Index89,
        [((1, 32, 64, 96), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "64", "stride": "2"},
        },
    ),
    (
        Index90,
        [((1, 32, 64, 96), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "1", "stop": "64", "stride": "2"},
        },
    ),
    (
        Index73,
        [((3, 16, 6, 64, 32), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index74,
        [((3, 16, 6, 64, 32), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index75,
        [((3, 16, 6, 64, 32), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "2", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index91,
        [((1, 32, 32, 192), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "4", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index77,
        [((1, 32, 32, 192), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "4", "stride": "1"},
        },
    ),
    (
        Index92,
        [((1, 32, 32, 192), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "4", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index79,
        [((1, 32, 32, 192), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "4", "stride": "1"},
        },
    ),
    (
        Index93,
        [((1, 32, 32, 192), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "28", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index94,
        [((1, 32, 32, 192), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "28", "stride": "1"},
        },
    ),
    (
        Index95,
        [((1, 32, 32, 192), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "28", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index96,
        [((1, 32, 32, 192), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "28", "stride": "1"},
        },
    ),
    (
        Index97,
        [((1, 32, 32, 192), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "32", "stride": "2"},
        },
    ),
    (
        Index98,
        [((1, 32, 32, 192), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "1", "stop": "32", "stride": "2"},
        },
    ),
    (
        Index99,
        [((1, 32, 32, 192), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "-4", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index86,
        [((1, 32, 32, 192), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "-4", "stride": "1"},
        },
    ),
    (
        Index100,
        [((1, 32, 32, 192), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "-4", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index88,
        [((1, 32, 32, 192), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "-4", "stride": "1"},
        },
    ),
    (
        Index101,
        [((1, 16, 32, 192), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "32", "stride": "2"},
        },
    ),
    (
        Index102,
        [((1, 16, 32, 192), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "1", "stop": "32", "stride": "2"},
        },
    ),
    (
        Index73,
        [((3, 4, 12, 64, 32), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index74,
        [((3, 4, 12, 64, 32), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index75,
        [((3, 4, 12, 64, 32), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "2", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index103,
        [((1, 16, 16, 384), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "4", "stop": "16", "stride": "1"},
        },
    ),
    (
        Index77,
        [((1, 16, 16, 384), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "4", "stride": "1"},
        },
    ),
    (
        Index104,
        [((1, 16, 16, 384), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "4", "stop": "16", "stride": "1"},
        },
    ),
    (
        Index79,
        [((1, 16, 16, 384), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "4", "stride": "1"},
        },
    ),
    (
        Index105,
        [((1, 16, 16, 384), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "12", "stop": "16", "stride": "1"},
        },
    ),
    (
        Index106,
        [((1, 16, 16, 384), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "12", "stride": "1"},
        },
    ),
    (
        Index107,
        [((1, 16, 16, 384), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "12", "stop": "16", "stride": "1"},
        },
    ),
    (
        Index108,
        [((1, 16, 16, 384), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "12", "stride": "1"},
        },
    ),
    (
        Index109,
        [((1, 16, 16, 384), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "16", "stride": "2"},
        },
    ),
    (
        Index110,
        [((1, 16, 16, 384), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "1", "stop": "16", "stride": "2"},
        },
    ),
    (
        Index111,
        [((1, 16, 16, 384), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "-4", "stop": "16", "stride": "1"},
        },
    ),
    (
        Index86,
        [((1, 16, 16, 384), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "-4", "stride": "1"},
        },
    ),
    (
        Index112,
        [((1, 16, 16, 384), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "-4", "stop": "16", "stride": "1"},
        },
    ),
    (
        Index88,
        [((1, 16, 16, 384), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "-4", "stride": "1"},
        },
    ),
    (
        Index113,
        [((1, 8, 16, 384), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "16", "stride": "2"},
        },
    ),
    (
        Index114,
        [((1, 8, 16, 384), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "1", "stop": "16", "stride": "2"},
        },
    ),
    (
        Index73,
        [((3, 1, 24, 64, 32), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index74,
        [((3, 1, 24, 64, 32), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index75,
        [((3, 1, 24, 64, 32), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "2", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index0,
        [((1, 197, 1024), torch.float32)],
        {
            "model_names": ["pt_vit_google_vit_large_patch16_224_img_cls_hf", "pt_vit_vit_l_16_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index8,
        [((1, 197, 1024), torch.float32)],
        {
            "model_names": ["pt_beit_microsoft_beit_large_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "1", "stop": "197", "stride": "1"},
        },
    ),
    (
        Index115,
        [((1, 5880, 4), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v6_yolov6n_obj_det_torchhub",
                "pt_yolo_v6_yolov6s_obj_det_torchhub",
                "pt_yolo_v6_yolov6l_obj_det_torchhub",
                "pt_yolo_v6_yolov6m_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index116,
        [((1, 5880, 4), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v6_yolov6n_obj_det_torchhub",
                "pt_yolo_v6_yolov6s_obj_det_torchhub",
                "pt_yolo_v6_yolov6l_obj_det_torchhub",
                "pt_yolo_v6_yolov6m_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "2", "stop": "4", "stride": "1"},
        },
    ),
    (
        Index117,
        [((1, 3, 416, 416), torch.float32)],
        {
            "model_names": ["pt_yolox_yolox_tiny_obj_det_torchhub", "pt_yolox_yolox_nano_obj_det_torchhub"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "416", "stride": "2"},
        },
    ),
    (
        Index118,
        [((1, 3, 416, 416), torch.float32)],
        {
            "model_names": ["pt_yolox_yolox_tiny_obj_det_torchhub", "pt_yolox_yolox_nano_obj_det_torchhub"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "1", "stop": "416", "stride": "2"},
        },
    ),
    (
        Index119,
        [((1, 3, 208, 416), torch.float32)],
        {
            "model_names": ["pt_yolox_yolox_tiny_obj_det_torchhub", "pt_yolox_yolox_nano_obj_det_torchhub"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "416", "stride": "2"},
        },
    ),
    (
        Index120,
        [((1, 3, 208, 416), torch.float32)],
        {
            "model_names": ["pt_yolox_yolox_tiny_obj_det_torchhub", "pt_yolox_yolox_nano_obj_det_torchhub"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "1", "stop": "416", "stride": "2"},
        },
    ),
    (
        Index1,
        [((2, 1, 11), torch.float32)],
        {
            "model_names": ["pd_bert_chinese_roberta_base_qa_padlenlp"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index2,
        [((2, 1, 11), torch.float32)],
        {
            "model_names": ["pd_bert_chinese_roberta_base_qa_padlenlp"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index121,
        [((1, 224, 7, 7), torch.float32)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w32_pose_estimation_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index122,
        [((1, 224, 7, 7), torch.float32)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w32_pose_estimation_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "32", "stop": "96", "stride": "1"},
        },
    ),
    (
        Index123,
        [((1, 224, 7, 7), torch.float32)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w32_pose_estimation_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "96", "stop": "224", "stride": "1"},
        },
    ),
    (
        Index20,
        [((1, 224, 7, 7), torch.float32)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w32_pose_estimation_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index124,
        [((1, 224, 7, 7), torch.float32)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w32_pose_estimation_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "128", "stop": "192", "stride": "1"},
        },
    ),
    (
        Index125,
        [((1, 224, 7, 7), torch.float32)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w32_pose_estimation_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "192", "stop": "224", "stride": "1"},
        },
    ),
    (
        Index11,
        [((1, 128, 28, 28), torch.float32)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w32_pose_estimation_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index126,
        [((1, 128, 28, 28), torch.float32)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w32_pose_estimation_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "64", "stop": "96", "stride": "1"},
        },
    ),
    (
        Index127,
        [((1, 128, 28, 28), torch.float32)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w32_pose_estimation_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "96", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index121,
        [((1, 128, 28, 28), torch.float32)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w32_pose_estimation_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index128,
        [((1, 128, 28, 28), torch.float32)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w32_pose_estimation_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "32", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index129,
        [((1, 128, 28, 28), torch.float32)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w32_pose_estimation_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "64", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index130,
        [((1, 1, 1024, 72), torch.float32)],
        {
            "model_names": [
                "pt_nbeats_generic_basis_clm_hf",
                "pt_nbeats_trend_basis_clm_hf",
                "pt_nbeats_seasionality_basis_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "-1", "stop": "72", "stride": "1"},
        },
    ),
    (
        Index131,
        [((1024, 96), torch.float32)],
        {
            "model_names": ["pt_nbeats_generic_basis_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "-24", "stop": "96", "stride": "1"},
        },
    ),
    (
        Index132,
        [((1024, 96), torch.float32)],
        {
            "model_names": ["pt_nbeats_generic_basis_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "72", "stride": "1"},
        },
    ),
    (
        Index133,
        [((3, 197, 1, 768), torch.float32)],
        {
            "model_names": ["pt_vit_vit_b_16_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index134,
        [((3, 197, 1, 768), torch.float32)],
        {
            "model_names": ["pt_vit_vit_b_16_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index135,
        [((3, 197, 1, 768), torch.float32)],
        {
            "model_names": ["pt_vit_vit_b_16_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "2", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index136,
        [((1, 3, 640, 640), torch.float32)],
        {
            "model_names": [
                "pt_yolox_yolox_x_obj_det_torchhub",
                "pt_yolox_yolox_m_obj_det_torchhub",
                "pt_yolox_yolox_s_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "640", "stride": "2"},
        },
    ),
    (
        Index137,
        [((1, 3, 640, 640), torch.float32)],
        {
            "model_names": [
                "pt_yolox_yolox_x_obj_det_torchhub",
                "pt_yolox_yolox_m_obj_det_torchhub",
                "pt_yolox_yolox_s_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "1", "stop": "640", "stride": "2"},
        },
    ),
    (
        Index138,
        [((1, 3, 320, 640), torch.float32)],
        {
            "model_names": [
                "pt_yolox_yolox_x_obj_det_torchhub",
                "pt_yolox_yolox_m_obj_det_torchhub",
                "pt_yolox_yolox_s_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "640", "stride": "2"},
        },
    ),
    (
        Index139,
        [((1, 3, 320, 640), torch.float32)],
        {
            "model_names": [
                "pt_yolox_yolox_x_obj_det_torchhub",
                "pt_yolox_yolox_m_obj_det_torchhub",
                "pt_yolox_yolox_s_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "1", "stop": "640", "stride": "2"},
        },
    ),
    (
        Index140,
        [((732, 16), torch.float32)],
        {
            "model_names": ["pt_beit_microsoft_beit_large_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "729", "stride": "1"},
        },
    ),
    (
        Index141,
        [((732, 16), torch.float32)],
        {
            "model_names": ["pt_beit_microsoft_beit_large_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "729", "stop": "732", "stride": "1"},
        },
    ),
    (
        Index0,
        [((2, 1024), torch.float32)],
        {
            "model_names": [
                "pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf",
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                "onnx_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index64,
        [((2, 1024), torch.float32)],
        {
            "model_names": [
                "pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf",
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                "onnx_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index0,
        [((1, 197, 192), torch.float32)],
        {
            "model_names": ["pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index62,
        [((1, 1, 2048, 2048), torch.bool)],
        {
            "model_names": ["pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf", "pt_gptneo_eleutherai_gpt_neo_125m_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index142,
        [((1, 1, 256, 2048), torch.bool)],
        {
            "model_names": ["pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf", "pt_gptneo_eleutherai_gpt_neo_125m_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index17,
        [((1, 308, 7, 7), torch.float32)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w44_pose_estimation_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "44", "stride": "1"},
        },
    ),
    (
        Index143,
        [((1, 308, 7, 7), torch.float32)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w44_pose_estimation_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "44", "stop": "132", "stride": "1"},
        },
    ),
    (
        Index144,
        [((1, 308, 7, 7), torch.float32)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w44_pose_estimation_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "132", "stop": "308", "stride": "1"},
        },
    ),
    (
        Index145,
        [((1, 308, 7, 7), torch.float32)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w44_pose_estimation_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "176", "stride": "1"},
        },
    ),
    (
        Index146,
        [((1, 308, 7, 7), torch.float32)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w44_pose_estimation_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "176", "stop": "264", "stride": "1"},
        },
    ),
    (
        Index147,
        [((1, 308, 7, 7), torch.float32)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w44_pose_estimation_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "264", "stop": "308", "stride": "1"},
        },
    ),
    (
        Index148,
        [((1024, 8), torch.float32)],
        {
            "model_names": ["pt_nbeats_trend_basis_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "4", "stride": "1"},
        },
    ),
    (
        Index149,
        [((1024, 8), torch.float32)],
        {
            "model_names": ["pt_nbeats_trend_basis_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "4", "stop": "8", "stride": "1"},
        },
    ),
    (
        Index68,
        [((1, 16, 6, 64), torch.float32)],
        {
            "model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "32", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index67,
        [((1, 16, 6, 64), torch.float32)],
        {
            "model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index3,
        [((1, 514), torch.int64)],
        {
            "model_names": [
                "pt_roberta_xlm_roberta_base_mlm_hf",
                "pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index150,
        [((1, 56, 56, 96), torch.float32)],
        {
            "model_names": [
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "3", "stop": "56", "stride": "1"},
        },
    ),
    (
        Index151,
        [((1, 56, 56, 96), torch.float32)],
        {
            "model_names": [
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index152,
        [((1, 56, 56, 96), torch.float32)],
        {
            "model_names": [
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "3", "stop": "56", "stride": "1"},
        },
    ),
    (
        Index153,
        [((1, 56, 56, 96), torch.float32)],
        {
            "model_names": [
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index154,
        [((1, 56, 56, 96), torch.float32)],
        {
            "model_names": [
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "53", "stop": "56", "stride": "1"},
        },
    ),
    (
        Index155,
        [((1, 56, 56, 96), torch.float32)],
        {
            "model_names": [
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "53", "stride": "1"},
        },
    ),
    (
        Index156,
        [((1, 56, 56, 96), torch.float32)],
        {
            "model_names": [
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "53", "stop": "56", "stride": "1"},
        },
    ),
    (
        Index157,
        [((1, 56, 56, 96), torch.float32)],
        {
            "model_names": [
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "53", "stride": "1"},
        },
    ),
    (
        Index158,
        [((1, 56, 56, 96), torch.float32)],
        {
            "model_names": [
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "56", "stride": "2"},
        },
    ),
    (
        Index159,
        [((1, 56, 56, 96), torch.float32)],
        {
            "model_names": [
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "1", "stop": "56", "stride": "2"},
        },
    ),
    (
        Index160,
        [((1, 28, 56, 96), torch.float32)],
        {
            "model_names": [
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "56", "stride": "2"},
        },
    ),
    (
        Index161,
        [((1, 28, 56, 96), torch.float32)],
        {
            "model_names": [
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "1", "stop": "56", "stride": "2"},
        },
    ),
    (
        Index162,
        [((1, 28, 28, 192), torch.float32)],
        {
            "model_names": [
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "3", "stop": "28", "stride": "1"},
        },
    ),
    (
        Index151,
        [((1, 28, 28, 192), torch.float32)],
        {
            "model_names": [
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index163,
        [((1, 28, 28, 192), torch.float32)],
        {
            "model_names": [
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "3", "stop": "28", "stride": "1"},
        },
    ),
    (
        Index153,
        [((1, 28, 28, 192), torch.float32)],
        {
            "model_names": [
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index164,
        [((1, 28, 28, 192), torch.float32)],
        {
            "model_names": [
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "25", "stop": "28", "stride": "1"},
        },
    ),
    (
        Index165,
        [((1, 28, 28, 192), torch.float32)],
        {
            "model_names": [
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "25", "stride": "1"},
        },
    ),
    (
        Index166,
        [((1, 28, 28, 192), torch.float32)],
        {
            "model_names": [
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "25", "stop": "28", "stride": "1"},
        },
    ),
    (
        Index167,
        [((1, 28, 28, 192), torch.float32)],
        {
            "model_names": [
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "25", "stride": "1"},
        },
    ),
    (
        Index168,
        [((1, 28, 28, 192), torch.float32)],
        {
            "model_names": [
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "28", "stride": "2"},
        },
    ),
    (
        Index169,
        [((1, 28, 28, 192), torch.float32)],
        {
            "model_names": [
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "1", "stop": "28", "stride": "2"},
        },
    ),
    (
        Index170,
        [((1, 14, 28, 192), torch.float32)],
        {
            "model_names": [
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "28", "stride": "2"},
        },
    ),
    (
        Index171,
        [((1, 14, 28, 192), torch.float32)],
        {
            "model_names": [
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "1", "stop": "28", "stride": "2"},
        },
    ),
    (
        Index172,
        [((1, 14, 14, 384), torch.float32)],
        {
            "model_names": [
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "3", "stop": "14", "stride": "1"},
        },
    ),
    (
        Index151,
        [((1, 14, 14, 384), torch.float32)],
        {
            "model_names": [
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index173,
        [((1, 14, 14, 384), torch.float32)],
        {
            "model_names": [
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "3", "stop": "14", "stride": "1"},
        },
    ),
    (
        Index153,
        [((1, 14, 14, 384), torch.float32)],
        {
            "model_names": [
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index174,
        [((1, 14, 14, 384), torch.float32)],
        {
            "model_names": [
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "11", "stop": "14", "stride": "1"},
        },
    ),
    (
        Index175,
        [((1, 14, 14, 384), torch.float32)],
        {
            "model_names": [
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "11", "stride": "1"},
        },
    ),
    (
        Index176,
        [((1, 14, 14, 384), torch.float32)],
        {
            "model_names": [
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "11", "stop": "14", "stride": "1"},
        },
    ),
    (
        Index177,
        [((1, 14, 14, 384), torch.float32)],
        {
            "model_names": [
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "11", "stride": "1"},
        },
    ),
    (
        Index178,
        [((1, 14, 14, 384), torch.float32)],
        {
            "model_names": [
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "14", "stride": "2"},
        },
    ),
    (
        Index179,
        [((1, 14, 14, 384), torch.float32)],
        {
            "model_names": [
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "1", "stop": "14", "stride": "2"},
        },
    ),
    (
        Index180,
        [((1, 7, 14, 384), torch.float32)],
        {
            "model_names": [
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "14", "stride": "2"},
        },
    ),
    (
        Index181,
        [((1, 7, 14, 384), torch.float32)],
        {
            "model_names": [
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "1", "stop": "14", "stride": "2"},
        },
    ),
    (
        Index133,
        [((3, 1370, 1, 1280), torch.float32)],
        {
            "model_names": ["pt_vit_vit_h_14_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index134,
        [((3, 1370, 1, 1280), torch.float32)],
        {
            "model_names": ["pt_vit_vit_h_14_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index135,
        [((3, 1370, 1, 1280), torch.float32)],
        {
            "model_names": ["pt_vit_vit_h_14_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "2", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index0,
        [((1, 1370, 1280), torch.float32)],
        {
            "model_names": ["pt_vit_vit_h_14_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index0,
        [((1, 4251, 192), torch.float32)],
        {
            "model_names": ["pt_yolos_hustvl_yolos_tiny_obj_det_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index182,
        [((1, 4251, 192), torch.float32)],
        {
            "model_names": ["pt_yolos_hustvl_yolos_tiny_obj_det_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "1", "stop": "-100", "stride": "1"},
        },
    ),
    (
        Index183,
        [((1, 4251, 192), torch.float32)],
        {
            "model_names": ["pt_yolos_hustvl_yolos_tiny_obj_det_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "-100", "stop": "4251", "stride": "1"},
        },
    ),
    (
        Index184,
        [((1, 1445, 192), torch.float32)],
        {
            "model_names": ["pt_yolos_hustvl_yolos_tiny_obj_det_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "-100", "stop": "1445", "stride": "1"},
        },
    ),
    (
        Index185,
        [((1, 32, 160, 160), torch.float32)],
        {
            "model_names": [
                "pt_yolov8_default_obj_det_github",
                "onnx_yolov10_default_obj_det_github",
                "onnx_yolov8_default_obj_det_github",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "16", "stride": "1"},
        },
    ),
    (
        Index186,
        [((1, 32, 160, 160), torch.float32)],
        {
            "model_names": [
                "pt_yolov8_default_obj_det_github",
                "onnx_yolov10_default_obj_det_github",
                "onnx_yolov8_default_obj_det_github",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "16", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index121,
        [((1, 64, 80, 80), torch.float32)],
        {
            "model_names": [
                "pt_yolov8_default_obj_det_github",
                "onnx_yolov10_default_obj_det_github",
                "onnx_yolov8_default_obj_det_github",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index128,
        [((1, 64, 80, 80), torch.float32)],
        {
            "model_names": [
                "pt_yolov8_default_obj_det_github",
                "onnx_yolov10_default_obj_det_github",
                "onnx_yolov8_default_obj_det_github",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "32", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index11,
        [((1, 128, 40, 40), torch.float32)],
        {
            "model_names": [
                "pt_yolov8_default_obj_det_github",
                "onnx_yolov10_default_obj_det_github",
                "onnx_yolov8_default_obj_det_github",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index129,
        [((1, 128, 40, 40), torch.float32)],
        {
            "model_names": [
                "pt_yolov8_default_obj_det_github",
                "onnx_yolov10_default_obj_det_github",
                "onnx_yolov8_default_obj_det_github",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "64", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index20,
        [((1, 256, 20, 20), torch.float32)],
        {
            "model_names": [
                "pt_yolov8_default_obj_det_github",
                "onnx_yolov10_default_obj_det_github",
                "onnx_yolov8_default_obj_det_github",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index21,
        [((1, 256, 20, 20), torch.float32)],
        {
            "model_names": [
                "pt_yolov8_default_obj_det_github",
                "onnx_yolov10_default_obj_det_github",
                "onnx_yolov8_default_obj_det_github",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "128", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index187,
        [((1, 144, 8400), torch.float32)],
        {
            "model_names": [
                "pt_yolov8_default_obj_det_github",
                "onnx_yolov10_default_obj_det_github",
                "pt_yolov9_default_obj_det_github",
                "onnx_yolov8_default_obj_det_github",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index188,
        [((1, 144, 8400), torch.float32)],
        {
            "model_names": [
                "pt_yolov8_default_obj_det_github",
                "onnx_yolov10_default_obj_det_github",
                "pt_yolov9_default_obj_det_github",
                "onnx_yolov8_default_obj_det_github",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "64", "stop": "144", "stride": "1"},
        },
    ),
    (
        Index189,
        [((1, 4, 8400), torch.float32)],
        {
            "model_names": [
                "pt_yolov8_default_obj_det_github",
                "onnx_yolov10_default_obj_det_github",
                "pt_yolov9_default_obj_det_github",
                "onnx_yolov8_default_obj_det_github",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index190,
        [((1, 4, 8400), torch.float32)],
        {
            "model_names": [
                "pt_yolov8_default_obj_det_github",
                "onnx_yolov10_default_obj_det_github",
                "pt_yolov9_default_obj_det_github",
                "onnx_yolov8_default_obj_det_github",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "2", "stop": "4", "stride": "1"},
        },
    ),
    (
        Index191,
        [((1, 2, 128, 400), torch.float32)],
        {
            "model_names": ["onnx_yolov10_default_obj_det_github"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "64", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index192,
        [((1, 2, 128, 400), torch.float32)],
        {
            "model_names": ["onnx_yolov10_default_obj_det_github"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index193,
        [((1, 2, 128, 400), torch.float32)],
        {
            "model_names": ["onnx_yolov10_default_obj_det_github"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "32", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index194,
        [((1, 336, 7, 7), torch.float32)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w48_pose_estimation_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "48", "stride": "1"},
        },
    ),
    (
        Index195,
        [((1, 336, 7, 7), torch.float32)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w48_pose_estimation_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "48", "stop": "144", "stride": "1"},
        },
    ),
    (
        Index196,
        [((1, 336, 7, 7), torch.float32)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w48_pose_estimation_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "144", "stop": "336", "stride": "1"},
        },
    ),
    (
        Index23,
        [((1, 336, 7, 7), torch.float32)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w48_pose_estimation_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "192", "stride": "1"},
        },
    ),
    (
        Index24,
        [((1, 336, 7, 7), torch.float32)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w48_pose_estimation_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "192", "stop": "288", "stride": "1"},
        },
    ),
    (
        Index197,
        [((1, 336, 7, 7), torch.float32)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w48_pose_estimation_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "288", "stop": "336", "stride": "1"},
        },
    ),
    (
        Index49,
        [((1, 192, 28, 28), torch.float32)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w48_pose_estimation_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "96", "stride": "1"},
        },
    ),
    (
        Index198,
        [((1, 192, 28, 28), torch.float32)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w48_pose_estimation_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "96", "stop": "144", "stride": "1"},
        },
    ),
    (
        Index199,
        [((1, 192, 28, 28), torch.float32)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w48_pose_estimation_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "144", "stop": "192", "stride": "1"},
        },
    ),
    (
        Index194,
        [((1, 192, 28, 28), torch.float32)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w48_pose_estimation_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "48", "stride": "1"},
        },
    ),
    (
        Index200,
        [((1, 192, 28, 28), torch.float32)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w48_pose_estimation_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "48", "stop": "96", "stride": "1"},
        },
    ),
    (
        Index201,
        [((1, 192, 28, 28), torch.float32)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w48_pose_estimation_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "96", "stop": "192", "stride": "1"},
        },
    ),
    (
        Index68,
        [((1, 32, 256, 64), torch.float32)],
        {
            "model_names": ["pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf", "pt_phi1_microsoft_phi_1_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "32", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index67,
        [((1, 32, 256, 64), torch.float32)],
        {
            "model_names": ["pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf", "pt_phi1_microsoft_phi_1_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index68,
        [((1, 8, 256, 64), torch.float32)],
        {
            "model_names": ["pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "32", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index67,
        [((1, 8, 256, 64), torch.float32)],
        {
            "model_names": ["pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index67,
        [((1, 32, 256, 80), torch.float32)],
        {
            "model_names": ["pt_phi2_microsoft_phi_2_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index202,
        [((1, 32, 256, 80), torch.float32)],
        {
            "model_names": ["pt_phi2_microsoft_phi_2_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "32", "stop": "80", "stride": "1"},
        },
    ),
    (
        Index69,
        [((1, 32, 256, 32), torch.float32)],
        {
            "model_names": ["pt_phi2_microsoft_phi_2_clm_hf", "pt_phi1_microsoft_phi_1_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "16", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index70,
        [((1, 32, 256, 32), torch.float32)],
        {
            "model_names": ["pt_phi2_microsoft_phi_2_clm_hf", "pt_phi1_microsoft_phi_1_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "16", "stride": "1"},
        },
    ),
    (
        Index68,
        [((1, 16, 29, 64), torch.float32)],
        {
            "model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_chat_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "32", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index67,
        [((1, 16, 29, 64), torch.float32)],
        {
            "model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_chat_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index0,
        [((2, 4, 1), torch.int64)],
        {
            "model_names": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index64,
        [((2, 4, 1), torch.int64)],
        {
            "model_names": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index203,
        [((2, 4, 1), torch.int64)],
        {
            "model_names": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "2", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index204,
        [((2, 4, 1), torch.int64)],
        {
            "model_names": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "3", "stop": "4", "stride": "1"},
        },
    ),
    (
        Index0,
        [((2048, 1536), torch.float32)],
        {
            "model_names": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index133,
        [((3, 197, 1, 1024), torch.float32)],
        {
            "model_names": ["pt_vit_vit_l_16_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index134,
        [((3, 197, 1, 1024), torch.float32)],
        {
            "model_names": ["pt_vit_vit_l_16_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index135,
        [((3, 197, 1, 1024), torch.float32)],
        {
            "model_names": ["pt_vit_vit_l_16_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "2", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index205,
        [((2050, 2048), torch.float32)],
        {
            "model_names": ["pt_xglm_facebook_xglm_1_7b_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "2", "stop": "258", "stride": "1"},
        },
    ),
    (
        Index11,
        [((1, 128, 160, 160), torch.float32)],
        {
            "model_names": ["pt_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index129,
        [((1, 128, 160, 160), torch.float32)],
        {
            "model_names": ["pt_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "64", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index20,
        [((1, 256, 159, 159), torch.float32)],
        {
            "model_names": ["pt_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index21,
        [((1, 256, 159, 159), torch.float32)],
        {
            "model_names": ["pt_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "128", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index20,
        [((1, 256, 80, 80), torch.float32)],
        {
            "model_names": ["pt_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index21,
        [((1, 256, 80, 80), torch.float32)],
        {
            "model_names": ["pt_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "128", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index32,
        [((1, 512, 79, 79), torch.float32)],
        {
            "model_names": ["pt_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index206,
        [((1, 512, 79, 79), torch.float32)],
        {
            "model_names": ["pt_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "256", "stop": "512", "stride": "1"},
        },
    ),
    (
        Index32,
        [((1, 512, 40, 40), torch.float32)],
        {
            "model_names": ["pt_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index206,
        [((1, 512, 40, 40), torch.float32)],
        {
            "model_names": ["pt_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "256", "stop": "512", "stride": "1"},
        },
    ),
    (
        Index32,
        [((1, 512, 39, 39), torch.float32)],
        {
            "model_names": ["pt_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index206,
        [((1, 512, 39, 39), torch.float32)],
        {
            "model_names": ["pt_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "256", "stop": "512", "stride": "1"},
        },
    ),
    (
        Index32,
        [((1, 512, 20, 20), torch.float32)],
        {
            "model_names": ["pt_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index206,
        [((1, 512, 20, 20), torch.float32)],
        {
            "model_names": ["pt_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "256", "stop": "512", "stride": "1"},
        },
    ),
    (
        Index20,
        [((1, 256, 79, 79), torch.float32)],
        {
            "model_names": ["pt_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index21,
        [((1, 256, 79, 79), torch.float32)],
        {
            "model_names": ["pt_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "128", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index0,
        [((1, 6, 768), torch.float32)],
        {
            "model_names": ["onnx_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index0,
        [((1, 8, 768), torch.float32)],
        {
            "model_names": [
                "pd_blip_text_salesforce_blip_image_captioning_base_text_enc_padlenlp",
                "pd_bert_bert_base_uncased_seq_cls_padlenlp",
                "pd_chineseclip_text_ofa_sys_chinese_clip_vit_base_patch16_text_enc_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index1,
        [((2, 1, 9), torch.float32)],
        {
            "model_names": ["pd_ernie_1_0_qa_padlenlp", "pd_bert_bert_base_uncased_qa_padlenlp"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index2,
        [((2, 1, 9), torch.float32)],
        {
            "model_names": ["pd_ernie_1_0_qa_padlenlp", "pd_bert_bert_base_uncased_qa_padlenlp"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index1,
        [((25, 1, 288), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index2,
        [((25, 1, 288), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index10,
        [((25, 1, 288), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "2", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index207,
        [((25, 1, 288), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "3", "stop": "4", "stride": "1"},
        },
    ),
    (
        Index208,
        [((25, 1, 288), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "4", "stop": "5", "stride": "1"},
        },
    ),
    (
        Index209,
        [((25, 1, 288), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "5", "stop": "6", "stride": "1"},
        },
    ),
    (
        Index210,
        [((25, 1, 288), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "6", "stop": "7", "stride": "1"},
        },
    ),
    (
        Index211,
        [((25, 1, 288), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "7", "stop": "8", "stride": "1"},
        },
    ),
    (
        Index212,
        [((25, 1, 288), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "8", "stop": "9", "stride": "1"},
        },
    ),
    (
        Index213,
        [((25, 1, 288), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "9", "stop": "10", "stride": "1"},
        },
    ),
    (
        Index214,
        [((25, 1, 288), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "10", "stop": "11", "stride": "1"},
        },
    ),
    (
        Index215,
        [((25, 1, 288), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "11", "stop": "12", "stride": "1"},
        },
    ),
    (
        Index216,
        [((25, 1, 288), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "12", "stop": "13", "stride": "1"},
        },
    ),
    (
        Index217,
        [((25, 1, 288), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "13", "stop": "14", "stride": "1"},
        },
    ),
    (
        Index218,
        [((25, 1, 288), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "14", "stop": "15", "stride": "1"},
        },
    ),
    (
        Index219,
        [((25, 1, 288), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "15", "stop": "16", "stride": "1"},
        },
    ),
    (
        Index220,
        [((25, 1, 288), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "16", "stop": "17", "stride": "1"},
        },
    ),
    (
        Index221,
        [((25, 1, 288), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "17", "stop": "18", "stride": "1"},
        },
    ),
    (
        Index222,
        [((25, 1, 288), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "18", "stop": "19", "stride": "1"},
        },
    ),
    (
        Index223,
        [((25, 1, 288), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "19", "stop": "20", "stride": "1"},
        },
    ),
    (
        Index224,
        [((25, 1, 288), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "20", "stop": "21", "stride": "1"},
        },
    ),
    (
        Index225,
        [((25, 1, 288), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "21", "stop": "22", "stride": "1"},
        },
    ),
    (
        Index226,
        [((25, 1, 288), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "22", "stop": "23", "stride": "1"},
        },
    ),
    (
        Index227,
        [((25, 1, 288), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "23", "stop": "24", "stride": "1"},
        },
    ),
    (
        Index228,
        [((25, 1, 288), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "24", "stop": "25", "stride": "1"},
        },
    ),
    (
        Index229,
        [((1, 192), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "144", "stop": "192", "stride": "1"},
        },
    ),
    (
        Index230,
        [((1, 192), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "48", "stop": "96", "stride": "1"},
        },
    ),
    (
        Index231,
        [((1, 192), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "48", "stride": "1"},
        },
    ),
    (
        Index232,
        [((1, 192), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "96", "stop": "144", "stride": "1"},
        },
    ),
    (
        Index1,
        [((25, 1, 96), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index2,
        [((25, 1, 96), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index10,
        [((25, 1, 96), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "2", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index207,
        [((25, 1, 96), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "3", "stop": "4", "stride": "1"},
        },
    ),
    (
        Index208,
        [((25, 1, 96), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "4", "stop": "5", "stride": "1"},
        },
    ),
    (
        Index209,
        [((25, 1, 96), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "5", "stop": "6", "stride": "1"},
        },
    ),
    (
        Index210,
        [((25, 1, 96), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "6", "stop": "7", "stride": "1"},
        },
    ),
    (
        Index211,
        [((25, 1, 96), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "7", "stop": "8", "stride": "1"},
        },
    ),
    (
        Index212,
        [((25, 1, 96), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "8", "stop": "9", "stride": "1"},
        },
    ),
    (
        Index213,
        [((25, 1, 96), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "9", "stop": "10", "stride": "1"},
        },
    ),
    (
        Index214,
        [((25, 1, 96), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "10", "stop": "11", "stride": "1"},
        },
    ),
    (
        Index215,
        [((25, 1, 96), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "11", "stop": "12", "stride": "1"},
        },
    ),
    (
        Index216,
        [((25, 1, 96), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "12", "stop": "13", "stride": "1"},
        },
    ),
    (
        Index217,
        [((25, 1, 96), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "13", "stop": "14", "stride": "1"},
        },
    ),
    (
        Index218,
        [((25, 1, 96), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "14", "stop": "15", "stride": "1"},
        },
    ),
    (
        Index219,
        [((25, 1, 96), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "15", "stop": "16", "stride": "1"},
        },
    ),
    (
        Index220,
        [((25, 1, 96), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "16", "stop": "17", "stride": "1"},
        },
    ),
    (
        Index221,
        [((25, 1, 96), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "17", "stop": "18", "stride": "1"},
        },
    ),
    (
        Index222,
        [((25, 1, 96), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "18", "stop": "19", "stride": "1"},
        },
    ),
    (
        Index223,
        [((25, 1, 96), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "19", "stop": "20", "stride": "1"},
        },
    ),
    (
        Index224,
        [((25, 1, 96), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "20", "stop": "21", "stride": "1"},
        },
    ),
    (
        Index225,
        [((25, 1, 96), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "21", "stop": "22", "stride": "1"},
        },
    ),
    (
        Index226,
        [((25, 1, 96), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "22", "stop": "23", "stride": "1"},
        },
    ),
    (
        Index227,
        [((25, 1, 96), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "23", "stop": "24", "stride": "1"},
        },
    ),
    (
        Index228,
        [((25, 1, 96), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "24", "stop": "25", "stride": "1"},
        },
    ),
    (
        Index0,
        [((1, 32, 16, 3, 96), torch.float32)],
        {
            "model_names": ["pt_bloom_bigscience_bloom_1b1_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index64,
        [((1, 32, 16, 3, 96), torch.float32)],
        {
            "model_names": ["pt_bloom_bigscience_bloom_1b1_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index203,
        [((1, 32, 16, 3, 96), torch.float32)],
        {
            "model_names": ["pt_bloom_bigscience_bloom_1b1_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "2", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index62,
        [((3072, 1024), torch.float32)],
        {
            "model_names": ["pt_codegen_salesforce_codegen_350m_nl_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index233,
        [((3072, 1024), torch.float32)],
        {
            "model_names": ["pt_codegen_salesforce_codegen_350m_nl_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "768", "stop": "1024", "stride": "1"},
        },
    ),
    (
        Index234,
        [((3072, 1024), torch.float32)],
        {
            "model_names": ["pt_codegen_salesforce_codegen_350m_nl_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "1536", "stop": "1792", "stride": "1"},
        },
    ),
    (
        Index235,
        [((3072, 1024), torch.float32)],
        {
            "model_names": ["pt_codegen_salesforce_codegen_350m_nl_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "2304", "stop": "2560", "stride": "1"},
        },
    ),
    (
        Index236,
        [((3072, 1024), torch.float32)],
        {
            "model_names": ["pt_codegen_salesforce_codegen_350m_nl_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "512", "stop": "768", "stride": "1"},
        },
    ),
    (
        Index237,
        [((3072, 1024), torch.float32)],
        {
            "model_names": ["pt_codegen_salesforce_codegen_350m_nl_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "1280", "stop": "1536", "stride": "1"},
        },
    ),
    (
        Index238,
        [((3072, 1024), torch.float32)],
        {
            "model_names": ["pt_codegen_salesforce_codegen_350m_nl_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "2048", "stop": "2304", "stride": "1"},
        },
    ),
    (
        Index239,
        [((3072, 1024), torch.float32)],
        {
            "model_names": ["pt_codegen_salesforce_codegen_350m_nl_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "2816", "stop": "3072", "stride": "1"},
        },
    ),
    (
        Index240,
        [((3072, 1024), torch.float32)],
        {
            "model_names": ["pt_codegen_salesforce_codegen_350m_nl_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "256", "stop": "512", "stride": "1"},
        },
    ),
    (
        Index241,
        [((3072, 1024), torch.float32)],
        {
            "model_names": ["pt_codegen_salesforce_codegen_350m_nl_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "1024", "stop": "1280", "stride": "1"},
        },
    ),
    (
        Index242,
        [((3072, 1024), torch.float32)],
        {
            "model_names": ["pt_codegen_salesforce_codegen_350m_nl_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "1792", "stop": "2048", "stride": "1"},
        },
    ),
    (
        Index243,
        [((3072, 1024), torch.float32)],
        {
            "model_names": ["pt_codegen_salesforce_codegen_350m_nl_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "2560", "stop": "2816", "stride": "1"},
        },
    ),
    (
        Index67,
        [((1, 256, 16, 64), torch.float32)],
        {
            "model_names": ["pt_codegen_salesforce_codegen_350m_nl_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index68,
        [((1, 256, 16, 64), torch.float32)],
        {
            "model_names": ["pt_codegen_salesforce_codegen_350m_nl_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "32", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index244,
        [((1, 256, 16, 32), torch.float32)],
        {
            "model_names": ["pt_codegen_salesforce_codegen_350m_nl_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "1", "stop": "32", "stride": "2"},
        },
    ),
    (
        Index245,
        [((1, 256, 16, 32), torch.float32)],
        {
            "model_names": ["pt_codegen_salesforce_codegen_350m_nl_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "32", "stride": "2"},
        },
    ),
    (
        Index0,
        [((2, 768), torch.float32)],
        {
            "model_names": [
                "pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf",
                "pt_albert_twmkn9_albert_base_v2_squad2_qa_hf",
                "pt_opt_facebook_opt_125m_qa_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index64,
        [((2, 768), torch.float32)],
        {
            "model_names": [
                "pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf",
                "pt_albert_twmkn9_albert_base_v2_squad2_qa_hf",
                "pt_opt_facebook_opt_125m_qa_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index185,
        [((1, 64, 28, 28), torch.float32)],
        {
            "model_names": [
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "16", "stride": "1"},
        },
    ),
    (
        Index186,
        [((1, 64, 28, 28), torch.float32)],
        {
            "model_names": [
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "16", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index128,
        [((1, 64, 28, 28), torch.float32)],
        {
            "model_names": [
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "32", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index11,
        [((1, 112, 7, 7), torch.float32)],
        {
            "model_names": [
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index126,
        [((1, 112, 7, 7), torch.float32)],
        {
            "model_names": [
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "64", "stop": "96", "stride": "1"},
        },
    ),
    (
        Index246,
        [((1, 112, 7, 7), torch.float32)],
        {
            "model_names": [
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "96", "stop": "112", "stride": "1"},
        },
    ),
    (
        Index247,
        [((1, 4096, 6), torch.float32)],
        {
            "model_names": ["pt_mamba_state_spaces_mamba_370m_hf_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "2048", "stride": "1"},
        },
    ),
    (
        Index248,
        [((1, 4096, 6), torch.float32)],
        {
            "model_names": ["pt_mamba_state_spaces_mamba_370m_hf_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "2048", "stop": "4096", "stride": "1"},
        },
    ),
    (
        Index249,
        [((1, 2048, 9), torch.float32)],
        {
            "model_names": ["pt_mamba_state_spaces_mamba_370m_hf_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "6", "stride": "1"},
        },
    ),
    (
        Index187,
        [((96, 2048), torch.float32)],
        {
            "model_names": ["pt_mamba_state_spaces_mamba_370m_hf_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index250,
        [((96, 2048), torch.float32)],
        {
            "model_names": ["pt_mamba_state_spaces_mamba_370m_hf_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "64", "stop": "80", "stride": "1"},
        },
    ),
    (
        Index251,
        [((96, 2048), torch.float32)],
        {
            "model_names": ["pt_mamba_state_spaces_mamba_370m_hf_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "80", "stop": "96", "stride": "1"},
        },
    ),
    (
        Index0,
        [((1, 2048, 6, 16), torch.float32)],
        {
            "model_names": ["pt_mamba_state_spaces_mamba_370m_hf_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index64,
        [((1, 2048, 6, 16), torch.float32)],
        {
            "model_names": ["pt_mamba_state_spaces_mamba_370m_hf_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index203,
        [((1, 2048, 6, 16), torch.float32)],
        {
            "model_names": ["pt_mamba_state_spaces_mamba_370m_hf_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "2", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index204,
        [((1, 2048, 6, 16), torch.float32)],
        {
            "model_names": ["pt_mamba_state_spaces_mamba_370m_hf_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "3", "stop": "4", "stride": "1"},
        },
    ),
    (
        Index252,
        [((1, 2048, 6, 16), torch.float32)],
        {
            "model_names": ["pt_mamba_state_spaces_mamba_370m_hf_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "4", "stop": "5", "stride": "1"},
        },
    ),
    (
        Index253,
        [((1, 2048, 6, 16), torch.float32)],
        {
            "model_names": ["pt_mamba_state_spaces_mamba_370m_hf_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "5", "stop": "6", "stride": "1"},
        },
    ),
    (
        Index0,
        [((1, 6, 16), torch.float32)],
        {
            "model_names": ["pt_mamba_state_spaces_mamba_370m_hf_clm_hf", "pt_mamba_state_spaces_mamba_790m_hf_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index64,
        [((1, 6, 16), torch.float32)],
        {
            "model_names": ["pt_mamba_state_spaces_mamba_370m_hf_clm_hf", "pt_mamba_state_spaces_mamba_790m_hf_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index203,
        [((1, 6, 16), torch.float32)],
        {
            "model_names": ["pt_mamba_state_spaces_mamba_370m_hf_clm_hf", "pt_mamba_state_spaces_mamba_790m_hf_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "2", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index204,
        [((1, 6, 16), torch.float32)],
        {
            "model_names": ["pt_mamba_state_spaces_mamba_370m_hf_clm_hf", "pt_mamba_state_spaces_mamba_790m_hf_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "3", "stop": "4", "stride": "1"},
        },
    ),
    (
        Index252,
        [((1, 6, 16), torch.float32)],
        {
            "model_names": ["pt_mamba_state_spaces_mamba_370m_hf_clm_hf", "pt_mamba_state_spaces_mamba_790m_hf_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "4", "stop": "5", "stride": "1"},
        },
    ),
    (
        Index253,
        [((1, 6, 16), torch.float32)],
        {
            "model_names": ["pt_mamba_state_spaces_mamba_370m_hf_clm_hf", "pt_mamba_state_spaces_mamba_790m_hf_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "5", "stop": "6", "stride": "1"},
        },
    ),
    (
        Index71,
        [((1, 12, 39, 128), torch.float32)],
        {
            "model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "64", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index72,
        [((1, 12, 39, 128), torch.float32)],
        {
            "model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index71,
        [((1, 2, 39, 128), torch.float32)],
        {
            "model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "64", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index72,
        [((1, 2, 39, 128), torch.float32)],
        {
            "model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index73,
        [((3, 64, 3, 49, 32), torch.float32)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index74,
        [((3, 64, 3, 49, 32), torch.float32)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index75,
        [((3, 64, 3, 49, 32), torch.float32)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "2", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index73,
        [((3, 16, 6, 49, 32), torch.float32)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index74,
        [((3, 16, 6, 49, 32), torch.float32)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index75,
        [((3, 16, 6, 49, 32), torch.float32)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "2", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index73,
        [((3, 4, 12, 49, 32), torch.float32)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index74,
        [((3, 4, 12, 49, 32), torch.float32)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index75,
        [((3, 4, 12, 49, 32), torch.float32)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "2", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index73,
        [((3, 1, 24, 49, 32), torch.float32)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index74,
        [((3, 1, 24, 49, 32), torch.float32)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index75,
        [((3, 1, 24, 49, 32), torch.float32)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "2", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index0,
        [((1, 201, 768), torch.float32)],
        {
            "model_names": ["pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index0,
        [((1, 15, 768), torch.float32)],
        {
            "model_names": ["pd_bert_bert_base_japanese_seq_cls_padlenlp"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index73,
        [((3, 1, 8, 12, 15), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v4_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index74,
        [((3, 1, 8, 12, 15), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v4_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index75,
        [((3, 1, 8, 12, 15), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v4_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "2", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index9,
        [((1, 8, 522, 256), torch.float32)],
        {
            "model_names": ["pt_falcon3_tiiuae_falcon3_1b_base_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "128", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index3,
        [((1, 8, 522, 256), torch.float32)],
        {
            "model_names": ["pt_falcon3_tiiuae_falcon3_1b_base_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index9,
        [((1, 4, 522, 256), torch.float32)],
        {
            "model_names": ["pt_falcon3_tiiuae_falcon3_1b_base_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "128", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index3,
        [((1, 4, 522, 256), torch.float32)],
        {
            "model_names": ["pt_falcon3_tiiuae_falcon3_1b_base_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index1,
        [((1, 2, 30, 40), torch.float32)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index2,
        [((1, 2, 30, 40), torch.float32)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index1,
        [((1, 2, 60, 80), torch.float32)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index2,
        [((1, 2, 60, 80), torch.float32)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index1,
        [((1, 2, 120, 160), torch.float32)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index2,
        [((1, 2, 120, 160), torch.float32)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index254,
        [((1, 126, 7, 7), torch.float32)],
        {
            "model_names": [
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "18", "stride": "1"},
        },
    ),
    (
        Index255,
        [((1, 126, 7, 7), torch.float32)],
        {
            "model_names": [
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "18", "stop": "54", "stride": "1"},
        },
    ),
    (
        Index256,
        [((1, 126, 7, 7), torch.float32)],
        {
            "model_names": [
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "54", "stop": "126", "stride": "1"},
        },
    ),
    (
        Index257,
        [((1, 126, 7, 7), torch.float32)],
        {
            "model_names": [
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "72", "stride": "1"},
        },
    ),
    (
        Index258,
        [((1, 126, 7, 7), torch.float32)],
        {
            "model_names": [
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "72", "stop": "108", "stride": "1"},
        },
    ),
    (
        Index259,
        [((1, 126, 7, 7), torch.float32)],
        {
            "model_names": [
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "108", "stop": "126", "stride": "1"},
        },
    ),
    (
        Index254,
        [((1, 72, 28, 28), torch.float32)],
        {
            "model_names": [
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "18", "stride": "1"},
        },
    ),
    (
        Index260,
        [((1, 72, 28, 28), torch.float32)],
        {
            "model_names": [
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "36", "stride": "1"},
        },
    ),
    (
        Index261,
        [((1, 72, 28, 28), torch.float32)],
        {
            "model_names": [
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "36", "stop": "54", "stride": "1"},
        },
    ),
    (
        Index262,
        [((1, 72, 28, 28), torch.float32)],
        {
            "model_names": [
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "54", "stop": "72", "stride": "1"},
        },
    ),
    (
        Index263,
        [((1, 72, 28, 28), torch.float32)],
        {
            "model_names": [
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "18", "stop": "36", "stride": "1"},
        },
    ),
    (
        Index264,
        [((1, 72, 28, 28), torch.float32)],
        {
            "model_names": [
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "36", "stop": "72", "stride": "1"},
        },
    ),
    (
        Index265,
        [((1, 6144, 6), torch.float32)],
        {
            "model_names": ["pt_mamba_state_spaces_mamba_790m_hf_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "3072", "stride": "1"},
        },
    ),
    (
        Index266,
        [((1, 6144, 6), torch.float32)],
        {
            "model_names": ["pt_mamba_state_spaces_mamba_790m_hf_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "3072", "stop": "6144", "stride": "1"},
        },
    ),
    (
        Index249,
        [((1, 3072, 9), torch.float32)],
        {
            "model_names": ["pt_mamba_state_spaces_mamba_790m_hf_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "6", "stride": "1"},
        },
    ),
    (
        Index267,
        [((128, 3072), torch.float32)],
        {
            "model_names": ["pt_mamba_state_spaces_mamba_790m_hf_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "96", "stride": "1"},
        },
    ),
    (
        Index268,
        [((128, 3072), torch.float32)],
        {
            "model_names": ["pt_mamba_state_spaces_mamba_790m_hf_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "96", "stop": "112", "stride": "1"},
        },
    ),
    (
        Index269,
        [((128, 3072), torch.float32)],
        {
            "model_names": ["pt_mamba_state_spaces_mamba_790m_hf_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "112", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index0,
        [((1, 3072, 6, 16), torch.float32)],
        {
            "model_names": ["pt_mamba_state_spaces_mamba_790m_hf_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index64,
        [((1, 3072, 6, 16), torch.float32)],
        {
            "model_names": ["pt_mamba_state_spaces_mamba_790m_hf_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index203,
        [((1, 3072, 6, 16), torch.float32)],
        {
            "model_names": ["pt_mamba_state_spaces_mamba_790m_hf_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "2", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index204,
        [((1, 3072, 6, 16), torch.float32)],
        {
            "model_names": ["pt_mamba_state_spaces_mamba_790m_hf_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "3", "stop": "4", "stride": "1"},
        },
    ),
    (
        Index252,
        [((1, 3072, 6, 16), torch.float32)],
        {
            "model_names": ["pt_mamba_state_spaces_mamba_790m_hf_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "4", "stop": "5", "stride": "1"},
        },
    ),
    (
        Index253,
        [((1, 3072, 6, 16), torch.float32)],
        {
            "model_names": ["pt_mamba_state_spaces_mamba_790m_hf_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "5", "stop": "6", "stride": "1"},
        },
    ),
    (
        Index0,
        [((2, 2048), torch.float32)],
        {
            "model_names": ["pt_opt_facebook_opt_1_3b_qa_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index64,
        [((2, 2048), torch.float32)],
        {
            "model_names": ["pt_opt_facebook_opt_1_3b_qa_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index67,
        [((1, 32, 7, 64), torch.float32)],
        {
            "model_names": ["pt_phi1_microsoft_phi_1_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index68,
        [((1, 32, 7, 64), torch.float32)],
        {
            "model_names": ["pt_phi1_microsoft_phi_1_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "32", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index69,
        [((1, 32, 7, 32), torch.float32)],
        {
            "model_names": ["pt_phi1_microsoft_phi_1_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "16", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index70,
        [((1, 32, 7, 32), torch.float32)],
        {
            "model_names": ["pt_phi1_microsoft_phi_1_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "16", "stride": "1"},
        },
    ),
    (
        Index71,
        [((1, 12, 29, 128), torch.float32)],
        {
            "model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "64", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index72,
        [((1, 12, 29, 128), torch.float32)],
        {
            "model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index71,
        [((1, 2, 29, 128), torch.float32)],
        {
            "model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "64", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index72,
        [((1, 2, 29, 128), torch.float32)],
        {
            "model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index0,
        [((1, 11, 768), torch.float32)],
        {
            "model_names": ["pd_bert_chinese_roberta_base_seq_cls_padlenlp"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index11,
        [((1, 176, 27, 27), torch.float32)],
        {
            "model_names": ["pd_googlenet_base_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index12,
        [((1, 176, 27, 27), torch.float32)],
        {
            "model_names": ["pd_googlenet_base_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "64", "stop": "160", "stride": "1"},
        },
    ),
    (
        Index13,
        [((1, 176, 27, 27), torch.float32)],
        {
            "model_names": ["pd_googlenet_base_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "160", "stop": "176", "stride": "1"},
        },
    ),
    (
        Index20,
        [((1, 288, 27, 27), torch.float32)],
        {
            "model_names": ["pd_googlenet_base_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index21,
        [((1, 288, 27, 27), torch.float32)],
        {
            "model_names": ["pd_googlenet_base_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "128", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index22,
        [((1, 288, 27, 27), torch.float32)],
        {
            "model_names": ["pd_googlenet_base_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "256", "stop": "288", "stride": "1"},
        },
    ),
    (
        Index23,
        [((1, 304, 13, 13), torch.float32)],
        {
            "model_names": ["pd_googlenet_base_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "192", "stride": "1"},
        },
    ),
    (
        Index24,
        [((1, 304, 13, 13), torch.float32)],
        {
            "model_names": ["pd_googlenet_base_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "192", "stop": "288", "stride": "1"},
        },
    ),
    (
        Index25,
        [((1, 304, 13, 13), torch.float32)],
        {
            "model_names": ["pd_googlenet_base_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "288", "stop": "304", "stride": "1"},
        },
    ),
    (
        Index26,
        [((1, 296, 13, 13), torch.float32)],
        {
            "model_names": ["pd_googlenet_base_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "160", "stride": "1"},
        },
    ),
    (
        Index27,
        [((1, 296, 13, 13), torch.float32)],
        {
            "model_names": ["pd_googlenet_base_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "160", "stop": "272", "stride": "1"},
        },
    ),
    (
        Index28,
        [((1, 296, 13, 13), torch.float32)],
        {
            "model_names": ["pd_googlenet_base_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "272", "stop": "296", "stride": "1"},
        },
    ),
    (
        Index20,
        [((1, 280, 13, 13), torch.float32)],
        {
            "model_names": ["pd_googlenet_base_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index21,
        [((1, 280, 13, 13), torch.float32)],
        {
            "model_names": ["pd_googlenet_base_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "128", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index29,
        [((1, 280, 13, 13), torch.float32)],
        {
            "model_names": ["pd_googlenet_base_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "256", "stop": "280", "stride": "1"},
        },
    ),
    (
        Index30,
        [((1, 288, 13, 13), torch.float32)],
        {
            "model_names": ["pd_googlenet_base_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "112", "stride": "1"},
        },
    ),
    (
        Index31,
        [((1, 288, 13, 13), torch.float32)],
        {
            "model_names": ["pd_googlenet_base_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "112", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index22,
        [((1, 288, 13, 13), torch.float32)],
        {
            "model_names": ["pd_googlenet_base_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "256", "stop": "288", "stride": "1"},
        },
    ),
    (
        Index32,
        [((1, 448, 13, 13), torch.float32)],
        {
            "model_names": ["pd_googlenet_base_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index33,
        [((1, 448, 13, 13), torch.float32)],
        {
            "model_names": ["pd_googlenet_base_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "256", "stop": "416", "stride": "1"},
        },
    ),
    (
        Index34,
        [((1, 448, 13, 13), torch.float32)],
        {
            "model_names": ["pd_googlenet_base_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "416", "stop": "448", "stride": "1"},
        },
    ),
    (
        Index32,
        [((1, 448, 6, 6), torch.float32)],
        {
            "model_names": ["pd_googlenet_base_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index33,
        [((1, 448, 6, 6), torch.float32)],
        {
            "model_names": ["pd_googlenet_base_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "256", "stop": "416", "stride": "1"},
        },
    ),
    (
        Index34,
        [((1, 448, 6, 6), torch.float32)],
        {
            "model_names": ["pd_googlenet_base_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "416", "stop": "448", "stride": "1"},
        },
    ),
    (
        Index35,
        [((1, 624, 6, 6), torch.float32)],
        {
            "model_names": ["pd_googlenet_base_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "384", "stride": "1"},
        },
    ),
    (
        Index36,
        [((1, 624, 6, 6), torch.float32)],
        {
            "model_names": ["pd_googlenet_base_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "384", "stop": "576", "stride": "1"},
        },
    ),
    (
        Index37,
        [((1, 624, 6, 6), torch.float32)],
        {
            "model_names": ["pd_googlenet_base_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "576", "stop": "624", "stride": "1"},
        },
    ),
    (
        Index71,
        [((1, 16, 588, 128), torch.float32)],
        {
            "model_names": ["pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "64", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index72,
        [((1, 16, 588, 128), torch.float32)],
        {
            "model_names": ["pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index73,
        [((3, 1, 12, 257, 64), torch.float32)],
        {
            "model_names": ["pt_mgp_alibaba_damo_mgp_str_base_scene_text_recognition_hf"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index74,
        [((3, 1, 12, 257, 64), torch.float32)],
        {
            "model_names": ["pt_mgp_alibaba_damo_mgp_str_base_scene_text_recognition_hf"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index75,
        [((3, 1, 12, 257, 64), torch.float32)],
        {
            "model_names": ["pt_mgp_alibaba_damo_mgp_str_base_scene_text_recognition_hf"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "2", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index270,
        [((1, 256, 2), torch.float32)],
        {
            "model_names": ["pt_phi1_microsoft_phi_1_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "255", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index68,
        [((1, 14, 35, 64), torch.float32)],
        {
            "model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "32", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index67,
        [((1, 14, 35, 64), torch.float32)],
        {
            "model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index68,
        [((1, 2, 35, 64), torch.float32)],
        {
            "model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "32", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index67,
        [((1, 2, 35, 64), torch.float32)],
        {
            "model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index73,
        [((3, 64, 4, 64, 32), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index74,
        [((3, 64, 4, 64, 32), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index75,
        [((3, 64, 4, 64, 32), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "2", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index76,
        [((1, 64, 64, 128), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "4", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index77,
        [((1, 64, 64, 128), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "4", "stride": "1"},
        },
    ),
    (
        Index78,
        [((1, 64, 64, 128), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "4", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index79,
        [((1, 64, 64, 128), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "4", "stride": "1"},
        },
    ),
    (
        Index80,
        [((1, 64, 64, 128), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "60", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index44,
        [((1, 64, 64, 128), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "60", "stride": "1"},
        },
    ),
    (
        Index81,
        [((1, 64, 64, 128), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "60", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index82,
        [((1, 64, 64, 128), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "60", "stride": "1"},
        },
    ),
    (
        Index83,
        [((1, 64, 64, 128), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "64", "stride": "2"},
        },
    ),
    (
        Index84,
        [((1, 64, 64, 128), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "1", "stop": "64", "stride": "2"},
        },
    ),
    (
        Index89,
        [((1, 32, 64, 128), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "64", "stride": "2"},
        },
    ),
    (
        Index90,
        [((1, 32, 64, 128), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "1", "stop": "64", "stride": "2"},
        },
    ),
    (
        Index73,
        [((3, 16, 8, 64, 32), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index74,
        [((3, 16, 8, 64, 32), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index75,
        [((3, 16, 8, 64, 32), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "2", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index91,
        [((1, 32, 32, 256), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "4", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index77,
        [((1, 32, 32, 256), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "4", "stride": "1"},
        },
    ),
    (
        Index92,
        [((1, 32, 32, 256), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "4", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index79,
        [((1, 32, 32, 256), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "4", "stride": "1"},
        },
    ),
    (
        Index93,
        [((1, 32, 32, 256), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "28", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index94,
        [((1, 32, 32, 256), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "28", "stride": "1"},
        },
    ),
    (
        Index95,
        [((1, 32, 32, 256), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "28", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index96,
        [((1, 32, 32, 256), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "28", "stride": "1"},
        },
    ),
    (
        Index97,
        [((1, 32, 32, 256), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "32", "stride": "2"},
        },
    ),
    (
        Index98,
        [((1, 32, 32, 256), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "1", "stop": "32", "stride": "2"},
        },
    ),
    (
        Index101,
        [((1, 16, 32, 256), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "32", "stride": "2"},
        },
    ),
    (
        Index102,
        [((1, 16, 32, 256), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "1", "stop": "32", "stride": "2"},
        },
    ),
    (
        Index73,
        [((3, 4, 16, 64, 32), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index74,
        [((3, 4, 16, 64, 32), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index75,
        [((3, 4, 16, 64, 32), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "2", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index103,
        [((1, 16, 16, 512), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "4", "stop": "16", "stride": "1"},
        },
    ),
    (
        Index77,
        [((1, 16, 16, 512), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "4", "stride": "1"},
        },
    ),
    (
        Index104,
        [((1, 16, 16, 512), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "4", "stop": "16", "stride": "1"},
        },
    ),
    (
        Index79,
        [((1, 16, 16, 512), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "4", "stride": "1"},
        },
    ),
    (
        Index105,
        [((1, 16, 16, 512), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "12", "stop": "16", "stride": "1"},
        },
    ),
    (
        Index106,
        [((1, 16, 16, 512), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "12", "stride": "1"},
        },
    ),
    (
        Index107,
        [((1, 16, 16, 512), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "12", "stop": "16", "stride": "1"},
        },
    ),
    (
        Index108,
        [((1, 16, 16, 512), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "12", "stride": "1"},
        },
    ),
    (
        Index109,
        [((1, 16, 16, 512), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "16", "stride": "2"},
        },
    ),
    (
        Index110,
        [((1, 16, 16, 512), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "1", "stop": "16", "stride": "2"},
        },
    ),
    (
        Index113,
        [((1, 8, 16, 512), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "16", "stride": "2"},
        },
    ),
    (
        Index114,
        [((1, 8, 16, 512), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "1", "stop": "16", "stride": "2"},
        },
    ),
    (
        Index73,
        [((3, 1, 32, 64, 32), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index74,
        [((3, 1, 32, 64, 32), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index75,
        [((3, 1, 32, 64, 32), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "2", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index140,
        [((732, 12), torch.float32)],
        {
            "model_names": ["pt_beit_microsoft_beit_base_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "729", "stride": "1"},
        },
    ),
    (
        Index141,
        [((732, 12), torch.float32)],
        {
            "model_names": ["pt_beit_microsoft_beit_base_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "729", "stop": "732", "stride": "1"},
        },
    ),
    (
        Index63,
        [((1, 77), torch.int64)],
        {
            "model_names": ["pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "7", "stride": "1"},
        },
    ),
    (
        Index0,
        [((1, 197, 384), torch.float32)],
        {
            "model_names": ["pt_deit_facebook_deit_small_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index142,
        [((1, 1, 256, 1024), torch.bool)],
        {
            "model_names": ["pt_gpt_gpt2_text_gen_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index271,
        [((1, 280, 7, 7), torch.float32)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w40_pose_estimation_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "40", "stride": "1"},
        },
    ),
    (
        Index272,
        [((1, 280, 7, 7), torch.float32)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w40_pose_estimation_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "40", "stop": "120", "stride": "1"},
        },
    ),
    (
        Index273,
        [((1, 280, 7, 7), torch.float32)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w40_pose_estimation_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "120", "stop": "280", "stride": "1"},
        },
    ),
    (
        Index26,
        [((1, 280, 7, 7), torch.float32)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w40_pose_estimation_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "160", "stride": "1"},
        },
    ),
    (
        Index274,
        [((1, 280, 7, 7), torch.float32)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w40_pose_estimation_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "160", "stop": "240", "stride": "1"},
        },
    ),
    (
        Index275,
        [((1, 280, 7, 7), torch.float32)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w40_pose_estimation_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "240", "stop": "280", "stride": "1"},
        },
    ),
    (
        Index276,
        [((1, 160, 28, 28), torch.float32)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w40_pose_estimation_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "80", "stride": "1"},
        },
    ),
    (
        Index277,
        [((1, 160, 28, 28), torch.float32)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w40_pose_estimation_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "80", "stop": "120", "stride": "1"},
        },
    ),
    (
        Index278,
        [((1, 160, 28, 28), torch.float32)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w40_pose_estimation_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "120", "stop": "160", "stride": "1"},
        },
    ),
    (
        Index271,
        [((1, 160, 28, 28), torch.float32)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w40_pose_estimation_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "40", "stride": "1"},
        },
    ),
    (
        Index279,
        [((1, 160, 28, 28), torch.float32)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w40_pose_estimation_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "40", "stop": "80", "stride": "1"},
        },
    ),
    (
        Index280,
        [((1, 160, 28, 28), torch.float32)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w40_pose_estimation_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "80", "stop": "160", "stride": "1"},
        },
    ),
    (
        Index281,
        [((1024, 48), torch.float32)],
        {
            "model_names": ["pt_nbeats_seasionality_basis_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "12", "stop": "24", "stride": "1"},
        },
    ),
    (
        Index282,
        [((1024, 48), torch.float32)],
        {
            "model_names": ["pt_nbeats_seasionality_basis_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "12", "stride": "1"},
        },
    ),
    (
        Index283,
        [((1024, 48), torch.float32)],
        {
            "model_names": ["pt_nbeats_seasionality_basis_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "36", "stop": "48", "stride": "1"},
        },
    ),
    (
        Index284,
        [((1024, 48), torch.float32)],
        {
            "model_names": ["pt_nbeats_seasionality_basis_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "24", "stop": "36", "stride": "1"},
        },
    ),
    (
        Index133,
        [((3, 50, 1, 768), torch.float32)],
        {
            "model_names": ["pt_vit_vit_b_32_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index134,
        [((3, 50, 1, 768), torch.float32)],
        {
            "model_names": ["pt_vit_vit_b_32_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index135,
        [((3, 50, 1, 768), torch.float32)],
        {
            "model_names": ["pt_vit_vit_b_32_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "2", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index0,
        [((1, 50, 768), torch.float32)],
        {
            "model_names": ["pt_vit_vit_b_32_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "1", "stride": "1"},
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
