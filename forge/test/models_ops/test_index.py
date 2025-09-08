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
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=768, stride=1)
        return index_output_1


class Index1(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=768, stop=1536, stride=1)
        return index_output_1


class Index2(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=1536, stop=2304, stride=1)
        return index_output_1


class Index3(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=0, stop=768, stride=1)
        return index_output_1


class Index4(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=768, stop=1536, stride=1)
        return index_output_1


class Index5(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=1536, stop=2304, stride=1)
        return index_output_1


class Index6(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=64, stride=1)
        return index_output_1


class Index7(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=64, stop=192, stride=1)
        return index_output_1


class Index8(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=192, stop=448, stride=1)
        return index_output_1


class Index9(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=256, stride=1)
        return index_output_1


class Index10(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=256, stop=384, stride=1)
        return index_output_1


class Index11(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=384, stop=448, stride=1)
        return index_output_1


class Index12(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=256, stop=416, stride=1)
        return index_output_1


class Index13(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=416, stop=448, stride=1)
        return index_output_1


class Index14(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=128, stride=1)
        return index_output_1


class Index15(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=128, stop=192, stride=1)
        return index_output_1


class Index16(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=192, stop=256, stride=1)
        return index_output_1


class Index17(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=64, stop=128, stride=1)
        return index_output_1


class Index18(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=128, stop=256, stride=1)
        return index_output_1


class Index19(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-5, start=0, stop=1, stride=1)
        return index_output_1


class Index20(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-5, start=1, stop=2, stride=1)
        return index_output_1


class Index21(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-5, start=2, stop=3, stride=1)
        return index_output_1


class Index22(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=0, stop=14, stride=1)
        return index_output_1


class Index23(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=0, stop=128, stride=1)
        return index_output_1


class Index24(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=0, stop=384, stride=1)
        return index_output_1


class Index25(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=0, stop=25, stride=1)
        return index_output_1


class Index26(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=0, stop=8, stride=1)
        return index_output_1


class Index27(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=0, stop=16, stride=1)
        return index_output_1


class Index28(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=0, stop=9, stride=1)
        return index_output_1


class Index29(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=0, stop=4, stride=1)
        return index_output_1


class Index30(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=1, stride=1)
        return index_output_1


class Index31(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=1, stop=2, stride=1)
        return index_output_1


class Index32(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=0, stop=1, stride=1)
        return index_output_1


class Index33(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=1, stop=2, stride=1)
        return index_output_1


class Index34(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=256, stride=1)
        return index_output_1


class Index35(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=768, stop=1024, stride=1)
        return index_output_1


class Index36(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=1536, stop=1792, stride=1)
        return index_output_1


class Index37(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=2304, stop=2560, stride=1)
        return index_output_1


class Index38(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=512, stop=768, stride=1)
        return index_output_1


class Index39(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=1280, stop=1536, stride=1)
        return index_output_1


class Index40(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=2048, stop=2304, stride=1)
        return index_output_1


class Index41(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=2816, stop=3072, stride=1)
        return index_output_1


class Index42(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=256, stop=512, stride=1)
        return index_output_1


class Index43(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=1024, stop=1280, stride=1)
        return index_output_1


class Index44(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=1792, stop=2048, stride=1)
        return index_output_1


class Index45(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=2560, stop=2816, stride=1)
        return index_output_1


class Index46(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=0, stop=32, stride=1)
        return index_output_1


class Index47(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=32, stop=64, stride=1)
        return index_output_1


class Index48(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=1, stop=32, stride=2)
        return index_output_1


class Index49(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=0, stop=32, stride=2)
        return index_output_1


class Index50(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=0, stop=256, stride=1)
        return index_output_1


class Index51(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=18, stride=1)
        return index_output_1


class Index52(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=18, stop=54, stride=1)
        return index_output_1


class Index53(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=54, stop=126, stride=1)
        return index_output_1


class Index54(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=72, stride=1)
        return index_output_1


class Index55(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=72, stop=108, stride=1)
        return index_output_1


class Index56(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=108, stop=126, stride=1)
        return index_output_1


class Index57(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=36, stride=1)
        return index_output_1


class Index58(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=36, stop=54, stride=1)
        return index_output_1


class Index59(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=54, stop=72, stride=1)
        return index_output_1


class Index60(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=18, stop=36, stride=1)
        return index_output_1


class Index61(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=36, stop=72, stride=1)
        return index_output_1


class Index62(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=31, stop=32, stride=1)
        return index_output_1


class Index63(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=16, stop=32, stride=1)
        return index_output_1


class Index64(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=2, stop=3, stride=1)
        return index_output_1


class Index65(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=3, stop=4, stride=1)
        return index_output_1


class Index66(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=3, stop=56, stride=1)
        return index_output_1


class Index67(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=3, stride=1)
        return index_output_1


class Index68(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=3, stop=56, stride=1)
        return index_output_1


class Index69(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=3, stride=1)
        return index_output_1


class Index70(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=53, stop=56, stride=1)
        return index_output_1


class Index71(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=53, stride=1)
        return index_output_1


class Index72(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=53, stop=56, stride=1)
        return index_output_1


class Index73(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=53, stride=1)
        return index_output_1


class Index74(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=56, stride=2)
        return index_output_1


class Index75(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=1, stop=56, stride=2)
        return index_output_1


class Index76(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=56, stride=2)
        return index_output_1


class Index77(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=1, stop=56, stride=2)
        return index_output_1


class Index78(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=3, stop=28, stride=1)
        return index_output_1


class Index79(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=3, stop=28, stride=1)
        return index_output_1


class Index80(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=25, stop=28, stride=1)
        return index_output_1


class Index81(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=25, stride=1)
        return index_output_1


class Index82(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=25, stop=28, stride=1)
        return index_output_1


class Index83(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=25, stride=1)
        return index_output_1


class Index84(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=28, stride=2)
        return index_output_1


class Index85(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=1, stop=28, stride=2)
        return index_output_1


class Index86(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=28, stride=2)
        return index_output_1


class Index87(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=1, stop=28, stride=2)
        return index_output_1


class Index88(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=3, stop=14, stride=1)
        return index_output_1


class Index89(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=3, stop=14, stride=1)
        return index_output_1


class Index90(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=11, stop=14, stride=1)
        return index_output_1


class Index91(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=11, stride=1)
        return index_output_1


class Index92(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=11, stop=14, stride=1)
        return index_output_1


class Index93(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=11, stride=1)
        return index_output_1


class Index94(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=14, stride=2)
        return index_output_1


class Index95(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=1, stop=14, stride=2)
        return index_output_1


class Index96(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=14, stride=2)
        return index_output_1


class Index97(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=1, stop=14, stride=2)
        return index_output_1


class Index98(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=0, stop=2, stride=1)
        return index_output_1


class Index99(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=2, stop=4, stride=1)
        return index_output_1


class Index100(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=4, stop=85, stride=1)
        return index_output_1


class Index101(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=1, stride=1)
        return index_output_1


class Index102(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=1, stop=2, stride=1)
        return index_output_1


class Index103(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=2, stop=3, stride=1)
        return index_output_1


class Index104(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=80, stride=1)
        return index_output_1


class Index105(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=80, stop=160, stride=1)
        return index_output_1


class Index106(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=160, stride=1)
        return index_output_1


class Index107(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=160, stop=320, stride=1)
        return index_output_1


class Index108(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=320, stride=1)
        return index_output_1


class Index109(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=320, stop=640, stride=1)
        return index_output_1


class Index110(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=64, stride=1)
        return index_output_1


class Index111(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=64, stop=144, stride=1)
        return index_output_1


class Index112(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=2, stride=1)
        return index_output_1


class Index113(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=2, stop=4, stride=1)
        return index_output_1


class Index114(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=640, stride=2)
        return index_output_1


class Index115(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=1, stop=640, stride=2)
        return index_output_1


class Index116(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=0, stop=640, stride=2)
        return index_output_1


class Index117(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=1, stop=640, stride=2)
        return index_output_1


class Index118(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=16, stride=1)
        return index_output_1


class Index119(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=48, stride=1)
        return index_output_1


class Index120(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=24, stride=1)
        return index_output_1


class Index121(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=40, stride=1)
        return index_output_1


class Index122(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=120, stride=1)
        return index_output_1


class Index123(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=60, stride=1)
        return index_output_1


class Index124(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=60, stop=90, stride=1)
        return index_output_1


class Index125(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=90, stop=120, stride=1)
        return index_output_1


class Index126(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=30, stride=1)
        return index_output_1


class Index127(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=30, stop=60, stride=1)
        return index_output_1


class Index128(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=60, stop=120, stride=1)
        return index_output_1


class Index129(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=240, stride=1)
        return index_output_1


class Index130(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=200, stride=1)
        return index_output_1


class Index131(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=184, stride=1)
        return index_output_1


class Index132(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=480, stride=1)
        return index_output_1


class Index133(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=112, stride=1)
        return index_output_1


class Index134(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=672, stride=1)
        return index_output_1


class Index135(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=960, stride=1)
        return index_output_1


class Index136(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=16, stop=32, stride=1)
        return index_output_1


class Index137(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=32, stride=1)
        return index_output_1


class Index138(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=32, stop=64, stride=1)
        return index_output_1


class Index139(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=64, stop=128, stride=1)
        return index_output_1


class Index140(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=32, stride=1)
        return index_output_1


class Index141(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=32, stop=64, stride=1)
        return index_output_1


class Index142(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=1, stop=197, stride=1)
        return index_output_1


class Index143(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=3, stop=4, stride=1)
        return index_output_1


class Index144(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=4, stop=5, stride=1)
        return index_output_1


class Index145(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=5, stop=6, stride=1)
        return index_output_1


class Index146(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=6, stop=7, stride=1)
        return index_output_1


class Index147(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=7, stop=8, stride=1)
        return index_output_1


class Index148(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=8, stop=9, stride=1)
        return index_output_1


class Index149(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=9, stop=10, stride=1)
        return index_output_1


class Index150(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=10, stop=11, stride=1)
        return index_output_1


class Index151(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=11, stop=12, stride=1)
        return index_output_1


class Index152(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=12, stop=13, stride=1)
        return index_output_1


class Index153(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=13, stop=14, stride=1)
        return index_output_1


class Index154(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=14, stop=15, stride=1)
        return index_output_1


class Index155(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=15, stop=16, stride=1)
        return index_output_1


class Index156(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=16, stop=17, stride=1)
        return index_output_1


class Index157(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=17, stop=18, stride=1)
        return index_output_1


class Index158(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=18, stop=19, stride=1)
        return index_output_1


class Index159(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=19, stop=20, stride=1)
        return index_output_1


class Index160(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=20, stop=21, stride=1)
        return index_output_1


class Index161(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=21, stop=22, stride=1)
        return index_output_1


class Index162(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=22, stop=23, stride=1)
        return index_output_1


class Index163(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=23, stop=24, stride=1)
        return index_output_1


class Index164(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=24, stop=25, stride=1)
        return index_output_1


class Index165(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=144, stop=192, stride=1)
        return index_output_1


class Index166(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=48, stop=96, stride=1)
        return index_output_1


class Index167(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=0, stop=48, stride=1)
        return index_output_1


class Index168(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=96, stop=144, stride=1)
        return index_output_1


class Index169(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=729, stride=1)
        return index_output_1


class Index170(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=729, stop=732, stride=1)
        return index_output_1


class Index171(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=64, stop=128, stride=1)
        return index_output_1


class Index172(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=0, stop=64, stride=1)
        return index_output_1


class Index173(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=32, stop=96, stride=1)
        return index_output_1


class Index174(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=96, stop=224, stride=1)
        return index_output_1


class Index175(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=192, stop=224, stride=1)
        return index_output_1


class Index176(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=64, stop=96, stride=1)
        return index_output_1


class Index177(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=96, stop=128, stride=1)
        return index_output_1


class Index178(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=-1, stop=72, stride=1)
        return index_output_1


class Index179(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=-1, stop=-9223372036854775807, stride=-1)
        return index_output_1


class Index180(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=-24, stop=96, stride=1)
        return index_output_1


class Index181(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=0, stop=72, stride=1)
        return index_output_1


class Index182(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=0, stop=10, stride=1)
        return index_output_1


class Index183(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=416, stride=2)
        return index_output_1


class Index184(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=1, stop=416, stride=2)
        return index_output_1


class Index185(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=0, stop=416, stride=2)
        return index_output_1


class Index186(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=1, stop=416, stride=2)
        return index_output_1


class Index187(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=30, stop=90, stride=1)
        return index_output_1


class Index188(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=90, stop=210, stride=1)
        return index_output_1


class Index189(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=120, stop=180, stride=1)
        return index_output_1


class Index190(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=180, stop=210, stride=1)
        return index_output_1


class Index191(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=96, stride=1)
        return index_output_1


class Index192(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=96, stop=160, stride=1)
        return index_output_1


class Index193(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=160, stop=224, stride=1)
        return index_output_1


class Index194(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=384, stride=1)
        return index_output_1


class Index195(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=384, stop=576, stride=1)
        return index_output_1


class Index196(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=576, stop=768, stride=1)
        return index_output_1


class Index197(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=256, stop=640, stride=1)
        return index_output_1


class Index198(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=640, stop=1024, stride=1)
        return index_output_1


class Index199(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=-1, stop=1, stride=1)
        return index_output_1


class Index200(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=-61, stop=61, stride=1)
        return index_output_1


class Index201(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=64, stop=160, stride=1)
        return index_output_1


class Index202(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=160, stop=176, stride=1)
        return index_output_1


class Index203(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=88, stride=1)
        return index_output_1


class Index204(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=88, stop=132, stride=1)
        return index_output_1


class Index205(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=132, stop=176, stride=1)
        return index_output_1


class Index206(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=44, stride=1)
        return index_output_1


class Index207(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=44, stop=88, stride=1)
        return index_output_1


class Index208(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=88, stop=176, stride=1)
        return index_output_1


class Index209(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=256, stop=288, stride=1)
        return index_output_1


class Index210(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=192, stride=1)
        return index_output_1


class Index211(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=192, stop=288, stride=1)
        return index_output_1


class Index212(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=288, stop=304, stride=1)
        return index_output_1


class Index213(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=160, stop=272, stride=1)
        return index_output_1


class Index214(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=272, stop=296, stride=1)
        return index_output_1


class Index215(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=256, stop=280, stride=1)
        return index_output_1


class Index216(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=112, stop=256, stride=1)
        return index_output_1


class Index217(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=576, stop=624, stride=1)
        return index_output_1


class Index218(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=4, stop=64, stride=1)
        return index_output_1


class Index219(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=4, stride=1)
        return index_output_1


class Index220(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=4, stop=64, stride=1)
        return index_output_1


class Index221(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=4, stride=1)
        return index_output_1


class Index222(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=60, stop=64, stride=1)
        return index_output_1


class Index223(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=60, stop=64, stride=1)
        return index_output_1


class Index224(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=60, stride=1)
        return index_output_1


class Index225(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=64, stride=2)
        return index_output_1


class Index226(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=1, stop=64, stride=2)
        return index_output_1


class Index227(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=64, stride=2)
        return index_output_1


class Index228(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=1, stop=64, stride=2)
        return index_output_1


class Index229(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=4, stop=32, stride=1)
        return index_output_1


class Index230(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=4, stop=32, stride=1)
        return index_output_1


class Index231(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=28, stop=32, stride=1)
        return index_output_1


class Index232(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=28, stride=1)
        return index_output_1


class Index233(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=28, stop=32, stride=1)
        return index_output_1


class Index234(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=28, stride=1)
        return index_output_1


class Index235(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=32, stride=2)
        return index_output_1


class Index236(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=1, stop=32, stride=2)
        return index_output_1


class Index237(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=32, stride=2)
        return index_output_1


class Index238(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=1, stop=32, stride=2)
        return index_output_1


class Index239(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=4, stop=16, stride=1)
        return index_output_1


class Index240(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=4, stop=16, stride=1)
        return index_output_1


class Index241(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=12, stop=16, stride=1)
        return index_output_1


class Index242(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=12, stride=1)
        return index_output_1


class Index243(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=12, stop=16, stride=1)
        return index_output_1


class Index244(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=12, stride=1)
        return index_output_1


class Index245(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=16, stride=2)
        return index_output_1


class Index246(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=1, stop=16, stride=2)
        return index_output_1


class Index247(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=16, stride=2)
        return index_output_1


class Index248(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=1, stop=16, stride=2)
        return index_output_1


class Index249(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=96, stop=112, stride=1)
        return index_output_1


class Index250(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=44, stop=132, stride=1)
        return index_output_1


class Index251(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=132, stop=308, stride=1)
        return index_output_1


class Index252(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=176, stride=1)
        return index_output_1


class Index253(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=176, stop=264, stride=1)
        return index_output_1


class Index254(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=264, stop=308, stride=1)
        return index_output_1


class Index255(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=128, stop=256, stride=1)
        return index_output_1


class Index256(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=5, stride=1)
        return index_output_1


class Index257(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=0, stop=5, stride=1)
        return index_output_1


class Index258(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=4, stop=5, stride=1)
        return index_output_1


class Index259(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=-4, stop=64, stride=1)
        return index_output_1


class Index260(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=-4, stride=1)
        return index_output_1


class Index261(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=-4, stop=64, stride=1)
        return index_output_1


class Index262(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=-4, stride=1)
        return index_output_1


class Index263(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=-4, stop=32, stride=1)
        return index_output_1


class Index264(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=-4, stop=32, stride=1)
        return index_output_1


class Index265(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=-4, stop=16, stride=1)
        return index_output_1


class Index266(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=-4, stop=16, stride=1)
        return index_output_1


class Index267(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=40, stop=120, stride=1)
        return index_output_1


class Index268(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=120, stop=280, stride=1)
        return index_output_1


class Index269(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=160, stop=240, stride=1)
        return index_output_1


class Index270(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=240, stop=280, stride=1)
        return index_output_1


class Index271(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=80, stop=120, stride=1)
        return index_output_1


class Index272(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=120, stop=160, stride=1)
        return index_output_1


class Index273(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=40, stop=80, stride=1)
        return index_output_1


class Index274(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=48, stop=144, stride=1)
        return index_output_1


class Index275(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=144, stop=336, stride=1)
        return index_output_1


class Index276(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=288, stop=336, stride=1)
        return index_output_1


class Index277(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=96, stop=144, stride=1)
        return index_output_1


class Index278(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=144, stop=192, stride=1)
        return index_output_1


class Index279(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=48, stop=96, stride=1)
        return index_output_1


class Index280(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=96, stop=192, stride=1)
        return index_output_1


class Index281(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=4, stop=8, stride=1)
        return index_output_1


class Index282(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=24, stride=1)
        return index_output_1


class Index283(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=0, stop=1, stride=1)
        return index_output_1


class Index284(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=1, stop=2, stride=1)
        return index_output_1


class Index285(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=2, stop=3, stride=1)
        return index_output_1


class Index286(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=2, stop=258, stride=1)
        return index_output_1


class Index287(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=12, stop=24, stride=1)
        return index_output_1


class Index288(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=0, stop=12, stride=1)
        return index_output_1


class Index289(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=36, stop=48, stride=1)
        return index_output_1


class Index290(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=24, stop=36, stride=1)
        return index_output_1


class Index291(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=32, stop=80, stride=1)
        return index_output_1


class Index292(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=10, stop=11, stride=1)
        return index_output_1


class Index293(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=0, stop=7, stride=1)
        return index_output_1


class Index294(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=256, stop=512, stride=1)
        return index_output_1


class Index295(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=1, stop=577, stride=1)
        return index_output_1


class Index296(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=0, stop=3072, stride=1)
        return index_output_1


class Index297(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=3072, stop=6144, stride=1)
        return index_output_1


class Index298(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=6144, stop=9216, stride=1)
        return index_output_1


class Index299(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=8192, stop=16384, stride=1)
        return index_output_1


class Index300(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=8192, stride=1)
        return index_output_1


class Index301(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=160, stop=320, stride=1)
        return index_output_1


class Index302(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=0, stop=160, stride=1)
        return index_output_1


class Index303(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=2560, stride=1)
        return index_output_1


class Index304(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=2560, stop=5120, stride=1)
        return index_output_1


class Index305(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=0, stop=2560, stride=1)
        return index_output_1


class Index306(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=2560, stop=5120, stride=1)
        return index_output_1


class Index307(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=5120, stride=1)
        return index_output_1


class Index308(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=5120, stop=10240, stride=1)
        return index_output_1


class Index309(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=0, stop=5120, stride=1)
        return index_output_1


class Index310(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=5120, stop=10240, stride=1)
        return index_output_1


class Index311(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=-1, stop=25, stride=1)
        return index_output_1


class Index312(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=-1, stop=34, stride=1)
        return index_output_1


class Index313(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=0, stop=128, stride=2)
        return index_output_1


class Index314(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=1, stop=128, stride=2)
        return index_output_1


class Index315(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=-2, stride=1)
        return index_output_1


class Index316(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=-2, stop=-1, stride=1)
        return index_output_1


class Index317(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=72, stop=73, stride=1)
        return index_output_1


class Index318(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=512, stop=768, stride=1)
        return index_output_1


class Index319(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=768, stop=1024, stride=1)
        return index_output_1


class Index320(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=1024, stop=1280, stride=1)
        return index_output_1


class Index321(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=1280, stop=1536, stride=1)
        return index_output_1


class Index322(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=1536, stop=1792, stride=1)
        return index_output_1


class Index323(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=5120, stop=6400, stride=1)
        return index_output_1


class Index324(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=6400, stop=7680, stride=1)
        return index_output_1


class Index325(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=17920, stop=35840, stride=1)
        return index_output_1


class Index326(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=17920, stride=1)
        return index_output_1


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (
        Index0,
        [((2304, 768), torch.float32)],
        {
            "model_names": [
                "onnx_gpt_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
                "pt_gpt_gpt2_sequence_classification_seq_cls_hf",
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
                "onnx_nanogpt_financialsupport_nanogpt_text_gen_hf",
                "pt_gpt_gpt2_text_gen_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "768", "stride": "1"},
        },
    ),
    (
        Index1,
        [((2304, 768), torch.float32)],
        {
            "model_names": [
                "onnx_gpt_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
                "pt_gpt_gpt2_sequence_classification_seq_cls_hf",
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
                "onnx_nanogpt_financialsupport_nanogpt_text_gen_hf",
                "pt_gpt_gpt2_text_gen_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "768", "stop": "1536", "stride": "1"},
        },
    ),
    (
        Index2,
        [((2304, 768), torch.float32)],
        {
            "model_names": [
                "onnx_gpt_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
                "pt_gpt_gpt2_sequence_classification_seq_cls_hf",
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
                "onnx_nanogpt_financialsupport_nanogpt_text_gen_hf",
                "pt_gpt_gpt2_text_gen_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "1536", "stop": "2304", "stride": "1"},
        },
    ),
    (
        Index3,
        [((2304,), torch.float32)],
        {
            "model_names": [
                "onnx_gpt_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
                "pt_gpt_gpt2_sequence_classification_seq_cls_hf",
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
                "onnx_nanogpt_financialsupport_nanogpt_text_gen_hf",
                "pt_gpt_gpt2_text_gen_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "768", "stride": "1"},
        },
    ),
    (
        Index4,
        [((2304,), torch.float32)],
        {
            "model_names": [
                "onnx_gpt_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
                "pt_gpt_gpt2_sequence_classification_seq_cls_hf",
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
                "onnx_nanogpt_financialsupport_nanogpt_text_gen_hf",
                "pt_gpt_gpt2_text_gen_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "768", "stop": "1536", "stride": "1"},
        },
    ),
    (
        Index5,
        [((2304,), torch.float32)],
        {
            "model_names": [
                "onnx_gpt_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
                "pt_gpt_gpt2_sequence_classification_seq_cls_hf",
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
                "onnx_nanogpt_financialsupport_nanogpt_text_gen_hf",
                "pt_gpt_gpt2_text_gen_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "1536", "stop": "2304", "stride": "1"},
        },
    ),
    (
        Index6,
        [((1, 448, 7, 7), torch.float32)],
        {
            "model_names": ["onnx_hrnet_hrnetv2_w64_img_cls_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index7,
        [((1, 448, 7, 7), torch.float32)],
        {
            "model_names": ["onnx_hrnet_hrnetv2_w64_img_cls_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "64", "stop": "192", "stride": "1"},
        },
    ),
    (
        Index8,
        [((1, 448, 7, 7), torch.float32)],
        {
            "model_names": ["onnx_hrnet_hrnetv2_w64_img_cls_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "192", "stop": "448", "stride": "1"},
        },
    ),
    (
        Index9,
        [((1, 448, 7, 7), torch.float32)],
        {
            "model_names": ["onnx_hrnet_hrnetv2_w64_img_cls_osmr", "onnx_googlenet_googlenet_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index10,
        [((1, 448, 7, 7), torch.float32)],
        {
            "model_names": ["onnx_hrnet_hrnetv2_w64_img_cls_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "256", "stop": "384", "stride": "1"},
        },
    ),
    (
        Index11,
        [((1, 448, 7, 7), torch.float32)],
        {
            "model_names": ["onnx_hrnet_hrnetv2_w64_img_cls_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "384", "stop": "448", "stride": "1"},
        },
    ),
    (
        Index12,
        [((1, 448, 7, 7), torch.float32)],
        {
            "model_names": ["onnx_googlenet_googlenet_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "256", "stop": "416", "stride": "1"},
        },
    ),
    (
        Index13,
        [((1, 448, 7, 7), torch.float32)],
        {
            "model_names": ["onnx_googlenet_googlenet_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "416", "stop": "448", "stride": "1"},
        },
    ),
    (
        Index14,
        [((1, 256, 28, 28), torch.float32)],
        {
            "model_names": ["onnx_hrnet_hrnetv2_w64_img_cls_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index15,
        [((1, 256, 28, 28), torch.float32)],
        {
            "model_names": ["onnx_hrnet_hrnetv2_w64_img_cls_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "128", "stop": "192", "stride": "1"},
        },
    ),
    (
        Index16,
        [((1, 256, 28, 28), torch.float32)],
        {
            "model_names": ["onnx_hrnet_hrnetv2_w64_img_cls_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "192", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index6,
        [((1, 256, 28, 28), torch.float32)],
        {
            "model_names": ["onnx_hrnet_hrnetv2_w64_img_cls_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index17,
        [((1, 256, 28, 28), torch.float32)],
        {
            "model_names": ["onnx_hrnet_hrnetv2_w64_img_cls_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "64", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index18,
        [((1, 256, 28, 28), torch.float32)],
        {
            "model_names": ["onnx_hrnet_hrnetv2_w64_img_cls_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "128", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index19,
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
        Index20,
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
        Index21,
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
        Index22,
        [((1, 512), torch.int64)],
        {
            "model_names": ["pt_albert_squad2_qa_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "14", "stride": "1"},
        },
    ),
    (
        Index23,
        [((1, 512), torch.int64)],
        {
            "model_names": [
                "pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf",
                "pt_albert_base_v1_mlm_hf",
                "pt_albert_base_v1_token_cls_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
                "pt_albert_xxlarge_v2_token_cls_hf",
                "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "pt_albert_xlarge_v1_mlm_hf",
                "pt_albert_xlarge_v1_token_cls_hf",
                "pt_albert_xlarge_v2_mlm_hf",
                "pt_albert_xlarge_v2_token_cls_hf",
                "pt_bert_bert_base_uncased_mlm_hf",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_squeezebert_squeezebert_mnli_seq_cls_hf",
                "pt_albert_large_v2_mlm_hf",
                "pt_albert_large_v2_token_cls_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
                "pt_albert_base_v2_mlm_hf",
                "pt_albert_base_v2_token_cls_hf",
                "pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder",
                "pt_albert_large_v1_mlm_hf",
                "pt_albert_large_v1_token_cls_hf",
                "pt_distilbert_distilbert_base_uncased_mlm_hf",
                "pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder",
                "pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf",
                "pt_distilbert_distilbert_base_multilingual_cased_mlm_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder",
                "pt_albert_xxlarge_v1_mlm_hf",
                "pt_albert_xxlarge_v1_token_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index24,
        [((1, 512), torch.int64)],
        {
            "model_names": [
                "pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf",
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                "pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "384", "stride": "1"},
        },
    ),
    (
        Index25,
        [((1, 512), torch.int64)],
        {
            "model_names": [
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "25", "stride": "1"},
        },
    ),
    (
        Index26,
        [((1, 512), torch.int64)],
        {
            "model_names": ["pd_blip_text_salesforce_blip_image_captioning_base_text_enc_padlenlp"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "8", "stride": "1"},
        },
    ),
    (
        Index27,
        [((1, 512), torch.int64)],
        {
            "model_names": ["pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "16", "stride": "1"},
        },
    ),
    (
        Index28,
        [((1, 512), torch.int64)],
        {
            "model_names": ["pt_albert_imdb_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "9", "stride": "1"},
        },
    ),
    (
        Index29,
        [((1, 512), torch.int64)],
        {
            "model_names": ["pd_blip_salesforce_blip_image_captioning_base_img_captioning_padlenlp"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "4", "stride": "1"},
        },
    ),
    (
        Index30,
        [((2, 768), torch.float32)],
        {
            "model_names": [
                "pt_albert_squad2_qa_hf",
                "pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_opt_facebook_opt_125m_qa_hf",
                "pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index31,
        [((2, 768), torch.float32)],
        {
            "model_names": [
                "pt_albert_squad2_qa_hf",
                "pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_opt_facebook_opt_125m_qa_hf",
                "pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    pytest.param(
        (
            Index32,
            [((2,), torch.float32)],
            {
                "model_names": [
                    "pt_albert_squad2_qa_hf",
                    "pt_opt_facebook_opt_350m_qa_hf",
                    "pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf",
                    "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                    "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                    "pt_opt_facebook_opt_1_3b_qa_hf",
                    "pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf",
                    "pt_opt_facebook_opt_125m_qa_hf",
                    "pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader",
                    "onnx_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                ],
                "pcc": 0.99,
                "args": {"dim": "-1", "start": "0", "stop": "1", "stride": "1"},
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Index33,
            [((2,), torch.float32)],
            {
                "model_names": [
                    "pt_albert_squad2_qa_hf",
                    "pt_opt_facebook_opt_350m_qa_hf",
                    "pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf",
                    "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                    "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                    "pt_opt_facebook_opt_1_3b_qa_hf",
                    "pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf",
                    "pt_opt_facebook_opt_125m_qa_hf",
                    "pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader",
                    "onnx_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                ],
                "pcc": 0.99,
                "args": {"dim": "-1", "start": "1", "stop": "2", "stride": "1"},
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Index34,
        [((3072, 1024), torch.float32)],
        {
            "model_names": [
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_mono_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index35,
        [((3072, 1024), torch.float32)],
        {
            "model_names": [
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_mono_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "768", "stop": "1024", "stride": "1"},
        },
    ),
    (
        Index36,
        [((3072, 1024), torch.float32)],
        {
            "model_names": [
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_mono_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "1536", "stop": "1792", "stride": "1"},
        },
    ),
    (
        Index37,
        [((3072, 1024), torch.float32)],
        {
            "model_names": [
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_mono_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "2304", "stop": "2560", "stride": "1"},
        },
    ),
    (
        Index38,
        [((3072, 1024), torch.float32)],
        {
            "model_names": [
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_mono_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "512", "stop": "768", "stride": "1"},
        },
    ),
    (
        Index39,
        [((3072, 1024), torch.float32)],
        {
            "model_names": [
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_mono_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "1280", "stop": "1536", "stride": "1"},
        },
    ),
    (
        Index40,
        [((3072, 1024), torch.float32)],
        {
            "model_names": [
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_mono_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "2048", "stop": "2304", "stride": "1"},
        },
    ),
    (
        Index41,
        [((3072, 1024), torch.float32)],
        {
            "model_names": [
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_mono_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "2816", "stop": "3072", "stride": "1"},
        },
    ),
    (
        Index42,
        [((3072, 1024), torch.float32)],
        {
            "model_names": [
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_mono_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "256", "stop": "512", "stride": "1"},
        },
    ),
    (
        Index43,
        [((3072, 1024), torch.float32)],
        {
            "model_names": [
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_mono_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "1024", "stop": "1280", "stride": "1"},
        },
    ),
    (
        Index44,
        [((3072, 1024), torch.float32)],
        {
            "model_names": [
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_mono_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "1792", "stop": "2048", "stride": "1"},
        },
    ),
    (
        Index45,
        [((3072, 1024), torch.float32)],
        {
            "model_names": [
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_mono_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "2560", "stop": "2816", "stride": "1"},
        },
    ),
    (
        Index46,
        [((1, 256, 16, 64), torch.float32)],
        {
            "model_names": [
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index47,
        [((1, 256, 16, 64), torch.float32)],
        {
            "model_names": [
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "32", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index48,
        [((1, 256, 16, 32), torch.float32)],
        {
            "model_names": [
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "1", "stop": "32", "stride": "2"},
        },
    ),
    (
        Index49,
        [((1, 256, 16, 32), torch.float32)],
        {
            "model_names": [
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "32", "stride": "2"},
        },
    ),
    (
        Index50,
        [((1, 1, 256, 256), torch.float32)],
        {
            "model_names": [
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index51,
        [((1, 126, 7, 7), torch.bfloat16)],
        {
            "model_names": [
                "pt_hrnet_hrnetv2_w18_osmr_img_cls_osmr",
                "pt_hrnet_hrnet_w18_small_v2_osmr_img_cls_osmr",
                "pt_hrnet_hrnet_w18_ms_aug_in1k_img_cls_timm",
                "pt_hrnet_hrnet_w18_small_v2_img_cls_timm",
                "pt_hrnet_hrnet_w18_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "0", "stop": "18", "stride": "1"},
        },
    ),
    (
        Index52,
        [((1, 126, 7, 7), torch.bfloat16)],
        {
            "model_names": [
                "pt_hrnet_hrnetv2_w18_osmr_img_cls_osmr",
                "pt_hrnet_hrnet_w18_small_v2_osmr_img_cls_osmr",
                "pt_hrnet_hrnet_w18_ms_aug_in1k_img_cls_timm",
                "pt_hrnet_hrnet_w18_small_v2_img_cls_timm",
                "pt_hrnet_hrnet_w18_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "18", "stop": "54", "stride": "1"},
        },
    ),
    (
        Index53,
        [((1, 126, 7, 7), torch.bfloat16)],
        {
            "model_names": [
                "pt_hrnet_hrnetv2_w18_osmr_img_cls_osmr",
                "pt_hrnet_hrnet_w18_small_v2_osmr_img_cls_osmr",
                "pt_hrnet_hrnet_w18_ms_aug_in1k_img_cls_timm",
                "pt_hrnet_hrnet_w18_small_v2_img_cls_timm",
                "pt_hrnet_hrnet_w18_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "54", "stop": "126", "stride": "1"},
        },
    ),
    (
        Index54,
        [((1, 126, 7, 7), torch.bfloat16)],
        {
            "model_names": [
                "pt_hrnet_hrnetv2_w18_osmr_img_cls_osmr",
                "pt_hrnet_hrnet_w18_small_v2_osmr_img_cls_osmr",
                "pt_hrnet_hrnet_w18_ms_aug_in1k_img_cls_timm",
                "pt_hrnet_hrnet_w18_small_v2_img_cls_timm",
                "pt_hrnet_hrnet_w18_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "0", "stop": "72", "stride": "1"},
        },
    ),
    (
        Index55,
        [((1, 126, 7, 7), torch.bfloat16)],
        {
            "model_names": [
                "pt_hrnet_hrnetv2_w18_osmr_img_cls_osmr",
                "pt_hrnet_hrnet_w18_small_v2_osmr_img_cls_osmr",
                "pt_hrnet_hrnet_w18_ms_aug_in1k_img_cls_timm",
                "pt_hrnet_hrnet_w18_small_v2_img_cls_timm",
                "pt_hrnet_hrnet_w18_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "72", "stop": "108", "stride": "1"},
        },
    ),
    (
        Index56,
        [((1, 126, 7, 7), torch.bfloat16)],
        {
            "model_names": [
                "pt_hrnet_hrnetv2_w18_osmr_img_cls_osmr",
                "pt_hrnet_hrnet_w18_small_v2_osmr_img_cls_osmr",
                "pt_hrnet_hrnet_w18_ms_aug_in1k_img_cls_timm",
                "pt_hrnet_hrnet_w18_small_v2_img_cls_timm",
                "pt_hrnet_hrnet_w18_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "108", "stop": "126", "stride": "1"},
        },
    ),
    (
        Index57,
        [((1, 72, 28, 28), torch.bfloat16)],
        {
            "model_names": [
                "pt_hrnet_hrnetv2_w18_osmr_img_cls_osmr",
                "pt_hrnet_hrnet_w18_small_v2_osmr_img_cls_osmr",
                "pt_hrnet_hrnet_w18_ms_aug_in1k_img_cls_timm",
                "pt_hrnet_hrnet_w18_small_v2_img_cls_timm",
                "pt_hrnet_hrnet_w18_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "0", "stop": "36", "stride": "1"},
        },
    ),
    (
        Index58,
        [((1, 72, 28, 28), torch.bfloat16)],
        {
            "model_names": [
                "pt_hrnet_hrnetv2_w18_osmr_img_cls_osmr",
                "pt_hrnet_hrnet_w18_small_v2_osmr_img_cls_osmr",
                "pt_hrnet_hrnet_w18_ms_aug_in1k_img_cls_timm",
                "pt_hrnet_hrnet_w18_small_v2_img_cls_timm",
                "pt_hrnet_hrnet_w18_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "36", "stop": "54", "stride": "1"},
        },
    ),
    (
        Index59,
        [((1, 72, 28, 28), torch.bfloat16)],
        {
            "model_names": [
                "pt_hrnet_hrnetv2_w18_osmr_img_cls_osmr",
                "pt_hrnet_hrnet_w18_small_v2_osmr_img_cls_osmr",
                "pt_hrnet_hrnet_w18_ms_aug_in1k_img_cls_timm",
                "pt_hrnet_hrnet_w18_small_v2_img_cls_timm",
                "pt_hrnet_hrnet_w18_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "54", "stop": "72", "stride": "1"},
        },
    ),
    (
        Index51,
        [((1, 72, 28, 28), torch.bfloat16)],
        {
            "model_names": [
                "pt_hrnet_hrnetv2_w18_osmr_img_cls_osmr",
                "pt_hrnet_hrnet_w18_small_v2_osmr_img_cls_osmr",
                "pt_hrnet_hrnet_w18_ms_aug_in1k_img_cls_timm",
                "pt_hrnet_hrnet_w18_small_v2_img_cls_timm",
                "pt_hrnet_hrnet_w18_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "0", "stop": "18", "stride": "1"},
        },
    ),
    (
        Index60,
        [((1, 72, 28, 28), torch.bfloat16)],
        {
            "model_names": [
                "pt_hrnet_hrnetv2_w18_osmr_img_cls_osmr",
                "pt_hrnet_hrnet_w18_small_v2_osmr_img_cls_osmr",
                "pt_hrnet_hrnet_w18_ms_aug_in1k_img_cls_timm",
                "pt_hrnet_hrnet_w18_small_v2_img_cls_timm",
                "pt_hrnet_hrnet_w18_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "18", "stop": "36", "stride": "1"},
        },
    ),
    (
        Index61,
        [((1, 72, 28, 28), torch.bfloat16)],
        {
            "model_names": [
                "pt_hrnet_hrnetv2_w18_osmr_img_cls_osmr",
                "pt_hrnet_hrnet_w18_small_v2_osmr_img_cls_osmr",
                "pt_hrnet_hrnet_w18_ms_aug_in1k_img_cls_timm",
                "pt_hrnet_hrnet_w18_small_v2_img_cls_timm",
                "pt_hrnet_hrnet_w18_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "36", "stop": "72", "stride": "1"},
        },
    ),
    (
        Index62,
        [((1, 32, 2), torch.float32)],
        {
            "model_names": [
                "pt_opt_facebook_opt_125m_seq_cls_hf",
                "pt_opt_facebook_opt_1_3b_seq_cls_hf",
                "pt_opt_facebook_opt_350m_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "31", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index30,
        [((2, 512), torch.float32)],
        {
            "model_names": ["pt_opt_facebook_opt_350m_qa_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index31,
        [((2, 512), torch.float32)],
        {
            "model_names": ["pt_opt_facebook_opt_350m_qa_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index46,
        [((1, 32, 256, 64), torch.float32)],
        {
            "model_names": [
                "pt_phi_1_5_microsoft_phi_1_5_clm_hf",
                "pt_phi1_microsoft_phi_1_clm_hf",
                "pt_llama3_llama_3_2_1b_clm_hf",
                "pt_llama3_llama_3_2_1b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index47,
        [((1, 32, 256, 64), torch.float32)],
        {
            "model_names": [
                "pt_phi_1_5_microsoft_phi_1_5_clm_hf",
                "pt_phi1_microsoft_phi_1_clm_hf",
                "pt_llama3_llama_3_2_1b_clm_hf",
                "pt_llama3_llama_3_2_1b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "32", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index63,
        [((1, 32, 256, 32), torch.float32)],
        {
            "model_names": [
                "pt_phi_1_5_microsoft_phi_1_5_clm_hf",
                "pt_phi1_microsoft_phi_1_clm_hf",
                "pt_phi2_microsoft_phi_2_pytdml_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "16", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index27,
        [((1, 32, 256, 32), torch.float32)],
        {
            "model_names": [
                "pt_phi_1_5_microsoft_phi_1_5_clm_hf",
                "pt_phi1_microsoft_phi_1_clm_hf",
                "pt_phi2_microsoft_phi_2_pytdml_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "16", "stride": "1"},
        },
    ),
    (
        Index47,
        [((1, 14, 35, 64), torch.float32)],
        {
            "model_names": ["pt_qwen_coder_0_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "32", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index46,
        [((1, 14, 35, 64), torch.float32)],
        {
            "model_names": ["pt_qwen_coder_0_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index47,
        [((1, 2, 35, 64), torch.float32)],
        {
            "model_names": ["pt_qwen_coder_0_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "32", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index46,
        [((1, 2, 35, 64), torch.float32)],
        {
            "model_names": ["pt_qwen_coder_0_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index30,
        [((1, 4, 1), torch.int64)],
        {
            "model_names": [
                "pt_stereo_medium_music_generation_hf",
                "pt_stereo_small_music_generation_hf",
                "pt_stereo_large_music_generation_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index31,
        [((1, 4, 1), torch.int64)],
        {
            "model_names": [
                "pt_stereo_medium_music_generation_hf",
                "pt_stereo_small_music_generation_hf",
                "pt_stereo_large_music_generation_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index64,
        [((1, 4, 1), torch.int64)],
        {
            "model_names": [
                "pt_stereo_medium_music_generation_hf",
                "pt_stereo_small_music_generation_hf",
                "pt_stereo_large_music_generation_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "2", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index65,
        [((1, 4, 1), torch.int64)],
        {
            "model_names": [
                "pt_stereo_medium_music_generation_hf",
                "pt_stereo_small_music_generation_hf",
                "pt_stereo_large_music_generation_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "3", "stop": "4", "stride": "1"},
        },
    ),
    (
        Index30,
        [((2048, 1536), torch.float32)],
        {
            "model_names": ["pt_stereo_medium_music_generation_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index19,
        [((3, 64, 4, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-5", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index20,
        [((3, 64, 4, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-5", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index21,
        [((3, 64, 4, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-5", "start": "2", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index66,
        [((1, 56, 56, 128), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "3", "stop": "56", "stride": "1"},
        },
    ),
    (
        Index67,
        [((1, 56, 56, 128), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "0", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index68,
        [((1, 56, 56, 128), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "start": "3", "stop": "56", "stride": "1"},
        },
    ),
    (
        Index69,
        [((1, 56, 56, 128), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "start": "0", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index70,
        [((1, 56, 56, 128), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "53", "stop": "56", "stride": "1"},
        },
    ),
    (
        Index71,
        [((1, 56, 56, 128), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "0", "stop": "53", "stride": "1"},
        },
    ),
    (
        Index72,
        [((1, 56, 56, 128), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "start": "53", "stop": "56", "stride": "1"},
        },
    ),
    (
        Index73,
        [((1, 56, 56, 128), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "start": "0", "stop": "53", "stride": "1"},
        },
    ),
    (
        Index74,
        [((1, 56, 56, 128), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "0", "stop": "56", "stride": "2"},
        },
    ),
    (
        Index75,
        [((1, 56, 56, 128), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "1", "stop": "56", "stride": "2"},
        },
    ),
    (
        Index76,
        [((1, 28, 56, 128), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "start": "0", "stop": "56", "stride": "2"},
        },
    ),
    (
        Index77,
        [((1, 28, 56, 128), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "start": "1", "stop": "56", "stride": "2"},
        },
    ),
    (
        Index19,
        [((3, 16, 8, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-5", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index20,
        [((3, 16, 8, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-5", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index21,
        [((3, 16, 8, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-5", "start": "2", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index78,
        [((1, 28, 28, 256), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "3", "stop": "28", "stride": "1"},
        },
    ),
    (
        Index67,
        [((1, 28, 28, 256), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "0", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index79,
        [((1, 28, 28, 256), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "start": "3", "stop": "28", "stride": "1"},
        },
    ),
    (
        Index69,
        [((1, 28, 28, 256), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "start": "0", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index80,
        [((1, 28, 28, 256), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "25", "stop": "28", "stride": "1"},
        },
    ),
    (
        Index81,
        [((1, 28, 28, 256), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "0", "stop": "25", "stride": "1"},
        },
    ),
    (
        Index82,
        [((1, 28, 28, 256), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "start": "25", "stop": "28", "stride": "1"},
        },
    ),
    (
        Index83,
        [((1, 28, 28, 256), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "start": "0", "stop": "25", "stride": "1"},
        },
    ),
    (
        Index84,
        [((1, 28, 28, 256), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "0", "stop": "28", "stride": "2"},
        },
    ),
    (
        Index85,
        [((1, 28, 28, 256), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "1", "stop": "28", "stride": "2"},
        },
    ),
    (
        Index86,
        [((1, 14, 28, 256), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "start": "0", "stop": "28", "stride": "2"},
        },
    ),
    (
        Index87,
        [((1, 14, 28, 256), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "start": "1", "stop": "28", "stride": "2"},
        },
    ),
    (
        Index19,
        [((3, 4, 16, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-5", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index20,
        [((3, 4, 16, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-5", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index21,
        [((3, 4, 16, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-5", "start": "2", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index88,
        [((1, 14, 14, 512), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "3", "stop": "14", "stride": "1"},
        },
    ),
    (
        Index67,
        [((1, 14, 14, 512), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "0", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index89,
        [((1, 14, 14, 512), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "start": "3", "stop": "14", "stride": "1"},
        },
    ),
    (
        Index69,
        [((1, 14, 14, 512), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "start": "0", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index90,
        [((1, 14, 14, 512), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "11", "stop": "14", "stride": "1"},
        },
    ),
    (
        Index91,
        [((1, 14, 14, 512), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "0", "stop": "11", "stride": "1"},
        },
    ),
    (
        Index92,
        [((1, 14, 14, 512), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "start": "11", "stop": "14", "stride": "1"},
        },
    ),
    (
        Index93,
        [((1, 14, 14, 512), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "start": "0", "stop": "11", "stride": "1"},
        },
    ),
    (
        Index94,
        [((1, 14, 14, 512), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "0", "stop": "14", "stride": "2"},
        },
    ),
    (
        Index95,
        [((1, 14, 14, 512), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "1", "stop": "14", "stride": "2"},
        },
    ),
    (
        Index96,
        [((1, 7, 14, 512), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "start": "0", "stop": "14", "stride": "2"},
        },
    ),
    (
        Index97,
        [((1, 7, 14, 512), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "start": "1", "stop": "14", "stride": "2"},
        },
    ),
    (
        Index19,
        [((3, 1, 32, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-5", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index20,
        [((3, 1, 32, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-5", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index21,
        [((3, 1, 32, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-5", "start": "2", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index98,
        [((1, 3, 80, 80, 85), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5s_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_1280x1280",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_640x640",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "start": "0", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index99,
        [((1, 3, 80, 80, 85), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5s_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_1280x1280",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_640x640",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "start": "2", "stop": "4", "stride": "1"},
        },
    ),
    (
        Index100,
        [((1, 3, 80, 80, 85), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5s_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_1280x1280",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_640x640",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "start": "4", "stop": "85", "stride": "1"},
        },
    ),
    (
        Index101,
        [((3, 3, 2), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5s_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_1280x1280",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_480x480",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index102,
        [((3, 3, 2), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5s_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_1280x1280",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_480x480",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index103,
        [((3, 3, 2), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5s_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_1280x1280",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_480x480",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "2", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index98,
        [((1, 3, 40, 40, 85), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5s_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_1280x1280",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_640x640",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "start": "0", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index99,
        [((1, 3, 40, 40, 85), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5s_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_1280x1280",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_640x640",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "start": "2", "stop": "4", "stride": "1"},
        },
    ),
    (
        Index100,
        [((1, 3, 40, 40, 85), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5s_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_1280x1280",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_640x640",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "start": "4", "stop": "85", "stride": "1"},
        },
    ),
    (
        Index98,
        [((1, 3, 20, 20, 85), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5s_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_640x640",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "start": "0", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index99,
        [((1, 3, 20, 20, 85), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5s_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_640x640",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "start": "2", "stop": "4", "stride": "1"},
        },
    ),
    (
        Index100,
        [((1, 3, 20, 20, 85), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5s_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_640x640",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "start": "4", "stop": "85", "stride": "1"},
        },
    ),
    (
        Index104,
        [((1, 160, 160, 160), torch.bfloat16)],
        {
            "model_names": ["pt_yolov8_yolov8x_obj_det_github", "pt_yolov10_yolov10x_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "0", "stop": "80", "stride": "1"},
        },
    ),
    (
        Index105,
        [((1, 160, 160, 160), torch.bfloat16)],
        {
            "model_names": ["pt_yolov8_yolov8x_obj_det_github", "pt_yolov10_yolov10x_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "80", "stop": "160", "stride": "1"},
        },
    ),
    (
        Index106,
        [((1, 320, 80, 80), torch.bfloat16)],
        {
            "model_names": ["pt_yolov8_yolov8x_obj_det_github", "pt_yolov10_yolov10x_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "0", "stop": "160", "stride": "1"},
        },
    ),
    (
        Index107,
        [((1, 320, 80, 80), torch.bfloat16)],
        {
            "model_names": ["pt_yolov8_yolov8x_obj_det_github", "pt_yolov10_yolov10x_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "160", "stop": "320", "stride": "1"},
        },
    ),
    (
        Index108,
        [((1, 640, 40, 40), torch.bfloat16)],
        {
            "model_names": ["pt_yolov8_yolov8x_obj_det_github", "pt_yolov10_yolov10x_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "0", "stop": "320", "stride": "1"},
        },
    ),
    (
        Index109,
        [((1, 640, 40, 40), torch.bfloat16)],
        {
            "model_names": ["pt_yolov8_yolov8x_obj_det_github", "pt_yolov10_yolov10x_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "320", "stop": "640", "stride": "1"},
        },
    ),
    (
        Index108,
        [((1, 640, 20, 20), torch.bfloat16)],
        {
            "model_names": ["pt_yolov8_yolov8x_obj_det_github", "pt_yolov10_yolov10x_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "0", "stop": "320", "stride": "1"},
        },
    ),
    (
        Index109,
        [((1, 640, 20, 20), torch.bfloat16)],
        {
            "model_names": ["pt_yolov8_yolov8x_obj_det_github", "pt_yolov10_yolov10x_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "320", "stop": "640", "stride": "1"},
        },
    ),
    (
        Index110,
        [((1, 144, 8400), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolov8_yolov8x_obj_det_github",
                "pt_yolov10_yolov10x_obj_det_github",
                "pt_yolo_world_default_obj_det_github",
                "pt_yolov10_yolov10n_obj_det_github",
                "pt_yolov8_yolov8n_obj_det_github",
                "pt_yolov9_default_obj_det_github",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index111,
        [((1, 144, 8400), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolov8_yolov8x_obj_det_github",
                "pt_yolov10_yolov10x_obj_det_github",
                "pt_yolo_world_default_obj_det_github",
                "pt_yolov10_yolov10n_obj_det_github",
                "pt_yolov8_yolov8n_obj_det_github",
                "pt_yolov9_default_obj_det_github",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "start": "64", "stop": "144", "stride": "1"},
        },
    ),
    (
        Index112,
        [((1, 4, 8400), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolov8_yolov8x_obj_det_github",
                "pt_yolov10_yolov10x_obj_det_github",
                "pt_yolo_world_default_obj_det_github",
                "pt_yolov10_yolov10n_obj_det_github",
                "pt_yolov8_yolov8n_obj_det_github",
                "pt_yolov9_default_obj_det_github",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "start": "0", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index113,
        [((1, 4, 8400), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolov8_yolov8x_obj_det_github",
                "pt_yolov10_yolov10x_obj_det_github",
                "pt_yolo_world_default_obj_det_github",
                "pt_yolov10_yolov10n_obj_det_github",
                "pt_yolov8_yolov8n_obj_det_github",
                "pt_yolov9_default_obj_det_github",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "start": "2", "stop": "4", "stride": "1"},
        },
    ),
    (
        Index114,
        [((1, 3, 640, 640), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolox_yolox_m_obj_det_torchhub",
                "pt_yolox_yolox_s_obj_det_torchhub",
                "pt_yolox_yolox_x_obj_det_torchhub",
                "pt_yolox_yolox_l_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "start": "0", "stop": "640", "stride": "2"},
        },
    ),
    (
        Index115,
        [((1, 3, 640, 640), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolox_yolox_m_obj_det_torchhub",
                "pt_yolox_yolox_s_obj_det_torchhub",
                "pt_yolox_yolox_x_obj_det_torchhub",
                "pt_yolox_yolox_l_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "start": "1", "stop": "640", "stride": "2"},
        },
    ),
    (
        Index116,
        [((1, 3, 320, 640), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolox_yolox_m_obj_det_torchhub",
                "pt_yolox_yolox_s_obj_det_torchhub",
                "pt_yolox_yolox_x_obj_det_torchhub",
                "pt_yolox_yolox_l_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "start": "0", "stop": "640", "stride": "2"},
        },
    ),
    (
        Index117,
        [((1, 3, 320, 640), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolox_yolox_m_obj_det_torchhub",
                "pt_yolox_yolox_s_obj_det_torchhub",
                "pt_yolox_yolox_x_obj_det_torchhub",
                "pt_yolox_yolox_l_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "start": "1", "stop": "640", "stride": "2"},
        },
    ),
    (
        Index118,
        [((1, 16, 112, 112), torch.float32)],
        {
            "model_names": ["onnx_ghostnet_ghostnet_100_in1k_img_cls_timm", "onnx_ghostnet_ghostnet_100_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "16", "stride": "1"},
        },
    ),
    (
        Index119,
        [((1, 48, 112, 112), torch.float32)],
        {
            "model_names": ["onnx_ghostnet_ghostnet_100_in1k_img_cls_timm", "onnx_ghostnet_ghostnet_100_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "48", "stride": "1"},
        },
    ),
    (
        Index120,
        [((1, 24, 56, 56), torch.float32)],
        {
            "model_names": ["onnx_ghostnet_ghostnet_100_in1k_img_cls_timm", "onnx_ghostnet_ghostnet_100_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "24", "stride": "1"},
        },
    ),
    (
        Index54,
        [((1, 72, 56, 56), torch.float32)],
        {
            "model_names": ["onnx_ghostnet_ghostnet_100_in1k_img_cls_timm", "onnx_ghostnet_ghostnet_100_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "72", "stride": "1"},
        },
    ),
    (
        Index121,
        [((1, 40, 28, 28), torch.float32)],
        {
            "model_names": ["onnx_ghostnet_ghostnet_100_in1k_img_cls_timm", "onnx_ghostnet_ghostnet_100_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "40", "stride": "1"},
        },
    ),
    (
        Index122,
        [((1, 120, 28, 28), torch.float32)],
        {
            "model_names": ["onnx_ghostnet_ghostnet_100_in1k_img_cls_timm", "onnx_ghostnet_ghostnet_100_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "120", "stride": "1"},
        },
    ),
    (
        Index123,
        [((1, 120, 28, 28), torch.float32)],
        {
            "model_names": ["onnx_hrnet_hrnetv2_w30_img_cls_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "60", "stride": "1"},
        },
    ),
    (
        Index124,
        [((1, 120, 28, 28), torch.float32)],
        {
            "model_names": ["onnx_hrnet_hrnetv2_w30_img_cls_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "60", "stop": "90", "stride": "1"},
        },
    ),
    (
        Index125,
        [((1, 120, 28, 28), torch.float32)],
        {
            "model_names": ["onnx_hrnet_hrnetv2_w30_img_cls_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "90", "stop": "120", "stride": "1"},
        },
    ),
    (
        Index126,
        [((1, 120, 28, 28), torch.float32)],
        {
            "model_names": ["onnx_hrnet_hrnetv2_w30_img_cls_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "30", "stride": "1"},
        },
    ),
    (
        Index127,
        [((1, 120, 28, 28), torch.float32)],
        {
            "model_names": ["onnx_hrnet_hrnetv2_w30_img_cls_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "30", "stop": "60", "stride": "1"},
        },
    ),
    (
        Index128,
        [((1, 120, 28, 28), torch.float32)],
        {
            "model_names": ["onnx_hrnet_hrnetv2_w30_img_cls_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "60", "stop": "120", "stride": "1"},
        },
    ),
    (
        Index129,
        [((1, 240, 28, 28), torch.float32)],
        {
            "model_names": ["onnx_ghostnet_ghostnet_100_in1k_img_cls_timm", "onnx_ghostnet_ghostnet_100_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "240", "stride": "1"},
        },
    ),
    (
        Index104,
        [((1, 80, 14, 14), torch.float32)],
        {
            "model_names": ["onnx_ghostnet_ghostnet_100_in1k_img_cls_timm", "onnx_ghostnet_ghostnet_100_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "80", "stride": "1"},
        },
    ),
    (
        Index130,
        [((1, 200, 14, 14), torch.float32)],
        {
            "model_names": ["onnx_ghostnet_ghostnet_100_in1k_img_cls_timm", "onnx_ghostnet_ghostnet_100_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "200", "stride": "1"},
        },
    ),
    (
        Index131,
        [((1, 184, 14, 14), torch.float32)],
        {
            "model_names": ["onnx_ghostnet_ghostnet_100_in1k_img_cls_timm", "onnx_ghostnet_ghostnet_100_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "184", "stride": "1"},
        },
    ),
    (
        Index132,
        [((1, 480, 14, 14), torch.float32)],
        {
            "model_names": ["onnx_ghostnet_ghostnet_100_in1k_img_cls_timm", "onnx_ghostnet_ghostnet_100_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "480", "stride": "1"},
        },
    ),
    (
        Index133,
        [((1, 112, 14, 14), torch.float32)],
        {
            "model_names": ["onnx_ghostnet_ghostnet_100_in1k_img_cls_timm", "onnx_ghostnet_ghostnet_100_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "112", "stride": "1"},
        },
    ),
    (
        Index134,
        [((1, 672, 14, 14), torch.float32)],
        {
            "model_names": ["onnx_ghostnet_ghostnet_100_in1k_img_cls_timm", "onnx_ghostnet_ghostnet_100_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "672", "stride": "1"},
        },
    ),
    (
        Index106,
        [((1, 160, 7, 7), torch.float32)],
        {
            "model_names": ["onnx_ghostnet_ghostnet_100_in1k_img_cls_timm", "onnx_ghostnet_ghostnet_100_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "160", "stride": "1"},
        },
    ),
    (
        Index135,
        [((1, 960, 7, 7), torch.float32)],
        {
            "model_names": ["onnx_ghostnet_ghostnet_100_in1k_img_cls_timm", "onnx_ghostnet_ghostnet_100_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "960", "stride": "1"},
        },
    ),
    (
        Index30,
        [((1, 13, 384), torch.float32)],
        {
            "model_names": ["onnx_minilm_sentence_transformers_all_minilm_l6_v2_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index118,
        [((1, 32, 160, 160), torch.float32)],
        {
            "model_names": ["onnx_yolov10_default_obj_det_github", "onnx_yolov8_default_obj_det_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "16", "stride": "1"},
        },
    ),
    (
        Index136,
        [((1, 32, 160, 160), torch.float32)],
        {
            "model_names": ["onnx_yolov10_default_obj_det_github", "onnx_yolov8_default_obj_det_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "16", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index137,
        [((1, 64, 80, 80), torch.float32)],
        {
            "model_names": ["onnx_yolov10_default_obj_det_github", "onnx_yolov8_default_obj_det_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index138,
        [((1, 64, 80, 80), torch.float32)],
        {
            "model_names": ["onnx_yolov10_default_obj_det_github", "onnx_yolov8_default_obj_det_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "32", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index6,
        [((1, 128, 40, 40), torch.float32)],
        {
            "model_names": ["onnx_yolov10_default_obj_det_github", "onnx_yolov8_default_obj_det_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index17,
        [((1, 128, 40, 40), torch.float32)],
        {
            "model_names": ["onnx_yolov10_default_obj_det_github", "onnx_yolov8_default_obj_det_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "64", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index14,
        [((1, 256, 20, 20), torch.float32)],
        {
            "model_names": ["onnx_yolov10_default_obj_det_github", "onnx_yolov8_default_obj_det_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index18,
        [((1, 256, 20, 20), torch.float32)],
        {
            "model_names": ["onnx_yolov10_default_obj_det_github", "onnx_yolov8_default_obj_det_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "128", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index139,
        [((1, 2, 128, 400), torch.float32)],
        {
            "model_names": ["onnx_yolov10_default_obj_det_github"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "64", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index140,
        [((1, 2, 128, 400), torch.float32)],
        {
            "model_names": ["onnx_yolov10_default_obj_det_github"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index141,
        [((1, 2, 128, 400), torch.float32)],
        {
            "model_names": ["onnx_yolov10_default_obj_det_github"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "32", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index110,
        [((1, 144, 8400), torch.float32)],
        {
            "model_names": [
                "onnx_yolov10_default_obj_det_github",
                "onnx_yolov8_default_obj_det_github",
                "onnx_yolov9_default_obj_det_github",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index111,
        [((1, 144, 8400), torch.float32)],
        {
            "model_names": [
                "onnx_yolov10_default_obj_det_github",
                "onnx_yolov8_default_obj_det_github",
                "onnx_yolov9_default_obj_det_github",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "64", "stop": "144", "stride": "1"},
        },
    ),
    (
        Index112,
        [((1, 4, 8400), torch.float32)],
        {
            "model_names": [
                "onnx_yolov10_default_obj_det_github",
                "onnx_yolov8_default_obj_det_github",
                "onnx_yolov9_default_obj_det_github",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index113,
        [((1, 4, 8400), torch.float32)],
        {
            "model_names": [
                "onnx_yolov10_default_obj_det_github",
                "onnx_yolov8_default_obj_det_github",
                "onnx_yolov9_default_obj_det_github",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "2", "stop": "4", "stride": "1"},
        },
    ),
    (
        Index101,
        [((2, 1, 14), torch.float32)],
        {
            "model_names": ["pd_bert_bert_base_japanese_qa_padlenlp"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index102,
        [((2, 1, 14), torch.float32)],
        {
            "model_names": ["pd_bert_bert_base_japanese_qa_padlenlp"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index101,
        [((2, 1, 9), torch.float32)],
        {
            "model_names": ["pd_bert_bert_base_uncased_qa_padlenlp", "pd_ernie_1_0_qa_padlenlp"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index102,
        [((2, 1, 9), torch.float32)],
        {
            "model_names": ["pd_bert_bert_base_uncased_qa_padlenlp", "pd_ernie_1_0_qa_padlenlp"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index30,
        [((1, 197, 768), torch.float32)],
        {
            "model_names": [
                "pd_clip_vision_openai_clip_vit_base_patch16_img_enc_padlenlp",
                "pd_chineseclip_ofa_sys_chinese_clip_vit_base_patch16_img_text_pairing_padlenlp",
                "onnx_deit_facebook_deit_base_patch16_224_img_cls_hf",
                "onnx_vit_base_google_vit_base_patch16_224_img_cls_hf",
                "pd_chineseclip_vision_ofa_sys_chinese_clip_vit_base_patch16_img_enc_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index142,
        [((1, 197, 768), torch.float32)],
        {
            "model_names": ["onnx_beit_microsoft_beit_base_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "1", "stop": "197", "stride": "1"},
        },
    ),
    (
        Index101,
        [((25, 1, 288), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index102,
        [((25, 1, 288), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index103,
        [((25, 1, 288), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "2", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index143,
        [((25, 1, 288), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "3", "stop": "4", "stride": "1"},
        },
    ),
    (
        Index144,
        [((25, 1, 288), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "4", "stop": "5", "stride": "1"},
        },
    ),
    (
        Index145,
        [((25, 1, 288), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "5", "stop": "6", "stride": "1"},
        },
    ),
    (
        Index146,
        [((25, 1, 288), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "6", "stop": "7", "stride": "1"},
        },
    ),
    (
        Index147,
        [((25, 1, 288), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "7", "stop": "8", "stride": "1"},
        },
    ),
    (
        Index148,
        [((25, 1, 288), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "8", "stop": "9", "stride": "1"},
        },
    ),
    (
        Index149,
        [((25, 1, 288), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "9", "stop": "10", "stride": "1"},
        },
    ),
    (
        Index150,
        [((25, 1, 288), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "10", "stop": "11", "stride": "1"},
        },
    ),
    (
        Index151,
        [((25, 1, 288), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "11", "stop": "12", "stride": "1"},
        },
    ),
    (
        Index152,
        [((25, 1, 288), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "12", "stop": "13", "stride": "1"},
        },
    ),
    (
        Index153,
        [((25, 1, 288), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "13", "stop": "14", "stride": "1"},
        },
    ),
    (
        Index154,
        [((25, 1, 288), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "14", "stop": "15", "stride": "1"},
        },
    ),
    (
        Index155,
        [((25, 1, 288), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "15", "stop": "16", "stride": "1"},
        },
    ),
    (
        Index156,
        [((25, 1, 288), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "16", "stop": "17", "stride": "1"},
        },
    ),
    (
        Index157,
        [((25, 1, 288), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "17", "stop": "18", "stride": "1"},
        },
    ),
    (
        Index158,
        [((25, 1, 288), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "18", "stop": "19", "stride": "1"},
        },
    ),
    (
        Index159,
        [((25, 1, 288), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "19", "stop": "20", "stride": "1"},
        },
    ),
    (
        Index160,
        [((25, 1, 288), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "20", "stop": "21", "stride": "1"},
        },
    ),
    (
        Index161,
        [((25, 1, 288), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "21", "stop": "22", "stride": "1"},
        },
    ),
    (
        Index162,
        [((25, 1, 288), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "22", "stop": "23", "stride": "1"},
        },
    ),
    (
        Index163,
        [((25, 1, 288), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "23", "stop": "24", "stride": "1"},
        },
    ),
    (
        Index164,
        [((25, 1, 288), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "24", "stop": "25", "stride": "1"},
        },
    ),
    (
        Index165,
        [((1, 192), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "144", "stop": "192", "stride": "1"},
        },
    ),
    (
        Index166,
        [((1, 192), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "48", "stop": "96", "stride": "1"},
        },
    ),
    (
        Index167,
        [((1, 192), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "48", "stride": "1"},
        },
    ),
    (
        Index168,
        [((1, 192), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "96", "stop": "144", "stride": "1"},
        },
    ),
    (
        Index101,
        [((25, 1, 96), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index102,
        [((25, 1, 96), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index103,
        [((25, 1, 96), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "2", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index143,
        [((25, 1, 96), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "3", "stop": "4", "stride": "1"},
        },
    ),
    (
        Index144,
        [((25, 1, 96), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "4", "stop": "5", "stride": "1"},
        },
    ),
    (
        Index145,
        [((25, 1, 96), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "5", "stop": "6", "stride": "1"},
        },
    ),
    (
        Index146,
        [((25, 1, 96), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "6", "stop": "7", "stride": "1"},
        },
    ),
    (
        Index147,
        [((25, 1, 96), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "7", "stop": "8", "stride": "1"},
        },
    ),
    (
        Index148,
        [((25, 1, 96), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "8", "stop": "9", "stride": "1"},
        },
    ),
    (
        Index149,
        [((25, 1, 96), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "9", "stop": "10", "stride": "1"},
        },
    ),
    (
        Index150,
        [((25, 1, 96), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "10", "stop": "11", "stride": "1"},
        },
    ),
    (
        Index151,
        [((25, 1, 96), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "11", "stop": "12", "stride": "1"},
        },
    ),
    (
        Index152,
        [((25, 1, 96), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "12", "stop": "13", "stride": "1"},
        },
    ),
    (
        Index153,
        [((25, 1, 96), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "13", "stop": "14", "stride": "1"},
        },
    ),
    (
        Index154,
        [((25, 1, 96), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "14", "stop": "15", "stride": "1"},
        },
    ),
    (
        Index155,
        [((25, 1, 96), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "15", "stop": "16", "stride": "1"},
        },
    ),
    (
        Index156,
        [((25, 1, 96), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "16", "stop": "17", "stride": "1"},
        },
    ),
    (
        Index157,
        [((25, 1, 96), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "17", "stop": "18", "stride": "1"},
        },
    ),
    (
        Index158,
        [((25, 1, 96), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "18", "stop": "19", "stride": "1"},
        },
    ),
    (
        Index159,
        [((25, 1, 96), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "19", "stop": "20", "stride": "1"},
        },
    ),
    (
        Index160,
        [((25, 1, 96), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "20", "stop": "21", "stride": "1"},
        },
    ),
    (
        Index161,
        [((25, 1, 96), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "21", "stop": "22", "stride": "1"},
        },
    ),
    (
        Index162,
        [((25, 1, 96), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "22", "stop": "23", "stride": "1"},
        },
    ),
    (
        Index163,
        [((25, 1, 96), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "23", "stop": "24", "stride": "1"},
        },
    ),
    (
        Index164,
        [((25, 1, 96), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "24", "stop": "25", "stride": "1"},
        },
    ),
    (
        Index169,
        [((732, 16), torch.bfloat16)],
        {
            "model_names": ["pt_beit_large_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "start": "0", "stop": "729", "stride": "1"},
        },
    ),
    (
        Index170,
        [((732, 16), torch.bfloat16)],
        {
            "model_names": ["pt_beit_large_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "start": "729", "stop": "732", "stride": "1"},
        },
    ),
    (
        Index142,
        [((1, 197, 1024), torch.bfloat16)],
        {
            "model_names": ["pt_beit_large_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "start": "1", "stop": "197", "stride": "1"},
        },
    ),
    (
        Index30,
        [((1, 197, 1024), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_l_16_img_cls_torchvision", "pt_vit_large_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index171,
        [((1, 16, 588, 128), torch.float32)],
        {
            "model_names": ["pt_deepseek_1_3b_instruct_qa_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "64", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index172,
        [((1, 16, 588, 128), torch.float32)],
        {
            "model_names": ["pt_deepseek_1_3b_instruct_qa_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index137,
        [((1, 224, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w32_osmr_img_cls_osmr", "pt_hrnet_hrnet_w32_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "0", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index173,
        [((1, 224, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w32_osmr_img_cls_osmr", "pt_hrnet_hrnet_w32_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "32", "stop": "96", "stride": "1"},
        },
    ),
    (
        Index174,
        [((1, 224, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w32_osmr_img_cls_osmr", "pt_hrnet_hrnet_w32_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "96", "stop": "224", "stride": "1"},
        },
    ),
    (
        Index14,
        [((1, 224, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w32_osmr_img_cls_osmr", "pt_hrnet_hrnet_w32_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "0", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index15,
        [((1, 224, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w32_osmr_img_cls_osmr", "pt_hrnet_hrnet_w32_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "128", "stop": "192", "stride": "1"},
        },
    ),
    (
        Index175,
        [((1, 224, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w32_osmr_img_cls_osmr", "pt_hrnet_hrnet_w32_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "192", "stop": "224", "stride": "1"},
        },
    ),
    (
        Index6,
        [((1, 128, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w32_osmr_img_cls_osmr", "pt_hrnet_hrnet_w32_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index176,
        [((1, 128, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w32_osmr_img_cls_osmr", "pt_hrnet_hrnet_w32_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "64", "stop": "96", "stride": "1"},
        },
    ),
    (
        Index177,
        [((1, 128, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w32_osmr_img_cls_osmr", "pt_hrnet_hrnet_w32_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "96", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index137,
        [((1, 128, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w32_osmr_img_cls_osmr", "pt_hrnet_hrnet_w32_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "0", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index138,
        [((1, 128, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w32_osmr_img_cls_osmr", "pt_hrnet_hrnet_w32_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "32", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index17,
        [((1, 128, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w32_osmr_img_cls_osmr", "pt_hrnet_hrnet_w32_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "64", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index178,
        [((1, 1, 1024, 72), torch.float32)],
        {
            "model_names": [
                "pt_nbeats_generic_basis_time_series_forecasting_github",
                "pt_nbeats_trend_basis_time_series_forecasting_github",
                "pt_nbeats_seasonality_basis_time_series_forecasting_github",
                "onnx_nbeats_seasionality_basis_time_series_forecasting_github",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "-1", "stop": "72", "stride": "1"},
        },
    ),
    (
        Index179,
        [((1, 1, 1024, 72), torch.float32)],
        {
            "model_names": [
                "pt_nbeats_generic_basis_time_series_forecasting_github",
                "pt_nbeats_trend_basis_time_series_forecasting_github",
                "onnx_nbeats_seasionality_basis_time_series_forecasting_github",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "-1", "stop": "-9223372036854775807", "stride": "-1"},
        },
    ),
    (
        Index180,
        [((1024, 96), torch.float32)],
        {
            "model_names": ["pt_nbeats_generic_basis_time_series_forecasting_github"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "-24", "stop": "96", "stride": "1"},
        },
    ),
    (
        Index181,
        [((1024, 96), torch.float32)],
        {
            "model_names": ["pt_nbeats_generic_basis_time_series_forecasting_github"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "72", "stride": "1"},
        },
    ),
    (
        Index46,
        [((1, 32, 12, 64), torch.float32)],
        {
            "model_names": ["pt_phi_1_5_microsoft_phi_1_5_token_cls_hf", "pt_phi1_microsoft_phi_1_token_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index47,
        [((1, 32, 12, 64), torch.float32)],
        {
            "model_names": ["pt_phi_1_5_microsoft_phi_1_5_token_cls_hf", "pt_phi1_microsoft_phi_1_token_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "32", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index63,
        [((1, 32, 12, 32), torch.float32)],
        {
            "model_names": [
                "pt_phi_1_5_microsoft_phi_1_5_token_cls_hf",
                "pt_phi1_microsoft_phi_1_token_cls_hf",
                "pt_phi2_microsoft_phi_2_pytdml_token_cls_hf",
                "pt_phi2_microsoft_phi_2_token_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "16", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index27,
        [((1, 32, 12, 32), torch.float32)],
        {
            "model_names": [
                "pt_phi_1_5_microsoft_phi_1_5_token_cls_hf",
                "pt_phi1_microsoft_phi_1_token_cls_hf",
                "pt_phi2_microsoft_phi_2_pytdml_token_cls_hf",
                "pt_phi2_microsoft_phi_2_token_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "16", "stride": "1"},
        },
    ),
    (
        Index23,
        [((1, 514), torch.int64)],
        {
            "model_names": ["pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index182,
        [((1, 514), torch.int64)],
        {
            "model_names": ["pt_roberta_xlm_base_mlm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "10", "stride": "1"},
        },
    ),
    (
        Index30,
        [((1, 128, 768), torch.float32)],
        {
            "model_names": [
                "pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf",
                "onnx_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_squeezebert_squeezebert_mnli_seq_cls_hf",
                "pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder",
                "onnx_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf",
                "pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder",
                "pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index19,
        [((3, 64, 3, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_t_img_cls_torchvision", "pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-5", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index20,
        [((3, 64, 3, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_t_img_cls_torchvision", "pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-5", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index21,
        [((3, 64, 3, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_t_img_cls_torchvision", "pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-5", "start": "2", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index66,
        [((1, 56, 56, 96), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "3", "stop": "56", "stride": "1"},
        },
    ),
    (
        Index67,
        [((1, 56, 56, 96), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "0", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index68,
        [((1, 56, 56, 96), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "start": "3", "stop": "56", "stride": "1"},
        },
    ),
    (
        Index69,
        [((1, 56, 56, 96), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "start": "0", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index70,
        [((1, 56, 56, 96), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "53", "stop": "56", "stride": "1"},
        },
    ),
    (
        Index71,
        [((1, 56, 56, 96), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "0", "stop": "53", "stride": "1"},
        },
    ),
    (
        Index72,
        [((1, 56, 56, 96), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "start": "53", "stop": "56", "stride": "1"},
        },
    ),
    (
        Index73,
        [((1, 56, 56, 96), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "start": "0", "stop": "53", "stride": "1"},
        },
    ),
    (
        Index74,
        [((1, 56, 56, 96), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "0", "stop": "56", "stride": "2"},
        },
    ),
    (
        Index75,
        [((1, 56, 56, 96), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "1", "stop": "56", "stride": "2"},
        },
    ),
    (
        Index76,
        [((1, 28, 56, 96), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "start": "0", "stop": "56", "stride": "2"},
        },
    ),
    (
        Index77,
        [((1, 28, 56, 96), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "start": "1", "stop": "56", "stride": "2"},
        },
    ),
    (
        Index19,
        [((3, 16, 6, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_t_img_cls_torchvision", "pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-5", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index20,
        [((3, 16, 6, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_t_img_cls_torchvision", "pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-5", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index21,
        [((3, 16, 6, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_t_img_cls_torchvision", "pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-5", "start": "2", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index78,
        [((1, 28, 28, 192), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "3", "stop": "28", "stride": "1"},
        },
    ),
    (
        Index67,
        [((1, 28, 28, 192), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "0", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index79,
        [((1, 28, 28, 192), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "start": "3", "stop": "28", "stride": "1"},
        },
    ),
    (
        Index69,
        [((1, 28, 28, 192), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "start": "0", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index80,
        [((1, 28, 28, 192), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "25", "stop": "28", "stride": "1"},
        },
    ),
    (
        Index81,
        [((1, 28, 28, 192), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "0", "stop": "25", "stride": "1"},
        },
    ),
    (
        Index82,
        [((1, 28, 28, 192), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "start": "25", "stop": "28", "stride": "1"},
        },
    ),
    (
        Index83,
        [((1, 28, 28, 192), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "start": "0", "stop": "25", "stride": "1"},
        },
    ),
    (
        Index84,
        [((1, 28, 28, 192), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "0", "stop": "28", "stride": "2"},
        },
    ),
    (
        Index85,
        [((1, 28, 28, 192), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "1", "stop": "28", "stride": "2"},
        },
    ),
    (
        Index86,
        [((1, 14, 28, 192), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "start": "0", "stop": "28", "stride": "2"},
        },
    ),
    (
        Index87,
        [((1, 14, 28, 192), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "start": "1", "stop": "28", "stride": "2"},
        },
    ),
    (
        Index19,
        [((3, 4, 12, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_t_img_cls_torchvision", "pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-5", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index20,
        [((3, 4, 12, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_t_img_cls_torchvision", "pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-5", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index21,
        [((3, 4, 12, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_t_img_cls_torchvision", "pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-5", "start": "2", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index88,
        [((1, 14, 14, 384), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "3", "stop": "14", "stride": "1"},
        },
    ),
    (
        Index67,
        [((1, 14, 14, 384), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "0", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index89,
        [((1, 14, 14, 384), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "start": "3", "stop": "14", "stride": "1"},
        },
    ),
    (
        Index69,
        [((1, 14, 14, 384), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "start": "0", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index90,
        [((1, 14, 14, 384), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "11", "stop": "14", "stride": "1"},
        },
    ),
    (
        Index91,
        [((1, 14, 14, 384), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "0", "stop": "11", "stride": "1"},
        },
    ),
    (
        Index92,
        [((1, 14, 14, 384), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "start": "11", "stop": "14", "stride": "1"},
        },
    ),
    (
        Index93,
        [((1, 14, 14, 384), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "start": "0", "stop": "11", "stride": "1"},
        },
    ),
    (
        Index94,
        [((1, 14, 14, 384), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "0", "stop": "14", "stride": "2"},
        },
    ),
    (
        Index95,
        [((1, 14, 14, 384), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "1", "stop": "14", "stride": "2"},
        },
    ),
    (
        Index96,
        [((1, 7, 14, 384), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "start": "0", "stop": "14", "stride": "2"},
        },
    ),
    (
        Index97,
        [((1, 7, 14, 384), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "start": "1", "stop": "14", "stride": "2"},
        },
    ),
    (
        Index19,
        [((3, 1, 24, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_t_img_cls_torchvision", "pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-5", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index20,
        [((3, 1, 24, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_t_img_cls_torchvision", "pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-5", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index21,
        [((3, 1, 24, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_t_img_cls_torchvision", "pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-5", "start": "2", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index98,
        [((1, 3, 10, 10, 85), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5s_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_320x320",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "start": "0", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index99,
        [((1, 3, 10, 10, 85), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5s_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_320x320",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "start": "2", "stop": "4", "stride": "1"},
        },
    ),
    (
        Index100,
        [((1, 3, 10, 10, 85), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5s_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_320x320",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "start": "4", "stop": "85", "stride": "1"},
        },
    ),
    (
        Index183,
        [((1, 3, 416, 416), torch.bfloat16)],
        {
            "model_names": ["pt_yolox_yolox_tiny_obj_det_torchhub", "pt_yolox_yolox_nano_obj_det_torchhub"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "start": "0", "stop": "416", "stride": "2"},
        },
    ),
    (
        Index184,
        [((1, 3, 416, 416), torch.bfloat16)],
        {
            "model_names": ["pt_yolox_yolox_tiny_obj_det_torchhub", "pt_yolox_yolox_nano_obj_det_torchhub"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "start": "1", "stop": "416", "stride": "2"},
        },
    ),
    (
        Index185,
        [((1, 3, 208, 416), torch.bfloat16)],
        {
            "model_names": ["pt_yolox_yolox_tiny_obj_det_torchhub", "pt_yolox_yolox_nano_obj_det_torchhub"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "start": "0", "stop": "416", "stride": "2"},
        },
    ),
    (
        Index186,
        [((1, 3, 208, 416), torch.bfloat16)],
        {
            "model_names": ["pt_yolox_yolox_tiny_obj_det_torchhub", "pt_yolox_yolox_nano_obj_det_torchhub"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "start": "1", "stop": "416", "stride": "2"},
        },
    ),
    (
        Index126,
        [((1, 210, 7, 7), torch.float32)],
        {
            "model_names": ["onnx_hrnet_hrnetv2_w30_img_cls_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "30", "stride": "1"},
        },
    ),
    (
        Index187,
        [((1, 210, 7, 7), torch.float32)],
        {
            "model_names": ["onnx_hrnet_hrnetv2_w30_img_cls_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "30", "stop": "90", "stride": "1"},
        },
    ),
    (
        Index188,
        [((1, 210, 7, 7), torch.float32)],
        {
            "model_names": ["onnx_hrnet_hrnetv2_w30_img_cls_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "90", "stop": "210", "stride": "1"},
        },
    ),
    (
        Index122,
        [((1, 210, 7, 7), torch.float32)],
        {
            "model_names": ["onnx_hrnet_hrnetv2_w30_img_cls_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "120", "stride": "1"},
        },
    ),
    (
        Index189,
        [((1, 210, 7, 7), torch.float32)],
        {
            "model_names": ["onnx_hrnet_hrnetv2_w30_img_cls_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "120", "stop": "180", "stride": "1"},
        },
    ),
    (
        Index190,
        [((1, 210, 7, 7), torch.float32)],
        {
            "model_names": ["onnx_hrnet_hrnetv2_w30_img_cls_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "180", "stop": "210", "stride": "1"},
        },
    ),
    (
        Index191,
        [((1, 224, 35, 35), torch.float32)],
        {
            "model_names": [
                "onnx_inception_inception_v4_img_cls_timm",
                "onnx_inception_inception_v4_tf_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "96", "stride": "1"},
        },
    ),
    (
        Index192,
        [((1, 224, 35, 35), torch.float32)],
        {
            "model_names": [
                "onnx_inception_inception_v4_img_cls_timm",
                "onnx_inception_inception_v4_tf_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "96", "stop": "160", "stride": "1"},
        },
    ),
    (
        Index193,
        [((1, 224, 35, 35), torch.float32)],
        {
            "model_names": [
                "onnx_inception_inception_v4_img_cls_timm",
                "onnx_inception_inception_v4_tf_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "160", "stop": "224", "stride": "1"},
        },
    ),
    (
        Index194,
        [((1, 768, 17, 17), torch.float32)],
        {
            "model_names": [
                "onnx_inception_inception_v4_img_cls_timm",
                "onnx_inception_inception_v4_tf_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "384", "stride": "1"},
        },
    ),
    (
        Index195,
        [((1, 768, 17, 17), torch.float32)],
        {
            "model_names": [
                "onnx_inception_inception_v4_img_cls_timm",
                "onnx_inception_inception_v4_tf_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "384", "stop": "576", "stride": "1"},
        },
    ),
    (
        Index196,
        [((1, 768, 17, 17), torch.float32)],
        {
            "model_names": [
                "onnx_inception_inception_v4_img_cls_timm",
                "onnx_inception_inception_v4_tf_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "576", "stop": "768", "stride": "1"},
        },
    ),
    (
        Index9,
        [((1, 1024, 8, 8), torch.float32)],
        {
            "model_names": [
                "onnx_inception_inception_v4_img_cls_timm",
                "onnx_inception_inception_v4_tf_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index197,
        [((1, 1024, 8, 8), torch.float32)],
        {
            "model_names": [
                "onnx_inception_inception_v4_img_cls_timm",
                "onnx_inception_inception_v4_tf_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "256", "stop": "640", "stride": "1"},
        },
    ),
    (
        Index198,
        [((1, 1024, 8, 8), torch.float32)],
        {
            "model_names": [
                "onnx_inception_inception_v4_img_cls_timm",
                "onnx_inception_inception_v4_tf_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "640", "stop": "1024", "stride": "1"},
        },
    ),
    (
        Index19,
        [((3, 1, 12, 257, 64), torch.float32)],
        {
            "model_names": ["onnx_mgp_alibaba_damo_mgp_str_base_scene_text_recognition_hf"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index20,
        [((3, 1, 12, 257, 64), torch.float32)],
        {
            "model_names": ["onnx_mgp_alibaba_damo_mgp_str_base_scene_text_recognition_hf"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index21,
        [((3, 1, 12, 257, 64), torch.float32)],
        {
            "model_names": ["onnx_mgp_alibaba_damo_mgp_str_base_scene_text_recognition_hf"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "2", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index199,
        [((1, 8, 1, 1), torch.float32)],
        {
            "model_names": ["onnx_t5_t5_small_text_gen_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "-1", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index200,
        [((1, 8, 61, 61), torch.float32)],
        {
            "model_names": ["onnx_t5_t5_small_text_gen_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "-61", "stop": "61", "stride": "1"},
        },
    ),
    (
        Index46,
        [((1, 5, 16, 64), torch.float32)],
        {
            "model_names": [
                "pt_codegen_salesforce_codegen_350m_mono_clm_hf",
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index47,
        [((1, 5, 16, 64), torch.float32)],
        {
            "model_names": [
                "pt_codegen_salesforce_codegen_350m_mono_clm_hf",
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "32", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index48,
        [((1, 5, 16, 32), torch.float32)],
        {
            "model_names": [
                "pt_codegen_salesforce_codegen_350m_mono_clm_hf",
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "1", "stop": "32", "stride": "2"},
        },
    ),
    (
        Index49,
        [((1, 5, 16, 32), torch.float32)],
        {
            "model_names": [
                "pt_codegen_salesforce_codegen_350m_mono_clm_hf",
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "32", "stride": "2"},
        },
    ),
    (
        Index101,
        [((1, 3, 224, 224), torch.bfloat16)],
        {
            "model_names": ["pt_googlenet_googlenet_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index102,
        [((1, 3, 224, 224), torch.bfloat16)],
        {
            "model_names": ["pt_googlenet_googlenet_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index103,
        [((1, 3, 224, 224), torch.bfloat16)],
        {
            "model_names": ["pt_googlenet_googlenet_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "2", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index6,
        [((1, 176, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_googlenet_googlenet_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index201,
        [((1, 176, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_googlenet_googlenet_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "64", "stop": "160", "stride": "1"},
        },
    ),
    (
        Index202,
        [((1, 176, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_googlenet_googlenet_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "160", "stop": "176", "stride": "1"},
        },
    ),
    (
        Index203,
        [((1, 176, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w44_img_cls_timm", "pt_hrnet_hrnetv2_w44_osmr_img_cls_osmr"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "0", "stop": "88", "stride": "1"},
        },
    ),
    (
        Index204,
        [((1, 176, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w44_img_cls_timm", "pt_hrnet_hrnetv2_w44_osmr_img_cls_osmr"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "88", "stop": "132", "stride": "1"},
        },
    ),
    (
        Index205,
        [((1, 176, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w44_img_cls_timm", "pt_hrnet_hrnetv2_w44_osmr_img_cls_osmr"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "132", "stop": "176", "stride": "1"},
        },
    ),
    (
        Index206,
        [((1, 176, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w44_img_cls_timm", "pt_hrnet_hrnetv2_w44_osmr_img_cls_osmr"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "0", "stop": "44", "stride": "1"},
        },
    ),
    (
        Index207,
        [((1, 176, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w44_img_cls_timm", "pt_hrnet_hrnetv2_w44_osmr_img_cls_osmr"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "44", "stop": "88", "stride": "1"},
        },
    ),
    (
        Index208,
        [((1, 176, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w44_img_cls_timm", "pt_hrnet_hrnetv2_w44_osmr_img_cls_osmr"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "88", "stop": "176", "stride": "1"},
        },
    ),
    (
        Index14,
        [((1, 288, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_googlenet_googlenet_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "0", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index18,
        [((1, 288, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_googlenet_googlenet_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "128", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index209,
        [((1, 288, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_googlenet_googlenet_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "256", "stop": "288", "stride": "1"},
        },
    ),
    (
        Index210,
        [((1, 304, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_googlenet_googlenet_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "0", "stop": "192", "stride": "1"},
        },
    ),
    (
        Index211,
        [((1, 304, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_googlenet_googlenet_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "192", "stop": "288", "stride": "1"},
        },
    ),
    (
        Index212,
        [((1, 304, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_googlenet_googlenet_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "288", "stop": "304", "stride": "1"},
        },
    ),
    (
        Index106,
        [((1, 296, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_googlenet_googlenet_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "0", "stop": "160", "stride": "1"},
        },
    ),
    (
        Index213,
        [((1, 296, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_googlenet_googlenet_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "160", "stop": "272", "stride": "1"},
        },
    ),
    (
        Index214,
        [((1, 296, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_googlenet_googlenet_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "272", "stop": "296", "stride": "1"},
        },
    ),
    (
        Index14,
        [((1, 280, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_googlenet_googlenet_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "0", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index18,
        [((1, 280, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_googlenet_googlenet_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "128", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index215,
        [((1, 280, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_googlenet_googlenet_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "256", "stop": "280", "stride": "1"},
        },
    ),
    (
        Index133,
        [((1, 288, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_googlenet_googlenet_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "0", "stop": "112", "stride": "1"},
        },
    ),
    (
        Index216,
        [((1, 288, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_googlenet_googlenet_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "112", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index209,
        [((1, 288, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_googlenet_googlenet_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "256", "stop": "288", "stride": "1"},
        },
    ),
    (
        Index9,
        [((1, 448, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_googlenet_googlenet_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "0", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index12,
        [((1, 448, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_googlenet_googlenet_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "256", "stop": "416", "stride": "1"},
        },
    ),
    (
        Index13,
        [((1, 448, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_googlenet_googlenet_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "416", "stop": "448", "stride": "1"},
        },
    ),
    (
        Index9,
        [((1, 448, 7, 7), torch.bfloat16)],
        {
            "model_names": [
                "pt_googlenet_googlenet_img_cls_torchvision",
                "pt_hrnet_hrnet_w64_img_cls_timm",
                "pt_hrnet_hrnetv2_w64_osmr_img_cls_osmr",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "0", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index12,
        [((1, 448, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_googlenet_googlenet_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "256", "stop": "416", "stride": "1"},
        },
    ),
    (
        Index13,
        [((1, 448, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_googlenet_googlenet_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "416", "stop": "448", "stride": "1"},
        },
    ),
    (
        Index6,
        [((1, 448, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w64_img_cls_timm", "pt_hrnet_hrnetv2_w64_osmr_img_cls_osmr"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index7,
        [((1, 448, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w64_img_cls_timm", "pt_hrnet_hrnetv2_w64_osmr_img_cls_osmr"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "64", "stop": "192", "stride": "1"},
        },
    ),
    (
        Index8,
        [((1, 448, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w64_img_cls_timm", "pt_hrnet_hrnetv2_w64_osmr_img_cls_osmr"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "192", "stop": "448", "stride": "1"},
        },
    ),
    (
        Index10,
        [((1, 448, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w64_img_cls_timm", "pt_hrnet_hrnetv2_w64_osmr_img_cls_osmr"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "256", "stop": "384", "stride": "1"},
        },
    ),
    (
        Index11,
        [((1, 448, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w64_img_cls_timm", "pt_hrnet_hrnetv2_w64_osmr_img_cls_osmr"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "384", "stop": "448", "stride": "1"},
        },
    ),
    (
        Index194,
        [((1, 624, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_googlenet_googlenet_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "0", "stop": "384", "stride": "1"},
        },
    ),
    (
        Index195,
        [((1, 624, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_googlenet_googlenet_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "384", "stop": "576", "stride": "1"},
        },
    ),
    (
        Index217,
        [((1, 624, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_googlenet_googlenet_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "576", "stop": "624", "stride": "1"},
        },
    ),
    (
        Index191,
        [((1, 224, 35, 35), torch.bfloat16)],
        {
            "model_names": [
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_inception_inceptionv4_img_cls_osmr",
                "pt_inception_inception_v4_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "0", "stop": "96", "stride": "1"},
        },
    ),
    (
        Index192,
        [((1, 224, 35, 35), torch.bfloat16)],
        {
            "model_names": [
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_inception_inceptionv4_img_cls_osmr",
                "pt_inception_inception_v4_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "96", "stop": "160", "stride": "1"},
        },
    ),
    (
        Index193,
        [((1, 224, 35, 35), torch.bfloat16)],
        {
            "model_names": [
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_inception_inceptionv4_img_cls_osmr",
                "pt_inception_inception_v4_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "160", "stop": "224", "stride": "1"},
        },
    ),
    (
        Index194,
        [((1, 768, 17, 17), torch.bfloat16)],
        {
            "model_names": [
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_inception_inceptionv4_img_cls_osmr",
                "pt_inception_inception_v4_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "0", "stop": "384", "stride": "1"},
        },
    ),
    (
        Index195,
        [((1, 768, 17, 17), torch.bfloat16)],
        {
            "model_names": [
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_inception_inceptionv4_img_cls_osmr",
                "pt_inception_inception_v4_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "384", "stop": "576", "stride": "1"},
        },
    ),
    (
        Index196,
        [((1, 768, 17, 17), torch.bfloat16)],
        {
            "model_names": [
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_inception_inceptionv4_img_cls_osmr",
                "pt_inception_inception_v4_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "576", "stop": "768", "stride": "1"},
        },
    ),
    (
        Index9,
        [((1, 1024, 8, 8), torch.bfloat16)],
        {
            "model_names": [
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_inception_inceptionv4_img_cls_osmr",
                "pt_inception_inception_v4_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "0", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index197,
        [((1, 1024, 8, 8), torch.bfloat16)],
        {
            "model_names": [
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_inception_inceptionv4_img_cls_osmr",
                "pt_inception_inception_v4_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "256", "stop": "640", "stride": "1"},
        },
    ),
    (
        Index198,
        [((1, 1024, 8, 8), torch.bfloat16)],
        {
            "model_names": [
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_inception_inceptionv4_img_cls_osmr",
                "pt_inception_inception_v4_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "640", "stop": "1024", "stride": "1"},
        },
    ),
    (
        Index171,
        [((1, 12, 29, 128), torch.float32)],
        {
            "model_names": ["pt_qwen_v2_1_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "64", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index172,
        [((1, 12, 29, 128), torch.float32)],
        {
            "model_names": ["pt_qwen_v2_1_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index171,
        [((1, 2, 29, 128), torch.float32)],
        {
            "model_names": ["pt_qwen_v2_1_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "64", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index172,
        [((1, 2, 29, 128), torch.float32)],
        {
            "model_names": ["pt_qwen_v2_1_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index19,
        [((3, 64, 4, 64, 32), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index20,
        [((3, 64, 4, 64, 32), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index21,
        [((3, 64, 4, 64, 32), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "2", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index218,
        [((1, 64, 64, 128), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "4", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index219,
        [((1, 64, 64, 128), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "4", "stride": "1"},
        },
    ),
    (
        Index220,
        [((1, 64, 64, 128), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "4", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index221,
        [((1, 64, 64, 128), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "4", "stride": "1"},
        },
    ),
    (
        Index222,
        [((1, 64, 64, 128), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "60", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index123,
        [((1, 64, 64, 128), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "60", "stride": "1"},
        },
    ),
    (
        Index223,
        [((1, 64, 64, 128), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "60", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index224,
        [((1, 64, 64, 128), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "60", "stride": "1"},
        },
    ),
    (
        Index225,
        [((1, 64, 64, 128), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "64", "stride": "2"},
        },
    ),
    (
        Index226,
        [((1, 64, 64, 128), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "1", "stop": "64", "stride": "2"},
        },
    ),
    (
        Index227,
        [((1, 32, 64, 128), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "64", "stride": "2"},
        },
    ),
    (
        Index228,
        [((1, 32, 64, 128), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "1", "stop": "64", "stride": "2"},
        },
    ),
    (
        Index19,
        [((3, 16, 8, 64, 32), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index20,
        [((3, 16, 8, 64, 32), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index21,
        [((3, 16, 8, 64, 32), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "2", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index229,
        [((1, 32, 32, 256), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "4", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index219,
        [((1, 32, 32, 256), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "4", "stride": "1"},
        },
    ),
    (
        Index230,
        [((1, 32, 32, 256), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "4", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index221,
        [((1, 32, 32, 256), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "4", "stride": "1"},
        },
    ),
    (
        Index231,
        [((1, 32, 32, 256), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "28", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index232,
        [((1, 32, 32, 256), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "28", "stride": "1"},
        },
    ),
    (
        Index233,
        [((1, 32, 32, 256), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "28", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index234,
        [((1, 32, 32, 256), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "28", "stride": "1"},
        },
    ),
    (
        Index235,
        [((1, 32, 32, 256), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "32", "stride": "2"},
        },
    ),
    (
        Index236,
        [((1, 32, 32, 256), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "1", "stop": "32", "stride": "2"},
        },
    ),
    (
        Index237,
        [((1, 16, 32, 256), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "32", "stride": "2"},
        },
    ),
    (
        Index238,
        [((1, 16, 32, 256), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "1", "stop": "32", "stride": "2"},
        },
    ),
    (
        Index19,
        [((3, 4, 16, 64, 32), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index20,
        [((3, 4, 16, 64, 32), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index21,
        [((3, 4, 16, 64, 32), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "2", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index239,
        [((1, 16, 16, 512), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "4", "stop": "16", "stride": "1"},
        },
    ),
    (
        Index219,
        [((1, 16, 16, 512), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "4", "stride": "1"},
        },
    ),
    (
        Index240,
        [((1, 16, 16, 512), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "4", "stop": "16", "stride": "1"},
        },
    ),
    (
        Index221,
        [((1, 16, 16, 512), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "4", "stride": "1"},
        },
    ),
    (
        Index241,
        [((1, 16, 16, 512), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "12", "stop": "16", "stride": "1"},
        },
    ),
    (
        Index242,
        [((1, 16, 16, 512), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "12", "stride": "1"},
        },
    ),
    (
        Index243,
        [((1, 16, 16, 512), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "12", "stop": "16", "stride": "1"},
        },
    ),
    (
        Index244,
        [((1, 16, 16, 512), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "12", "stride": "1"},
        },
    ),
    (
        Index245,
        [((1, 16, 16, 512), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "16", "stride": "2"},
        },
    ),
    (
        Index246,
        [((1, 16, 16, 512), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "1", "stop": "16", "stride": "2"},
        },
    ),
    (
        Index247,
        [((1, 8, 16, 512), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "16", "stride": "2"},
        },
    ),
    (
        Index248,
        [((1, 8, 16, 512), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "1", "stop": "16", "stride": "2"},
        },
    ),
    (
        Index19,
        [((3, 1, 32, 64, 32), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index20,
        [((3, 1, 32, 64, 32), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index21,
        [((3, 1, 32, 64, 32), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "2", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index98,
        [((1, 3, 60, 60, 85), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5m_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_480x480",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "start": "0", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index99,
        [((1, 3, 60, 60, 85), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5m_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_480x480",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "start": "2", "stop": "4", "stride": "1"},
        },
    ),
    (
        Index100,
        [((1, 3, 60, 60, 85), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5m_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_480x480",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "start": "4", "stop": "85", "stride": "1"},
        },
    ),
    (
        Index98,
        [((1, 3, 30, 30, 85), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5m_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_480x480",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "start": "0", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index99,
        [((1, 3, 30, 30, 85), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5m_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_480x480",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "start": "2", "stop": "4", "stride": "1"},
        },
    ),
    (
        Index100,
        [((1, 3, 30, 30, 85), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5m_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_480x480",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "start": "4", "stop": "85", "stride": "1"},
        },
    ),
    (
        Index98,
        [((1, 3, 15, 15, 85), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5m_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_480x480",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "start": "0", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index99,
        [((1, 3, 15, 15, 85), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5m_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_480x480",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "start": "2", "stop": "4", "stride": "1"},
        },
    ),
    (
        Index100,
        [((1, 3, 15, 15, 85), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5m_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_480x480",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "start": "4", "stop": "85", "stride": "1"},
        },
    ),
    (
        Index98,
        [((1, 5880, 4), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolo_v6_yolov6n_obj_det_torchhub",
                "pt_yolo_v6_yolov6m_obj_det_torchhub",
                "pt_yolo_v6_yolov6s_obj_det_torchhub",
                "pt_yolo_v6_yolov6l_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "start": "0", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index99,
        [((1, 5880, 4), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolo_v6_yolov6n_obj_det_torchhub",
                "pt_yolo_v6_yolov6m_obj_det_torchhub",
                "pt_yolo_v6_yolov6s_obj_det_torchhub",
                "pt_yolo_v6_yolov6l_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "start": "2", "stop": "4", "stride": "1"},
        },
    ),
    (
        Index118,
        [((1, 64, 28, 28), torch.float32)],
        {
            "model_names": ["onnx_hrnet_hrnet_w18_small_v1_img_cls_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "16", "stride": "1"},
        },
    ),
    (
        Index136,
        [((1, 64, 28, 28), torch.float32)],
        {
            "model_names": ["onnx_hrnet_hrnet_w18_small_v1_img_cls_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "16", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index138,
        [((1, 64, 28, 28), torch.float32)],
        {
            "model_names": ["onnx_hrnet_hrnet_w18_small_v1_img_cls_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "32", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index6,
        [((1, 112, 7, 7), torch.float32)],
        {
            "model_names": ["onnx_hrnet_hrnet_w18_small_v1_img_cls_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index176,
        [((1, 112, 7, 7), torch.float32)],
        {
            "model_names": ["onnx_hrnet_hrnet_w18_small_v1_img_cls_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "64", "stop": "96", "stride": "1"},
        },
    ),
    (
        Index249,
        [((1, 112, 7, 7), torch.float32)],
        {
            "model_names": ["onnx_hrnet_hrnet_w18_small_v1_img_cls_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "96", "stop": "112", "stride": "1"},
        },
    ),
    (
        Index206,
        [((1, 308, 7, 7), torch.float32)],
        {
            "model_names": ["onnx_hrnet_hrnetv2_w44_img_cls_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "44", "stride": "1"},
        },
    ),
    (
        Index250,
        [((1, 308, 7, 7), torch.float32)],
        {
            "model_names": ["onnx_hrnet_hrnetv2_w44_img_cls_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "44", "stop": "132", "stride": "1"},
        },
    ),
    (
        Index251,
        [((1, 308, 7, 7), torch.float32)],
        {
            "model_names": ["onnx_hrnet_hrnetv2_w44_img_cls_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "132", "stop": "308", "stride": "1"},
        },
    ),
    (
        Index252,
        [((1, 308, 7, 7), torch.float32)],
        {
            "model_names": ["onnx_hrnet_hrnetv2_w44_img_cls_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "176", "stride": "1"},
        },
    ),
    (
        Index253,
        [((1, 308, 7, 7), torch.float32)],
        {
            "model_names": ["onnx_hrnet_hrnetv2_w44_img_cls_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "176", "stop": "264", "stride": "1"},
        },
    ),
    (
        Index254,
        [((1, 308, 7, 7), torch.float32)],
        {
            "model_names": ["onnx_hrnet_hrnetv2_w44_img_cls_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "264", "stop": "308", "stride": "1"},
        },
    ),
    (
        Index203,
        [((1, 176, 28, 28), torch.float32)],
        {
            "model_names": ["onnx_hrnet_hrnetv2_w44_img_cls_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "88", "stride": "1"},
        },
    ),
    (
        Index204,
        [((1, 176, 28, 28), torch.float32)],
        {
            "model_names": ["onnx_hrnet_hrnetv2_w44_img_cls_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "88", "stop": "132", "stride": "1"},
        },
    ),
    (
        Index205,
        [((1, 176, 28, 28), torch.float32)],
        {
            "model_names": ["onnx_hrnet_hrnetv2_w44_img_cls_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "132", "stop": "176", "stride": "1"},
        },
    ),
    (
        Index206,
        [((1, 176, 28, 28), torch.float32)],
        {
            "model_names": ["onnx_hrnet_hrnetv2_w44_img_cls_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "44", "stride": "1"},
        },
    ),
    (
        Index207,
        [((1, 176, 28, 28), torch.float32)],
        {
            "model_names": ["onnx_hrnet_hrnetv2_w44_img_cls_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "44", "stop": "88", "stride": "1"},
        },
    ),
    (
        Index208,
        [((1, 176, 28, 28), torch.float32)],
        {
            "model_names": ["onnx_hrnet_hrnetv2_w44_img_cls_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "88", "stop": "176", "stride": "1"},
        },
    ),
    (
        Index6,
        [((1, 176, 28, 28), torch.float32)],
        {
            "model_names": ["onnx_googlenet_googlenet_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index201,
        [((1, 176, 28, 28), torch.float32)],
        {
            "model_names": ["onnx_googlenet_googlenet_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "64", "stop": "160", "stride": "1"},
        },
    ),
    (
        Index202,
        [((1, 176, 28, 28), torch.float32)],
        {
            "model_names": ["onnx_googlenet_googlenet_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "160", "stop": "176", "stride": "1"},
        },
    ),
    (
        Index30,
        [((1, 201, 768), torch.float32)],
        {
            "model_names": ["onnx_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index98,
        [((1, 3, 40, 40, 85), torch.float32)],
        {
            "model_names": [
                "onnx_yolo_v5_yolov5n_img_cls_torchhub_320x320",
                "onnx_yolo_v5_yolov5s_img_cls_torchhub_320x320",
                "onnx_yolo_v5_yolov5m_img_cls_torchhub_320x320",
                "onnx_yolo_v5_yolov5l_img_cls_torchhub_320x320",
                "onnx_yolo_v5_yolov5x_img_cls_torchhub_320x320",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index99,
        [((1, 3, 40, 40, 85), torch.float32)],
        {
            "model_names": [
                "onnx_yolo_v5_yolov5n_img_cls_torchhub_320x320",
                "onnx_yolo_v5_yolov5s_img_cls_torchhub_320x320",
                "onnx_yolo_v5_yolov5m_img_cls_torchhub_320x320",
                "onnx_yolo_v5_yolov5l_img_cls_torchhub_320x320",
                "onnx_yolo_v5_yolov5x_img_cls_torchhub_320x320",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "2", "stop": "4", "stride": "1"},
        },
    ),
    (
        Index100,
        [((1, 3, 40, 40, 85), torch.float32)],
        {
            "model_names": [
                "onnx_yolo_v5_yolov5n_img_cls_torchhub_320x320",
                "onnx_yolo_v5_yolov5s_img_cls_torchhub_320x320",
                "onnx_yolo_v5_yolov5m_img_cls_torchhub_320x320",
                "onnx_yolo_v5_yolov5l_img_cls_torchhub_320x320",
                "onnx_yolo_v5_yolov5x_img_cls_torchhub_320x320",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "4", "stop": "85", "stride": "1"},
        },
    ),
    (
        Index98,
        [((1, 3, 20, 20, 85), torch.float32)],
        {
            "model_names": [
                "onnx_yolo_v5_yolov5n_img_cls_torchhub_320x320",
                "onnx_yolo_v5_yolov5s_img_cls_torchhub_320x320",
                "onnx_yolo_v5_yolov5m_img_cls_torchhub_320x320",
                "onnx_yolo_v5_yolov5l_img_cls_torchhub_320x320",
                "onnx_yolo_v5_yolov5x_img_cls_torchhub_320x320",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index99,
        [((1, 3, 20, 20, 85), torch.float32)],
        {
            "model_names": [
                "onnx_yolo_v5_yolov5n_img_cls_torchhub_320x320",
                "onnx_yolo_v5_yolov5s_img_cls_torchhub_320x320",
                "onnx_yolo_v5_yolov5m_img_cls_torchhub_320x320",
                "onnx_yolo_v5_yolov5l_img_cls_torchhub_320x320",
                "onnx_yolo_v5_yolov5x_img_cls_torchhub_320x320",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "2", "stop": "4", "stride": "1"},
        },
    ),
    (
        Index100,
        [((1, 3, 20, 20, 85), torch.float32)],
        {
            "model_names": [
                "onnx_yolo_v5_yolov5n_img_cls_torchhub_320x320",
                "onnx_yolo_v5_yolov5s_img_cls_torchhub_320x320",
                "onnx_yolo_v5_yolov5m_img_cls_torchhub_320x320",
                "onnx_yolo_v5_yolov5l_img_cls_torchhub_320x320",
                "onnx_yolo_v5_yolov5x_img_cls_torchhub_320x320",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "4", "stop": "85", "stride": "1"},
        },
    ),
    (
        Index98,
        [((1, 3, 10, 10, 85), torch.float32)],
        {
            "model_names": [
                "onnx_yolo_v5_yolov5n_img_cls_torchhub_320x320",
                "onnx_yolo_v5_yolov5s_img_cls_torchhub_320x320",
                "onnx_yolo_v5_yolov5m_img_cls_torchhub_320x320",
                "onnx_yolo_v5_yolov5l_img_cls_torchhub_320x320",
                "onnx_yolo_v5_yolov5x_img_cls_torchhub_320x320",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index99,
        [((1, 3, 10, 10, 85), torch.float32)],
        {
            "model_names": [
                "onnx_yolo_v5_yolov5n_img_cls_torchhub_320x320",
                "onnx_yolo_v5_yolov5s_img_cls_torchhub_320x320",
                "onnx_yolo_v5_yolov5m_img_cls_torchhub_320x320",
                "onnx_yolo_v5_yolov5l_img_cls_torchhub_320x320",
                "onnx_yolo_v5_yolov5x_img_cls_torchhub_320x320",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "2", "stop": "4", "stride": "1"},
        },
    ),
    (
        Index100,
        [((1, 3, 10, 10, 85), torch.float32)],
        {
            "model_names": [
                "onnx_yolo_v5_yolov5n_img_cls_torchhub_320x320",
                "onnx_yolo_v5_yolov5s_img_cls_torchhub_320x320",
                "onnx_yolo_v5_yolov5m_img_cls_torchhub_320x320",
                "onnx_yolo_v5_yolov5l_img_cls_torchhub_320x320",
                "onnx_yolo_v5_yolov5x_img_cls_torchhub_320x320",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "4", "stop": "85", "stride": "1"},
        },
    ),
    (
        Index30,
        [((1, 15, 768), torch.float32)],
        {
            "model_names": ["pd_bert_bert_base_japanese_seq_cls_padlenlp"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index30,
        [((1, 8, 768), torch.float32)],
        {
            "model_names": [
                "pd_bert_bert_base_uncased_seq_cls_padlenlp",
                "pd_blip_text_salesforce_blip_image_captioning_base_text_enc_padlenlp",
                "pd_chineseclip_text_ofa_sys_chinese_clip_vit_base_patch16_text_enc_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index30,
        [((1, 25, 768), torch.float32)],
        {
            "model_names": [
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index255,
        [((1, 8, 522, 256), torch.float32)],
        {
            "model_names": ["pt_falcon3_tiiuae_falcon3_1b_base_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "128", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index23,
        [((1, 8, 522, 256), torch.float32)],
        {
            "model_names": ["pt_falcon3_tiiuae_falcon3_1b_base_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index255,
        [((1, 4, 522, 256), torch.float32)],
        {
            "model_names": [
                "pt_falcon3_tiiuae_falcon3_1b_base_clm_hf",
                "pt_falcon3_tiiuae_falcon3_3b_base_clm_hf",
                "pt_falcon3_tiiuae_falcon3_7b_base_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "128", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index23,
        [((1, 4, 522, 256), torch.float32)],
        {
            "model_names": [
                "pt_falcon3_tiiuae_falcon3_1b_base_clm_hf",
                "pt_falcon3_tiiuae_falcon3_3b_base_clm_hf",
                "pt_falcon3_tiiuae_falcon3_7b_base_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index256,
        [((1, 1, 2048, 2048), torch.bool)],
        {
            "model_names": [
                "pt_gptneo_gpt_neo_1_3b_seq_cls_hf",
                "pt_gptneo_gpt_neo_125m_seq_cls_hf",
                "pt_gptneo_gpt_neo_2_7b_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "5", "stride": "1"},
        },
    ),
    (
        Index34,
        [((1, 1, 2048, 2048), torch.bool)],
        {
            "model_names": [
                "pt_gptneo_gpt_neo_125m_clm_hf",
                "pt_gptneo_gpt_neo_1_3b_clm_hf",
                "pt_gptneo_gpt_neo_2_7b_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index257,
        [((1, 1, 5, 2048), torch.bool)],
        {
            "model_names": [
                "pt_gptneo_gpt_neo_1_3b_seq_cls_hf",
                "pt_gptneo_gpt_neo_125m_seq_cls_hf",
                "pt_gptneo_gpt_neo_2_7b_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "5", "stride": "1"},
        },
    ),
    (
        Index258,
        [((1, 5, 2), torch.float32)],
        {
            "model_names": [
                "pt_gptneo_gpt_neo_1_3b_seq_cls_hf",
                "pt_gptneo_gpt_neo_125m_seq_cls_hf",
                "pt_phi1_microsoft_phi_1_seq_cls_hf",
                "pt_phi_1_5_microsoft_phi_1_5_seq_cls_hf",
                "pt_gptneo_gpt_neo_2_7b_seq_cls_hf",
                "pt_phi3_microsoft_phi_3_mini_128k_instruct_seq_cls_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "4", "stop": "5", "stride": "1"},
        },
    ),
    (
        Index114,
        [((1, 3, 640, 640), torch.float32)],
        {
            "model_names": [
                "pt_yolox_yolox_s_obj_det_torchhub",
                "pt_yolox_yolox_l_obj_det_torchhub",
                "pt_yolox_yolox_m_obj_det_torchhub",
                "pt_yolox_yolox_x_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "640", "stride": "2"},
        },
    ),
    (
        Index115,
        [((1, 3, 640, 640), torch.float32)],
        {
            "model_names": [
                "pt_yolox_yolox_s_obj_det_torchhub",
                "pt_yolox_yolox_l_obj_det_torchhub",
                "pt_yolox_yolox_m_obj_det_torchhub",
                "pt_yolox_yolox_x_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "1", "stop": "640", "stride": "2"},
        },
    ),
    (
        Index116,
        [((1, 3, 320, 640), torch.float32)],
        {
            "model_names": [
                "pt_yolox_yolox_s_obj_det_torchhub",
                "pt_yolox_yolox_l_obj_det_torchhub",
                "pt_yolox_yolox_m_obj_det_torchhub",
                "pt_yolox_yolox_x_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "640", "stride": "2"},
        },
    ),
    (
        Index117,
        [((1, 3, 320, 640), torch.float32)],
        {
            "model_names": [
                "pt_yolox_yolox_s_obj_det_torchhub",
                "pt_yolox_yolox_l_obj_det_torchhub",
                "pt_yolox_yolox_m_obj_det_torchhub",
                "pt_yolox_yolox_x_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "1", "stop": "640", "stride": "2"},
        },
    ),
    (
        Index218,
        [((1, 64, 64, 96), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
                "pt_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "4", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index219,
        [((1, 64, 64, 96), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
                "pt_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "4", "stride": "1"},
        },
    ),
    (
        Index220,
        [((1, 64, 64, 96), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
                "pt_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "4", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index221,
        [((1, 64, 64, 96), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
                "pt_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "4", "stride": "1"},
        },
    ),
    (
        Index259,
        [((1, 64, 64, 96), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "-4", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index260,
        [((1, 64, 64, 96), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "-4", "stride": "1"},
        },
    ),
    (
        Index261,
        [((1, 64, 64, 96), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "-4", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index262,
        [((1, 64, 64, 96), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "-4", "stride": "1"},
        },
    ),
    (
        Index225,
        [((1, 64, 64, 96), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
                "pt_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "64", "stride": "2"},
        },
    ),
    (
        Index226,
        [((1, 64, 64, 96), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
                "pt_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "1", "stop": "64", "stride": "2"},
        },
    ),
    (
        Index222,
        [((1, 64, 64, 96), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_t_img_cls_torchvision",
                "pt_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "60", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index123,
        [((1, 64, 64, 96), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_t_img_cls_torchvision",
                "pt_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "60", "stride": "1"},
        },
    ),
    (
        Index223,
        [((1, 64, 64, 96), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_t_img_cls_torchvision",
                "pt_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "60", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index224,
        [((1, 64, 64, 96), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_t_img_cls_torchvision",
                "pt_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "60", "stride": "1"},
        },
    ),
    (
        Index227,
        [((1, 32, 64, 96), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
                "pt_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "64", "stride": "2"},
        },
    ),
    (
        Index228,
        [((1, 32, 64, 96), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
                "pt_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "1", "stop": "64", "stride": "2"},
        },
    ),
    (
        Index229,
        [((1, 32, 32, 192), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
                "pt_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "4", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index219,
        [((1, 32, 32, 192), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
                "pt_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "4", "stride": "1"},
        },
    ),
    (
        Index230,
        [((1, 32, 32, 192), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
                "pt_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "4", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index221,
        [((1, 32, 32, 192), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
                "pt_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "4", "stride": "1"},
        },
    ),
    (
        Index263,
        [((1, 32, 32, 192), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "-4", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index260,
        [((1, 32, 32, 192), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "-4", "stride": "1"},
        },
    ),
    (
        Index264,
        [((1, 32, 32, 192), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "-4", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index262,
        [((1, 32, 32, 192), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "-4", "stride": "1"},
        },
    ),
    (
        Index235,
        [((1, 32, 32, 192), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
                "pt_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "32", "stride": "2"},
        },
    ),
    (
        Index236,
        [((1, 32, 32, 192), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
                "pt_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "1", "stop": "32", "stride": "2"},
        },
    ),
    (
        Index231,
        [((1, 32, 32, 192), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_t_img_cls_torchvision",
                "pt_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "28", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index232,
        [((1, 32, 32, 192), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_t_img_cls_torchvision",
                "pt_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "28", "stride": "1"},
        },
    ),
    (
        Index233,
        [((1, 32, 32, 192), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_t_img_cls_torchvision",
                "pt_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "28", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index234,
        [((1, 32, 32, 192), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_t_img_cls_torchvision",
                "pt_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "28", "stride": "1"},
        },
    ),
    (
        Index237,
        [((1, 16, 32, 192), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
                "pt_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "32", "stride": "2"},
        },
    ),
    (
        Index238,
        [((1, 16, 32, 192), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
                "pt_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "1", "stop": "32", "stride": "2"},
        },
    ),
    (
        Index239,
        [((1, 16, 16, 384), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
                "pt_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "4", "stop": "16", "stride": "1"},
        },
    ),
    (
        Index219,
        [((1, 16, 16, 384), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
                "pt_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "4", "stride": "1"},
        },
    ),
    (
        Index240,
        [((1, 16, 16, 384), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
                "pt_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "4", "stop": "16", "stride": "1"},
        },
    ),
    (
        Index221,
        [((1, 16, 16, 384), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
                "pt_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "4", "stride": "1"},
        },
    ),
    (
        Index265,
        [((1, 16, 16, 384), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "-4", "stop": "16", "stride": "1"},
        },
    ),
    (
        Index260,
        [((1, 16, 16, 384), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "-4", "stride": "1"},
        },
    ),
    (
        Index266,
        [((1, 16, 16, 384), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "-4", "stop": "16", "stride": "1"},
        },
    ),
    (
        Index262,
        [((1, 16, 16, 384), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "-4", "stride": "1"},
        },
    ),
    (
        Index245,
        [((1, 16, 16, 384), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
                "pt_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "16", "stride": "2"},
        },
    ),
    (
        Index246,
        [((1, 16, 16, 384), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
                "pt_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "1", "stop": "16", "stride": "2"},
        },
    ),
    (
        Index241,
        [((1, 16, 16, 384), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_t_img_cls_torchvision",
                "pt_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "12", "stop": "16", "stride": "1"},
        },
    ),
    (
        Index242,
        [((1, 16, 16, 384), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_t_img_cls_torchvision",
                "pt_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "12", "stride": "1"},
        },
    ),
    (
        Index243,
        [((1, 16, 16, 384), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_t_img_cls_torchvision",
                "pt_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "12", "stop": "16", "stride": "1"},
        },
    ),
    (
        Index244,
        [((1, 16, 16, 384), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_t_img_cls_torchvision",
                "pt_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "12", "stride": "1"},
        },
    ),
    (
        Index247,
        [((1, 8, 16, 384), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
                "pt_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "16", "stride": "2"},
        },
    ),
    (
        Index248,
        [((1, 8, 16, 384), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
                "pt_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "1", "stop": "16", "stride": "2"},
        },
    ),
    (
        Index19,
        [((3, 1, 12, 577, 64), torch.float32)],
        {
            "model_names": [
                "pd_blip_vision_salesforce_blip_image_captioning_base_img_enc_padlenlp",
                "pd_blip_salesforce_blip_image_captioning_base_img_captioning_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index20,
        [((3, 1, 12, 577, 64), torch.float32)],
        {
            "model_names": [
                "pd_blip_vision_salesforce_blip_image_captioning_base_img_enc_padlenlp",
                "pd_blip_salesforce_blip_image_captioning_base_img_captioning_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index21,
        [((3, 1, 12, 577, 64), torch.float32)],
        {
            "model_names": [
                "pd_blip_vision_salesforce_blip_image_captioning_base_img_enc_padlenlp",
                "pd_blip_salesforce_blip_image_captioning_base_img_captioning_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "2", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index30,
        [((1, 577, 768), torch.float32)],
        {
            "model_names": [
                "pd_blip_vision_salesforce_blip_image_captioning_base_img_enc_padlenlp",
                "pd_blip_salesforce_blip_image_captioning_base_img_captioning_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index30,
        [((4, 5, 768), torch.float32)],
        {
            "model_names": ["pd_chineseclip_ofa_sys_chinese_clip_vit_base_patch16_img_text_pairing_padlenlp"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index30,
        [((1, 197, 192), torch.bfloat16)],
        {
            "model_names": ["pt_deit_tiny_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index101,
        [((1, 2, 30, 40), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_default_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index102,
        [((1, 2, 30, 40), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_default_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index101,
        [((1, 2, 60, 80), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_default_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index102,
        [((1, 2, 60, 80), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_default_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index101,
        [((1, 2, 120, 160), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_default_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index102,
        [((1, 2, 120, 160), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_default_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index118,
        [((1, 64, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w18_small_img_cls_timm", "pt_hrnet_hrnet_w18_small_v1_osmr_img_cls_osmr"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "0", "stop": "16", "stride": "1"},
        },
    ),
    (
        Index136,
        [((1, 64, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w18_small_img_cls_timm", "pt_hrnet_hrnet_w18_small_v1_osmr_img_cls_osmr"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "16", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index138,
        [((1, 64, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w18_small_img_cls_timm", "pt_hrnet_hrnet_w18_small_v1_osmr_img_cls_osmr"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "32", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index6,
        [((1, 112, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w18_small_img_cls_timm", "pt_hrnet_hrnet_w18_small_v1_osmr_img_cls_osmr"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index176,
        [((1, 112, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w18_small_img_cls_timm", "pt_hrnet_hrnet_w18_small_v1_osmr_img_cls_osmr"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "64", "stop": "96", "stride": "1"},
        },
    ),
    (
        Index249,
        [((1, 112, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w18_small_img_cls_timm", "pt_hrnet_hrnet_w18_small_v1_osmr_img_cls_osmr"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "96", "stop": "112", "stride": "1"},
        },
    ),
    (
        Index126,
        [((1, 210, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w30_img_cls_timm", "pt_hrnet_hrnetv2_w30_osmr_img_cls_osmr"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "0", "stop": "30", "stride": "1"},
        },
    ),
    (
        Index187,
        [((1, 210, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w30_img_cls_timm", "pt_hrnet_hrnetv2_w30_osmr_img_cls_osmr"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "30", "stop": "90", "stride": "1"},
        },
    ),
    (
        Index188,
        [((1, 210, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w30_img_cls_timm", "pt_hrnet_hrnetv2_w30_osmr_img_cls_osmr"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "90", "stop": "210", "stride": "1"},
        },
    ),
    (
        Index122,
        [((1, 210, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w30_img_cls_timm", "pt_hrnet_hrnetv2_w30_osmr_img_cls_osmr"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "0", "stop": "120", "stride": "1"},
        },
    ),
    (
        Index189,
        [((1, 210, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w30_img_cls_timm", "pt_hrnet_hrnetv2_w30_osmr_img_cls_osmr"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "120", "stop": "180", "stride": "1"},
        },
    ),
    (
        Index190,
        [((1, 210, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w30_img_cls_timm", "pt_hrnet_hrnetv2_w30_osmr_img_cls_osmr"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "180", "stop": "210", "stride": "1"},
        },
    ),
    (
        Index123,
        [((1, 120, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w30_img_cls_timm", "pt_hrnet_hrnetv2_w30_osmr_img_cls_osmr"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "0", "stop": "60", "stride": "1"},
        },
    ),
    (
        Index124,
        [((1, 120, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w30_img_cls_timm", "pt_hrnet_hrnetv2_w30_osmr_img_cls_osmr"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "60", "stop": "90", "stride": "1"},
        },
    ),
    (
        Index125,
        [((1, 120, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w30_img_cls_timm", "pt_hrnet_hrnetv2_w30_osmr_img_cls_osmr"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "90", "stop": "120", "stride": "1"},
        },
    ),
    (
        Index126,
        [((1, 120, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w30_img_cls_timm", "pt_hrnet_hrnetv2_w30_osmr_img_cls_osmr"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "0", "stop": "30", "stride": "1"},
        },
    ),
    (
        Index127,
        [((1, 120, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w30_img_cls_timm", "pt_hrnet_hrnetv2_w30_osmr_img_cls_osmr"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "30", "stop": "60", "stride": "1"},
        },
    ),
    (
        Index128,
        [((1, 120, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w30_img_cls_timm", "pt_hrnet_hrnetv2_w30_osmr_img_cls_osmr"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "60", "stop": "120", "stride": "1"},
        },
    ),
    (
        Index47,
        [((1, 16, 6, 64), torch.float32)],
        {
            "model_names": ["pt_qwen1_5_0_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "32", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index46,
        [((1, 16, 6, 64), torch.float32)],
        {
            "model_names": ["pt_qwen1_5_0_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index171,
        [((4, 32, 31, 128), torch.bfloat16)],
        {
            "model_names": ["pt_qwen_v3_embedding_4b_sentence_embed_gen_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "start": "64", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index172,
        [((4, 32, 31, 128), torch.bfloat16)],
        {
            "model_names": ["pt_qwen_v3_embedding_4b_sentence_embed_gen_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index171,
        [((4, 8, 31, 128), torch.bfloat16)],
        {
            "model_names": [
                "pt_qwen_v3_embedding_4b_sentence_embed_gen_hf",
                "pt_qwen_v3_embedding_0_6b_sentence_embed_gen_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "start": "64", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index172,
        [((4, 8, 31, 128), torch.bfloat16)],
        {
            "model_names": [
                "pt_qwen_v3_embedding_4b_sentence_embed_gen_hf",
                "pt_qwen_v3_embedding_0_6b_sentence_embed_gen_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index19,
        [((3, 64, 3, 64, 32), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_t_img_cls_torchvision", "pt_swin_swin_v2_s_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index20,
        [((3, 64, 3, 64, 32), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_t_img_cls_torchvision", "pt_swin_swin_v2_s_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index21,
        [((3, 64, 3, 64, 32), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_t_img_cls_torchvision", "pt_swin_swin_v2_s_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "2", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index19,
        [((3, 16, 6, 64, 32), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_t_img_cls_torchvision", "pt_swin_swin_v2_s_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index20,
        [((3, 16, 6, 64, 32), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_t_img_cls_torchvision", "pt_swin_swin_v2_s_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index21,
        [((3, 16, 6, 64, 32), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_t_img_cls_torchvision", "pt_swin_swin_v2_s_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "2", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index19,
        [((3, 4, 12, 64, 32), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_t_img_cls_torchvision", "pt_swin_swin_v2_s_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index20,
        [((3, 4, 12, 64, 32), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_t_img_cls_torchvision", "pt_swin_swin_v2_s_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index21,
        [((3, 4, 12, 64, 32), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_t_img_cls_torchvision", "pt_swin_swin_v2_s_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "2", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index19,
        [((3, 1, 24, 64, 32), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_t_img_cls_torchvision", "pt_swin_swin_v2_s_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index20,
        [((3, 1, 24, 64, 32), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_t_img_cls_torchvision", "pt_swin_swin_v2_s_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index21,
        [((3, 1, 24, 64, 32), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_t_img_cls_torchvision", "pt_swin_swin_v2_s_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "2", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index30,
        [((1, 197, 768), torch.bfloat16)],
        {
            "model_names": [
                "pt_vit_base_img_cls_hf",
                "pt_deit_base_distilled_img_cls_hf",
                "pt_vit_vit_b_16_img_cls_torchvision",
                "pt_deit_base_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index142,
        [((1, 197, 768), torch.bfloat16)],
        {
            "model_names": ["pt_beit_base_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "start": "1", "stop": "197", "stride": "1"},
        },
    ),
    (
        Index183,
        [((1, 3, 416, 416), torch.float32)],
        {
            "model_names": ["pt_yolox_yolox_tiny_obj_det_torchhub", "pt_yolox_yolox_nano_obj_det_torchhub"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "416", "stride": "2"},
        },
    ),
    (
        Index184,
        [((1, 3, 416, 416), torch.float32)],
        {
            "model_names": ["pt_yolox_yolox_tiny_obj_det_torchhub", "pt_yolox_yolox_nano_obj_det_torchhub"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "1", "stop": "416", "stride": "2"},
        },
    ),
    (
        Index185,
        [((1, 3, 208, 416), torch.float32)],
        {
            "model_names": ["pt_yolox_yolox_tiny_obj_det_torchhub", "pt_yolox_yolox_nano_obj_det_torchhub"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "416", "stride": "2"},
        },
    ),
    (
        Index186,
        [((1, 3, 208, 416), torch.float32)],
        {
            "model_names": ["pt_yolox_yolox_tiny_obj_det_torchhub", "pt_yolox_yolox_nano_obj_det_torchhub"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "1", "stop": "416", "stride": "2"},
        },
    ),
    (
        Index30,
        [((1, 197, 384), torch.float32)],
        {
            "model_names": ["onnx_deit_facebook_deit_small_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index51,
        [((1, 126, 7, 7), torch.float32)],
        {
            "model_names": ["onnx_hrnet_hrnetv2_w18_img_cls_osmr", "onnx_hrnet_hrnet_w18_small_v2_img_cls_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "18", "stride": "1"},
        },
    ),
    (
        Index52,
        [((1, 126, 7, 7), torch.float32)],
        {
            "model_names": ["onnx_hrnet_hrnetv2_w18_img_cls_osmr", "onnx_hrnet_hrnet_w18_small_v2_img_cls_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "18", "stop": "54", "stride": "1"},
        },
    ),
    (
        Index53,
        [((1, 126, 7, 7), torch.float32)],
        {
            "model_names": ["onnx_hrnet_hrnetv2_w18_img_cls_osmr", "onnx_hrnet_hrnet_w18_small_v2_img_cls_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "54", "stop": "126", "stride": "1"},
        },
    ),
    (
        Index54,
        [((1, 126, 7, 7), torch.float32)],
        {
            "model_names": ["onnx_hrnet_hrnetv2_w18_img_cls_osmr", "onnx_hrnet_hrnet_w18_small_v2_img_cls_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "72", "stride": "1"},
        },
    ),
    (
        Index55,
        [((1, 126, 7, 7), torch.float32)],
        {
            "model_names": ["onnx_hrnet_hrnetv2_w18_img_cls_osmr", "onnx_hrnet_hrnet_w18_small_v2_img_cls_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "72", "stop": "108", "stride": "1"},
        },
    ),
    (
        Index56,
        [((1, 126, 7, 7), torch.float32)],
        {
            "model_names": ["onnx_hrnet_hrnetv2_w18_img_cls_osmr", "onnx_hrnet_hrnet_w18_small_v2_img_cls_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "108", "stop": "126", "stride": "1"},
        },
    ),
    (
        Index57,
        [((1, 72, 28, 28), torch.float32)],
        {
            "model_names": ["onnx_hrnet_hrnetv2_w18_img_cls_osmr", "onnx_hrnet_hrnet_w18_small_v2_img_cls_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "36", "stride": "1"},
        },
    ),
    (
        Index58,
        [((1, 72, 28, 28), torch.float32)],
        {
            "model_names": ["onnx_hrnet_hrnetv2_w18_img_cls_osmr", "onnx_hrnet_hrnet_w18_small_v2_img_cls_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "36", "stop": "54", "stride": "1"},
        },
    ),
    (
        Index59,
        [((1, 72, 28, 28), torch.float32)],
        {
            "model_names": ["onnx_hrnet_hrnetv2_w18_img_cls_osmr", "onnx_hrnet_hrnet_w18_small_v2_img_cls_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "54", "stop": "72", "stride": "1"},
        },
    ),
    (
        Index51,
        [((1, 72, 28, 28), torch.float32)],
        {
            "model_names": ["onnx_hrnet_hrnetv2_w18_img_cls_osmr", "onnx_hrnet_hrnet_w18_small_v2_img_cls_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "18", "stride": "1"},
        },
    ),
    (
        Index60,
        [((1, 72, 28, 28), torch.float32)],
        {
            "model_names": ["onnx_hrnet_hrnetv2_w18_img_cls_osmr", "onnx_hrnet_hrnet_w18_small_v2_img_cls_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "18", "stop": "36", "stride": "1"},
        },
    ),
    (
        Index61,
        [((1, 72, 28, 28), torch.float32)],
        {
            "model_names": ["onnx_hrnet_hrnetv2_w18_img_cls_osmr", "onnx_hrnet_hrnet_w18_small_v2_img_cls_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "36", "stop": "72", "stride": "1"},
        },
    ),
    (
        Index6,
        [((1, 176, 27, 27), torch.float32)],
        {
            "model_names": ["pd_googlenet_base_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index201,
        [((1, 176, 27, 27), torch.float32)],
        {
            "model_names": ["pd_googlenet_base_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "64", "stop": "160", "stride": "1"},
        },
    ),
    (
        Index202,
        [((1, 176, 27, 27), torch.float32)],
        {
            "model_names": ["pd_googlenet_base_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "160", "stop": "176", "stride": "1"},
        },
    ),
    (
        Index14,
        [((1, 288, 27, 27), torch.float32)],
        {
            "model_names": ["pd_googlenet_base_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index18,
        [((1, 288, 27, 27), torch.float32)],
        {
            "model_names": ["pd_googlenet_base_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "128", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index209,
        [((1, 288, 27, 27), torch.float32)],
        {
            "model_names": ["pd_googlenet_base_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "256", "stop": "288", "stride": "1"},
        },
    ),
    (
        Index210,
        [((1, 304, 13, 13), torch.float32)],
        {
            "model_names": ["pd_googlenet_base_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "192", "stride": "1"},
        },
    ),
    (
        Index211,
        [((1, 304, 13, 13), torch.float32)],
        {
            "model_names": ["pd_googlenet_base_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "192", "stop": "288", "stride": "1"},
        },
    ),
    (
        Index212,
        [((1, 304, 13, 13), torch.float32)],
        {
            "model_names": ["pd_googlenet_base_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "288", "stop": "304", "stride": "1"},
        },
    ),
    (
        Index106,
        [((1, 296, 13, 13), torch.float32)],
        {
            "model_names": ["pd_googlenet_base_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "160", "stride": "1"},
        },
    ),
    (
        Index213,
        [((1, 296, 13, 13), torch.float32)],
        {
            "model_names": ["pd_googlenet_base_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "160", "stop": "272", "stride": "1"},
        },
    ),
    (
        Index214,
        [((1, 296, 13, 13), torch.float32)],
        {
            "model_names": ["pd_googlenet_base_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "272", "stop": "296", "stride": "1"},
        },
    ),
    (
        Index14,
        [((1, 280, 13, 13), torch.float32)],
        {
            "model_names": ["pd_googlenet_base_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index18,
        [((1, 280, 13, 13), torch.float32)],
        {
            "model_names": ["pd_googlenet_base_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "128", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index215,
        [((1, 280, 13, 13), torch.float32)],
        {
            "model_names": ["pd_googlenet_base_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "256", "stop": "280", "stride": "1"},
        },
    ),
    (
        Index133,
        [((1, 288, 13, 13), torch.float32)],
        {
            "model_names": ["pd_googlenet_base_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "112", "stride": "1"},
        },
    ),
    (
        Index216,
        [((1, 288, 13, 13), torch.float32)],
        {
            "model_names": ["pd_googlenet_base_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "112", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index209,
        [((1, 288, 13, 13), torch.float32)],
        {
            "model_names": ["pd_googlenet_base_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "256", "stop": "288", "stride": "1"},
        },
    ),
    (
        Index9,
        [((1, 448, 13, 13), torch.float32)],
        {
            "model_names": ["pd_googlenet_base_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index12,
        [((1, 448, 13, 13), torch.float32)],
        {
            "model_names": ["pd_googlenet_base_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "256", "stop": "416", "stride": "1"},
        },
    ),
    (
        Index13,
        [((1, 448, 13, 13), torch.float32)],
        {
            "model_names": ["pd_googlenet_base_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "416", "stop": "448", "stride": "1"},
        },
    ),
    (
        Index9,
        [((1, 448, 6, 6), torch.float32)],
        {
            "model_names": ["pd_googlenet_base_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index12,
        [((1, 448, 6, 6), torch.float32)],
        {
            "model_names": ["pd_googlenet_base_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "256", "stop": "416", "stride": "1"},
        },
    ),
    (
        Index13,
        [((1, 448, 6, 6), torch.float32)],
        {
            "model_names": ["pd_googlenet_base_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "416", "stop": "448", "stride": "1"},
        },
    ),
    (
        Index194,
        [((1, 624, 6, 6), torch.float32)],
        {
            "model_names": ["pd_googlenet_base_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "384", "stride": "1"},
        },
    ),
    (
        Index195,
        [((1, 624, 6, 6), torch.float32)],
        {
            "model_names": ["pd_googlenet_base_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "384", "stop": "576", "stride": "1"},
        },
    ),
    (
        Index217,
        [((1, 624, 6, 6), torch.float32)],
        {
            "model_names": ["pd_googlenet_base_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "576", "stop": "624", "stride": "1"},
        },
    ),
    (
        Index30,
        [((1, 16, 768), torch.float32)],
        {
            "model_names": ["pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index30,
        [((2, 1024), torch.float32)],
        {
            "model_names": [
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                "pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf",
                "onnx_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index31,
        [((2, 1024), torch.float32)],
        {
            "model_names": [
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                "pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf",
                "onnx_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index30,
        [((1, 32, 16, 3, 96), torch.float32)],
        {
            "model_names": ["pt_bloom_default_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index31,
        [((1, 32, 16, 3, 96), torch.float32)],
        {
            "model_names": ["pt_bloom_default_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index64,
        [((1, 32, 16, 3, 96), torch.float32)],
        {
            "model_names": ["pt_bloom_default_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "2", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index121,
        [((1, 280, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w40_img_cls_timm", "pt_hrnet_hrnetv2_w40_osmr_img_cls_osmr"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "0", "stop": "40", "stride": "1"},
        },
    ),
    (
        Index267,
        [((1, 280, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w40_img_cls_timm", "pt_hrnet_hrnetv2_w40_osmr_img_cls_osmr"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "40", "stop": "120", "stride": "1"},
        },
    ),
    (
        Index268,
        [((1, 280, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w40_img_cls_timm", "pt_hrnet_hrnetv2_w40_osmr_img_cls_osmr"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "120", "stop": "280", "stride": "1"},
        },
    ),
    (
        Index106,
        [((1, 280, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w40_img_cls_timm", "pt_hrnet_hrnetv2_w40_osmr_img_cls_osmr"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "0", "stop": "160", "stride": "1"},
        },
    ),
    (
        Index269,
        [((1, 280, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w40_img_cls_timm", "pt_hrnet_hrnetv2_w40_osmr_img_cls_osmr"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "160", "stop": "240", "stride": "1"},
        },
    ),
    (
        Index270,
        [((1, 280, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w40_img_cls_timm", "pt_hrnet_hrnetv2_w40_osmr_img_cls_osmr"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "240", "stop": "280", "stride": "1"},
        },
    ),
    (
        Index104,
        [((1, 160, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w40_img_cls_timm", "pt_hrnet_hrnetv2_w40_osmr_img_cls_osmr"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "0", "stop": "80", "stride": "1"},
        },
    ),
    (
        Index271,
        [((1, 160, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w40_img_cls_timm", "pt_hrnet_hrnetv2_w40_osmr_img_cls_osmr"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "80", "stop": "120", "stride": "1"},
        },
    ),
    (
        Index272,
        [((1, 160, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w40_img_cls_timm", "pt_hrnet_hrnetv2_w40_osmr_img_cls_osmr"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "120", "stop": "160", "stride": "1"},
        },
    ),
    (
        Index121,
        [((1, 160, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w40_img_cls_timm", "pt_hrnet_hrnetv2_w40_osmr_img_cls_osmr"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "0", "stop": "40", "stride": "1"},
        },
    ),
    (
        Index273,
        [((1, 160, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w40_img_cls_timm", "pt_hrnet_hrnetv2_w40_osmr_img_cls_osmr"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "40", "stop": "80", "stride": "1"},
        },
    ),
    (
        Index105,
        [((1, 160, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w40_img_cls_timm", "pt_hrnet_hrnetv2_w40_osmr_img_cls_osmr"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "80", "stop": "160", "stride": "1"},
        },
    ),
    (
        Index206,
        [((1, 308, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w44_img_cls_timm", "pt_hrnet_hrnetv2_w44_osmr_img_cls_osmr"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "0", "stop": "44", "stride": "1"},
        },
    ),
    (
        Index250,
        [((1, 308, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w44_img_cls_timm", "pt_hrnet_hrnetv2_w44_osmr_img_cls_osmr"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "44", "stop": "132", "stride": "1"},
        },
    ),
    (
        Index251,
        [((1, 308, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w44_img_cls_timm", "pt_hrnet_hrnetv2_w44_osmr_img_cls_osmr"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "132", "stop": "308", "stride": "1"},
        },
    ),
    (
        Index252,
        [((1, 308, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w44_img_cls_timm", "pt_hrnet_hrnetv2_w44_osmr_img_cls_osmr"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "0", "stop": "176", "stride": "1"},
        },
    ),
    (
        Index253,
        [((1, 308, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w44_img_cls_timm", "pt_hrnet_hrnetv2_w44_osmr_img_cls_osmr"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "176", "stop": "264", "stride": "1"},
        },
    ),
    (
        Index254,
        [((1, 308, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w44_img_cls_timm", "pt_hrnet_hrnetv2_w44_osmr_img_cls_osmr"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "264", "stop": "308", "stride": "1"},
        },
    ),
    (
        Index119,
        [((1, 336, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w48_img_cls_timm", "pt_hrnet_hrnetv2_w48_osmr_img_cls_osmr"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "0", "stop": "48", "stride": "1"},
        },
    ),
    (
        Index274,
        [((1, 336, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w48_img_cls_timm", "pt_hrnet_hrnetv2_w48_osmr_img_cls_osmr"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "48", "stop": "144", "stride": "1"},
        },
    ),
    (
        Index275,
        [((1, 336, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w48_img_cls_timm", "pt_hrnet_hrnetv2_w48_osmr_img_cls_osmr"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "144", "stop": "336", "stride": "1"},
        },
    ),
    (
        Index210,
        [((1, 336, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w48_img_cls_timm", "pt_hrnet_hrnetv2_w48_osmr_img_cls_osmr"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "0", "stop": "192", "stride": "1"},
        },
    ),
    (
        Index211,
        [((1, 336, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w48_img_cls_timm", "pt_hrnet_hrnetv2_w48_osmr_img_cls_osmr"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "192", "stop": "288", "stride": "1"},
        },
    ),
    (
        Index276,
        [((1, 336, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w48_img_cls_timm", "pt_hrnet_hrnetv2_w48_osmr_img_cls_osmr"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "288", "stop": "336", "stride": "1"},
        },
    ),
    (
        Index191,
        [((1, 192, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w48_img_cls_timm", "pt_hrnet_hrnetv2_w48_osmr_img_cls_osmr"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "0", "stop": "96", "stride": "1"},
        },
    ),
    (
        Index277,
        [((1, 192, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w48_img_cls_timm", "pt_hrnet_hrnetv2_w48_osmr_img_cls_osmr"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "96", "stop": "144", "stride": "1"},
        },
    ),
    (
        Index278,
        [((1, 192, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w48_img_cls_timm", "pt_hrnet_hrnetv2_w48_osmr_img_cls_osmr"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "144", "stop": "192", "stride": "1"},
        },
    ),
    (
        Index119,
        [((1, 192, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w48_img_cls_timm", "pt_hrnet_hrnetv2_w48_osmr_img_cls_osmr"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "0", "stop": "48", "stride": "1"},
        },
    ),
    (
        Index279,
        [((1, 192, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w48_img_cls_timm", "pt_hrnet_hrnetv2_w48_osmr_img_cls_osmr"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "48", "stop": "96", "stride": "1"},
        },
    ),
    (
        Index280,
        [((1, 192, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w48_img_cls_timm", "pt_hrnet_hrnetv2_w48_osmr_img_cls_osmr"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "96", "stop": "192", "stride": "1"},
        },
    ),
    (
        Index29,
        [((1024, 8), torch.float32)],
        {
            "model_names": ["pt_nbeats_trend_basis_time_series_forecasting_github"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "4", "stride": "1"},
        },
    ),
    (
        Index281,
        [((1024, 8), torch.float32)],
        {
            "model_names": ["pt_nbeats_trend_basis_time_series_forecasting_github"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "4", "stop": "8", "stride": "1"},
        },
    ),
    (
        Index30,
        [((2, 2048), torch.float32)],
        {
            "model_names": ["pt_opt_facebook_opt_1_3b_qa_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index31,
        [((2, 2048), torch.float32)],
        {
            "model_names": ["pt_opt_facebook_opt_1_3b_qa_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index30,
        [((1, 1876, 768), torch.float32)],
        {
            "model_names": ["pt_speecht5_tts_tts_text_to_speech_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index282,
        [((1, 600, 768), torch.float32)],
        {
            "model_names": ["pt_speecht5_tts_tts_text_to_speech_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "24", "stride": "1"},
        },
    ),
    (
        Index30,
        [((1, 201, 768), torch.bfloat16)],
        {
            "model_names": ["pt_vilt_vqa_qa_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index283,
        [((3, 1370, 1, 1280), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_h_14_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-4", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index284,
        [((3, 1370, 1, 1280), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_h_14_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-4", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index285,
        [((3, 1370, 1, 1280), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_h_14_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-4", "start": "2", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index30,
        [((1, 1370, 1280), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_h_14_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index286,
        [((2050, 2048), torch.float32)],
        {
            "model_names": ["pt_xglm_xglm_1_7b_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "2", "stop": "258", "stride": "1"},
        },
    ),
    (
        Index50,
        [((1, 1, 256, 2048), torch.bool)],
        {
            "model_names": [
                "pt_gptneo_gpt_neo_125m_clm_hf",
                "pt_gptneo_gpt_neo_1_3b_clm_hf",
                "pt_gptneo_gpt_neo_2_7b_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index47,
        [((1, 32, 4, 64), torch.float32)],
        {
            "model_names": ["pt_llama3_llama_3_2_1b_seq_cls_hf", "pt_llama3_llama_3_2_1b_instruct_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "32", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index46,
        [((1, 32, 4, 64), torch.float32)],
        {
            "model_names": ["pt_llama3_llama_3_2_1b_seq_cls_hf", "pt_llama3_llama_3_2_1b_instruct_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index47,
        [((1, 8, 4, 64), torch.float32)],
        {
            "model_names": ["pt_llama3_llama_3_2_1b_seq_cls_hf", "pt_llama3_llama_3_2_1b_instruct_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "32", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index46,
        [((1, 8, 4, 64), torch.float32)],
        {
            "model_names": ["pt_llama3_llama_3_2_1b_seq_cls_hf", "pt_llama3_llama_3_2_1b_instruct_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index65,
        [((1, 4, 2), torch.float32)],
        {
            "model_names": [
                "pt_llama3_llama_3_2_1b_seq_cls_hf",
                "pt_llama3_llama_3_2_1b_instruct_seq_cls_hf",
                "pt_llama3_llama_3_1_8b_instruct_seq_cls_hf",
                "pt_llama3_llama_3_1_8b_seq_cls_hf",
                "pt_llama3_llama_3_2_3b_instruct_seq_cls_hf",
                "pt_llama3_llama_3_2_3b_seq_cls_hf",
                "pt_llama3_llama_3_8b_instruct_seq_cls_hf",
                "pt_llama3_llama_3_8b_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "3", "stop": "4", "stride": "1"},
        },
    ),
    (
        Index287,
        [((1024, 48), torch.float32)],
        {
            "model_names": [
                "pt_nbeats_seasonality_basis_time_series_forecasting_github",
                "onnx_nbeats_seasionality_basis_time_series_forecasting_github",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "12", "stop": "24", "stride": "1"},
        },
    ),
    (
        Index288,
        [((1024, 48), torch.float32)],
        {
            "model_names": [
                "pt_nbeats_seasonality_basis_time_series_forecasting_github",
                "onnx_nbeats_seasionality_basis_time_series_forecasting_github",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "12", "stride": "1"},
        },
    ),
    (
        Index289,
        [((1024, 48), torch.float32)],
        {
            "model_names": [
                "pt_nbeats_seasonality_basis_time_series_forecasting_github",
                "onnx_nbeats_seasionality_basis_time_series_forecasting_github",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "36", "stop": "48", "stride": "1"},
        },
    ),
    (
        Index290,
        [((1024, 48), torch.float32)],
        {
            "model_names": [
                "pt_nbeats_seasonality_basis_time_series_forecasting_github",
                "onnx_nbeats_seasionality_basis_time_series_forecasting_github",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "24", "stop": "36", "stride": "1"},
        },
    ),
    (
        Index46,
        [((1, 32, 11, 80), torch.float32)],
        {
            "model_names": ["pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf", "pt_phi2_microsoft_phi_2_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index291,
        [((1, 32, 11, 80), torch.float32)],
        {
            "model_names": ["pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf", "pt_phi2_microsoft_phi_2_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "32", "stop": "80", "stride": "1"},
        },
    ),
    (
        Index63,
        [((1, 32, 11, 32), torch.float32)],
        {
            "model_names": ["pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf", "pt_phi2_microsoft_phi_2_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "16", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index27,
        [((1, 32, 11, 32), torch.float32)],
        {
            "model_names": ["pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf", "pt_phi2_microsoft_phi_2_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "16", "stride": "1"},
        },
    ),
    (
        Index292,
        [((1, 11, 2), torch.float32)],
        {
            "model_names": ["pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf", "pt_phi2_microsoft_phi_2_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "10", "stop": "11", "stride": "1"},
        },
    ),
    (
        Index171,
        [((1, 16, 128, 128), torch.float32)],
        {
            "model_names": ["pt_qwen_v3_0_6b_clm_hf", "pt_qwen_v3_1_7b_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "64", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index172,
        [((1, 16, 128, 128), torch.float32)],
        {
            "model_names": ["pt_qwen_v3_0_6b_clm_hf", "pt_qwen_v3_1_7b_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index171,
        [((1, 8, 128, 128), torch.float32)],
        {
            "model_names": ["pt_qwen_v3_0_6b_clm_hf", "pt_qwen_v3_1_7b_clm_hf", "pt_qwen_v3_4b_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "64", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index172,
        [((1, 8, 128, 128), torch.float32)],
        {
            "model_names": ["pt_qwen_v3_0_6b_clm_hf", "pt_qwen_v3_1_7b_clm_hf", "pt_qwen_v3_4b_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index283,
        [((3, 197, 1, 768), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_b_16_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-4", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index284,
        [((3, 197, 1, 768), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_b_16_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-4", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index285,
        [((3, 197, 1, 768), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_b_16_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-4", "start": "2", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index283,
        [((3, 50, 1, 1024), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_l_32_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-4", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index284,
        [((3, 50, 1, 1024), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_l_32_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-4", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index285,
        [((3, 50, 1, 1024), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_l_32_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-4", "start": "2", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index30,
        [((1, 50, 1024), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_l_32_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index139,
        [((1, 5, 128, 400), torch.bfloat16)],
        {
            "model_names": ["pt_yolov10_yolov10x_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "start": "64", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index140,
        [((1, 5, 128, 400), torch.bfloat16)],
        {
            "model_names": ["pt_yolov10_yolov10x_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "start": "0", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index141,
        [((1, 5, 128, 400), torch.bfloat16)],
        {
            "model_names": ["pt_yolov10_yolov10x_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "start": "32", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index142,
        [((1, 197, 1024), torch.float32)],
        {
            "model_names": ["onnx_beit_microsoft_beit_large_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "1", "stop": "197", "stride": "1"},
        },
    ),
    (
        Index30,
        [((1, 197, 1024), torch.float32)],
        {
            "model_names": ["onnx_vit_base_google_vit_large_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index30,
        [((1, 197, 192), torch.float32)],
        {
            "model_names": ["onnx_deit_facebook_deit_tiny_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index119,
        [((1, 336, 7, 7), torch.float32)],
        {
            "model_names": ["onnx_hrnet_hrnetv2_w48_img_cls_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "48", "stride": "1"},
        },
    ),
    (
        Index274,
        [((1, 336, 7, 7), torch.float32)],
        {
            "model_names": ["onnx_hrnet_hrnetv2_w48_img_cls_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "48", "stop": "144", "stride": "1"},
        },
    ),
    (
        Index275,
        [((1, 336, 7, 7), torch.float32)],
        {
            "model_names": ["onnx_hrnet_hrnetv2_w48_img_cls_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "144", "stop": "336", "stride": "1"},
        },
    ),
    (
        Index210,
        [((1, 336, 7, 7), torch.float32)],
        {
            "model_names": ["onnx_hrnet_hrnetv2_w48_img_cls_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "192", "stride": "1"},
        },
    ),
    (
        Index211,
        [((1, 336, 7, 7), torch.float32)],
        {
            "model_names": ["onnx_hrnet_hrnetv2_w48_img_cls_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "192", "stop": "288", "stride": "1"},
        },
    ),
    (
        Index276,
        [((1, 336, 7, 7), torch.float32)],
        {
            "model_names": ["onnx_hrnet_hrnetv2_w48_img_cls_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "288", "stop": "336", "stride": "1"},
        },
    ),
    (
        Index191,
        [((1, 192, 28, 28), torch.float32)],
        {
            "model_names": ["onnx_hrnet_hrnetv2_w48_img_cls_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "96", "stride": "1"},
        },
    ),
    (
        Index277,
        [((1, 192, 28, 28), torch.float32)],
        {
            "model_names": ["onnx_hrnet_hrnetv2_w48_img_cls_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "96", "stop": "144", "stride": "1"},
        },
    ),
    (
        Index278,
        [((1, 192, 28, 28), torch.float32)],
        {
            "model_names": ["onnx_hrnet_hrnetv2_w48_img_cls_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "144", "stop": "192", "stride": "1"},
        },
    ),
    (
        Index119,
        [((1, 192, 28, 28), torch.float32)],
        {
            "model_names": ["onnx_hrnet_hrnetv2_w48_img_cls_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "48", "stride": "1"},
        },
    ),
    (
        Index279,
        [((1, 192, 28, 28), torch.float32)],
        {
            "model_names": ["onnx_hrnet_hrnetv2_w48_img_cls_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "48", "stop": "96", "stride": "1"},
        },
    ),
    (
        Index280,
        [((1, 192, 28, 28), torch.float32)],
        {
            "model_names": ["onnx_hrnet_hrnetv2_w48_img_cls_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "96", "stop": "192", "stride": "1"},
        },
    ),
    (
        Index293,
        [((1, 77), torch.int64)],
        {
            "model_names": ["pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "7", "stride": "1"},
        },
    ),
    (
        Index46,
        [((1, 32, 5, 64), torch.float32)],
        {
            "model_names": ["pt_phi1_microsoft_phi_1_seq_cls_hf", "pt_phi_1_5_microsoft_phi_1_5_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index47,
        [((1, 32, 5, 64), torch.float32)],
        {
            "model_names": ["pt_phi1_microsoft_phi_1_seq_cls_hf", "pt_phi_1_5_microsoft_phi_1_5_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "32", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index63,
        [((1, 32, 5, 32), torch.float32)],
        {
            "model_names": ["pt_phi1_microsoft_phi_1_seq_cls_hf", "pt_phi_1_5_microsoft_phi_1_5_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "16", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index27,
        [((1, 32, 5, 32), torch.float32)],
        {
            "model_names": ["pt_phi1_microsoft_phi_1_seq_cls_hf", "pt_phi_1_5_microsoft_phi_1_5_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "16", "stride": "1"},
        },
    ),
    (
        Index30,
        [((1, 204, 768), torch.bfloat16)],
        {
            "model_names": ["pt_vilt_mlm_mlm_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index283,
        [((3, 50, 1, 768), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_b_32_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-4", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index284,
        [((3, 50, 1, 768), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_b_32_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-4", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index285,
        [((3, 50, 1, 768), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_b_32_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-4", "start": "2", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index30,
        [((1, 50, 768), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_b_32_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index98,
        [((1, 3, 160, 160, 85), torch.bfloat16)],
        {
            "model_names": ["pt_yolo_v5_yolov5s_img_cls_torchhub_1280x1280"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "start": "0", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index99,
        [((1, 3, 160, 160, 85), torch.bfloat16)],
        {
            "model_names": ["pt_yolo_v5_yolov5s_img_cls_torchhub_1280x1280"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "start": "2", "stop": "4", "stride": "1"},
        },
    ),
    (
        Index100,
        [((1, 3, 160, 160, 85), torch.bfloat16)],
        {
            "model_names": ["pt_yolo_v5_yolov5s_img_cls_torchhub_1280x1280"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "start": "4", "stop": "85", "stride": "1"},
        },
    ),
    (
        Index293,
        [((1, 1, 7, 7), torch.float32)],
        {
            "model_names": ["onnx_nanogpt_financialsupport_nanogpt_text_gen_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "7", "stride": "1"},
        },
    ),
    (
        Index6,
        [((1, 128, 160, 160), torch.float32)],
        {
            "model_names": ["onnx_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index17,
        [((1, 128, 160, 160), torch.float32)],
        {
            "model_names": ["onnx_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "64", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index14,
        [((1, 256, 159, 159), torch.float32)],
        {
            "model_names": ["onnx_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index18,
        [((1, 256, 159, 159), torch.float32)],
        {
            "model_names": ["onnx_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "128", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index14,
        [((1, 256, 80, 80), torch.float32)],
        {
            "model_names": ["onnx_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index18,
        [((1, 256, 80, 80), torch.float32)],
        {
            "model_names": ["onnx_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "128", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index9,
        [((1, 512, 79, 79), torch.float32)],
        {
            "model_names": ["onnx_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index294,
        [((1, 512, 79, 79), torch.float32)],
        {
            "model_names": ["onnx_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "256", "stop": "512", "stride": "1"},
        },
    ),
    (
        Index9,
        [((1, 512, 40, 40), torch.float32)],
        {
            "model_names": ["onnx_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index294,
        [((1, 512, 40, 40), torch.float32)],
        {
            "model_names": ["onnx_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "256", "stop": "512", "stride": "1"},
        },
    ),
    (
        Index9,
        [((1, 512, 39, 39), torch.float32)],
        {
            "model_names": ["onnx_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index294,
        [((1, 512, 39, 39), torch.float32)],
        {
            "model_names": ["onnx_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "256", "stop": "512", "stride": "1"},
        },
    ),
    (
        Index9,
        [((1, 512, 20, 20), torch.float32)],
        {
            "model_names": ["onnx_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index294,
        [((1, 512, 20, 20), torch.float32)],
        {
            "model_names": ["onnx_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "256", "stop": "512", "stride": "1"},
        },
    ),
    (
        Index14,
        [((1, 256, 79, 79), torch.float32)],
        {
            "model_names": ["onnx_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index18,
        [((1, 256, 79, 79), torch.float32)],
        {
            "model_names": ["onnx_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "128", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index30,
        [((1, 9, 768), torch.float32)],
        {
            "model_names": [
                "pt_albert_imdb_seq_cls_hf",
                "pd_ernie_1_0_seq_cls_padlenlp",
                "pd_roberta_rbt4_ch_seq_cls_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index169,
        [((732, 12), torch.bfloat16)],
        {
            "model_names": ["pt_beit_base_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "start": "0", "stop": "729", "stride": "1"},
        },
    ),
    (
        Index170,
        [((732, 12), torch.bfloat16)],
        {
            "model_names": ["pt_beit_base_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "start": "729", "stop": "732", "stride": "1"},
        },
    ),
    (
        Index171,
        [((1, 12, 35, 128), torch.float32)],
        {
            "model_names": ["pt_qwen_coder_1_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "64", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index172,
        [((1, 12, 35, 128), torch.float32)],
        {
            "model_names": ["pt_qwen_coder_1_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index171,
        [((1, 2, 35, 128), torch.float32)],
        {
            "model_names": ["pt_qwen_coder_1_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "64", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index172,
        [((1, 2, 35, 128), torch.float32)],
        {
            "model_names": ["pt_qwen_coder_1_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index30,
        [((2048, 1024), torch.float32)],
        {
            "model_names": ["pt_stereo_small_music_generation_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index283,
        [((3, 197, 1, 1024), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_l_16_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-4", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index284,
        [((3, 197, 1, 1024), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_l_16_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-4", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index285,
        [((3, 197, 1, 1024), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_l_16_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-4", "start": "2", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index137,
        [((1, 64, 160, 160), torch.bfloat16)],
        {
            "model_names": ["pt_yolo_world_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "0", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index138,
        [((1, 64, 160, 160), torch.bfloat16)],
        {
            "model_names": ["pt_yolo_world_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "32", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index6,
        [((1, 128, 80, 80), torch.bfloat16)],
        {
            "model_names": ["pt_yolo_world_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index17,
        [((1, 128, 80, 80), torch.bfloat16)],
        {
            "model_names": ["pt_yolo_world_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "64", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index14,
        [((1, 256, 40, 40), torch.bfloat16)],
        {
            "model_names": ["pt_yolo_world_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "0", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index18,
        [((1, 256, 40, 40), torch.bfloat16)],
        {
            "model_names": ["pt_yolo_world_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "128", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index9,
        [((1, 512, 20, 20), torch.bfloat16)],
        {
            "model_names": ["pt_yolo_world_default_obj_det_github", "pt_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "0", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index294,
        [((1, 512, 20, 20), torch.bfloat16)],
        {
            "model_names": ["pt_yolo_world_default_obj_det_github", "pt_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "256", "stop": "512", "stride": "1"},
        },
    ),
    (
        Index118,
        [((1, 32, 160, 160), torch.bfloat16)],
        {
            "model_names": ["pt_yolov10_yolov10n_obj_det_github", "pt_yolov8_yolov8n_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "0", "stop": "16", "stride": "1"},
        },
    ),
    (
        Index136,
        [((1, 32, 160, 160), torch.bfloat16)],
        {
            "model_names": ["pt_yolov10_yolov10n_obj_det_github", "pt_yolov8_yolov8n_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "16", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index137,
        [((1, 64, 80, 80), torch.bfloat16)],
        {
            "model_names": ["pt_yolov10_yolov10n_obj_det_github", "pt_yolov8_yolov8n_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "0", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index138,
        [((1, 64, 80, 80), torch.bfloat16)],
        {
            "model_names": ["pt_yolov10_yolov10n_obj_det_github", "pt_yolov8_yolov8n_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "32", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index6,
        [((1, 128, 40, 40), torch.bfloat16)],
        {
            "model_names": ["pt_yolov10_yolov10n_obj_det_github", "pt_yolov8_yolov8n_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index17,
        [((1, 128, 40, 40), torch.bfloat16)],
        {
            "model_names": ["pt_yolov10_yolov10n_obj_det_github", "pt_yolov8_yolov8n_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "64", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index14,
        [((1, 256, 20, 20), torch.bfloat16)],
        {
            "model_names": ["pt_yolov10_yolov10n_obj_det_github", "pt_yolov8_yolov8n_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "0", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index18,
        [((1, 256, 20, 20), torch.bfloat16)],
        {
            "model_names": ["pt_yolov10_yolov10n_obj_det_github", "pt_yolov8_yolov8n_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "128", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index139,
        [((1, 2, 128, 400), torch.bfloat16)],
        {
            "model_names": ["pt_yolov10_yolov10n_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "start": "64", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index140,
        [((1, 2, 128, 400), torch.bfloat16)],
        {
            "model_names": ["pt_yolov10_yolov10n_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "start": "0", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index141,
        [((1, 2, 128, 400), torch.bfloat16)],
        {
            "model_names": ["pt_yolov10_yolov10n_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "start": "32", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index30,
        [((1, 6, 768), torch.float32)],
        {
            "model_names": ["onnx_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index293,
        [((2, 1, 7, 7), torch.float32)],
        {
            "model_names": ["onnx_clip_openai_clip_vit_base_patch32_text_gen_hf_text"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "7", "stride": "1"},
        },
    ),
    (
        Index101,
        [((1, 3, 224, 224), torch.float32)],
        {
            "model_names": ["onnx_googlenet_googlenet_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index102,
        [((1, 3, 224, 224), torch.float32)],
        {
            "model_names": ["onnx_googlenet_googlenet_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index103,
        [((1, 3, 224, 224), torch.float32)],
        {
            "model_names": ["onnx_googlenet_googlenet_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "2", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index14,
        [((1, 288, 28, 28), torch.float32)],
        {
            "model_names": ["onnx_googlenet_googlenet_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index18,
        [((1, 288, 28, 28), torch.float32)],
        {
            "model_names": ["onnx_googlenet_googlenet_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "128", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index209,
        [((1, 288, 28, 28), torch.float32)],
        {
            "model_names": ["onnx_googlenet_googlenet_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "256", "stop": "288", "stride": "1"},
        },
    ),
    (
        Index210,
        [((1, 304, 14, 14), torch.float32)],
        {
            "model_names": ["onnx_googlenet_googlenet_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "192", "stride": "1"},
        },
    ),
    (
        Index211,
        [((1, 304, 14, 14), torch.float32)],
        {
            "model_names": ["onnx_googlenet_googlenet_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "192", "stop": "288", "stride": "1"},
        },
    ),
    (
        Index212,
        [((1, 304, 14, 14), torch.float32)],
        {
            "model_names": ["onnx_googlenet_googlenet_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "288", "stop": "304", "stride": "1"},
        },
    ),
    (
        Index106,
        [((1, 296, 14, 14), torch.float32)],
        {
            "model_names": ["onnx_googlenet_googlenet_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "160", "stride": "1"},
        },
    ),
    (
        Index213,
        [((1, 296, 14, 14), torch.float32)],
        {
            "model_names": ["onnx_googlenet_googlenet_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "160", "stop": "272", "stride": "1"},
        },
    ),
    (
        Index214,
        [((1, 296, 14, 14), torch.float32)],
        {
            "model_names": ["onnx_googlenet_googlenet_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "272", "stop": "296", "stride": "1"},
        },
    ),
    (
        Index14,
        [((1, 280, 14, 14), torch.float32)],
        {
            "model_names": ["onnx_googlenet_googlenet_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index18,
        [((1, 280, 14, 14), torch.float32)],
        {
            "model_names": ["onnx_googlenet_googlenet_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "128", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index215,
        [((1, 280, 14, 14), torch.float32)],
        {
            "model_names": ["onnx_googlenet_googlenet_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "256", "stop": "280", "stride": "1"},
        },
    ),
    (
        Index133,
        [((1, 288, 14, 14), torch.float32)],
        {
            "model_names": ["onnx_googlenet_googlenet_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "112", "stride": "1"},
        },
    ),
    (
        Index216,
        [((1, 288, 14, 14), torch.float32)],
        {
            "model_names": ["onnx_googlenet_googlenet_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "112", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index209,
        [((1, 288, 14, 14), torch.float32)],
        {
            "model_names": ["onnx_googlenet_googlenet_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "256", "stop": "288", "stride": "1"},
        },
    ),
    (
        Index9,
        [((1, 448, 14, 14), torch.float32)],
        {
            "model_names": ["onnx_googlenet_googlenet_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index12,
        [((1, 448, 14, 14), torch.float32)],
        {
            "model_names": ["onnx_googlenet_googlenet_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "256", "stop": "416", "stride": "1"},
        },
    ),
    (
        Index13,
        [((1, 448, 14, 14), torch.float32)],
        {
            "model_names": ["onnx_googlenet_googlenet_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "416", "stop": "448", "stride": "1"},
        },
    ),
    (
        Index194,
        [((1, 624, 7, 7), torch.float32)],
        {
            "model_names": ["onnx_googlenet_googlenet_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "384", "stride": "1"},
        },
    ),
    (
        Index195,
        [((1, 624, 7, 7), torch.float32)],
        {
            "model_names": ["onnx_googlenet_googlenet_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "384", "stop": "576", "stride": "1"},
        },
    ),
    (
        Index217,
        [((1, 624, 7, 7), torch.float32)],
        {
            "model_names": ["onnx_googlenet_googlenet_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "576", "stop": "624", "stride": "1"},
        },
    ),
    (
        Index101,
        [((2, 1, 11), torch.float32)],
        {
            "model_names": ["pd_bert_chinese_roberta_base_qa_padlenlp"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index102,
        [((2, 1, 11), torch.float32)],
        {
            "model_names": ["pd_bert_chinese_roberta_base_qa_padlenlp"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index30,
        [((2, 4, 768), torch.float32)],
        {
            "model_names": ["pd_blip_salesforce_blip_image_captioning_base_img_captioning_padlenlp"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index19,
        [((3, 1, 12, 257, 64), torch.bfloat16)],
        {
            "model_names": ["pt_mgp_default_scene_text_recognition_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-5", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index20,
        [((3, 1, 12, 257, 64), torch.bfloat16)],
        {
            "model_names": ["pt_mgp_default_scene_text_recognition_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-5", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index21,
        [((3, 1, 12, 257, 64), torch.bfloat16)],
        {
            "model_names": ["pt_mgp_default_scene_text_recognition_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-5", "start": "2", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index47,
        [((1, 16, 29, 64), torch.float32)],
        {
            "model_names": ["pt_qwen1_5_0_5b_chat_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "32", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index46,
        [((1, 16, 29, 64), torch.float32)],
        {
            "model_names": ["pt_qwen1_5_0_5b_chat_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index47,
        [((1, 14, 39, 64), torch.float32)],
        {
            "model_names": ["pt_qwen_v2_0_5b_instruct_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "32", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index46,
        [((1, 14, 39, 64), torch.float32)],
        {
            "model_names": ["pt_qwen_v2_0_5b_instruct_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index47,
        [((1, 2, 39, 64), torch.float32)],
        {
            "model_names": ["pt_qwen_v2_0_5b_instruct_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "32", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index46,
        [((1, 2, 39, 64), torch.float32)],
        {
            "model_names": ["pt_qwen_v2_0_5b_instruct_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index30,
        [((1, 334, 64, 3, 64), torch.float32)],
        {
            "model_names": ["onnx_fuyu_adept_fuyu_8b_clm_hf", "pt_fuyu_adept_fuyu_8b_qa_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index31,
        [((1, 334, 64, 3, 64), torch.float32)],
        {
            "model_names": ["onnx_fuyu_adept_fuyu_8b_clm_hf", "pt_fuyu_adept_fuyu_8b_qa_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index64,
        [((1, 334, 64, 3, 64), torch.float32)],
        {
            "model_names": ["onnx_fuyu_adept_fuyu_8b_clm_hf", "pt_fuyu_adept_fuyu_8b_qa_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "2", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index46,
        [((1, 64, 334, 64), torch.float32)],
        {
            "model_names": ["onnx_fuyu_adept_fuyu_8b_clm_hf", "pt_fuyu_adept_fuyu_8b_qa_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index47,
        [((1, 64, 334, 64), torch.float32)],
        {
            "model_names": ["onnx_fuyu_adept_fuyu_8b_clm_hf", "pt_fuyu_adept_fuyu_8b_qa_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "32", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index63,
        [((1, 64, 334, 32), torch.float32)],
        {
            "model_names": ["onnx_fuyu_adept_fuyu_8b_clm_hf", "pt_fuyu_adept_fuyu_8b_qa_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "16", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index27,
        [((1, 64, 334, 32), torch.float32)],
        {
            "model_names": ["onnx_fuyu_adept_fuyu_8b_clm_hf", "pt_fuyu_adept_fuyu_8b_qa_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "16", "stride": "1"},
        },
    ),
    (
        Index101,
        [((1, 2, 30, 40), torch.float32)],
        {
            "model_names": ["onnx_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index102,
        [((1, 2, 30, 40), torch.float32)],
        {
            "model_names": ["onnx_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index101,
        [((1, 2, 60, 80), torch.float32)],
        {
            "model_names": ["onnx_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index102,
        [((1, 2, 60, 80), torch.float32)],
        {
            "model_names": ["onnx_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index101,
        [((1, 2, 120, 160), torch.float32)],
        {
            "model_names": ["onnx_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index102,
        [((1, 2, 120, 160), torch.float32)],
        {
            "model_names": ["onnx_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index295,
        [((1, 577, 1024), torch.float32)],
        {
            "model_names": ["pt_llava_1_5_7b_cond_gen_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "1", "stop": "577", "stride": "1"},
        },
    ),
    (
        Index171,
        [((1, 32, 596, 128), torch.float32)],
        {
            "model_names": ["pt_llava_1_5_7b_cond_gen_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "64", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index172,
        [((1, 32, 596, 128), torch.float32)],
        {
            "model_names": ["pt_llava_1_5_7b_cond_gen_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index171,
        [((1, 32, 12, 128), torch.float32)],
        {
            "model_names": [
                "pt_ministral_ministral_3b_instruct_clm_hf",
                "pt_ministral_ministral_8b_instruct_clm_hf",
                "pt_mistral_7b_clm_hf",
                "pt_mistral_7b_instruct_v03_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "64", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index172,
        [((1, 32, 12, 128), torch.float32)],
        {
            "model_names": [
                "pt_ministral_ministral_3b_instruct_clm_hf",
                "pt_ministral_ministral_8b_instruct_clm_hf",
                "pt_mistral_7b_clm_hf",
                "pt_mistral_7b_instruct_v03_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index171,
        [((1, 8, 12, 128), torch.float32)],
        {
            "model_names": [
                "pt_ministral_ministral_3b_instruct_clm_hf",
                "pt_ministral_ministral_8b_instruct_clm_hf",
                "pt_mistral_7b_clm_hf",
                "pt_mistral_7b_instruct_v03_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "64", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index172,
        [((1, 8, 12, 128), torch.float32)],
        {
            "model_names": [
                "pt_ministral_ministral_3b_instruct_clm_hf",
                "pt_ministral_ministral_8b_instruct_clm_hf",
                "pt_mistral_7b_clm_hf",
                "pt_mistral_7b_instruct_v03_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index296,
        [((1, 256, 9216), torch.float32)],
        {
            "model_names": [
                "pt_phi3_5_mini_instruct_clm_hf",
                "pt_phi3_microsoft_phi_3_mini_128k_instruct_clm_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "3072", "stride": "1"},
        },
    ),
    (
        Index297,
        [((1, 256, 9216), torch.float32)],
        {
            "model_names": [
                "pt_phi3_5_mini_instruct_clm_hf",
                "pt_phi3_microsoft_phi_3_mini_128k_instruct_clm_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "3072", "stop": "6144", "stride": "1"},
        },
    ),
    (
        Index298,
        [((1, 256, 9216), torch.float32)],
        {
            "model_names": [
                "pt_phi3_5_mini_instruct_clm_hf",
                "pt_phi3_microsoft_phi_3_mini_128k_instruct_clm_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "6144", "stop": "9216", "stride": "1"},
        },
    ),
    (
        Index166,
        [((1, 32, 256, 96), torch.float32)],
        {
            "model_names": [
                "pt_phi3_5_mini_instruct_clm_hf",
                "pt_phi3_microsoft_phi_3_mini_128k_instruct_clm_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "48", "stop": "96", "stride": "1"},
        },
    ),
    (
        Index167,
        [((1, 32, 256, 96), torch.float32)],
        {
            "model_names": [
                "pt_phi3_5_mini_instruct_clm_hf",
                "pt_phi3_microsoft_phi_3_mini_128k_instruct_clm_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "48", "stride": "1"},
        },
    ),
    (
        Index299,
        [((16384, 3072), torch.float32)],
        {
            "model_names": [
                "pt_phi3_5_mini_instruct_clm_hf",
                "pt_phi3_microsoft_phi_3_mini_128k_instruct_clm_hf",
                "pt_phi3_microsoft_phi_3_mini_128k_instruct_seq_cls_hf",
                "pt_phi3_microsoft_phi_3_mini_128k_instruct_token_cls_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "8192", "stop": "16384", "stride": "1"},
        },
    ),
    (
        Index300,
        [((16384, 3072), torch.float32)],
        {
            "model_names": [
                "pt_phi3_5_mini_instruct_clm_hf",
                "pt_phi3_microsoft_phi_3_mini_128k_instruct_clm_hf",
                "pt_phi3_microsoft_phi_3_mini_128k_instruct_seq_cls_hf",
                "pt_phi3_microsoft_phi_3_mini_128k_instruct_token_cls_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "8192", "stride": "1"},
        },
    ),
    (
        Index30,
        [((2048, 2048), torch.float32)],
        {
            "model_names": ["pt_stereo_large_music_generation_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index30,
        [((1, 11, 768), torch.float32)],
        {
            "model_names": ["pd_bert_chinese_roberta_base_seq_cls_padlenlp"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index30,
        [((1, 197, 384), torch.bfloat16)],
        {
            "model_names": ["pt_deit_small_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index14,
        [((1, 256, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w64_img_cls_timm", "pt_hrnet_hrnetv2_w64_osmr_img_cls_osmr"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "0", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index15,
        [((1, 256, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w64_img_cls_timm", "pt_hrnet_hrnetv2_w64_osmr_img_cls_osmr"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "128", "stop": "192", "stride": "1"},
        },
    ),
    (
        Index16,
        [((1, 256, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w64_img_cls_timm", "pt_hrnet_hrnetv2_w64_osmr_img_cls_osmr"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "192", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index6,
        [((1, 256, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w64_img_cls_timm", "pt_hrnet_hrnetv2_w64_osmr_img_cls_osmr"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index17,
        [((1, 256, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w64_img_cls_timm", "pt_hrnet_hrnetv2_w64_osmr_img_cls_osmr"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "64", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index18,
        [((1, 256, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w64_img_cls_timm", "pt_hrnet_hrnetv2_w64_osmr_img_cls_osmr"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "128", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index32,
        [((50,), torch.bfloat16)],
        {
            "model_names": ["pt_stable_diffusion_stable_diffusion_xl_base_1_0_cond_gen_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index301,
        [((2, 320), torch.bfloat16)],
        {
            "model_names": ["pt_stable_diffusion_stable_diffusion_xl_base_1_0_cond_gen_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "start": "160", "stop": "320", "stride": "1"},
        },
    ),
    (
        Index302,
        [((2, 320), torch.bfloat16)],
        {
            "model_names": ["pt_stable_diffusion_stable_diffusion_xl_base_1_0_cond_gen_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "start": "0", "stop": "160", "stride": "1"},
        },
    ),
    (
        Index303,
        [((5120, 640), torch.bfloat16)],
        {
            "model_names": ["pt_stable_diffusion_stable_diffusion_xl_base_1_0_cond_gen_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "start": "0", "stop": "2560", "stride": "1"},
        },
    ),
    (
        Index304,
        [((5120, 640), torch.bfloat16)],
        {
            "model_names": ["pt_stable_diffusion_stable_diffusion_xl_base_1_0_cond_gen_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "start": "2560", "stop": "5120", "stride": "1"},
        },
    ),
    (
        Index305,
        [((5120,), torch.bfloat16)],
        {
            "model_names": ["pt_stable_diffusion_stable_diffusion_xl_base_1_0_cond_gen_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "start": "0", "stop": "2560", "stride": "1"},
        },
    ),
    (
        Index306,
        [((5120,), torch.bfloat16)],
        {
            "model_names": ["pt_stable_diffusion_stable_diffusion_xl_base_1_0_cond_gen_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "start": "2560", "stop": "5120", "stride": "1"},
        },
    ),
    (
        Index307,
        [((10240, 1280), torch.bfloat16)],
        {
            "model_names": ["pt_stable_diffusion_stable_diffusion_xl_base_1_0_cond_gen_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "start": "0", "stop": "5120", "stride": "1"},
        },
    ),
    (
        Index308,
        [((10240, 1280), torch.bfloat16)],
        {
            "model_names": ["pt_stable_diffusion_stable_diffusion_xl_base_1_0_cond_gen_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "start": "5120", "stop": "10240", "stride": "1"},
        },
    ),
    (
        Index309,
        [((10240,), torch.bfloat16)],
        {
            "model_names": ["pt_stable_diffusion_stable_diffusion_xl_base_1_0_cond_gen_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "start": "0", "stop": "5120", "stride": "1"},
        },
    ),
    (
        Index310,
        [((10240,), torch.bfloat16)],
        {
            "model_names": ["pt_stable_diffusion_stable_diffusion_xl_base_1_0_cond_gen_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "start": "5120", "stop": "10240", "stride": "1"},
        },
    ),
    (
        Index286,
        [((2050, 1024), torch.float32)],
        {
            "model_names": ["pt_xglm_xglm_564m_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "2", "stop": "258", "stride": "1"},
        },
    ),
    (
        Index6,
        [((1, 128, 160, 160), torch.bfloat16)],
        {
            "model_names": ["pt_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index17,
        [((1, 128, 160, 160), torch.bfloat16)],
        {
            "model_names": ["pt_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "64", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index14,
        [((1, 256, 159, 159), torch.bfloat16)],
        {
            "model_names": ["pt_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "0", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index18,
        [((1, 256, 159, 159), torch.bfloat16)],
        {
            "model_names": ["pt_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "128", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index14,
        [((1, 256, 80, 80), torch.bfloat16)],
        {
            "model_names": ["pt_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "0", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index18,
        [((1, 256, 80, 80), torch.bfloat16)],
        {
            "model_names": ["pt_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "128", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index9,
        [((1, 512, 79, 79), torch.bfloat16)],
        {
            "model_names": ["pt_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "0", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index294,
        [((1, 512, 79, 79), torch.bfloat16)],
        {
            "model_names": ["pt_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "256", "stop": "512", "stride": "1"},
        },
    ),
    (
        Index9,
        [((1, 512, 40, 40), torch.bfloat16)],
        {
            "model_names": ["pt_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "0", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index294,
        [((1, 512, 40, 40), torch.bfloat16)],
        {
            "model_names": ["pt_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "256", "stop": "512", "stride": "1"},
        },
    ),
    (
        Index9,
        [((1, 512, 39, 39), torch.bfloat16)],
        {
            "model_names": ["pt_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "0", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index294,
        [((1, 512, 39, 39), torch.bfloat16)],
        {
            "model_names": ["pt_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "256", "stop": "512", "stride": "1"},
        },
    ),
    (
        Index14,
        [((1, 256, 79, 79), torch.bfloat16)],
        {
            "model_names": ["pt_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "0", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index18,
        [((1, 256, 79, 79), torch.bfloat16)],
        {
            "model_names": ["pt_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "128", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index311,
        [((1, 25, 34), torch.bfloat16)],
        {
            "model_names": ["pt_detr_resnet_50_obj_det_hf", "pt_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "start": "-1", "stop": "25", "stride": "1"},
        },
    ),
    (
        Index312,
        [((1, 25, 34), torch.bfloat16)],
        {
            "model_names": ["pt_detr_resnet_50_obj_det_hf", "pt_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "start": "-1", "stop": "34", "stride": "1"},
        },
    ),
    (
        Index313,
        [((1, 25, 34, 128), torch.bfloat16)],
        {
            "model_names": ["pt_detr_resnet_50_obj_det_hf", "pt_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "start": "0", "stop": "128", "stride": "2"},
        },
    ),
    (
        Index314,
        [((1, 25, 34, 128), torch.bfloat16)],
        {
            "model_names": ["pt_detr_resnet_50_obj_det_hf", "pt_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "start": "1", "stop": "128", "stride": "2"},
        },
    ),
    (
        Index47,
        [((1, 8, 256, 64), torch.float32)],
        {
            "model_names": ["pt_llama3_llama_3_2_1b_clm_hf", "pt_llama3_llama_3_2_1b_instruct_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "32", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index46,
        [((1, 8, 256, 64), torch.float32)],
        {
            "model_names": ["pt_llama3_llama_3_2_1b_clm_hf", "pt_llama3_llama_3_2_1b_instruct_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index171,
        [((1, 32, 1063, 128), torch.float32)],
        {
            "model_names": ["pt_deepseek_7b_instruct_qa_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "64", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index172,
        [((1, 32, 1063, 128), torch.float32)],
        {
            "model_names": ["pt_deepseek_7b_instruct_qa_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index255,
        [((1, 12, 522, 256), torch.float32)],
        {
            "model_names": ["pt_falcon3_tiiuae_falcon3_3b_base_clm_hf", "pt_falcon3_tiiuae_falcon3_7b_base_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "128", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index23,
        [((1, 12, 522, 256), torch.float32)],
        {
            "model_names": ["pt_falcon3_tiiuae_falcon3_3b_base_clm_hf", "pt_falcon3_tiiuae_falcon3_7b_base_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index315,
        [((1, 6, 73, 64), torch.float32)],
        {
            "model_names": ["pt_falcon_tiiuae_falcon_7b_instruct_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "-2", "stride": "1"},
        },
    ),
    (
        Index316,
        [((1, 6, 73, 64), torch.float32)],
        {
            "model_names": ["pt_falcon_tiiuae_falcon_7b_instruct_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "-2", "stop": "-1", "stride": "1"},
        },
    ),
    (
        Index317,
        [((1, 6, 73, 64), torch.float32)],
        {
            "model_names": ["pt_falcon_tiiuae_falcon_7b_instruct_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "72", "stop": "73", "stride": "1"},
        },
    ),
    (
        Index47,
        [((1, 71, 6, 64), torch.float32)],
        {
            "model_names": ["pt_falcon_tiiuae_falcon_7b_instruct_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "32", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index46,
        [((1, 71, 6, 64), torch.float32)],
        {
            "model_names": ["pt_falcon_tiiuae_falcon_7b_instruct_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index47,
        [((1, 1, 6, 64), torch.float32)],
        {
            "model_names": ["pt_falcon_tiiuae_falcon_7b_instruct_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "32", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index46,
        [((1, 1, 6, 64), torch.float32)],
        {
            "model_names": ["pt_falcon_tiiuae_falcon_7b_instruct_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index255,
        [((1, 8, 356, 256), torch.float32)],
        {
            "model_names": ["pt_gemma_google_gemma_1_1_2b_it_qa_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "128", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index23,
        [((1, 8, 356, 256), torch.float32)],
        {
            "model_names": ["pt_gemma_google_gemma_1_1_2b_it_qa_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index255,
        [((1, 1, 356, 256), torch.float32)],
        {
            "model_names": ["pt_gemma_google_gemma_1_1_2b_it_qa_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "128", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index23,
        [((1, 1, 356, 256), torch.float32)],
        {
            "model_names": ["pt_gemma_google_gemma_1_1_2b_it_qa_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index255,
        [((1, 16, 356, 256), torch.float32)],
        {
            "model_names": ["pt_gemma_google_gemma_1_1_7b_it_qa_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "128", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index23,
        [((1, 16, 356, 256), torch.float32)],
        {
            "model_names": ["pt_gemma_google_gemma_1_1_7b_it_qa_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index255,
        [((1, 8, 512, 256), torch.float32)],
        {
            "model_names": ["pt_gemma_google_gemma_2b_text_gen_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "128", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index23,
        [((1, 8, 512, 256), torch.float32)],
        {
            "model_names": ["pt_gemma_google_gemma_2b_text_gen_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index255,
        [((1, 1, 512, 256), torch.float32)],
        {
            "model_names": ["pt_gemma_google_gemma_2b_text_gen_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "128", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index23,
        [((1, 1, 512, 256), torch.float32)],
        {
            "model_names": ["pt_gemma_google_gemma_2b_text_gen_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index171,
        [((1, 32, 256, 128), torch.float32)],
        {
            "model_names": [
                "pt_llama3_huggyllama_7b_clm_hf",
                "pt_llama3_llama_3_1_8b_clm_hf",
                "pt_llama3_llama_3_1_8b_instruct_clm_hf",
                "pt_llama3_llama_3_8b_clm_hf",
                "pt_llama3_llama_3_8b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "64", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index172,
        [((1, 32, 256, 128), torch.float32)],
        {
            "model_names": [
                "pt_llama3_huggyllama_7b_clm_hf",
                "pt_llama3_llama_3_1_8b_clm_hf",
                "pt_llama3_llama_3_1_8b_instruct_clm_hf",
                "pt_llama3_llama_3_8b_clm_hf",
                "pt_llama3_llama_3_8b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index171,
        [((1, 32, 4, 128), torch.float32)],
        {
            "model_names": [
                "pt_llama3_huggyllama_7b_seq_cls_hf",
                "pt_llama3_llama_3_1_8b_instruct_seq_cls_hf",
                "pt_llama3_llama_3_1_8b_seq_cls_hf",
                "pt_llama3_llama_3_8b_instruct_seq_cls_hf",
                "pt_llama3_llama_3_8b_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "64", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index172,
        [((1, 32, 4, 128), torch.float32)],
        {
            "model_names": [
                "pt_llama3_huggyllama_7b_seq_cls_hf",
                "pt_llama3_llama_3_1_8b_instruct_seq_cls_hf",
                "pt_llama3_llama_3_1_8b_seq_cls_hf",
                "pt_llama3_llama_3_8b_instruct_seq_cls_hf",
                "pt_llama3_llama_3_8b_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index171,
        [((1, 8, 256, 128), torch.float32)],
        {
            "model_names": [
                "pt_llama3_llama_3_1_8b_clm_hf",
                "pt_llama3_llama_3_1_8b_instruct_clm_hf",
                "pt_llama3_llama_3_2_3b_clm_hf",
                "pt_llama3_llama_3_2_3b_instruct_clm_hf",
                "pt_llama3_llama_3_8b_clm_hf",
                "pt_llama3_llama_3_8b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "64", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index172,
        [((1, 8, 256, 128), torch.float32)],
        {
            "model_names": [
                "pt_llama3_llama_3_1_8b_clm_hf",
                "pt_llama3_llama_3_1_8b_instruct_clm_hf",
                "pt_llama3_llama_3_2_3b_clm_hf",
                "pt_llama3_llama_3_2_3b_instruct_clm_hf",
                "pt_llama3_llama_3_8b_clm_hf",
                "pt_llama3_llama_3_8b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index171,
        [((1, 8, 4, 128), torch.float32)],
        {
            "model_names": [
                "pt_llama3_llama_3_1_8b_instruct_seq_cls_hf",
                "pt_llama3_llama_3_1_8b_seq_cls_hf",
                "pt_llama3_llama_3_2_3b_instruct_seq_cls_hf",
                "pt_llama3_llama_3_2_3b_seq_cls_hf",
                "pt_llama3_llama_3_8b_instruct_seq_cls_hf",
                "pt_llama3_llama_3_8b_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "64", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index172,
        [((1, 8, 4, 128), torch.float32)],
        {
            "model_names": [
                "pt_llama3_llama_3_1_8b_instruct_seq_cls_hf",
                "pt_llama3_llama_3_1_8b_seq_cls_hf",
                "pt_llama3_llama_3_2_3b_instruct_seq_cls_hf",
                "pt_llama3_llama_3_2_3b_seq_cls_hf",
                "pt_llama3_llama_3_8b_instruct_seq_cls_hf",
                "pt_llama3_llama_3_8b_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index171,
        [((1, 24, 256, 128), torch.float32)],
        {
            "model_names": ["pt_llama3_llama_3_2_3b_clm_hf", "pt_llama3_llama_3_2_3b_instruct_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "64", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index172,
        [((1, 24, 256, 128), torch.float32)],
        {
            "model_names": ["pt_llama3_llama_3_2_3b_clm_hf", "pt_llama3_llama_3_2_3b_instruct_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index171,
        [((1, 24, 4, 128), torch.float32)],
        {
            "model_names": ["pt_llama3_llama_3_2_3b_instruct_seq_cls_hf", "pt_llama3_llama_3_2_3b_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "64", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index172,
        [((1, 24, 4, 128), torch.float32)],
        {
            "model_names": ["pt_llama3_llama_3_2_3b_instruct_seq_cls_hf", "pt_llama3_llama_3_2_3b_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index9,
        [((1, 1792, 56, 56), torch.bfloat16)],
        {
            "model_names": ["pt_monodle_dla34_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "0", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index294,
        [((1, 1792, 56, 56), torch.bfloat16)],
        {
            "model_names": ["pt_monodle_dla34_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "256", "stop": "512", "stride": "1"},
        },
    ),
    (
        Index318,
        [((1, 1792, 56, 56), torch.bfloat16)],
        {
            "model_names": ["pt_monodle_dla34_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "512", "stop": "768", "stride": "1"},
        },
    ),
    (
        Index319,
        [((1, 1792, 56, 56), torch.bfloat16)],
        {
            "model_names": ["pt_monodle_dla34_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "768", "stop": "1024", "stride": "1"},
        },
    ),
    (
        Index320,
        [((1, 1792, 56, 56), torch.bfloat16)],
        {
            "model_names": ["pt_monodle_dla34_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "1024", "stop": "1280", "stride": "1"},
        },
    ),
    (
        Index321,
        [((1, 1792, 56, 56), torch.bfloat16)],
        {
            "model_names": ["pt_monodle_dla34_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "1280", "stop": "1536", "stride": "1"},
        },
    ),
    (
        Index322,
        [((1, 1792, 56, 56), torch.bfloat16)],
        {
            "model_names": ["pt_monodle_dla34_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "1536", "stop": "1792", "stride": "1"},
        },
    ),
    (
        Index46,
        [((1, 32, 256, 80), torch.float32)],
        {
            "model_names": ["pt_phi2_microsoft_phi_2_pytdml_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index291,
        [((1, 32, 256, 80), torch.float32)],
        {
            "model_names": ["pt_phi2_microsoft_phi_2_pytdml_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "32", "stop": "80", "stride": "1"},
        },
    ),
    (
        Index46,
        [((1, 32, 12, 80), torch.float32)],
        {
            "model_names": ["pt_phi2_microsoft_phi_2_pytdml_token_cls_hf", "pt_phi2_microsoft_phi_2_token_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index291,
        [((1, 32, 12, 80), torch.float32)],
        {
            "model_names": ["pt_phi2_microsoft_phi_2_pytdml_token_cls_hf", "pt_phi2_microsoft_phi_2_token_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "32", "stop": "80", "stride": "1"},
        },
    ),
    (
        Index296,
        [((1, 5, 9216), torch.float32)],
        {
            "model_names": [
                "pt_phi3_microsoft_phi_3_mini_128k_instruct_seq_cls_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "3072", "stride": "1"},
        },
    ),
    (
        Index297,
        [((1, 5, 9216), torch.float32)],
        {
            "model_names": [
                "pt_phi3_microsoft_phi_3_mini_128k_instruct_seq_cls_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "3072", "stop": "6144", "stride": "1"},
        },
    ),
    (
        Index298,
        [((1, 5, 9216), torch.float32)],
        {
            "model_names": [
                "pt_phi3_microsoft_phi_3_mini_128k_instruct_seq_cls_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "6144", "stop": "9216", "stride": "1"},
        },
    ),
    (
        Index166,
        [((1, 32, 5, 96), torch.float32)],
        {
            "model_names": [
                "pt_phi3_microsoft_phi_3_mini_128k_instruct_seq_cls_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "48", "stop": "96", "stride": "1"},
        },
    ),
    (
        Index167,
        [((1, 32, 5, 96), torch.float32)],
        {
            "model_names": [
                "pt_phi3_microsoft_phi_3_mini_128k_instruct_seq_cls_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "48", "stride": "1"},
        },
    ),
    (
        Index296,
        [((1, 13, 9216), torch.float32)],
        {
            "model_names": [
                "pt_phi3_microsoft_phi_3_mini_128k_instruct_token_cls_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "3072", "stride": "1"},
        },
    ),
    (
        Index297,
        [((1, 13, 9216), torch.float32)],
        {
            "model_names": [
                "pt_phi3_microsoft_phi_3_mini_128k_instruct_token_cls_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "3072", "stop": "6144", "stride": "1"},
        },
    ),
    (
        Index298,
        [((1, 13, 9216), torch.float32)],
        {
            "model_names": [
                "pt_phi3_microsoft_phi_3_mini_128k_instruct_token_cls_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "6144", "stop": "9216", "stride": "1"},
        },
    ),
    (
        Index166,
        [((1, 32, 13, 96), torch.float32)],
        {
            "model_names": [
                "pt_phi3_microsoft_phi_3_mini_128k_instruct_token_cls_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "48", "stop": "96", "stride": "1"},
        },
    ),
    (
        Index167,
        [((1, 32, 13, 96), torch.float32)],
        {
            "model_names": [
                "pt_phi3_microsoft_phi_3_mini_128k_instruct_token_cls_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "48", "stride": "1"},
        },
    ),
    (
        Index309,
        [((1, 5, 7680), torch.float32)],
        {
            "model_names": ["pt_phi4_microsoft_phi_4_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "5120", "stride": "1"},
        },
    ),
    (
        Index323,
        [((1, 5, 7680), torch.float32)],
        {
            "model_names": ["pt_phi4_microsoft_phi_4_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "5120", "stop": "6400", "stride": "1"},
        },
    ),
    (
        Index324,
        [((1, 5, 7680), torch.float32)],
        {
            "model_names": ["pt_phi4_microsoft_phi_4_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "6400", "stop": "7680", "stride": "1"},
        },
    ),
    (
        Index171,
        [((1, 40, 5, 128), torch.float32)],
        {
            "model_names": ["pt_phi4_microsoft_phi_4_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "64", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index172,
        [((1, 40, 5, 128), torch.float32)],
        {
            "model_names": ["pt_phi4_microsoft_phi_4_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index171,
        [((1, 10, 5, 128), torch.float32)],
        {
            "model_names": ["pt_phi4_microsoft_phi_4_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "64", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index172,
        [((1, 10, 5, 128), torch.float32)],
        {
            "model_names": ["pt_phi4_microsoft_phi_4_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index325,
        [((35840, 5120), torch.float32)],
        {
            "model_names": ["pt_phi4_microsoft_phi_4_seq_cls_hf", "pt_phi4_microsoft_phi_4_token_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "17920", "stop": "35840", "stride": "1"},
        },
    ),
    (
        Index326,
        [((35840, 5120), torch.float32)],
        {
            "model_names": ["pt_phi4_microsoft_phi_4_seq_cls_hf", "pt_phi4_microsoft_phi_4_token_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "17920", "stride": "1"},
        },
    ),
    (
        Index309,
        [((1, 12, 7680), torch.float32)],
        {
            "model_names": ["pt_phi4_microsoft_phi_4_token_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "5120", "stride": "1"},
        },
    ),
    (
        Index323,
        [((1, 12, 7680), torch.float32)],
        {
            "model_names": ["pt_phi4_microsoft_phi_4_token_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "5120", "stop": "6400", "stride": "1"},
        },
    ),
    (
        Index324,
        [((1, 12, 7680), torch.float32)],
        {
            "model_names": ["pt_phi4_microsoft_phi_4_token_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "6400", "stop": "7680", "stride": "1"},
        },
    ),
    (
        Index171,
        [((1, 40, 12, 128), torch.float32)],
        {
            "model_names": ["pt_phi4_microsoft_phi_4_token_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "64", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index172,
        [((1, 40, 12, 128), torch.float32)],
        {
            "model_names": ["pt_phi4_microsoft_phi_4_token_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index171,
        [((1, 10, 12, 128), torch.float32)],
        {
            "model_names": ["pt_phi4_microsoft_phi_4_token_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "64", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index172,
        [((1, 10, 12, 128), torch.float32)],
        {
            "model_names": ["pt_phi4_microsoft_phi_4_token_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index171,
        [((1, 28, 13, 128), torch.float32)],
        {
            "model_names": ["pt_qwen_v2_qwen_qwen2_7b_token_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "64", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index172,
        [((1, 28, 13, 128), torch.float32)],
        {
            "model_names": ["pt_qwen_v2_qwen_qwen2_7b_token_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index171,
        [((1, 4, 13, 128), torch.float32)],
        {
            "model_names": ["pt_qwen_v2_qwen_qwen2_7b_token_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "64", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index172,
        [((1, 4, 13, 128), torch.float32)],
        {
            "model_names": ["pt_qwen_v2_qwen_qwen2_7b_token_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index171,
        [((1, 32, 128, 128), torch.float32)],
        {
            "model_names": ["pt_qwen_v3_4b_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "64", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index172,
        [((1, 32, 128, 128), torch.float32)],
        {
            "model_names": ["pt_qwen_v3_4b_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index171,
        [((4, 16, 31, 128), torch.bfloat16)],
        {
            "model_names": ["pt_qwen_v3_embedding_0_6b_sentence_embed_gen_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "start": "64", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index172,
        [((4, 16, 31, 128), torch.bfloat16)],
        {
            "model_names": ["pt_qwen_v3_embedding_0_6b_sentence_embed_gen_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
]


@pytest.mark.nightly_models_ops
@pytest.mark.parametrize("forge_module_and_shapes_dtypes", forge_modules_and_shapes_dtypes_list, ids=ids_func)
@pytest.mark.parametrize("training_test", [False, True], ids=["inference", "training"])
def test_module(forge_module_and_shapes_dtypes, training_test):

    record_forge_op_name("Index")

    forge_module, operand_shapes_dtypes, metadata = forge_module_and_shapes_dtypes

    pcc = metadata.get("pcc")

    for metadata_name, metadata_value in metadata.items():
        if metadata_name in ["pcc"]:
            continue
        elif metadata_name == "model_names":
            record_op_model_names(metadata_value)
        elif metadata_name == "args":
            record_forge_op_args(metadata_value)
        else:
            logger.warning(
                "No utility function available in forge property handler to record %s property", metadata_name
            )

    max_int = 1000
    inputs = [
        Tensor.create_from_shape(operand_shape, operand_dtype, max_int=max_int, requires_grad=training_test)
        for operand_shape, operand_dtype in operand_shapes_dtypes
    ]

    framework_model = forge_module(forge_module.__name__)

    for name, parameter in framework_model._parameters.items():
        parameter_tensor = Tensor.create_torch_tensor(
            shape=parameter.shape.get_pytorch_shape(),
            dtype=parameter.pt_data_format,
            max_int=max_int,
            requires_grad=training_test,
        )
        framework_model.set_parameter(name, parameter_tensor)

    for name, constant in framework_model._constants.items():
        constant_tensor = Tensor.create_torch_tensor(
            shape=constant.shape.get_pytorch_shape(),
            dtype=constant.pt_data_format,
            max_int=max_int,
            requires_grad=training_test,
        )
        framework_model.set_constant(name, constant_tensor)

    record_single_op_operands_info(framework_model, inputs)

    compiler_cfg = forge.config.CompilerConfig()
    if "default_df_override" in metadata.keys():
        compiler_cfg.default_df_override = forge.DataFormat.from_json(metadata["default_df_override"])

    compiled_model = compile(framework_model, sample_inputs=inputs, compiler_cfg=compiler_cfg, training=training_test)

    verify(
        inputs,
        framework_model,
        compiled_model,
        with_backward=training_test,
        verify_cfg=VerifyConfig(value_checker=AutomaticValueChecker(pcc=pcc)),
    )
