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
from forge.verify.config import VerifyConfig
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
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=1, stop=2, stride=1)
        return index_output_1


class Index2(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=2, stop=3, stride=1)
        return index_output_1


class Index3(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=3, stop=4, stride=1)
        return index_output_1


class Index4(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=0, stop=7, stride=1)
        return index_output_1


class Index5(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=11, stride=1)
        return index_output_1


class Index6(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=0, stop=128, stride=1)
        return index_output_1


class Index7(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=0, stop=384, stride=1)
        return index_output_1


class Index8(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=0, stop=1, stride=1)
        return index_output_1


class Index9(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=1, stop=2, stride=1)
        return index_output_1


class Index10(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=256, stride=1)
        return index_output_1


class Index11(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=768, stop=1024, stride=1)
        return index_output_1


class Index12(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=1536, stop=1792, stride=1)
        return index_output_1


class Index13(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=2304, stop=2560, stride=1)
        return index_output_1


class Index14(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=512, stop=768, stride=1)
        return index_output_1


class Index15(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=1280, stop=1536, stride=1)
        return index_output_1


class Index16(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=2048, stop=2304, stride=1)
        return index_output_1


class Index17(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=2816, stop=3072, stride=1)
        return index_output_1


class Index18(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=256, stop=512, stride=1)
        return index_output_1


class Index19(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=1024, stop=1280, stride=1)
        return index_output_1


class Index20(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=1792, stop=2048, stride=1)
        return index_output_1


class Index21(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=2560, stop=2816, stride=1)
        return index_output_1


class Index22(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=0, stop=32, stride=1)
        return index_output_1


class Index23(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=32, stop=64, stride=1)
        return index_output_1


class Index24(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=1, stop=32, stride=2)
        return index_output_1


class Index25(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=0, stop=32, stride=2)
        return index_output_1


class Index26(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=-2, stride=1)
        return index_output_1


class Index27(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=-2, stop=-1, stride=1)
        return index_output_1


class Index28(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=72, stop=73, stride=1)
        return index_output_1


class Index29(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=16, stop=32, stride=1)
        return index_output_1


class Index30(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=0, stop=16, stride=1)
        return index_output_1


class Index31(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=128, stop=256, stride=1)
        return index_output_1


class Index32(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=768, stride=1)
        return index_output_1


class Index33(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=768, stop=1536, stride=1)
        return index_output_1


class Index34(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=1536, stop=2304, stride=1)
        return index_output_1


class Index35(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=0, stop=768, stride=1)
        return index_output_1


class Index36(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=768, stop=1536, stride=1)
        return index_output_1


class Index37(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=1536, stop=2304, stride=1)
        return index_output_1


class Index38(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=0, stop=256, stride=1)
        return index_output_1


class Index39(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=64, stop=128, stride=1)
        return index_output_1


class Index40(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=0, stop=64, stride=1)
        return index_output_1


class Index41(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=32, stop=80, stride=1)
        return index_output_1


class Index42(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=10, stop=11, stride=1)
        return index_output_1


class Index43(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=2, stop=258, stride=1)
        return index_output_1


class Index44(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=-1, stop=72, stride=1)
        return index_output_1


class Index45(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=-24, stop=96, stride=1)
        return index_output_1


class Index46(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=0, stop=72, stride=1)
        return index_output_1


class Index47(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=0, stop=4, stride=1)
        return index_output_1


class Index48(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=4, stop=8, stride=1)
        return index_output_1


class Index49(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=12, stop=24, stride=1)
        return index_output_1


class Index50(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=0, stop=12, stride=1)
        return index_output_1


class Index51(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=36, stop=48, stride=1)
        return index_output_1


class Index52(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=24, stop=36, stride=1)
        return index_output_1


class Index53(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=1, stride=1)
        return index_output_1


class Index54(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=1, stop=2, stride=1)
        return index_output_1


class Index55(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=2, stop=3, stride=1)
        return index_output_1


class Index56(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=64, stride=1)
        return index_output_1


class Index57(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=64, stop=160, stride=1)
        return index_output_1


class Index58(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=160, stop=176, stride=1)
        return index_output_1


class Index59(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=88, stride=1)
        return index_output_1


class Index60(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=88, stop=132, stride=1)
        return index_output_1


class Index61(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=132, stop=176, stride=1)
        return index_output_1


class Index62(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=44, stride=1)
        return index_output_1


class Index63(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=44, stop=88, stride=1)
        return index_output_1


class Index64(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=88, stop=176, stride=1)
        return index_output_1


class Index65(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=128, stride=1)
        return index_output_1


class Index66(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=128, stop=256, stride=1)
        return index_output_1


class Index67(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=256, stop=288, stride=1)
        return index_output_1


class Index68(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=192, stride=1)
        return index_output_1


class Index69(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=192, stop=288, stride=1)
        return index_output_1


class Index70(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=288, stop=304, stride=1)
        return index_output_1


class Index71(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=160, stride=1)
        return index_output_1


class Index72(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=160, stop=272, stride=1)
        return index_output_1


class Index73(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=272, stop=296, stride=1)
        return index_output_1


class Index74(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=256, stop=280, stride=1)
        return index_output_1


class Index75(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=112, stride=1)
        return index_output_1


class Index76(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=112, stop=256, stride=1)
        return index_output_1


class Index77(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=256, stride=1)
        return index_output_1


class Index78(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=256, stop=416, stride=1)
        return index_output_1


class Index79(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=416, stop=448, stride=1)
        return index_output_1


class Index80(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=64, stop=192, stride=1)
        return index_output_1


class Index81(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=192, stop=448, stride=1)
        return index_output_1


class Index82(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=256, stop=384, stride=1)
        return index_output_1


class Index83(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=384, stop=448, stride=1)
        return index_output_1


class Index84(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=384, stride=1)
        return index_output_1


class Index85(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=384, stop=576, stride=1)
        return index_output_1


class Index86(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=576, stop=624, stride=1)
        return index_output_1


class Index87(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=18, stride=1)
        return index_output_1


class Index88(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=18, stop=54, stride=1)
        return index_output_1


class Index89(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=54, stop=126, stride=1)
        return index_output_1


class Index90(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=72, stride=1)
        return index_output_1


class Index91(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=72, stop=108, stride=1)
        return index_output_1


class Index92(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=108, stop=126, stride=1)
        return index_output_1


class Index93(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=36, stride=1)
        return index_output_1


class Index94(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=36, stop=54, stride=1)
        return index_output_1


class Index95(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=54, stop=72, stride=1)
        return index_output_1


class Index96(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=18, stop=36, stride=1)
        return index_output_1


class Index97(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=36, stop=72, stride=1)
        return index_output_1


class Index98(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=128, stop=192, stride=1)
        return index_output_1


class Index99(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=192, stop=256, stride=1)
        return index_output_1


class Index100(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=64, stop=128, stride=1)
        return index_output_1


class Index101(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=30, stride=1)
        return index_output_1


class Index102(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=30, stop=90, stride=1)
        return index_output_1


class Index103(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=90, stop=210, stride=1)
        return index_output_1


class Index104(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=120, stride=1)
        return index_output_1


class Index105(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=120, stop=180, stride=1)
        return index_output_1


class Index106(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=180, stop=210, stride=1)
        return index_output_1


class Index107(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=60, stride=1)
        return index_output_1


class Index108(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=60, stop=90, stride=1)
        return index_output_1


class Index109(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=90, stop=120, stride=1)
        return index_output_1


class Index110(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=30, stop=60, stride=1)
        return index_output_1


class Index111(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=60, stop=120, stride=1)
        return index_output_1


class Index112(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=32, stride=1)
        return index_output_1


class Index113(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=32, stop=96, stride=1)
        return index_output_1


class Index114(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=96, stop=224, stride=1)
        return index_output_1


class Index115(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=192, stop=224, stride=1)
        return index_output_1


class Index116(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=64, stop=96, stride=1)
        return index_output_1


class Index117(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=96, stop=128, stride=1)
        return index_output_1


class Index118(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=32, stop=64, stride=1)
        return index_output_1


class Index119(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=40, stride=1)
        return index_output_1


class Index120(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=40, stop=120, stride=1)
        return index_output_1


class Index121(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=120, stop=280, stride=1)
        return index_output_1


class Index122(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=160, stop=240, stride=1)
        return index_output_1


class Index123(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=240, stop=280, stride=1)
        return index_output_1


class Index124(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=80, stride=1)
        return index_output_1


class Index125(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=80, stop=120, stride=1)
        return index_output_1


class Index126(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=120, stop=160, stride=1)
        return index_output_1


class Index127(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=40, stop=80, stride=1)
        return index_output_1


class Index128(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=80, stop=160, stride=1)
        return index_output_1


class Index129(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=48, stride=1)
        return index_output_1


class Index130(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=48, stop=144, stride=1)
        return index_output_1


class Index131(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=144, stop=336, stride=1)
        return index_output_1


class Index132(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=288, stop=336, stride=1)
        return index_output_1


class Index133(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=96, stride=1)
        return index_output_1


class Index134(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=96, stop=144, stride=1)
        return index_output_1


class Index135(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=144, stop=192, stride=1)
        return index_output_1


class Index136(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=48, stop=96, stride=1)
        return index_output_1


class Index137(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=96, stop=192, stride=1)
        return index_output_1


class Index138(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=44, stop=132, stride=1)
        return index_output_1


class Index139(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=132, stop=308, stride=1)
        return index_output_1


class Index140(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=176, stride=1)
        return index_output_1


class Index141(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=176, stop=264, stride=1)
        return index_output_1


class Index142(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=264, stop=308, stride=1)
        return index_output_1


class Index143(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=16, stride=1)
        return index_output_1


class Index144(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=16, stop=32, stride=1)
        return index_output_1


class Index145(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=96, stop=112, stride=1)
        return index_output_1


class Index146(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=96, stop=160, stride=1)
        return index_output_1


class Index147(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=160, stop=224, stride=1)
        return index_output_1


class Index148(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=576, stop=768, stride=1)
        return index_output_1


class Index149(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=256, stop=640, stride=1)
        return index_output_1


class Index150(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=640, stop=1024, stride=1)
        return index_output_1


class Index151(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=256, stop=512, stride=1)
        return index_output_1


class Index152(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=512, stop=768, stride=1)
        return index_output_1


class Index153(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=768, stop=1024, stride=1)
        return index_output_1


class Index154(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=1024, stop=1280, stride=1)
        return index_output_1


class Index155(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=1280, stop=1536, stride=1)
        return index_output_1


class Index156(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=1536, stop=1792, stride=1)
        return index_output_1


class Index157(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=4, stop=64, stride=1)
        return index_output_1


class Index158(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=4, stride=1)
        return index_output_1


class Index159(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=4, stop=64, stride=1)
        return index_output_1


class Index160(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=4, stride=1)
        return index_output_1


class Index161(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=60, stop=64, stride=1)
        return index_output_1


class Index162(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=60, stop=64, stride=1)
        return index_output_1


class Index163(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=60, stride=1)
        return index_output_1


class Index164(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=64, stride=2)
        return index_output_1


class Index165(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=1, stop=64, stride=2)
        return index_output_1


class Index166(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=64, stride=2)
        return index_output_1


class Index167(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=1, stop=64, stride=2)
        return index_output_1


class Index168(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=4, stop=32, stride=1)
        return index_output_1


class Index169(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=4, stop=32, stride=1)
        return index_output_1


class Index170(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=28, stop=32, stride=1)
        return index_output_1


class Index171(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=28, stride=1)
        return index_output_1


class Index172(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=28, stop=32, stride=1)
        return index_output_1


class Index173(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=28, stride=1)
        return index_output_1


class Index174(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=32, stride=2)
        return index_output_1


class Index175(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=1, stop=32, stride=2)
        return index_output_1


class Index176(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=32, stride=2)
        return index_output_1


class Index177(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=1, stop=32, stride=2)
        return index_output_1


class Index178(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=4, stop=16, stride=1)
        return index_output_1


class Index179(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=4, stop=16, stride=1)
        return index_output_1


class Index180(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=12, stop=16, stride=1)
        return index_output_1


class Index181(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=12, stride=1)
        return index_output_1


class Index182(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=12, stop=16, stride=1)
        return index_output_1


class Index183(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=12, stride=1)
        return index_output_1


class Index184(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=16, stride=2)
        return index_output_1


class Index185(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=1, stop=16, stride=2)
        return index_output_1


class Index186(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=16, stride=2)
        return index_output_1


class Index187(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=1, stop=16, stride=2)
        return index_output_1


class Index188(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=0, stop=2, stride=1)
        return index_output_1


class Index189(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=2, stop=4, stride=1)
        return index_output_1


class Index190(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=640, stride=2)
        return index_output_1


class Index191(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=1, stop=640, stride=2)
        return index_output_1


class Index192(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=0, stop=640, stride=2)
        return index_output_1


class Index193(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=1, stop=640, stride=2)
        return index_output_1


class Index194(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=416, stride=2)
        return index_output_1


class Index195(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=1, stop=416, stride=2)
        return index_output_1


class Index196(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=0, stop=416, stride=2)
        return index_output_1


class Index197(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=1, stop=416, stride=2)
        return index_output_1


def ids_func(param):
    forge_module, shapes_dtypes, _ = param
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (
        Index0,
        [((2, 4, 1), torch.float32)],
        {"model_name": ["pt_musicgen_large", "pt_musicgen_small", "pt_musicgen_medium"]},
    ),
    (
        Index1,
        [((2, 4, 1), torch.float32)],
        {"model_name": ["pt_musicgen_large", "pt_musicgen_small", "pt_musicgen_medium"]},
    ),
    (
        Index2,
        [((2, 4, 1), torch.float32)],
        {"model_name": ["pt_musicgen_large", "pt_musicgen_small", "pt_musicgen_medium"]},
    ),
    (
        Index3,
        [((2, 4, 1), torch.float32)],
        {"model_name": ["pt_musicgen_large", "pt_musicgen_small", "pt_musicgen_medium"]},
    ),
    (Index0, [((2048, 2048), torch.float32)], {"model_name": ["pt_musicgen_large"]}),
    (Index0, [((2048, 1024), torch.float32)], {"model_name": ["pt_musicgen_small"]}),
    (Index0, [((2048, 1536), torch.float32)], {"model_name": ["pt_musicgen_medium"]}),
    (Index4, [((1, 77), torch.float32)], {"model_name": ["pt_clip_vit_base_patch32_text"]}),
    (Index5, [((1, 204, 768), torch.float32)], {"model_name": ["pt_ViLt_maskedlm"]}),
    (Index0, [((1, 201, 768), torch.float32)], {"model_name": ["pt_ViLt_question_answering"]}),
    (
        Index6,
        [((1, 512), torch.float32)],
        {
            "model_name": [
                "pt_albert_base_v2_masked_lm",
                "pt_albert_base_v2_masked_lm",
                "pt_albert_xlarge_v1_token_cls",
                "pt_albert_xlarge_v1_token_cls",
                "pt_albert_large_v1_masked_lm",
                "pt_albert_large_v1_masked_lm",
                "pt_albert_xxlarge_v1_token_cls",
                "pt_albert_xxlarge_v1_token_cls",
                "pt_albert_large_v2_token_cls",
                "pt_albert_large_v2_token_cls",
                "pt_albert_large_v2_masked_lm",
                "pt_albert_large_v2_masked_lm",
                "pt_albert_xlarge_v2_token_cls",
                "pt_albert_xlarge_v2_token_cls",
                "pt_albert_base_v1_masked_lm",
                "pt_albert_base_v1_masked_lm",
                "pt_albert_xxlarge_v2_masked_lm",
                "pt_albert_xxlarge_v2_masked_lm",
                "pt_albert_xlarge_v1_masked_lm",
                "pt_albert_xlarge_v1_masked_lm",
                "pt_albert_xxlarge_v1_masked_lm",
                "pt_albert_xxlarge_v1_masked_lm",
                "pt_albert_large_v1_token_cls",
                "pt_albert_large_v1_token_cls",
                "pt_albert_base_v1_token_cls",
                "pt_albert_base_v1_token_cls",
                "pt_albert_xlarge_v2_masked_lm",
                "pt_albert_xlarge_v2_masked_lm",
                "pt_albert_xxlarge_v2_token_cls",
                "pt_albert_xxlarge_v2_token_cls",
                "pt_albert_base_v2_token_cls",
                "pt_albert_base_v2_token_cls",
                "pt_bert_masked_lm",
                "pt_bert_masked_lm",
                "pt_bert_sequence_classification",
                "pt_bert_sequence_classification",
                "pt_distilbert_masked_lm",
                "pt_distilbert_sequence_classification",
                "pt_distilbert_token_classification",
                "pt_dpr_ctx_encoder_single_nq_base",
                "pt_dpr_reader_single_nq_base",
                "pt_dpr_reader_single_nq_base",
                "pt_dpr_reader_multiset_base",
                "pt_dpr_reader_multiset_base",
                "pt_dpr_question_encoder_single_nq_base",
                "pt_dpr_ctx_encoder_multiset_base",
                "pt_dpr_question_encoder_multiset_base",
                "pt_squeezebert",
            ]
        },
    ),
    (
        Index7,
        [((1, 512), torch.float32)],
        {"model_name": ["pt_bert_qa", "pt_bert_qa", "pt_distilbert_question_answering"]},
    ),
    (Index0, [((2, 1024), torch.float32)], {"model_name": ["pt_bert_qa"]}),
    (Index1, [((2, 1024), torch.float32)], {"model_name": ["pt_bert_qa"]}),
    (
        Index8,
        [((2,), torch.float32)],
        {
            "model_name": [
                "pt_bert_qa",
                "pt_distilbert_question_answering",
                "pt_dpr_reader_single_nq_base",
                "pt_dpr_reader_multiset_base",
                "pt_opt_1_3b_qa",
                "pt_opt_350m_qa",
                "pt_opt_125m_qa",
            ]
        },
    ),
    (
        Index9,
        [((2,), torch.float32)],
        {
            "model_name": [
                "pt_bert_qa",
                "pt_distilbert_question_answering",
                "pt_dpr_reader_single_nq_base",
                "pt_dpr_reader_multiset_base",
                "pt_opt_1_3b_qa",
                "pt_opt_350m_qa",
                "pt_opt_125m_qa",
            ]
        },
    ),
    (Index10, [((3072, 1024), torch.float32)], {"model_name": ["pt_codegen_350M_mono"]}),
    (Index11, [((3072, 1024), torch.float32)], {"model_name": ["pt_codegen_350M_mono"]}),
    (Index12, [((3072, 1024), torch.float32)], {"model_name": ["pt_codegen_350M_mono"]}),
    (Index13, [((3072, 1024), torch.float32)], {"model_name": ["pt_codegen_350M_mono"]}),
    (Index14, [((3072, 1024), torch.float32)], {"model_name": ["pt_codegen_350M_mono"]}),
    (Index15, [((3072, 1024), torch.float32)], {"model_name": ["pt_codegen_350M_mono"]}),
    (Index16, [((3072, 1024), torch.float32)], {"model_name": ["pt_codegen_350M_mono"]}),
    (Index17, [((3072, 1024), torch.float32)], {"model_name": ["pt_codegen_350M_mono"]}),
    (Index18, [((3072, 1024), torch.float32)], {"model_name": ["pt_codegen_350M_mono"]}),
    (Index19, [((3072, 1024), torch.float32)], {"model_name": ["pt_codegen_350M_mono"]}),
    (Index20, [((3072, 1024), torch.float32)], {"model_name": ["pt_codegen_350M_mono"]}),
    (Index21, [((3072, 1024), torch.float32)], {"model_name": ["pt_codegen_350M_mono"]}),
    (Index22, [((1, 256, 16, 64), torch.float32)], {"model_name": ["pt_codegen_350M_mono"]}),
    (Index23, [((1, 256, 16, 64), torch.float32)], {"model_name": ["pt_codegen_350M_mono"]}),
    (Index24, [((1, 256, 16, 32), torch.float32)], {"model_name": ["pt_codegen_350M_mono"]}),
    (Index25, [((1, 256, 16, 32), torch.float32)], {"model_name": ["pt_codegen_350M_mono"]}),
    (
        Index0,
        [((1, 128, 768), torch.float32)],
        {
            "model_name": [
                "pt_distilbert_sequence_classification",
                "pt_dpr_ctx_encoder_single_nq_base",
                "pt_dpr_reader_single_nq_base",
                "pt_dpr_reader_multiset_base",
                "pt_dpr_question_encoder_single_nq_base",
                "pt_dpr_ctx_encoder_multiset_base",
                "pt_dpr_question_encoder_multiset_base",
                "pt_roberta_sentiment",
                "pt_squeezebert",
            ]
        },
    ),
    (
        Index0,
        [((2, 768), torch.float32)],
        {
            "model_name": [
                "pt_distilbert_question_answering",
                "pt_dpr_reader_single_nq_base",
                "pt_dpr_reader_multiset_base",
                "pt_opt_125m_qa",
            ]
        },
    ),
    (
        Index1,
        [((2, 768), torch.float32)],
        {
            "model_name": [
                "pt_distilbert_question_answering",
                "pt_dpr_reader_single_nq_base",
                "pt_dpr_reader_multiset_base",
                "pt_opt_125m_qa",
            ]
        },
    ),
    (Index26, [((1, 6, 73, 64), torch.float32)], {"model_name": ["pt_falcon"]}),
    (Index27, [((1, 6, 73, 64), torch.float32)], {"model_name": ["pt_falcon"]}),
    (Index28, [((1, 6, 73, 64), torch.float32)], {"model_name": ["pt_falcon"]}),
    (Index23, [((1, 71, 6, 64), torch.float32)], {"model_name": ["pt_falcon"]}),
    (Index22, [((1, 71, 6, 64), torch.float32)], {"model_name": ["pt_falcon"]}),
    (Index23, [((1, 1, 6, 64), torch.float32)], {"model_name": ["pt_falcon"]}),
    (Index22, [((1, 1, 6, 64), torch.float32)], {"model_name": ["pt_falcon"]}),
    (Index0, [((1, 334, 64, 3, 64), torch.float32)], {"model_name": ["pt_fuyu_8b"]}),
    (Index1, [((1, 334, 64, 3, 64), torch.float32)], {"model_name": ["pt_fuyu_8b"]}),
    (Index2, [((1, 334, 64, 3, 64), torch.float32)], {"model_name": ["pt_fuyu_8b"]}),
    (Index22, [((1, 64, 334, 64), torch.float32)], {"model_name": ["pt_fuyu_8b"]}),
    (Index23, [((1, 64, 334, 64), torch.float32)], {"model_name": ["pt_fuyu_8b"]}),
    (Index29, [((1, 64, 334, 32), torch.float32)], {"model_name": ["pt_fuyu_8b"]}),
    (Index30, [((1, 64, 334, 32), torch.float32)], {"model_name": ["pt_fuyu_8b"]}),
    (Index31, [((1, 8, 7, 256), torch.float32)], {"model_name": ["pt_gemma_2b"]}),
    (Index6, [((1, 8, 7, 256), torch.float32)], {"model_name": ["pt_gemma_2b"]}),
    (Index31, [((1, 1, 7, 256), torch.float32)], {"model_name": ["pt_gemma_2b"]}),
    (Index6, [((1, 1, 7, 256), torch.float32)], {"model_name": ["pt_gemma_2b"]}),
    (Index32, [((2304, 768), torch.float32)], {"model_name": ["pt_gpt2_generation"]}),
    (Index33, [((2304, 768), torch.float32)], {"model_name": ["pt_gpt2_generation"]}),
    (Index34, [((2304, 768), torch.float32)], {"model_name": ["pt_gpt2_generation"]}),
    (Index35, [((2304,), torch.float32)], {"model_name": ["pt_gpt2_generation"]}),
    (Index36, [((2304,), torch.float32)], {"model_name": ["pt_gpt2_generation"]}),
    (Index37, [((2304,), torch.float32)], {"model_name": ["pt_gpt2_generation"]}),
    (Index10, [((1, 1, 1024, 1024), torch.float32)], {"model_name": ["pt_gpt2_generation"]}),
    (Index38, [((1, 1, 256, 1024), torch.float32)], {"model_name": ["pt_gpt2_generation"]}),
    (
        Index10,
        [((1, 1, 2048, 2048), torch.float32)],
        {
            "model_name": [
                "pt_gpt_neo_2_7B_causal_lm",
                "pt_gpt_neo_2_7B_causal_lm",
                "pt_gpt_neo_1_3B_causal_lm",
                "pt_gpt_neo_1_3B_causal_lm",
                "pt_gpt_neo_125M_causal_lm",
                "pt_gpt_neo_125M_causal_lm",
            ]
        },
    ),
    (
        Index38,
        [((1, 1, 256, 2048), torch.float32)],
        {"model_name": ["pt_gpt_neo_2_7B_causal_lm", "pt_gpt_neo_1_3B_causal_lm", "pt_gpt_neo_125M_causal_lm"]},
    ),
    (
        Index39,
        [((1, 32, 256, 128), torch.float32)],
        {
            "model_name": [
                "pt_Llama_3_1_8B_Instruct_causal_lm",
                "pt_Meta_Llama_3_8B_causal_lm",
                "pt_Meta_Llama_3_8B_Instruct_causal_lm",
                "pt_Llama_3_1_8B_causal_lm",
            ]
        },
    ),
    (
        Index40,
        [((1, 32, 256, 128), torch.float32)],
        {
            "model_name": [
                "pt_Llama_3_1_8B_Instruct_causal_lm",
                "pt_Meta_Llama_3_8B_causal_lm",
                "pt_Meta_Llama_3_8B_Instruct_causal_lm",
                "pt_Llama_3_1_8B_causal_lm",
            ]
        },
    ),
    (
        Index39,
        [((1, 8, 256, 128), torch.float32)],
        {
            "model_name": [
                "pt_Llama_3_1_8B_Instruct_causal_lm",
                "pt_Meta_Llama_3_8B_causal_lm",
                "pt_Meta_Llama_3_8B_Instruct_causal_lm",
                "pt_Llama_3_1_8B_causal_lm",
            ]
        },
    ),
    (
        Index40,
        [((1, 8, 256, 128), torch.float32)],
        {
            "model_name": [
                "pt_Llama_3_1_8B_Instruct_causal_lm",
                "pt_Meta_Llama_3_8B_causal_lm",
                "pt_Meta_Llama_3_8B_Instruct_causal_lm",
                "pt_Llama_3_1_8B_causal_lm",
            ]
        },
    ),
    (
        Index23,
        [((1, 32, 256, 64), torch.float32)],
        {"model_name": ["pt_Llama_3_2_1B_Instruct_causal_lm", "pt_Llama_3_2_1B_causal_lm"]},
    ),
    (
        Index22,
        [((1, 32, 256, 64), torch.float32)],
        {"model_name": ["pt_Llama_3_2_1B_Instruct_causal_lm", "pt_Llama_3_2_1B_causal_lm"]},
    ),
    (
        Index23,
        [((1, 8, 256, 64), torch.float32)],
        {"model_name": ["pt_Llama_3_2_1B_Instruct_causal_lm", "pt_Llama_3_2_1B_causal_lm"]},
    ),
    (
        Index22,
        [((1, 8, 256, 64), torch.float32)],
        {"model_name": ["pt_Llama_3_2_1B_Instruct_causal_lm", "pt_Llama_3_2_1B_causal_lm"]},
    ),
    (
        Index39,
        [((1, 32, 4, 128), torch.float32)],
        {
            "model_name": [
                "pt_Llama_3_1_8B_Instruct_seq_cls",
                "pt_Meta_Llama_3_8B_seq_cls",
                "pt_Llama_3_1_8B_seq_cls",
                "pt_Meta_Llama_3_8B_Instruct_seq_cls",
            ]
        },
    ),
    (
        Index40,
        [((1, 32, 4, 128), torch.float32)],
        {
            "model_name": [
                "pt_Llama_3_1_8B_Instruct_seq_cls",
                "pt_Meta_Llama_3_8B_seq_cls",
                "pt_Llama_3_1_8B_seq_cls",
                "pt_Meta_Llama_3_8B_Instruct_seq_cls",
            ]
        },
    ),
    (
        Index39,
        [((1, 8, 4, 128), torch.float32)],
        {
            "model_name": [
                "pt_Llama_3_1_8B_Instruct_seq_cls",
                "pt_Meta_Llama_3_8B_seq_cls",
                "pt_Llama_3_1_8B_seq_cls",
                "pt_Meta_Llama_3_8B_Instruct_seq_cls",
            ]
        },
    ),
    (
        Index40,
        [((1, 8, 4, 128), torch.float32)],
        {
            "model_name": [
                "pt_Llama_3_1_8B_Instruct_seq_cls",
                "pt_Meta_Llama_3_8B_seq_cls",
                "pt_Llama_3_1_8B_seq_cls",
                "pt_Meta_Llama_3_8B_Instruct_seq_cls",
            ]
        },
    ),
    (
        Index3,
        [((1, 4, 2), torch.float32)],
        {
            "model_name": [
                "pt_Llama_3_1_8B_Instruct_seq_cls",
                "pt_Meta_Llama_3_8B_seq_cls",
                "pt_Llama_3_2_1B_Instruct_seq_cls",
                "pt_Llama_3_2_1B_seq_cls",
                "pt_Llama_3_1_8B_seq_cls",
                "pt_Meta_Llama_3_8B_Instruct_seq_cls",
            ]
        },
    ),
    (
        Index23,
        [((1, 32, 4, 64), torch.float32)],
        {"model_name": ["pt_Llama_3_2_1B_Instruct_seq_cls", "pt_Llama_3_2_1B_seq_cls"]},
    ),
    (
        Index22,
        [((1, 32, 4, 64), torch.float32)],
        {"model_name": ["pt_Llama_3_2_1B_Instruct_seq_cls", "pt_Llama_3_2_1B_seq_cls"]},
    ),
    (
        Index23,
        [((1, 8, 4, 64), torch.float32)],
        {"model_name": ["pt_Llama_3_2_1B_Instruct_seq_cls", "pt_Llama_3_2_1B_seq_cls"]},
    ),
    (
        Index22,
        [((1, 8, 4, 64), torch.float32)],
        {"model_name": ["pt_Llama_3_2_1B_Instruct_seq_cls", "pt_Llama_3_2_1B_seq_cls"]},
    ),
    (Index39, [((1, 32, 128, 128), torch.float32)], {"model_name": ["pt_Mistral_7B_v0_1"]}),
    (Index40, [((1, 32, 128, 128), torch.float32)], {"model_name": ["pt_Mistral_7B_v0_1"]}),
    (Index39, [((1, 8, 128, 128), torch.float32)], {"model_name": ["pt_Mistral_7B_v0_1"]}),
    (Index40, [((1, 8, 128, 128), torch.float32)], {"model_name": ["pt_Mistral_7B_v0_1"]}),
    (Index0, [((2, 2048), torch.float32)], {"model_name": ["pt_opt_1_3b_qa"]}),
    (Index1, [((2, 2048), torch.float32)], {"model_name": ["pt_opt_1_3b_qa"]}),
    (Index0, [((2, 512), torch.float32)], {"model_name": ["pt_opt_350m_qa"]}),
    (Index1, [((2, 512), torch.float32)], {"model_name": ["pt_opt_350m_qa"]}),
    (Index22, [((1, 32, 12, 80), torch.float32)], {"model_name": ["pt_phi_2_pytdml_token_cls", "pt_phi_2_token_cls"]}),
    (Index41, [((1, 32, 12, 80), torch.float32)], {"model_name": ["pt_phi_2_pytdml_token_cls", "pt_phi_2_token_cls"]}),
    (Index29, [((1, 32, 12, 32), torch.float32)], {"model_name": ["pt_phi_2_pytdml_token_cls", "pt_phi_2_token_cls"]}),
    (Index30, [((1, 32, 12, 32), torch.float32)], {"model_name": ["pt_phi_2_pytdml_token_cls", "pt_phi_2_token_cls"]}),
    (Index22, [((1, 32, 256, 80), torch.float32)], {"model_name": ["pt_phi_2_causal_lm", "pt_phi_2_pytdml_causal_lm"]}),
    (Index41, [((1, 32, 256, 80), torch.float32)], {"model_name": ["pt_phi_2_causal_lm", "pt_phi_2_pytdml_causal_lm"]}),
    (Index29, [((1, 32, 256, 32), torch.float32)], {"model_name": ["pt_phi_2_causal_lm", "pt_phi_2_pytdml_causal_lm"]}),
    (Index30, [((1, 32, 256, 32), torch.float32)], {"model_name": ["pt_phi_2_causal_lm", "pt_phi_2_pytdml_causal_lm"]}),
    (Index22, [((1, 32, 11, 80), torch.float32)], {"model_name": ["pt_phi_2_seq_cls", "pt_phi_2_pytdml_seq_cls"]}),
    (Index41, [((1, 32, 11, 80), torch.float32)], {"model_name": ["pt_phi_2_seq_cls", "pt_phi_2_pytdml_seq_cls"]}),
    (Index29, [((1, 32, 11, 32), torch.float32)], {"model_name": ["pt_phi_2_seq_cls", "pt_phi_2_pytdml_seq_cls"]}),
    (Index30, [((1, 32, 11, 32), torch.float32)], {"model_name": ["pt_phi_2_seq_cls", "pt_phi_2_pytdml_seq_cls"]}),
    (Index42, [((1, 11, 2), torch.float32)], {"model_name": ["pt_phi_2_seq_cls", "pt_phi_2_pytdml_seq_cls"]}),
    (Index23, [((1, 16, 29, 64), torch.float32)], {"model_name": ["pt_qwen_chat"]}),
    (Index22, [((1, 16, 29, 64), torch.float32)], {"model_name": ["pt_qwen_chat"]}),
    (Index23, [((1, 16, 6, 64), torch.float32)], {"model_name": ["pt_qwen_causal_lm"]}),
    (Index22, [((1, 16, 6, 64), torch.float32)], {"model_name": ["pt_qwen_causal_lm"]}),
    (
        Index39,
        [((1, 12, 35, 128), torch.float32)],
        {"model_name": ["pt_Qwen_Qwen2_5_Coder_1_5B_Instruct", "pt_Qwen_Qwen2_5_Coder_1_5B"]},
    ),
    (
        Index40,
        [((1, 12, 35, 128), torch.float32)],
        {"model_name": ["pt_Qwen_Qwen2_5_Coder_1_5B_Instruct", "pt_Qwen_Qwen2_5_Coder_1_5B"]},
    ),
    (
        Index39,
        [((1, 2, 35, 128), torch.float32)],
        {
            "model_name": [
                "pt_Qwen_Qwen2_5_Coder_1_5B_Instruct",
                "pt_Qwen_Qwen2_5_Coder_3B",
                "pt_Qwen_Qwen2_5_Coder_1_5B",
                "pt_Qwen_Qwen2_5_Coder_3B_Instruct",
            ]
        },
    ),
    (
        Index40,
        [((1, 2, 35, 128), torch.float32)],
        {
            "model_name": [
                "pt_Qwen_Qwen2_5_Coder_1_5B_Instruct",
                "pt_Qwen_Qwen2_5_Coder_3B",
                "pt_Qwen_Qwen2_5_Coder_1_5B",
                "pt_Qwen_Qwen2_5_Coder_3B_Instruct",
            ]
        },
    ),
    (
        Index39,
        [((1, 16, 35, 128), torch.float32)],
        {"model_name": ["pt_Qwen_Qwen2_5_Coder_3B", "pt_Qwen_Qwen2_5_Coder_3B_Instruct"]},
    ),
    (
        Index40,
        [((1, 16, 35, 128), torch.float32)],
        {"model_name": ["pt_Qwen_Qwen2_5_Coder_3B", "pt_Qwen_Qwen2_5_Coder_3B_Instruct"]},
    ),
    (
        Index39,
        [((1, 28, 35, 128), torch.float32)],
        {"model_name": ["pt_Qwen_Qwen2_5_Coder_7B", "pt_Qwen_Qwen2_5_Coder_7B_Instruct"]},
    ),
    (
        Index40,
        [((1, 28, 35, 128), torch.float32)],
        {"model_name": ["pt_Qwen_Qwen2_5_Coder_7B", "pt_Qwen_Qwen2_5_Coder_7B_Instruct"]},
    ),
    (
        Index39,
        [((1, 4, 35, 128), torch.float32)],
        {"model_name": ["pt_Qwen_Qwen2_5_Coder_7B", "pt_Qwen_Qwen2_5_Coder_7B_Instruct"]},
    ),
    (
        Index40,
        [((1, 4, 35, 128), torch.float32)],
        {"model_name": ["pt_Qwen_Qwen2_5_Coder_7B", "pt_Qwen_Qwen2_5_Coder_7B_Instruct"]},
    ),
    (Index23, [((1, 14, 35, 64), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_Coder_0_5B"]}),
    (Index22, [((1, 14, 35, 64), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_Coder_0_5B"]}),
    (Index23, [((1, 2, 35, 64), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_Coder_0_5B"]}),
    (Index22, [((1, 2, 35, 64), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_Coder_0_5B"]}),
    (Index39, [((1, 12, 39, 128), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_1_5B_Instruct"]}),
    (Index40, [((1, 12, 39, 128), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_1_5B_Instruct"]}),
    (
        Index39,
        [((1, 2, 39, 128), torch.float32)],
        {"model_name": ["pt_Qwen_Qwen2_5_1_5B_Instruct", "pt_Qwen_Qwen2_5_3B_Instruct"]},
    ),
    (
        Index40,
        [((1, 2, 39, 128), torch.float32)],
        {"model_name": ["pt_Qwen_Qwen2_5_1_5B_Instruct", "pt_Qwen_Qwen2_5_3B_Instruct"]},
    ),
    (Index39, [((1, 12, 29, 128), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_1_5B"]}),
    (Index40, [((1, 12, 29, 128), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_1_5B"]}),
    (Index39, [((1, 2, 29, 128), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_1_5B", "pt_Qwen_Qwen2_5_3B"]}),
    (Index40, [((1, 2, 29, 128), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_1_5B", "pt_Qwen_Qwen2_5_3B"]}),
    (Index39, [((1, 28, 29, 128), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_7B"]}),
    (Index40, [((1, 28, 29, 128), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_7B"]}),
    (Index39, [((1, 4, 29, 128), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_7B"]}),
    (Index40, [((1, 4, 29, 128), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_7B"]}),
    (Index23, [((1, 14, 39, 64), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_0_5B_Instruct"]}),
    (Index22, [((1, 14, 39, 64), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_0_5B_Instruct"]}),
    (Index23, [((1, 2, 39, 64), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_0_5B_Instruct"]}),
    (Index22, [((1, 2, 39, 64), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_0_5B_Instruct"]}),
    (Index23, [((1, 14, 29, 64), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_0_5B"]}),
    (Index22, [((1, 14, 29, 64), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_0_5B"]}),
    (Index23, [((1, 2, 29, 64), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_0_5B"]}),
    (Index22, [((1, 2, 29, 64), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_0_5B"]}),
    (Index39, [((1, 16, 39, 128), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_3B_Instruct"]}),
    (Index40, [((1, 16, 39, 128), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_3B_Instruct"]}),
    (Index39, [((1, 28, 39, 128), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_7B_Instruct"]}),
    (Index40, [((1, 28, 39, 128), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_7B_Instruct"]}),
    (Index39, [((1, 4, 39, 128), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_7B_Instruct"]}),
    (Index40, [((1, 4, 39, 128), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_7B_Instruct"]}),
    (Index39, [((1, 16, 29, 128), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_3B"]}),
    (Index40, [((1, 16, 29, 128), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_3B"]}),
    (Index6, [((1, 514), torch.float32)], {"model_name": ["pt_roberta_masked_lm", "pt_roberta_sentiment"]}),
    (Index43, [((2050, 2048), torch.float32)], {"model_name": ["pt_xglm_1_7B"]}),
    (Index43, [((2050, 1024), torch.float32)], {"model_name": ["pt_xglm_564M"]}),
    (
        Index44,
        [((1, 1, 1024, 72), torch.float32)],
        {"model_name": ["nbeats_generic", "nbeats_trend", "nbeats_seasonality"]},
    ),
    (Index45, [((1024, 96), torch.float32)], {"model_name": ["nbeats_generic"]}),
    (Index46, [((1024, 96), torch.float32)], {"model_name": ["nbeats_generic"]}),
    (Index47, [((1024, 8), torch.float32)], {"model_name": ["nbeats_trend"]}),
    (Index48, [((1024, 8), torch.float32)], {"model_name": ["nbeats_trend"]}),
    (Index49, [((1024, 48), torch.float32)], {"model_name": ["nbeats_seasonality"]}),
    (Index50, [((1024, 48), torch.float32)], {"model_name": ["nbeats_seasonality"]}),
    (Index51, [((1024, 48), torch.float32)], {"model_name": ["nbeats_seasonality"]}),
    (Index52, [((1024, 48), torch.float32)], {"model_name": ["nbeats_seasonality"]}),
    (Index0, [((1, 197, 384), torch.float32)], {"model_name": ["pt_deit_small_patch16_224"]}),
    (
        Index0,
        [((1, 197, 768), torch.float32)],
        {"model_name": ["pt_deit_base_patch16_224", "pt_deit_base_distilled_patch16_224", "pt_vit_base_patch16_224"]},
    ),
    (Index0, [((1, 197, 192), torch.float32)], {"model_name": ["pt_deit_tiny_patch16_224"]}),
    (Index53, [((1, 3, 224, 224), torch.float32)], {"model_name": ["pt_googlenet"]}),
    (Index54, [((1, 3, 224, 224), torch.float32)], {"model_name": ["pt_googlenet"]}),
    (Index55, [((1, 3, 224, 224), torch.float32)], {"model_name": ["pt_googlenet"]}),
    (Index56, [((1, 176, 28, 28), torch.float32)], {"model_name": ["pt_googlenet"]}),
    (Index57, [((1, 176, 28, 28), torch.float32)], {"model_name": ["pt_googlenet"]}),
    (Index58, [((1, 176, 28, 28), torch.float32)], {"model_name": ["pt_googlenet"]}),
    (
        Index59,
        [((1, 176, 28, 28), torch.float32)],
        {"model_name": ["pt_hrnet_timm_hrnet_w44", "pt_hrnet_osmr_hrnetv2_w44"]},
    ),
    (
        Index60,
        [((1, 176, 28, 28), torch.float32)],
        {"model_name": ["pt_hrnet_timm_hrnet_w44", "pt_hrnet_osmr_hrnetv2_w44"]},
    ),
    (
        Index61,
        [((1, 176, 28, 28), torch.float32)],
        {"model_name": ["pt_hrnet_timm_hrnet_w44", "pt_hrnet_osmr_hrnetv2_w44"]},
    ),
    (
        Index62,
        [((1, 176, 28, 28), torch.float32)],
        {"model_name": ["pt_hrnet_timm_hrnet_w44", "pt_hrnet_osmr_hrnetv2_w44"]},
    ),
    (
        Index63,
        [((1, 176, 28, 28), torch.float32)],
        {"model_name": ["pt_hrnet_timm_hrnet_w44", "pt_hrnet_osmr_hrnetv2_w44"]},
    ),
    (
        Index64,
        [((1, 176, 28, 28), torch.float32)],
        {"model_name": ["pt_hrnet_timm_hrnet_w44", "pt_hrnet_osmr_hrnetv2_w44"]},
    ),
    (Index65, [((1, 288, 28, 28), torch.float32)], {"model_name": ["pt_googlenet"]}),
    (Index66, [((1, 288, 28, 28), torch.float32)], {"model_name": ["pt_googlenet"]}),
    (Index67, [((1, 288, 28, 28), torch.float32)], {"model_name": ["pt_googlenet"]}),
    (Index68, [((1, 304, 14, 14), torch.float32)], {"model_name": ["pt_googlenet"]}),
    (Index69, [((1, 304, 14, 14), torch.float32)], {"model_name": ["pt_googlenet"]}),
    (Index70, [((1, 304, 14, 14), torch.float32)], {"model_name": ["pt_googlenet"]}),
    (Index71, [((1, 296, 14, 14), torch.float32)], {"model_name": ["pt_googlenet"]}),
    (Index72, [((1, 296, 14, 14), torch.float32)], {"model_name": ["pt_googlenet"]}),
    (Index73, [((1, 296, 14, 14), torch.float32)], {"model_name": ["pt_googlenet"]}),
    (Index65, [((1, 280, 14, 14), torch.float32)], {"model_name": ["pt_googlenet"]}),
    (Index66, [((1, 280, 14, 14), torch.float32)], {"model_name": ["pt_googlenet"]}),
    (Index74, [((1, 280, 14, 14), torch.float32)], {"model_name": ["pt_googlenet"]}),
    (Index75, [((1, 288, 14, 14), torch.float32)], {"model_name": ["pt_googlenet"]}),
    (Index76, [((1, 288, 14, 14), torch.float32)], {"model_name": ["pt_googlenet"]}),
    (Index67, [((1, 288, 14, 14), torch.float32)], {"model_name": ["pt_googlenet"]}),
    (Index77, [((1, 448, 14, 14), torch.float32)], {"model_name": ["pt_googlenet"]}),
    (Index78, [((1, 448, 14, 14), torch.float32)], {"model_name": ["pt_googlenet"]}),
    (Index79, [((1, 448, 14, 14), torch.float32)], {"model_name": ["pt_googlenet"]}),
    (
        Index77,
        [((1, 448, 7, 7), torch.float32)],
        {"model_name": ["pt_googlenet", "pt_hrnet_osmr_hrnetv2_w64", "pt_hrnet_timm_hrnet_w64"]},
    ),
    (Index78, [((1, 448, 7, 7), torch.float32)], {"model_name": ["pt_googlenet"]}),
    (Index79, [((1, 448, 7, 7), torch.float32)], {"model_name": ["pt_googlenet"]}),
    (
        Index56,
        [((1, 448, 7, 7), torch.float32)],
        {"model_name": ["pt_hrnet_osmr_hrnetv2_w64", "pt_hrnet_timm_hrnet_w64"]},
    ),
    (
        Index80,
        [((1, 448, 7, 7), torch.float32)],
        {"model_name": ["pt_hrnet_osmr_hrnetv2_w64", "pt_hrnet_timm_hrnet_w64"]},
    ),
    (
        Index81,
        [((1, 448, 7, 7), torch.float32)],
        {"model_name": ["pt_hrnet_osmr_hrnetv2_w64", "pt_hrnet_timm_hrnet_w64"]},
    ),
    (
        Index82,
        [((1, 448, 7, 7), torch.float32)],
        {"model_name": ["pt_hrnet_osmr_hrnetv2_w64", "pt_hrnet_timm_hrnet_w64"]},
    ),
    (
        Index83,
        [((1, 448, 7, 7), torch.float32)],
        {"model_name": ["pt_hrnet_osmr_hrnetv2_w64", "pt_hrnet_timm_hrnet_w64"]},
    ),
    (Index84, [((1, 624, 7, 7), torch.float32)], {"model_name": ["pt_googlenet"]}),
    (Index85, [((1, 624, 7, 7), torch.float32)], {"model_name": ["pt_googlenet"]}),
    (Index86, [((1, 624, 7, 7), torch.float32)], {"model_name": ["pt_googlenet"]}),
    (
        Index87,
        [((1, 126, 7, 7), torch.float32)],
        {
            "model_name": [
                "pt_hrnet_timm_hrnet_w18",
                "pt_hrnet_osmr_hrnet_w18_small_v2",
                "pt_hrnet_osmr_hrnetv2_w18",
                "pt_hrnet_timm_hrnet_w18_small_v2",
            ]
        },
    ),
    (
        Index88,
        [((1, 126, 7, 7), torch.float32)],
        {
            "model_name": [
                "pt_hrnet_timm_hrnet_w18",
                "pt_hrnet_osmr_hrnet_w18_small_v2",
                "pt_hrnet_osmr_hrnetv2_w18",
                "pt_hrnet_timm_hrnet_w18_small_v2",
            ]
        },
    ),
    (
        Index89,
        [((1, 126, 7, 7), torch.float32)],
        {
            "model_name": [
                "pt_hrnet_timm_hrnet_w18",
                "pt_hrnet_osmr_hrnet_w18_small_v2",
                "pt_hrnet_osmr_hrnetv2_w18",
                "pt_hrnet_timm_hrnet_w18_small_v2",
            ]
        },
    ),
    (
        Index90,
        [((1, 126, 7, 7), torch.float32)],
        {
            "model_name": [
                "pt_hrnet_timm_hrnet_w18",
                "pt_hrnet_osmr_hrnet_w18_small_v2",
                "pt_hrnet_osmr_hrnetv2_w18",
                "pt_hrnet_timm_hrnet_w18_small_v2",
            ]
        },
    ),
    (
        Index91,
        [((1, 126, 7, 7), torch.float32)],
        {
            "model_name": [
                "pt_hrnet_timm_hrnet_w18",
                "pt_hrnet_osmr_hrnet_w18_small_v2",
                "pt_hrnet_osmr_hrnetv2_w18",
                "pt_hrnet_timm_hrnet_w18_small_v2",
            ]
        },
    ),
    (
        Index92,
        [((1, 126, 7, 7), torch.float32)],
        {
            "model_name": [
                "pt_hrnet_timm_hrnet_w18",
                "pt_hrnet_osmr_hrnet_w18_small_v2",
                "pt_hrnet_osmr_hrnetv2_w18",
                "pt_hrnet_timm_hrnet_w18_small_v2",
            ]
        },
    ),
    (
        Index93,
        [((1, 72, 28, 28), torch.float32)],
        {
            "model_name": [
                "pt_hrnet_timm_hrnet_w18",
                "pt_hrnet_osmr_hrnet_w18_small_v2",
                "pt_hrnet_osmr_hrnetv2_w18",
                "pt_hrnet_timm_hrnet_w18_small_v2",
            ]
        },
    ),
    (
        Index94,
        [((1, 72, 28, 28), torch.float32)],
        {
            "model_name": [
                "pt_hrnet_timm_hrnet_w18",
                "pt_hrnet_osmr_hrnet_w18_small_v2",
                "pt_hrnet_osmr_hrnetv2_w18",
                "pt_hrnet_timm_hrnet_w18_small_v2",
            ]
        },
    ),
    (
        Index95,
        [((1, 72, 28, 28), torch.float32)],
        {
            "model_name": [
                "pt_hrnet_timm_hrnet_w18",
                "pt_hrnet_osmr_hrnet_w18_small_v2",
                "pt_hrnet_osmr_hrnetv2_w18",
                "pt_hrnet_timm_hrnet_w18_small_v2",
            ]
        },
    ),
    (
        Index87,
        [((1, 72, 28, 28), torch.float32)],
        {
            "model_name": [
                "pt_hrnet_timm_hrnet_w18",
                "pt_hrnet_osmr_hrnet_w18_small_v2",
                "pt_hrnet_osmr_hrnetv2_w18",
                "pt_hrnet_timm_hrnet_w18_small_v2",
            ]
        },
    ),
    (
        Index96,
        [((1, 72, 28, 28), torch.float32)],
        {
            "model_name": [
                "pt_hrnet_timm_hrnet_w18",
                "pt_hrnet_osmr_hrnet_w18_small_v2",
                "pt_hrnet_osmr_hrnetv2_w18",
                "pt_hrnet_timm_hrnet_w18_small_v2",
            ]
        },
    ),
    (
        Index97,
        [((1, 72, 28, 28), torch.float32)],
        {
            "model_name": [
                "pt_hrnet_timm_hrnet_w18",
                "pt_hrnet_osmr_hrnet_w18_small_v2",
                "pt_hrnet_osmr_hrnetv2_w18",
                "pt_hrnet_timm_hrnet_w18_small_v2",
            ]
        },
    ),
    (
        Index65,
        [((1, 256, 28, 28), torch.float32)],
        {"model_name": ["pt_hrnet_osmr_hrnetv2_w64", "pt_hrnet_timm_hrnet_w64"]},
    ),
    (
        Index98,
        [((1, 256, 28, 28), torch.float32)],
        {"model_name": ["pt_hrnet_osmr_hrnetv2_w64", "pt_hrnet_timm_hrnet_w64"]},
    ),
    (
        Index99,
        [((1, 256, 28, 28), torch.float32)],
        {"model_name": ["pt_hrnet_osmr_hrnetv2_w64", "pt_hrnet_timm_hrnet_w64"]},
    ),
    (
        Index56,
        [((1, 256, 28, 28), torch.float32)],
        {"model_name": ["pt_hrnet_osmr_hrnetv2_w64", "pt_hrnet_timm_hrnet_w64"]},
    ),
    (
        Index100,
        [((1, 256, 28, 28), torch.float32)],
        {"model_name": ["pt_hrnet_osmr_hrnetv2_w64", "pt_hrnet_timm_hrnet_w64"]},
    ),
    (
        Index66,
        [((1, 256, 28, 28), torch.float32)],
        {"model_name": ["pt_hrnet_osmr_hrnetv2_w64", "pt_hrnet_timm_hrnet_w64"]},
    ),
    (
        Index101,
        [((1, 210, 7, 7), torch.float32)],
        {"model_name": ["pt_hrnet_timm_hrnet_w30", "pt_hrnet_osmr_hrnetv2_w30"]},
    ),
    (
        Index102,
        [((1, 210, 7, 7), torch.float32)],
        {"model_name": ["pt_hrnet_timm_hrnet_w30", "pt_hrnet_osmr_hrnetv2_w30"]},
    ),
    (
        Index103,
        [((1, 210, 7, 7), torch.float32)],
        {"model_name": ["pt_hrnet_timm_hrnet_w30", "pt_hrnet_osmr_hrnetv2_w30"]},
    ),
    (
        Index104,
        [((1, 210, 7, 7), torch.float32)],
        {"model_name": ["pt_hrnet_timm_hrnet_w30", "pt_hrnet_osmr_hrnetv2_w30"]},
    ),
    (
        Index105,
        [((1, 210, 7, 7), torch.float32)],
        {"model_name": ["pt_hrnet_timm_hrnet_w30", "pt_hrnet_osmr_hrnetv2_w30"]},
    ),
    (
        Index106,
        [((1, 210, 7, 7), torch.float32)],
        {"model_name": ["pt_hrnet_timm_hrnet_w30", "pt_hrnet_osmr_hrnetv2_w30"]},
    ),
    (
        Index107,
        [((1, 120, 28, 28), torch.float32)],
        {"model_name": ["pt_hrnet_timm_hrnet_w30", "pt_hrnet_osmr_hrnetv2_w30"]},
    ),
    (
        Index108,
        [((1, 120, 28, 28), torch.float32)],
        {"model_name": ["pt_hrnet_timm_hrnet_w30", "pt_hrnet_osmr_hrnetv2_w30"]},
    ),
    (
        Index109,
        [((1, 120, 28, 28), torch.float32)],
        {"model_name": ["pt_hrnet_timm_hrnet_w30", "pt_hrnet_osmr_hrnetv2_w30"]},
    ),
    (
        Index101,
        [((1, 120, 28, 28), torch.float32)],
        {"model_name": ["pt_hrnet_timm_hrnet_w30", "pt_hrnet_osmr_hrnetv2_w30"]},
    ),
    (
        Index110,
        [((1, 120, 28, 28), torch.float32)],
        {"model_name": ["pt_hrnet_timm_hrnet_w30", "pt_hrnet_osmr_hrnetv2_w30"]},
    ),
    (
        Index111,
        [((1, 120, 28, 28), torch.float32)],
        {"model_name": ["pt_hrnet_timm_hrnet_w30", "pt_hrnet_osmr_hrnetv2_w30"]},
    ),
    (
        Index112,
        [((1, 224, 7, 7), torch.float32)],
        {"model_name": ["pt_hrnet_timm_hrnet_w32", "pt_hrnet_osmr_hrnetv2_w32"]},
    ),
    (
        Index113,
        [((1, 224, 7, 7), torch.float32)],
        {"model_name": ["pt_hrnet_timm_hrnet_w32", "pt_hrnet_osmr_hrnetv2_w32"]},
    ),
    (
        Index114,
        [((1, 224, 7, 7), torch.float32)],
        {"model_name": ["pt_hrnet_timm_hrnet_w32", "pt_hrnet_osmr_hrnetv2_w32"]},
    ),
    (
        Index65,
        [((1, 224, 7, 7), torch.float32)],
        {"model_name": ["pt_hrnet_timm_hrnet_w32", "pt_hrnet_osmr_hrnetv2_w32"]},
    ),
    (
        Index98,
        [((1, 224, 7, 7), torch.float32)],
        {"model_name": ["pt_hrnet_timm_hrnet_w32", "pt_hrnet_osmr_hrnetv2_w32"]},
    ),
    (
        Index115,
        [((1, 224, 7, 7), torch.float32)],
        {"model_name": ["pt_hrnet_timm_hrnet_w32", "pt_hrnet_osmr_hrnetv2_w32"]},
    ),
    (
        Index56,
        [((1, 128, 28, 28), torch.float32)],
        {"model_name": ["pt_hrnet_timm_hrnet_w32", "pt_hrnet_osmr_hrnetv2_w32"]},
    ),
    (
        Index116,
        [((1, 128, 28, 28), torch.float32)],
        {"model_name": ["pt_hrnet_timm_hrnet_w32", "pt_hrnet_osmr_hrnetv2_w32"]},
    ),
    (
        Index117,
        [((1, 128, 28, 28), torch.float32)],
        {"model_name": ["pt_hrnet_timm_hrnet_w32", "pt_hrnet_osmr_hrnetv2_w32"]},
    ),
    (
        Index112,
        [((1, 128, 28, 28), torch.float32)],
        {"model_name": ["pt_hrnet_timm_hrnet_w32", "pt_hrnet_osmr_hrnetv2_w32"]},
    ),
    (
        Index118,
        [((1, 128, 28, 28), torch.float32)],
        {"model_name": ["pt_hrnet_timm_hrnet_w32", "pt_hrnet_osmr_hrnetv2_w32"]},
    ),
    (
        Index100,
        [((1, 128, 28, 28), torch.float32)],
        {"model_name": ["pt_hrnet_timm_hrnet_w32", "pt_hrnet_osmr_hrnetv2_w32"]},
    ),
    (
        Index119,
        [((1, 280, 7, 7), torch.float32)],
        {"model_name": ["pt_hrnet_osmr_hrnetv2_w40", "pt_hrnet_timm_hrnet_w40"]},
    ),
    (
        Index120,
        [((1, 280, 7, 7), torch.float32)],
        {"model_name": ["pt_hrnet_osmr_hrnetv2_w40", "pt_hrnet_timm_hrnet_w40"]},
    ),
    (
        Index121,
        [((1, 280, 7, 7), torch.float32)],
        {"model_name": ["pt_hrnet_osmr_hrnetv2_w40", "pt_hrnet_timm_hrnet_w40"]},
    ),
    (
        Index71,
        [((1, 280, 7, 7), torch.float32)],
        {"model_name": ["pt_hrnet_osmr_hrnetv2_w40", "pt_hrnet_timm_hrnet_w40"]},
    ),
    (
        Index122,
        [((1, 280, 7, 7), torch.float32)],
        {"model_name": ["pt_hrnet_osmr_hrnetv2_w40", "pt_hrnet_timm_hrnet_w40"]},
    ),
    (
        Index123,
        [((1, 280, 7, 7), torch.float32)],
        {"model_name": ["pt_hrnet_osmr_hrnetv2_w40", "pt_hrnet_timm_hrnet_w40"]},
    ),
    (
        Index124,
        [((1, 160, 28, 28), torch.float32)],
        {"model_name": ["pt_hrnet_osmr_hrnetv2_w40", "pt_hrnet_timm_hrnet_w40"]},
    ),
    (
        Index125,
        [((1, 160, 28, 28), torch.float32)],
        {"model_name": ["pt_hrnet_osmr_hrnetv2_w40", "pt_hrnet_timm_hrnet_w40"]},
    ),
    (
        Index126,
        [((1, 160, 28, 28), torch.float32)],
        {"model_name": ["pt_hrnet_osmr_hrnetv2_w40", "pt_hrnet_timm_hrnet_w40"]},
    ),
    (
        Index119,
        [((1, 160, 28, 28), torch.float32)],
        {"model_name": ["pt_hrnet_osmr_hrnetv2_w40", "pt_hrnet_timm_hrnet_w40"]},
    ),
    (
        Index127,
        [((1, 160, 28, 28), torch.float32)],
        {"model_name": ["pt_hrnet_osmr_hrnetv2_w40", "pt_hrnet_timm_hrnet_w40"]},
    ),
    (
        Index128,
        [((1, 160, 28, 28), torch.float32)],
        {"model_name": ["pt_hrnet_osmr_hrnetv2_w40", "pt_hrnet_timm_hrnet_w40"]},
    ),
    (
        Index129,
        [((1, 336, 7, 7), torch.float32)],
        {"model_name": ["pt_hrnet_timm_hrnet_w48", "pt_hrnet_osmr_hrnetv2_w48"]},
    ),
    (
        Index130,
        [((1, 336, 7, 7), torch.float32)],
        {"model_name": ["pt_hrnet_timm_hrnet_w48", "pt_hrnet_osmr_hrnetv2_w48"]},
    ),
    (
        Index131,
        [((1, 336, 7, 7), torch.float32)],
        {"model_name": ["pt_hrnet_timm_hrnet_w48", "pt_hrnet_osmr_hrnetv2_w48"]},
    ),
    (
        Index68,
        [((1, 336, 7, 7), torch.float32)],
        {"model_name": ["pt_hrnet_timm_hrnet_w48", "pt_hrnet_osmr_hrnetv2_w48"]},
    ),
    (
        Index69,
        [((1, 336, 7, 7), torch.float32)],
        {"model_name": ["pt_hrnet_timm_hrnet_w48", "pt_hrnet_osmr_hrnetv2_w48"]},
    ),
    (
        Index132,
        [((1, 336, 7, 7), torch.float32)],
        {"model_name": ["pt_hrnet_timm_hrnet_w48", "pt_hrnet_osmr_hrnetv2_w48"]},
    ),
    (
        Index133,
        [((1, 192, 28, 28), torch.float32)],
        {"model_name": ["pt_hrnet_timm_hrnet_w48", "pt_hrnet_osmr_hrnetv2_w48"]},
    ),
    (
        Index134,
        [((1, 192, 28, 28), torch.float32)],
        {"model_name": ["pt_hrnet_timm_hrnet_w48", "pt_hrnet_osmr_hrnetv2_w48"]},
    ),
    (
        Index135,
        [((1, 192, 28, 28), torch.float32)],
        {"model_name": ["pt_hrnet_timm_hrnet_w48", "pt_hrnet_osmr_hrnetv2_w48"]},
    ),
    (
        Index129,
        [((1, 192, 28, 28), torch.float32)],
        {"model_name": ["pt_hrnet_timm_hrnet_w48", "pt_hrnet_osmr_hrnetv2_w48"]},
    ),
    (
        Index136,
        [((1, 192, 28, 28), torch.float32)],
        {"model_name": ["pt_hrnet_timm_hrnet_w48", "pt_hrnet_osmr_hrnetv2_w48"]},
    ),
    (
        Index137,
        [((1, 192, 28, 28), torch.float32)],
        {"model_name": ["pt_hrnet_timm_hrnet_w48", "pt_hrnet_osmr_hrnetv2_w48"]},
    ),
    (
        Index62,
        [((1, 308, 7, 7), torch.float32)],
        {"model_name": ["pt_hrnet_timm_hrnet_w44", "pt_hrnet_osmr_hrnetv2_w44"]},
    ),
    (
        Index138,
        [((1, 308, 7, 7), torch.float32)],
        {"model_name": ["pt_hrnet_timm_hrnet_w44", "pt_hrnet_osmr_hrnetv2_w44"]},
    ),
    (
        Index139,
        [((1, 308, 7, 7), torch.float32)],
        {"model_name": ["pt_hrnet_timm_hrnet_w44", "pt_hrnet_osmr_hrnetv2_w44"]},
    ),
    (
        Index140,
        [((1, 308, 7, 7), torch.float32)],
        {"model_name": ["pt_hrnet_timm_hrnet_w44", "pt_hrnet_osmr_hrnetv2_w44"]},
    ),
    (
        Index141,
        [((1, 308, 7, 7), torch.float32)],
        {"model_name": ["pt_hrnet_timm_hrnet_w44", "pt_hrnet_osmr_hrnetv2_w44"]},
    ),
    (
        Index142,
        [((1, 308, 7, 7), torch.float32)],
        {"model_name": ["pt_hrnet_timm_hrnet_w44", "pt_hrnet_osmr_hrnetv2_w44"]},
    ),
    (
        Index143,
        [((1, 64, 28, 28), torch.float32)],
        {"model_name": ["pt_hrnet_timm_hrnet_w18_small", "pt_hrnet_osmr_hrnet_w18_small_v1"]},
    ),
    (
        Index144,
        [((1, 64, 28, 28), torch.float32)],
        {"model_name": ["pt_hrnet_timm_hrnet_w18_small", "pt_hrnet_osmr_hrnet_w18_small_v1"]},
    ),
    (
        Index118,
        [((1, 64, 28, 28), torch.float32)],
        {"model_name": ["pt_hrnet_timm_hrnet_w18_small", "pt_hrnet_osmr_hrnet_w18_small_v1"]},
    ),
    (
        Index56,
        [((1, 112, 7, 7), torch.float32)],
        {"model_name": ["pt_hrnet_timm_hrnet_w18_small", "pt_hrnet_osmr_hrnet_w18_small_v1"]},
    ),
    (
        Index116,
        [((1, 112, 7, 7), torch.float32)],
        {"model_name": ["pt_hrnet_timm_hrnet_w18_small", "pt_hrnet_osmr_hrnet_w18_small_v1"]},
    ),
    (
        Index145,
        [((1, 112, 7, 7), torch.float32)],
        {"model_name": ["pt_hrnet_timm_hrnet_w18_small", "pt_hrnet_osmr_hrnet_w18_small_v1"]},
    ),
    (Index133, [((1, 224, 35, 35), torch.float32)], {"model_name": ["pt_timm_inception_v4", "pt_osmr_inception_v4"]}),
    (Index146, [((1, 224, 35, 35), torch.float32)], {"model_name": ["pt_timm_inception_v4", "pt_osmr_inception_v4"]}),
    (Index147, [((1, 224, 35, 35), torch.float32)], {"model_name": ["pt_timm_inception_v4", "pt_osmr_inception_v4"]}),
    (Index84, [((1, 768, 17, 17), torch.float32)], {"model_name": ["pt_timm_inception_v4", "pt_osmr_inception_v4"]}),
    (Index85, [((1, 768, 17, 17), torch.float32)], {"model_name": ["pt_timm_inception_v4", "pt_osmr_inception_v4"]}),
    (Index148, [((1, 768, 17, 17), torch.float32)], {"model_name": ["pt_timm_inception_v4", "pt_osmr_inception_v4"]}),
    (Index77, [((1, 1024, 8, 8), torch.float32)], {"model_name": ["pt_timm_inception_v4", "pt_osmr_inception_v4"]}),
    (Index149, [((1, 1024, 8, 8), torch.float32)], {"model_name": ["pt_timm_inception_v4", "pt_osmr_inception_v4"]}),
    (Index150, [((1, 1024, 8, 8), torch.float32)], {"model_name": ["pt_timm_inception_v4", "pt_osmr_inception_v4"]}),
    (Index77, [((1, 1792, 56, 56), torch.float32)], {"model_name": ["pt_monodle"]}),
    (Index151, [((1, 1792, 56, 56), torch.float32)], {"model_name": ["pt_monodle"]}),
    (Index152, [((1, 1792, 56, 56), torch.float32)], {"model_name": ["pt_monodle"]}),
    (Index153, [((1, 1792, 56, 56), torch.float32)], {"model_name": ["pt_monodle"]}),
    (Index154, [((1, 1792, 56, 56), torch.float32)], {"model_name": ["pt_monodle"]}),
    (Index155, [((1, 1792, 56, 56), torch.float32)], {"model_name": ["pt_monodle"]}),
    (Index156, [((1, 1792, 56, 56), torch.float32)], {"model_name": ["pt_monodle"]}),
    (Index157, [((1, 64, 64, 96), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Index158, [((1, 64, 64, 96), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Index159, [((1, 64, 64, 96), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Index160, [((1, 64, 64, 96), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Index161, [((1, 64, 64, 96), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Index107, [((1, 64, 64, 96), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Index162, [((1, 64, 64, 96), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Index163, [((1, 64, 64, 96), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Index164, [((1, 64, 64, 96), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Index165, [((1, 64, 64, 96), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Index166, [((1, 32, 64, 96), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Index167, [((1, 32, 64, 96), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Index168, [((1, 32, 32, 192), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Index158, [((1, 32, 32, 192), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Index169, [((1, 32, 32, 192), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Index160, [((1, 32, 32, 192), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Index170, [((1, 32, 32, 192), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Index171, [((1, 32, 32, 192), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Index172, [((1, 32, 32, 192), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Index173, [((1, 32, 32, 192), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Index174, [((1, 32, 32, 192), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Index175, [((1, 32, 32, 192), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Index176, [((1, 16, 32, 192), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Index177, [((1, 16, 32, 192), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Index178, [((1, 16, 16, 384), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Index158, [((1, 16, 16, 384), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Index179, [((1, 16, 16, 384), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Index160, [((1, 16, 16, 384), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Index180, [((1, 16, 16, 384), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Index181, [((1, 16, 16, 384), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Index182, [((1, 16, 16, 384), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Index183, [((1, 16, 16, 384), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Index184, [((1, 16, 16, 384), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Index185, [((1, 16, 16, 384), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Index186, [((1, 8, 16, 384), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Index187, [((1, 8, 16, 384), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Index0, [((1, 197, 1024), torch.float32)], {"model_name": ["pt_vit_large_patch16_224"]}),
    (
        Index188,
        [((1, 5880, 4), torch.float32)],
        {"model_name": ["pt_yolov6m", "pt_yolov6n", "pt_yolov6l", "pt_yolov6s"]},
    ),
    (
        Index189,
        [((1, 5880, 4), torch.float32)],
        {"model_name": ["pt_yolov6m", "pt_yolov6n", "pt_yolov6l", "pt_yolov6s"]},
    ),
    (
        Index190,
        [((1, 3, 640, 640), torch.float32)],
        {"model_name": ["pt_yolox_m", "pt_yolox_s", "pt_yolox_x", "pt_yolox_l"]},
    ),
    (
        Index191,
        [((1, 3, 640, 640), torch.float32)],
        {"model_name": ["pt_yolox_m", "pt_yolox_s", "pt_yolox_x", "pt_yolox_l"]},
    ),
    (
        Index192,
        [((1, 3, 320, 640), torch.float32)],
        {"model_name": ["pt_yolox_m", "pt_yolox_s", "pt_yolox_x", "pt_yolox_l"]},
    ),
    (
        Index193,
        [((1, 3, 320, 640), torch.float32)],
        {"model_name": ["pt_yolox_m", "pt_yolox_s", "pt_yolox_x", "pt_yolox_l"]},
    ),
    (Index194, [((1, 3, 416, 416), torch.float32)], {"model_name": ["pt_yolox_nano", "pt_yolox_tiny"]}),
    (Index195, [((1, 3, 416, 416), torch.float32)], {"model_name": ["pt_yolox_nano", "pt_yolox_tiny"]}),
    (Index196, [((1, 3, 208, 416), torch.float32)], {"model_name": ["pt_yolox_nano", "pt_yolox_tiny"]}),
    (Index197, [((1, 3, 208, 416), torch.float32)], {"model_name": ["pt_yolox_nano", "pt_yolox_tiny"]}),
]


@pytest.mark.push
@pytest.mark.parametrize("forge_module_and_shapes_dtypes", forge_modules_and_shapes_dtypes_list, ids=ids_func)
def test_module(forge_module_and_shapes_dtypes, record_property):
    record_property("frontend", "tt-forge-fe")

    forge_module, operand_shapes_dtypes, metadata = forge_module_and_shapes_dtypes

    for metadata_name, metadata_value in metadata.items():
        record_property(metadata_name, metadata_value)

    inputs = [
        Tensor.create_from_shape(operand_shape, operand_dtype) for operand_shape, operand_dtype in operand_shapes_dtypes
    ]

    framework_model = forge_module(forge_module.__name__)
    framework_model.process_framework_parameters()

    compiled_model = compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)
