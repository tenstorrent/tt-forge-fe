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
        index_output_1 = forge.op.Index("", index_input_0, dim=-5, start=0, stop=1, stride=1)
        return index_output_1


class Index4(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-5, start=1, stop=2, stride=1)
        return index_output_1


class Index5(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-5, start=2, stop=3, stride=1)
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
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=0, stop=6, stride=1)
        return index_output_1


class Index8(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=0, stop=384, stride=1)
        return index_output_1


class Index9(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=0, stop=14, stride=1)
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
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=1, stop=2, stride=1)
        return index_output_1


class Index27(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=0, stop=1, stride=1)
        return index_output_1


class Index28(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=1, stop=2, stride=1)
        return index_output_1


class Index29(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=32, stride=1)
        return index_output_1


class Index30(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=31, stop=32, stride=1)
        return index_output_1


class Index31(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=32, stop=80, stride=1)
        return index_output_1


class Index32(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=16, stop=32, stride=1)
        return index_output_1


class Index33(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=0, stop=16, stride=1)
        return index_output_1


class Index34(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=2, stop=3, stride=1)
        return index_output_1


class Index35(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=0, stop=7, stride=1)
        return index_output_1


class Index36(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=768, stride=1)
        return index_output_1


class Index37(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=768, stop=1536, stride=1)
        return index_output_1


class Index38(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=1536, stop=2304, stride=1)
        return index_output_1


class Index39(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=0, stop=768, stride=1)
        return index_output_1


class Index40(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=768, stop=1536, stride=1)
        return index_output_1


class Index41(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=1536, stop=2304, stride=1)
        return index_output_1


class Index42(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=7, stride=1)
        return index_output_1


class Index43(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=3, stop=4, stride=1)
        return index_output_1


class Index44(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=16, stride=1)
        return index_output_1


class Index45(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=16, stop=32, stride=1)
        return index_output_1


class Index46(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=32, stride=1)
        return index_output_1


class Index47(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=32, stop=64, stride=1)
        return index_output_1


class Index48(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=64, stride=1)
        return index_output_1


class Index49(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=64, stop=128, stride=1)
        return index_output_1


class Index50(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=128, stride=1)
        return index_output_1


class Index51(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=128, stop=256, stride=1)
        return index_output_1


class Index52(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=64, stride=1)
        return index_output_1


class Index53(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=64, stop=144, stride=1)
        return index_output_1


class Index54(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=2, stride=1)
        return index_output_1


class Index55(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=2, stop=4, stride=1)
        return index_output_1


class Index56(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=2, stop=3, stride=1)
        return index_output_1


class Index57(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=3, stop=4, stride=1)
        return index_output_1


class Index58(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=4, stop=5, stride=1)
        return index_output_1


class Index59(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=5, stop=6, stride=1)
        return index_output_1


class Index60(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=6, stop=7, stride=1)
        return index_output_1


class Index61(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=7, stop=8, stride=1)
        return index_output_1


class Index62(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=8, stop=9, stride=1)
        return index_output_1


class Index63(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=9, stop=10, stride=1)
        return index_output_1


class Index64(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=10, stop=11, stride=1)
        return index_output_1


class Index65(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=11, stop=12, stride=1)
        return index_output_1


class Index66(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=12, stop=13, stride=1)
        return index_output_1


class Index67(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=13, stop=14, stride=1)
        return index_output_1


class Index68(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=14, stop=15, stride=1)
        return index_output_1


class Index69(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=15, stop=16, stride=1)
        return index_output_1


class Index70(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=16, stop=17, stride=1)
        return index_output_1


class Index71(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=17, stop=18, stride=1)
        return index_output_1


class Index72(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=18, stop=19, stride=1)
        return index_output_1


class Index73(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=19, stop=20, stride=1)
        return index_output_1


class Index74(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=20, stop=21, stride=1)
        return index_output_1


class Index75(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=21, stop=22, stride=1)
        return index_output_1


class Index76(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=22, stop=23, stride=1)
        return index_output_1


class Index77(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=23, stop=24, stride=1)
        return index_output_1


class Index78(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=24, stop=25, stride=1)
        return index_output_1


class Index79(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=144, stop=192, stride=1)
        return index_output_1


class Index80(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=48, stop=96, stride=1)
        return index_output_1


class Index81(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=0, stop=48, stride=1)
        return index_output_1


class Index82(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=96, stop=144, stride=1)
        return index_output_1


class Index83(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=4, stop=64, stride=1)
        return index_output_1


class Index84(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=4, stride=1)
        return index_output_1


class Index85(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=4, stop=64, stride=1)
        return index_output_1


class Index86(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=4, stride=1)
        return index_output_1


class Index87(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=-4, stop=64, stride=1)
        return index_output_1


class Index88(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=-4, stride=1)
        return index_output_1


class Index89(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=-4, stop=64, stride=1)
        return index_output_1


class Index90(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=-4, stride=1)
        return index_output_1


class Index91(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=64, stride=2)
        return index_output_1


class Index92(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=1, stop=64, stride=2)
        return index_output_1


class Index93(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=64, stride=2)
        return index_output_1


class Index94(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=1, stop=64, stride=2)
        return index_output_1


class Index95(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=4, stop=32, stride=1)
        return index_output_1


class Index96(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=4, stop=32, stride=1)
        return index_output_1


class Index97(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=-4, stop=32, stride=1)
        return index_output_1


class Index98(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=-4, stop=32, stride=1)
        return index_output_1


class Index99(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=32, stride=2)
        return index_output_1


class Index100(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=1, stop=32, stride=2)
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
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=-4, stop=16, stride=1)
        return index_output_1


class Index106(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=-4, stop=16, stride=1)
        return index_output_1


class Index107(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=16, stride=2)
        return index_output_1


class Index108(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=1, stop=16, stride=2)
        return index_output_1


class Index109(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=16, stride=2)
        return index_output_1


class Index110(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=1, stop=16, stride=2)
        return index_output_1


class Index111(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=64, stop=128, stride=1)
        return index_output_1


class Index112(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=0, stop=64, stride=1)
        return index_output_1


class Index113(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=0, stop=256, stride=1)
        return index_output_1


class Index114(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=2048, stride=1)
        return index_output_1


class Index115(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=2048, stop=4096, stride=1)
        return index_output_1


class Index116(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=64, stop=80, stride=1)
        return index_output_1


class Index117(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=80, stop=96, stride=1)
        return index_output_1


class Index118(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=4, stop=5, stride=1)
        return index_output_1


class Index119(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=5, stop=6, stride=1)
        return index_output_1


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (
        Index0,
        [((1, 6, 768), torch.float32)],
        {
            "model_names": [
                "onnx_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf",
                "pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "1", "stride": "1"},
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
        Index3,
        [((3, 1, 8, 12, 15), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index4,
        [((3, 1, 8, 12, 15), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index5,
        [((3, 1, 8, 12, 15), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "2", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index6,
        [((1, 512), torch.int64)],
        {
            "model_names": [
                "pt_albert_base_v1_token_cls_hf",
                "pt_albert_base_v2_mlm_hf",
                "pt_albert_xlarge_v2_token_cls_hf",
                "pt_albert_xxlarge_v1_mlm_hf",
                "pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf",
                "pt_distilbert_distilbert_base_uncased_mlm_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_albert_base_v2_token_cls_hf",
                "pt_albert_large_v1_mlm_hf",
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder",
                "pt_albert_large_v2_token_cls_hf",
                "pt_albert_xlarge_v1_mlm_hf",
                "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
                "pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index7,
        [((1, 512), torch.int64)],
        {
            "model_names": ["pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "6", "stride": "1"},
        },
    ),
    (
        Index8,
        [((1, 512), torch.int64)],
        {
            "model_names": [
                "pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf",
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "384", "stride": "1"},
        },
    ),
    (
        Index9,
        [((1, 512), torch.int64)],
        {
            "model_names": ["pt_albert_twmkn9_albert_base_v2_squad2_qa_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "14", "stride": "1"},
        },
    ),
    (
        Index10,
        [((3072, 1024), torch.float32)],
        {
            "model_names": [
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index11,
        [((3072, 1024), torch.float32)],
        {
            "model_names": [
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "768", "stop": "1024", "stride": "1"},
        },
    ),
    (
        Index12,
        [((3072, 1024), torch.float32)],
        {
            "model_names": [
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "1536", "stop": "1792", "stride": "1"},
        },
    ),
    (
        Index13,
        [((3072, 1024), torch.float32)],
        {
            "model_names": [
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "2304", "stop": "2560", "stride": "1"},
        },
    ),
    (
        Index14,
        [((3072, 1024), torch.float32)],
        {
            "model_names": [
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "512", "stop": "768", "stride": "1"},
        },
    ),
    (
        Index15,
        [((3072, 1024), torch.float32)],
        {
            "model_names": [
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "1280", "stop": "1536", "stride": "1"},
        },
    ),
    (
        Index16,
        [((3072, 1024), torch.float32)],
        {
            "model_names": [
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "2048", "stop": "2304", "stride": "1"},
        },
    ),
    (
        Index17,
        [((3072, 1024), torch.float32)],
        {
            "model_names": [
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "2816", "stop": "3072", "stride": "1"},
        },
    ),
    (
        Index18,
        [((3072, 1024), torch.float32)],
        {
            "model_names": [
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "256", "stop": "512", "stride": "1"},
        },
    ),
    (
        Index19,
        [((3072, 1024), torch.float32)],
        {
            "model_names": [
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "1024", "stop": "1280", "stride": "1"},
        },
    ),
    (
        Index20,
        [((3072, 1024), torch.float32)],
        {
            "model_names": [
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "1792", "stop": "2048", "stride": "1"},
        },
    ),
    (
        Index21,
        [((3072, 1024), torch.float32)],
        {
            "model_names": [
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "2560", "stop": "2816", "stride": "1"},
        },
    ),
    (
        Index22,
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
        Index23,
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
        Index24,
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
        Index25,
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
        Index0,
        [((1, 128, 768), torch.float32)],
        {
            "model_names": [
                "pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index0,
        [((2, 768), torch.float32)],
        {
            "model_names": [
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_opt_facebook_opt_125m_qa_hf",
                "pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf",
                "pt_albert_twmkn9_albert_base_v2_squad2_qa_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index26,
        [((2, 768), torch.float32)],
        {
            "model_names": [
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_opt_facebook_opt_125m_qa_hf",
                "pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf",
                "pt_albert_twmkn9_albert_base_v2_squad2_qa_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index27,
        [((2,), torch.float32)],
        {
            "model_names": [
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_opt_facebook_opt_125m_qa_hf",
                "onnx_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                "pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf",
                "pt_opt_facebook_opt_350m_qa_hf",
                "pt_albert_twmkn9_albert_base_v2_squad2_qa_hf",
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index28,
        [((2,), torch.float32)],
        {
            "model_names": [
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_opt_facebook_opt_125m_qa_hf",
                "onnx_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                "pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf",
                "pt_opt_facebook_opt_350m_qa_hf",
                "pt_albert_twmkn9_albert_base_v2_squad2_qa_hf",
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index29,
        [((1, 1, 2048, 2048), torch.bool)],
        {
            "model_names": ["pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index10,
        [((1, 1, 2048, 2048), torch.bool)],
        {
            "model_names": ["pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index22,
        [((1, 1, 32, 2048), torch.bool)],
        {
            "model_names": ["pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index30,
        [((1, 32, 2), torch.float32)],
        {
            "model_names": ["pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "31", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index23,
        [((1, 32, 256, 64), torch.float32)],
        {
            "model_names": ["pt_llama3_meta_llama_llama_3_2_1b_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "32", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index22,
        [((1, 32, 256, 64), torch.float32)],
        {
            "model_names": ["pt_llama3_meta_llama_llama_3_2_1b_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index23,
        [((1, 8, 256, 64), torch.float32)],
        {
            "model_names": ["pt_llama3_meta_llama_llama_3_2_1b_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "32", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index22,
        [((1, 8, 256, 64), torch.float32)],
        {
            "model_names": ["pt_llama3_meta_llama_llama_3_2_1b_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index22,
        [((1, 32, 256, 80), torch.float32)],
        {
            "model_names": ["pt_phi2_microsoft_phi_2_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index31,
        [((1, 32, 256, 80), torch.float32)],
        {
            "model_names": ["pt_phi2_microsoft_phi_2_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "32", "stop": "80", "stride": "1"},
        },
    ),
    (
        Index32,
        [((1, 32, 256, 32), torch.float32)],
        {
            "model_names": ["pt_phi2_microsoft_phi_2_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "16", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index33,
        [((1, 32, 256, 32), torch.float32)],
        {
            "model_names": ["pt_phi2_microsoft_phi_2_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "16", "stride": "1"},
        },
    ),
    (
        Index0,
        [((1, 204, 768), torch.float32)],
        {
            "model_names": ["pt_vilt_dandelin_vilt_b32_mlm_mlm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index0,
        [((2, 1024), torch.float32)],
        {
            "model_names": [
                "onnx_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index26,
        [((2, 1024), torch.float32)],
        {
            "model_names": [
                "onnx_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
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
        Index0,
        [((1, 8, 768), torch.float32)],
        {
            "model_names": [
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
        Index0,
        [((1, 32, 16, 3, 96), torch.float32)],
        {
            "model_names": ["pt_bloom_bigscience_bloom_1b1_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index26,
        [((1, 32, 16, 3, 96), torch.float32)],
        {
            "model_names": ["pt_bloom_bigscience_bloom_1b1_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index34,
        [((1, 32, 16, 3, 96), torch.float32)],
        {
            "model_names": ["pt_bloom_bigscience_bloom_1b1_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "2", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index35,
        [((1, 77), torch.int64)],
        {
            "model_names": ["pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "7", "stride": "1"},
        },
    ),
    (
        Index36,
        [((2304, 768), torch.float32)],
        {
            "model_names": [
                "pt_gpt_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "768", "stride": "1"},
        },
    ),
    (
        Index37,
        [((2304, 768), torch.float32)],
        {
            "model_names": [
                "pt_gpt_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "768", "stop": "1536", "stride": "1"},
        },
    ),
    (
        Index38,
        [((2304, 768), torch.float32)],
        {
            "model_names": [
                "pt_gpt_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "1536", "stop": "2304", "stride": "1"},
        },
    ),
    (
        Index39,
        [((2304,), torch.float32)],
        {
            "model_names": [
                "pt_gpt_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "768", "stride": "1"},
        },
    ),
    (
        Index40,
        [((2304,), torch.float32)],
        {
            "model_names": [
                "pt_gpt_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "768", "stop": "1536", "stride": "1"},
        },
    ),
    (
        Index41,
        [((2304,), torch.float32)],
        {
            "model_names": [
                "pt_gpt_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "1536", "stop": "2304", "stride": "1"},
        },
    ),
    (
        Index42,
        [((1, 1, 1024, 1024), torch.bool)],
        {
            "model_names": [
                "pt_gpt_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "7", "stride": "1"},
        },
    ),
    (
        Index35,
        [((1, 1, 7, 1024), torch.bool)],
        {
            "model_names": [
                "pt_gpt_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "7", "stride": "1"},
        },
    ),
    (
        Index23,
        [((1, 32, 4, 64), torch.float32)],
        {
            "model_names": ["pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "32", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index22,
        [((1, 32, 4, 64), torch.float32)],
        {
            "model_names": ["pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index23,
        [((1, 8, 4, 64), torch.float32)],
        {
            "model_names": ["pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "32", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index22,
        [((1, 8, 4, 64), torch.float32)],
        {
            "model_names": ["pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index43,
        [((1, 4, 2), torch.float32)],
        {
            "model_names": ["pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "3", "stop": "4", "stride": "1"},
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
        Index26,
        [((2, 512), torch.float32)],
        {
            "model_names": ["pt_opt_facebook_opt_350m_qa_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index22,
        [((1, 32, 7, 64), torch.float32)],
        {
            "model_names": ["pt_phi_1_5_microsoft_phi_1_5_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index23,
        [((1, 32, 7, 64), torch.float32)],
        {
            "model_names": ["pt_phi_1_5_microsoft_phi_1_5_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "32", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index32,
        [((1, 32, 7, 32), torch.float32)],
        {
            "model_names": ["pt_phi_1_5_microsoft_phi_1_5_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "16", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index33,
        [((1, 32, 7, 32), torch.float32)],
        {
            "model_names": ["pt_phi_1_5_microsoft_phi_1_5_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "16", "stride": "1"},
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
        [((1, 197, 768), torch.float32)],
        {
            "model_names": ["onnx_vit_base_google_vit_base_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index44,
        [((1, 32, 160, 160), torch.float32)],
        {
            "model_names": ["onnx_yolov8_default_obj_det_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "16", "stride": "1"},
        },
    ),
    (
        Index45,
        [((1, 32, 160, 160), torch.float32)],
        {
            "model_names": ["onnx_yolov8_default_obj_det_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "16", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index46,
        [((1, 64, 80, 80), torch.float32)],
        {
            "model_names": ["onnx_yolov8_default_obj_det_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index47,
        [((1, 64, 80, 80), torch.float32)],
        {
            "model_names": ["onnx_yolov8_default_obj_det_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "32", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index48,
        [((1, 128, 40, 40), torch.float32)],
        {
            "model_names": ["onnx_yolov8_default_obj_det_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index49,
        [((1, 128, 40, 40), torch.float32)],
        {
            "model_names": ["onnx_yolov8_default_obj_det_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "64", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index50,
        [((1, 256, 20, 20), torch.float32)],
        {
            "model_names": ["onnx_yolov8_default_obj_det_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index51,
        [((1, 256, 20, 20), torch.float32)],
        {
            "model_names": ["onnx_yolov8_default_obj_det_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "128", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index52,
        [((1, 144, 8400), torch.float32)],
        {
            "model_names": ["onnx_yolov8_default_obj_det_github"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index53,
        [((1, 144, 8400), torch.float32)],
        {
            "model_names": ["onnx_yolov8_default_obj_det_github"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "64", "stop": "144", "stride": "1"},
        },
    ),
    (
        Index54,
        [((1, 4, 8400), torch.float32)],
        {
            "model_names": ["onnx_yolov8_default_obj_det_github"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index55,
        [((1, 4, 8400), torch.float32)],
        {
            "model_names": ["onnx_yolov8_default_obj_det_github"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "2", "stop": "4", "stride": "1"},
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
        Index1,
        [((25, 1, 288), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index2,
        [((25, 1, 288), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index56,
        [((25, 1, 288), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "2", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index57,
        [((25, 1, 288), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "3", "stop": "4", "stride": "1"},
        },
    ),
    (
        Index58,
        [((25, 1, 288), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "4", "stop": "5", "stride": "1"},
        },
    ),
    (
        Index59,
        [((25, 1, 288), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "5", "stop": "6", "stride": "1"},
        },
    ),
    (
        Index60,
        [((25, 1, 288), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "6", "stop": "7", "stride": "1"},
        },
    ),
    (
        Index61,
        [((25, 1, 288), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "7", "stop": "8", "stride": "1"},
        },
    ),
    (
        Index62,
        [((25, 1, 288), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "8", "stop": "9", "stride": "1"},
        },
    ),
    (
        Index63,
        [((25, 1, 288), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "9", "stop": "10", "stride": "1"},
        },
    ),
    (
        Index64,
        [((25, 1, 288), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "10", "stop": "11", "stride": "1"},
        },
    ),
    (
        Index65,
        [((25, 1, 288), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "11", "stop": "12", "stride": "1"},
        },
    ),
    (
        Index66,
        [((25, 1, 288), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "12", "stop": "13", "stride": "1"},
        },
    ),
    (
        Index67,
        [((25, 1, 288), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "13", "stop": "14", "stride": "1"},
        },
    ),
    (
        Index68,
        [((25, 1, 288), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "14", "stop": "15", "stride": "1"},
        },
    ),
    (
        Index69,
        [((25, 1, 288), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "15", "stop": "16", "stride": "1"},
        },
    ),
    (
        Index70,
        [((25, 1, 288), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "16", "stop": "17", "stride": "1"},
        },
    ),
    (
        Index71,
        [((25, 1, 288), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "17", "stop": "18", "stride": "1"},
        },
    ),
    (
        Index72,
        [((25, 1, 288), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "18", "stop": "19", "stride": "1"},
        },
    ),
    (
        Index73,
        [((25, 1, 288), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "19", "stop": "20", "stride": "1"},
        },
    ),
    (
        Index74,
        [((25, 1, 288), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "20", "stop": "21", "stride": "1"},
        },
    ),
    (
        Index75,
        [((25, 1, 288), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "21", "stop": "22", "stride": "1"},
        },
    ),
    (
        Index76,
        [((25, 1, 288), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "22", "stop": "23", "stride": "1"},
        },
    ),
    (
        Index77,
        [((25, 1, 288), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "23", "stop": "24", "stride": "1"},
        },
    ),
    (
        Index78,
        [((25, 1, 288), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "24", "stop": "25", "stride": "1"},
        },
    ),
    (
        Index79,
        [((1, 192), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "144", "stop": "192", "stride": "1"},
        },
    ),
    (
        Index80,
        [((1, 192), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "48", "stop": "96", "stride": "1"},
        },
    ),
    (
        Index81,
        [((1, 192), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "48", "stride": "1"},
        },
    ),
    (
        Index82,
        [((1, 192), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "96", "stop": "144", "stride": "1"},
        },
    ),
    (
        Index1,
        [((25, 1, 96), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index2,
        [((25, 1, 96), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index56,
        [((25, 1, 96), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "2", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index57,
        [((25, 1, 96), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "3", "stop": "4", "stride": "1"},
        },
    ),
    (
        Index58,
        [((25, 1, 96), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "4", "stop": "5", "stride": "1"},
        },
    ),
    (
        Index59,
        [((25, 1, 96), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "5", "stop": "6", "stride": "1"},
        },
    ),
    (
        Index60,
        [((25, 1, 96), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "6", "stop": "7", "stride": "1"},
        },
    ),
    (
        Index61,
        [((25, 1, 96), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "7", "stop": "8", "stride": "1"},
        },
    ),
    (
        Index62,
        [((25, 1, 96), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "8", "stop": "9", "stride": "1"},
        },
    ),
    (
        Index63,
        [((25, 1, 96), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "9", "stop": "10", "stride": "1"},
        },
    ),
    (
        Index64,
        [((25, 1, 96), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "10", "stop": "11", "stride": "1"},
        },
    ),
    (
        Index65,
        [((25, 1, 96), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "11", "stop": "12", "stride": "1"},
        },
    ),
    (
        Index66,
        [((25, 1, 96), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "12", "stop": "13", "stride": "1"},
        },
    ),
    (
        Index67,
        [((25, 1, 96), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "13", "stop": "14", "stride": "1"},
        },
    ),
    (
        Index68,
        [((25, 1, 96), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "14", "stop": "15", "stride": "1"},
        },
    ),
    (
        Index69,
        [((25, 1, 96), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "15", "stop": "16", "stride": "1"},
        },
    ),
    (
        Index70,
        [((25, 1, 96), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "16", "stop": "17", "stride": "1"},
        },
    ),
    (
        Index71,
        [((25, 1, 96), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "17", "stop": "18", "stride": "1"},
        },
    ),
    (
        Index72,
        [((25, 1, 96), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "18", "stop": "19", "stride": "1"},
        },
    ),
    (
        Index73,
        [((25, 1, 96), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "19", "stop": "20", "stride": "1"},
        },
    ),
    (
        Index74,
        [((25, 1, 96), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "20", "stop": "21", "stride": "1"},
        },
    ),
    (
        Index75,
        [((25, 1, 96), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "21", "stop": "22", "stride": "1"},
        },
    ),
    (
        Index76,
        [((25, 1, 96), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "22", "stop": "23", "stride": "1"},
        },
    ),
    (
        Index77,
        [((25, 1, 96), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "23", "stop": "24", "stride": "1"},
        },
    ),
    (
        Index78,
        [((25, 1, 96), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "24", "stop": "25", "stride": "1"},
        },
    ),
    (
        Index83,
        [((1, 64, 64, 96), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "4", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index84,
        [((1, 64, 64, 96), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "4", "stride": "1"},
        },
    ),
    (
        Index85,
        [((1, 64, 64, 96), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "4", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index86,
        [((1, 64, 64, 96), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "4", "stride": "1"},
        },
    ),
    (
        Index87,
        [((1, 64, 64, 96), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "-4", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index88,
        [((1, 64, 64, 96), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "-4", "stride": "1"},
        },
    ),
    (
        Index89,
        [((1, 64, 64, 96), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "-4", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index90,
        [((1, 64, 64, 96), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "-4", "stride": "1"},
        },
    ),
    (
        Index91,
        [((1, 64, 64, 96), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "64", "stride": "2"},
        },
    ),
    (
        Index92,
        [((1, 64, 64, 96), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "1", "stop": "64", "stride": "2"},
        },
    ),
    (
        Index93,
        [((1, 32, 64, 96), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "64", "stride": "2"},
        },
    ),
    (
        Index94,
        [((1, 32, 64, 96), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "1", "stop": "64", "stride": "2"},
        },
    ),
    (
        Index95,
        [((1, 32, 32, 192), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "4", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index84,
        [((1, 32, 32, 192), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "4", "stride": "1"},
        },
    ),
    (
        Index96,
        [((1, 32, 32, 192), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "4", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index86,
        [((1, 32, 32, 192), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "4", "stride": "1"},
        },
    ),
    (
        Index97,
        [((1, 32, 32, 192), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "-4", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index88,
        [((1, 32, 32, 192), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "-4", "stride": "1"},
        },
    ),
    (
        Index98,
        [((1, 32, 32, 192), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "-4", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index90,
        [((1, 32, 32, 192), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "-4", "stride": "1"},
        },
    ),
    (
        Index99,
        [((1, 32, 32, 192), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "32", "stride": "2"},
        },
    ),
    (
        Index100,
        [((1, 32, 32, 192), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "1", "stop": "32", "stride": "2"},
        },
    ),
    (
        Index101,
        [((1, 16, 32, 192), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "32", "stride": "2"},
        },
    ),
    (
        Index102,
        [((1, 16, 32, 192), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "1", "stop": "32", "stride": "2"},
        },
    ),
    (
        Index103,
        [((1, 16, 16, 384), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "4", "stop": "16", "stride": "1"},
        },
    ),
    (
        Index84,
        [((1, 16, 16, 384), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "4", "stride": "1"},
        },
    ),
    (
        Index104,
        [((1, 16, 16, 384), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "4", "stop": "16", "stride": "1"},
        },
    ),
    (
        Index86,
        [((1, 16, 16, 384), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "4", "stride": "1"},
        },
    ),
    (
        Index105,
        [((1, 16, 16, 384), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "-4", "stop": "16", "stride": "1"},
        },
    ),
    (
        Index88,
        [((1, 16, 16, 384), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "-4", "stride": "1"},
        },
    ),
    (
        Index106,
        [((1, 16, 16, 384), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "-4", "stop": "16", "stride": "1"},
        },
    ),
    (
        Index90,
        [((1, 16, 16, 384), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "-4", "stride": "1"},
        },
    ),
    (
        Index107,
        [((1, 16, 16, 384), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "16", "stride": "2"},
        },
    ),
    (
        Index108,
        [((1, 16, 16, 384), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "1", "stop": "16", "stride": "2"},
        },
    ),
    (
        Index109,
        [((1, 8, 16, 384), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "16", "stride": "2"},
        },
    ),
    (
        Index110,
        [((1, 8, 16, 384), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "1", "stop": "16", "stride": "2"},
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
        Index111,
        [((1, 16, 588, 128), torch.float32)],
        {
            "model_names": ["pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "64", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index112,
        [((1, 16, 588, 128), torch.float32)],
        {
            "model_names": ["pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index113,
        [((1, 1, 256, 2048), torch.bool)],
        {
            "model_names": ["pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index114,
        [((1, 4096, 6), torch.float32)],
        {
            "model_names": ["pt_mamba_state_spaces_mamba_370m_hf_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "2048", "stride": "1"},
        },
    ),
    (
        Index115,
        [((1, 4096, 6), torch.float32)],
        {
            "model_names": ["pt_mamba_state_spaces_mamba_370m_hf_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "2048", "stop": "4096", "stride": "1"},
        },
    ),
    (
        Index7,
        [((1, 2048, 9), torch.float32)],
        {
            "model_names": ["pt_mamba_state_spaces_mamba_370m_hf_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "6", "stride": "1"},
        },
    ),
    (
        Index52,
        [((96, 2048), torch.float32)],
        {
            "model_names": ["pt_mamba_state_spaces_mamba_370m_hf_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index116,
        [((96, 2048), torch.float32)],
        {
            "model_names": ["pt_mamba_state_spaces_mamba_370m_hf_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "64", "stop": "80", "stride": "1"},
        },
    ),
    (
        Index117,
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
        Index26,
        [((1, 2048, 6, 16), torch.float32)],
        {
            "model_names": ["pt_mamba_state_spaces_mamba_370m_hf_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index34,
        [((1, 2048, 6, 16), torch.float32)],
        {
            "model_names": ["pt_mamba_state_spaces_mamba_370m_hf_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "2", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index43,
        [((1, 2048, 6, 16), torch.float32)],
        {
            "model_names": ["pt_mamba_state_spaces_mamba_370m_hf_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "3", "stop": "4", "stride": "1"},
        },
    ),
    (
        Index118,
        [((1, 2048, 6, 16), torch.float32)],
        {
            "model_names": ["pt_mamba_state_spaces_mamba_370m_hf_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "4", "stop": "5", "stride": "1"},
        },
    ),
    (
        Index119,
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
            "model_names": ["pt_mamba_state_spaces_mamba_370m_hf_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index26,
        [((1, 6, 16), torch.float32)],
        {
            "model_names": ["pt_mamba_state_spaces_mamba_370m_hf_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index34,
        [((1, 6, 16), torch.float32)],
        {
            "model_names": ["pt_mamba_state_spaces_mamba_370m_hf_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "2", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index43,
        [((1, 6, 16), torch.float32)],
        {
            "model_names": ["pt_mamba_state_spaces_mamba_370m_hf_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "3", "stop": "4", "stride": "1"},
        },
    ),
    (
        Index118,
        [((1, 6, 16), torch.float32)],
        {
            "model_names": ["pt_mamba_state_spaces_mamba_370m_hf_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "4", "stop": "5", "stride": "1"},
        },
    ),
    (
        Index119,
        [((1, 6, 16), torch.float32)],
        {
            "model_names": ["pt_mamba_state_spaces_mamba_370m_hf_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "5", "stop": "6", "stride": "1"},
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
        Index26,
        [((2, 4, 1), torch.int64)],
        {
            "model_names": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index34,
        [((2, 4, 1), torch.int64)],
        {
            "model_names": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "2", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index43,
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
