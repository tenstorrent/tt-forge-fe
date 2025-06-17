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
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=0, stop=8, stride=1)
        return index_output_1


class Index2(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=0, stop=14, stride=1)
        return index_output_1


class Index3(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=0, stop=9, stride=1)
        return index_output_1


class Index4(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=0, stop=384, stride=1)
        return index_output_1


class Index5(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=0, stop=16, stride=1)
        return index_output_1


class Index6(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=1, stride=1)
        return index_output_1


class Index7(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=1, stop=2, stride=1)
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
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=0, stop=32, stride=1)
        return index_output_1


class Index11(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=32, stop=64, stride=1)
        return index_output_1


class Index12(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=16, stop=32, stride=1)
        return index_output_1


class Index13(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=1, stride=1)
        return index_output_1


class Index14(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=1, stop=2, stride=1)
        return index_output_1


class Index15(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=64, stop=128, stride=1)
        return index_output_1


class Index16(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=0, stop=64, stride=1)
        return index_output_1


class Index17(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=768, stride=1)
        return index_output_1


class Index18(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=768, stop=1536, stride=1)
        return index_output_1


class Index19(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=1536, stop=2304, stride=1)
        return index_output_1


class Index20(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=0, stop=768, stride=1)
        return index_output_1


class Index21(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=768, stop=1536, stride=1)
        return index_output_1


class Index22(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=1536, stop=2304, stride=1)
        return index_output_1


class Index23(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=16, stride=1)
        return index_output_1


class Index24(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=16, stop=32, stride=1)
        return index_output_1


class Index25(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=32, stop=64, stride=1)
        return index_output_1


class Index26(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=64, stride=1)
        return index_output_1


class Index27(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=64, stop=96, stride=1)
        return index_output_1


class Index28(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=96, stop=112, stride=1)
        return index_output_1


class Index29(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=96, stride=1)
        return index_output_1


class Index30(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=96, stop=160, stride=1)
        return index_output_1


class Index31(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=160, stop=224, stride=1)
        return index_output_1


class Index32(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=384, stride=1)
        return index_output_1


class Index33(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=384, stop=576, stride=1)
        return index_output_1


class Index34(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=576, stop=768, stride=1)
        return index_output_1


class Index35(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=256, stride=1)
        return index_output_1


class Index36(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=256, stop=640, stride=1)
        return index_output_1


class Index37(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=640, stop=1024, stride=1)
        return index_output_1


class Index38(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=3072, stride=1)
        return index_output_1


class Index39(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=3072, stop=6144, stride=1)
        return index_output_1


class Index40(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=0, stop=6, stride=1)
        return index_output_1


class Index41(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=96, stride=1)
        return index_output_1


class Index42(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=96, stop=112, stride=1)
        return index_output_1


class Index43(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=112, stop=128, stride=1)
        return index_output_1


class Index44(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=2, stop=3, stride=1)
        return index_output_1


class Index45(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=3, stop=4, stride=1)
        return index_output_1


class Index46(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=4, stop=5, stride=1)
        return index_output_1


class Index47(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=5, stop=6, stride=1)
        return index_output_1


class Index48(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=-1, stop=72, stride=1)
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
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=32, stop=80, stride=1)
        return index_output_1


class Index54(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=0, stop=1, stride=1)
        return index_output_1


class Index55(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=1, stop=2, stride=1)
        return index_output_1


class Index56(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=2, stop=3, stride=1)
        return index_output_1


class Index57(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=0, stop=2, stride=1)
        return index_output_1


class Index58(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=2, stop=4, stride=1)
        return index_output_1


class Index59(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=80, stride=1)
        return index_output_1


class Index60(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=80, stop=160, stride=1)
        return index_output_1


class Index61(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=160, stride=1)
        return index_output_1


class Index62(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=160, stop=320, stride=1)
        return index_output_1


class Index63(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=320, stride=1)
        return index_output_1


class Index64(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=320, stop=640, stride=1)
        return index_output_1


class Index65(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=64, stop=128, stride=1)
        return index_output_1


class Index66(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=32, stride=1)
        return index_output_1


class Index67(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=32, stop=64, stride=1)
        return index_output_1


class Index68(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=64, stride=1)
        return index_output_1


class Index69(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=64, stop=144, stride=1)
        return index_output_1


class Index70(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=2, stride=1)
        return index_output_1


class Index71(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=2, stop=4, stride=1)
        return index_output_1


class Index72(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=640, stride=2)
        return index_output_1


class Index73(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=1, stop=640, stride=2)
        return index_output_1


class Index74(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=0, stop=640, stride=2)
        return index_output_1


class Index75(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=1, stop=640, stride=2)
        return index_output_1


class Index76(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=3, stop=4, stride=1)
        return index_output_1


class Index77(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=4, stop=5, stride=1)
        return index_output_1


class Index78(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=5, stop=6, stride=1)
        return index_output_1


class Index79(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=6, stop=7, stride=1)
        return index_output_1


class Index80(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=7, stop=8, stride=1)
        return index_output_1


class Index81(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=8, stop=9, stride=1)
        return index_output_1


class Index82(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=9, stop=10, stride=1)
        return index_output_1


class Index83(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=10, stop=11, stride=1)
        return index_output_1


class Index84(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=11, stop=12, stride=1)
        return index_output_1


class Index85(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=12, stop=13, stride=1)
        return index_output_1


class Index86(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=13, stop=14, stride=1)
        return index_output_1


class Index87(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=14, stop=15, stride=1)
        return index_output_1


class Index88(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=15, stop=16, stride=1)
        return index_output_1


class Index89(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=16, stop=17, stride=1)
        return index_output_1


class Index90(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=17, stop=18, stride=1)
        return index_output_1


class Index91(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=18, stop=19, stride=1)
        return index_output_1


class Index92(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=19, stop=20, stride=1)
        return index_output_1


class Index93(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=20, stop=21, stride=1)
        return index_output_1


class Index94(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=21, stop=22, stride=1)
        return index_output_1


class Index95(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=22, stop=23, stride=1)
        return index_output_1


class Index96(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=23, stop=24, stride=1)
        return index_output_1


class Index97(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=24, stop=25, stride=1)
        return index_output_1


class Index98(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=25, stop=26, stride=1)
        return index_output_1


class Index99(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=26, stop=27, stride=1)
        return index_output_1


class Index100(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=27, stop=28, stride=1)
        return index_output_1


class Index101(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=28, stop=29, stride=1)
        return index_output_1


class Index102(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=29, stop=30, stride=1)
        return index_output_1


class Index103(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=30, stop=31, stride=1)
        return index_output_1


class Index104(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=31, stop=32, stride=1)
        return index_output_1


class Index105(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=32, stop=33, stride=1)
        return index_output_1


class Index106(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=33, stop=34, stride=1)
        return index_output_1


class Index107(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=34, stop=35, stride=1)
        return index_output_1


class Index108(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=35, stop=36, stride=1)
        return index_output_1


class Index109(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=36, stop=37, stride=1)
        return index_output_1


class Index110(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=37, stop=38, stride=1)
        return index_output_1


class Index111(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=38, stop=39, stride=1)
        return index_output_1


class Index112(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=39, stop=40, stride=1)
        return index_output_1


class Index113(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=40, stop=41, stride=1)
        return index_output_1


class Index114(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=41, stop=42, stride=1)
        return index_output_1


class Index115(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=42, stop=43, stride=1)
        return index_output_1


class Index116(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=43, stop=44, stride=1)
        return index_output_1


class Index117(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=44, stop=45, stride=1)
        return index_output_1


class Index118(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=45, stop=46, stride=1)
        return index_output_1


class Index119(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=46, stop=47, stride=1)
        return index_output_1


class Index120(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=47, stop=48, stride=1)
        return index_output_1


class Index121(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=48, stop=49, stride=1)
        return index_output_1


class Index122(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=49, stop=50, stride=1)
        return index_output_1


class Index123(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=50, stop=51, stride=1)
        return index_output_1


class Index124(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=51, stop=52, stride=1)
        return index_output_1


class Index125(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=52, stop=53, stride=1)
        return index_output_1


class Index126(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=53, stop=54, stride=1)
        return index_output_1


class Index127(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=54, stop=55, stride=1)
        return index_output_1


class Index128(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=55, stop=56, stride=1)
        return index_output_1


class Index129(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=56, stop=57, stride=1)
        return index_output_1


class Index130(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=57, stop=58, stride=1)
        return index_output_1


class Index131(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=58, stop=59, stride=1)
        return index_output_1


class Index132(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=59, stop=60, stride=1)
        return index_output_1


class Index133(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=60, stop=61, stride=1)
        return index_output_1


class Index134(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=61, stop=62, stride=1)
        return index_output_1


class Index135(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=62, stop=63, stride=1)
        return index_output_1


class Index136(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=63, stop=64, stride=1)
        return index_output_1


class Index137(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=64, stop=65, stride=1)
        return index_output_1


class Index138(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=65, stop=66, stride=1)
        return index_output_1


class Index139(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=66, stop=67, stride=1)
        return index_output_1


class Index140(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=67, stop=68, stride=1)
        return index_output_1


class Index141(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=68, stop=69, stride=1)
        return index_output_1


class Index142(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=69, stop=70, stride=1)
        return index_output_1


class Index143(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=70, stop=71, stride=1)
        return index_output_1


class Index144(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=71, stop=72, stride=1)
        return index_output_1


class Index145(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=72, stop=73, stride=1)
        return index_output_1


class Index146(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=73, stop=74, stride=1)
        return index_output_1


class Index147(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=74, stop=75, stride=1)
        return index_output_1


class Index148(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=75, stop=76, stride=1)
        return index_output_1


class Index149(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=76, stop=77, stride=1)
        return index_output_1


class Index150(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=77, stop=78, stride=1)
        return index_output_1


class Index151(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=78, stop=79, stride=1)
        return index_output_1


class Index152(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=79, stop=80, stride=1)
        return index_output_1


class Index153(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=80, stop=81, stride=1)
        return index_output_1


class Index154(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=81, stop=82, stride=1)
        return index_output_1


class Index155(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=82, stop=83, stride=1)
        return index_output_1


class Index156(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=83, stop=84, stride=1)
        return index_output_1


class Index157(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=84, stop=85, stride=1)
        return index_output_1


class Index158(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=85, stop=86, stride=1)
        return index_output_1


class Index159(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=86, stop=87, stride=1)
        return index_output_1


class Index160(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=87, stop=88, stride=1)
        return index_output_1


class Index161(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=88, stop=89, stride=1)
        return index_output_1


class Index162(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=89, stop=90, stride=1)
        return index_output_1


class Index163(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=90, stop=91, stride=1)
        return index_output_1


class Index164(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=91, stop=92, stride=1)
        return index_output_1


class Index165(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=92, stop=93, stride=1)
        return index_output_1


class Index166(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=93, stop=94, stride=1)
        return index_output_1


class Index167(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=94, stop=95, stride=1)
        return index_output_1


class Index168(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=95, stop=96, stride=1)
        return index_output_1


class Index169(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=96, stop=97, stride=1)
        return index_output_1


class Index170(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=97, stop=98, stride=1)
        return index_output_1


class Index171(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=98, stop=99, stride=1)
        return index_output_1


class Index172(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=99, stop=100, stride=1)
        return index_output_1


class Index173(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=100, stop=101, stride=1)
        return index_output_1


class Index174(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=101, stop=102, stride=1)
        return index_output_1


class Index175(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=102, stop=103, stride=1)
        return index_output_1


class Index176(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=103, stop=104, stride=1)
        return index_output_1


class Index177(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=104, stop=105, stride=1)
        return index_output_1


class Index178(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=105, stop=106, stride=1)
        return index_output_1


class Index179(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=106, stop=107, stride=1)
        return index_output_1


class Index180(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=107, stop=108, stride=1)
        return index_output_1


class Index181(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=108, stop=109, stride=1)
        return index_output_1


class Index182(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=109, stop=110, stride=1)
        return index_output_1


class Index183(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=110, stop=111, stride=1)
        return index_output_1


class Index184(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=111, stop=112, stride=1)
        return index_output_1


class Index185(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=112, stop=113, stride=1)
        return index_output_1


class Index186(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=113, stop=114, stride=1)
        return index_output_1


class Index187(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=114, stop=115, stride=1)
        return index_output_1


class Index188(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=115, stop=116, stride=1)
        return index_output_1


class Index189(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=116, stop=117, stride=1)
        return index_output_1


class Index190(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=117, stop=118, stride=1)
        return index_output_1


class Index191(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=118, stop=119, stride=1)
        return index_output_1


class Index192(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=119, stop=120, stride=1)
        return index_output_1


class Index193(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=120, stop=121, stride=1)
        return index_output_1


class Index194(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=121, stop=122, stride=1)
        return index_output_1


class Index195(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=122, stop=123, stride=1)
        return index_output_1


class Index196(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=123, stop=124, stride=1)
        return index_output_1


class Index197(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=124, stop=125, stride=1)
        return index_output_1


class Index198(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=125, stop=126, stride=1)
        return index_output_1


class Index199(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=126, stop=127, stride=1)
        return index_output_1


class Index200(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=127, stop=128, stride=1)
        return index_output_1


class Index201(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=128, stop=129, stride=1)
        return index_output_1


class Index202(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=129, stop=130, stride=1)
        return index_output_1


class Index203(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=130, stop=131, stride=1)
        return index_output_1


class Index204(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=131, stop=132, stride=1)
        return index_output_1


class Index205(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=132, stop=133, stride=1)
        return index_output_1


class Index206(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=133, stop=134, stride=1)
        return index_output_1


class Index207(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=134, stop=135, stride=1)
        return index_output_1


class Index208(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=135, stop=136, stride=1)
        return index_output_1


class Index209(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=136, stop=137, stride=1)
        return index_output_1


class Index210(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=137, stop=138, stride=1)
        return index_output_1


class Index211(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=138, stop=139, stride=1)
        return index_output_1


class Index212(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=139, stop=140, stride=1)
        return index_output_1


class Index213(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=140, stop=141, stride=1)
        return index_output_1


class Index214(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=141, stop=142, stride=1)
        return index_output_1


class Index215(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=142, stop=143, stride=1)
        return index_output_1


class Index216(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=143, stop=144, stride=1)
        return index_output_1


class Index217(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=144, stop=145, stride=1)
        return index_output_1


class Index218(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=145, stop=146, stride=1)
        return index_output_1


class Index219(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=146, stop=147, stride=1)
        return index_output_1


class Index220(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=147, stop=148, stride=1)
        return index_output_1


class Index221(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=148, stop=149, stride=1)
        return index_output_1


class Index222(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=149, stop=150, stride=1)
        return index_output_1


class Index223(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=150, stop=151, stride=1)
        return index_output_1


class Index224(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=151, stop=152, stride=1)
        return index_output_1


class Index225(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=152, stop=153, stride=1)
        return index_output_1


class Index226(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=153, stop=154, stride=1)
        return index_output_1


class Index227(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=154, stop=155, stride=1)
        return index_output_1


class Index228(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=155, stop=156, stride=1)
        return index_output_1


class Index229(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=156, stop=157, stride=1)
        return index_output_1


class Index230(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=157, stop=158, stride=1)
        return index_output_1


class Index231(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=158, stop=159, stride=1)
        return index_output_1


class Index232(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=159, stop=160, stride=1)
        return index_output_1


class Index233(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=160, stop=161, stride=1)
        return index_output_1


class Index234(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=161, stop=162, stride=1)
        return index_output_1


class Index235(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=162, stop=163, stride=1)
        return index_output_1


class Index236(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=163, stop=164, stride=1)
        return index_output_1


class Index237(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=164, stop=165, stride=1)
        return index_output_1


class Index238(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=165, stop=166, stride=1)
        return index_output_1


class Index239(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=166, stop=167, stride=1)
        return index_output_1


class Index240(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=167, stop=168, stride=1)
        return index_output_1


class Index241(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=168, stop=169, stride=1)
        return index_output_1


class Index242(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=169, stop=170, stride=1)
        return index_output_1


class Index243(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=170, stop=171, stride=1)
        return index_output_1


class Index244(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=171, stop=172, stride=1)
        return index_output_1


class Index245(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=172, stop=173, stride=1)
        return index_output_1


class Index246(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=173, stop=174, stride=1)
        return index_output_1


class Index247(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=174, stop=175, stride=1)
        return index_output_1


class Index248(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=175, stop=176, stride=1)
        return index_output_1


class Index249(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=176, stop=177, stride=1)
        return index_output_1


class Index250(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=177, stop=178, stride=1)
        return index_output_1


class Index251(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=178, stop=179, stride=1)
        return index_output_1


class Index252(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=179, stop=180, stride=1)
        return index_output_1


class Index253(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=180, stop=181, stride=1)
        return index_output_1


class Index254(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=181, stop=182, stride=1)
        return index_output_1


class Index255(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=182, stop=183, stride=1)
        return index_output_1


class Index256(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=183, stop=184, stride=1)
        return index_output_1


class Index257(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=184, stop=185, stride=1)
        return index_output_1


class Index258(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=185, stop=186, stride=1)
        return index_output_1


class Index259(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=186, stop=187, stride=1)
        return index_output_1


class Index260(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=187, stop=188, stride=1)
        return index_output_1


class Index261(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=188, stop=189, stride=1)
        return index_output_1


class Index262(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=189, stop=190, stride=1)
        return index_output_1


class Index263(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=190, stop=191, stride=1)
        return index_output_1


class Index264(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=191, stop=192, stride=1)
        return index_output_1


class Index265(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=192, stop=193, stride=1)
        return index_output_1


class Index266(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=193, stop=194, stride=1)
        return index_output_1


class Index267(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=194, stop=195, stride=1)
        return index_output_1


class Index268(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=195, stop=196, stride=1)
        return index_output_1


class Index269(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=196, stop=197, stride=1)
        return index_output_1


class Index270(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=197, stop=198, stride=1)
        return index_output_1


class Index271(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=198, stop=199, stride=1)
        return index_output_1


class Index272(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=199, stop=200, stride=1)
        return index_output_1


class Index273(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=200, stop=201, stride=1)
        return index_output_1


class Index274(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=201, stop=202, stride=1)
        return index_output_1


class Index275(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=202, stop=203, stride=1)
        return index_output_1


class Index276(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=203, stop=204, stride=1)
        return index_output_1


class Index277(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=204, stop=205, stride=1)
        return index_output_1


class Index278(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=205, stop=206, stride=1)
        return index_output_1


class Index279(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=206, stop=207, stride=1)
        return index_output_1


class Index280(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=207, stop=208, stride=1)
        return index_output_1


class Index281(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=208, stop=209, stride=1)
        return index_output_1


class Index282(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=209, stop=210, stride=1)
        return index_output_1


class Index283(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=210, stop=211, stride=1)
        return index_output_1


class Index284(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=211, stop=212, stride=1)
        return index_output_1


class Index285(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=212, stop=213, stride=1)
        return index_output_1


class Index286(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=213, stop=214, stride=1)
        return index_output_1


class Index287(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=214, stop=215, stride=1)
        return index_output_1


class Index288(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=215, stop=216, stride=1)
        return index_output_1


class Index289(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=216, stop=217, stride=1)
        return index_output_1


class Index290(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=217, stop=218, stride=1)
        return index_output_1


class Index291(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=218, stop=219, stride=1)
        return index_output_1


class Index292(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=219, stop=220, stride=1)
        return index_output_1


class Index293(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=220, stop=221, stride=1)
        return index_output_1


class Index294(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=221, stop=222, stride=1)
        return index_output_1


class Index295(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=222, stop=223, stride=1)
        return index_output_1


class Index296(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=223, stop=224, stride=1)
        return index_output_1


class Index297(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=224, stop=225, stride=1)
        return index_output_1


class Index298(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=225, stop=226, stride=1)
        return index_output_1


class Index299(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=226, stop=227, stride=1)
        return index_output_1


class Index300(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=227, stop=228, stride=1)
        return index_output_1


class Index301(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=228, stop=229, stride=1)
        return index_output_1


class Index302(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=229, stop=230, stride=1)
        return index_output_1


class Index303(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=230, stop=231, stride=1)
        return index_output_1


class Index304(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=231, stop=232, stride=1)
        return index_output_1


class Index305(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=232, stop=233, stride=1)
        return index_output_1


class Index306(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=233, stop=234, stride=1)
        return index_output_1


class Index307(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=234, stop=235, stride=1)
        return index_output_1


class Index308(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=235, stop=236, stride=1)
        return index_output_1


class Index309(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=236, stop=237, stride=1)
        return index_output_1


class Index310(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=237, stop=238, stride=1)
        return index_output_1


class Index311(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=238, stop=239, stride=1)
        return index_output_1


class Index312(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=239, stop=240, stride=1)
        return index_output_1


class Index313(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=240, stop=241, stride=1)
        return index_output_1


class Index314(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=241, stop=242, stride=1)
        return index_output_1


class Index315(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=242, stop=243, stride=1)
        return index_output_1


class Index316(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=243, stop=244, stride=1)
        return index_output_1


class Index317(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=244, stop=245, stride=1)
        return index_output_1


class Index318(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=245, stop=246, stride=1)
        return index_output_1


class Index319(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=246, stop=247, stride=1)
        return index_output_1


class Index320(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=247, stop=248, stride=1)
        return index_output_1


class Index321(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=248, stop=249, stride=1)
        return index_output_1


class Index322(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=249, stop=250, stride=1)
        return index_output_1


class Index323(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=250, stop=251, stride=1)
        return index_output_1


class Index324(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=251, stop=252, stride=1)
        return index_output_1


class Index325(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=252, stop=253, stride=1)
        return index_output_1


class Index326(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=253, stop=254, stride=1)
        return index_output_1


class Index327(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=254, stop=255, stride=1)
        return index_output_1


class Index328(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=255, stop=256, stride=1)
        return index_output_1


class Index329(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=256, stop=257, stride=1)
        return index_output_1


class Index330(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=257, stop=258, stride=1)
        return index_output_1


class Index331(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=258, stop=259, stride=1)
        return index_output_1


class Index332(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=259, stop=260, stride=1)
        return index_output_1


class Index333(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=260, stop=261, stride=1)
        return index_output_1


class Index334(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=261, stop=262, stride=1)
        return index_output_1


class Index335(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=262, stop=263, stride=1)
        return index_output_1


class Index336(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=263, stop=264, stride=1)
        return index_output_1


class Index337(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=264, stop=265, stride=1)
        return index_output_1


class Index338(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=265, stop=266, stride=1)
        return index_output_1


class Index339(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=266, stop=267, stride=1)
        return index_output_1


class Index340(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=267, stop=268, stride=1)
        return index_output_1


class Index341(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=268, stop=269, stride=1)
        return index_output_1


class Index342(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=269, stop=270, stride=1)
        return index_output_1


class Index343(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=270, stop=271, stride=1)
        return index_output_1


class Index344(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=271, stop=272, stride=1)
        return index_output_1


class Index345(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=272, stop=273, stride=1)
        return index_output_1


class Index346(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=273, stop=274, stride=1)
        return index_output_1


class Index347(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=274, stop=275, stride=1)
        return index_output_1


class Index348(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=275, stop=276, stride=1)
        return index_output_1


class Index349(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=276, stop=277, stride=1)
        return index_output_1


class Index350(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=277, stop=278, stride=1)
        return index_output_1


class Index351(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=278, stop=279, stride=1)
        return index_output_1


class Index352(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=279, stop=280, stride=1)
        return index_output_1


class Index353(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=280, stop=281, stride=1)
        return index_output_1


class Index354(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=281, stop=282, stride=1)
        return index_output_1


class Index355(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=282, stop=283, stride=1)
        return index_output_1


class Index356(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=283, stop=284, stride=1)
        return index_output_1


class Index357(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=284, stop=285, stride=1)
        return index_output_1


class Index358(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=285, stop=286, stride=1)
        return index_output_1


class Index359(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=286, stop=287, stride=1)
        return index_output_1


class Index360(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=287, stop=288, stride=1)
        return index_output_1


class Index361(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=288, stop=289, stride=1)
        return index_output_1


class Index362(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=289, stop=290, stride=1)
        return index_output_1


class Index363(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=290, stop=291, stride=1)
        return index_output_1


class Index364(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=291, stop=292, stride=1)
        return index_output_1


class Index365(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=292, stop=293, stride=1)
        return index_output_1


class Index366(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=293, stop=294, stride=1)
        return index_output_1


class Index367(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=294, stop=295, stride=1)
        return index_output_1


class Index368(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=295, stop=296, stride=1)
        return index_output_1


class Index369(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=296, stop=297, stride=1)
        return index_output_1


class Index370(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=297, stop=298, stride=1)
        return index_output_1


class Index371(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=298, stop=299, stride=1)
        return index_output_1


class Index372(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-4, start=299, stop=300, stride=1)
        return index_output_1


class Index373(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=6, stop=7, stride=1)
        return index_output_1


class Index374(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=7, stop=8, stride=1)
        return index_output_1


class Index375(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=8, stop=9, stride=1)
        return index_output_1


class Index376(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=9, stop=10, stride=1)
        return index_output_1


class Index377(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=10, stop=11, stride=1)
        return index_output_1


class Index378(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=11, stop=12, stride=1)
        return index_output_1


class Index379(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=12, stop=13, stride=1)
        return index_output_1


class Index380(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=13, stop=14, stride=1)
        return index_output_1


class Index381(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=2, stop=3, stride=1)
        return index_output_1


class Index382(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=3, stop=4, stride=1)
        return index_output_1


class Index383(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=4, stop=5, stride=1)
        return index_output_1


class Index384(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=5, stop=6, stride=1)
        return index_output_1


class Index385(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=6, stop=7, stride=1)
        return index_output_1


class Index386(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=7, stop=8, stride=1)
        return index_output_1


class Index387(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=8, stop=9, stride=1)
        return index_output_1


class Index388(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=9, stop=10, stride=1)
        return index_output_1


class Index389(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=10, stop=11, stride=1)
        return index_output_1


class Index390(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=11, stop=12, stride=1)
        return index_output_1


class Index391(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=12, stop=13, stride=1)
        return index_output_1


class Index392(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=13, stop=14, stride=1)
        return index_output_1


class Index393(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=14, stop=15, stride=1)
        return index_output_1


class Index394(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=15, stop=16, stride=1)
        return index_output_1


class Index395(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=16, stop=17, stride=1)
        return index_output_1


class Index396(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=17, stop=18, stride=1)
        return index_output_1


class Index397(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=18, stop=19, stride=1)
        return index_output_1


class Index398(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=19, stop=20, stride=1)
        return index_output_1


class Index399(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=20, stop=21, stride=1)
        return index_output_1


class Index400(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=21, stop=22, stride=1)
        return index_output_1


class Index401(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=22, stop=23, stride=1)
        return index_output_1


class Index402(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=23, stop=24, stride=1)
        return index_output_1


class Index403(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=24, stop=25, stride=1)
        return index_output_1


class Index404(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=25, stop=26, stride=1)
        return index_output_1


class Index405(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=26, stop=27, stride=1)
        return index_output_1


class Index406(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=27, stop=28, stride=1)
        return index_output_1


class Index407(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=28, stop=29, stride=1)
        return index_output_1


class Index408(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=29, stop=30, stride=1)
        return index_output_1


class Index409(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=30, stop=31, stride=1)
        return index_output_1


class Index410(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=31, stop=32, stride=1)
        return index_output_1


class Index411(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=32, stop=33, stride=1)
        return index_output_1


class Index412(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=33, stop=34, stride=1)
        return index_output_1


class Index413(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=34, stop=35, stride=1)
        return index_output_1


class Index414(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=35, stop=36, stride=1)
        return index_output_1


class Index415(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=36, stop=37, stride=1)
        return index_output_1


class Index416(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=37, stop=38, stride=1)
        return index_output_1


class Index417(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=38, stop=39, stride=1)
        return index_output_1


class Index418(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=39, stop=40, stride=1)
        return index_output_1


class Index419(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=40, stop=41, stride=1)
        return index_output_1


class Index420(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=41, stop=42, stride=1)
        return index_output_1


class Index421(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=42, stop=43, stride=1)
        return index_output_1


class Index422(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=43, stop=44, stride=1)
        return index_output_1


class Index423(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=44, stop=45, stride=1)
        return index_output_1


class Index424(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=45, stop=46, stride=1)
        return index_output_1


class Index425(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=46, stop=47, stride=1)
        return index_output_1


class Index426(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=47, stop=48, stride=1)
        return index_output_1


class Index427(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=48, stop=49, stride=1)
        return index_output_1


class Index428(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=49, stop=50, stride=1)
        return index_output_1


class Index429(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=50, stop=51, stride=1)
        return index_output_1


class Index430(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=51, stop=52, stride=1)
        return index_output_1


class Index431(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=52, stop=53, stride=1)
        return index_output_1


class Index432(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=53, stop=54, stride=1)
        return index_output_1


class Index433(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=54, stop=55, stride=1)
        return index_output_1


class Index434(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=55, stop=56, stride=1)
        return index_output_1


class Index435(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=56, stop=57, stride=1)
        return index_output_1


class Index436(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=57, stop=58, stride=1)
        return index_output_1


class Index437(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=58, stop=59, stride=1)
        return index_output_1


class Index438(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=59, stop=60, stride=1)
        return index_output_1


class Index439(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=60, stop=61, stride=1)
        return index_output_1


class Index440(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=61, stop=62, stride=1)
        return index_output_1


class Index441(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=62, stop=63, stride=1)
        return index_output_1


class Index442(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=63, stop=64, stride=1)
        return index_output_1


class Index443(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=14, stop=15, stride=1)
        return index_output_1


class Index444(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=15, stop=16, stride=1)
        return index_output_1


class Index445(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=16, stop=17, stride=1)
        return index_output_1


class Index446(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=17, stop=18, stride=1)
        return index_output_1


class Index447(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=18, stop=19, stride=1)
        return index_output_1


class Index448(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=19, stop=20, stride=1)
        return index_output_1


class Index449(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=20, stop=21, stride=1)
        return index_output_1


class Index450(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=21, stop=22, stride=1)
        return index_output_1


class Index451(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=22, stop=23, stride=1)
        return index_output_1


class Index452(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=23, stop=24, stride=1)
        return index_output_1


class Index453(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=24, stop=25, stride=1)
        return index_output_1


class Index454(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=25, stop=26, stride=1)
        return index_output_1


class Index455(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=26, stop=27, stride=1)
        return index_output_1


class Index456(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=27, stop=28, stride=1)
        return index_output_1


class Index457(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=28, stop=29, stride=1)
        return index_output_1


class Index458(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=29, stop=30, stride=1)
        return index_output_1


class Index459(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=30, stop=31, stride=1)
        return index_output_1


class Index460(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=31, stop=32, stride=1)
        return index_output_1


class Index461(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=32, stop=33, stride=1)
        return index_output_1


class Index462(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=33, stop=34, stride=1)
        return index_output_1


class Index463(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=34, stop=35, stride=1)
        return index_output_1


class Index464(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=35, stop=36, stride=1)
        return index_output_1


class Index465(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=36, stop=37, stride=1)
        return index_output_1


class Index466(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=37, stop=38, stride=1)
        return index_output_1


class Index467(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=38, stop=39, stride=1)
        return index_output_1


class Index468(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=39, stop=40, stride=1)
        return index_output_1


class Index469(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=40, stop=41, stride=1)
        return index_output_1


class Index470(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=41, stop=42, stride=1)
        return index_output_1


class Index471(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=42, stop=43, stride=1)
        return index_output_1


class Index472(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=43, stop=44, stride=1)
        return index_output_1


class Index473(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=44, stop=45, stride=1)
        return index_output_1


class Index474(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=45, stop=46, stride=1)
        return index_output_1


class Index475(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=46, stop=47, stride=1)
        return index_output_1


class Index476(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=47, stop=48, stride=1)
        return index_output_1


class Index477(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=48, stop=49, stride=1)
        return index_output_1


class Index478(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=49, stop=50, stride=1)
        return index_output_1


class Index479(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=50, stop=51, stride=1)
        return index_output_1


class Index480(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=51, stop=52, stride=1)
        return index_output_1


class Index481(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=52, stop=53, stride=1)
        return index_output_1


class Index482(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=53, stop=54, stride=1)
        return index_output_1


class Index483(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=54, stop=55, stride=1)
        return index_output_1


class Index484(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=55, stop=56, stride=1)
        return index_output_1


class Index485(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=56, stop=57, stride=1)
        return index_output_1


class Index486(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=57, stop=58, stride=1)
        return index_output_1


class Index487(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=58, stop=59, stride=1)
        return index_output_1


class Index488(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=59, stop=60, stride=1)
        return index_output_1


class Index489(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=60, stop=61, stride=1)
        return index_output_1


class Index490(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=61, stop=62, stride=1)
        return index_output_1


class Index491(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=62, stop=63, stride=1)
        return index_output_1


class Index492(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=63, stop=64, stride=1)
        return index_output_1


class Index493(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=1, stop=5, stride=1)
        return index_output_1


class Index494(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=1, stop=4, stride=1)
        return index_output_1


class Index495(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=1, stop=4, stride=1)
        return index_output_1


class Index496(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=32, stride=1)
        return index_output_1


class Index497(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=64, stop=128, stride=1)
        return index_output_1


class Index498(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=128, stride=1)
        return index_output_1


class Index499(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=128, stop=256, stride=1)
        return index_output_1


class Index500(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=64, stop=160, stride=1)
        return index_output_1


class Index501(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=160, stop=176, stride=1)
        return index_output_1


class Index502(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=256, stop=288, stride=1)
        return index_output_1


class Index503(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=192, stride=1)
        return index_output_1


class Index504(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=192, stop=288, stride=1)
        return index_output_1


class Index505(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=288, stop=304, stride=1)
        return index_output_1


class Index506(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=160, stop=272, stride=1)
        return index_output_1


class Index507(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=272, stop=296, stride=1)
        return index_output_1


class Index508(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=256, stop=280, stride=1)
        return index_output_1


class Index509(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=112, stride=1)
        return index_output_1


class Index510(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=112, stop=256, stride=1)
        return index_output_1


class Index511(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=256, stop=416, stride=1)
        return index_output_1


class Index512(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=416, stop=448, stride=1)
        return index_output_1


class Index513(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=576, stop=624, stride=1)
        return index_output_1


class Index514(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=144, stop=192, stride=1)
        return index_output_1


class Index515(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=48, stop=96, stride=1)
        return index_output_1


class Index516(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=0, stop=48, stride=1)
        return index_output_1


class Index517(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=96, stop=144, stride=1)
        return index_output_1


class Index518(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=729, stride=1)
        return index_output_1


class Index519(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=729, stop=732, stride=1)
        return index_output_1


class Index520(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=1, stop=197, stride=1)
        return index_output_1


class Index521(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=-1, stop=25, stride=1)
        return index_output_1


class Index522(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=-1, stop=34, stride=1)
        return index_output_1


class Index523(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=0, stop=128, stride=2)
        return index_output_1


class Index524(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=1, stop=128, stride=2)
        return index_output_1


class Index525(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=88, stride=1)
        return index_output_1


class Index526(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=88, stop=132, stride=1)
        return index_output_1


class Index527(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=132, stop=176, stride=1)
        return index_output_1


class Index528(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=44, stride=1)
        return index_output_1


class Index529(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=44, stop=88, stride=1)
        return index_output_1


class Index530(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=88, stop=176, stride=1)
        return index_output_1


class Index531(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=64, stop=192, stride=1)
        return index_output_1


class Index532(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=192, stop=448, stride=1)
        return index_output_1


class Index533(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=256, stop=384, stride=1)
        return index_output_1


class Index534(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=384, stop=448, stride=1)
        return index_output_1


class Index535(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=256, stride=1)
        return index_output_1


class Index536(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=0, stop=256, stride=1)
        return index_output_1


class Index537(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-5, start=0, stop=1, stride=1)
        return index_output_1


class Index538(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-5, start=1, stop=2, stride=1)
        return index_output_1


class Index539(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-5, start=2, stop=3, stride=1)
        return index_output_1


class Index540(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=768, stop=1024, stride=1)
        return index_output_1


class Index541(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=1536, stop=1792, stride=1)
        return index_output_1


class Index542(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=2304, stop=2560, stride=1)
        return index_output_1


class Index543(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=512, stop=768, stride=1)
        return index_output_1


class Index544(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=1280, stop=1536, stride=1)
        return index_output_1


class Index545(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=2048, stop=2304, stride=1)
        return index_output_1


class Index546(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=2816, stop=3072, stride=1)
        return index_output_1


class Index547(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=256, stop=512, stride=1)
        return index_output_1


class Index548(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=1024, stop=1280, stride=1)
        return index_output_1


class Index549(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=1792, stop=2048, stride=1)
        return index_output_1


class Index550(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=2560, stop=2816, stride=1)
        return index_output_1


class Index551(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=1, stop=32, stride=2)
        return index_output_1


class Index552(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=0, stop=32, stride=2)
        return index_output_1


class Index553(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=3, stop=56, stride=1)
        return index_output_1


class Index554(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=3, stride=1)
        return index_output_1


class Index555(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=3, stop=56, stride=1)
        return index_output_1


class Index556(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=3, stride=1)
        return index_output_1


class Index557(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=53, stop=56, stride=1)
        return index_output_1


class Index558(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=53, stride=1)
        return index_output_1


class Index559(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=53, stop=56, stride=1)
        return index_output_1


class Index560(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=53, stride=1)
        return index_output_1


class Index561(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=56, stride=2)
        return index_output_1


class Index562(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=1, stop=56, stride=2)
        return index_output_1


class Index563(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=56, stride=2)
        return index_output_1


class Index564(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=1, stop=56, stride=2)
        return index_output_1


class Index565(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=3, stop=28, stride=1)
        return index_output_1


class Index566(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=3, stop=28, stride=1)
        return index_output_1


class Index567(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=25, stop=28, stride=1)
        return index_output_1


class Index568(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=25, stride=1)
        return index_output_1


class Index569(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=25, stop=28, stride=1)
        return index_output_1


class Index570(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=25, stride=1)
        return index_output_1


class Index571(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=28, stride=2)
        return index_output_1


class Index572(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=1, stop=28, stride=2)
        return index_output_1


class Index573(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=28, stride=2)
        return index_output_1


class Index574(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=1, stop=28, stride=2)
        return index_output_1


class Index575(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=3, stop=14, stride=1)
        return index_output_1


class Index576(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=3, stop=14, stride=1)
        return index_output_1


class Index577(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=11, stop=14, stride=1)
        return index_output_1


class Index578(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=11, stride=1)
        return index_output_1


class Index579(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=11, stop=14, stride=1)
        return index_output_1


class Index580(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=11, stride=1)
        return index_output_1


class Index581(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=14, stride=2)
        return index_output_1


class Index582(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=1, stop=14, stride=2)
        return index_output_1


class Index583(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=14, stride=2)
        return index_output_1


class Index584(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=1, stop=14, stride=2)
        return index_output_1


class Index585(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=2, stop=258, stride=1)
        return index_output_1


class Index586(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=40, stride=1)
        return index_output_1


class Index587(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=40, stop=120, stride=1)
        return index_output_1


class Index588(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=120, stop=280, stride=1)
        return index_output_1


class Index589(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=160, stop=240, stride=1)
        return index_output_1


class Index590(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=240, stop=280, stride=1)
        return index_output_1


class Index591(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=80, stop=120, stride=1)
        return index_output_1


class Index592(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=120, stop=160, stride=1)
        return index_output_1


class Index593(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=40, stop=80, stride=1)
        return index_output_1


class Index594(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=416, stride=2)
        return index_output_1


class Index595(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=1, stop=416, stride=2)
        return index_output_1


class Index596(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=0, stop=416, stride=2)
        return index_output_1


class Index597(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=1, stop=416, stride=2)
        return index_output_1


class Index598(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=0, stop=7, stride=1)
        return index_output_1


class Index599(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=32, stop=96, stride=1)
        return index_output_1


class Index600(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=96, stop=224, stride=1)
        return index_output_1


class Index601(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=128, stop=192, stride=1)
        return index_output_1


class Index602(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=192, stop=224, stride=1)
        return index_output_1


class Index603(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=96, stop=128, stride=1)
        return index_output_1


class Index604(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=-24, stop=96, stride=1)
        return index_output_1


class Index605(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=0, stop=72, stride=1)
        return index_output_1


class Index606(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=4, stop=64, stride=1)
        return index_output_1


class Index607(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=4, stride=1)
        return index_output_1


class Index608(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=4, stop=64, stride=1)
        return index_output_1


class Index609(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=4, stride=1)
        return index_output_1


class Index610(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=-4, stop=64, stride=1)
        return index_output_1


class Index611(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=-4, stride=1)
        return index_output_1


class Index612(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=-4, stop=64, stride=1)
        return index_output_1


class Index613(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=-4, stride=1)
        return index_output_1


class Index614(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=64, stride=2)
        return index_output_1


class Index615(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=1, stop=64, stride=2)
        return index_output_1


class Index616(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=60, stop=64, stride=1)
        return index_output_1


class Index617(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=60, stride=1)
        return index_output_1


class Index618(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=60, stop=64, stride=1)
        return index_output_1


class Index619(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=60, stride=1)
        return index_output_1


class Index620(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=64, stride=2)
        return index_output_1


class Index621(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=1, stop=64, stride=2)
        return index_output_1


class Index622(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=4, stop=32, stride=1)
        return index_output_1


class Index623(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=4, stop=32, stride=1)
        return index_output_1


class Index624(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=-4, stop=32, stride=1)
        return index_output_1


class Index625(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=-4, stop=32, stride=1)
        return index_output_1


class Index626(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=32, stride=2)
        return index_output_1


class Index627(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=1, stop=32, stride=2)
        return index_output_1


class Index628(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=28, stop=32, stride=1)
        return index_output_1


class Index629(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=28, stride=1)
        return index_output_1


class Index630(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=28, stop=32, stride=1)
        return index_output_1


class Index631(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=28, stride=1)
        return index_output_1


class Index632(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=32, stride=2)
        return index_output_1


class Index633(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=1, stop=32, stride=2)
        return index_output_1


class Index634(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=4, stop=16, stride=1)
        return index_output_1


class Index635(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=4, stop=16, stride=1)
        return index_output_1


class Index636(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=-4, stop=16, stride=1)
        return index_output_1


class Index637(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=-4, stop=16, stride=1)
        return index_output_1


class Index638(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=16, stride=2)
        return index_output_1


class Index639(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=1, stop=16, stride=2)
        return index_output_1


class Index640(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=12, stop=16, stride=1)
        return index_output_1


class Index641(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=12, stride=1)
        return index_output_1


class Index642(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=12, stop=16, stride=1)
        return index_output_1


class Index643(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=12, stride=1)
        return index_output_1


class Index644(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=16, stride=2)
        return index_output_1


class Index645(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=1, stop=16, stride=2)
        return index_output_1


class Index646(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=18, stride=1)
        return index_output_1


class Index647(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=18, stop=54, stride=1)
        return index_output_1


class Index648(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=54, stop=126, stride=1)
        return index_output_1


class Index649(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=72, stride=1)
        return index_output_1


class Index650(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=72, stop=108, stride=1)
        return index_output_1


class Index651(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=108, stop=126, stride=1)
        return index_output_1


class Index652(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=36, stride=1)
        return index_output_1


class Index653(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=36, stop=54, stride=1)
        return index_output_1


class Index654(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=54, stop=72, stride=1)
        return index_output_1


class Index655(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=18, stop=36, stride=1)
        return index_output_1


class Index656(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=36, stop=72, stride=1)
        return index_output_1


class Index657(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=192, stop=256, stride=1)
        return index_output_1


class Index658(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=2048, stride=1)
        return index_output_1


class Index659(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=2048, stop=4096, stride=1)
        return index_output_1


class Index660(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=64, stop=80, stride=1)
        return index_output_1


class Index661(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=80, stop=96, stride=1)
        return index_output_1


class Index662(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=256, stop=512, stride=1)
        return index_output_1


class Index663(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=48, stride=1)
        return index_output_1


class Index664(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=48, stop=144, stride=1)
        return index_output_1


class Index665(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=144, stop=336, stride=1)
        return index_output_1


class Index666(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=288, stop=336, stride=1)
        return index_output_1


class Index667(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=96, stop=144, stride=1)
        return index_output_1


class Index668(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=144, stop=192, stride=1)
        return index_output_1


class Index669(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=48, stop=96, stride=1)
        return index_output_1


class Index670(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=96, stop=192, stride=1)
        return index_output_1


class Index671(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=1, stop=-100, stride=1)
        return index_output_1


class Index672(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=-100, stop=4251, stride=1)
        return index_output_1


class Index673(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=-100, stop=1445, stride=1)
        return index_output_1


class Index674(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=128, stop=256, stride=1)
        return index_output_1


class Index675(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=30, stride=1)
        return index_output_1


class Index676(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=30, stop=90, stride=1)
        return index_output_1


class Index677(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=90, stop=210, stride=1)
        return index_output_1


class Index678(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=120, stride=1)
        return index_output_1


class Index679(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=120, stop=180, stride=1)
        return index_output_1


class Index680(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=180, stop=210, stride=1)
        return index_output_1


class Index681(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=60, stop=90, stride=1)
        return index_output_1


class Index682(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=90, stop=120, stride=1)
        return index_output_1


class Index683(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=30, stop=60, stride=1)
        return index_output_1


class Index684(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=60, stop=120, stride=1)
        return index_output_1


class Index685(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=255, stop=256, stride=1)
        return index_output_1


class Index686(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=-2, stride=1)
        return index_output_1


class Index687(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=-2, stop=-1, stride=1)
        return index_output_1


class Index688(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=72, stop=73, stride=1)
        return index_output_1


class Index689(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=44, stop=132, stride=1)
        return index_output_1


class Index690(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=132, stop=308, stride=1)
        return index_output_1


class Index691(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=0, stop=176, stride=1)
        return index_output_1


class Index692(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=176, stop=264, stride=1)
        return index_output_1


class Index693(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=264, stop=308, stride=1)
        return index_output_1


class Index694(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=4096, stride=1)
        return index_output_1


class Index695(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=4096, stop=8192, stride=1)
        return index_output_1


class Index696(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=128, stride=1)
        return index_output_1


class Index697(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=128, stop=144, stride=1)
        return index_output_1


class Index698(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=144, stop=160, stride=1)
        return index_output_1


class Index699(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=5120, stride=1)
        return index_output_1


class Index700(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=5120, stop=10240, stride=1)
        return index_output_1


class Index701(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=160, stride=1)
        return index_output_1


class Index702(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=160, stop=176, stride=1)
        return index_output_1


class Index703(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=176, stop=192, stride=1)
        return index_output_1


class Index704(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=0, stop=3072, stride=1)
        return index_output_1


class Index705(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=3072, stop=6144, stride=1)
        return index_output_1


class Index706(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=6144, stop=9216, stride=1)
        return index_output_1


class Index707(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=8192, stop=16384, stride=1)
        return index_output_1


class Index708(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=8192, stride=1)
        return index_output_1


class Index709(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=0, stop=5120, stride=1)
        return index_output_1


class Index710(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=5120, stop=6400, stride=1)
        return index_output_1


class Index711(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=6400, stop=7680, stride=1)
        return index_output_1


class Index712(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=17920, stop=35840, stride=1)
        return index_output_1


class Index713(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-2, start=0, stop=17920, stride=1)
        return index_output_1


class Index714(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=8192, stop=16384, stride=1)
        return index_output_1


class Index715(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=0, stop=8192, stride=1)
        return index_output_1


class Index716(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=512, stop=768, stride=1)
        return index_output_1


class Index717(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=768, stop=1024, stride=1)
        return index_output_1


class Index718(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=1024, stop=1280, stride=1)
        return index_output_1


class Index719(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=1280, stop=1536, stride=1)
        return index_output_1


class Index720(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-3, start=1536, stop=1792, stride=1)
        return index_output_1


class Index721(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=0, stop=4, stride=1)
        return index_output_1


class Index722(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, index_input_0):
        index_output_1 = forge.op.Index("", index_input_0, dim=-1, start=4, stop=8, stride=1)
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
                "pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
                "pt_albert_xlarge_v2_token_cls_hf",
                "pt_albert_xxlarge_v2_token_cls_hf",
                "pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder",
                "pt_albert_base_v2_mlm_hf",
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
                "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf",
                "pt_albert_xlarge_v1_mlm_hf",
                "pt_albert_xlarge_v2_mlm_hf",
                "pt_albert_xxlarge_v1_mlm_hf",
                "pt_bert_bert_base_uncased_mlm_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder",
                "pt_albert_large_v1_mlm_hf",
                "pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader",
                "pt_albert_base_v1_token_cls_hf",
                "pt_albert_base_v2_token_cls_hf",
                "pt_albert_large_v1_token_cls_hf",
                "pt_albert_large_v2_mlm_hf",
                "pt_albert_large_v2_token_cls_hf",
                "pt_albert_xlarge_v1_token_cls_hf",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_albert_base_v1_mlm_hf",
                "pt_distilbert_distilbert_base_uncased_mlm_hf",
                "pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf",
                "pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_distilbert_distilbert_base_multilingual_cased_mlm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index1,
        [((1, 512), torch.int64)],
        {
            "model_names": ["pd_blip_text_salesforce_blip_image_captioning_base_text_enc_padlenlp"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "8", "stride": "1"},
        },
    ),
    (
        Index2,
        [((1, 512), torch.int64)],
        {
            "model_names": ["pt_albert_twmkn9_albert_base_v2_squad2_qa_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "14", "stride": "1"},
        },
    ),
    (
        Index3,
        [((1, 512), torch.int64)],
        {
            "model_names": ["pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "9", "stride": "1"},
        },
    ),
    (
        Index4,
        [((1, 512), torch.int64)],
        {
            "model_names": [
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                "pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf",
                "pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "384", "stride": "1"},
        },
    ),
    (
        Index5,
        [((1, 512), torch.int64)],
        {
            "model_names": ["pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "16", "stride": "1"},
        },
    ),
    (
        Index6,
        [((1, 128, 768), torch.float32)],
        {
            "model_names": [
                "pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf",
                "pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder",
                "pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index6,
        [((2, 768), torch.float32)],
        {
            "model_names": [
                "pt_opt_facebook_opt_125m_qa_hf",
                "pt_albert_twmkn9_albert_base_v2_squad2_qa_hf",
                "pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index7,
        [((2, 768), torch.float32)],
        {
            "model_names": [
                "pt_opt_facebook_opt_125m_qa_hf",
                "pt_albert_twmkn9_albert_base_v2_squad2_qa_hf",
                "pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index8,
        [((2,), torch.float32)],
        {
            "model_names": [
                "pt_opt_facebook_opt_125m_qa_hf",
                "pt_opt_facebook_opt_350m_qa_hf",
                "pt_albert_twmkn9_albert_base_v2_squad2_qa_hf",
                "pt_opt_facebook_opt_1_3b_qa_hf",
                "pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader",
                "onnx_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf",
                "pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index9,
        [((2,), torch.float32)],
        {
            "model_names": [
                "pt_opt_facebook_opt_125m_qa_hf",
                "pt_opt_facebook_opt_350m_qa_hf",
                "pt_albert_twmkn9_albert_base_v2_squad2_qa_hf",
                "pt_opt_facebook_opt_1_3b_qa_hf",
                "pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader",
                "onnx_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf",
                "pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index10,
        [((1, 32, 7, 64), torch.float32)],
        {
            "model_names": ["pt_phi1_microsoft_phi_1_clm_hf", "pt_phi_1_5_microsoft_phi_1_5_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index11,
        [((1, 32, 7, 64), torch.float32)],
        {
            "model_names": ["pt_phi1_microsoft_phi_1_clm_hf", "pt_phi_1_5_microsoft_phi_1_5_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "32", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index12,
        [((1, 32, 7, 32), torch.float32)],
        {
            "model_names": ["pt_phi1_microsoft_phi_1_clm_hf", "pt_phi_1_5_microsoft_phi_1_5_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "16", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index5,
        [((1, 32, 7, 32), torch.float32)],
        {
            "model_names": ["pt_phi1_microsoft_phi_1_clm_hf", "pt_phi_1_5_microsoft_phi_1_5_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "16", "stride": "1"},
        },
    ),
    (
        Index11,
        [((1, 16, 29, 64), torch.float32)],
        {
            "model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_chat_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "32", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index10,
        [((1, 16, 29, 64), torch.float32)],
        {
            "model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_chat_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index6,
        [((1, 11, 768), torch.float32)],
        {
            "model_names": ["pd_bert_chinese_roberta_base_seq_cls_padlenlp"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index13,
        [((2, 1, 9), torch.float32)],
        {
            "model_names": ["pd_ernie_1_0_qa_padlenlp", "pd_bert_bert_base_uncased_qa_padlenlp"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index14,
        [((2, 1, 9), torch.float32)],
        {
            "model_names": ["pd_ernie_1_0_qa_padlenlp", "pd_bert_bert_base_uncased_qa_padlenlp"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index15,
        [((1, 16, 588, 128), torch.float32)],
        {
            "model_names": ["pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "64", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index16,
        [((1, 16, 588, 128), torch.float32)],
        {
            "model_names": ["pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index17,
        [((2304, 768), torch.float32)],
        {
            "model_names": [
                "pt_gpt_gpt2_text_gen_hf",
                "pt_gpt_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "768", "stride": "1"},
        },
    ),
    (
        Index18,
        [((2304, 768), torch.float32)],
        {
            "model_names": [
                "pt_gpt_gpt2_text_gen_hf",
                "pt_gpt_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "768", "stop": "1536", "stride": "1"},
        },
    ),
    (
        Index19,
        [((2304, 768), torch.float32)],
        {
            "model_names": [
                "pt_gpt_gpt2_text_gen_hf",
                "pt_gpt_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "1536", "stop": "2304", "stride": "1"},
        },
    ),
    (
        Index20,
        [((2304,), torch.float32)],
        {
            "model_names": [
                "pt_gpt_gpt2_text_gen_hf",
                "pt_gpt_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "768", "stride": "1"},
        },
    ),
    (
        Index21,
        [((2304,), torch.float32)],
        {
            "model_names": [
                "pt_gpt_gpt2_text_gen_hf",
                "pt_gpt_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "768", "stop": "1536", "stride": "1"},
        },
    ),
    (
        Index22,
        [((2304,), torch.float32)],
        {
            "model_names": [
                "pt_gpt_gpt2_text_gen_hf",
                "pt_gpt_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "1536", "stop": "2304", "stride": "1"},
        },
    ),
    (
        Index23,
        [((1, 64, 28, 28), torch.bfloat16)],
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
        Index24,
        [((1, 64, 28, 28), torch.bfloat16)],
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
        Index25,
        [((1, 64, 28, 28), torch.bfloat16)],
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
        Index26,
        [((1, 112, 7, 7), torch.bfloat16)],
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
        Index27,
        [((1, 112, 7, 7), torch.bfloat16)],
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
        Index28,
        [((1, 112, 7, 7), torch.bfloat16)],
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
        Index29,
        [((1, 224, 35, 35), torch.bfloat16)],
        {
            "model_names": [
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_inception_v4_img_cls_osmr",
                "pt_inception_inception_v4_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "96", "stride": "1"},
        },
    ),
    (
        Index30,
        [((1, 224, 35, 35), torch.bfloat16)],
        {
            "model_names": [
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_inception_v4_img_cls_osmr",
                "pt_inception_inception_v4_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "96", "stop": "160", "stride": "1"},
        },
    ),
    (
        Index31,
        [((1, 224, 35, 35), torch.bfloat16)],
        {
            "model_names": [
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_inception_v4_img_cls_osmr",
                "pt_inception_inception_v4_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "160", "stop": "224", "stride": "1"},
        },
    ),
    (
        Index32,
        [((1, 768, 17, 17), torch.bfloat16)],
        {
            "model_names": [
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_inception_v4_img_cls_osmr",
                "pt_inception_inception_v4_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "384", "stride": "1"},
        },
    ),
    (
        Index33,
        [((1, 768, 17, 17), torch.bfloat16)],
        {
            "model_names": [
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_inception_v4_img_cls_osmr",
                "pt_inception_inception_v4_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "384", "stop": "576", "stride": "1"},
        },
    ),
    (
        Index34,
        [((1, 768, 17, 17), torch.bfloat16)],
        {
            "model_names": [
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_inception_v4_img_cls_osmr",
                "pt_inception_inception_v4_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "576", "stop": "768", "stride": "1"},
        },
    ),
    (
        Index35,
        [((1, 1024, 8, 8), torch.bfloat16)],
        {
            "model_names": [
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_inception_v4_img_cls_osmr",
                "pt_inception_inception_v4_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index36,
        [((1, 1024, 8, 8), torch.bfloat16)],
        {
            "model_names": [
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_inception_v4_img_cls_osmr",
                "pt_inception_inception_v4_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "256", "stop": "640", "stride": "1"},
        },
    ),
    (
        Index37,
        [((1, 1024, 8, 8), torch.bfloat16)],
        {
            "model_names": [
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_inception_v4_img_cls_osmr",
                "pt_inception_inception_v4_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "640", "stop": "1024", "stride": "1"},
        },
    ),
    (
        Index38,
        [((1, 6144, 6), torch.float32)],
        {
            "model_names": ["pt_mamba_state_spaces_mamba_790m_hf_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "3072", "stride": "1"},
        },
    ),
    (
        Index39,
        [((1, 6144, 6), torch.float32)],
        {
            "model_names": ["pt_mamba_state_spaces_mamba_790m_hf_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "3072", "stop": "6144", "stride": "1"},
        },
    ),
    (
        Index40,
        [((1, 3072, 9), torch.float32)],
        {
            "model_names": ["pt_mamba_state_spaces_mamba_790m_hf_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "6", "stride": "1"},
        },
    ),
    (
        Index41,
        [((128, 3072), torch.float32)],
        {
            "model_names": ["pt_mamba_state_spaces_mamba_790m_hf_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "96", "stride": "1"},
        },
    ),
    (
        Index42,
        [((128, 3072), torch.float32)],
        {
            "model_names": ["pt_mamba_state_spaces_mamba_790m_hf_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "96", "stop": "112", "stride": "1"},
        },
    ),
    (
        Index43,
        [((128, 3072), torch.float32)],
        {
            "model_names": ["pt_mamba_state_spaces_mamba_790m_hf_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "112", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index6,
        [((1, 3072, 6, 16), torch.float32)],
        {
            "model_names": ["pt_mamba_state_spaces_mamba_790m_hf_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index7,
        [((1, 3072, 6, 16), torch.float32)],
        {
            "model_names": ["pt_mamba_state_spaces_mamba_790m_hf_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index44,
        [((1, 3072, 6, 16), torch.float32)],
        {
            "model_names": ["pt_mamba_state_spaces_mamba_790m_hf_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "2", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index45,
        [((1, 3072, 6, 16), torch.float32)],
        {
            "model_names": ["pt_mamba_state_spaces_mamba_790m_hf_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "3", "stop": "4", "stride": "1"},
        },
    ),
    (
        Index46,
        [((1, 3072, 6, 16), torch.float32)],
        {
            "model_names": ["pt_mamba_state_spaces_mamba_790m_hf_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "4", "stop": "5", "stride": "1"},
        },
    ),
    (
        Index47,
        [((1, 3072, 6, 16), torch.float32)],
        {
            "model_names": ["pt_mamba_state_spaces_mamba_790m_hf_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "5", "stop": "6", "stride": "1"},
        },
    ),
    (
        Index6,
        [((1, 6, 16), torch.float32)],
        {
            "model_names": [
                "pt_mamba_state_spaces_mamba_790m_hf_clm_hf",
                "pt_mamba_state_spaces_mamba_370m_hf_clm_hf",
                "pt_mamba_state_spaces_mamba_1_4b_hf_clm_hf",
                "pt_mamba_state_spaces_mamba_2_8b_hf_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index7,
        [((1, 6, 16), torch.float32)],
        {
            "model_names": [
                "pt_mamba_state_spaces_mamba_790m_hf_clm_hf",
                "pt_mamba_state_spaces_mamba_370m_hf_clm_hf",
                "pt_mamba_state_spaces_mamba_1_4b_hf_clm_hf",
                "pt_mamba_state_spaces_mamba_2_8b_hf_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index44,
        [((1, 6, 16), torch.float32)],
        {
            "model_names": [
                "pt_mamba_state_spaces_mamba_790m_hf_clm_hf",
                "pt_mamba_state_spaces_mamba_370m_hf_clm_hf",
                "pt_mamba_state_spaces_mamba_1_4b_hf_clm_hf",
                "pt_mamba_state_spaces_mamba_2_8b_hf_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "2", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index45,
        [((1, 6, 16), torch.float32)],
        {
            "model_names": [
                "pt_mamba_state_spaces_mamba_790m_hf_clm_hf",
                "pt_mamba_state_spaces_mamba_370m_hf_clm_hf",
                "pt_mamba_state_spaces_mamba_1_4b_hf_clm_hf",
                "pt_mamba_state_spaces_mamba_2_8b_hf_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "3", "stop": "4", "stride": "1"},
        },
    ),
    (
        Index46,
        [((1, 6, 16), torch.float32)],
        {
            "model_names": [
                "pt_mamba_state_spaces_mamba_790m_hf_clm_hf",
                "pt_mamba_state_spaces_mamba_370m_hf_clm_hf",
                "pt_mamba_state_spaces_mamba_1_4b_hf_clm_hf",
                "pt_mamba_state_spaces_mamba_2_8b_hf_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "4", "stop": "5", "stride": "1"},
        },
    ),
    (
        Index47,
        [((1, 6, 16), torch.float32)],
        {
            "model_names": [
                "pt_mamba_state_spaces_mamba_790m_hf_clm_hf",
                "pt_mamba_state_spaces_mamba_370m_hf_clm_hf",
                "pt_mamba_state_spaces_mamba_1_4b_hf_clm_hf",
                "pt_mamba_state_spaces_mamba_2_8b_hf_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "5", "stop": "6", "stride": "1"},
        },
    ),
    (
        Index48,
        [((1, 1, 1024, 72), torch.float32)],
        {
            "model_names": [
                "pt_nbeats_seasionality_basis_clm_hf",
                "pt_nbeats_generic_basis_clm_hf",
                "pt_nbeats_trend_basis_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "-1", "stop": "72", "stride": "1"},
        },
    ),
    (
        Index49,
        [((1024, 48), torch.float32)],
        {
            "model_names": ["pt_nbeats_seasionality_basis_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "12", "stop": "24", "stride": "1"},
        },
    ),
    (
        Index50,
        [((1024, 48), torch.float32)],
        {
            "model_names": ["pt_nbeats_seasionality_basis_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "12", "stride": "1"},
        },
    ),
    (
        Index51,
        [((1024, 48), torch.float32)],
        {
            "model_names": ["pt_nbeats_seasionality_basis_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "36", "stop": "48", "stride": "1"},
        },
    ),
    (
        Index52,
        [((1024, 48), torch.float32)],
        {
            "model_names": ["pt_nbeats_seasionality_basis_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "24", "stop": "36", "stride": "1"},
        },
    ),
    (
        Index6,
        [((2, 512), torch.float32)],
        {
            "model_names": ["pt_opt_facebook_opt_350m_qa_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index7,
        [((2, 512), torch.float32)],
        {
            "model_names": ["pt_opt_facebook_opt_350m_qa_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index10,
        [((1, 32, 256, 80), torch.float32)],
        {
            "model_names": ["pt_phi2_microsoft_phi_2_clm_hf", "pt_phi2_microsoft_phi_2_pytdml_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index53,
        [((1, 32, 256, 80), torch.float32)],
        {
            "model_names": ["pt_phi2_microsoft_phi_2_clm_hf", "pt_phi2_microsoft_phi_2_pytdml_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "32", "stop": "80", "stride": "1"},
        },
    ),
    (
        Index12,
        [((1, 32, 256, 32), torch.float32)],
        {
            "model_names": [
                "pt_phi2_microsoft_phi_2_clm_hf",
                "pt_phi1_microsoft_phi_1_seq_cls_hf",
                "pt_phi2_microsoft_phi_2_pytdml_clm_hf",
                "pt_phi_1_5_microsoft_phi_1_5_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "16", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index5,
        [((1, 32, 256, 32), torch.float32)],
        {
            "model_names": [
                "pt_phi2_microsoft_phi_2_clm_hf",
                "pt_phi1_microsoft_phi_1_seq_cls_hf",
                "pt_phi2_microsoft_phi_2_pytdml_clm_hf",
                "pt_phi_1_5_microsoft_phi_1_5_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "16", "stride": "1"},
        },
    ),
    (
        Index11,
        [((1, 14, 39, 64), torch.float32)],
        {
            "model_names": ["pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "32", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index10,
        [((1, 14, 39, 64), torch.float32)],
        {
            "model_names": ["pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index11,
        [((1, 2, 39, 64), torch.float32)],
        {
            "model_names": ["pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "32", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index10,
        [((1, 2, 39, 64), torch.float32)],
        {
            "model_names": ["pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index15,
        [((1, 12, 29, 128), torch.float32)],
        {
            "model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "64", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index16,
        [((1, 12, 29, 128), torch.float32)],
        {
            "model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index15,
        [((1, 2, 29, 128), torch.float32)],
        {
            "model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf", "pt_qwen_v2_qwen_qwen2_5_3b_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "64", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index16,
        [((1, 2, 29, 128), torch.float32)],
        {
            "model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf", "pt_qwen_v2_qwen_qwen2_5_3b_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index15,
        [((1, 12, 39, 128), torch.float32)],
        {
            "model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "64", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index16,
        [((1, 12, 39, 128), torch.float32)],
        {
            "model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index15,
        [((1, 2, 39, 128), torch.float32)],
        {
            "model_names": [
                "pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf",
                "pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "64", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index16,
        [((1, 2, 39, 128), torch.float32)],
        {
            "model_names": [
                "pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf",
                "pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index54,
        [((3, 50, 1, 1024), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_l_32_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index55,
        [((3, 50, 1, 1024), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_l_32_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index56,
        [((3, 50, 1, 1024), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_l_32_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "2", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index6,
        [((1, 50, 1024), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_l_32_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index57,
        [((1, 5880, 4), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolo_v6_yolov6n_obj_det_torchhub",
                "pt_yolo_v6_yolov6m_obj_det_torchhub",
                "pt_yolo_v6_yolov6s_obj_det_torchhub",
                "pt_yolo_v6_yolov6l_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index58,
        [((1, 5880, 4), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolo_v6_yolov6n_obj_det_torchhub",
                "pt_yolo_v6_yolov6m_obj_det_torchhub",
                "pt_yolo_v6_yolov6s_obj_det_torchhub",
                "pt_yolo_v6_yolov6l_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "2", "stop": "4", "stride": "1"},
        },
    ),
    (
        Index59,
        [((1, 160, 160, 160), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolov10_yolov10x_obj_det_github",
                "pt_yolov10_yolov10n_obj_det_github",
                "pt_yolov8_yolov8x_obj_det_github",
                "pt_yolov8_yolov8n_obj_det_github",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "80", "stride": "1"},
        },
    ),
    (
        Index60,
        [((1, 160, 160, 160), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolov10_yolov10x_obj_det_github",
                "pt_yolov10_yolov10n_obj_det_github",
                "pt_yolov8_yolov8x_obj_det_github",
                "pt_yolov8_yolov8n_obj_det_github",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "80", "stop": "160", "stride": "1"},
        },
    ),
    (
        Index61,
        [((1, 320, 80, 80), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolov10_yolov10x_obj_det_github",
                "pt_yolov10_yolov10n_obj_det_github",
                "pt_yolov8_yolov8x_obj_det_github",
                "pt_yolov8_yolov8n_obj_det_github",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "160", "stride": "1"},
        },
    ),
    (
        Index62,
        [((1, 320, 80, 80), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolov10_yolov10x_obj_det_github",
                "pt_yolov10_yolov10n_obj_det_github",
                "pt_yolov8_yolov8x_obj_det_github",
                "pt_yolov8_yolov8n_obj_det_github",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "160", "stop": "320", "stride": "1"},
        },
    ),
    (
        Index63,
        [((1, 640, 40, 40), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolov10_yolov10x_obj_det_github",
                "pt_yolov10_yolov10n_obj_det_github",
                "pt_yolov8_yolov8x_obj_det_github",
                "pt_yolov8_yolov8n_obj_det_github",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "320", "stride": "1"},
        },
    ),
    (
        Index64,
        [((1, 640, 40, 40), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolov10_yolov10x_obj_det_github",
                "pt_yolov10_yolov10n_obj_det_github",
                "pt_yolov8_yolov8x_obj_det_github",
                "pt_yolov8_yolov8n_obj_det_github",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "320", "stop": "640", "stride": "1"},
        },
    ),
    (
        Index63,
        [((1, 640, 20, 20), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolov10_yolov10x_obj_det_github",
                "pt_yolov10_yolov10n_obj_det_github",
                "pt_yolov8_yolov8x_obj_det_github",
                "pt_yolov8_yolov8n_obj_det_github",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "320", "stride": "1"},
        },
    ),
    (
        Index64,
        [((1, 640, 20, 20), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolov10_yolov10x_obj_det_github",
                "pt_yolov10_yolov10n_obj_det_github",
                "pt_yolov8_yolov8x_obj_det_github",
                "pt_yolov8_yolov8n_obj_det_github",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "320", "stop": "640", "stride": "1"},
        },
    ),
    (
        Index65,
        [((1, 5, 128, 400), torch.bfloat16)],
        {
            "model_names": ["pt_yolov10_yolov10x_obj_det_github", "pt_yolov10_yolov10n_obj_det_github"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "64", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index66,
        [((1, 5, 128, 400), torch.bfloat16)],
        {
            "model_names": ["pt_yolov10_yolov10x_obj_det_github", "pt_yolov10_yolov10n_obj_det_github"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index67,
        [((1, 5, 128, 400), torch.bfloat16)],
        {
            "model_names": ["pt_yolov10_yolov10x_obj_det_github", "pt_yolov10_yolov10n_obj_det_github"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "32", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index68,
        [((1, 144, 8400), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolov10_yolov10x_obj_det_github",
                "pt_yolov10_yolov10n_obj_det_github",
                "pt_yolov8_yolov8x_obj_det_github",
                "pt_yolov8_yolov8n_obj_det_github",
                "pt_yolov9_default_obj_det_github",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index69,
        [((1, 144, 8400), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolov10_yolov10x_obj_det_github",
                "pt_yolov10_yolov10n_obj_det_github",
                "pt_yolov8_yolov8x_obj_det_github",
                "pt_yolov8_yolov8n_obj_det_github",
                "pt_yolov9_default_obj_det_github",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "64", "stop": "144", "stride": "1"},
        },
    ),
    (
        Index70,
        [((1, 4, 8400), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolov10_yolov10x_obj_det_github",
                "pt_yolov10_yolov10n_obj_det_github",
                "pt_yolov8_yolov8x_obj_det_github",
                "pt_yolov8_yolov8n_obj_det_github",
                "pt_yolov9_default_obj_det_github",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index71,
        [((1, 4, 8400), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolov10_yolov10x_obj_det_github",
                "pt_yolov10_yolov10n_obj_det_github",
                "pt_yolov8_yolov8x_obj_det_github",
                "pt_yolov8_yolov8n_obj_det_github",
                "pt_yolov9_default_obj_det_github",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "2", "stop": "4", "stride": "1"},
        },
    ),
    (
        Index72,
        [((1, 3, 640, 640), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolox_yolox_x_obj_det_torchhub",
                "pt_yolox_yolox_l_obj_det_torchhub",
                "pt_yolox_yolox_m_obj_det_torchhub",
                "pt_yolox_yolox_s_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "640", "stride": "2"},
        },
    ),
    (
        Index73,
        [((1, 3, 640, 640), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolox_yolox_x_obj_det_torchhub",
                "pt_yolox_yolox_l_obj_det_torchhub",
                "pt_yolox_yolox_m_obj_det_torchhub",
                "pt_yolox_yolox_s_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "1", "stop": "640", "stride": "2"},
        },
    ),
    (
        Index74,
        [((1, 3, 320, 640), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolox_yolox_x_obj_det_torchhub",
                "pt_yolox_yolox_l_obj_det_torchhub",
                "pt_yolox_yolox_m_obj_det_torchhub",
                "pt_yolox_yolox_s_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "640", "stride": "2"},
        },
    ),
    (
        Index75,
        [((1, 3, 320, 640), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolox_yolox_x_obj_det_torchhub",
                "pt_yolox_yolox_l_obj_det_torchhub",
                "pt_yolox_yolox_m_obj_det_torchhub",
                "pt_yolox_yolox_s_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "1", "stop": "640", "stride": "2"},
        },
    ),
    (
        Index54,
        [((3, 300, 196, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index55,
        [((3, 300, 196, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index56,
        [((3, 300, 196, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "2", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index54,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index55,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index56,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "2", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index76,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "3", "stop": "4", "stride": "1"},
        },
    ),
    (
        Index77,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "4", "stop": "5", "stride": "1"},
        },
    ),
    (
        Index78,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "5", "stop": "6", "stride": "1"},
        },
    ),
    (
        Index79,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "6", "stop": "7", "stride": "1"},
        },
    ),
    (
        Index80,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "7", "stop": "8", "stride": "1"},
        },
    ),
    (
        Index81,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "8", "stop": "9", "stride": "1"},
        },
    ),
    (
        Index82,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "9", "stop": "10", "stride": "1"},
        },
    ),
    (
        Index83,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "10", "stop": "11", "stride": "1"},
        },
    ),
    (
        Index84,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "11", "stop": "12", "stride": "1"},
        },
    ),
    (
        Index85,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "12", "stop": "13", "stride": "1"},
        },
    ),
    (
        Index86,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "13", "stop": "14", "stride": "1"},
        },
    ),
    (
        Index87,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "14", "stop": "15", "stride": "1"},
        },
    ),
    (
        Index88,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "15", "stop": "16", "stride": "1"},
        },
    ),
    (
        Index89,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "16", "stop": "17", "stride": "1"},
        },
    ),
    (
        Index90,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "17", "stop": "18", "stride": "1"},
        },
    ),
    (
        Index91,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "18", "stop": "19", "stride": "1"},
        },
    ),
    (
        Index92,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "19", "stop": "20", "stride": "1"},
        },
    ),
    (
        Index93,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "20", "stop": "21", "stride": "1"},
        },
    ),
    (
        Index94,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "21", "stop": "22", "stride": "1"},
        },
    ),
    (
        Index95,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "22", "stop": "23", "stride": "1"},
        },
    ),
    (
        Index96,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "23", "stop": "24", "stride": "1"},
        },
    ),
    (
        Index97,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "24", "stop": "25", "stride": "1"},
        },
    ),
    (
        Index98,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "25", "stop": "26", "stride": "1"},
        },
    ),
    (
        Index99,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "26", "stop": "27", "stride": "1"},
        },
    ),
    (
        Index100,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "27", "stop": "28", "stride": "1"},
        },
    ),
    (
        Index101,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "28", "stop": "29", "stride": "1"},
        },
    ),
    (
        Index102,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "29", "stop": "30", "stride": "1"},
        },
    ),
    (
        Index103,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "30", "stop": "31", "stride": "1"},
        },
    ),
    (
        Index104,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "31", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index105,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "32", "stop": "33", "stride": "1"},
        },
    ),
    (
        Index106,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "33", "stop": "34", "stride": "1"},
        },
    ),
    (
        Index107,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "34", "stop": "35", "stride": "1"},
        },
    ),
    (
        Index108,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "35", "stop": "36", "stride": "1"},
        },
    ),
    (
        Index109,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "36", "stop": "37", "stride": "1"},
        },
    ),
    (
        Index110,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "37", "stop": "38", "stride": "1"},
        },
    ),
    (
        Index111,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "38", "stop": "39", "stride": "1"},
        },
    ),
    (
        Index112,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "39", "stop": "40", "stride": "1"},
        },
    ),
    (
        Index113,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "40", "stop": "41", "stride": "1"},
        },
    ),
    (
        Index114,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "41", "stop": "42", "stride": "1"},
        },
    ),
    (
        Index115,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "42", "stop": "43", "stride": "1"},
        },
    ),
    (
        Index116,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "43", "stop": "44", "stride": "1"},
        },
    ),
    (
        Index117,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "44", "stop": "45", "stride": "1"},
        },
    ),
    (
        Index118,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "45", "stop": "46", "stride": "1"},
        },
    ),
    (
        Index119,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "46", "stop": "47", "stride": "1"},
        },
    ),
    (
        Index120,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "47", "stop": "48", "stride": "1"},
        },
    ),
    (
        Index121,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "48", "stop": "49", "stride": "1"},
        },
    ),
    (
        Index122,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "49", "stop": "50", "stride": "1"},
        },
    ),
    (
        Index123,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "50", "stop": "51", "stride": "1"},
        },
    ),
    (
        Index124,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "51", "stop": "52", "stride": "1"},
        },
    ),
    (
        Index125,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "52", "stop": "53", "stride": "1"},
        },
    ),
    (
        Index126,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "53", "stop": "54", "stride": "1"},
        },
    ),
    (
        Index127,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "54", "stop": "55", "stride": "1"},
        },
    ),
    (
        Index128,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "55", "stop": "56", "stride": "1"},
        },
    ),
    (
        Index129,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "56", "stop": "57", "stride": "1"},
        },
    ),
    (
        Index130,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "57", "stop": "58", "stride": "1"},
        },
    ),
    (
        Index131,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "58", "stop": "59", "stride": "1"},
        },
    ),
    (
        Index132,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "59", "stop": "60", "stride": "1"},
        },
    ),
    (
        Index133,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "60", "stop": "61", "stride": "1"},
        },
    ),
    (
        Index134,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "61", "stop": "62", "stride": "1"},
        },
    ),
    (
        Index135,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "62", "stop": "63", "stride": "1"},
        },
    ),
    (
        Index136,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "63", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index137,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "64", "stop": "65", "stride": "1"},
        },
    ),
    (
        Index138,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "65", "stop": "66", "stride": "1"},
        },
    ),
    (
        Index139,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "66", "stop": "67", "stride": "1"},
        },
    ),
    (
        Index140,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "67", "stop": "68", "stride": "1"},
        },
    ),
    (
        Index141,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "68", "stop": "69", "stride": "1"},
        },
    ),
    (
        Index142,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "69", "stop": "70", "stride": "1"},
        },
    ),
    (
        Index143,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "70", "stop": "71", "stride": "1"},
        },
    ),
    (
        Index144,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "71", "stop": "72", "stride": "1"},
        },
    ),
    (
        Index145,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "72", "stop": "73", "stride": "1"},
        },
    ),
    (
        Index146,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "73", "stop": "74", "stride": "1"},
        },
    ),
    (
        Index147,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "74", "stop": "75", "stride": "1"},
        },
    ),
    (
        Index148,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "75", "stop": "76", "stride": "1"},
        },
    ),
    (
        Index149,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "76", "stop": "77", "stride": "1"},
        },
    ),
    (
        Index150,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "77", "stop": "78", "stride": "1"},
        },
    ),
    (
        Index151,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "78", "stop": "79", "stride": "1"},
        },
    ),
    (
        Index152,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "79", "stop": "80", "stride": "1"},
        },
    ),
    (
        Index153,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "80", "stop": "81", "stride": "1"},
        },
    ),
    (
        Index154,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "81", "stop": "82", "stride": "1"},
        },
    ),
    (
        Index155,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "82", "stop": "83", "stride": "1"},
        },
    ),
    (
        Index156,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "83", "stop": "84", "stride": "1"},
        },
    ),
    (
        Index157,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "84", "stop": "85", "stride": "1"},
        },
    ),
    (
        Index158,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "85", "stop": "86", "stride": "1"},
        },
    ),
    (
        Index159,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "86", "stop": "87", "stride": "1"},
        },
    ),
    (
        Index160,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "87", "stop": "88", "stride": "1"},
        },
    ),
    (
        Index161,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "88", "stop": "89", "stride": "1"},
        },
    ),
    (
        Index162,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "89", "stop": "90", "stride": "1"},
        },
    ),
    (
        Index163,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "90", "stop": "91", "stride": "1"},
        },
    ),
    (
        Index164,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "91", "stop": "92", "stride": "1"},
        },
    ),
    (
        Index165,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "92", "stop": "93", "stride": "1"},
        },
    ),
    (
        Index166,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "93", "stop": "94", "stride": "1"},
        },
    ),
    (
        Index167,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "94", "stop": "95", "stride": "1"},
        },
    ),
    (
        Index168,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "95", "stop": "96", "stride": "1"},
        },
    ),
    (
        Index169,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "96", "stop": "97", "stride": "1"},
        },
    ),
    (
        Index170,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "97", "stop": "98", "stride": "1"},
        },
    ),
    (
        Index171,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "98", "stop": "99", "stride": "1"},
        },
    ),
    (
        Index172,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "99", "stop": "100", "stride": "1"},
        },
    ),
    (
        Index173,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "100", "stop": "101", "stride": "1"},
        },
    ),
    (
        Index174,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "101", "stop": "102", "stride": "1"},
        },
    ),
    (
        Index175,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "102", "stop": "103", "stride": "1"},
        },
    ),
    (
        Index176,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "103", "stop": "104", "stride": "1"},
        },
    ),
    (
        Index177,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "104", "stop": "105", "stride": "1"},
        },
    ),
    (
        Index178,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "105", "stop": "106", "stride": "1"},
        },
    ),
    (
        Index179,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "106", "stop": "107", "stride": "1"},
        },
    ),
    (
        Index180,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "107", "stop": "108", "stride": "1"},
        },
    ),
    (
        Index181,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "108", "stop": "109", "stride": "1"},
        },
    ),
    (
        Index182,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "109", "stop": "110", "stride": "1"},
        },
    ),
    (
        Index183,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "110", "stop": "111", "stride": "1"},
        },
    ),
    (
        Index184,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "111", "stop": "112", "stride": "1"},
        },
    ),
    (
        Index185,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "112", "stop": "113", "stride": "1"},
        },
    ),
    (
        Index186,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "113", "stop": "114", "stride": "1"},
        },
    ),
    (
        Index187,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "114", "stop": "115", "stride": "1"},
        },
    ),
    (
        Index188,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "115", "stop": "116", "stride": "1"},
        },
    ),
    (
        Index189,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "116", "stop": "117", "stride": "1"},
        },
    ),
    (
        Index190,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "117", "stop": "118", "stride": "1"},
        },
    ),
    (
        Index191,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "118", "stop": "119", "stride": "1"},
        },
    ),
    (
        Index192,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "119", "stop": "120", "stride": "1"},
        },
    ),
    (
        Index193,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "120", "stop": "121", "stride": "1"},
        },
    ),
    (
        Index194,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "121", "stop": "122", "stride": "1"},
        },
    ),
    (
        Index195,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "122", "stop": "123", "stride": "1"},
        },
    ),
    (
        Index196,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "123", "stop": "124", "stride": "1"},
        },
    ),
    (
        Index197,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "124", "stop": "125", "stride": "1"},
        },
    ),
    (
        Index198,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "125", "stop": "126", "stride": "1"},
        },
    ),
    (
        Index199,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "126", "stop": "127", "stride": "1"},
        },
    ),
    (
        Index200,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "127", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index201,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "128", "stop": "129", "stride": "1"},
        },
    ),
    (
        Index202,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "129", "stop": "130", "stride": "1"},
        },
    ),
    (
        Index203,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "130", "stop": "131", "stride": "1"},
        },
    ),
    (
        Index204,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "131", "stop": "132", "stride": "1"},
        },
    ),
    (
        Index205,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "132", "stop": "133", "stride": "1"},
        },
    ),
    (
        Index206,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "133", "stop": "134", "stride": "1"},
        },
    ),
    (
        Index207,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "134", "stop": "135", "stride": "1"},
        },
    ),
    (
        Index208,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "135", "stop": "136", "stride": "1"},
        },
    ),
    (
        Index209,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "136", "stop": "137", "stride": "1"},
        },
    ),
    (
        Index210,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "137", "stop": "138", "stride": "1"},
        },
    ),
    (
        Index211,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "138", "stop": "139", "stride": "1"},
        },
    ),
    (
        Index212,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "139", "stop": "140", "stride": "1"},
        },
    ),
    (
        Index213,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "140", "stop": "141", "stride": "1"},
        },
    ),
    (
        Index214,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "141", "stop": "142", "stride": "1"},
        },
    ),
    (
        Index215,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "142", "stop": "143", "stride": "1"},
        },
    ),
    (
        Index216,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "143", "stop": "144", "stride": "1"},
        },
    ),
    (
        Index217,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "144", "stop": "145", "stride": "1"},
        },
    ),
    (
        Index218,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "145", "stop": "146", "stride": "1"},
        },
    ),
    (
        Index219,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "146", "stop": "147", "stride": "1"},
        },
    ),
    (
        Index220,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "147", "stop": "148", "stride": "1"},
        },
    ),
    (
        Index221,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "148", "stop": "149", "stride": "1"},
        },
    ),
    (
        Index222,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "149", "stop": "150", "stride": "1"},
        },
    ),
    (
        Index223,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "150", "stop": "151", "stride": "1"},
        },
    ),
    (
        Index224,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "151", "stop": "152", "stride": "1"},
        },
    ),
    (
        Index225,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "152", "stop": "153", "stride": "1"},
        },
    ),
    (
        Index226,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "153", "stop": "154", "stride": "1"},
        },
    ),
    (
        Index227,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "154", "stop": "155", "stride": "1"},
        },
    ),
    (
        Index228,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "155", "stop": "156", "stride": "1"},
        },
    ),
    (
        Index229,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "156", "stop": "157", "stride": "1"},
        },
    ),
    (
        Index230,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "157", "stop": "158", "stride": "1"},
        },
    ),
    (
        Index231,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "158", "stop": "159", "stride": "1"},
        },
    ),
    (
        Index232,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "159", "stop": "160", "stride": "1"},
        },
    ),
    (
        Index233,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "160", "stop": "161", "stride": "1"},
        },
    ),
    (
        Index234,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "161", "stop": "162", "stride": "1"},
        },
    ),
    (
        Index235,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "162", "stop": "163", "stride": "1"},
        },
    ),
    (
        Index236,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "163", "stop": "164", "stride": "1"},
        },
    ),
    (
        Index237,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "164", "stop": "165", "stride": "1"},
        },
    ),
    (
        Index238,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "165", "stop": "166", "stride": "1"},
        },
    ),
    (
        Index239,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "166", "stop": "167", "stride": "1"},
        },
    ),
    (
        Index240,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "167", "stop": "168", "stride": "1"},
        },
    ),
    (
        Index241,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "168", "stop": "169", "stride": "1"},
        },
    ),
    (
        Index242,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "169", "stop": "170", "stride": "1"},
        },
    ),
    (
        Index243,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "170", "stop": "171", "stride": "1"},
        },
    ),
    (
        Index244,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "171", "stop": "172", "stride": "1"},
        },
    ),
    (
        Index245,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "172", "stop": "173", "stride": "1"},
        },
    ),
    (
        Index246,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "173", "stop": "174", "stride": "1"},
        },
    ),
    (
        Index247,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "174", "stop": "175", "stride": "1"},
        },
    ),
    (
        Index248,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "175", "stop": "176", "stride": "1"},
        },
    ),
    (
        Index249,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "176", "stop": "177", "stride": "1"},
        },
    ),
    (
        Index250,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "177", "stop": "178", "stride": "1"},
        },
    ),
    (
        Index251,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "178", "stop": "179", "stride": "1"},
        },
    ),
    (
        Index252,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "179", "stop": "180", "stride": "1"},
        },
    ),
    (
        Index253,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "180", "stop": "181", "stride": "1"},
        },
    ),
    (
        Index254,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "181", "stop": "182", "stride": "1"},
        },
    ),
    (
        Index255,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "182", "stop": "183", "stride": "1"},
        },
    ),
    (
        Index256,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "183", "stop": "184", "stride": "1"},
        },
    ),
    (
        Index257,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "184", "stop": "185", "stride": "1"},
        },
    ),
    (
        Index258,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "185", "stop": "186", "stride": "1"},
        },
    ),
    (
        Index259,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "186", "stop": "187", "stride": "1"},
        },
    ),
    (
        Index260,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "187", "stop": "188", "stride": "1"},
        },
    ),
    (
        Index261,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "188", "stop": "189", "stride": "1"},
        },
    ),
    (
        Index262,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "189", "stop": "190", "stride": "1"},
        },
    ),
    (
        Index263,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "190", "stop": "191", "stride": "1"},
        },
    ),
    (
        Index264,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "191", "stop": "192", "stride": "1"},
        },
    ),
    (
        Index265,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "192", "stop": "193", "stride": "1"},
        },
    ),
    (
        Index266,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "193", "stop": "194", "stride": "1"},
        },
    ),
    (
        Index267,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "194", "stop": "195", "stride": "1"},
        },
    ),
    (
        Index268,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "195", "stop": "196", "stride": "1"},
        },
    ),
    (
        Index269,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "196", "stop": "197", "stride": "1"},
        },
    ),
    (
        Index270,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "197", "stop": "198", "stride": "1"},
        },
    ),
    (
        Index271,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "198", "stop": "199", "stride": "1"},
        },
    ),
    (
        Index272,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "199", "stop": "200", "stride": "1"},
        },
    ),
    (
        Index273,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "200", "stop": "201", "stride": "1"},
        },
    ),
    (
        Index274,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "201", "stop": "202", "stride": "1"},
        },
    ),
    (
        Index275,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "202", "stop": "203", "stride": "1"},
        },
    ),
    (
        Index276,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "203", "stop": "204", "stride": "1"},
        },
    ),
    (
        Index277,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "204", "stop": "205", "stride": "1"},
        },
    ),
    (
        Index278,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "205", "stop": "206", "stride": "1"},
        },
    ),
    (
        Index279,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "206", "stop": "207", "stride": "1"},
        },
    ),
    (
        Index280,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "207", "stop": "208", "stride": "1"},
        },
    ),
    (
        Index281,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "208", "stop": "209", "stride": "1"},
        },
    ),
    (
        Index282,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "209", "stop": "210", "stride": "1"},
        },
    ),
    (
        Index283,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "210", "stop": "211", "stride": "1"},
        },
    ),
    (
        Index284,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "211", "stop": "212", "stride": "1"},
        },
    ),
    (
        Index285,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "212", "stop": "213", "stride": "1"},
        },
    ),
    (
        Index286,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "213", "stop": "214", "stride": "1"},
        },
    ),
    (
        Index287,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "214", "stop": "215", "stride": "1"},
        },
    ),
    (
        Index288,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "215", "stop": "216", "stride": "1"},
        },
    ),
    (
        Index289,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "216", "stop": "217", "stride": "1"},
        },
    ),
    (
        Index290,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "217", "stop": "218", "stride": "1"},
        },
    ),
    (
        Index291,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "218", "stop": "219", "stride": "1"},
        },
    ),
    (
        Index292,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "219", "stop": "220", "stride": "1"},
        },
    ),
    (
        Index293,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "220", "stop": "221", "stride": "1"},
        },
    ),
    (
        Index294,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "221", "stop": "222", "stride": "1"},
        },
    ),
    (
        Index295,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "222", "stop": "223", "stride": "1"},
        },
    ),
    (
        Index296,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "223", "stop": "224", "stride": "1"},
        },
    ),
    (
        Index297,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "224", "stop": "225", "stride": "1"},
        },
    ),
    (
        Index298,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "225", "stop": "226", "stride": "1"},
        },
    ),
    (
        Index299,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "226", "stop": "227", "stride": "1"},
        },
    ),
    (
        Index300,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "227", "stop": "228", "stride": "1"},
        },
    ),
    (
        Index301,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "228", "stop": "229", "stride": "1"},
        },
    ),
    (
        Index302,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "229", "stop": "230", "stride": "1"},
        },
    ),
    (
        Index303,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "230", "stop": "231", "stride": "1"},
        },
    ),
    (
        Index304,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "231", "stop": "232", "stride": "1"},
        },
    ),
    (
        Index305,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "232", "stop": "233", "stride": "1"},
        },
    ),
    (
        Index306,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "233", "stop": "234", "stride": "1"},
        },
    ),
    (
        Index307,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "234", "stop": "235", "stride": "1"},
        },
    ),
    (
        Index308,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "235", "stop": "236", "stride": "1"},
        },
    ),
    (
        Index309,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "236", "stop": "237", "stride": "1"},
        },
    ),
    (
        Index310,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "237", "stop": "238", "stride": "1"},
        },
    ),
    (
        Index311,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "238", "stop": "239", "stride": "1"},
        },
    ),
    (
        Index312,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "239", "stop": "240", "stride": "1"},
        },
    ),
    (
        Index313,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "240", "stop": "241", "stride": "1"},
        },
    ),
    (
        Index314,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "241", "stop": "242", "stride": "1"},
        },
    ),
    (
        Index315,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "242", "stop": "243", "stride": "1"},
        },
    ),
    (
        Index316,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "243", "stop": "244", "stride": "1"},
        },
    ),
    (
        Index317,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "244", "stop": "245", "stride": "1"},
        },
    ),
    (
        Index318,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "245", "stop": "246", "stride": "1"},
        },
    ),
    (
        Index319,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "246", "stop": "247", "stride": "1"},
        },
    ),
    (
        Index320,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "247", "stop": "248", "stride": "1"},
        },
    ),
    (
        Index321,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "248", "stop": "249", "stride": "1"},
        },
    ),
    (
        Index322,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "249", "stop": "250", "stride": "1"},
        },
    ),
    (
        Index323,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "250", "stop": "251", "stride": "1"},
        },
    ),
    (
        Index324,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "251", "stop": "252", "stride": "1"},
        },
    ),
    (
        Index325,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "252", "stop": "253", "stride": "1"},
        },
    ),
    (
        Index326,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "253", "stop": "254", "stride": "1"},
        },
    ),
    (
        Index327,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "254", "stop": "255", "stride": "1"},
        },
    ),
    (
        Index328,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "255", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index329,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "256", "stop": "257", "stride": "1"},
        },
    ),
    (
        Index330,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "257", "stop": "258", "stride": "1"},
        },
    ),
    (
        Index331,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "258", "stop": "259", "stride": "1"},
        },
    ),
    (
        Index332,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "259", "stop": "260", "stride": "1"},
        },
    ),
    (
        Index333,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "260", "stop": "261", "stride": "1"},
        },
    ),
    (
        Index334,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "261", "stop": "262", "stride": "1"},
        },
    ),
    (
        Index335,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "262", "stop": "263", "stride": "1"},
        },
    ),
    (
        Index336,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "263", "stop": "264", "stride": "1"},
        },
    ),
    (
        Index337,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "264", "stop": "265", "stride": "1"},
        },
    ),
    (
        Index338,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "265", "stop": "266", "stride": "1"},
        },
    ),
    (
        Index339,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "266", "stop": "267", "stride": "1"},
        },
    ),
    (
        Index340,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "267", "stop": "268", "stride": "1"},
        },
    ),
    (
        Index341,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "268", "stop": "269", "stride": "1"},
        },
    ),
    (
        Index342,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "269", "stop": "270", "stride": "1"},
        },
    ),
    (
        Index343,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "270", "stop": "271", "stride": "1"},
        },
    ),
    (
        Index344,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "271", "stop": "272", "stride": "1"},
        },
    ),
    (
        Index345,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "272", "stop": "273", "stride": "1"},
        },
    ),
    (
        Index346,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "273", "stop": "274", "stride": "1"},
        },
    ),
    (
        Index347,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "274", "stop": "275", "stride": "1"},
        },
    ),
    (
        Index348,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "275", "stop": "276", "stride": "1"},
        },
    ),
    (
        Index349,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "276", "stop": "277", "stride": "1"},
        },
    ),
    (
        Index350,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "277", "stop": "278", "stride": "1"},
        },
    ),
    (
        Index351,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "278", "stop": "279", "stride": "1"},
        },
    ),
    (
        Index352,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "279", "stop": "280", "stride": "1"},
        },
    ),
    (
        Index353,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "280", "stop": "281", "stride": "1"},
        },
    ),
    (
        Index354,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "281", "stop": "282", "stride": "1"},
        },
    ),
    (
        Index355,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "282", "stop": "283", "stride": "1"},
        },
    ),
    (
        Index356,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "283", "stop": "284", "stride": "1"},
        },
    ),
    (
        Index357,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "284", "stop": "285", "stride": "1"},
        },
    ),
    (
        Index358,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "285", "stop": "286", "stride": "1"},
        },
    ),
    (
        Index359,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "286", "stop": "287", "stride": "1"},
        },
    ),
    (
        Index360,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "287", "stop": "288", "stride": "1"},
        },
    ),
    (
        Index361,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "288", "stop": "289", "stride": "1"},
        },
    ),
    (
        Index362,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "289", "stop": "290", "stride": "1"},
        },
    ),
    (
        Index363,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "290", "stop": "291", "stride": "1"},
        },
    ),
    (
        Index364,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "291", "stop": "292", "stride": "1"},
        },
    ),
    (
        Index365,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "292", "stop": "293", "stride": "1"},
        },
    ),
    (
        Index366,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "293", "stop": "294", "stride": "1"},
        },
    ),
    (
        Index367,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "294", "stop": "295", "stride": "1"},
        },
    ),
    (
        Index368,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "295", "stop": "296", "stride": "1"},
        },
    ),
    (
        Index369,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "296", "stop": "297", "stride": "1"},
        },
    ),
    (
        Index370,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "297", "stop": "298", "stride": "1"},
        },
    ),
    (
        Index371,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "298", "stop": "299", "stride": "1"},
        },
    ),
    (
        Index372,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "299", "stop": "300", "stride": "1"},
        },
    ),
    (
        Index6,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index7,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index44,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "2", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index45,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "3", "stop": "4", "stride": "1"},
        },
    ),
    (
        Index46,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "4", "stop": "5", "stride": "1"},
        },
    ),
    (
        Index47,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "5", "stop": "6", "stride": "1"},
        },
    ),
    (
        Index373,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "6", "stop": "7", "stride": "1"},
        },
    ),
    (
        Index374,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "7", "stop": "8", "stride": "1"},
        },
    ),
    (
        Index375,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "8", "stop": "9", "stride": "1"},
        },
    ),
    (
        Index376,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "9", "stop": "10", "stride": "1"},
        },
    ),
    (
        Index377,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "10", "stop": "11", "stride": "1"},
        },
    ),
    (
        Index378,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "11", "stop": "12", "stride": "1"},
        },
    ),
    (
        Index379,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "12", "stop": "13", "stride": "1"},
        },
    ),
    (
        Index380,
        [((300, 14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "13", "stop": "14", "stride": "1"},
        },
    ),
    (
        Index13,
        [((14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index14,
        [((14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index381,
        [((14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "2", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index382,
        [((14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "3", "stop": "4", "stride": "1"},
        },
    ),
    (
        Index383,
        [((14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "4", "stop": "5", "stride": "1"},
        },
    ),
    (
        Index384,
        [((14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "5", "stop": "6", "stride": "1"},
        },
    ),
    (
        Index385,
        [((14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "6", "stop": "7", "stride": "1"},
        },
    ),
    (
        Index386,
        [((14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "7", "stop": "8", "stride": "1"},
        },
    ),
    (
        Index387,
        [((14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "8", "stop": "9", "stride": "1"},
        },
    ),
    (
        Index388,
        [((14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "9", "stop": "10", "stride": "1"},
        },
    ),
    (
        Index389,
        [((14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "10", "stop": "11", "stride": "1"},
        },
    ),
    (
        Index390,
        [((14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "11", "stop": "12", "stride": "1"},
        },
    ),
    (
        Index391,
        [((14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "12", "stop": "13", "stride": "1"},
        },
    ),
    (
        Index392,
        [((14, 14, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "13", "stop": "14", "stride": "1"},
        },
    ),
    (
        Index26,
        [((1, 70, 70, 768), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index68,
        [((1, 64, 70, 768), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index54,
        [((3, 12, 4096, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index55,
        [((3, 12, 4096, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index56,
        [((3, 12, 4096, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "2", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index54,
        [((12, 64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index55,
        [((12, 64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index56,
        [((12, 64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "2", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index76,
        [((12, 64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "3", "stop": "4", "stride": "1"},
        },
    ),
    (
        Index77,
        [((12, 64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "4", "stop": "5", "stride": "1"},
        },
    ),
    (
        Index78,
        [((12, 64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "5", "stop": "6", "stride": "1"},
        },
    ),
    (
        Index79,
        [((12, 64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "6", "stop": "7", "stride": "1"},
        },
    ),
    (
        Index80,
        [((12, 64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "7", "stop": "8", "stride": "1"},
        },
    ),
    (
        Index81,
        [((12, 64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "8", "stop": "9", "stride": "1"},
        },
    ),
    (
        Index82,
        [((12, 64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "9", "stop": "10", "stride": "1"},
        },
    ),
    (
        Index83,
        [((12, 64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "10", "stop": "11", "stride": "1"},
        },
    ),
    (
        Index84,
        [((12, 64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "11", "stop": "12", "stride": "1"},
        },
    ),
    (
        Index6,
        [((12, 64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index7,
        [((12, 64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index44,
        [((12, 64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "2", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index45,
        [((12, 64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "3", "stop": "4", "stride": "1"},
        },
    ),
    (
        Index46,
        [((12, 64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "4", "stop": "5", "stride": "1"},
        },
    ),
    (
        Index47,
        [((12, 64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "5", "stop": "6", "stride": "1"},
        },
    ),
    (
        Index373,
        [((12, 64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "6", "stop": "7", "stride": "1"},
        },
    ),
    (
        Index374,
        [((12, 64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "7", "stop": "8", "stride": "1"},
        },
    ),
    (
        Index375,
        [((12, 64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "8", "stop": "9", "stride": "1"},
        },
    ),
    (
        Index376,
        [((12, 64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "9", "stop": "10", "stride": "1"},
        },
    ),
    (
        Index377,
        [((12, 64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "10", "stop": "11", "stride": "1"},
        },
    ),
    (
        Index378,
        [((12, 64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "11", "stop": "12", "stride": "1"},
        },
    ),
    (
        Index379,
        [((12, 64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "12", "stop": "13", "stride": "1"},
        },
    ),
    (
        Index380,
        [((12, 64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "13", "stop": "14", "stride": "1"},
        },
    ),
    (
        Index393,
        [((12, 64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "14", "stop": "15", "stride": "1"},
        },
    ),
    (
        Index394,
        [((12, 64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "15", "stop": "16", "stride": "1"},
        },
    ),
    (
        Index395,
        [((12, 64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "16", "stop": "17", "stride": "1"},
        },
    ),
    (
        Index396,
        [((12, 64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "17", "stop": "18", "stride": "1"},
        },
    ),
    (
        Index397,
        [((12, 64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "18", "stop": "19", "stride": "1"},
        },
    ),
    (
        Index398,
        [((12, 64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "19", "stop": "20", "stride": "1"},
        },
    ),
    (
        Index399,
        [((12, 64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "20", "stop": "21", "stride": "1"},
        },
    ),
    (
        Index400,
        [((12, 64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "21", "stop": "22", "stride": "1"},
        },
    ),
    (
        Index401,
        [((12, 64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "22", "stop": "23", "stride": "1"},
        },
    ),
    (
        Index402,
        [((12, 64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "23", "stop": "24", "stride": "1"},
        },
    ),
    (
        Index403,
        [((12, 64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "24", "stop": "25", "stride": "1"},
        },
    ),
    (
        Index404,
        [((12, 64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "25", "stop": "26", "stride": "1"},
        },
    ),
    (
        Index405,
        [((12, 64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "26", "stop": "27", "stride": "1"},
        },
    ),
    (
        Index406,
        [((12, 64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "27", "stop": "28", "stride": "1"},
        },
    ),
    (
        Index407,
        [((12, 64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "28", "stop": "29", "stride": "1"},
        },
    ),
    (
        Index408,
        [((12, 64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "29", "stop": "30", "stride": "1"},
        },
    ),
    (
        Index409,
        [((12, 64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "30", "stop": "31", "stride": "1"},
        },
    ),
    (
        Index410,
        [((12, 64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "31", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index411,
        [((12, 64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "32", "stop": "33", "stride": "1"},
        },
    ),
    (
        Index412,
        [((12, 64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "33", "stop": "34", "stride": "1"},
        },
    ),
    (
        Index413,
        [((12, 64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "34", "stop": "35", "stride": "1"},
        },
    ),
    (
        Index414,
        [((12, 64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "35", "stop": "36", "stride": "1"},
        },
    ),
    (
        Index415,
        [((12, 64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "36", "stop": "37", "stride": "1"},
        },
    ),
    (
        Index416,
        [((12, 64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "37", "stop": "38", "stride": "1"},
        },
    ),
    (
        Index417,
        [((12, 64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "38", "stop": "39", "stride": "1"},
        },
    ),
    (
        Index418,
        [((12, 64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "39", "stop": "40", "stride": "1"},
        },
    ),
    (
        Index419,
        [((12, 64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "40", "stop": "41", "stride": "1"},
        },
    ),
    (
        Index420,
        [((12, 64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "41", "stop": "42", "stride": "1"},
        },
    ),
    (
        Index421,
        [((12, 64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "42", "stop": "43", "stride": "1"},
        },
    ),
    (
        Index422,
        [((12, 64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "43", "stop": "44", "stride": "1"},
        },
    ),
    (
        Index423,
        [((12, 64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "44", "stop": "45", "stride": "1"},
        },
    ),
    (
        Index424,
        [((12, 64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "45", "stop": "46", "stride": "1"},
        },
    ),
    (
        Index425,
        [((12, 64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "46", "stop": "47", "stride": "1"},
        },
    ),
    (
        Index426,
        [((12, 64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "47", "stop": "48", "stride": "1"},
        },
    ),
    (
        Index427,
        [((12, 64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "48", "stop": "49", "stride": "1"},
        },
    ),
    (
        Index428,
        [((12, 64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "49", "stop": "50", "stride": "1"},
        },
    ),
    (
        Index429,
        [((12, 64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "50", "stop": "51", "stride": "1"},
        },
    ),
    (
        Index430,
        [((12, 64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "51", "stop": "52", "stride": "1"},
        },
    ),
    (
        Index431,
        [((12, 64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "52", "stop": "53", "stride": "1"},
        },
    ),
    (
        Index432,
        [((12, 64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "53", "stop": "54", "stride": "1"},
        },
    ),
    (
        Index433,
        [((12, 64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "54", "stop": "55", "stride": "1"},
        },
    ),
    (
        Index434,
        [((12, 64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "55", "stop": "56", "stride": "1"},
        },
    ),
    (
        Index435,
        [((12, 64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "56", "stop": "57", "stride": "1"},
        },
    ),
    (
        Index436,
        [((12, 64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "57", "stop": "58", "stride": "1"},
        },
    ),
    (
        Index437,
        [((12, 64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "58", "stop": "59", "stride": "1"},
        },
    ),
    (
        Index438,
        [((12, 64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "59", "stop": "60", "stride": "1"},
        },
    ),
    (
        Index439,
        [((12, 64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "60", "stop": "61", "stride": "1"},
        },
    ),
    (
        Index440,
        [((12, 64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "61", "stop": "62", "stride": "1"},
        },
    ),
    (
        Index441,
        [((12, 64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "62", "stop": "63", "stride": "1"},
        },
    ),
    (
        Index442,
        [((12, 64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "63", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index13,
        [((64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index14,
        [((64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index381,
        [((64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "2", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index382,
        [((64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "3", "stop": "4", "stride": "1"},
        },
    ),
    (
        Index383,
        [((64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "4", "stop": "5", "stride": "1"},
        },
    ),
    (
        Index384,
        [((64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "5", "stop": "6", "stride": "1"},
        },
    ),
    (
        Index385,
        [((64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "6", "stop": "7", "stride": "1"},
        },
    ),
    (
        Index386,
        [((64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "7", "stop": "8", "stride": "1"},
        },
    ),
    (
        Index387,
        [((64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "8", "stop": "9", "stride": "1"},
        },
    ),
    (
        Index388,
        [((64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "9", "stop": "10", "stride": "1"},
        },
    ),
    (
        Index389,
        [((64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "10", "stop": "11", "stride": "1"},
        },
    ),
    (
        Index390,
        [((64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "11", "stop": "12", "stride": "1"},
        },
    ),
    (
        Index391,
        [((64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "12", "stop": "13", "stride": "1"},
        },
    ),
    (
        Index392,
        [((64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "13", "stop": "14", "stride": "1"},
        },
    ),
    (
        Index443,
        [((64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "14", "stop": "15", "stride": "1"},
        },
    ),
    (
        Index444,
        [((64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "15", "stop": "16", "stride": "1"},
        },
    ),
    (
        Index445,
        [((64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "16", "stop": "17", "stride": "1"},
        },
    ),
    (
        Index446,
        [((64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "17", "stop": "18", "stride": "1"},
        },
    ),
    (
        Index447,
        [((64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "18", "stop": "19", "stride": "1"},
        },
    ),
    (
        Index448,
        [((64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "19", "stop": "20", "stride": "1"},
        },
    ),
    (
        Index449,
        [((64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "20", "stop": "21", "stride": "1"},
        },
    ),
    (
        Index450,
        [((64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "21", "stop": "22", "stride": "1"},
        },
    ),
    (
        Index451,
        [((64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "22", "stop": "23", "stride": "1"},
        },
    ),
    (
        Index452,
        [((64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "23", "stop": "24", "stride": "1"},
        },
    ),
    (
        Index453,
        [((64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "24", "stop": "25", "stride": "1"},
        },
    ),
    (
        Index454,
        [((64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "25", "stop": "26", "stride": "1"},
        },
    ),
    (
        Index455,
        [((64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "26", "stop": "27", "stride": "1"},
        },
    ),
    (
        Index456,
        [((64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "27", "stop": "28", "stride": "1"},
        },
    ),
    (
        Index457,
        [((64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "28", "stop": "29", "stride": "1"},
        },
    ),
    (
        Index458,
        [((64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "29", "stop": "30", "stride": "1"},
        },
    ),
    (
        Index459,
        [((64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "30", "stop": "31", "stride": "1"},
        },
    ),
    (
        Index460,
        [((64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "31", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index461,
        [((64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "32", "stop": "33", "stride": "1"},
        },
    ),
    (
        Index462,
        [((64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "33", "stop": "34", "stride": "1"},
        },
    ),
    (
        Index463,
        [((64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "34", "stop": "35", "stride": "1"},
        },
    ),
    (
        Index464,
        [((64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "35", "stop": "36", "stride": "1"},
        },
    ),
    (
        Index465,
        [((64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "36", "stop": "37", "stride": "1"},
        },
    ),
    (
        Index466,
        [((64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "37", "stop": "38", "stride": "1"},
        },
    ),
    (
        Index467,
        [((64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "38", "stop": "39", "stride": "1"},
        },
    ),
    (
        Index468,
        [((64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "39", "stop": "40", "stride": "1"},
        },
    ),
    (
        Index469,
        [((64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "40", "stop": "41", "stride": "1"},
        },
    ),
    (
        Index470,
        [((64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "41", "stop": "42", "stride": "1"},
        },
    ),
    (
        Index471,
        [((64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "42", "stop": "43", "stride": "1"},
        },
    ),
    (
        Index472,
        [((64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "43", "stop": "44", "stride": "1"},
        },
    ),
    (
        Index473,
        [((64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "44", "stop": "45", "stride": "1"},
        },
    ),
    (
        Index474,
        [((64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "45", "stop": "46", "stride": "1"},
        },
    ),
    (
        Index475,
        [((64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "46", "stop": "47", "stride": "1"},
        },
    ),
    (
        Index476,
        [((64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "47", "stop": "48", "stride": "1"},
        },
    ),
    (
        Index477,
        [((64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "48", "stop": "49", "stride": "1"},
        },
    ),
    (
        Index478,
        [((64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "49", "stop": "50", "stride": "1"},
        },
    ),
    (
        Index479,
        [((64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "50", "stop": "51", "stride": "1"},
        },
    ),
    (
        Index480,
        [((64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "51", "stop": "52", "stride": "1"},
        },
    ),
    (
        Index481,
        [((64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "52", "stop": "53", "stride": "1"},
        },
    ),
    (
        Index482,
        [((64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "53", "stop": "54", "stride": "1"},
        },
    ),
    (
        Index483,
        [((64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "54", "stop": "55", "stride": "1"},
        },
    ),
    (
        Index484,
        [((64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "55", "stop": "56", "stride": "1"},
        },
    ),
    (
        Index485,
        [((64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "56", "stop": "57", "stride": "1"},
        },
    ),
    (
        Index486,
        [((64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "57", "stop": "58", "stride": "1"},
        },
    ),
    (
        Index487,
        [((64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "58", "stop": "59", "stride": "1"},
        },
    ),
    (
        Index488,
        [((64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "59", "stop": "60", "stride": "1"},
        },
    ),
    (
        Index489,
        [((64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "60", "stop": "61", "stride": "1"},
        },
    ),
    (
        Index490,
        [((64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "61", "stop": "62", "stride": "1"},
        },
    ),
    (
        Index491,
        [((64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "62", "stop": "63", "stride": "1"},
        },
    ),
    (
        Index492,
        [((64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "63", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index6,
        [((1, 1, 5, 256), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index493,
        [((1, 1, 5, 256), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "1", "stop": "5", "stride": "1"},
        },
    ),
    (
        Index494,
        [((1, 1, 4), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "1", "stop": "4", "stride": "1"},
        },
    ),
    (
        Index6,
        [((1, 1, 4, 256), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index7,
        [((1, 1, 4, 256), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index44,
        [((1, 1, 4, 256), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "2", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index45,
        [((1, 1, 4, 256), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "3", "stop": "4", "stride": "1"},
        },
    ),
    (
        Index495,
        [((1, 1, 4, 256, 256), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "1", "stop": "4", "stride": "1"},
        },
    ),
    (
        Index23,
        [((1, 32, 160, 160), torch.float32)],
        {
            "model_names": ["onnx_yolov8_default_obj_det_github", "onnx_yolov10_default_obj_det_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "16", "stride": "1"},
        },
    ),
    (
        Index24,
        [((1, 32, 160, 160), torch.float32)],
        {
            "model_names": ["onnx_yolov8_default_obj_det_github", "onnx_yolov10_default_obj_det_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "16", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index496,
        [((1, 64, 80, 80), torch.float32)],
        {
            "model_names": ["onnx_yolov8_default_obj_det_github", "onnx_yolov10_default_obj_det_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index25,
        [((1, 64, 80, 80), torch.float32)],
        {
            "model_names": ["onnx_yolov8_default_obj_det_github", "onnx_yolov10_default_obj_det_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "32", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index26,
        [((1, 128, 40, 40), torch.float32)],
        {
            "model_names": ["onnx_yolov8_default_obj_det_github", "onnx_yolov10_default_obj_det_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index497,
        [((1, 128, 40, 40), torch.float32)],
        {
            "model_names": ["onnx_yolov8_default_obj_det_github", "onnx_yolov10_default_obj_det_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "64", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index498,
        [((1, 256, 20, 20), torch.float32)],
        {
            "model_names": ["onnx_yolov8_default_obj_det_github", "onnx_yolov10_default_obj_det_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index499,
        [((1, 256, 20, 20), torch.float32)],
        {
            "model_names": ["onnx_yolov8_default_obj_det_github", "onnx_yolov10_default_obj_det_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "128", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index68,
        [((1, 144, 8400), torch.float32)],
        {
            "model_names": ["onnx_yolov8_default_obj_det_github", "onnx_yolov10_default_obj_det_github"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index69,
        [((1, 144, 8400), torch.float32)],
        {
            "model_names": ["onnx_yolov8_default_obj_det_github", "onnx_yolov10_default_obj_det_github"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "64", "stop": "144", "stride": "1"},
        },
    ),
    (
        Index70,
        [((1, 4, 8400), torch.float32)],
        {
            "model_names": ["onnx_yolov8_default_obj_det_github", "onnx_yolov10_default_obj_det_github"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index71,
        [((1, 4, 8400), torch.float32)],
        {
            "model_names": ["onnx_yolov8_default_obj_det_github", "onnx_yolov10_default_obj_det_github"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "2", "stop": "4", "stride": "1"},
        },
    ),
    (
        Index6,
        [((1, 8, 768), torch.float32)],
        {
            "model_names": [
                "pd_blip_text_salesforce_blip_image_captioning_base_text_enc_padlenlp",
                "pd_chineseclip_text_ofa_sys_chinese_clip_vit_base_patch16_text_enc_padlenlp",
                "pd_bert_bert_base_uncased_seq_cls_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index6,
        [((1, 9, 768), torch.float32)],
        {
            "model_names": [
                "pd_ernie_1_0_seq_cls_padlenlp",
                "pd_roberta_rbt4_ch_seq_cls_padlenlp",
                "pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index26,
        [((1, 176, 27, 27), torch.float32)],
        {
            "model_names": ["pd_googlenet_base_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index500,
        [((1, 176, 27, 27), torch.float32)],
        {
            "model_names": ["pd_googlenet_base_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "64", "stop": "160", "stride": "1"},
        },
    ),
    (
        Index501,
        [((1, 176, 27, 27), torch.float32)],
        {
            "model_names": ["pd_googlenet_base_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "160", "stop": "176", "stride": "1"},
        },
    ),
    (
        Index498,
        [((1, 288, 27, 27), torch.float32)],
        {
            "model_names": ["pd_googlenet_base_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index499,
        [((1, 288, 27, 27), torch.float32)],
        {
            "model_names": ["pd_googlenet_base_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "128", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index502,
        [((1, 288, 27, 27), torch.float32)],
        {
            "model_names": ["pd_googlenet_base_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "256", "stop": "288", "stride": "1"},
        },
    ),
    (
        Index503,
        [((1, 304, 13, 13), torch.float32)],
        {
            "model_names": ["pd_googlenet_base_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "192", "stride": "1"},
        },
    ),
    (
        Index504,
        [((1, 304, 13, 13), torch.float32)],
        {
            "model_names": ["pd_googlenet_base_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "192", "stop": "288", "stride": "1"},
        },
    ),
    (
        Index505,
        [((1, 304, 13, 13), torch.float32)],
        {
            "model_names": ["pd_googlenet_base_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "288", "stop": "304", "stride": "1"},
        },
    ),
    (
        Index61,
        [((1, 296, 13, 13), torch.float32)],
        {
            "model_names": ["pd_googlenet_base_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "160", "stride": "1"},
        },
    ),
    (
        Index506,
        [((1, 296, 13, 13), torch.float32)],
        {
            "model_names": ["pd_googlenet_base_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "160", "stop": "272", "stride": "1"},
        },
    ),
    (
        Index507,
        [((1, 296, 13, 13), torch.float32)],
        {
            "model_names": ["pd_googlenet_base_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "272", "stop": "296", "stride": "1"},
        },
    ),
    (
        Index498,
        [((1, 280, 13, 13), torch.float32)],
        {
            "model_names": ["pd_googlenet_base_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index499,
        [((1, 280, 13, 13), torch.float32)],
        {
            "model_names": ["pd_googlenet_base_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "128", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index508,
        [((1, 280, 13, 13), torch.float32)],
        {
            "model_names": ["pd_googlenet_base_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "256", "stop": "280", "stride": "1"},
        },
    ),
    (
        Index509,
        [((1, 288, 13, 13), torch.float32)],
        {
            "model_names": ["pd_googlenet_base_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "112", "stride": "1"},
        },
    ),
    (
        Index510,
        [((1, 288, 13, 13), torch.float32)],
        {
            "model_names": ["pd_googlenet_base_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "112", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index502,
        [((1, 288, 13, 13), torch.float32)],
        {
            "model_names": ["pd_googlenet_base_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "256", "stop": "288", "stride": "1"},
        },
    ),
    (
        Index35,
        [((1, 448, 13, 13), torch.float32)],
        {
            "model_names": ["pd_googlenet_base_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index511,
        [((1, 448, 13, 13), torch.float32)],
        {
            "model_names": ["pd_googlenet_base_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "256", "stop": "416", "stride": "1"},
        },
    ),
    (
        Index512,
        [((1, 448, 13, 13), torch.float32)],
        {
            "model_names": ["pd_googlenet_base_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "416", "stop": "448", "stride": "1"},
        },
    ),
    (
        Index35,
        [((1, 448, 6, 6), torch.float32)],
        {
            "model_names": ["pd_googlenet_base_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index511,
        [((1, 448, 6, 6), torch.float32)],
        {
            "model_names": ["pd_googlenet_base_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "256", "stop": "416", "stride": "1"},
        },
    ),
    (
        Index512,
        [((1, 448, 6, 6), torch.float32)],
        {
            "model_names": ["pd_googlenet_base_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "416", "stop": "448", "stride": "1"},
        },
    ),
    (
        Index32,
        [((1, 624, 6, 6), torch.float32)],
        {
            "model_names": ["pd_googlenet_base_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "384", "stride": "1"},
        },
    ),
    (
        Index33,
        [((1, 624, 6, 6), torch.float32)],
        {
            "model_names": ["pd_googlenet_base_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "384", "stop": "576", "stride": "1"},
        },
    ),
    (
        Index513,
        [((1, 624, 6, 6), torch.float32)],
        {
            "model_names": ["pd_googlenet_base_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "576", "stop": "624", "stride": "1"},
        },
    ),
    (
        Index13,
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
        Index14,
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
        Index381,
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
        Index382,
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
        Index383,
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
        Index384,
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
        Index385,
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
        Index386,
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
        Index387,
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
        Index388,
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
        Index389,
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
        Index390,
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
        Index391,
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
        Index392,
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
        Index443,
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
        Index444,
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
        Index445,
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
        Index446,
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
        Index447,
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
        Index448,
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
        Index449,
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
        Index450,
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
        Index451,
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
        Index452,
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
        Index453,
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
        Index514,
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
        Index515,
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
        Index516,
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
        Index517,
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
        Index13,
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
        Index14,
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
        Index381,
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
        Index382,
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
        Index383,
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
        Index384,
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
        Index385,
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
        Index386,
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
        Index387,
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
        Index388,
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
        Index389,
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
        Index390,
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
        Index391,
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
        Index392,
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
        Index443,
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
        Index444,
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
        Index445,
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
        Index446,
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
        Index447,
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
        Index448,
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
        Index449,
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
        Index450,
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
        Index451,
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
        Index452,
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
        Index453,
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
        Index518,
        [((732, 12), torch.bfloat16)],
        {
            "model_names": ["pt_beit_microsoft_beit_base_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "729", "stride": "1"},
        },
    ),
    (
        Index519,
        [((732, 12), torch.bfloat16)],
        {
            "model_names": ["pt_beit_microsoft_beit_base_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "729", "stop": "732", "stride": "1"},
        },
    ),
    (
        Index520,
        [((1, 197, 768), torch.bfloat16)],
        {
            "model_names": ["pt_beit_microsoft_beit_base_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "1", "stop": "197", "stride": "1"},
        },
    ),
    (
        Index6,
        [((1, 197, 768), torch.bfloat16)],
        {
            "model_names": [
                "pt_deit_facebook_deit_base_distilled_patch16_224_img_cls_hf",
                "pt_vit_vit_b_16_img_cls_torchvision",
                "pt_deit_facebook_deit_base_patch16_224_img_cls_hf",
                "pt_vit_google_vit_base_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index518,
        [((732, 16), torch.bfloat16)],
        {
            "model_names": ["pt_beit_microsoft_beit_large_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "729", "stride": "1"},
        },
    ),
    (
        Index519,
        [((732, 16), torch.bfloat16)],
        {
            "model_names": ["pt_beit_microsoft_beit_large_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "729", "stop": "732", "stride": "1"},
        },
    ),
    (
        Index520,
        [((1, 197, 1024), torch.bfloat16)],
        {
            "model_names": ["pt_beit_microsoft_beit_large_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "1", "stop": "197", "stride": "1"},
        },
    ),
    (
        Index6,
        [((1, 197, 1024), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_l_16_img_cls_torchvision", "pt_vit_google_vit_large_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index6,
        [((1, 32, 16, 3, 96), torch.float32)],
        {
            "model_names": ["pt_bloom_bigscience_bloom_1b1_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index7,
        [((1, 32, 16, 3, 96), torch.float32)],
        {
            "model_names": ["pt_bloom_bigscience_bloom_1b1_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index44,
        [((1, 32, 16, 3, 96), torch.float32)],
        {
            "model_names": ["pt_bloom_bigscience_bloom_1b1_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "2", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index6,
        [((1, 197, 384), torch.bfloat16)],
        {
            "model_names": ["pt_deit_facebook_deit_small_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index521,
        [((1, 25, 34), torch.bfloat16)],
        {
            "model_names": [
                "pt_detr_facebook_detr_resnet_50_obj_det_hf",
                "pt_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "-1", "stop": "25", "stride": "1"},
        },
    ),
    (
        Index522,
        [((1, 25, 34), torch.bfloat16)],
        {
            "model_names": [
                "pt_detr_facebook_detr_resnet_50_obj_det_hf",
                "pt_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "-1", "stop": "34", "stride": "1"},
        },
    ),
    (
        Index523,
        [((1, 25, 34, 128), torch.bfloat16)],
        {
            "model_names": [
                "pt_detr_facebook_detr_resnet_50_obj_det_hf",
                "pt_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "128", "stride": "2"},
        },
    ),
    (
        Index524,
        [((1, 25, 34, 128), torch.bfloat16)],
        {
            "model_names": [
                "pt_detr_facebook_detr_resnet_50_obj_det_hf",
                "pt_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "1", "stop": "128", "stride": "2"},
        },
    ),
    (
        Index13,
        [((1, 3, 224, 224), torch.bfloat16)],
        {
            "model_names": ["pt_googlenet_base_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index14,
        [((1, 3, 224, 224), torch.bfloat16)],
        {
            "model_names": ["pt_googlenet_base_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index381,
        [((1, 3, 224, 224), torch.bfloat16)],
        {
            "model_names": ["pt_googlenet_base_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "2", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index26,
        [((1, 176, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_googlenet_base_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index500,
        [((1, 176, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_googlenet_base_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "64", "stop": "160", "stride": "1"},
        },
    ),
    (
        Index501,
        [((1, 176, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_googlenet_base_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "160", "stop": "176", "stride": "1"},
        },
    ),
    (
        Index525,
        [((1, 176, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w44_pose_estimation_timm", "pt_hrnet_hrnetv2_w44_pose_estimation_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "88", "stride": "1"},
        },
    ),
    (
        Index526,
        [((1, 176, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w44_pose_estimation_timm", "pt_hrnet_hrnetv2_w44_pose_estimation_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "88", "stop": "132", "stride": "1"},
        },
    ),
    (
        Index527,
        [((1, 176, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w44_pose_estimation_timm", "pt_hrnet_hrnetv2_w44_pose_estimation_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "132", "stop": "176", "stride": "1"},
        },
    ),
    (
        Index528,
        [((1, 176, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w44_pose_estimation_timm", "pt_hrnet_hrnetv2_w44_pose_estimation_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "44", "stride": "1"},
        },
    ),
    (
        Index529,
        [((1, 176, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w44_pose_estimation_timm", "pt_hrnet_hrnetv2_w44_pose_estimation_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "44", "stop": "88", "stride": "1"},
        },
    ),
    (
        Index530,
        [((1, 176, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w44_pose_estimation_timm", "pt_hrnet_hrnetv2_w44_pose_estimation_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "88", "stop": "176", "stride": "1"},
        },
    ),
    (
        Index498,
        [((1, 288, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_googlenet_base_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index499,
        [((1, 288, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_googlenet_base_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "128", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index502,
        [((1, 288, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_googlenet_base_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "256", "stop": "288", "stride": "1"},
        },
    ),
    (
        Index503,
        [((1, 304, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_googlenet_base_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "192", "stride": "1"},
        },
    ),
    (
        Index504,
        [((1, 304, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_googlenet_base_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "192", "stop": "288", "stride": "1"},
        },
    ),
    (
        Index505,
        [((1, 304, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_googlenet_base_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "288", "stop": "304", "stride": "1"},
        },
    ),
    (
        Index61,
        [((1, 296, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_googlenet_base_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "160", "stride": "1"},
        },
    ),
    (
        Index506,
        [((1, 296, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_googlenet_base_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "160", "stop": "272", "stride": "1"},
        },
    ),
    (
        Index507,
        [((1, 296, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_googlenet_base_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "272", "stop": "296", "stride": "1"},
        },
    ),
    (
        Index498,
        [((1, 280, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_googlenet_base_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index499,
        [((1, 280, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_googlenet_base_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "128", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index508,
        [((1, 280, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_googlenet_base_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "256", "stop": "280", "stride": "1"},
        },
    ),
    (
        Index509,
        [((1, 288, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_googlenet_base_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "112", "stride": "1"},
        },
    ),
    (
        Index510,
        [((1, 288, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_googlenet_base_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "112", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index502,
        [((1, 288, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_googlenet_base_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "256", "stop": "288", "stride": "1"},
        },
    ),
    (
        Index35,
        [((1, 448, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_googlenet_base_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index511,
        [((1, 448, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_googlenet_base_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "256", "stop": "416", "stride": "1"},
        },
    ),
    (
        Index512,
        [((1, 448, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_googlenet_base_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "416", "stop": "448", "stride": "1"},
        },
    ),
    (
        Index35,
        [((1, 448, 7, 7), torch.bfloat16)],
        {
            "model_names": [
                "pt_googlenet_base_img_cls_torchvision",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
                "pt_hrnet_hrnet_w64_pose_estimation_timm",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index511,
        [((1, 448, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_googlenet_base_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "256", "stop": "416", "stride": "1"},
        },
    ),
    (
        Index512,
        [((1, 448, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_googlenet_base_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "416", "stop": "448", "stride": "1"},
        },
    ),
    (
        Index26,
        [((1, 448, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w64_pose_estimation_osmr", "pt_hrnet_hrnet_w64_pose_estimation_timm"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index531,
        [((1, 448, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w64_pose_estimation_osmr", "pt_hrnet_hrnet_w64_pose_estimation_timm"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "64", "stop": "192", "stride": "1"},
        },
    ),
    (
        Index532,
        [((1, 448, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w64_pose_estimation_osmr", "pt_hrnet_hrnet_w64_pose_estimation_timm"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "192", "stop": "448", "stride": "1"},
        },
    ),
    (
        Index533,
        [((1, 448, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w64_pose_estimation_osmr", "pt_hrnet_hrnet_w64_pose_estimation_timm"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "256", "stop": "384", "stride": "1"},
        },
    ),
    (
        Index534,
        [((1, 448, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w64_pose_estimation_osmr", "pt_hrnet_hrnet_w64_pose_estimation_timm"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "384", "stop": "448", "stride": "1"},
        },
    ),
    (
        Index32,
        [((1, 624, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_googlenet_base_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "384", "stride": "1"},
        },
    ),
    (
        Index33,
        [((1, 624, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_googlenet_base_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "384", "stop": "576", "stride": "1"},
        },
    ),
    (
        Index513,
        [((1, 624, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_googlenet_base_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "576", "stop": "624", "stride": "1"},
        },
    ),
    (
        Index535,
        [((1, 1, 2048, 2048), torch.bool)],
        {
            "model_names": [
                "pt_gptneo_eleutherai_gpt_neo_125m_clm_hf",
                "pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf",
                "pt_gptneo_eleutherai_gpt_neo_2_7b_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index66,
        [((1, 1, 2048, 2048), torch.bool)],
        {
            "model_names": [
                "pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf",
                "pt_gptneo_eleutherai_gpt_neo_125m_seq_cls_hf",
                "pt_gptneo_eleutherai_gpt_neo_2_7b_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index536,
        [((1, 1, 256, 2048), torch.bool)],
        {
            "model_names": [
                "pt_gptneo_eleutherai_gpt_neo_125m_clm_hf",
                "pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf",
                "pt_gptneo_eleutherai_gpt_neo_2_7b_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index537,
        [((3, 1, 12, 257, 64), torch.bfloat16)],
        {
            "model_names": ["pt_mgp_alibaba_damo_mgp_str_base_scene_text_recognition_hf"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index538,
        [((3, 1, 12, 257, 64), torch.bfloat16)],
        {
            "model_names": ["pt_mgp_alibaba_damo_mgp_str_base_scene_text_recognition_hf"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index539,
        [((3, 1, 12, 257, 64), torch.bfloat16)],
        {
            "model_names": ["pt_mgp_alibaba_damo_mgp_str_base_scene_text_recognition_hf"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "2", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index10,
        [((1, 32, 12, 64), torch.float32)],
        {
            "model_names": ["pt_phi1_microsoft_phi_1_token_cls_hf", "pt_phi_1_5_microsoft_phi_1_5_token_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index11,
        [((1, 32, 12, 64), torch.float32)],
        {
            "model_names": ["pt_phi1_microsoft_phi_1_token_cls_hf", "pt_phi_1_5_microsoft_phi_1_5_token_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "32", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index12,
        [((1, 32, 12, 32), torch.float32)],
        {
            "model_names": [
                "pt_phi1_microsoft_phi_1_token_cls_hf",
                "pt_phi_1_5_microsoft_phi_1_5_token_cls_hf",
                "pt_phi2_microsoft_phi_2_pytdml_token_cls_hf",
                "pt_phi2_microsoft_phi_2_token_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "16", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index5,
        [((1, 32, 12, 32), torch.float32)],
        {
            "model_names": [
                "pt_phi1_microsoft_phi_1_token_cls_hf",
                "pt_phi_1_5_microsoft_phi_1_5_token_cls_hf",
                "pt_phi2_microsoft_phi_2_pytdml_token_cls_hf",
                "pt_phi2_microsoft_phi_2_token_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "16", "stride": "1"},
        },
    ),
    (
        Index11,
        [((1, 16, 6, 64), torch.float32)],
        {
            "model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "32", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index10,
        [((1, 16, 6, 64), torch.float32)],
        {
            "model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index535,
        [((3072, 1024), torch.float32)],
        {
            "model_names": [
                "pt_codegen_salesforce_codegen_350m_mono_clm_hf",
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index540,
        [((3072, 1024), torch.float32)],
        {
            "model_names": [
                "pt_codegen_salesforce_codegen_350m_mono_clm_hf",
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "768", "stop": "1024", "stride": "1"},
        },
    ),
    (
        Index541,
        [((3072, 1024), torch.float32)],
        {
            "model_names": [
                "pt_codegen_salesforce_codegen_350m_mono_clm_hf",
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "1536", "stop": "1792", "stride": "1"},
        },
    ),
    (
        Index542,
        [((3072, 1024), torch.float32)],
        {
            "model_names": [
                "pt_codegen_salesforce_codegen_350m_mono_clm_hf",
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "2304", "stop": "2560", "stride": "1"},
        },
    ),
    (
        Index543,
        [((3072, 1024), torch.float32)],
        {
            "model_names": [
                "pt_codegen_salesforce_codegen_350m_mono_clm_hf",
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "512", "stop": "768", "stride": "1"},
        },
    ),
    (
        Index544,
        [((3072, 1024), torch.float32)],
        {
            "model_names": [
                "pt_codegen_salesforce_codegen_350m_mono_clm_hf",
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "1280", "stop": "1536", "stride": "1"},
        },
    ),
    (
        Index545,
        [((3072, 1024), torch.float32)],
        {
            "model_names": [
                "pt_codegen_salesforce_codegen_350m_mono_clm_hf",
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "2048", "stop": "2304", "stride": "1"},
        },
    ),
    (
        Index546,
        [((3072, 1024), torch.float32)],
        {
            "model_names": [
                "pt_codegen_salesforce_codegen_350m_mono_clm_hf",
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "2816", "stop": "3072", "stride": "1"},
        },
    ),
    (
        Index547,
        [((3072, 1024), torch.float32)],
        {
            "model_names": [
                "pt_codegen_salesforce_codegen_350m_mono_clm_hf",
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "256", "stop": "512", "stride": "1"},
        },
    ),
    (
        Index548,
        [((3072, 1024), torch.float32)],
        {
            "model_names": [
                "pt_codegen_salesforce_codegen_350m_mono_clm_hf",
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "1024", "stop": "1280", "stride": "1"},
        },
    ),
    (
        Index549,
        [((3072, 1024), torch.float32)],
        {
            "model_names": [
                "pt_codegen_salesforce_codegen_350m_mono_clm_hf",
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "1792", "stop": "2048", "stride": "1"},
        },
    ),
    (
        Index550,
        [((3072, 1024), torch.float32)],
        {
            "model_names": [
                "pt_codegen_salesforce_codegen_350m_mono_clm_hf",
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "2560", "stop": "2816", "stride": "1"},
        },
    ),
    (
        Index10,
        [((1, 256, 16, 64), torch.float32)],
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
        Index11,
        [((1, 256, 16, 64), torch.float32)],
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
        Index551,
        [((1, 256, 16, 32), torch.float32)],
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
        Index552,
        [((1, 256, 16, 32), torch.float32)],
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
        Index13,
        [((1, 2, 30, 40), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index14,
        [((1, 2, 30, 40), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index13,
        [((1, 2, 60, 80), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index14,
        [((1, 2, 60, 80), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index13,
        [((1, 2, 120, 160), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index14,
        [((1, 2, 120, 160), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index15,
        [((1, 12, 35, 128), torch.float32)],
        {
            "model_names": [
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "64", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index16,
        [((1, 12, 35, 128), torch.float32)],
        {
            "model_names": [
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index15,
        [((1, 2, 35, 128), torch.float32)],
        {
            "model_names": [
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "64", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index16,
        [((1, 2, 35, 128), torch.float32)],
        {
            "model_names": [
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index537,
        [((3, 64, 3, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index538,
        [((3, 64, 3, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index539,
        [((3, 64, 3, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "2", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index553,
        [((1, 56, 56, 96), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "3", "stop": "56", "stride": "1"},
        },
    ),
    (
        Index554,
        [((1, 56, 56, 96), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index555,
        [((1, 56, 56, 96), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "3", "stop": "56", "stride": "1"},
        },
    ),
    (
        Index556,
        [((1, 56, 56, 96), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index557,
        [((1, 56, 56, 96), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "53", "stop": "56", "stride": "1"},
        },
    ),
    (
        Index558,
        [((1, 56, 56, 96), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "53", "stride": "1"},
        },
    ),
    (
        Index559,
        [((1, 56, 56, 96), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "53", "stop": "56", "stride": "1"},
        },
    ),
    (
        Index560,
        [((1, 56, 56, 96), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "53", "stride": "1"},
        },
    ),
    (
        Index561,
        [((1, 56, 56, 96), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "56", "stride": "2"},
        },
    ),
    (
        Index562,
        [((1, 56, 56, 96), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "1", "stop": "56", "stride": "2"},
        },
    ),
    (
        Index563,
        [((1, 28, 56, 96), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "56", "stride": "2"},
        },
    ),
    (
        Index564,
        [((1, 28, 56, 96), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "1", "stop": "56", "stride": "2"},
        },
    ),
    (
        Index537,
        [((3, 16, 6, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index538,
        [((3, 16, 6, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index539,
        [((3, 16, 6, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "2", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index565,
        [((1, 28, 28, 192), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "3", "stop": "28", "stride": "1"},
        },
    ),
    (
        Index554,
        [((1, 28, 28, 192), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index566,
        [((1, 28, 28, 192), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "3", "stop": "28", "stride": "1"},
        },
    ),
    (
        Index556,
        [((1, 28, 28, 192), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index567,
        [((1, 28, 28, 192), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "25", "stop": "28", "stride": "1"},
        },
    ),
    (
        Index568,
        [((1, 28, 28, 192), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "25", "stride": "1"},
        },
    ),
    (
        Index569,
        [((1, 28, 28, 192), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "25", "stop": "28", "stride": "1"},
        },
    ),
    (
        Index570,
        [((1, 28, 28, 192), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "25", "stride": "1"},
        },
    ),
    (
        Index571,
        [((1, 28, 28, 192), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "28", "stride": "2"},
        },
    ),
    (
        Index572,
        [((1, 28, 28, 192), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "1", "stop": "28", "stride": "2"},
        },
    ),
    (
        Index573,
        [((1, 14, 28, 192), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "28", "stride": "2"},
        },
    ),
    (
        Index574,
        [((1, 14, 28, 192), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "1", "stop": "28", "stride": "2"},
        },
    ),
    (
        Index537,
        [((3, 4, 12, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index538,
        [((3, 4, 12, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index539,
        [((3, 4, 12, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "2", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index575,
        [((1, 14, 14, 384), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "3", "stop": "14", "stride": "1"},
        },
    ),
    (
        Index554,
        [((1, 14, 14, 384), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index576,
        [((1, 14, 14, 384), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "3", "stop": "14", "stride": "1"},
        },
    ),
    (
        Index556,
        [((1, 14, 14, 384), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index577,
        [((1, 14, 14, 384), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "11", "stop": "14", "stride": "1"},
        },
    ),
    (
        Index578,
        [((1, 14, 14, 384), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "11", "stride": "1"},
        },
    ),
    (
        Index579,
        [((1, 14, 14, 384), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "11", "stop": "14", "stride": "1"},
        },
    ),
    (
        Index580,
        [((1, 14, 14, 384), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "11", "stride": "1"},
        },
    ),
    (
        Index581,
        [((1, 14, 14, 384), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "14", "stride": "2"},
        },
    ),
    (
        Index582,
        [((1, 14, 14, 384), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "1", "stop": "14", "stride": "2"},
        },
    ),
    (
        Index583,
        [((1, 7, 14, 384), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "14", "stride": "2"},
        },
    ),
    (
        Index584,
        [((1, 7, 14, 384), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "1", "stop": "14", "stride": "2"},
        },
    ),
    (
        Index537,
        [((3, 1, 24, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index538,
        [((3, 1, 24, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index539,
        [((3, 1, 24, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "2", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index585,
        [((2050, 1024), torch.float32)],
        {
            "model_names": ["pt_xglm_facebook_xglm_564m_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "2", "stop": "258", "stride": "1"},
        },
    ),
    (
        Index6,
        [((1, 13, 384), torch.float32)],
        {
            "model_names": ["onnx_minilm_sentence_transformers_all_minilm_l6_v2_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index586,
        [((1, 280, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w40_pose_estimation_osmr", "pt_hrnet_hrnet_w40_pose_estimation_timm"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "40", "stride": "1"},
        },
    ),
    (
        Index587,
        [((1, 280, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w40_pose_estimation_osmr", "pt_hrnet_hrnet_w40_pose_estimation_timm"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "40", "stop": "120", "stride": "1"},
        },
    ),
    (
        Index588,
        [((1, 280, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w40_pose_estimation_osmr", "pt_hrnet_hrnet_w40_pose_estimation_timm"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "120", "stop": "280", "stride": "1"},
        },
    ),
    (
        Index61,
        [((1, 280, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w40_pose_estimation_osmr", "pt_hrnet_hrnet_w40_pose_estimation_timm"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "160", "stride": "1"},
        },
    ),
    (
        Index589,
        [((1, 280, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w40_pose_estimation_osmr", "pt_hrnet_hrnet_w40_pose_estimation_timm"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "160", "stop": "240", "stride": "1"},
        },
    ),
    (
        Index590,
        [((1, 280, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w40_pose_estimation_osmr", "pt_hrnet_hrnet_w40_pose_estimation_timm"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "240", "stop": "280", "stride": "1"},
        },
    ),
    (
        Index59,
        [((1, 160, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w40_pose_estimation_osmr", "pt_hrnet_hrnet_w40_pose_estimation_timm"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "80", "stride": "1"},
        },
    ),
    (
        Index591,
        [((1, 160, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w40_pose_estimation_osmr", "pt_hrnet_hrnet_w40_pose_estimation_timm"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "80", "stop": "120", "stride": "1"},
        },
    ),
    (
        Index592,
        [((1, 160, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w40_pose_estimation_osmr", "pt_hrnet_hrnet_w40_pose_estimation_timm"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "120", "stop": "160", "stride": "1"},
        },
    ),
    (
        Index586,
        [((1, 160, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w40_pose_estimation_osmr", "pt_hrnet_hrnet_w40_pose_estimation_timm"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "40", "stride": "1"},
        },
    ),
    (
        Index593,
        [((1, 160, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w40_pose_estimation_osmr", "pt_hrnet_hrnet_w40_pose_estimation_timm"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "40", "stop": "80", "stride": "1"},
        },
    ),
    (
        Index60,
        [((1, 160, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w40_pose_estimation_osmr", "pt_hrnet_hrnet_w40_pose_estimation_timm"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "80", "stop": "160", "stride": "1"},
        },
    ),
    (
        Index6,
        [((2, 2048), torch.float32)],
        {
            "model_names": ["pt_opt_facebook_opt_1_3b_qa_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index7,
        [((2, 2048), torch.float32)],
        {
            "model_names": ["pt_opt_facebook_opt_1_3b_qa_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index537,
        [((3, 64, 4, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index538,
        [((3, 64, 4, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index539,
        [((3, 64, 4, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "2", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index553,
        [((1, 56, 56, 128), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "3", "stop": "56", "stride": "1"},
        },
    ),
    (
        Index554,
        [((1, 56, 56, 128), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index555,
        [((1, 56, 56, 128), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "3", "stop": "56", "stride": "1"},
        },
    ),
    (
        Index556,
        [((1, 56, 56, 128), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index557,
        [((1, 56, 56, 128), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "53", "stop": "56", "stride": "1"},
        },
    ),
    (
        Index558,
        [((1, 56, 56, 128), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "53", "stride": "1"},
        },
    ),
    (
        Index559,
        [((1, 56, 56, 128), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "53", "stop": "56", "stride": "1"},
        },
    ),
    (
        Index560,
        [((1, 56, 56, 128), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "53", "stride": "1"},
        },
    ),
    (
        Index561,
        [((1, 56, 56, 128), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "56", "stride": "2"},
        },
    ),
    (
        Index562,
        [((1, 56, 56, 128), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "1", "stop": "56", "stride": "2"},
        },
    ),
    (
        Index563,
        [((1, 28, 56, 128), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "56", "stride": "2"},
        },
    ),
    (
        Index564,
        [((1, 28, 56, 128), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "1", "stop": "56", "stride": "2"},
        },
    ),
    (
        Index537,
        [((3, 16, 8, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index538,
        [((3, 16, 8, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index539,
        [((3, 16, 8, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "2", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index565,
        [((1, 28, 28, 256), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "3", "stop": "28", "stride": "1"},
        },
    ),
    (
        Index554,
        [((1, 28, 28, 256), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index566,
        [((1, 28, 28, 256), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "3", "stop": "28", "stride": "1"},
        },
    ),
    (
        Index556,
        [((1, 28, 28, 256), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index567,
        [((1, 28, 28, 256), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "25", "stop": "28", "stride": "1"},
        },
    ),
    (
        Index568,
        [((1, 28, 28, 256), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "25", "stride": "1"},
        },
    ),
    (
        Index569,
        [((1, 28, 28, 256), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "25", "stop": "28", "stride": "1"},
        },
    ),
    (
        Index570,
        [((1, 28, 28, 256), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "25", "stride": "1"},
        },
    ),
    (
        Index571,
        [((1, 28, 28, 256), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "28", "stride": "2"},
        },
    ),
    (
        Index572,
        [((1, 28, 28, 256), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "1", "stop": "28", "stride": "2"},
        },
    ),
    (
        Index573,
        [((1, 14, 28, 256), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "28", "stride": "2"},
        },
    ),
    (
        Index574,
        [((1, 14, 28, 256), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "1", "stop": "28", "stride": "2"},
        },
    ),
    (
        Index537,
        [((3, 4, 16, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index538,
        [((3, 4, 16, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index539,
        [((3, 4, 16, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "2", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index575,
        [((1, 14, 14, 512), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "3", "stop": "14", "stride": "1"},
        },
    ),
    (
        Index554,
        [((1, 14, 14, 512), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index576,
        [((1, 14, 14, 512), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "3", "stop": "14", "stride": "1"},
        },
    ),
    (
        Index556,
        [((1, 14, 14, 512), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index577,
        [((1, 14, 14, 512), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "11", "stop": "14", "stride": "1"},
        },
    ),
    (
        Index578,
        [((1, 14, 14, 512), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "11", "stride": "1"},
        },
    ),
    (
        Index579,
        [((1, 14, 14, 512), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "11", "stop": "14", "stride": "1"},
        },
    ),
    (
        Index580,
        [((1, 14, 14, 512), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "11", "stride": "1"},
        },
    ),
    (
        Index581,
        [((1, 14, 14, 512), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "14", "stride": "2"},
        },
    ),
    (
        Index582,
        [((1, 14, 14, 512), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "1", "stop": "14", "stride": "2"},
        },
    ),
    (
        Index583,
        [((1, 7, 14, 512), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "14", "stride": "2"},
        },
    ),
    (
        Index584,
        [((1, 7, 14, 512), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "1", "stop": "14", "stride": "2"},
        },
    ),
    (
        Index537,
        [((3, 1, 32, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index538,
        [((3, 1, 32, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index539,
        [((3, 1, 32, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "2", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index585,
        [((2050, 2048), torch.float32)],
        {
            "model_names": ["pt_xglm_facebook_xglm_1_7b_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "2", "stop": "258", "stride": "1"},
        },
    ),
    (
        Index594,
        [((1, 3, 416, 416), torch.bfloat16)],
        {
            "model_names": ["pt_yolox_yolox_tiny_obj_det_torchhub", "pt_yolox_yolox_nano_obj_det_torchhub"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "416", "stride": "2"},
        },
    ),
    (
        Index595,
        [((1, 3, 416, 416), torch.bfloat16)],
        {
            "model_names": ["pt_yolox_yolox_tiny_obj_det_torchhub", "pt_yolox_yolox_nano_obj_det_torchhub"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "1", "stop": "416", "stride": "2"},
        },
    ),
    (
        Index596,
        [((1, 3, 208, 416), torch.bfloat16)],
        {
            "model_names": ["pt_yolox_yolox_tiny_obj_det_torchhub", "pt_yolox_yolox_nano_obj_det_torchhub"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "416", "stride": "2"},
        },
    ),
    (
        Index597,
        [((1, 3, 208, 416), torch.bfloat16)],
        {
            "model_names": ["pt_yolox_yolox_tiny_obj_det_torchhub", "pt_yolox_yolox_nano_obj_det_torchhub"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "1", "stop": "416", "stride": "2"},
        },
    ),
    (
        Index6,
        [((1, 6, 768), torch.float32)],
        {
            "model_names": ["onnx_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index13,
        [((2, 1, 14), torch.float32)],
        {
            "model_names": ["pd_bert_bert_base_japanese_qa_padlenlp"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index14,
        [((2, 1, 14), torch.float32)],
        {
            "model_names": ["pd_bert_bert_base_japanese_qa_padlenlp"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index537,
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
        Index538,
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
        Index539,
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
        Index598,
        [((1, 77), torch.int64)],
        {
            "model_names": ["pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "7", "stride": "1"},
        },
    ),
    (
        Index6,
        [((1, 197, 192), torch.bfloat16)],
        {
            "model_names": ["pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index10,
        [((1, 1, 32, 2048), torch.bool)],
        {
            "model_names": [
                "pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf",
                "pt_gptneo_eleutherai_gpt_neo_125m_seq_cls_hf",
                "pt_gptneo_eleutherai_gpt_neo_2_7b_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index410,
        [((1, 32, 2), torch.float32)],
        {
            "model_names": [
                "pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf",
                "pt_gptneo_eleutherai_gpt_neo_125m_seq_cls_hf",
                "pt_gptneo_eleutherai_gpt_neo_2_7b_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "31", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index496,
        [((1, 224, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w32_pose_estimation_osmr", "pt_hrnet_hrnet_w32_pose_estimation_timm"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index599,
        [((1, 224, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w32_pose_estimation_osmr", "pt_hrnet_hrnet_w32_pose_estimation_timm"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "32", "stop": "96", "stride": "1"},
        },
    ),
    (
        Index600,
        [((1, 224, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w32_pose_estimation_osmr", "pt_hrnet_hrnet_w32_pose_estimation_timm"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "96", "stop": "224", "stride": "1"},
        },
    ),
    (
        Index498,
        [((1, 224, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w32_pose_estimation_osmr", "pt_hrnet_hrnet_w32_pose_estimation_timm"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index601,
        [((1, 224, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w32_pose_estimation_osmr", "pt_hrnet_hrnet_w32_pose_estimation_timm"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "128", "stop": "192", "stride": "1"},
        },
    ),
    (
        Index602,
        [((1, 224, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w32_pose_estimation_osmr", "pt_hrnet_hrnet_w32_pose_estimation_timm"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "192", "stop": "224", "stride": "1"},
        },
    ),
    (
        Index26,
        [((1, 128, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w32_pose_estimation_osmr", "pt_hrnet_hrnet_w32_pose_estimation_timm"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index27,
        [((1, 128, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w32_pose_estimation_osmr", "pt_hrnet_hrnet_w32_pose_estimation_timm"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "64", "stop": "96", "stride": "1"},
        },
    ),
    (
        Index603,
        [((1, 128, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w32_pose_estimation_osmr", "pt_hrnet_hrnet_w32_pose_estimation_timm"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "96", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index496,
        [((1, 128, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w32_pose_estimation_osmr", "pt_hrnet_hrnet_w32_pose_estimation_timm"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index25,
        [((1, 128, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w32_pose_estimation_osmr", "pt_hrnet_hrnet_w32_pose_estimation_timm"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "32", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index497,
        [((1, 128, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w32_pose_estimation_osmr", "pt_hrnet_hrnet_w32_pose_estimation_timm"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "64", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index11,
        [((1, 32, 4, 64), torch.float32)],
        {
            "model_names": [
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "32", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index10,
        [((1, 32, 4, 64), torch.float32)],
        {
            "model_names": [
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index11,
        [((1, 8, 4, 64), torch.float32)],
        {
            "model_names": [
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "32", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index10,
        [((1, 8, 4, 64), torch.float32)],
        {
            "model_names": [
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index45,
        [((1, 4, 2), torch.float32)],
        {
            "model_names": [
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_3b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_3b_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "3", "stop": "4", "stride": "1"},
        },
    ),
    (
        Index604,
        [((1024, 96), torch.float32)],
        {
            "model_names": ["pt_nbeats_generic_basis_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "-24", "stop": "96", "stride": "1"},
        },
    ),
    (
        Index605,
        [((1024, 96), torch.float32)],
        {
            "model_names": ["pt_nbeats_generic_basis_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "72", "stride": "1"},
        },
    ),
    (
        Index11,
        [((1, 14, 29, 64), torch.float32)],
        {
            "model_names": ["pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "32", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index10,
        [((1, 14, 29, 64), torch.float32)],
        {
            "model_names": ["pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index11,
        [((1, 2, 29, 64), torch.float32)],
        {
            "model_names": ["pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "32", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index10,
        [((1, 2, 29, 64), torch.float32)],
        {
            "model_names": ["pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index6,
        [((2, 4, 1), torch.int64)],
        {
            "model_names": [
                "pt_stereo_facebook_musicgen_small_music_generation_hf",
                "pt_stereo_facebook_musicgen_large_music_generation_hf",
                "pt_stereo_facebook_musicgen_medium_music_generation_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index7,
        [((2, 4, 1), torch.int64)],
        {
            "model_names": [
                "pt_stereo_facebook_musicgen_small_music_generation_hf",
                "pt_stereo_facebook_musicgen_large_music_generation_hf",
                "pt_stereo_facebook_musicgen_medium_music_generation_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index44,
        [((2, 4, 1), torch.int64)],
        {
            "model_names": [
                "pt_stereo_facebook_musicgen_small_music_generation_hf",
                "pt_stereo_facebook_musicgen_large_music_generation_hf",
                "pt_stereo_facebook_musicgen_medium_music_generation_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "2", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index45,
        [((2, 4, 1), torch.int64)],
        {
            "model_names": [
                "pt_stereo_facebook_musicgen_small_music_generation_hf",
                "pt_stereo_facebook_musicgen_large_music_generation_hf",
                "pt_stereo_facebook_musicgen_medium_music_generation_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "3", "stop": "4", "stride": "1"},
        },
    ),
    (
        Index6,
        [((2048, 1024), torch.float32)],
        {
            "model_names": ["pt_stereo_facebook_musicgen_small_music_generation_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index54,
        [((3, 197, 1, 1024), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_l_16_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index55,
        [((3, 197, 1, 1024), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_l_16_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index56,
        [((3, 197, 1, 1024), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_l_16_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "2", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index6,
        [((2, 1024), torch.float32)],
        {
            "model_names": [
                "onnx_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                "pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index7,
        [((2, 1024), torch.float32)],
        {
            "model_names": [
                "onnx_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                "pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index606,
        [((1, 64, 64, 96), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "4", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index607,
        [((1, 64, 64, 96), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "4", "stride": "1"},
        },
    ),
    (
        Index608,
        [((1, 64, 64, 96), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "4", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index609,
        [((1, 64, 64, 96), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "4", "stride": "1"},
        },
    ),
    (
        Index610,
        [((1, 64, 64, 96), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "-4", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index611,
        [((1, 64, 64, 96), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "-4", "stride": "1"},
        },
    ),
    (
        Index612,
        [((1, 64, 64, 96), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "-4", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index613,
        [((1, 64, 64, 96), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "-4", "stride": "1"},
        },
    ),
    (
        Index614,
        [((1, 64, 64, 96), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "64", "stride": "2"},
        },
    ),
    (
        Index615,
        [((1, 64, 64, 96), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "1", "stop": "64", "stride": "2"},
        },
    ),
    (
        Index616,
        [((1, 64, 64, 96), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "60", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index617,
        [((1, 64, 64, 96), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "60", "stride": "1"},
        },
    ),
    (
        Index618,
        [((1, 64, 64, 96), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "60", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index619,
        [((1, 64, 64, 96), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "60", "stride": "1"},
        },
    ),
    (
        Index620,
        [((1, 32, 64, 96), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "64", "stride": "2"},
        },
    ),
    (
        Index621,
        [((1, 32, 64, 96), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "1", "stop": "64", "stride": "2"},
        },
    ),
    (
        Index622,
        [((1, 32, 32, 192), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "4", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index607,
        [((1, 32, 32, 192), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "4", "stride": "1"},
        },
    ),
    (
        Index623,
        [((1, 32, 32, 192), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "4", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index609,
        [((1, 32, 32, 192), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "4", "stride": "1"},
        },
    ),
    (
        Index624,
        [((1, 32, 32, 192), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "-4", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index611,
        [((1, 32, 32, 192), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "-4", "stride": "1"},
        },
    ),
    (
        Index625,
        [((1, 32, 32, 192), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "-4", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index613,
        [((1, 32, 32, 192), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "-4", "stride": "1"},
        },
    ),
    (
        Index626,
        [((1, 32, 32, 192), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "32", "stride": "2"},
        },
    ),
    (
        Index627,
        [((1, 32, 32, 192), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "1", "stop": "32", "stride": "2"},
        },
    ),
    (
        Index628,
        [((1, 32, 32, 192), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "28", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index629,
        [((1, 32, 32, 192), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "28", "stride": "1"},
        },
    ),
    (
        Index630,
        [((1, 32, 32, 192), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "28", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index631,
        [((1, 32, 32, 192), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "28", "stride": "1"},
        },
    ),
    (
        Index632,
        [((1, 16, 32, 192), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "32", "stride": "2"},
        },
    ),
    (
        Index633,
        [((1, 16, 32, 192), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "1", "stop": "32", "stride": "2"},
        },
    ),
    (
        Index634,
        [((1, 16, 16, 384), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "4", "stop": "16", "stride": "1"},
        },
    ),
    (
        Index607,
        [((1, 16, 16, 384), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "4", "stride": "1"},
        },
    ),
    (
        Index635,
        [((1, 16, 16, 384), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "4", "stop": "16", "stride": "1"},
        },
    ),
    (
        Index609,
        [((1, 16, 16, 384), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "4", "stride": "1"},
        },
    ),
    (
        Index636,
        [((1, 16, 16, 384), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "-4", "stop": "16", "stride": "1"},
        },
    ),
    (
        Index611,
        [((1, 16, 16, 384), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "-4", "stride": "1"},
        },
    ),
    (
        Index637,
        [((1, 16, 16, 384), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "-4", "stop": "16", "stride": "1"},
        },
    ),
    (
        Index613,
        [((1, 16, 16, 384), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "-4", "stride": "1"},
        },
    ),
    (
        Index638,
        [((1, 16, 16, 384), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "16", "stride": "2"},
        },
    ),
    (
        Index639,
        [((1, 16, 16, 384), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "1", "stop": "16", "stride": "2"},
        },
    ),
    (
        Index640,
        [((1, 16, 16, 384), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "12", "stop": "16", "stride": "1"},
        },
    ),
    (
        Index641,
        [((1, 16, 16, 384), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "12", "stride": "1"},
        },
    ),
    (
        Index642,
        [((1, 16, 16, 384), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "12", "stop": "16", "stride": "1"},
        },
    ),
    (
        Index643,
        [((1, 16, 16, 384), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "12", "stride": "1"},
        },
    ),
    (
        Index644,
        [((1, 8, 16, 384), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "16", "stride": "2"},
        },
    ),
    (
        Index645,
        [((1, 8, 16, 384), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "1", "stop": "16", "stride": "2"},
        },
    ),
    (
        Index6,
        [((1, 15, 768), torch.float32)],
        {
            "model_names": ["pd_bert_bert_base_japanese_seq_cls_padlenlp"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index646,
        [((1, 126, 7, 7), torch.bfloat16)],
        {
            "model_names": [
                "pt_hrnet_hrnet_w18_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "18", "stride": "1"},
        },
    ),
    (
        Index647,
        [((1, 126, 7, 7), torch.bfloat16)],
        {
            "model_names": [
                "pt_hrnet_hrnet_w18_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "18", "stop": "54", "stride": "1"},
        },
    ),
    (
        Index648,
        [((1, 126, 7, 7), torch.bfloat16)],
        {
            "model_names": [
                "pt_hrnet_hrnet_w18_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "54", "stop": "126", "stride": "1"},
        },
    ),
    (
        Index649,
        [((1, 126, 7, 7), torch.bfloat16)],
        {
            "model_names": [
                "pt_hrnet_hrnet_w18_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "72", "stride": "1"},
        },
    ),
    (
        Index650,
        [((1, 126, 7, 7), torch.bfloat16)],
        {
            "model_names": [
                "pt_hrnet_hrnet_w18_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "72", "stop": "108", "stride": "1"},
        },
    ),
    (
        Index651,
        [((1, 126, 7, 7), torch.bfloat16)],
        {
            "model_names": [
                "pt_hrnet_hrnet_w18_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "108", "stop": "126", "stride": "1"},
        },
    ),
    (
        Index652,
        [((1, 72, 28, 28), torch.bfloat16)],
        {
            "model_names": [
                "pt_hrnet_hrnet_w18_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "36", "stride": "1"},
        },
    ),
    (
        Index653,
        [((1, 72, 28, 28), torch.bfloat16)],
        {
            "model_names": [
                "pt_hrnet_hrnet_w18_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "36", "stop": "54", "stride": "1"},
        },
    ),
    (
        Index654,
        [((1, 72, 28, 28), torch.bfloat16)],
        {
            "model_names": [
                "pt_hrnet_hrnet_w18_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "54", "stop": "72", "stride": "1"},
        },
    ),
    (
        Index646,
        [((1, 72, 28, 28), torch.bfloat16)],
        {
            "model_names": [
                "pt_hrnet_hrnet_w18_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "18", "stride": "1"},
        },
    ),
    (
        Index655,
        [((1, 72, 28, 28), torch.bfloat16)],
        {
            "model_names": [
                "pt_hrnet_hrnet_w18_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "18", "stop": "36", "stride": "1"},
        },
    ),
    (
        Index656,
        [((1, 72, 28, 28), torch.bfloat16)],
        {
            "model_names": [
                "pt_hrnet_hrnet_w18_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "36", "stop": "72", "stride": "1"},
        },
    ),
    (
        Index498,
        [((1, 256, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w64_pose_estimation_osmr", "pt_hrnet_hrnet_w64_pose_estimation_timm"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index601,
        [((1, 256, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w64_pose_estimation_osmr", "pt_hrnet_hrnet_w64_pose_estimation_timm"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "128", "stop": "192", "stride": "1"},
        },
    ),
    (
        Index657,
        [((1, 256, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w64_pose_estimation_osmr", "pt_hrnet_hrnet_w64_pose_estimation_timm"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "192", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index26,
        [((1, 256, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w64_pose_estimation_osmr", "pt_hrnet_hrnet_w64_pose_estimation_timm"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index497,
        [((1, 256, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w64_pose_estimation_osmr", "pt_hrnet_hrnet_w64_pose_estimation_timm"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "64", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index499,
        [((1, 256, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w64_pose_estimation_osmr", "pt_hrnet_hrnet_w64_pose_estimation_timm"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "128", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index11,
        [((1, 32, 256, 64), torch.float32)],
        {
            "model_names": [
                "pt_llama3_meta_llama_llama_3_2_1b_clm_hf",
                "pt_phi1_microsoft_phi_1_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf",
                "pt_phi_1_5_microsoft_phi_1_5_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "32", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index10,
        [((1, 32, 256, 64), torch.float32)],
        {
            "model_names": [
                "pt_llama3_meta_llama_llama_3_2_1b_clm_hf",
                "pt_phi1_microsoft_phi_1_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf",
                "pt_phi_1_5_microsoft_phi_1_5_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index11,
        [((1, 8, 256, 64), torch.float32)],
        {
            "model_names": [
                "pt_llama3_meta_llama_llama_3_2_1b_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "32", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index10,
        [((1, 8, 256, 64), torch.float32)],
        {
            "model_names": [
                "pt_llama3_meta_llama_llama_3_2_1b_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index658,
        [((1, 4096, 6), torch.float32)],
        {
            "model_names": ["pt_mamba_state_spaces_mamba_370m_hf_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "2048", "stride": "1"},
        },
    ),
    (
        Index659,
        [((1, 4096, 6), torch.float32)],
        {
            "model_names": ["pt_mamba_state_spaces_mamba_370m_hf_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "2048", "stop": "4096", "stride": "1"},
        },
    ),
    (
        Index40,
        [((1, 2048, 9), torch.float32)],
        {
            "model_names": ["pt_mamba_state_spaces_mamba_370m_hf_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "6", "stride": "1"},
        },
    ),
    (
        Index68,
        [((96, 2048), torch.float32)],
        {
            "model_names": ["pt_mamba_state_spaces_mamba_370m_hf_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index660,
        [((96, 2048), torch.float32)],
        {
            "model_names": ["pt_mamba_state_spaces_mamba_370m_hf_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "64", "stop": "80", "stride": "1"},
        },
    ),
    (
        Index661,
        [((96, 2048), torch.float32)],
        {
            "model_names": ["pt_mamba_state_spaces_mamba_370m_hf_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "80", "stop": "96", "stride": "1"},
        },
    ),
    (
        Index6,
        [((1, 2048, 6, 16), torch.float32)],
        {
            "model_names": ["pt_mamba_state_spaces_mamba_370m_hf_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index7,
        [((1, 2048, 6, 16), torch.float32)],
        {
            "model_names": ["pt_mamba_state_spaces_mamba_370m_hf_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index44,
        [((1, 2048, 6, 16), torch.float32)],
        {
            "model_names": ["pt_mamba_state_spaces_mamba_370m_hf_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "2", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index45,
        [((1, 2048, 6, 16), torch.float32)],
        {
            "model_names": ["pt_mamba_state_spaces_mamba_370m_hf_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "3", "stop": "4", "stride": "1"},
        },
    ),
    (
        Index46,
        [((1, 2048, 6, 16), torch.float32)],
        {
            "model_names": ["pt_mamba_state_spaces_mamba_370m_hf_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "4", "stop": "5", "stride": "1"},
        },
    ),
    (
        Index47,
        [((1, 2048, 6, 16), torch.float32)],
        {
            "model_names": ["pt_mamba_state_spaces_mamba_370m_hf_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "5", "stop": "6", "stride": "1"},
        },
    ),
    (
        Index11,
        [((1, 14, 35, 64), torch.float32)],
        {
            "model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "32", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index10,
        [((1, 14, 35, 64), torch.float32)],
        {
            "model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index11,
        [((1, 2, 35, 64), torch.float32)],
        {
            "model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "32", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index10,
        [((1, 2, 35, 64), torch.float32)],
        {
            "model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index6,
        [((1, 201, 768), torch.bfloat16)],
        {
            "model_names": ["pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index54,
        [((3, 50, 1, 768), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_b_32_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index55,
        [((3, 50, 1, 768), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_b_32_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index56,
        [((3, 50, 1, 768), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_b_32_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "2", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index6,
        [((1, 50, 768), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_b_32_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index35,
        [((1, 512, 20, 20), torch.bfloat16)],
        {
            "model_names": ["pt_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index662,
        [((1, 512, 20, 20), torch.bfloat16)],
        {
            "model_names": ["pt_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "256", "stop": "512", "stride": "1"},
        },
    ),
    (
        Index6,
        [((1, 197, 768), torch.float32)],
        {
            "model_names": ["onnx_vit_base_google_vit_base_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index6,
        [((1, 16, 768), torch.float32)],
        {
            "model_names": ["pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index663,
        [((1, 336, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w48_pose_estimation_osmr", "pt_hrnet_hrnet_w48_pose_estimation_timm"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "48", "stride": "1"},
        },
    ),
    (
        Index664,
        [((1, 336, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w48_pose_estimation_osmr", "pt_hrnet_hrnet_w48_pose_estimation_timm"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "48", "stop": "144", "stride": "1"},
        },
    ),
    (
        Index665,
        [((1, 336, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w48_pose_estimation_osmr", "pt_hrnet_hrnet_w48_pose_estimation_timm"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "144", "stop": "336", "stride": "1"},
        },
    ),
    (
        Index503,
        [((1, 336, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w48_pose_estimation_osmr", "pt_hrnet_hrnet_w48_pose_estimation_timm"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "192", "stride": "1"},
        },
    ),
    (
        Index504,
        [((1, 336, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w48_pose_estimation_osmr", "pt_hrnet_hrnet_w48_pose_estimation_timm"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "192", "stop": "288", "stride": "1"},
        },
    ),
    (
        Index666,
        [((1, 336, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w48_pose_estimation_osmr", "pt_hrnet_hrnet_w48_pose_estimation_timm"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "288", "stop": "336", "stride": "1"},
        },
    ),
    (
        Index29,
        [((1, 192, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w48_pose_estimation_osmr", "pt_hrnet_hrnet_w48_pose_estimation_timm"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "96", "stride": "1"},
        },
    ),
    (
        Index667,
        [((1, 192, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w48_pose_estimation_osmr", "pt_hrnet_hrnet_w48_pose_estimation_timm"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "96", "stop": "144", "stride": "1"},
        },
    ),
    (
        Index668,
        [((1, 192, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w48_pose_estimation_osmr", "pt_hrnet_hrnet_w48_pose_estimation_timm"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "144", "stop": "192", "stride": "1"},
        },
    ),
    (
        Index663,
        [((1, 192, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w48_pose_estimation_osmr", "pt_hrnet_hrnet_w48_pose_estimation_timm"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "48", "stride": "1"},
        },
    ),
    (
        Index669,
        [((1, 192, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w48_pose_estimation_osmr", "pt_hrnet_hrnet_w48_pose_estimation_timm"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "48", "stop": "96", "stride": "1"},
        },
    ),
    (
        Index670,
        [((1, 192, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnetv2_w48_pose_estimation_osmr", "pt_hrnet_hrnet_w48_pose_estimation_timm"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "96", "stop": "192", "stride": "1"},
        },
    ),
    (
        Index537,
        [((3, 64, 4, 64, 32), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index538,
        [((3, 64, 4, 64, 32), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index539,
        [((3, 64, 4, 64, 32), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "2", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index606,
        [((1, 64, 64, 128), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "4", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index607,
        [((1, 64, 64, 128), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "4", "stride": "1"},
        },
    ),
    (
        Index608,
        [((1, 64, 64, 128), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "4", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index609,
        [((1, 64, 64, 128), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "4", "stride": "1"},
        },
    ),
    (
        Index616,
        [((1, 64, 64, 128), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "60", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index617,
        [((1, 64, 64, 128), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "60", "stride": "1"},
        },
    ),
    (
        Index618,
        [((1, 64, 64, 128), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "60", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index619,
        [((1, 64, 64, 128), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "60", "stride": "1"},
        },
    ),
    (
        Index614,
        [((1, 64, 64, 128), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "64", "stride": "2"},
        },
    ),
    (
        Index615,
        [((1, 64, 64, 128), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "1", "stop": "64", "stride": "2"},
        },
    ),
    (
        Index620,
        [((1, 32, 64, 128), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "64", "stride": "2"},
        },
    ),
    (
        Index621,
        [((1, 32, 64, 128), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "1", "stop": "64", "stride": "2"},
        },
    ),
    (
        Index537,
        [((3, 16, 8, 64, 32), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index538,
        [((3, 16, 8, 64, 32), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index539,
        [((3, 16, 8, 64, 32), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "2", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index622,
        [((1, 32, 32, 256), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "4", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index607,
        [((1, 32, 32, 256), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "4", "stride": "1"},
        },
    ),
    (
        Index623,
        [((1, 32, 32, 256), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "4", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index609,
        [((1, 32, 32, 256), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "4", "stride": "1"},
        },
    ),
    (
        Index628,
        [((1, 32, 32, 256), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "28", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index629,
        [((1, 32, 32, 256), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "28", "stride": "1"},
        },
    ),
    (
        Index630,
        [((1, 32, 32, 256), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "28", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index631,
        [((1, 32, 32, 256), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "28", "stride": "1"},
        },
    ),
    (
        Index626,
        [((1, 32, 32, 256), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "32", "stride": "2"},
        },
    ),
    (
        Index627,
        [((1, 32, 32, 256), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "1", "stop": "32", "stride": "2"},
        },
    ),
    (
        Index632,
        [((1, 16, 32, 256), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "32", "stride": "2"},
        },
    ),
    (
        Index633,
        [((1, 16, 32, 256), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "1", "stop": "32", "stride": "2"},
        },
    ),
    (
        Index537,
        [((3, 4, 16, 64, 32), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index538,
        [((3, 4, 16, 64, 32), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index539,
        [((3, 4, 16, 64, 32), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "2", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index634,
        [((1, 16, 16, 512), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "4", "stop": "16", "stride": "1"},
        },
    ),
    (
        Index607,
        [((1, 16, 16, 512), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "4", "stride": "1"},
        },
    ),
    (
        Index635,
        [((1, 16, 16, 512), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "4", "stop": "16", "stride": "1"},
        },
    ),
    (
        Index609,
        [((1, 16, 16, 512), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "4", "stride": "1"},
        },
    ),
    (
        Index640,
        [((1, 16, 16, 512), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "12", "stop": "16", "stride": "1"},
        },
    ),
    (
        Index641,
        [((1, 16, 16, 512), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "12", "stride": "1"},
        },
    ),
    (
        Index642,
        [((1, 16, 16, 512), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "12", "stop": "16", "stride": "1"},
        },
    ),
    (
        Index643,
        [((1, 16, 16, 512), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "12", "stride": "1"},
        },
    ),
    (
        Index638,
        [((1, 16, 16, 512), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "16", "stride": "2"},
        },
    ),
    (
        Index639,
        [((1, 16, 16, 512), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "1", "stop": "16", "stride": "2"},
        },
    ),
    (
        Index644,
        [((1, 8, 16, 512), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "16", "stride": "2"},
        },
    ),
    (
        Index645,
        [((1, 8, 16, 512), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "1", "stop": "16", "stride": "2"},
        },
    ),
    (
        Index537,
        [((3, 1, 32, 64, 32), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index538,
        [((3, 1, 32, 64, 32), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index539,
        [((3, 1, 32, 64, 32), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "2", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index537,
        [((3, 64, 3, 64, 32), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index538,
        [((3, 64, 3, 64, 32), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index539,
        [((3, 64, 3, 64, 32), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "2", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index537,
        [((3, 16, 6, 64, 32), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index538,
        [((3, 16, 6, 64, 32), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index539,
        [((3, 16, 6, 64, 32), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "2", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index537,
        [((3, 4, 12, 64, 32), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index538,
        [((3, 4, 12, 64, 32), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index539,
        [((3, 4, 12, 64, 32), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "2", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index537,
        [((3, 1, 24, 64, 32), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index538,
        [((3, 1, 24, 64, 32), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index539,
        [((3, 1, 24, 64, 32), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-5", "start": "2", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index6,
        [((1, 204, 768), torch.bfloat16)],
        {
            "model_names": ["pt_vilt_dandelin_vilt_b32_mlm_mlm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index54,
        [((3, 1370, 1, 1280), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_h_14_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index55,
        [((3, 1370, 1, 1280), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_h_14_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index56,
        [((3, 1370, 1, 1280), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_h_14_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "2", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index6,
        [((1, 1370, 1280), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_h_14_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index6,
        [((1, 4251, 192), torch.bfloat16)],
        {
            "model_names": ["pt_yolos_hustvl_yolos_tiny_obj_det_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index671,
        [((1, 4251, 192), torch.bfloat16)],
        {
            "model_names": ["pt_yolos_hustvl_yolos_tiny_obj_det_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "1", "stop": "-100", "stride": "1"},
        },
    ),
    (
        Index672,
        [((1, 4251, 192), torch.bfloat16)],
        {
            "model_names": ["pt_yolos_hustvl_yolos_tiny_obj_det_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "-100", "stop": "4251", "stride": "1"},
        },
    ),
    (
        Index673,
        [((1, 1445, 192), torch.bfloat16)],
        {
            "model_names": ["pt_yolos_hustvl_yolos_tiny_obj_det_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "-100", "stop": "1445", "stride": "1"},
        },
    ),
    (
        Index13,
        [((2, 1, 11), torch.float32)],
        {
            "model_names": ["pd_bert_chinese_roberta_base_qa_padlenlp"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index14,
        [((2, 1, 11), torch.float32)],
        {
            "model_names": ["pd_bert_chinese_roberta_base_qa_padlenlp"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index674,
        [((1, 8, 522, 256), torch.float32)],
        {
            "model_names": ["pt_falcon3_tiiuae_falcon3_1b_base_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "128", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index0,
        [((1, 8, 522, 256), torch.float32)],
        {
            "model_names": ["pt_falcon3_tiiuae_falcon3_1b_base_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index674,
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
        Index0,
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
        Index674,
        [((1, 8, 207, 256), torch.float32)],
        {
            "model_names": ["pt_gemma_google_gemma_2_2b_it_qa_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "128", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index0,
        [((1, 8, 207, 256), torch.float32)],
        {
            "model_names": ["pt_gemma_google_gemma_2_2b_it_qa_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index674,
        [((1, 4, 207, 256), torch.float32)],
        {
            "model_names": ["pt_gemma_google_gemma_2_2b_it_qa_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "128", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index0,
        [((1, 4, 207, 256), torch.float32)],
        {
            "model_names": ["pt_gemma_google_gemma_2_2b_it_qa_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index675,
        [((1, 210, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w30_pose_estimation_timm", "pt_hrnet_hrnetv2_w30_pose_estimation_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "30", "stride": "1"},
        },
    ),
    (
        Index676,
        [((1, 210, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w30_pose_estimation_timm", "pt_hrnet_hrnetv2_w30_pose_estimation_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "30", "stop": "90", "stride": "1"},
        },
    ),
    (
        Index677,
        [((1, 210, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w30_pose_estimation_timm", "pt_hrnet_hrnetv2_w30_pose_estimation_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "90", "stop": "210", "stride": "1"},
        },
    ),
    (
        Index678,
        [((1, 210, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w30_pose_estimation_timm", "pt_hrnet_hrnetv2_w30_pose_estimation_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "120", "stride": "1"},
        },
    ),
    (
        Index679,
        [((1, 210, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w30_pose_estimation_timm", "pt_hrnet_hrnetv2_w30_pose_estimation_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "120", "stop": "180", "stride": "1"},
        },
    ),
    (
        Index680,
        [((1, 210, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w30_pose_estimation_timm", "pt_hrnet_hrnetv2_w30_pose_estimation_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "180", "stop": "210", "stride": "1"},
        },
    ),
    (
        Index617,
        [((1, 120, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w30_pose_estimation_timm", "pt_hrnet_hrnetv2_w30_pose_estimation_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "60", "stride": "1"},
        },
    ),
    (
        Index681,
        [((1, 120, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w30_pose_estimation_timm", "pt_hrnet_hrnetv2_w30_pose_estimation_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "60", "stop": "90", "stride": "1"},
        },
    ),
    (
        Index682,
        [((1, 120, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w30_pose_estimation_timm", "pt_hrnet_hrnetv2_w30_pose_estimation_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "90", "stop": "120", "stride": "1"},
        },
    ),
    (
        Index675,
        [((1, 120, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w30_pose_estimation_timm", "pt_hrnet_hrnetv2_w30_pose_estimation_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "30", "stride": "1"},
        },
    ),
    (
        Index683,
        [((1, 120, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w30_pose_estimation_timm", "pt_hrnet_hrnetv2_w30_pose_estimation_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "30", "stop": "60", "stride": "1"},
        },
    ),
    (
        Index684,
        [((1, 120, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w30_pose_estimation_timm", "pt_hrnet_hrnetv2_w30_pose_estimation_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "60", "stop": "120", "stride": "1"},
        },
    ),
    (
        Index685,
        [((1, 256, 2), torch.float32)],
        {
            "model_names": ["pt_phi1_microsoft_phi_1_seq_cls_hf", "pt_phi_1_5_microsoft_phi_1_5_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "255", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index0,
        [((1, 514), torch.int64)],
        {
            "model_names": [
                "pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf",
                "pt_roberta_xlm_roberta_base_mlm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index54,
        [((3, 197, 1, 768), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_b_16_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index55,
        [((3, 197, 1, 768), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_b_16_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index56,
        [((3, 197, 1, 768), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_b_16_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-4", "start": "2", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index26,
        [((1, 128, 160, 160), torch.bfloat16)],
        {
            "model_names": ["pt_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index497,
        [((1, 128, 160, 160), torch.bfloat16)],
        {
            "model_names": ["pt_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "64", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index498,
        [((1, 256, 159, 159), torch.bfloat16)],
        {
            "model_names": ["pt_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index499,
        [((1, 256, 159, 159), torch.bfloat16)],
        {
            "model_names": ["pt_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "128", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index498,
        [((1, 256, 80, 80), torch.bfloat16)],
        {
            "model_names": ["pt_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index499,
        [((1, 256, 80, 80), torch.bfloat16)],
        {
            "model_names": ["pt_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "128", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index35,
        [((1, 512, 79, 79), torch.bfloat16)],
        {
            "model_names": ["pt_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index662,
        [((1, 512, 79, 79), torch.bfloat16)],
        {
            "model_names": ["pt_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "256", "stop": "512", "stride": "1"},
        },
    ),
    (
        Index35,
        [((1, 512, 40, 40), torch.bfloat16)],
        {
            "model_names": ["pt_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index662,
        [((1, 512, 40, 40), torch.bfloat16)],
        {
            "model_names": ["pt_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "256", "stop": "512", "stride": "1"},
        },
    ),
    (
        Index35,
        [((1, 512, 39, 39), torch.bfloat16)],
        {
            "model_names": ["pt_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index662,
        [((1, 512, 39, 39), torch.bfloat16)],
        {
            "model_names": ["pt_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "256", "stop": "512", "stride": "1"},
        },
    ),
    (
        Index498,
        [((1, 256, 79, 79), torch.bfloat16)],
        {
            "model_names": ["pt_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index499,
        [((1, 256, 79, 79), torch.bfloat16)],
        {
            "model_names": ["pt_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "128", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index6,
        [((1, 197, 1024), torch.float32)],
        {
            "model_names": ["onnx_vit_base_google_vit_large_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index15,
        [((1, 24, 44, 128), torch.float32)],
        {
            "model_names": [
                "pt_cogito_deepcogito_cogito_v1_preview_llama_3b_text_gen_hf",
                "onnx_cogito_deepcogito_cogito_v1_preview_llama_3b_text_gen_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "64", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index16,
        [((1, 24, 44, 128), torch.float32)],
        {
            "model_names": [
                "pt_cogito_deepcogito_cogito_v1_preview_llama_3b_text_gen_hf",
                "onnx_cogito_deepcogito_cogito_v1_preview_llama_3b_text_gen_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index15,
        [((1, 8, 44, 128), torch.float32)],
        {
            "model_names": [
                "pt_cogito_deepcogito_cogito_v1_preview_llama_3b_text_gen_hf",
                "onnx_cogito_deepcogito_cogito_v1_preview_llama_3b_text_gen_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "64", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index16,
        [((1, 8, 44, 128), torch.float32)],
        {
            "model_names": [
                "pt_cogito_deepcogito_cogito_v1_preview_llama_3b_text_gen_hf",
                "onnx_cogito_deepcogito_cogito_v1_preview_llama_3b_text_gen_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index15,
        [((1, 32, 39, 128), torch.float32)],
        {
            "model_names": ["pt_deepseek_deepseek_math_7b_instruct_qa_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "64", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index16,
        [((1, 32, 39, 128), torch.float32)],
        {
            "model_names": ["pt_deepseek_deepseek_math_7b_instruct_qa_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index686,
        [((1, 6, 73, 64), torch.float32)],
        {
            "model_names": ["pt_falcon_tiiuae_falcon_7b_instruct_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "-2", "stride": "1"},
        },
    ),
    (
        Index687,
        [((1, 6, 73, 64), torch.float32)],
        {
            "model_names": ["pt_falcon_tiiuae_falcon_7b_instruct_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "-2", "stop": "-1", "stride": "1"},
        },
    ),
    (
        Index688,
        [((1, 6, 73, 64), torch.float32)],
        {
            "model_names": ["pt_falcon_tiiuae_falcon_7b_instruct_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "72", "stop": "73", "stride": "1"},
        },
    ),
    (
        Index11,
        [((1, 71, 6, 64), torch.float32)],
        {
            "model_names": ["pt_falcon_tiiuae_falcon_7b_instruct_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "32", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index10,
        [((1, 71, 6, 64), torch.float32)],
        {
            "model_names": ["pt_falcon_tiiuae_falcon_7b_instruct_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index11,
        [((1, 1, 6, 64), torch.float32)],
        {
            "model_names": ["pt_falcon_tiiuae_falcon_7b_instruct_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "32", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index10,
        [((1, 1, 6, 64), torch.float32)],
        {
            "model_names": ["pt_falcon_tiiuae_falcon_7b_instruct_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index6,
        [((1, 334, 64, 3, 64), torch.float32)],
        {
            "model_names": ["pt_fuyu_adept_fuyu_8b_qa_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index7,
        [((1, 334, 64, 3, 64), torch.float32)],
        {
            "model_names": ["pt_fuyu_adept_fuyu_8b_qa_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index44,
        [((1, 334, 64, 3, 64), torch.float32)],
        {
            "model_names": ["pt_fuyu_adept_fuyu_8b_qa_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "2", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index10,
        [((1, 64, 334, 64), torch.float32)],
        {
            "model_names": ["pt_fuyu_adept_fuyu_8b_qa_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index11,
        [((1, 64, 334, 64), torch.float32)],
        {
            "model_names": ["pt_fuyu_adept_fuyu_8b_qa_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "32", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index12,
        [((1, 64, 334, 32), torch.float32)],
        {
            "model_names": ["pt_fuyu_adept_fuyu_8b_qa_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "16", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index5,
        [((1, 64, 334, 32), torch.float32)],
        {
            "model_names": ["pt_fuyu_adept_fuyu_8b_qa_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "16", "stride": "1"},
        },
    ),
    (
        Index528,
        [((1, 308, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w44_pose_estimation_timm", "pt_hrnet_hrnetv2_w44_pose_estimation_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "44", "stride": "1"},
        },
    ),
    (
        Index689,
        [((1, 308, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w44_pose_estimation_timm", "pt_hrnet_hrnetv2_w44_pose_estimation_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "44", "stop": "132", "stride": "1"},
        },
    ),
    (
        Index690,
        [((1, 308, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w44_pose_estimation_timm", "pt_hrnet_hrnetv2_w44_pose_estimation_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "132", "stop": "308", "stride": "1"},
        },
    ),
    (
        Index691,
        [((1, 308, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w44_pose_estimation_timm", "pt_hrnet_hrnetv2_w44_pose_estimation_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "0", "stop": "176", "stride": "1"},
        },
    ),
    (
        Index692,
        [((1, 308, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w44_pose_estimation_timm", "pt_hrnet_hrnetv2_w44_pose_estimation_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "176", "stop": "264", "stride": "1"},
        },
    ),
    (
        Index693,
        [((1, 308, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w44_pose_estimation_timm", "pt_hrnet_hrnetv2_w44_pose_estimation_osmr"],
            "pcc": 0.99,
            "args": {"dim": "-3", "start": "264", "stop": "308", "stride": "1"},
        },
    ),
    (
        Index15,
        [((1, 32, 32, 128), torch.float32)],
        {
            "model_names": ["pt_llama3_huggyllama_llama_7b_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "64", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index16,
        [((1, 32, 32, 128), torch.float32)],
        {
            "model_names": ["pt_llama3_huggyllama_llama_7b_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index15,
        [((1, 32, 4, 128), torch.float32)],
        {
            "model_names": [
                "pt_llama3_huggyllama_llama_7b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "64", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index16,
        [((1, 32, 4, 128), torch.float32)],
        {
            "model_names": [
                "pt_llama3_huggyllama_llama_7b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index694,
        [((1, 8192, 6), torch.float32)],
        {
            "model_names": ["pt_mamba_state_spaces_mamba_1_4b_hf_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "4096", "stride": "1"},
        },
    ),
    (
        Index695,
        [((1, 8192, 6), torch.float32)],
        {
            "model_names": ["pt_mamba_state_spaces_mamba_1_4b_hf_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "4096", "stop": "8192", "stride": "1"},
        },
    ),
    (
        Index40,
        [((1, 4096, 9), torch.float32)],
        {
            "model_names": ["pt_mamba_state_spaces_mamba_1_4b_hf_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "6", "stride": "1"},
        },
    ),
    (
        Index696,
        [((160, 4096), torch.float32)],
        {
            "model_names": ["pt_mamba_state_spaces_mamba_1_4b_hf_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index697,
        [((160, 4096), torch.float32)],
        {
            "model_names": ["pt_mamba_state_spaces_mamba_1_4b_hf_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "128", "stop": "144", "stride": "1"},
        },
    ),
    (
        Index698,
        [((160, 4096), torch.float32)],
        {
            "model_names": ["pt_mamba_state_spaces_mamba_1_4b_hf_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "144", "stop": "160", "stride": "1"},
        },
    ),
    (
        Index6,
        [((1, 4096, 6, 16), torch.float32)],
        {
            "model_names": ["pt_mamba_state_spaces_mamba_1_4b_hf_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index7,
        [((1, 4096, 6, 16), torch.float32)],
        {
            "model_names": ["pt_mamba_state_spaces_mamba_1_4b_hf_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index44,
        [((1, 4096, 6, 16), torch.float32)],
        {
            "model_names": ["pt_mamba_state_spaces_mamba_1_4b_hf_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "2", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index45,
        [((1, 4096, 6, 16), torch.float32)],
        {
            "model_names": ["pt_mamba_state_spaces_mamba_1_4b_hf_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "3", "stop": "4", "stride": "1"},
        },
    ),
    (
        Index46,
        [((1, 4096, 6, 16), torch.float32)],
        {
            "model_names": ["pt_mamba_state_spaces_mamba_1_4b_hf_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "4", "stop": "5", "stride": "1"},
        },
    ),
    (
        Index47,
        [((1, 4096, 6, 16), torch.float32)],
        {
            "model_names": ["pt_mamba_state_spaces_mamba_1_4b_hf_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "5", "stop": "6", "stride": "1"},
        },
    ),
    (
        Index699,
        [((1, 10240, 6), torch.float32)],
        {
            "model_names": ["pt_mamba_state_spaces_mamba_2_8b_hf_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "5120", "stride": "1"},
        },
    ),
    (
        Index700,
        [((1, 10240, 6), torch.float32)],
        {
            "model_names": ["pt_mamba_state_spaces_mamba_2_8b_hf_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "5120", "stop": "10240", "stride": "1"},
        },
    ),
    (
        Index40,
        [((1, 5120, 9), torch.float32)],
        {
            "model_names": ["pt_mamba_state_spaces_mamba_2_8b_hf_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "6", "stride": "1"},
        },
    ),
    (
        Index701,
        [((192, 5120), torch.float32)],
        {
            "model_names": ["pt_mamba_state_spaces_mamba_2_8b_hf_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "160", "stride": "1"},
        },
    ),
    (
        Index702,
        [((192, 5120), torch.float32)],
        {
            "model_names": ["pt_mamba_state_spaces_mamba_2_8b_hf_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "160", "stop": "176", "stride": "1"},
        },
    ),
    (
        Index703,
        [((192, 5120), torch.float32)],
        {
            "model_names": ["pt_mamba_state_spaces_mamba_2_8b_hf_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "176", "stop": "192", "stride": "1"},
        },
    ),
    (
        Index6,
        [((1, 5120, 6, 16), torch.float32)],
        {
            "model_names": ["pt_mamba_state_spaces_mamba_2_8b_hf_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index7,
        [((1, 5120, 6, 16), torch.float32)],
        {
            "model_names": ["pt_mamba_state_spaces_mamba_2_8b_hf_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "1", "stop": "2", "stride": "1"},
        },
    ),
    (
        Index44,
        [((1, 5120, 6, 16), torch.float32)],
        {
            "model_names": ["pt_mamba_state_spaces_mamba_2_8b_hf_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "2", "stop": "3", "stride": "1"},
        },
    ),
    (
        Index45,
        [((1, 5120, 6, 16), torch.float32)],
        {
            "model_names": ["pt_mamba_state_spaces_mamba_2_8b_hf_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "3", "stop": "4", "stride": "1"},
        },
    ),
    (
        Index46,
        [((1, 5120, 6, 16), torch.float32)],
        {
            "model_names": ["pt_mamba_state_spaces_mamba_2_8b_hf_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "4", "stop": "5", "stride": "1"},
        },
    ),
    (
        Index47,
        [((1, 5120, 6, 16), torch.float32)],
        {
            "model_names": ["pt_mamba_state_spaces_mamba_2_8b_hf_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "5", "stop": "6", "stride": "1"},
        },
    ),
    (
        Index10,
        [((1, 32, 11, 80), torch.float32)],
        {
            "model_names": ["pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf", "pt_phi2_microsoft_phi_2_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index53,
        [((1, 32, 11, 80), torch.float32)],
        {
            "model_names": ["pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf", "pt_phi2_microsoft_phi_2_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "32", "stop": "80", "stride": "1"},
        },
    ),
    (
        Index12,
        [((1, 32, 11, 32), torch.float32)],
        {
            "model_names": ["pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf", "pt_phi2_microsoft_phi_2_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "16", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index5,
        [((1, 32, 11, 32), torch.float32)],
        {
            "model_names": ["pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf", "pt_phi2_microsoft_phi_2_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "16", "stride": "1"},
        },
    ),
    (
        Index377,
        [((1, 11, 2), torch.float32)],
        {
            "model_names": ["pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf", "pt_phi2_microsoft_phi_2_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "10", "stop": "11", "stride": "1"},
        },
    ),
    (
        Index10,
        [((1, 32, 12, 80), torch.float32)],
        {
            "model_names": ["pt_phi2_microsoft_phi_2_pytdml_token_cls_hf", "pt_phi2_microsoft_phi_2_token_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index53,
        [((1, 32, 12, 80), torch.float32)],
        {
            "model_names": ["pt_phi2_microsoft_phi_2_pytdml_token_cls_hf", "pt_phi2_microsoft_phi_2_token_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "32", "stop": "80", "stride": "1"},
        },
    ),
    (
        Index704,
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
        Index705,
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
        Index706,
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
        Index515,
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
        Index516,
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
        Index707,
        [((16384, 3072), torch.float32)],
        {
            "model_names": [
                "pt_phi3_microsoft_phi_3_mini_128k_instruct_seq_cls_hf",
                "pt_phi3_microsoft_phi_3_mini_128k_instruct_token_cls_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf",
                "pt_phi3_microsoft_phi_3_mini_128k_instruct_clm_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf",
                "pt_phi3_5_microsoft_phi_3_5_mini_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "8192", "stop": "16384", "stride": "1"},
        },
    ),
    (
        Index708,
        [((16384, 3072), torch.float32)],
        {
            "model_names": [
                "pt_phi3_microsoft_phi_3_mini_128k_instruct_seq_cls_hf",
                "pt_phi3_microsoft_phi_3_mini_128k_instruct_token_cls_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf",
                "pt_phi3_microsoft_phi_3_mini_128k_instruct_clm_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf",
                "pt_phi3_5_microsoft_phi_3_5_mini_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "8192", "stride": "1"},
        },
    ),
    (
        Index46,
        [((1, 5, 2), torch.float32)],
        {
            "model_names": [
                "pt_phi3_microsoft_phi_3_mini_128k_instruct_seq_cls_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "4", "stop": "5", "stride": "1"},
        },
    ),
    (
        Index704,
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
        Index705,
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
        Index706,
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
        Index515,
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
        Index516,
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
        Index709,
        [((1, 256, 7680), torch.float32)],
        {
            "model_names": ["pt_phi4_microsoft_phi_4_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "5120", "stride": "1"},
        },
    ),
    (
        Index710,
        [((1, 256, 7680), torch.float32)],
        {
            "model_names": ["pt_phi4_microsoft_phi_4_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "5120", "stop": "6400", "stride": "1"},
        },
    ),
    (
        Index711,
        [((1, 256, 7680), torch.float32)],
        {
            "model_names": ["pt_phi4_microsoft_phi_4_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "6400", "stop": "7680", "stride": "1"},
        },
    ),
    (
        Index15,
        [((1, 40, 256, 128), torch.float32)],
        {
            "model_names": ["pt_phi4_microsoft_phi_4_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "64", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index16,
        [((1, 40, 256, 128), torch.float32)],
        {
            "model_names": ["pt_phi4_microsoft_phi_4_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index15,
        [((1, 10, 256, 128), torch.float32)],
        {
            "model_names": ["pt_phi4_microsoft_phi_4_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "64", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index16,
        [((1, 10, 256, 128), torch.float32)],
        {
            "model_names": ["pt_phi4_microsoft_phi_4_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index712,
        [((35840, 5120), torch.float32)],
        {
            "model_names": [
                "pt_phi4_microsoft_phi_4_seq_cls_hf",
                "pt_phi4_microsoft_phi_4_token_cls_hf",
                "pt_phi4_microsoft_phi_4_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "17920", "stop": "35840", "stride": "1"},
        },
    ),
    (
        Index713,
        [((35840, 5120), torch.float32)],
        {
            "model_names": [
                "pt_phi4_microsoft_phi_4_seq_cls_hf",
                "pt_phi4_microsoft_phi_4_token_cls_hf",
                "pt_phi4_microsoft_phi_4_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "17920", "stride": "1"},
        },
    ),
    (
        Index709,
        [((1, 12, 7680), torch.float32)],
        {
            "model_names": ["pt_phi4_microsoft_phi_4_token_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "5120", "stride": "1"},
        },
    ),
    (
        Index710,
        [((1, 12, 7680), torch.float32)],
        {
            "model_names": ["pt_phi4_microsoft_phi_4_token_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "5120", "stop": "6400", "stride": "1"},
        },
    ),
    (
        Index711,
        [((1, 12, 7680), torch.float32)],
        {
            "model_names": ["pt_phi4_microsoft_phi_4_token_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "6400", "stop": "7680", "stride": "1"},
        },
    ),
    (
        Index15,
        [((1, 40, 12, 128), torch.float32)],
        {
            "model_names": ["pt_phi4_microsoft_phi_4_token_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "64", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index16,
        [((1, 40, 12, 128), torch.float32)],
        {
            "model_names": ["pt_phi4_microsoft_phi_4_token_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index15,
        [((1, 10, 12, 128), torch.float32)],
        {
            "model_names": ["pt_phi4_microsoft_phi_4_token_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "64", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index16,
        [((1, 10, 12, 128), torch.float32)],
        {
            "model_names": ["pt_phi4_microsoft_phi_4_token_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index15,
        [((1, 16, 35, 128), torch.float32)],
        {
            "model_names": [
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "64", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index16,
        [((1, 16, 35, 128), torch.float32)],
        {
            "model_names": [
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index15,
        [((1, 28, 35, 128), torch.float32)],
        {
            "model_names": [
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "64", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index16,
        [((1, 28, 35, 128), torch.float32)],
        {
            "model_names": [
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index15,
        [((1, 4, 35, 128), torch.float32)],
        {
            "model_names": [
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "64", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index16,
        [((1, 4, 35, 128), torch.float32)],
        {
            "model_names": [
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index15,
        [((1, 28, 13, 128), torch.float32)],
        {
            "model_names": ["pt_qwen_v2_qwen_qwen2_7b_token_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "64", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index16,
        [((1, 28, 13, 128), torch.float32)],
        {
            "model_names": ["pt_qwen_v2_qwen_qwen2_7b_token_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index15,
        [((1, 4, 13, 128), torch.float32)],
        {
            "model_names": ["pt_qwen_v2_qwen_qwen2_7b_token_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "64", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index16,
        [((1, 4, 13, 128), torch.float32)],
        {
            "model_names": ["pt_qwen_v2_qwen_qwen2_7b_token_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index6,
        [((2048, 2048), torch.float32)],
        {
            "model_names": ["pt_stereo_facebook_musicgen_large_music_generation_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "1", "stride": "1"},
        },
    ),
    (
        Index704,
        [((1, 256, 9216), torch.float32)],
        {
            "model_names": [
                "onnx_phi3_microsoft_phi_3_mini_128k_instruct_clm_hf",
                "onnx_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf",
                "onnx_phi3_5_microsoft_phi_3_5_mini_instruct_clm_hf",
                "pt_phi3_microsoft_phi_3_mini_128k_instruct_clm_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf",
                "pt_phi3_5_microsoft_phi_3_5_mini_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "3072", "stride": "1"},
        },
    ),
    (
        Index705,
        [((1, 256, 9216), torch.float32)],
        {
            "model_names": [
                "onnx_phi3_microsoft_phi_3_mini_128k_instruct_clm_hf",
                "onnx_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf",
                "onnx_phi3_5_microsoft_phi_3_5_mini_instruct_clm_hf",
                "pt_phi3_microsoft_phi_3_mini_128k_instruct_clm_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf",
                "pt_phi3_5_microsoft_phi_3_5_mini_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "3072", "stop": "6144", "stride": "1"},
        },
    ),
    (
        Index706,
        [((1, 256, 9216), torch.float32)],
        {
            "model_names": [
                "onnx_phi3_microsoft_phi_3_mini_128k_instruct_clm_hf",
                "onnx_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf",
                "onnx_phi3_5_microsoft_phi_3_5_mini_instruct_clm_hf",
                "pt_phi3_microsoft_phi_3_mini_128k_instruct_clm_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf",
                "pt_phi3_5_microsoft_phi_3_5_mini_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "6144", "stop": "9216", "stride": "1"},
        },
    ),
    (
        Index515,
        [((1, 32, 256, 96), torch.float32)],
        {
            "model_names": [
                "onnx_phi3_microsoft_phi_3_mini_128k_instruct_clm_hf",
                "onnx_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf",
                "onnx_phi3_5_microsoft_phi_3_5_mini_instruct_clm_hf",
                "pt_phi3_microsoft_phi_3_mini_128k_instruct_clm_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf",
                "pt_phi3_5_microsoft_phi_3_5_mini_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "48", "stop": "96", "stride": "1"},
        },
    ),
    (
        Index516,
        [((1, 32, 256, 96), torch.float32)],
        {
            "model_names": [
                "onnx_phi3_microsoft_phi_3_mini_128k_instruct_clm_hf",
                "onnx_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf",
                "onnx_phi3_5_microsoft_phi_3_5_mini_instruct_clm_hf",
                "pt_phi3_microsoft_phi_3_mini_128k_instruct_clm_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf",
                "pt_phi3_5_microsoft_phi_3_5_mini_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "48", "stride": "1"},
        },
    ),
    (
        Index714,
        [((1, 256, 16384), torch.float32)],
        {
            "model_names": [
                "onnx_phi3_microsoft_phi_3_mini_128k_instruct_clm_hf",
                "onnx_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf",
                "onnx_phi3_5_microsoft_phi_3_5_mini_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "8192", "stop": "16384", "stride": "1"},
        },
    ),
    (
        Index715,
        [((1, 256, 16384), torch.float32)],
        {
            "model_names": [
                "onnx_phi3_microsoft_phi_3_mini_128k_instruct_clm_hf",
                "onnx_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf",
                "onnx_phi3_5_microsoft_phi_3_5_mini_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "8192", "stride": "1"},
        },
    ),
    (
        Index674,
        [((1, 8, 107, 256), torch.float32)],
        {
            "model_names": ["pt_gemma_google_gemma_1_1_2b_it_qa_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "128", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index0,
        [((1, 8, 107, 256), torch.float32)],
        {
            "model_names": ["pt_gemma_google_gemma_1_1_2b_it_qa_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index674,
        [((1, 1, 107, 256), torch.float32)],
        {
            "model_names": ["pt_gemma_google_gemma_1_1_2b_it_qa_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "128", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index0,
        [((1, 1, 107, 256), torch.float32)],
        {
            "model_names": ["pt_gemma_google_gemma_1_1_2b_it_qa_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index674,
        [((1, 16, 107, 256), torch.float32)],
        {
            "model_names": ["pt_gemma_google_gemma_1_1_7b_it_qa_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "128", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index0,
        [((1, 16, 107, 256), torch.float32)],
        {
            "model_names": ["pt_gemma_google_gemma_1_1_7b_it_qa_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index15,
        [((1, 32, 256, 128), torch.float32)],
        {
            "model_names": ["pt_llama3_meta_llama_llama_3_1_8b_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "64", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index16,
        [((1, 32, 256, 128), torch.float32)],
        {
            "model_names": ["pt_llama3_meta_llama_llama_3_1_8b_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index15,
        [((1, 8, 256, 128), torch.float32)],
        {
            "model_names": [
                "pt_llama3_meta_llama_llama_3_1_8b_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_3b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "64", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index16,
        [((1, 8, 256, 128), torch.float32)],
        {
            "model_names": [
                "pt_llama3_meta_llama_llama_3_1_8b_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_3b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index15,
        [((1, 8, 4, 128), torch.float32)],
        {
            "model_names": [
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_3b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_3b_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "64", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index16,
        [((1, 8, 4, 128), torch.float32)],
        {
            "model_names": [
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_3b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_3b_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index15,
        [((1, 24, 4, 128), torch.float32)],
        {
            "model_names": [
                "pt_llama3_meta_llama_llama_3_2_3b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_3b_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "64", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index16,
        [((1, 24, 4, 128), torch.float32)],
        {
            "model_names": [
                "pt_llama3_meta_llama_llama_3_2_3b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_3b_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index15,
        [((1, 32, 10, 128), torch.float32)],
        {
            "model_names": ["pt_ministral_ministral_ministral_3b_instruct_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "64", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index16,
        [((1, 32, 10, 128), torch.float32)],
        {
            "model_names": ["pt_ministral_ministral_ministral_3b_instruct_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index15,
        [((1, 8, 10, 128), torch.float32)],
        {
            "model_names": ["pt_ministral_ministral_ministral_3b_instruct_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "64", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index16,
        [((1, 8, 10, 128), torch.float32)],
        {
            "model_names": ["pt_ministral_ministral_ministral_3b_instruct_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index15,
        [((1, 32, 8, 128), torch.float32)],
        {
            "model_names": ["pt_ministral_mistralai_ministral_8b_instruct_2410_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "64", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index16,
        [((1, 32, 8, 128), torch.float32)],
        {
            "model_names": ["pt_ministral_mistralai_ministral_8b_instruct_2410_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index15,
        [((1, 8, 8, 128), torch.float32)],
        {
            "model_names": ["pt_ministral_mistralai_ministral_8b_instruct_2410_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "64", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index16,
        [((1, 8, 8, 128), torch.float32)],
        {
            "model_names": ["pt_ministral_mistralai_ministral_8b_instruct_2410_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index15,
        [((1, 32, 135, 128), torch.float32)],
        {
            "model_names": ["pt_mistral_mistralai_mistral_7b_instruct_v0_3_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "64", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index16,
        [((1, 32, 135, 128), torch.float32)],
        {
            "model_names": ["pt_mistral_mistralai_mistral_7b_instruct_v0_3_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index15,
        [((1, 8, 135, 128), torch.float32)],
        {
            "model_names": ["pt_mistral_mistralai_mistral_7b_instruct_v0_3_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "64", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index16,
        [((1, 8, 135, 128), torch.float32)],
        {
            "model_names": ["pt_mistral_mistralai_mistral_7b_instruct_v0_3_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index15,
        [((1, 32, 128, 128), torch.float32)],
        {
            "model_names": ["pt_mistral_mistralai_mistral_7b_v0_1_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "64", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index16,
        [((1, 32, 128, 128), torch.float32)],
        {
            "model_names": ["pt_mistral_mistralai_mistral_7b_v0_1_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index15,
        [((1, 8, 128, 128), torch.float32)],
        {
            "model_names": ["pt_mistral_mistralai_mistral_7b_v0_1_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "64", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index16,
        [((1, 8, 128, 128), torch.float32)],
        {
            "model_names": ["pt_mistral_mistralai_mistral_7b_v0_1_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index709,
        [((1, 6, 7680), torch.float32)],
        {
            "model_names": ["pt_phi4_microsoft_phi_4_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "5120", "stride": "1"},
        },
    ),
    (
        Index710,
        [((1, 6, 7680), torch.float32)],
        {
            "model_names": ["pt_phi4_microsoft_phi_4_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "5120", "stop": "6400", "stride": "1"},
        },
    ),
    (
        Index711,
        [((1, 6, 7680), torch.float32)],
        {
            "model_names": ["pt_phi4_microsoft_phi_4_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "6400", "stop": "7680", "stride": "1"},
        },
    ),
    (
        Index15,
        [((1, 40, 6, 128), torch.float32)],
        {
            "model_names": ["pt_phi4_microsoft_phi_4_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "64", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index16,
        [((1, 40, 6, 128), torch.float32)],
        {
            "model_names": ["pt_phi4_microsoft_phi_4_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index15,
        [((1, 10, 6, 128), torch.float32)],
        {
            "model_names": ["pt_phi4_microsoft_phi_4_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "64", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index16,
        [((1, 10, 6, 128), torch.float32)],
        {
            "model_names": ["pt_phi4_microsoft_phi_4_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index65,
        [((1, 2, 128, 400), torch.float32)],
        {
            "model_names": ["onnx_yolov10_default_obj_det_github"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "64", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index66,
        [((1, 2, 128, 400), torch.float32)],
        {
            "model_names": ["onnx_yolov10_default_obj_det_github"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "0", "stop": "32", "stride": "1"},
        },
    ),
    (
        Index67,
        [((1, 2, 128, 400), torch.float32)],
        {
            "model_names": ["onnx_yolov10_default_obj_det_github"],
            "pcc": 0.99,
            "args": {"dim": "-2", "start": "32", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index674,
        [((1, 12, 522, 256), torch.float32)],
        {
            "model_names": ["pt_falcon3_tiiuae_falcon3_3b_base_clm_hf", "pt_falcon3_tiiuae_falcon3_7b_base_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "128", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index0,
        [((1, 12, 522, 256), torch.float32)],
        {
            "model_names": ["pt_falcon3_tiiuae_falcon3_3b_base_clm_hf", "pt_falcon3_tiiuae_falcon3_7b_base_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index15,
        [((1, 24, 256, 128), torch.float32)],
        {
            "model_names": ["pt_llama3_meta_llama_llama_3_2_3b_instruct_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "64", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index16,
        [((1, 24, 256, 128), torch.float32)],
        {
            "model_names": ["pt_llama3_meta_llama_llama_3_2_3b_instruct_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index35,
        [((1, 1792, 56, 56), torch.bfloat16)],
        {
            "model_names": ["pt_monodle_base_obj_det_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "0", "stop": "256", "stride": "1"},
        },
    ),
    (
        Index662,
        [((1, 1792, 56, 56), torch.bfloat16)],
        {
            "model_names": ["pt_monodle_base_obj_det_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "256", "stop": "512", "stride": "1"},
        },
    ),
    (
        Index716,
        [((1, 1792, 56, 56), torch.bfloat16)],
        {
            "model_names": ["pt_monodle_base_obj_det_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "512", "stop": "768", "stride": "1"},
        },
    ),
    (
        Index717,
        [((1, 1792, 56, 56), torch.bfloat16)],
        {
            "model_names": ["pt_monodle_base_obj_det_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "768", "stop": "1024", "stride": "1"},
        },
    ),
    (
        Index718,
        [((1, 1792, 56, 56), torch.bfloat16)],
        {
            "model_names": ["pt_monodle_base_obj_det_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "1024", "stop": "1280", "stride": "1"},
        },
    ),
    (
        Index719,
        [((1, 1792, 56, 56), torch.bfloat16)],
        {
            "model_names": ["pt_monodle_base_obj_det_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "1280", "stop": "1536", "stride": "1"},
        },
    ),
    (
        Index720,
        [((1, 1792, 56, 56), torch.bfloat16)],
        {
            "model_names": ["pt_monodle_base_obj_det_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-3", "start": "1536", "stop": "1792", "stride": "1"},
        },
    ),
    (
        Index721,
        [((1024, 8), torch.float32)],
        {
            "model_names": ["pt_nbeats_trend_basis_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "4", "stride": "1"},
        },
    ),
    (
        Index722,
        [((1024, 8), torch.float32)],
        {
            "model_names": ["pt_nbeats_trend_basis_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "4", "stop": "8", "stride": "1"},
        },
    ),
    (
        Index15,
        [((1, 16, 29, 128), torch.float32)],
        {
            "model_names": ["pt_qwen_v2_qwen_qwen2_5_3b_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "64", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index16,
        [((1, 16, 29, 128), torch.float32)],
        {
            "model_names": ["pt_qwen_v2_qwen_qwen2_5_3b_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index15,
        [((1, 16, 39, 128), torch.float32)],
        {
            "model_names": ["pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "64", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index16,
        [((1, 16, 39, 128), torch.float32)],
        {
            "model_names": ["pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index15,
        [((1, 28, 29, 128), torch.float32)],
        {
            "model_names": ["pt_qwen_v2_qwen_qwen2_5_7b_clm_hf", "pt_qwen_v2_qwen_qwen2_5_7b_instruct_1m_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "64", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index16,
        [((1, 28, 29, 128), torch.float32)],
        {
            "model_names": ["pt_qwen_v2_qwen_qwen2_5_7b_clm_hf", "pt_qwen_v2_qwen_qwen2_5_7b_instruct_1m_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index15,
        [((1, 4, 29, 128), torch.float32)],
        {
            "model_names": ["pt_qwen_v2_qwen_qwen2_5_7b_clm_hf", "pt_qwen_v2_qwen_qwen2_5_7b_instruct_1m_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "64", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index16,
        [((1, 4, 29, 128), torch.float32)],
        {
            "model_names": ["pt_qwen_v2_qwen_qwen2_5_7b_clm_hf", "pt_qwen_v2_qwen_qwen2_5_7b_instruct_1m_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index15,
        [((1, 28, 39, 128), torch.float32)],
        {
            "model_names": ["pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "64", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index16,
        [((1, 28, 39, 128), torch.float32)],
        {
            "model_names": ["pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index15,
        [((1, 4, 39, 128), torch.float32)],
        {
            "model_names": ["pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "64", "stop": "128", "stride": "1"},
        },
    ),
    (
        Index16,
        [((1, 4, 39, 128), torch.float32)],
        {
            "model_names": ["pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "start": "0", "stop": "64", "stride": "1"},
        },
    ),
    (
        Index6,
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

    compiler_cfg = forge.config.CompilerConfig()
    if "default_df_override" in metadata.keys():
        compiler_cfg.default_df_override = forge.DataFormat.from_json(metadata["default_df_override"])

    compiled_model = compile(framework_model, sample_inputs=inputs, compiler_cfg=compiler_cfg)

    verify(inputs, framework_model, compiled_model, VerifyConfig(value_checker=AutomaticValueChecker(pcc=pcc)))
