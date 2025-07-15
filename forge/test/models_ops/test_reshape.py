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


class Reshape0(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(11, 312))
        return reshape_output_1


class Reshape1(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 11, 12, 26))
        return reshape_output_1


class Reshape2(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 11, 312))
        return reshape_output_1


class Reshape3(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 11, 26))
        return reshape_output_1


class Reshape4(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 26, 11))
        return reshape_output_1


class Reshape5(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 11, 11))
        return reshape_output_1


class Reshape6(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 11, 11))
        return reshape_output_1


class Reshape7(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 11, 26))
        return reshape_output_1


class Reshape8(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2048, 1, 1))
        return reshape_output_1


class Reshape9(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2048))
        return reshape_output_1


class Reshape10(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(128, 1024))
        return reshape_output_1


class Reshape11(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 16, 64))
        return reshape_output_1


class Reshape12(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 1024))
        return reshape_output_1


class Reshape13(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 128, 64))
        return reshape_output_1


class Reshape14(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 128, 128))
        return reshape_output_1


class Reshape15(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 128, 128))
        return reshape_output_1


class Reshape16(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 128, 64))
        return reshape_output_1


class Reshape17(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1000))
        return reshape_output_1


class Reshape18(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(128, 768))
        return reshape_output_1


class Reshape19(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 12, 64))
        return reshape_output_1


class Reshape20(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 768))
        return reshape_output_1


class Reshape21(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 128, 64))
        return reshape_output_1


class Reshape22(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 128, 128))
        return reshape_output_1


class Reshape23(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 128, 128))
        return reshape_output_1


class Reshape24(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 128, 64))
        return reshape_output_1


class Reshape25(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 1))
        return reshape_output_1


class Reshape26(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128))
        return reshape_output_1


class Reshape27(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 768))
        return reshape_output_1


class Reshape28(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1,))
        return reshape_output_1


class Reshape29(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(40, 1, 3, 3))
        return reshape_output_1


class Reshape30(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(24, 1, 3, 3))
        return reshape_output_1


class Reshape31(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(144, 1, 3, 3))
        return reshape_output_1


class Reshape32(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(192, 1, 3, 3))
        return reshape_output_1


class Reshape33(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(192, 1, 5, 5))
        return reshape_output_1


class Reshape34(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(288, 1, 5, 5))
        return reshape_output_1


class Reshape35(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(288, 1, 3, 3))
        return reshape_output_1


class Reshape36(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(576, 1, 3, 3))
        return reshape_output_1


class Reshape37(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(576, 1, 5, 5))
        return reshape_output_1


class Reshape38(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(816, 1, 5, 5))
        return reshape_output_1


class Reshape39(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1392, 1, 5, 5))
        return reshape_output_1


class Reshape40(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1392, 1, 3, 3))
        return reshape_output_1


class Reshape41(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2304, 1, 3, 3))
        return reshape_output_1


class Reshape42(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1536, 1, 1))
        return reshape_output_1


class Reshape43(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(48, 1, 3, 3))
        return reshape_output_1


class Reshape44(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(336, 1, 5, 5))
        return reshape_output_1


class Reshape45(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(336, 1, 3, 3))
        return reshape_output_1


class Reshape46(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(672, 1, 3, 3))
        return reshape_output_1


class Reshape47(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(672, 1, 5, 5))
        return reshape_output_1


class Reshape48(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(960, 1, 5, 5))
        return reshape_output_1


class Reshape49(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1632, 1, 5, 5))
        return reshape_output_1


class Reshape50(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1632, 1, 3, 3))
        return reshape_output_1


class Reshape51(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2688, 1, 3, 3))
        return reshape_output_1


class Reshape52(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1792, 1, 1))
        return reshape_output_1


class Reshape53(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(56, 1, 3, 3))
        return reshape_output_1


class Reshape54(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 1, 3, 3))
        return reshape_output_1


class Reshape55(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(240, 1, 3, 3))
        return reshape_output_1


class Reshape56(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(240, 1, 5, 5))
        return reshape_output_1


class Reshape57(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(432, 1, 5, 5))
        return reshape_output_1


class Reshape58(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(432, 1, 3, 3))
        return reshape_output_1


class Reshape59(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(864, 1, 3, 3))
        return reshape_output_1


class Reshape60(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(864, 1, 5, 5))
        return reshape_output_1


class Reshape61(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1200, 1, 5, 5))
        return reshape_output_1


class Reshape62(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2064, 1, 5, 5))
        return reshape_output_1


class Reshape63(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2064, 1, 3, 3))
        return reshape_output_1


class Reshape64(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(3456, 1, 3, 3))
        return reshape_output_1


class Reshape65(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2304, 1, 1))
        return reshape_output_1


class Reshape66(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(4, 2048))
        return reshape_output_1


class Reshape67(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 32, 64))
        return reshape_output_1


class Reshape68(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 2048))
        return reshape_output_1


class Reshape69(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 4, 64))
        return reshape_output_1


class Reshape70(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 8, 64))
        return reshape_output_1


class Reshape71(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 4, 4))
        return reshape_output_1


class Reshape72(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 4, 4))
        return reshape_output_1


class Reshape73(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 4, 64))
        return reshape_output_1


class Reshape74(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 8192))
        return reshape_output_1


class Reshape75(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2))
        return reshape_output_1


class Reshape76(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 3, 16, 16, 16, 16))
        return reshape_output_1


class Reshape77(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(256, 768))
        return reshape_output_1


class Reshape78(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 512))
        return reshape_output_1


class Reshape79(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 512, 1))
        return reshape_output_1


class Reshape80(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(256, 512))
        return reshape_output_1


class Reshape81(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 8, 64))
        return reshape_output_1


class Reshape82(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 16, 512))
        return reshape_output_1


class Reshape83(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1024, 256, 1, 1))
        return reshape_output_1


class Reshape84(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1024, 512))
        return reshape_output_1


class Reshape85(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1024, 512, 1))
        return reshape_output_1


class Reshape86(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(256, 1024, 1, 1))
        return reshape_output_1


class Reshape87(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 512, 196, 1))
        return reshape_output_1


class Reshape88(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 1, 3, 3))
        return reshape_output_1


class Reshape89(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(128, 1, 3, 3))
        return reshape_output_1


class Reshape90(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(256, 1, 3, 3))
        return reshape_output_1


class Reshape91(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(512, 1, 3, 3))
        return reshape_output_1


class Reshape92(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1024, 1, 3, 3))
        return reshape_output_1


class Reshape93(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1024, 1, 1))
        return reshape_output_1


class Reshape94(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1024))
        return reshape_output_1


class Reshape95(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 512, 1, 1))
        return reshape_output_1


class Reshape96(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 512))
        return reshape_output_1


class Reshape97(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(256, 2048))
        return reshape_output_1


class Reshape98(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 32, 64))
        return reshape_output_1


class Reshape99(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 2048))
        return reshape_output_1


class Reshape100(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 256, 64))
        return reshape_output_1


class Reshape101(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 256, 256))
        return reshape_output_1


class Reshape102(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 256, 256))
        return reshape_output_1


class Reshape103(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 256, 64))
        return reshape_output_1


class Reshape104(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 8192))
        return reshape_output_1


class Reshape105(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(5, 2048))
        return reshape_output_1


class Reshape106(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 5, 32, 64))
        return reshape_output_1


class Reshape107(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 5, 2048))
        return reshape_output_1


class Reshape108(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 5, 64))
        return reshape_output_1


class Reshape109(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 5, 5))
        return reshape_output_1


class Reshape110(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 5, 5))
        return reshape_output_1


class Reshape111(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 5, 64))
        return reshape_output_1


class Reshape112(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 5, 8192))
        return reshape_output_1


class Reshape113(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 7392, 1, 1))
        return reshape_output_1


class Reshape114(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2016, 1, 1))
        return reshape_output_1


class Reshape115(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 16384, 1))
        return reshape_output_1


class Reshape116(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16384, 1, 32))
        return reshape_output_1


class Reshape117(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 128, 32))
        return reshape_output_1


class Reshape118(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 128, 128))
        return reshape_output_1


class Reshape119(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 256))
        return reshape_output_1


class Reshape120(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(256, 32))
        return reshape_output_1


class Reshape121(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 1, 32))
        return reshape_output_1


class Reshape122(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 32))
        return reshape_output_1


class Reshape123(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 16384, 256))
        return reshape_output_1


class Reshape124(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16384, 256))
        return reshape_output_1


class Reshape125(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 128, 128))
        return reshape_output_1


class Reshape126(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 16384, 1))
        return reshape_output_1


class Reshape127(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 64, 4096, 1))
        return reshape_output_1


class Reshape128(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4096, 2, 32))
        return reshape_output_1


class Reshape129(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 64, 64, 64))
        return reshape_output_1


class Reshape130(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 4096, 32))
        return reshape_output_1


class Reshape131(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 64, 256))
        return reshape_output_1


class Reshape132(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(256, 64))
        return reshape_output_1


class Reshape133(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 2, 32))
        return reshape_output_1


class Reshape134(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 1, 64))
        return reshape_output_1


class Reshape135(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 64))
        return reshape_output_1


class Reshape136(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 256, 32))
        return reshape_output_1


class Reshape137(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2, 4096, 256))
        return reshape_output_1


class Reshape138(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 4096, 256))
        return reshape_output_1


class Reshape139(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 32, 256))
        return reshape_output_1


class Reshape140(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2, 4096, 32))
        return reshape_output_1


class Reshape141(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(4096, 64))
        return reshape_output_1


class Reshape142(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4096, 64))
        return reshape_output_1


class Reshape143(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 64, 64))
        return reshape_output_1


class Reshape144(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 4096, 1))
        return reshape_output_1


class Reshape145(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 160, 1024, 1))
        return reshape_output_1


class Reshape146(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1024, 5, 32))
        return reshape_output_1


class Reshape147(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 32, 160))
        return reshape_output_1


class Reshape148(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(5, 1024, 32))
        return reshape_output_1


class Reshape149(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 160, 32, 32))
        return reshape_output_1


class Reshape150(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 160, 256))
        return reshape_output_1


class Reshape151(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(256, 160))
        return reshape_output_1


class Reshape152(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 5, 32))
        return reshape_output_1


class Reshape153(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 160))
        return reshape_output_1


class Reshape154(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(5, 256, 32))
        return reshape_output_1


class Reshape155(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 5, 1024, 256))
        return reshape_output_1


class Reshape156(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(5, 1024, 256))
        return reshape_output_1


class Reshape157(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(5, 32, 256))
        return reshape_output_1


class Reshape158(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 5, 1024, 32))
        return reshape_output_1


class Reshape159(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1024, 160))
        return reshape_output_1


class Reshape160(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1024, 160))
        return reshape_output_1


class Reshape161(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 640, 32, 32))
        return reshape_output_1


class Reshape162(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(640, 1, 3, 3))
        return reshape_output_1


class Reshape163(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 640, 1024, 1))
        return reshape_output_1


class Reshape164(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 256, 1))
        return reshape_output_1


class Reshape165(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(256, 256))
        return reshape_output_1


class Reshape166(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 8, 32))
        return reshape_output_1


class Reshape167(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 256))
        return reshape_output_1


class Reshape168(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 256, 32))
        return reshape_output_1


class Reshape169(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 256, 256))
        return reshape_output_1


class Reshape170(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 256, 256))
        return reshape_output_1


class Reshape171(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 32, 256))
        return reshape_output_1


class Reshape172(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 256, 32))
        return reshape_output_1


class Reshape173(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1024, 16, 16))
        return reshape_output_1


class Reshape174(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1024, 256, 1))
        return reshape_output_1


class Reshape175(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 7, 8, 7, 96))
        return reshape_output_1


class Reshape176(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(3136, 96))
        return reshape_output_1


class Reshape177(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 49, 288))
        return reshape_output_1


class Reshape178(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 49, 3, 3, 32))
        return reshape_output_1


class Reshape179(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 3, 49, 32))
        return reshape_output_1


class Reshape180(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(192, 49, 32))
        return reshape_output_1


class Reshape181(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 3, 49, 49))
        return reshape_output_1


class Reshape182(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(49, 49, 3))
        return reshape_output_1


class Reshape183(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(192, 49, 49))
        return reshape_output_1


class Reshape184(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 64, 3, 49, 49))
        return reshape_output_1


class Reshape185(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(192, 32, 49))
        return reshape_output_1


class Reshape186(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 49, 96))
        return reshape_output_1


class Reshape187(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 56, 56, 96))
        return reshape_output_1


class Reshape188(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 8, 7, 7, 96))
        return reshape_output_1


class Reshape189(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 56, 56, 384))
        return reshape_output_1


class Reshape190(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(3136, 384))
        return reshape_output_1


class Reshape191(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(784, 384))
        return reshape_output_1


class Reshape192(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 28, 28, 192))
        return reshape_output_1


class Reshape193(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 49, 192))
        return reshape_output_1


class Reshape194(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 7, 4, 7, 192))
        return reshape_output_1


class Reshape195(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(784, 192))
        return reshape_output_1


class Reshape196(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 49, 576))
        return reshape_output_1


class Reshape197(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 49, 3, 6, 32))
        return reshape_output_1


class Reshape198(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 6, 49, 32))
        return reshape_output_1


class Reshape199(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(96, 49, 32))
        return reshape_output_1


class Reshape200(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 6, 49, 49))
        return reshape_output_1


class Reshape201(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(49, 49, 6))
        return reshape_output_1


class Reshape202(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(96, 49, 49))
        return reshape_output_1


class Reshape203(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 6, 49, 49))
        return reshape_output_1


class Reshape204(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(96, 32, 49))
        return reshape_output_1


class Reshape205(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 4, 7, 7, 192))
        return reshape_output_1


class Reshape206(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 28, 28, 768))
        return reshape_output_1


class Reshape207(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(784, 768))
        return reshape_output_1


class Reshape208(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(196, 768))
        return reshape_output_1


class Reshape209(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 14, 14, 384))
        return reshape_output_1


class Reshape210(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(4, 49, 384))
        return reshape_output_1


class Reshape211(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2, 7, 2, 7, 384))
        return reshape_output_1


class Reshape212(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(196, 384))
        return reshape_output_1


class Reshape213(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(4, 49, 1152))
        return reshape_output_1


class Reshape214(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(4, 49, 3, 12, 32))
        return reshape_output_1


class Reshape215(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(4, 12, 49, 32))
        return reshape_output_1


class Reshape216(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(48, 49, 32))
        return reshape_output_1


class Reshape217(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(4, 12, 49, 49))
        return reshape_output_1


class Reshape218(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(49, 49, 12))
        return reshape_output_1


class Reshape219(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(48, 49, 49))
        return reshape_output_1


class Reshape220(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 12, 49, 49))
        return reshape_output_1


class Reshape221(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(48, 32, 49))
        return reshape_output_1


class Reshape222(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2, 2, 7, 7, 384))
        return reshape_output_1


class Reshape223(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 14, 14, 1536))
        return reshape_output_1


class Reshape224(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(196, 1536))
        return reshape_output_1


class Reshape225(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(49, 1536))
        return reshape_output_1


class Reshape226(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 7, 7, 768))
        return reshape_output_1


class Reshape227(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 49, 768))
        return reshape_output_1


class Reshape228(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 7, 1, 7, 768))
        return reshape_output_1


class Reshape229(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(49, 768))
        return reshape_output_1


class Reshape230(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 49, 2304))
        return reshape_output_1


class Reshape231(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 49, 3, 24, 32))
        return reshape_output_1


class Reshape232(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 24, 49, 32))
        return reshape_output_1


class Reshape233(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(24, 49, 32))
        return reshape_output_1


class Reshape234(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 24, 49, 49))
        return reshape_output_1


class Reshape235(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(49, 49, 24))
        return reshape_output_1


class Reshape236(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(24, 49, 49))
        return reshape_output_1


class Reshape237(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(24, 32, 49))
        return reshape_output_1


class Reshape238(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 1, 7, 7, 768))
        return reshape_output_1


class Reshape239(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 7, 7, 3072))
        return reshape_output_1


class Reshape240(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(49, 3072))
        return reshape_output_1


class Reshape241(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 768, 1, 1))
        return reshape_output_1


class Reshape242(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 25088, 1, 1))
        return reshape_output_1


class Reshape243(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 25088))
        return reshape_output_1


class Reshape244(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 512, 49, 1))
        return reshape_output_1


class Reshape245(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 768, 196, 1))
        return reshape_output_1


class Reshape246(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 768, 196))
        return reshape_output_1


class Reshape247(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(197, 768))
        return reshape_output_1


class Reshape248(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 197, 12, 64))
        return reshape_output_1


class Reshape249(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 197, 768))
        return reshape_output_1


class Reshape250(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(197, 1, 768))
        return reshape_output_1


class Reshape251(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 197, 64))
        return reshape_output_1


class Reshape252(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 197, 197))
        return reshape_output_1


class Reshape253(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 197, 197))
        return reshape_output_1


class Reshape254(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 197, 64))
        return reshape_output_1


class Reshape255(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1024, 49))
        return reshape_output_1


class Reshape256(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(50, 1024))
        return reshape_output_1


class Reshape257(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(50, 1, 3072))
        return reshape_output_1


class Reshape258(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(50, 1, 3, 1024))
        return reshape_output_1


class Reshape259(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(50, 16, 64))
        return reshape_output_1


class Reshape260(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 50, 64))
        return reshape_output_1


class Reshape261(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 50, 64))
        return reshape_output_1


class Reshape262(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 50, 50))
        return reshape_output_1


class Reshape263(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 50, 50))
        return reshape_output_1


class Reshape264(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(50, 1, 1024))
        return reshape_output_1


class Reshape265(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(728, 1, 3, 3))
        return reshape_output_1


class Reshape266(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1536, 1, 3, 3))
        return reshape_output_1


class Reshape267(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 255, 80, 80))
        return reshape_output_1


class Reshape268(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 255, 6400))
        return reshape_output_1


class Reshape269(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 3, 85, 6400))
        return reshape_output_1


class Reshape270(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 19200, 85))
        return reshape_output_1


class Reshape271(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 255, 40, 40))
        return reshape_output_1


class Reshape272(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 255, 1600))
        return reshape_output_1


class Reshape273(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 3, 85, 1600))
        return reshape_output_1


class Reshape274(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4800, 85))
        return reshape_output_1


class Reshape275(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 255, 20, 20))
        return reshape_output_1


class Reshape276(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 255, 400))
        return reshape_output_1


class Reshape277(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 3, 85, 400))
        return reshape_output_1


class Reshape278(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1200, 85))
        return reshape_output_1


class Reshape279(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 255, 60, 60))
        return reshape_output_1


class Reshape280(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 255, 3600))
        return reshape_output_1


class Reshape281(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 3, 85, 3600))
        return reshape_output_1


class Reshape282(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 10800, 85))
        return reshape_output_1


class Reshape283(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 255, 30, 30))
        return reshape_output_1


class Reshape284(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 255, 900))
        return reshape_output_1


class Reshape285(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 3, 85, 900))
        return reshape_output_1


class Reshape286(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2700, 85))
        return reshape_output_1


class Reshape287(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 255, 15, 15))
        return reshape_output_1


class Reshape288(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 255, 225))
        return reshape_output_1


class Reshape289(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 3, 85, 225))
        return reshape_output_1


class Reshape290(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 675, 85))
        return reshape_output_1


class Reshape291(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 144, 6400))
        return reshape_output_1


class Reshape292(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 144, 1600))
        return reshape_output_1


class Reshape293(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 144, 400))
        return reshape_output_1


class Reshape294(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 16, 8400))
        return reshape_output_1


class Reshape295(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 8400))
        return reshape_output_1


class Reshape296(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 85, 6400, 1))
        return reshape_output_1


class Reshape297(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 85, 1600, 1))
        return reshape_output_1


class Reshape298(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 85, 400, 1))
        return reshape_output_1


class Reshape299(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 64, 16384))
        return reshape_output_1


class Reshape300(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16384, 1, 64))
        return reshape_output_1


class Reshape301(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 128, 64))
        return reshape_output_1


class Reshape302(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 64, 128, 128))
        return reshape_output_1


class Reshape303(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 128, 128))
        return reshape_output_1


class Reshape304(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 16384))
        return reshape_output_1


class Reshape305(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 4096))
        return reshape_output_1


class Reshape306(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4096, 2, 64))
        return reshape_output_1


class Reshape307(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 64, 64, 128))
        return reshape_output_1


class Reshape308(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 4096, 64))
        return reshape_output_1


class Reshape309(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 64, 64))
        return reshape_output_1


class Reshape310(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 256))
        return reshape_output_1


class Reshape311(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(256, 128))
        return reshape_output_1


class Reshape312(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 2, 64))
        return reshape_output_1


class Reshape313(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 128))
        return reshape_output_1


class Reshape314(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 64, 256))
        return reshape_output_1


class Reshape315(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 256, 64))
        return reshape_output_1


class Reshape316(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2, 4096, 64))
        return reshape_output_1


class Reshape317(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(4096, 128))
        return reshape_output_1


class Reshape318(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4096, 128))
        return reshape_output_1


class Reshape319(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 512, 64, 64))
        return reshape_output_1


class Reshape320(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 512, 4096))
        return reshape_output_1


class Reshape321(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 320, 1024))
        return reshape_output_1


class Reshape322(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1024, 5, 64))
        return reshape_output_1


class Reshape323(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 32, 320))
        return reshape_output_1


class Reshape324(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(5, 1024, 64))
        return reshape_output_1


class Reshape325(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 320, 32, 32))
        return reshape_output_1


class Reshape326(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 320, 256))
        return reshape_output_1


class Reshape327(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(256, 320))
        return reshape_output_1


class Reshape328(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 5, 64))
        return reshape_output_1


class Reshape329(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 320))
        return reshape_output_1


class Reshape330(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(5, 64, 256))
        return reshape_output_1


class Reshape331(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(5, 256, 64))
        return reshape_output_1


class Reshape332(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 5, 1024, 64))
        return reshape_output_1


class Reshape333(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1024, 320))
        return reshape_output_1


class Reshape334(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1024, 320))
        return reshape_output_1


class Reshape335(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1280, 32, 32))
        return reshape_output_1


class Reshape336(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1280, 1024))
        return reshape_output_1


class Reshape337(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 512, 256))
        return reshape_output_1


class Reshape338(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 256, 64))
        return reshape_output_1


class Reshape339(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 64, 256))
        return reshape_output_1


class Reshape340(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 256, 64))
        return reshape_output_1


class Reshape341(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2048, 16, 16))
        return reshape_output_1


class Reshape342(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2048, 256))
        return reshape_output_1


class Reshape343(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 768, 16, 16))
        return reshape_output_1


class Reshape344(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 768, 32, 32))
        return reshape_output_1


class Reshape345(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 768, 64, 64))
        return reshape_output_1


class Reshape346(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 768, 128, 128))
        return reshape_output_1


class Reshape347(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(9, 768))
        return reshape_output_1


class Reshape348(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 9, 12, 64))
        return reshape_output_1


class Reshape349(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 9, 768))
        return reshape_output_1


class Reshape350(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 9, 64))
        return reshape_output_1


class Reshape351(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 64, 9))
        return reshape_output_1


class Reshape352(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 9, 9))
        return reshape_output_1


class Reshape353(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 9, 9))
        return reshape_output_1


class Reshape354(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 9, 64))
        return reshape_output_1


class Reshape355(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1024, 196, 1))
        return reshape_output_1


class Reshape356(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1024, 196))
        return reshape_output_1


class Reshape357(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(197, 1024))
        return reshape_output_1


class Reshape358(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 197, 16, 64))
        return reshape_output_1


class Reshape359(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 197, 1024))
        return reshape_output_1


class Reshape360(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(197, 1, 1024))
        return reshape_output_1


class Reshape361(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 197, 64))
        return reshape_output_1


class Reshape362(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 197, 197))
        return reshape_output_1


class Reshape363(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 27, 27, 16))
        return reshape_output_1


class Reshape364(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(729, 16))
        return reshape_output_1


class Reshape365(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(197, 197, 16))
        return reshape_output_1


class Reshape366(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 197, 197))
        return reshape_output_1


class Reshape367(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 197, 64))
        return reshape_output_1


class Reshape368(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(480, 1, 5, 5))
        return reshape_output_1


class Reshape369(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(480, 1, 3, 3))
        return reshape_output_1


class Reshape370(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(960, 1, 3, 3))
        return reshape_output_1


class Reshape371(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1344, 1, 5, 5))
        return reshape_output_1


class Reshape372(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2304, 1, 5, 5))
        return reshape_output_1


class Reshape373(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(3840, 1, 3, 3))
        return reshape_output_1


class Reshape374(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2560, 1, 1))
        return reshape_output_1


class Reshape375(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 1, 3, 3))
        return reshape_output_1


class Reshape376(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 1, 3, 3))
        return reshape_output_1


class Reshape377(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 1, 3, 3))
        return reshape_output_1


class Reshape378(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(36, 1, 3, 3))
        return reshape_output_1


class Reshape379(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(72, 1, 5, 5))
        return reshape_output_1


class Reshape380(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(20, 1, 3, 3))
        return reshape_output_1


class Reshape381(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(24, 1, 5, 5))
        return reshape_output_1


class Reshape382(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(60, 1, 3, 3))
        return reshape_output_1


class Reshape383(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(120, 1, 3, 3))
        return reshape_output_1


class Reshape384(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(100, 1, 3, 3))
        return reshape_output_1


class Reshape385(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(92, 1, 3, 3))
        return reshape_output_1


class Reshape386(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(80, 1, 3, 3))
        return reshape_output_1


class Reshape387(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(112, 1, 5, 5))
        return reshape_output_1


class Reshape388(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1280, 1, 1))
        return reshape_output_1


class Reshape389(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(96, 1, 3, 3))
        return reshape_output_1


class Reshape390(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(384, 1, 3, 3))
        return reshape_output_1


class Reshape391(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(768, 1, 3, 3))
        return reshape_output_1


class Reshape392(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(720, 1, 3, 3))
        return reshape_output_1


class Reshape393(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(72, 1, 3, 3))
        return reshape_output_1


class Reshape394(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(120, 1, 5, 5))
        return reshape_output_1


class Reshape395(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(200, 1, 3, 3))
        return reshape_output_1


class Reshape396(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(184, 1, 3, 3))
        return reshape_output_1


class Reshape397(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(11, 2560))
        return reshape_output_1


class Reshape398(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 11, 32, 80))
        return reshape_output_1


class Reshape399(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 11, 2560))
        return reshape_output_1


class Reshape400(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 11, 80))
        return reshape_output_1


class Reshape401(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 11, 11))
        return reshape_output_1


class Reshape402(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 11, 11))
        return reshape_output_1


class Reshape403(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 11, 80))
        return reshape_output_1


class Reshape404(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 11, 10240))
        return reshape_output_1


class Reshape405(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(6, 1024))
        return reshape_output_1


class Reshape406(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 6, 16, 64))
        return reshape_output_1


class Reshape407(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 6, 1024))
        return reshape_output_1


class Reshape408(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 6, 64))
        return reshape_output_1


class Reshape409(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 6, 6))
        return reshape_output_1


class Reshape410(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 6, 6))
        return reshape_output_1


class Reshape411(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 6, 64))
        return reshape_output_1


class Reshape412(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 6, 2816))
        return reshape_output_1


class Reshape413(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(35, 1536))
        return reshape_output_1


class Reshape414(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 35, 12, 128))
        return reshape_output_1


class Reshape415(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 35, 1536))
        return reshape_output_1


class Reshape416(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 35, 128))
        return reshape_output_1


class Reshape417(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 35, 256))
        return reshape_output_1


class Reshape418(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 35, 2, 128))
        return reshape_output_1


class Reshape419(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 35, 35))
        return reshape_output_1


class Reshape420(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 35, 35))
        return reshape_output_1


class Reshape421(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 35, 128))
        return reshape_output_1


class Reshape422(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 35, 8960))
        return reshape_output_1


class Reshape423(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(29, 1536))
        return reshape_output_1


class Reshape424(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 29, 12, 128))
        return reshape_output_1


class Reshape425(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 29, 1536))
        return reshape_output_1


class Reshape426(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 29, 128))
        return reshape_output_1


class Reshape427(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 29, 256))
        return reshape_output_1


class Reshape428(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 29, 2, 128))
        return reshape_output_1


class Reshape429(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 29, 29))
        return reshape_output_1


class Reshape430(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 29, 29))
        return reshape_output_1


class Reshape431(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 29, 128))
        return reshape_output_1


class Reshape432(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 29, 8960))
        return reshape_output_1


class Reshape433(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 3024, 1, 1))
        return reshape_output_1


class Reshape434(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 3712, 1, 1))
        return reshape_output_1


class Reshape435(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2520, 1, 1))
        return reshape_output_1


class Reshape436(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1008, 1, 1))
        return reshape_output_1


class Reshape437(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 8, 8, 8, 96))
        return reshape_output_1


class Reshape438(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(4096, 96))
        return reshape_output_1


class Reshape439(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 64, 64, 96))
        return reshape_output_1


class Reshape440(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 64, 288))
        return reshape_output_1


class Reshape441(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 64, 3, 3, 32))
        return reshape_output_1


class Reshape442(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 3, 64, 32))
        return reshape_output_1


class Reshape443(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(192, 64, 32))
        return reshape_output_1


class Reshape444(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 3, 64, 64))
        return reshape_output_1


class Reshape445(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(225, 2))
        return reshape_output_1


class Reshape446(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 15, 15, 512))
        return reshape_output_1


class Reshape447(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(225, 512))
        return reshape_output_1


class Reshape448(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(225, 3))
        return reshape_output_1


class Reshape449(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 64, 3))
        return reshape_output_1


class Reshape450(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(192, 64, 64))
        return reshape_output_1


class Reshape451(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 64, 3, 64, 64))
        return reshape_output_1


class Reshape452(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(192, 32, 64))
        return reshape_output_1


class Reshape453(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 64, 96))
        return reshape_output_1


class Reshape454(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 64, 64, 384))
        return reshape_output_1


class Reshape455(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(4096, 384))
        return reshape_output_1


class Reshape456(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1024, 384))
        return reshape_output_1


class Reshape457(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 32, 192))
        return reshape_output_1


class Reshape458(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 64, 192))
        return reshape_output_1


class Reshape459(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 8, 4, 8, 192))
        return reshape_output_1


class Reshape460(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1024, 192))
        return reshape_output_1


class Reshape461(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 64, 576))
        return reshape_output_1


class Reshape462(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 64, 3, 6, 32))
        return reshape_output_1


class Reshape463(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 6, 64, 32))
        return reshape_output_1


class Reshape464(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(96, 64, 32))
        return reshape_output_1


class Reshape465(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 6, 64, 64))
        return reshape_output_1


class Reshape466(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(225, 6))
        return reshape_output_1


class Reshape467(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 64, 6))
        return reshape_output_1


class Reshape468(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(96, 64, 64))
        return reshape_output_1


class Reshape469(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 6, 64, 64))
        return reshape_output_1


class Reshape470(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(96, 32, 64))
        return reshape_output_1


class Reshape471(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 4, 8, 8, 192))
        return reshape_output_1


class Reshape472(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 32, 768))
        return reshape_output_1


class Reshape473(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1024, 768))
        return reshape_output_1


class Reshape474(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 16, 384))
        return reshape_output_1


class Reshape475(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(4, 64, 384))
        return reshape_output_1


class Reshape476(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2, 8, 2, 8, 384))
        return reshape_output_1


class Reshape477(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(256, 384))
        return reshape_output_1


class Reshape478(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(4, 64, 1152))
        return reshape_output_1


class Reshape479(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(4, 64, 3, 12, 32))
        return reshape_output_1


class Reshape480(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(4, 12, 64, 32))
        return reshape_output_1


class Reshape481(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(48, 64, 32))
        return reshape_output_1


class Reshape482(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(4, 12, 64, 64))
        return reshape_output_1


class Reshape483(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(225, 12))
        return reshape_output_1


class Reshape484(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 64, 12))
        return reshape_output_1


class Reshape485(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(48, 64, 64))
        return reshape_output_1


class Reshape486(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 12, 64, 64))
        return reshape_output_1


class Reshape487(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(48, 32, 64))
        return reshape_output_1


class Reshape488(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2, 2, 8, 8, 384))
        return reshape_output_1


class Reshape489(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 16, 1536))
        return reshape_output_1


class Reshape490(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(256, 1536))
        return reshape_output_1


class Reshape491(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 1536))
        return reshape_output_1


class Reshape492(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 8, 768))
        return reshape_output_1


class Reshape493(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 64, 768))
        return reshape_output_1


class Reshape494(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 8, 1, 8, 768))
        return reshape_output_1


class Reshape495(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 768))
        return reshape_output_1


class Reshape496(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 64, 2304))
        return reshape_output_1


class Reshape497(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 64, 3, 24, 32))
        return reshape_output_1


class Reshape498(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 24, 64, 32))
        return reshape_output_1


class Reshape499(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(24, 64, 32))
        return reshape_output_1


class Reshape500(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 24, 64, 64))
        return reshape_output_1


class Reshape501(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(225, 24))
        return reshape_output_1


class Reshape502(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 64, 24))
        return reshape_output_1


class Reshape503(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(24, 64, 64))
        return reshape_output_1


class Reshape504(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(24, 32, 64))
        return reshape_output_1


class Reshape505(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 1, 8, 8, 768))
        return reshape_output_1


class Reshape506(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 8, 3072))
        return reshape_output_1


class Reshape507(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 3072))
        return reshape_output_1


class Reshape508(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 513))
        return reshape_output_1


class Reshape509(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(513, 768))
        return reshape_output_1


class Reshape510(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 513, 12, 64))
        return reshape_output_1


class Reshape511(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 513, 768))
        return reshape_output_1


class Reshape512(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 513, 64))
        return reshape_output_1


class Reshape513(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 513, 513))
        return reshape_output_1


class Reshape514(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 513, 513))
        return reshape_output_1


class Reshape515(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 64, 513))
        return reshape_output_1


class Reshape516(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 513, 64))
        return reshape_output_1


class Reshape517(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 61))
        return reshape_output_1


class Reshape518(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(61, 768))
        return reshape_output_1


class Reshape519(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 61, 12, 64))
        return reshape_output_1


class Reshape520(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 61, 768))
        return reshape_output_1


class Reshape521(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 61, 64))
        return reshape_output_1


class Reshape522(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 61, 61))
        return reshape_output_1


class Reshape523(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 61, 61))
        return reshape_output_1


class Reshape524(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 64, 61))
        return reshape_output_1


class Reshape525(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 61, 64))
        return reshape_output_1


class Reshape526(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 513, 61))
        return reshape_output_1


class Reshape527(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 513, 61))
        return reshape_output_1


class Reshape528(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 255, 10, 10))
        return reshape_output_1


class Reshape529(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 255, 100))
        return reshape_output_1


class Reshape530(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 3, 85, 100))
        return reshape_output_1


class Reshape531(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 300, 85))
        return reshape_output_1


class Reshape532(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(128, 2048))
        return reshape_output_1


class Reshape533(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 16, 128))
        return reshape_output_1


class Reshape534(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 2048))
        return reshape_output_1


class Reshape535(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 27, 27, 12))
        return reshape_output_1


class Reshape536(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(729, 12))
        return reshape_output_1


class Reshape537(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(197, 197, 12))
        return reshape_output_1


class Reshape538(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 192, 196, 1))
        return reshape_output_1


class Reshape539(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(197, 192))
        return reshape_output_1


class Reshape540(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 197, 3, 64))
        return reshape_output_1


class Reshape541(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 197, 192))
        return reshape_output_1


class Reshape542(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(3, 197, 64))
        return reshape_output_1


class Reshape543(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 3, 197, 197))
        return reshape_output_1


class Reshape544(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(3, 197, 197))
        return reshape_output_1


class Reshape545(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 3, 197, 64))
        return reshape_output_1


class Reshape546(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 192))
        return reshape_output_1


class Reshape547(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(384, 1, 5, 5))
        return reshape_output_1


class Reshape548(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(768, 1, 5, 5))
        return reshape_output_1


class Reshape549(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1056, 1, 5, 5))
        return reshape_output_1


class Reshape550(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1824, 1, 5, 5))
        return reshape_output_1


class Reshape551(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1824, 1, 3, 3))
        return reshape_output_1


class Reshape552(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(3072, 1, 3, 3))
        return reshape_output_1


class Reshape553(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(88, 1, 3, 3))
        return reshape_output_1


class Reshape554(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(96, 1, 5, 5))
        return reshape_output_1


class Reshape555(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(144, 1, 5, 5))
        return reshape_output_1


class Reshape556(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(29, 896))
        return reshape_output_1


class Reshape557(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 29, 14, 64))
        return reshape_output_1


class Reshape558(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 29, 896))
        return reshape_output_1


class Reshape559(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(14, 29, 64))
        return reshape_output_1


class Reshape560(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 29, 128))
        return reshape_output_1


class Reshape561(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 29, 2, 64))
        return reshape_output_1


class Reshape562(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 14, 29, 29))
        return reshape_output_1


class Reshape563(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(14, 29, 29))
        return reshape_output_1


class Reshape564(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 14, 29, 64))
        return reshape_output_1


class Reshape565(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 29, 4864))
        return reshape_output_1


class Reshape566(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 64, 16384, 1))
        return reshape_output_1


class Reshape567(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 16384, 1))
        return reshape_output_1


class Reshape568(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 4096, 1))
        return reshape_output_1


class Reshape569(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 512, 4096, 1))
        return reshape_output_1


class Reshape570(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 320, 1024, 1))
        return reshape_output_1


class Reshape571(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1280, 1, 3, 3))
        return reshape_output_1


class Reshape572(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1280, 1024, 1))
        return reshape_output_1


class Reshape573(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 512, 256, 1))
        return reshape_output_1


class Reshape574(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2048, 1, 3, 3))
        return reshape_output_1


class Reshape575(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2048, 256, 1))
        return reshape_output_1


class Reshape576(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 64, 128))
        return reshape_output_1


class Reshape577(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 64, 128))
        return reshape_output_1


class Reshape578(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 768, 128, 1))
        return reshape_output_1


class Reshape579(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(768, 768, 1, 1))
        return reshape_output_1


class Reshape580(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 768, 128))
        return reshape_output_1


class Reshape581(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(197, 1, 2304))
        return reshape_output_1


class Reshape582(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(197, 1, 3, 768))
        return reshape_output_1


class Reshape583(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(197, 12, 64))
        return reshape_output_1


class Reshape584(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(197, 1, 3072))
        return reshape_output_1


class Reshape585(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(197, 1, 3, 1024))
        return reshape_output_1


class Reshape586(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(197, 16, 64))
        return reshape_output_1


class Reshape587(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(160, 1, 3, 3))
        return reshape_output_1


class Reshape588(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(224, 1, 3, 3))
        return reshape_output_1


class Reshape589(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256))
        return reshape_output_1


class Reshape590(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 1024))
        return reshape_output_1


class Reshape591(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 4, 256))
        return reshape_output_1


class Reshape592(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(256, 1024))
        return reshape_output_1


class Reshape593(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 16, 64))
        return reshape_output_1


class Reshape594(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 256, 64))
        return reshape_output_1


class Reshape595(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 256, 256))
        return reshape_output_1


class Reshape596(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 256, 256))
        return reshape_output_1


class Reshape597(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 256, 64))
        return reshape_output_1


class Reshape598(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 17, 4480))
        return reshape_output_1


class Reshape599(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 4480))
        return reshape_output_1


class Reshape600(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 17, 1120))
        return reshape_output_1


class Reshape601(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 1120))
        return reshape_output_1


class Reshape602(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 17, 280))
        return reshape_output_1


class Reshape603(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 280))
        return reshape_output_1


class Reshape604(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 80, 4480))
        return reshape_output_1


class Reshape605(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 80, 1120))
        return reshape_output_1


class Reshape606(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 80, 280))
        return reshape_output_1


class Reshape607(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64,))
        return reshape_output_1


class Reshape608(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(256,))
        return reshape_output_1


class Reshape609(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(512,))
        return reshape_output_1


class Reshape610(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(128,))
        return reshape_output_1


class Reshape611(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1024,))
        return reshape_output_1


class Reshape612(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2048,))
        return reshape_output_1


class Reshape613(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 576))
        return reshape_output_1


class Reshape614(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1280))
        return reshape_output_1


class Reshape615(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 192, 196))
        return reshape_output_1


class Reshape616(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(3, 64, 197))
        return reshape_output_1


class Reshape617(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 9216, 1, 1))
        return reshape_output_1


class Reshape618(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(25, 1, 96))
        return reshape_output_1


class Reshape619(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 16, 32, 1))
        return reshape_output_1


class Reshape620(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 64, 256))
        return reshape_output_1


class Reshape621(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 4096))
        return reshape_output_1


class Reshape622(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1920, 1, 1))
        return reshape_output_1


class Reshape623(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(528, 1, 3, 3))
        return reshape_output_1


class Reshape624(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(528, 1, 5, 5))
        return reshape_output_1


class Reshape625(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(720, 1, 5, 5))
        return reshape_output_1


class Reshape626(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1248, 1, 5, 5))
        return reshape_output_1


class Reshape627(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1248, 1, 3, 3))
        return reshape_output_1


class Reshape628(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2112, 1, 3, 3))
        return reshape_output_1


class Reshape629(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1408, 1, 1))
        return reshape_output_1


class Reshape630(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 12, 64))
        return reshape_output_1


class Reshape631(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 768))
        return reshape_output_1


class Reshape632(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 256, 64))
        return reshape_output_1


class Reshape633(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 256, 256))
        return reshape_output_1


class Reshape634(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 256, 256))
        return reshape_output_1


class Reshape635(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 256, 64))
        return reshape_output_1


class Reshape636(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 3072))
        return reshape_output_1


class Reshape637(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(256, 3072))
        return reshape_output_1


class Reshape638(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32))
        return reshape_output_1


class Reshape639(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 768))
        return reshape_output_1


class Reshape640(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 12, 64))
        return reshape_output_1


class Reshape641(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 768))
        return reshape_output_1


class Reshape642(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 32, 64))
        return reshape_output_1


class Reshape643(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 2048))
        return reshape_output_1


class Reshape644(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 32, 32))
        return reshape_output_1


class Reshape645(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 32, 32))
        return reshape_output_1


class Reshape646(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 32, 64))
        return reshape_output_1


class Reshape647(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 12, 64))
        return reshape_output_1


class Reshape648(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 2))
        return reshape_output_1


class Reshape649(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 2048))
        return reshape_output_1


class Reshape650(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 12, 12))
        return reshape_output_1


class Reshape651(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 12, 12))
        return reshape_output_1


class Reshape652(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 8192))
        return reshape_output_1


class Reshape653(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(513, 512))
        return reshape_output_1


class Reshape654(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 513, 8, 64))
        return reshape_output_1


class Reshape655(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 513, 512))
        return reshape_output_1


class Reshape656(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 513, 64))
        return reshape_output_1


class Reshape657(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 513, 513))
        return reshape_output_1


class Reshape658(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 513, 513))
        return reshape_output_1


class Reshape659(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 64, 513))
        return reshape_output_1


class Reshape660(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 513, 64))
        return reshape_output_1


class Reshape661(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(61, 512))
        return reshape_output_1


class Reshape662(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 61, 8, 64))
        return reshape_output_1


class Reshape663(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 61, 512))
        return reshape_output_1


class Reshape664(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 61, 64))
        return reshape_output_1


class Reshape665(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 61, 61))
        return reshape_output_1


class Reshape666(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 61, 61))
        return reshape_output_1


class Reshape667(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 64, 61))
        return reshape_output_1


class Reshape668(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 61, 64))
        return reshape_output_1


class Reshape669(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 513, 61))
        return reshape_output_1


class Reshape670(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 513, 61))
        return reshape_output_1


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (
        Reshape0,
        [((1, 11, 312), torch.float32)],
        {"model_names": ["pd_albert_chinese_tiny_mlm_padlenlp"], "pcc": 0.99, "args": {"shape": "(11, 312)"}},
    ),
    (
        Reshape1,
        [((1, 11, 312), torch.float32)],
        {"model_names": ["pd_albert_chinese_tiny_mlm_padlenlp"], "pcc": 0.99, "args": {"shape": "(1, 11, 12, 26)"}},
    ),
    (
        Reshape2,
        [((11, 312), torch.float32)],
        {"model_names": ["pd_albert_chinese_tiny_mlm_padlenlp"], "pcc": 0.99, "args": {"shape": "(1, 11, 312)"}},
    ),
    (
        Reshape3,
        [((1, 12, 11, 26), torch.float32)],
        {"model_names": ["pd_albert_chinese_tiny_mlm_padlenlp"], "pcc": 0.99, "args": {"shape": "(12, 11, 26)"}},
    ),
    (
        Reshape4,
        [((1, 12, 26, 11), torch.float32)],
        {"model_names": ["pd_albert_chinese_tiny_mlm_padlenlp"], "pcc": 0.99, "args": {"shape": "(12, 26, 11)"}},
    ),
    (
        Reshape5,
        [((12, 11, 11), torch.float32)],
        {"model_names": ["pd_albert_chinese_tiny_mlm_padlenlp"], "pcc": 0.99, "args": {"shape": "(1, 12, 11, 11)"}},
    ),
    (
        Reshape6,
        [((1, 12, 11, 11), torch.float32)],
        {"model_names": ["pd_albert_chinese_tiny_mlm_padlenlp"], "pcc": 0.99, "args": {"shape": "(12, 11, 11)"}},
    ),
    (
        Reshape7,
        [((12, 11, 26), torch.float32)],
        {"model_names": ["pd_albert_chinese_tiny_mlm_padlenlp"], "pcc": 0.99, "args": {"shape": "(1, 12, 11, 26)"}},
    ),
    (
        Reshape0,
        [((1, 11, 12, 26), torch.float32)],
        {"model_names": ["pd_albert_chinese_tiny_mlm_padlenlp"], "pcc": 0.99, "args": {"shape": "(11, 312)"}},
    ),
    (
        Reshape8,
        [((1, 2048, 1, 1), torch.float32)],
        {
            "model_names": ["pd_resnet_101_img_cls_paddlemodels", "pd_resnet_152_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {"shape": "(1, 2048, 1, 1)"},
        },
    ),
    (
        Reshape9,
        [((1, 2048, 1, 1), torch.float32)],
        {"model_names": ["onnx_xception_xception71_tf_in1k_img_cls_timm"], "pcc": 0.99, "args": {"shape": "(1, 2048)"}},
    ),
    (
        Reshape10,
        [((1, 128, 1024), torch.float32)],
        {
            "model_names": [
                "pt_albert_large_v2_token_cls_hf",
                "pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(128, 1024)"},
        },
    ),
    (
        Reshape11,
        [((1, 128, 1024), torch.float32)],
        {
            "model_names": [
                "pt_albert_large_v2_token_cls_hf",
                "pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 128, 16, 64)"},
        },
    ),
    (
        Reshape12,
        [((128, 1024), torch.float32)],
        {
            "model_names": [
                "pt_albert_large_v2_token_cls_hf",
                "pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 128, 1024)"},
        },
    ),
    (
        Reshape13,
        [((1, 16, 128, 64), torch.float32)],
        {
            "model_names": [
                "pt_albert_large_v2_token_cls_hf",
                "pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(16, 128, 64)"},
        },
    ),
    (
        Reshape14,
        [((16, 128, 128), torch.float32)],
        {
            "model_names": [
                "pt_albert_large_v2_token_cls_hf",
                "pt_albert_xlarge_v2_token_cls_hf",
                "pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 16, 128, 128)"},
        },
    ),
    (
        Reshape15,
        [((1, 16, 128, 128), torch.float32)],
        {
            "model_names": [
                "pt_albert_large_v2_token_cls_hf",
                "pt_albert_xlarge_v2_token_cls_hf",
                "pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(16, 128, 128)"},
        },
    ),
    (
        Reshape16,
        [((16, 128, 64), torch.float32)],
        {
            "model_names": [
                "pt_albert_large_v2_token_cls_hf",
                "pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 16, 128, 64)"},
        },
    ),
    (
        Reshape10,
        [((1, 128, 16, 64), torch.float32)],
        {
            "model_names": [
                "pt_albert_large_v2_token_cls_hf",
                "pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(128, 1024)"},
        },
    ),
    (
        Reshape17,
        [((1, 1000, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_dla_dla46_c_visual_bb_torchvision",
                "pt_dla_dla60_visual_bb_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 1000)"},
        },
    ),
    (
        Reshape18,
        [((1, 128, 768), torch.float32)],
        {
            "model_names": [
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(128, 768)"},
        },
    ),
    (
        Reshape19,
        [((1, 128, 768), torch.float32)],
        {
            "model_names": [
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 128, 12, 64)"},
        },
    ),
    (
        Reshape20,
        [((128, 768), torch.float32)],
        {
            "model_names": [
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 128, 768)"},
        },
    ),
    (
        Reshape21,
        [((1, 12, 128, 64), torch.float32)],
        {
            "model_names": [
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(12, 128, 64)"},
        },
    ),
    (
        Reshape22,
        [((12, 128, 128), torch.float32)],
        {
            "model_names": [
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 12, 128, 128)"},
        },
    ),
    (
        Reshape23,
        [((1, 12, 128, 128), torch.float32)],
        {
            "model_names": [
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(12, 128, 128)"},
        },
    ),
    (
        Reshape24,
        [((12, 128, 64), torch.float32)],
        {
            "model_names": [
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 12, 128, 64)"},
        },
    ),
    (
        Reshape18,
        [((1, 128, 12, 64), torch.float32)],
        {
            "model_names": [
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(128, 768)"},
        },
    ),
    (
        Reshape25,
        [((128, 1), torch.float32)],
        {
            "model_names": ["pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader"],
            "pcc": 0.99,
            "args": {"shape": "(1, 128, 1)"},
        },
    ),
    (
        Reshape26,
        [((1, 128), torch.float32)],
        {
            "model_names": ["pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader"],
            "pcc": 0.99,
            "args": {"shape": "(1, 128)"},
        },
    ),
    (
        Reshape27,
        [((1, 1, 768), torch.float32)],
        {
            "model_names": [
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 768)"},
        },
    ),
    (
        Reshape28,
        [((1, 1), torch.float32)],
        {
            "model_names": ["pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader"],
            "pcc": 0.99,
            "args": {"shape": "(1,)"},
        },
    ),
    (
        Reshape29,
        [((40, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(40, 1, 3, 3)"},
        },
    ),
    (
        Reshape30,
        [((24, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv1_google_mobilenet_v1_0_75_192_img_cls_hf",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(24, 1, 3, 3)"},
        },
    ),
    (
        Reshape31,
        [((144, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(144, 1, 3, 3)"},
        },
    ),
    (
        Reshape32,
        [((192, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
                "pt_mobilenetv1_google_mobilenet_v1_0_75_192_img_cls_hf",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(192, 1, 3, 3)"},
        },
    ),
    (
        Reshape33,
        [((192, 1, 5, 5), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(192, 1, 5, 5)"},
        },
    ),
    (
        Reshape34,
        [((288, 1, 5, 5), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(288, 1, 5, 5)"},
        },
    ),
    (
        Reshape35,
        [((288, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(288, 1, 3, 3)"},
        },
    ),
    (
        Reshape36,
        [((576, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(576, 1, 3, 3)"},
        },
    ),
    (
        Reshape37,
        [((576, 1, 5, 5), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(576, 1, 5, 5)"},
        },
    ),
    (
        Reshape38,
        [((816, 1, 5, 5), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(816, 1, 5, 5)"},
        },
    ),
    (
        Reshape39,
        [((1392, 1, 5, 5), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1392, 1, 5, 5)"},
        },
    ),
    (
        Reshape40,
        [((1392, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1392, 1, 3, 3)"},
        },
    ),
    (
        Reshape41,
        [((2304, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(2304, 1, 3, 3)"},
        },
    ),
    (
        Reshape42,
        [((1, 1536, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_inception_inception_v4_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 1536, 1, 1)"},
        },
    ),
    (
        Reshape43,
        [((48, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv1_google_mobilenet_v1_0_75_192_img_cls_hf",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(48, 1, 3, 3)"},
        },
    ),
    (
        Reshape44,
        [((336, 1, 5, 5), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(336, 1, 5, 5)"},
        },
    ),
    (
        Reshape45,
        [((336, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(336, 1, 3, 3)"},
        },
    ),
    (
        Reshape46,
        [((672, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(672, 1, 3, 3)"},
        },
    ),
    (
        Reshape47,
        [((672, 1, 5, 5), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(672, 1, 5, 5)"},
        },
    ),
    (
        Reshape48,
        [((960, 1, 5, 5), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(960, 1, 5, 5)"},
        },
    ),
    (
        Reshape49,
        [((1632, 1, 5, 5), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1632, 1, 5, 5)"},
        },
    ),
    (
        Reshape50,
        [((1632, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1632, 1, 3, 3)"},
        },
    ),
    (
        Reshape51,
        [((2688, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(2688, 1, 3, 3)"},
        },
    ),
    (
        Reshape52,
        [((1, 1792, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 1792, 1, 1)"},
        },
    ),
    (
        Reshape53,
        [((56, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(56, 1, 3, 3)"},
        },
    ),
    (
        Reshape54,
        [((32, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                "pt_mobilenetv1_google_mobilenet_v1_1_0_224_img_cls_hf",
                "pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                "pt_mobilenetv1_basic_img_cls_torchvision",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(32, 1, 3, 3)"},
        },
    ),
    (
        Reshape55,
        [((240, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(240, 1, 3, 3)"},
        },
    ),
    (
        Reshape56,
        [((240, 1, 5, 5), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(240, 1, 5, 5)"},
        },
    ),
    (
        Reshape57,
        [((432, 1, 5, 5), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b6_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(432, 1, 5, 5)"},
        },
    ),
    (
        Reshape58,
        [((432, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(432, 1, 3, 3)"},
        },
    ),
    (
        Reshape59,
        [((864, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b6_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(864, 1, 3, 3)"},
        },
    ),
    (
        Reshape60,
        [((864, 1, 5, 5), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b6_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(864, 1, 5, 5)"},
        },
    ),
    (
        Reshape61,
        [((1200, 1, 5, 5), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b6_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1200, 1, 5, 5)"},
        },
    ),
    (
        Reshape62,
        [((2064, 1, 5, 5), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b6_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(2064, 1, 5, 5)"},
        },
    ),
    (
        Reshape63,
        [((2064, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b6_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(2064, 1, 3, 3)"},
        },
    ),
    (
        Reshape64,
        [((3456, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b6_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(3456, 1, 3, 3)"},
        },
    ),
    (
        Reshape65,
        [((1, 2304, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b6_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 2304, 1, 1)"},
        },
    ),
    (
        Reshape8,
        [((1, 2048, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_resnet_resnet152_img_cls_torchvision",
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_mobilenetv3_ssd_resnet101_img_cls_torchvision",
                "pt_resnext_resnext101_32x8d_img_cls_torchhub",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet152_img_cls_torchvision",
                "pt_resnet_50_img_cls_timm",
                "pt_xception_xception_img_cls_timm",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 2048, 1, 1)"},
        },
    ),
    (
        Reshape9,
        [((1, 2048, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_resnext_resnext101_64x4d_img_cls_osmr",
                "pt_resnext_resnext50_32x4d_img_cls_osmr",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 2048)"},
        },
    ),
    (
        Reshape66,
        [((1, 4, 2048), torch.float32)],
        {
            "model_names": ["pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(4, 2048)"},
        },
    ),
    (
        Reshape67,
        [((4, 2048), torch.float32)],
        {
            "model_names": ["pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 4, 32, 64)"},
        },
    ),
    (
        Reshape68,
        [((4, 2048), torch.float32)],
        {
            "model_names": ["pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 4, 2048)"},
        },
    ),
    (
        Reshape69,
        [((1, 32, 4, 64), torch.float32)],
        {
            "model_names": ["pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(32, 4, 64)"},
        },
    ),
    (
        Reshape70,
        [((4, 512), torch.float32)],
        {
            "model_names": ["pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 4, 8, 64)"},
        },
    ),
    (
        Reshape69,
        [((1, 8, 4, 4, 64), torch.float32)],
        {
            "model_names": ["pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(32, 4, 64)"},
        },
    ),
    (
        Reshape71,
        [((32, 4, 4), torch.float32)],
        {
            "model_names": ["pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 32, 4, 4)"},
        },
    ),
    (
        Reshape72,
        [((1, 32, 4, 4), torch.float32)],
        {
            "model_names": ["pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(32, 4, 4)"},
        },
    ),
    (
        Reshape73,
        [((32, 4, 64), torch.float32)],
        {
            "model_names": ["pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 32, 4, 64)"},
        },
    ),
    (
        Reshape66,
        [((1, 4, 32, 64), torch.float32)],
        {
            "model_names": ["pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(4, 2048)"},
        },
    ),
    (
        Reshape74,
        [((4, 8192), torch.float32)],
        {
            "model_names": ["pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 4, 8192)"},
        },
    ),
    (
        Reshape75,
        [((1, 1, 2), torch.float32)],
        {
            "model_names": [
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf",
                "pt_phi_1_5_microsoft_phi_1_5_seq_cls_hf",
                "pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 2)"},
        },
    ),
    (
        Reshape76,
        [((1, 3, 256, 256), torch.bfloat16)],
        {
            "model_names": ["pt_mlp_mixer_base_img_cls_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 3, 16, 16, 16, 16)"},
        },
    ),
    (
        Reshape77,
        [((1, 16, 16, 16, 16, 3), torch.bfloat16)],
        {
            "model_names": ["pt_mlp_mixer_base_img_cls_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(256, 768)"},
        },
    ),
    (
        Reshape78,
        [((256, 512), torch.bfloat16)],
        {
            "model_names": [
                "pt_mlp_mixer_base_img_cls_github",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 256, 512)"},
        },
    ),
    (
        Reshape79,
        [((1, 256, 512), torch.bfloat16)],
        {
            "model_names": ["pt_mlp_mixer_base_img_cls_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 256, 512, 1)"},
        },
    ),
    (
        Reshape80,
        [((1, 256, 512), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(256, 512)"},
        },
    ),
    (
        Reshape81,
        [((1, 256, 512), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 256, 8, 64)"},
        },
    ),
    (
        Reshape78,
        [((1, 256, 512), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b1_img_cls_hf", "pt_segformer_nvidia_mit_b3_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 256, 512)"},
        },
    ),
    (
        Reshape82,
        [((1, 256, 512), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 16, 16, 512)"},
        },
    ),
    (
        Reshape83,
        [((1024, 256, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mlp_mixer_base_img_cls_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1024, 256, 1, 1)"},
        },
    ),
    (
        Reshape84,
        [((1, 1024, 512, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mlp_mixer_base_img_cls_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 1024, 512)"},
        },
    ),
    (
        Reshape85,
        [((1, 1024, 512), torch.bfloat16)],
        {
            "model_names": ["pt_mlp_mixer_base_img_cls_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 1024, 512, 1)"},
        },
    ),
    (
        Reshape86,
        [((256, 1024, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mlp_mixer_base_img_cls_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(256, 1024, 1, 1)"},
        },
    ),
    (
        Reshape78,
        [((1, 256, 512, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mlp_mixer_base_img_cls_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 256, 512)"},
        },
    ),
    (
        Reshape87,
        [((1, 512, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_mlp_mixer_mixer_s16_224_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 512, 196, 1)"},
        },
    ),
    (
        Reshape88,
        [((64, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv1_google_mobilenet_v1_1_0_224_img_cls_hf",
                "pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv1_basic_img_cls_torchvision",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_xception_xception_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(64, 1, 3, 3)"},
        },
    ),
    (
        Reshape89,
        [((128, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv1_google_mobilenet_v1_1_0_224_img_cls_hf",
                "pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
                "pt_mobilenetv1_basic_img_cls_torchvision",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_xception_xception_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(128, 1, 3, 3)"},
        },
    ),
    (
        Reshape90,
        [((256, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv1_google_mobilenet_v1_1_0_224_img_cls_hf",
                "pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
                "pt_mobilenetv1_basic_img_cls_torchvision",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_xception_xception_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(256, 1, 3, 3)"},
        },
    ),
    (
        Reshape91,
        [((512, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv1_google_mobilenet_v1_1_0_224_img_cls_hf",
                "pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
                "pt_mobilenetv1_basic_img_cls_torchvision",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(512, 1, 3, 3)"},
        },
    ),
    (
        Reshape92,
        [((1024, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv1_google_mobilenet_v1_1_0_224_img_cls_hf",
                "pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
                "pt_mobilenetv1_basic_img_cls_torchvision",
                "pt_xception_xception_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1024, 1, 3, 3)"},
        },
    ),
    (
        Reshape93,
        [((1, 1024, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv1_google_mobilenet_v1_1_0_224_img_cls_hf",
                "pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 1024, 1, 1)"},
        },
    ),
    (
        Reshape94,
        [((1, 1024, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_vovnet_vovnet39_img_cls_osmr",
                "pt_mobilenetv1_basic_img_cls_torchvision",
                "pt_vovnet_v1_vovnet39_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 1024)"},
        },
    ),
    (
        Reshape95,
        [((1, 512, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_ssd_resnet34_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 512, 1, 1)"},
        },
    ),
    (
        Reshape96,
        [((1, 512, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_vovnet_vovnet27s_img_cls_osmr"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 512)"},
        },
    ),
    (
        Reshape97,
        [((1, 256, 2048), torch.float32)],
        {"model_names": ["pt_phi1_microsoft_phi_1_clm_hf"], "pcc": 0.99, "args": {"shape": "(256, 2048)"}},
    ),
    (
        Reshape98,
        [((1, 256, 2048), torch.float32)],
        {"model_names": ["pt_phi1_microsoft_phi_1_clm_hf"], "pcc": 0.99, "args": {"shape": "(1, 256, 32, 64)"}},
    ),
    (
        Reshape99,
        [((256, 2048), torch.float32)],
        {"model_names": ["pt_phi1_microsoft_phi_1_clm_hf"], "pcc": 0.99, "args": {"shape": "(1, 256, 2048)"}},
    ),
    (
        Reshape100,
        [((1, 32, 256, 64), torch.float32)],
        {"model_names": ["pt_phi1_microsoft_phi_1_clm_hf"], "pcc": 0.99, "args": {"shape": "(32, 256, 64)"}},
    ),
    (
        Reshape101,
        [((32, 256, 256), torch.float32)],
        {"model_names": ["pt_phi1_microsoft_phi_1_clm_hf"], "pcc": 0.99, "args": {"shape": "(1, 32, 256, 256)"}},
    ),
    (
        Reshape102,
        [((1, 32, 256, 256), torch.float32)],
        {"model_names": ["pt_phi1_microsoft_phi_1_clm_hf"], "pcc": 0.99, "args": {"shape": "(32, 256, 256)"}},
    ),
    (
        Reshape103,
        [((32, 256, 64), torch.float32)],
        {"model_names": ["pt_phi1_microsoft_phi_1_clm_hf"], "pcc": 0.99, "args": {"shape": "(1, 32, 256, 64)"}},
    ),
    (
        Reshape97,
        [((1, 256, 32, 64), torch.float32)],
        {"model_names": ["pt_phi1_microsoft_phi_1_clm_hf"], "pcc": 0.99, "args": {"shape": "(256, 2048)"}},
    ),
    (
        Reshape104,
        [((256, 8192), torch.float32)],
        {"model_names": ["pt_phi1_microsoft_phi_1_clm_hf"], "pcc": 0.99, "args": {"shape": "(1, 256, 8192)"}},
    ),
    (
        Reshape105,
        [((1, 5, 2048), torch.float32)],
        {"model_names": ["pt_phi_1_5_microsoft_phi_1_5_seq_cls_hf"], "pcc": 0.99, "args": {"shape": "(5, 2048)"}},
    ),
    (
        Reshape106,
        [((1, 5, 2048), torch.float32)],
        {"model_names": ["pt_phi_1_5_microsoft_phi_1_5_seq_cls_hf"], "pcc": 0.99, "args": {"shape": "(1, 5, 32, 64)"}},
    ),
    (
        Reshape107,
        [((5, 2048), torch.float32)],
        {"model_names": ["pt_phi_1_5_microsoft_phi_1_5_seq_cls_hf"], "pcc": 0.99, "args": {"shape": "(1, 5, 2048)"}},
    ),
    (
        Reshape108,
        [((1, 32, 5, 64), torch.float32)],
        {"model_names": ["pt_phi_1_5_microsoft_phi_1_5_seq_cls_hf"], "pcc": 0.99, "args": {"shape": "(32, 5, 64)"}},
    ),
    (
        Reshape109,
        [((32, 5, 5), torch.float32)],
        {"model_names": ["pt_phi_1_5_microsoft_phi_1_5_seq_cls_hf"], "pcc": 0.99, "args": {"shape": "(1, 32, 5, 5)"}},
    ),
    (
        Reshape110,
        [((1, 32, 5, 5), torch.float32)],
        {"model_names": ["pt_phi_1_5_microsoft_phi_1_5_seq_cls_hf"], "pcc": 0.99, "args": {"shape": "(32, 5, 5)"}},
    ),
    (
        Reshape111,
        [((32, 5, 64), torch.float32)],
        {"model_names": ["pt_phi_1_5_microsoft_phi_1_5_seq_cls_hf"], "pcc": 0.99, "args": {"shape": "(1, 32, 5, 64)"}},
    ),
    (
        Reshape105,
        [((1, 5, 32, 64), torch.float32)],
        {"model_names": ["pt_phi_1_5_microsoft_phi_1_5_seq_cls_hf"], "pcc": 0.99, "args": {"shape": "(5, 2048)"}},
    ),
    (
        Reshape112,
        [((5, 8192), torch.float32)],
        {"model_names": ["pt_phi_1_5_microsoft_phi_1_5_seq_cls_hf"], "pcc": 0.99, "args": {"shape": "(1, 5, 8192)"}},
    ),
    (
        Reshape113,
        [((1, 7392, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_128gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 7392, 1, 1)"},
        },
    ),
    (
        Reshape114,
        [((1, 2016, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_8gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 2016, 1, 1)"},
        },
    ),
    (
        Reshape115,
        [((1, 32, 128, 128), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 32, 16384, 1)"},
        },
    ),
    (
        Reshape116,
        [((1, 16384, 32), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 16384, 1, 32)"},
        },
    ),
    (
        Reshape117,
        [((1, 16384, 32), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 128, 128, 32)"},
        },
    ),
    (
        Reshape118,
        [((1, 32, 16384), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 32, 128, 128)"},
        },
    ),
    (
        Reshape119,
        [((1, 32, 16, 16), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 32, 256)"},
        },
    ),
    (
        Reshape120,
        [((1, 256, 32), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(256, 32)"},
        },
    ),
    (
        Reshape121,
        [((1, 256, 32), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 256, 1, 32)"},
        },
    ),
    (
        Reshape122,
        [((256, 32), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 256, 32)"},
        },
    ),
    (
        Reshape123,
        [((1, 16384, 256), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 1, 16384, 256)"},
        },
    ),
    (
        Reshape124,
        [((1, 1, 16384, 256), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 16384, 256)"},
        },
    ),
    (
        Reshape119,
        [((1, 1, 32, 256), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 32, 256)"},
        },
    ),
    (
        Reshape125,
        [((1, 128, 16384), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 128, 128, 128)"},
        },
    ),
    (
        Reshape126,
        [((1, 128, 128, 128), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 128, 16384, 1)"},
        },
    ),
    (
        Reshape127,
        [((1, 64, 64, 64), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 64, 4096, 1)"},
        },
    ),
    (
        Reshape128,
        [((1, 4096, 64), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 4096, 2, 32)"},
        },
    ),
    (
        Reshape129,
        [((1, 4096, 64), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 64, 64, 64)"},
        },
    ),
    (
        Reshape130,
        [((1, 2, 4096, 32), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(2, 4096, 32)"},
        },
    ),
    (
        Reshape129,
        [((1, 64, 4096), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 64, 64, 64)"},
        },
    ),
    (
        Reshape131,
        [((1, 64, 16, 16), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 64, 256)"},
        },
    ),
    (
        Reshape132,
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
            "args": {"shape": "(256, 64)"},
        },
    ),
    (
        Reshape133,
        [((1, 256, 64), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 256, 2, 32)"},
        },
    ),
    (
        Reshape134,
        [((1, 256, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 256, 1, 64)"},
        },
    ),
    (
        Reshape135,
        [((256, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 256, 64)"},
        },
    ),
    (
        Reshape136,
        [((1, 2, 256, 32), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(2, 256, 32)"},
        },
    ),
    (
        Reshape137,
        [((2, 4096, 256), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 2, 4096, 256)"},
        },
    ),
    (
        Reshape138,
        [((1, 2, 4096, 256), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(2, 4096, 256)"},
        },
    ),
    (
        Reshape139,
        [((1, 2, 32, 256), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(2, 32, 256)"},
        },
    ),
    (
        Reshape140,
        [((2, 4096, 32), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 2, 4096, 32)"},
        },
    ),
    (
        Reshape141,
        [((1, 4096, 2, 32), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(4096, 64)"},
        },
    ),
    (
        Reshape142,
        [((4096, 64), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 4096, 64)"},
        },
    ),
    (
        Reshape143,
        [((1, 256, 4096), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 256, 64, 64)"},
        },
    ),
    (
        Reshape144,
        [((1, 256, 64, 64), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 256, 4096, 1)"},
        },
    ),
    (
        Reshape145,
        [((1, 160, 32, 32), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 160, 1024, 1)"},
        },
    ),
    (
        Reshape146,
        [((1, 1024, 160), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 1024, 5, 32)"},
        },
    ),
    (
        Reshape147,
        [((1, 1024, 160), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 32, 32, 160)"},
        },
    ),
    (
        Reshape148,
        [((1, 5, 1024, 32), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(5, 1024, 32)"},
        },
    ),
    (
        Reshape149,
        [((1, 160, 1024), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 160, 32, 32)"},
        },
    ),
    (
        Reshape150,
        [((1, 160, 16, 16), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 160, 256)"},
        },
    ),
    (
        Reshape151,
        [((1, 256, 160), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(256, 160)"},
        },
    ),
    (
        Reshape152,
        [((1, 256, 160), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 256, 5, 32)"},
        },
    ),
    (
        Reshape153,
        [((256, 160), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 256, 160)"},
        },
    ),
    (
        Reshape154,
        [((1, 5, 256, 32), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(5, 256, 32)"},
        },
    ),
    (
        Reshape155,
        [((5, 1024, 256), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 5, 1024, 256)"},
        },
    ),
    (
        Reshape156,
        [((1, 5, 1024, 256), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(5, 1024, 256)"},
        },
    ),
    (
        Reshape157,
        [((1, 5, 32, 256), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(5, 32, 256)"},
        },
    ),
    (
        Reshape158,
        [((5, 1024, 32), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 5, 1024, 32)"},
        },
    ),
    (
        Reshape159,
        [((1, 1024, 5, 32), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1024, 160)"},
        },
    ),
    (
        Reshape160,
        [((1024, 160), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 1024, 160)"},
        },
    ),
    (
        Reshape161,
        [((1, 640, 1024), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 640, 32, 32)"},
        },
    ),
    (
        Reshape162,
        [((640, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(640, 1, 3, 3)"},
        },
    ),
    (
        Reshape163,
        [((1, 640, 32, 32), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 640, 1024, 1)"},
        },
    ),
    (
        Reshape164,
        [((1, 256, 16, 16), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 256, 256, 1)"},
        },
    ),
    (
        Reshape165,
        [((1, 256, 256), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(256, 256)"},
        },
    ),
    (
        Reshape166,
        [((1, 256, 256), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 256, 8, 32)"},
        },
    ),
    (
        Reshape167,
        [((1, 256, 256), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 256, 256)"},
        },
    ),
    (
        Reshape167,
        [((256, 256), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 256, 256)"},
        },
    ),
    (
        Reshape168,
        [((1, 8, 256, 32), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(8, 256, 32)"},
        },
    ),
    (
        Reshape169,
        [((8, 256, 256), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 8, 256, 256)"},
        },
    ),
    (
        Reshape170,
        [((1, 8, 256, 256), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(8, 256, 256)"},
        },
    ),
    (
        Reshape171,
        [((1, 8, 32, 256), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(8, 32, 256)"},
        },
    ),
    (
        Reshape172,
        [((8, 256, 32), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 8, 256, 32)"},
        },
    ),
    (
        Reshape165,
        [((1, 256, 8, 32), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(256, 256)"},
        },
    ),
    (
        Reshape173,
        [((1, 1024, 256), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 1024, 16, 16)"},
        },
    ),
    (
        Reshape174,
        [((1, 1024, 16, 16), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 1024, 256, 1)"},
        },
    ),
    (
        Reshape175,
        [((1, 56, 56, 96), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 8, 7, 8, 7, 96)"},
        },
    ),
    (
        Reshape176,
        [((1, 56, 56, 96), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(3136, 96)"},
        },
    ),
    (
        Reshape176,
        [((1, 8, 8, 7, 7, 96), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(3136, 96)"},
        },
    ),
    (
        Reshape177,
        [((3136, 288), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(64, 49, 288)"},
        },
    ),
    (
        Reshape178,
        [((64, 49, 288), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(64, 49, 3, 3, 32)"},
        },
    ),
    (
        Reshape179,
        [((1, 64, 3, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(64, 3, 49, 32)"},
        },
    ),
    (
        Reshape180,
        [((1, 64, 3, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(192, 49, 32)"},
        },
    ),
    (
        Reshape180,
        [((64, 3, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(192, 49, 32)"},
        },
    ),
    (
        Reshape181,
        [((192, 49, 49), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(64, 3, 49, 49)"},
        },
    ),
    (
        Reshape182,
        [((2401, 3), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(49, 49, 3)"},
        },
    ),
    (
        Reshape183,
        [((64, 3, 49, 49), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(192, 49, 49)"},
        },
    ),
    (
        Reshape184,
        [((64, 3, 49, 49), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 64, 3, 49, 49)"},
        },
    ),
    (
        Reshape185,
        [((64, 3, 32, 49), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(192, 32, 49)"},
        },
    ),
    (
        Reshape179,
        [((192, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(64, 3, 49, 32)"},
        },
    ),
    (
        Reshape176,
        [((64, 49, 3, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(3136, 96)"},
        },
    ),
    (
        Reshape186,
        [((3136, 96), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(64, 49, 96)"},
        },
    ),
    (
        Reshape187,
        [((3136, 96), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 56, 56, 96)"},
        },
    ),
    (
        Reshape188,
        [((64, 49, 96), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 8, 8, 7, 7, 96)"},
        },
    ),
    (
        Reshape187,
        [((1, 8, 7, 8, 7, 96), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 56, 56, 96)"},
        },
    ),
    (
        Reshape189,
        [((3136, 384), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 56, 56, 384)"},
        },
    ),
    (
        Reshape190,
        [((1, 56, 56, 384), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(3136, 384)"},
        },
    ),
    (
        Reshape181,
        [((1, 64, 3, 49, 49), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(64, 3, 49, 49)"},
        },
    ),
    (
        Reshape191,
        [((1, 28, 28, 384), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(784, 384)"},
        },
    ),
    (
        Reshape192,
        [((784, 192), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 28, 28, 192)"},
        },
    ),
    (
        Reshape193,
        [((784, 192), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(16, 49, 192)"},
        },
    ),
    (
        Reshape194,
        [((1, 28, 28, 192), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 4, 7, 4, 7, 192)"},
        },
    ),
    (
        Reshape195,
        [((1, 28, 28, 192), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(784, 192)"},
        },
    ),
    (
        Reshape195,
        [((1, 4, 4, 7, 7, 192), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(784, 192)"},
        },
    ),
    (
        Reshape196,
        [((784, 576), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(16, 49, 576)"},
        },
    ),
    (
        Reshape197,
        [((16, 49, 576), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(16, 49, 3, 6, 32)"},
        },
    ),
    (
        Reshape198,
        [((1, 16, 6, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(16, 6, 49, 32)"},
        },
    ),
    (
        Reshape199,
        [((1, 16, 6, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(96, 49, 32)"},
        },
    ),
    (
        Reshape199,
        [((16, 6, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(96, 49, 32)"},
        },
    ),
    (
        Reshape200,
        [((96, 49, 49), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(16, 6, 49, 49)"},
        },
    ),
    (
        Reshape201,
        [((2401, 6), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(49, 49, 6)"},
        },
    ),
    (
        Reshape202,
        [((16, 6, 49, 49), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(96, 49, 49)"},
        },
    ),
    (
        Reshape203,
        [((16, 6, 49, 49), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 16, 6, 49, 49)"},
        },
    ),
    (
        Reshape204,
        [((16, 6, 32, 49), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(96, 32, 49)"},
        },
    ),
    (
        Reshape198,
        [((96, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(16, 6, 49, 32)"},
        },
    ),
    (
        Reshape195,
        [((16, 49, 6, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(784, 192)"},
        },
    ),
    (
        Reshape205,
        [((16, 49, 192), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 4, 4, 7, 7, 192)"},
        },
    ),
    (
        Reshape192,
        [((1, 4, 7, 4, 7, 192), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 28, 28, 192)"},
        },
    ),
    (
        Reshape206,
        [((784, 768), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 28, 28, 768)"},
        },
    ),
    (
        Reshape207,
        [((1, 28, 28, 768), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(784, 768)"},
        },
    ),
    (
        Reshape200,
        [((1, 16, 6, 49, 49), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(16, 6, 49, 49)"},
        },
    ),
    (
        Reshape208,
        [((1, 14, 14, 768), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(196, 768)"},
        },
    ),
    (
        Reshape209,
        [((196, 384), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 14, 14, 384)"},
        },
    ),
    (
        Reshape210,
        [((196, 384), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(4, 49, 384)"},
        },
    ),
    (
        Reshape211,
        [((1, 14, 14, 384), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 2, 7, 2, 7, 384)"},
        },
    ),
    (
        Reshape212,
        [((1, 14, 14, 384), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(196, 384)"},
        },
    ),
    (
        Reshape212,
        [((1, 2, 2, 7, 7, 384), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(196, 384)"},
        },
    ),
    (
        Reshape213,
        [((196, 1152), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(4, 49, 1152)"},
        },
    ),
    (
        Reshape214,
        [((4, 49, 1152), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(4, 49, 3, 12, 32)"},
        },
    ),
    (
        Reshape215,
        [((1, 4, 12, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(4, 12, 49, 32)"},
        },
    ),
    (
        Reshape216,
        [((1, 4, 12, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(48, 49, 32)"},
        },
    ),
    (
        Reshape216,
        [((4, 12, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(48, 49, 32)"},
        },
    ),
    (
        Reshape217,
        [((48, 49, 49), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(4, 12, 49, 49)"},
        },
    ),
    (
        Reshape218,
        [((2401, 12), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(49, 49, 12)"},
        },
    ),
    (
        Reshape219,
        [((4, 12, 49, 49), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(48, 49, 49)"},
        },
    ),
    (
        Reshape220,
        [((4, 12, 49, 49), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 4, 12, 49, 49)"},
        },
    ),
    (
        Reshape221,
        [((4, 12, 32, 49), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(48, 32, 49)"},
        },
    ),
    (
        Reshape215,
        [((48, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(4, 12, 49, 32)"},
        },
    ),
    (
        Reshape212,
        [((4, 49, 12, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(196, 384)"},
        },
    ),
    (
        Reshape222,
        [((4, 49, 384), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 2, 2, 7, 7, 384)"},
        },
    ),
    (
        Reshape209,
        [((1, 2, 7, 2, 7, 384), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 14, 14, 384)"},
        },
    ),
    (
        Reshape223,
        [((196, 1536), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 14, 14, 1536)"},
        },
    ),
    (
        Reshape224,
        [((1, 14, 14, 1536), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(196, 1536)"},
        },
    ),
    (
        Reshape217,
        [((1, 4, 12, 49, 49), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(4, 12, 49, 49)"},
        },
    ),
    (
        Reshape225,
        [((1, 7, 7, 1536), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(49, 1536)"},
        },
    ),
    (
        Reshape226,
        [((49, 768), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 7, 7, 768)"},
        },
    ),
    (
        Reshape227,
        [((49, 768), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 49, 768)"},
        },
    ),
    (
        Reshape228,
        [((1, 7, 7, 768), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 1, 7, 1, 7, 768)"},
        },
    ),
    (
        Reshape229,
        [((1, 7, 7, 768), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(49, 768)"},
        },
    ),
    (
        Reshape229,
        [((1, 1, 1, 7, 7, 768), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(49, 768)"},
        },
    ),
    (
        Reshape230,
        [((49, 2304), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 49, 2304)"},
        },
    ),
    (
        Reshape231,
        [((1, 49, 2304), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 49, 3, 24, 32)"},
        },
    ),
    (
        Reshape232,
        [((1, 1, 24, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 24, 49, 32)"},
        },
    ),
    (
        Reshape233,
        [((1, 1, 24, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(24, 49, 32)"},
        },
    ),
    (
        Reshape233,
        [((1, 24, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(24, 49, 32)"},
        },
    ),
    (
        Reshape234,
        [((24, 49, 49), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 24, 49, 49)"},
        },
    ),
    (
        Reshape235,
        [((2401, 24), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(49, 49, 24)"},
        },
    ),
    (
        Reshape236,
        [((1, 24, 49, 49), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(24, 49, 49)"},
        },
    ),
    (
        Reshape237,
        [((1, 24, 32, 49), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(24, 32, 49)"},
        },
    ),
    (
        Reshape232,
        [((24, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 24, 49, 32)"},
        },
    ),
    (
        Reshape229,
        [((1, 49, 24, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(49, 768)"},
        },
    ),
    (
        Reshape238,
        [((1, 49, 768), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 1, 1, 7, 7, 768)"},
        },
    ),
    (
        Reshape226,
        [((1, 1, 7, 1, 7, 768), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 7, 7, 768)"},
        },
    ),
    (
        Reshape239,
        [((49, 3072), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 7, 7, 3072)"},
        },
    ),
    (
        Reshape240,
        [((1, 7, 7, 3072), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(49, 3072)"},
        },
    ),
    (
        Reshape241,
        [((1, 768, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_mobilenetv1_google_mobilenet_v1_0_75_192_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 768, 1, 1)"},
        },
    ),
    (
        Reshape242,
        [((1, 512, 7, 7), torch.bfloat16)],
        {
            "model_names": [
                "pt_vgg_19_obj_det_hf",
                "pt_vgg_vgg16_img_cls_torchvision",
                "pt_vgg_vgg11_img_cls_torchvision",
                "pt_vgg_vgg13_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 25088, 1, 1)"},
        },
    ),
    (
        Reshape243,
        [((1, 512, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_vgg_vgg16_obj_det_osmr", "pt_vgg_bn_vgg19b_obj_det_osmr"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 25088)"},
        },
    ),
    (
        Reshape244,
        [((1, 512, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_mlp_mixer_mixer_s32_224_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 512, 49, 1)"},
        },
    ),
    (
        Reshape245,
        [((1, 768, 14, 14), torch.bfloat16)],
        {
            "model_names": [
                "pt_vit_google_vit_base_patch16_224_img_cls_hf",
                "pt_mlp_mixer_mixer_b16_224_in21k_img_cls_timm",
                "pt_beit_microsoft_beit_base_patch16_224_img_cls_hf",
                "pt_mlp_mixer_mixer_b16_224_miil_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 768, 196, 1)"},
        },
    ),
    (
        Reshape246,
        [((1, 768, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_b_16_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 768, 196)"},
        },
    ),
    (
        Reshape247,
        [((1, 197, 768), torch.bfloat16)],
        {
            "model_names": [
                "pt_vit_google_vit_base_patch16_224_img_cls_hf",
                "pt_beit_microsoft_beit_base_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(197, 768)"},
        },
    ),
    (
        Reshape248,
        [((1, 197, 768), torch.bfloat16)],
        {
            "model_names": [
                "pt_vit_google_vit_base_patch16_224_img_cls_hf",
                "pt_beit_microsoft_beit_base_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 197, 12, 64)"},
        },
    ),
    (
        Reshape249,
        [((197, 768), torch.bfloat16)],
        {
            "model_names": [
                "pt_vit_google_vit_base_patch16_224_img_cls_hf",
                "pt_beit_microsoft_beit_base_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 197, 768)"},
        },
    ),
    (
        Reshape248,
        [((197, 768), torch.bfloat16)],
        {
            "model_names": ["pt_beit_microsoft_beit_base_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 197, 12, 64)"},
        },
    ),
    (
        Reshape250,
        [((197, 768), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_b_16_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(197, 1, 768)"},
        },
    ),
    (
        Reshape251,
        [((1, 12, 197, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_vit_google_vit_base_patch16_224_img_cls_hf",
                "pt_beit_microsoft_beit_base_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(12, 197, 64)"},
        },
    ),
    (
        Reshape252,
        [((12, 197, 197), torch.bfloat16)],
        {
            "model_names": [
                "pt_vit_google_vit_base_patch16_224_img_cls_hf",
                "pt_beit_microsoft_beit_base_patch16_224_img_cls_hf",
                "pt_vit_vit_b_16_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 12, 197, 197)"},
        },
    ),
    (
        Reshape253,
        [((1, 12, 197, 197), torch.bfloat16)],
        {
            "model_names": [
                "pt_vit_google_vit_base_patch16_224_img_cls_hf",
                "pt_beit_microsoft_beit_base_patch16_224_img_cls_hf",
                "pt_vit_vit_b_16_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(12, 197, 197)"},
        },
    ),
    (
        Reshape254,
        [((12, 197, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_vit_google_vit_base_patch16_224_img_cls_hf",
                "pt_beit_microsoft_beit_base_patch16_224_img_cls_hf",
                "pt_vit_vit_b_16_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 12, 197, 64)"},
        },
    ),
    (
        Reshape251,
        [((12, 197, 64), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_b_16_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(12, 197, 64)"},
        },
    ),
    (
        Reshape247,
        [((1, 197, 12, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_vit_google_vit_base_patch16_224_img_cls_hf",
                "pt_beit_microsoft_beit_base_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(197, 768)"},
        },
    ),
    (
        Reshape27,
        [((1, 1, 768), torch.bfloat16)],
        {
            "model_names": ["pt_vit_google_vit_base_patch16_224_img_cls_hf", "pt_vit_vit_b_16_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 768)"},
        },
    ),
    (
        Reshape255,
        [((1, 1024, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_l_32_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 1024, 49)"},
        },
    ),
    (
        Reshape256,
        [((50, 1, 1024), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_l_32_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(50, 1024)"},
        },
    ),
    (
        Reshape257,
        [((50, 3072), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_l_32_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(50, 1, 3072)"},
        },
    ),
    (
        Reshape258,
        [((50, 1, 3072), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_l_32_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(50, 1, 3, 1024)"},
        },
    ),
    (
        Reshape259,
        [((1, 50, 1, 1024), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_l_32_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(50, 16, 64)"},
        },
    ),
    (
        Reshape260,
        [((16, 50, 64), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_l_32_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(16, 50, 64)"},
        },
    ),
    (
        Reshape261,
        [((16, 50, 64), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_l_32_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 16, 50, 64)"},
        },
    ),
    (
        Reshape262,
        [((16, 50, 50), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_l_32_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 16, 50, 50)"},
        },
    ),
    (
        Reshape263,
        [((1, 16, 50, 50), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_l_32_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(16, 50, 50)"},
        },
    ),
    (
        Reshape256,
        [((50, 1, 16, 64), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_l_32_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(50, 1024)"},
        },
    ),
    (
        Reshape264,
        [((50, 1024), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_l_32_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(50, 1, 1024)"},
        },
    ),
    (
        Reshape94,
        [((1, 1, 1024), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_l_32_img_cls_torchvision", "pt_vit_vit_l_16_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 1024)"},
        },
    ),
    (
        Reshape265,
        [((728, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": [
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
                "pt_xception_xception_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(728, 1, 3, 3)"},
        },
    ),
    (
        Reshape266,
        [((1536, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": [
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
                "pt_xception_xception_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1536, 1, 3, 3)"},
        },
    ),
    (
        Reshape267,
        [((1, 3, 85, 80, 80), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5l_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_640x640",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 255, 80, 80)"},
        },
    ),
    (
        Reshape268,
        [((1, 255, 80, 80), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5l_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_640x640",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 1, 255, 6400)"},
        },
    ),
    (
        Reshape269,
        [((1, 1, 255, 6400), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5l_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_640x640",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 3, 85, 6400)"},
        },
    ),
    (
        Reshape270,
        [((1, 3, 6400, 85), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5l_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_640x640",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 19200, 85)"},
        },
    ),
    (
        Reshape271,
        [((1, 3, 85, 40, 40), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5l_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_320x320",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 255, 40, 40)"},
        },
    ),
    (
        Reshape272,
        [((1, 255, 40, 40), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5l_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_320x320",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 1, 255, 1600)"},
        },
    ),
    (
        Reshape273,
        [((1, 1, 255, 1600), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5l_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_320x320",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 3, 85, 1600)"},
        },
    ),
    (
        Reshape274,
        [((1, 3, 1600, 85), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5l_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_320x320",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 4800, 85)"},
        },
    ),
    (
        Reshape275,
        [((1, 3, 85, 20, 20), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5l_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_320x320",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 255, 20, 20)"},
        },
    ),
    (
        Reshape276,
        [((1, 255, 20, 20), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5l_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_320x320",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 1, 255, 400)"},
        },
    ),
    (
        Reshape277,
        [((1, 1, 255, 400), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5l_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_320x320",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 3, 85, 400)"},
        },
    ),
    (
        Reshape278,
        [((1, 3, 400, 85), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5l_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_320x320",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 1200, 85)"},
        },
    ),
    (
        Reshape279,
        [((1, 3, 85, 60, 60), torch.float32)],
        {
            "model_names": ["pt_yolo_v5_yolov5n_img_cls_torchhub_480x480"],
            "pcc": 0.99,
            "args": {"shape": "(1, 255, 60, 60)"},
        },
    ),
    (
        Reshape280,
        [((1, 255, 60, 60), torch.float32)],
        {
            "model_names": ["pt_yolo_v5_yolov5n_img_cls_torchhub_480x480"],
            "pcc": 0.99,
            "args": {"shape": "(1, 1, 255, 3600)"},
        },
    ),
    (
        Reshape281,
        [((1, 1, 255, 3600), torch.float32)],
        {
            "model_names": ["pt_yolo_v5_yolov5n_img_cls_torchhub_480x480"],
            "pcc": 0.99,
            "args": {"shape": "(1, 3, 85, 3600)"},
        },
    ),
    (
        Reshape282,
        [((1, 3, 3600, 85), torch.float32)],
        {
            "model_names": ["pt_yolo_v5_yolov5n_img_cls_torchhub_480x480"],
            "pcc": 0.99,
            "args": {"shape": "(1, 10800, 85)"},
        },
    ),
    (
        Reshape283,
        [((1, 3, 85, 30, 30), torch.float32)],
        {
            "model_names": ["pt_yolo_v5_yolov5n_img_cls_torchhub_480x480"],
            "pcc": 0.99,
            "args": {"shape": "(1, 255, 30, 30)"},
        },
    ),
    (
        Reshape284,
        [((1, 255, 30, 30), torch.float32)],
        {
            "model_names": ["pt_yolo_v5_yolov5n_img_cls_torchhub_480x480"],
            "pcc": 0.99,
            "args": {"shape": "(1, 1, 255, 900)"},
        },
    ),
    (
        Reshape285,
        [((1, 1, 255, 900), torch.float32)],
        {
            "model_names": ["pt_yolo_v5_yolov5n_img_cls_torchhub_480x480"],
            "pcc": 0.99,
            "args": {"shape": "(1, 3, 85, 900)"},
        },
    ),
    (
        Reshape286,
        [((1, 3, 900, 85), torch.float32)],
        {
            "model_names": ["pt_yolo_v5_yolov5n_img_cls_torchhub_480x480"],
            "pcc": 0.99,
            "args": {"shape": "(1, 2700, 85)"},
        },
    ),
    (
        Reshape287,
        [((1, 3, 85, 15, 15), torch.float32)],
        {
            "model_names": ["pt_yolo_v5_yolov5n_img_cls_torchhub_480x480"],
            "pcc": 0.99,
            "args": {"shape": "(1, 255, 15, 15)"},
        },
    ),
    (
        Reshape288,
        [((1, 255, 15, 15), torch.float32)],
        {
            "model_names": ["pt_yolo_v5_yolov5n_img_cls_torchhub_480x480"],
            "pcc": 0.99,
            "args": {"shape": "(1, 1, 255, 225)"},
        },
    ),
    (
        Reshape289,
        [((1, 1, 255, 225), torch.float32)],
        {
            "model_names": ["pt_yolo_v5_yolov5n_img_cls_torchhub_480x480"],
            "pcc": 0.99,
            "args": {"shape": "(1, 3, 85, 225)"},
        },
    ),
    (
        Reshape290,
        [((1, 3, 225, 85), torch.float32)],
        {
            "model_names": ["pt_yolo_v5_yolov5n_img_cls_torchhub_480x480"],
            "pcc": 0.99,
            "args": {"shape": "(1, 675, 85)"},
        },
    ),
    (
        Reshape291,
        [((1, 144, 80, 80), torch.bfloat16)],
        {
            "model_names": ["pt_yolov8_yolov8x_obj_det_github", "pt_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 144, 6400)"},
        },
    ),
    (
        Reshape292,
        [((1, 144, 40, 40), torch.bfloat16)],
        {
            "model_names": ["pt_yolov8_yolov8x_obj_det_github", "pt_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 144, 1600)"},
        },
    ),
    (
        Reshape293,
        [((1, 144, 20, 20), torch.bfloat16)],
        {
            "model_names": ["pt_yolov8_yolov8x_obj_det_github", "pt_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 144, 400)"},
        },
    ),
    (
        Reshape294,
        [((1, 64, 8400), torch.bfloat16)],
        {
            "model_names": ["pt_yolov8_yolov8x_obj_det_github", "pt_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 4, 16, 8400)"},
        },
    ),
    (
        Reshape295,
        [((1, 1, 4, 8400), torch.bfloat16)],
        {
            "model_names": ["pt_yolov8_yolov8x_obj_det_github", "pt_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 4, 8400)"},
        },
    ),
    (
        Reshape296,
        [((1, 85, 80, 80), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolox_yolox_l_obj_det_torchhub",
                "pt_yolox_yolox_m_obj_det_torchhub",
                "pt_yolox_yolox_darknet_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 85, 6400, 1)"},
        },
    ),
    (
        Reshape297,
        [((1, 85, 40, 40), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolox_yolox_l_obj_det_torchhub",
                "pt_yolox_yolox_m_obj_det_torchhub",
                "pt_yolox_yolox_darknet_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 85, 1600, 1)"},
        },
    ),
    (
        Reshape298,
        [((1, 85, 20, 20), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolox_yolox_l_obj_det_torchhub",
                "pt_yolox_yolox_m_obj_det_torchhub",
                "pt_yolox_yolox_darknet_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 85, 400, 1)"},
        },
    ),
    (
        Reshape299,
        [((1, 64, 128, 128), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 64, 16384)"},
        },
    ),
    (
        Reshape300,
        [((1, 16384, 64), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 16384, 1, 64)"},
        },
    ),
    (
        Reshape301,
        [((1, 16384, 64), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 128, 128, 64)"},
        },
    ),
    (
        Reshape302,
        [((1, 64, 16384), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 64, 128, 128)"},
        },
    ),
    (
        Reshape131,
        [((1, 64, 16, 16), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 64, 256)"},
        },
    ),
    (
        Reshape132,
        [((1, 256, 64), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"shape": "(256, 64)"},
        },
    ),
    (
        Reshape134,
        [((1, 256, 64), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 256, 1, 64)"},
        },
    ),
    (
        Reshape135,
        [((256, 64), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 256, 64)"},
        },
    ),
    (
        Reshape131,
        [((1, 1, 64, 256), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 64, 256)"},
        },
    ),
    (
        Reshape123,
        [((1, 16384, 256), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 1, 16384, 256)"},
        },
    ),
    (
        Reshape124,
        [((1, 1, 16384, 256), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 16384, 256)"},
        },
    ),
    (
        Reshape303,
        [((1, 256, 16384), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 256, 128, 128)"},
        },
    ),
    (
        Reshape304,
        [((1, 256, 128, 128), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 256, 16384)"},
        },
    ),
    (
        Reshape305,
        [((1, 128, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 128, 4096)"},
        },
    ),
    (
        Reshape306,
        [((1, 4096, 128), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 4096, 2, 64)"},
        },
    ),
    (
        Reshape307,
        [((1, 4096, 128), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 64, 64, 128)"},
        },
    ),
    (
        Reshape308,
        [((1, 2, 4096, 64), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"shape": "(2, 4096, 64)"},
        },
    ),
    (
        Reshape309,
        [((1, 128, 4096), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 128, 64, 64)"},
        },
    ),
    (
        Reshape310,
        [((1, 128, 16, 16), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 128, 256)"},
        },
    ),
    (
        Reshape311,
        [((1, 256, 128), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"shape": "(256, 128)"},
        },
    ),
    (
        Reshape312,
        [((1, 256, 128), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 256, 2, 64)"},
        },
    ),
    (
        Reshape313,
        [((256, 128), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 256, 128)"},
        },
    ),
    (
        Reshape314,
        [((1, 2, 64, 256), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"shape": "(2, 64, 256)"},
        },
    ),
    (
        Reshape137,
        [((2, 4096, 256), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 2, 4096, 256)"},
        },
    ),
    (
        Reshape138,
        [((1, 2, 4096, 256), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"shape": "(2, 4096, 256)"},
        },
    ),
    (
        Reshape315,
        [((1, 2, 256, 64), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"shape": "(2, 256, 64)"},
        },
    ),
    (
        Reshape316,
        [((2, 4096, 64), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 2, 4096, 64)"},
        },
    ),
    (
        Reshape317,
        [((1, 4096, 2, 64), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"shape": "(4096, 128)"},
        },
    ),
    (
        Reshape318,
        [((4096, 128), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 4096, 128)"},
        },
    ),
    (
        Reshape319,
        [((1, 512, 4096), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 512, 64, 64)"},
        },
    ),
    (
        Reshape320,
        [((1, 512, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 512, 4096)"},
        },
    ),
    (
        Reshape321,
        [((1, 320, 32, 32), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 320, 1024)"},
        },
    ),
    (
        Reshape322,
        [((1, 1024, 320), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 1024, 5, 64)"},
        },
    ),
    (
        Reshape323,
        [((1, 1024, 320), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 32, 32, 320)"},
        },
    ),
    (
        Reshape324,
        [((1, 5, 1024, 64), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"shape": "(5, 1024, 64)"},
        },
    ),
    (
        Reshape325,
        [((1, 320, 1024), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 320, 32, 32)"},
        },
    ),
    (
        Reshape326,
        [((1, 320, 16, 16), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 320, 256)"},
        },
    ),
    (
        Reshape327,
        [((1, 256, 320), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"shape": "(256, 320)"},
        },
    ),
    (
        Reshape328,
        [((1, 256, 320), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 256, 5, 64)"},
        },
    ),
    (
        Reshape329,
        [((256, 320), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 256, 320)"},
        },
    ),
    (
        Reshape330,
        [((1, 5, 64, 256), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"shape": "(5, 64, 256)"},
        },
    ),
    (
        Reshape155,
        [((5, 1024, 256), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 5, 1024, 256)"},
        },
    ),
    (
        Reshape156,
        [((1, 5, 1024, 256), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"shape": "(5, 1024, 256)"},
        },
    ),
    (
        Reshape331,
        [((1, 5, 256, 64), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"shape": "(5, 256, 64)"},
        },
    ),
    (
        Reshape332,
        [((5, 1024, 64), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 5, 1024, 64)"},
        },
    ),
    (
        Reshape333,
        [((1, 1024, 5, 64), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1024, 320)"},
        },
    ),
    (
        Reshape334,
        [((1024, 320), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 1024, 320)"},
        },
    ),
    (
        Reshape335,
        [((1, 1280, 1024), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 1280, 32, 32)"},
        },
    ),
    (
        Reshape336,
        [((1, 1280, 32, 32), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 1280, 1024)"},
        },
    ),
    (
        Reshape337,
        [((1, 512, 16, 16), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 512, 256)"},
        },
    ),
    (
        Reshape80,
        [((1, 256, 512), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"shape": "(256, 512)"},
        },
    ),
    (
        Reshape81,
        [((1, 256, 512), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 256, 8, 64)"},
        },
    ),
    (
        Reshape82,
        [((1, 256, 512), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 16, 16, 512)"},
        },
    ),
    (
        Reshape78,
        [((256, 512), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 256, 512)"},
        },
    ),
    (
        Reshape338,
        [((1, 8, 256, 64), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"shape": "(8, 256, 64)"},
        },
    ),
    (
        Reshape339,
        [((1, 8, 64, 256), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"shape": "(8, 64, 256)"},
        },
    ),
    (
        Reshape169,
        [((8, 256, 256), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 8, 256, 256)"},
        },
    ),
    (
        Reshape170,
        [((1, 8, 256, 256), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"shape": "(8, 256, 256)"},
        },
    ),
    (
        Reshape340,
        [((8, 256, 64), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 8, 256, 64)"},
        },
    ),
    (
        Reshape80,
        [((1, 256, 8, 64), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"shape": "(256, 512)"},
        },
    ),
    (
        Reshape341,
        [((1, 2048, 256), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 2048, 16, 16)"},
        },
    ),
    (
        Reshape342,
        [((1, 2048, 16, 16), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 2048, 256)"},
        },
    ),
    (
        Reshape343,
        [((1, 768, 256), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 768, 16, 16)"},
        },
    ),
    (
        Reshape344,
        [((1, 768, 1024), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 768, 32, 32)"},
        },
    ),
    (
        Reshape345,
        [((1, 768, 4096), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 768, 64, 64)"},
        },
    ),
    (
        Reshape346,
        [((1, 768, 16384), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 768, 128, 128)"},
        },
    ),
    (
        Reshape347,
        [((1, 9, 768), torch.float32)],
        {
            "model_names": [
                "pd_roberta_rbt4_ch_seq_cls_padlenlp",
                "pd_bert_chinese_roberta_base_mlm_padlenlp",
                "pd_bert_bert_base_uncased_mlm_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"shape": "(9, 768)"},
        },
    ),
    (
        Reshape348,
        [((1, 9, 768), torch.float32)],
        {
            "model_names": [
                "pd_roberta_rbt4_ch_seq_cls_padlenlp",
                "pd_bert_chinese_roberta_base_mlm_padlenlp",
                "pd_bert_bert_base_uncased_mlm_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 9, 12, 64)"},
        },
    ),
    (
        Reshape349,
        [((9, 768), torch.float32)],
        {
            "model_names": [
                "pd_roberta_rbt4_ch_seq_cls_padlenlp",
                "pd_bert_chinese_roberta_base_mlm_padlenlp",
                "pd_bert_bert_base_uncased_mlm_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 9, 768)"},
        },
    ),
    (
        Reshape350,
        [((1, 12, 9, 64), torch.float32)],
        {
            "model_names": [
                "pd_roberta_rbt4_ch_seq_cls_padlenlp",
                "pd_bert_chinese_roberta_base_mlm_padlenlp",
                "pd_bert_bert_base_uncased_mlm_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"shape": "(12, 9, 64)"},
        },
    ),
    (
        Reshape351,
        [((1, 12, 64, 9), torch.float32)],
        {
            "model_names": [
                "pd_roberta_rbt4_ch_seq_cls_padlenlp",
                "pd_bert_chinese_roberta_base_mlm_padlenlp",
                "pd_bert_bert_base_uncased_mlm_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"shape": "(12, 64, 9)"},
        },
    ),
    (
        Reshape352,
        [((12, 9, 9), torch.float32)],
        {
            "model_names": [
                "pd_roberta_rbt4_ch_seq_cls_padlenlp",
                "pd_bert_chinese_roberta_base_mlm_padlenlp",
                "pd_bert_bert_base_uncased_mlm_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 12, 9, 9)"},
        },
    ),
    (
        Reshape353,
        [((1, 12, 9, 9), torch.float32)],
        {
            "model_names": [
                "pd_roberta_rbt4_ch_seq_cls_padlenlp",
                "pd_bert_chinese_roberta_base_mlm_padlenlp",
                "pd_bert_bert_base_uncased_mlm_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"shape": "(12, 9, 9)"},
        },
    ),
    (
        Reshape354,
        [((12, 9, 64), torch.float32)],
        {
            "model_names": [
                "pd_roberta_rbt4_ch_seq_cls_padlenlp",
                "pd_bert_chinese_roberta_base_mlm_padlenlp",
                "pd_bert_bert_base_uncased_mlm_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 12, 9, 64)"},
        },
    ),
    (
        Reshape347,
        [((1, 9, 12, 64), torch.float32)],
        {
            "model_names": [
                "pd_roberta_rbt4_ch_seq_cls_padlenlp",
                "pd_bert_chinese_roberta_base_mlm_padlenlp",
                "pd_bert_bert_base_uncased_mlm_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"shape": "(9, 768)"},
        },
    ),
    (
        Reshape355,
        [((1, 1024, 14, 14), torch.bfloat16)],
        {
            "model_names": [
                "pt_beit_microsoft_beit_large_patch16_224_img_cls_hf",
                "pt_mlp_mixer_mixer_l16_224_in21k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 1024, 196, 1)"},
        },
    ),
    (
        Reshape356,
        [((1, 1024, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_l_16_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 1024, 196)"},
        },
    ),
    (
        Reshape357,
        [((1, 197, 1024), torch.bfloat16)],
        {
            "model_names": ["pt_beit_microsoft_beit_large_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(197, 1024)"},
        },
    ),
    (
        Reshape358,
        [((1, 197, 1024), torch.bfloat16)],
        {
            "model_names": ["pt_beit_microsoft_beit_large_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 197, 16, 64)"},
        },
    ),
    (
        Reshape359,
        [((197, 1024), torch.bfloat16)],
        {
            "model_names": ["pt_beit_microsoft_beit_large_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 197, 1024)"},
        },
    ),
    (
        Reshape358,
        [((197, 1024), torch.bfloat16)],
        {
            "model_names": ["pt_beit_microsoft_beit_large_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 197, 16, 64)"},
        },
    ),
    (
        Reshape360,
        [((197, 1024), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_l_16_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(197, 1, 1024)"},
        },
    ),
    (
        Reshape361,
        [((1, 16, 197, 64), torch.bfloat16)],
        {
            "model_names": ["pt_beit_microsoft_beit_large_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(16, 197, 64)"},
        },
    ),
    (
        Reshape362,
        [((16, 197, 197), torch.bfloat16)],
        {
            "model_names": [
                "pt_beit_microsoft_beit_large_patch16_224_img_cls_hf",
                "pt_vit_vit_l_16_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 16, 197, 197)"},
        },
    ),
    (
        Reshape363,
        [((729, 16), torch.bfloat16)],
        {
            "model_names": ["pt_beit_microsoft_beit_large_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 27, 27, 16)"},
        },
    ),
    (
        Reshape364,
        [((1, 27, 27, 16), torch.bfloat16)],
        {
            "model_names": ["pt_beit_microsoft_beit_large_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(729, 16)"},
        },
    ),
    (
        Reshape365,
        [((38809, 16), torch.bfloat16)],
        {
            "model_names": ["pt_beit_microsoft_beit_large_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(197, 197, 16)"},
        },
    ),
    (
        Reshape366,
        [((1, 16, 197, 197), torch.bfloat16)],
        {
            "model_names": [
                "pt_beit_microsoft_beit_large_patch16_224_img_cls_hf",
                "pt_vit_vit_l_16_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(16, 197, 197)"},
        },
    ),
    (
        Reshape367,
        [((16, 197, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_beit_microsoft_beit_large_patch16_224_img_cls_hf",
                "pt_vit_vit_l_16_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 16, 197, 64)"},
        },
    ),
    (
        Reshape361,
        [((16, 197, 64), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_l_16_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(16, 197, 64)"},
        },
    ),
    (
        Reshape357,
        [((1, 197, 16, 64), torch.bfloat16)],
        {
            "model_names": ["pt_beit_microsoft_beit_large_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(197, 1024)"},
        },
    ),
    (
        Reshape368,
        [((480, 1, 5, 5), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b7_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(480, 1, 5, 5)"},
        },
    ),
    (
        Reshape369,
        [((480, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(480, 1, 3, 3)"},
        },
    ),
    (
        Reshape370,
        [((960, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(960, 1, 3, 3)"},
        },
    ),
    (
        Reshape371,
        [((1344, 1, 5, 5), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b7_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1344, 1, 5, 5)"},
        },
    ),
    (
        Reshape372,
        [((2304, 1, 5, 5), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b7_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(2304, 1, 5, 5)"},
        },
    ),
    (
        Reshape373,
        [((3840, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b7_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(3840, 1, 3, 3)"},
        },
    ),
    (
        Reshape374,
        [((1, 2560, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b7_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 2560, 1, 1)"},
        },
    ),
    (
        Reshape375,
        [((8, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": ["pt_ghostnet_ghostnet_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(8, 1, 3, 3)"},
        },
    ),
    (
        Reshape376,
        [((12, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": ["pt_ghostnet_ghostnet_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(12, 1, 3, 3)"},
        },
    ),
    (
        Reshape377,
        [((16, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(16, 1, 3, 3)"},
        },
    ),
    (
        Reshape378,
        [((36, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": ["pt_ghostnet_ghostnet_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(36, 1, 3, 3)"},
        },
    ),
    (
        Reshape379,
        [((72, 1, 5, 5), torch.bfloat16)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(72, 1, 5, 5)"},
        },
    ),
    (
        Reshape380,
        [((20, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": ["pt_ghostnet_ghostnet_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(20, 1, 3, 3)"},
        },
    ),
    (
        Reshape381,
        [((24, 1, 5, 5), torch.bfloat16)],
        {
            "model_names": ["pt_ghostnet_ghostnet_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(24, 1, 5, 5)"},
        },
    ),
    (
        Reshape382,
        [((60, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": ["pt_ghostnet_ghostnet_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(60, 1, 3, 3)"},
        },
    ),
    (
        Reshape383,
        [((120, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": ["pt_ghostnet_ghostnet_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(120, 1, 3, 3)"},
        },
    ),
    (
        Reshape384,
        [((100, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": ["pt_ghostnet_ghostnet_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(100, 1, 3, 3)"},
        },
    ),
    (
        Reshape385,
        [((92, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": ["pt_ghostnet_ghostnet_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(92, 1, 3, 3)"},
        },
    ),
    (
        Reshape386,
        [((80, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": ["pt_ghostnet_ghostnet_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(80, 1, 3, 3)"},
        },
    ),
    (
        Reshape387,
        [((112, 1, 5, 5), torch.bfloat16)],
        {
            "model_names": ["pt_ghostnet_ghostnet_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(112, 1, 5, 5)"},
        },
    ),
    (
        Reshape388,
        [((1, 1280, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                "pt_mobilenetv2_basic_img_cls_torchhub",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 1280, 1, 1)"},
        },
    ),
    (
        Reshape389,
        [((96, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv1_google_mobilenet_v1_0_75_192_img_cls_hf",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(96, 1, 3, 3)"},
        },
    ),
    (
        Reshape390,
        [((384, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv1_google_mobilenet_v1_0_75_192_img_cls_hf",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(384, 1, 3, 3)"},
        },
    ),
    (
        Reshape391,
        [((768, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv1_google_mobilenet_v1_0_75_192_img_cls_hf",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(768, 1, 3, 3)"},
        },
    ),
    (
        Reshape392,
        [((720, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(720, 1, 3, 3)"},
        },
    ),
    (
        Reshape393,
        [((72, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(72, 1, 3, 3)"},
        },
    ),
    (
        Reshape394,
        [((120, 1, 5, 5), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(120, 1, 5, 5)"},
        },
    ),
    (
        Reshape395,
        [((200, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(200, 1, 3, 3)"},
        },
    ),
    (
        Reshape396,
        [((184, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(184, 1, 3, 3)"},
        },
    ),
    (
        Reshape397,
        [((1, 11, 2560), torch.float32)],
        {"model_names": ["pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf"], "pcc": 0.99, "args": {"shape": "(11, 2560)"}},
    ),
    (
        Reshape398,
        [((1, 11, 2560), torch.float32)],
        {
            "model_names": ["pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 11, 32, 80)"},
        },
    ),
    (
        Reshape399,
        [((11, 2560), torch.float32)],
        {"model_names": ["pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf"], "pcc": 0.99, "args": {"shape": "(1, 11, 2560)"}},
    ),
    (
        Reshape400,
        [((1, 32, 11, 80), torch.float32)],
        {"model_names": ["pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf"], "pcc": 0.99, "args": {"shape": "(32, 11, 80)"}},
    ),
    (
        Reshape401,
        [((32, 11, 11), torch.float32)],
        {
            "model_names": ["pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 32, 11, 11)"},
        },
    ),
    (
        Reshape402,
        [((1, 32, 11, 11), torch.float32)],
        {"model_names": ["pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf"], "pcc": 0.99, "args": {"shape": "(32, 11, 11)"}},
    ),
    (
        Reshape403,
        [((32, 11, 80), torch.float32)],
        {
            "model_names": ["pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 32, 11, 80)"},
        },
    ),
    (
        Reshape397,
        [((1, 11, 32, 80), torch.float32)],
        {"model_names": ["pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf"], "pcc": 0.99, "args": {"shape": "(11, 2560)"}},
    ),
    (
        Reshape404,
        [((11, 10240), torch.float32)],
        {
            "model_names": ["pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 11, 10240)"},
        },
    ),
    (
        Reshape405,
        [((1, 6, 1024), torch.float32)],
        {"model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"], "pcc": 0.99, "args": {"shape": "(6, 1024)"}},
    ),
    (
        Reshape406,
        [((1, 6, 1024), torch.float32)],
        {"model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"], "pcc": 0.99, "args": {"shape": "(1, 6, 16, 64)"}},
    ),
    (
        Reshape407,
        [((6, 1024), torch.float32)],
        {"model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"], "pcc": 0.99, "args": {"shape": "(1, 6, 1024)"}},
    ),
    (
        Reshape408,
        [((1, 16, 6, 64), torch.float32)],
        {"model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"], "pcc": 0.99, "args": {"shape": "(16, 6, 64)"}},
    ),
    (
        Reshape409,
        [((16, 6, 6), torch.float32)],
        {"model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"], "pcc": 0.99, "args": {"shape": "(1, 16, 6, 6)"}},
    ),
    (
        Reshape410,
        [((1, 16, 6, 6), torch.float32)],
        {"model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"], "pcc": 0.99, "args": {"shape": "(16, 6, 6)"}},
    ),
    (
        Reshape411,
        [((16, 6, 64), torch.float32)],
        {"model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"], "pcc": 0.99, "args": {"shape": "(1, 16, 6, 64)"}},
    ),
    (
        Reshape405,
        [((1, 6, 16, 64), torch.float32)],
        {"model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"], "pcc": 0.99, "args": {"shape": "(6, 1024)"}},
    ),
    (
        Reshape412,
        [((6, 2816), torch.float32)],
        {"model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"], "pcc": 0.99, "args": {"shape": "(1, 6, 2816)"}},
    ),
    (
        Reshape413,
        [((1, 35, 1536), torch.float32)],
        {"model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf"], "pcc": 0.99, "args": {"shape": "(35, 1536)"}},
    ),
    (
        Reshape414,
        [((1, 35, 1536), torch.float32)],
        {
            "model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 35, 12, 128)"},
        },
    ),
    (
        Reshape415,
        [((35, 1536), torch.float32)],
        {
            "model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 35, 1536)"},
        },
    ),
    (
        Reshape416,
        [((1, 12, 35, 128), torch.float32)],
        {
            "model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(12, 35, 128)"},
        },
    ),
    (
        Reshape417,
        [((35, 256), torch.float32)],
        {
            "model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 35, 256)"},
        },
    ),
    (
        Reshape418,
        [((1, 35, 256), torch.float32)],
        {
            "model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 35, 2, 128)"},
        },
    ),
    (
        Reshape416,
        [((1, 2, 6, 35, 128), torch.float32)],
        {
            "model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(12, 35, 128)"},
        },
    ),
    (
        Reshape419,
        [((12, 35, 35), torch.float32)],
        {
            "model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 12, 35, 35)"},
        },
    ),
    (
        Reshape420,
        [((1, 12, 35, 35), torch.float32)],
        {
            "model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(12, 35, 35)"},
        },
    ),
    (
        Reshape421,
        [((12, 35, 128), torch.float32)],
        {
            "model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 12, 35, 128)"},
        },
    ),
    (
        Reshape413,
        [((1, 35, 12, 128), torch.float32)],
        {"model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf"], "pcc": 0.99, "args": {"shape": "(35, 1536)"}},
    ),
    (
        Reshape422,
        [((35, 8960), torch.float32)],
        {
            "model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 35, 8960)"},
        },
    ),
    (
        Reshape423,
        [((1, 29, 1536), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99, "args": {"shape": "(29, 1536)"}},
    ),
    (
        Reshape424,
        [((1, 29, 1536), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99, "args": {"shape": "(1, 29, 12, 128)"}},
    ),
    (
        Reshape425,
        [((29, 1536), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99, "args": {"shape": "(1, 29, 1536)"}},
    ),
    (
        Reshape426,
        [((1, 12, 29, 128), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99, "args": {"shape": "(12, 29, 128)"}},
    ),
    (
        Reshape427,
        [((29, 256), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99, "args": {"shape": "(1, 29, 256)"}},
    ),
    (
        Reshape428,
        [((1, 29, 256), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99, "args": {"shape": "(1, 29, 2, 128)"}},
    ),
    (
        Reshape426,
        [((1, 2, 6, 29, 128), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99, "args": {"shape": "(12, 29, 128)"}},
    ),
    (
        Reshape429,
        [((12, 29, 29), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99, "args": {"shape": "(1, 12, 29, 29)"}},
    ),
    (
        Reshape430,
        [((1, 12, 29, 29), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99, "args": {"shape": "(12, 29, 29)"}},
    ),
    (
        Reshape431,
        [((12, 29, 128), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99, "args": {"shape": "(1, 12, 29, 128)"}},
    ),
    (
        Reshape423,
        [((1, 29, 12, 128), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99, "args": {"shape": "(29, 1536)"}},
    ),
    (
        Reshape432,
        [((29, 8960), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99, "args": {"shape": "(1, 29, 8960)"}},
    ),
    (
        Reshape433,
        [((1, 3024, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_facebook_regnet_y_160_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 3024, 1, 1)"},
        },
    ),
    (
        Reshape434,
        [((1, 3712, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_facebook_regnet_y_320_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 3712, 1, 1)"},
        },
    ),
    (
        Reshape435,
        [((1, 2520, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_x_32gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 2520, 1, 1)"},
        },
    ),
    (
        Reshape436,
        [((1, 1008, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_x_3_2gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 1008, 1, 1)"},
        },
    ),
    (
        Reshape437,
        [((1, 64, 64, 96), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(1, 8, 8, 8, 8, 96)"},
        },
    ),
    (
        Reshape438,
        [((1, 64, 64, 96), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(4096, 96)"}},
    ),
    (
        Reshape438,
        [((1, 8, 8, 8, 8, 96), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(4096, 96)"}},
    ),
    (
        Reshape439,
        [((1, 8, 8, 8, 8, 96), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(1, 64, 64, 96)"}},
    ),
    (
        Reshape440,
        [((4096, 288), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(64, 64, 288)"}},
    ),
    (
        Reshape441,
        [((64, 64, 288), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(64, 64, 3, 3, 32)"},
        },
    ),
    (
        Reshape442,
        [((1, 64, 3, 64, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(64, 3, 64, 32)"}},
    ),
    (
        Reshape443,
        [((64, 3, 64, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(192, 64, 32)"}},
    ),
    (
        Reshape444,
        [((192, 64, 64), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(64, 3, 64, 64)"}},
    ),
    (
        Reshape445,
        [((1, 15, 15, 2), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(225, 2)"}},
    ),
    (
        Reshape446,
        [((225, 512), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(1, 15, 15, 512)"}},
    ),
    (
        Reshape447,
        [((1, 15, 15, 512), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(225, 512)"}},
    ),
    (
        Reshape448,
        [((225, 3), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(225, 3)"}},
    ),
    (
        Reshape449,
        [((4096, 3), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(64, 64, 3)"}},
    ),
    (
        Reshape450,
        [((64, 3, 64, 64), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(192, 64, 64)"}},
    ),
    (
        Reshape451,
        [((64, 3, 64, 64), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(1, 64, 3, 64, 64)"},
        },
    ),
    (
        Reshape452,
        [((64, 3, 32, 64), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(192, 32, 64)"}},
    ),
    (
        Reshape442,
        [((192, 64, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(64, 3, 64, 32)"}},
    ),
    (
        Reshape438,
        [((64, 64, 3, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(4096, 96)"}},
    ),
    (
        Reshape453,
        [((4096, 96), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(64, 64, 96)"}},
    ),
    (
        Reshape439,
        [((4096, 96), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(1, 64, 64, 96)"}},
    ),
    (
        Reshape437,
        [((64, 64, 96), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(1, 8, 8, 8, 8, 96)"},
        },
    ),
    (
        Reshape454,
        [((4096, 384), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(1, 64, 64, 384)"}},
    ),
    (
        Reshape455,
        [((1, 64, 64, 384), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(4096, 384)"}},
    ),
    (
        Reshape444,
        [((1, 64, 3, 64, 64), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(64, 3, 64, 64)"}},
    ),
    (
        Reshape456,
        [((1, 32, 32, 384), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(1024, 384)"}},
    ),
    (
        Reshape457,
        [((1024, 192), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(1, 32, 32, 192)"}},
    ),
    (
        Reshape458,
        [((1024, 192), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(16, 64, 192)"}},
    ),
    (
        Reshape459,
        [((1, 32, 32, 192), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(1, 4, 8, 4, 8, 192)"},
        },
    ),
    (
        Reshape460,
        [((1, 32, 32, 192), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(1024, 192)"}},
    ),
    (
        Reshape460,
        [((1, 4, 4, 8, 8, 192), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(1024, 192)"}},
    ),
    (
        Reshape461,
        [((1024, 576), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(16, 64, 576)"}},
    ),
    (
        Reshape462,
        [((16, 64, 576), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(16, 64, 3, 6, 32)"},
        },
    ),
    (
        Reshape463,
        [((1, 16, 6, 64, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(16, 6, 64, 32)"}},
    ),
    (
        Reshape464,
        [((16, 6, 64, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(96, 64, 32)"}},
    ),
    (
        Reshape465,
        [((96, 64, 64), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(16, 6, 64, 64)"}},
    ),
    (
        Reshape466,
        [((225, 6), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(225, 6)"}},
    ),
    (
        Reshape467,
        [((4096, 6), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(64, 64, 6)"}},
    ),
    (
        Reshape468,
        [((16, 6, 64, 64), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(96, 64, 64)"}},
    ),
    (
        Reshape469,
        [((16, 6, 64, 64), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(1, 16, 6, 64, 64)"},
        },
    ),
    (
        Reshape470,
        [((16, 6, 32, 64), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(96, 32, 64)"}},
    ),
    (
        Reshape463,
        [((96, 64, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(16, 6, 64, 32)"}},
    ),
    (
        Reshape460,
        [((16, 64, 6, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(1024, 192)"}},
    ),
    (
        Reshape471,
        [((16, 64, 192), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(1, 4, 4, 8, 8, 192)"},
        },
    ),
    (
        Reshape457,
        [((1, 4, 8, 4, 8, 192), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(1, 32, 32, 192)"}},
    ),
    (
        Reshape472,
        [((1024, 768), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(1, 32, 32, 768)"}},
    ),
    (
        Reshape473,
        [((1, 32, 32, 768), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(1024, 768)"}},
    ),
    (
        Reshape465,
        [((1, 16, 6, 64, 64), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(16, 6, 64, 64)"}},
    ),
    (
        Reshape77,
        [((1, 16, 16, 768), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(256, 768)"}},
    ),
    (
        Reshape474,
        [((256, 384), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(1, 16, 16, 384)"}},
    ),
    (
        Reshape475,
        [((256, 384), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(4, 64, 384)"}},
    ),
    (
        Reshape476,
        [((1, 16, 16, 384), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(1, 2, 8, 2, 8, 384)"},
        },
    ),
    (
        Reshape477,
        [((1, 16, 16, 384), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(256, 384)"}},
    ),
    (
        Reshape477,
        [((1, 2, 2, 8, 8, 384), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(256, 384)"}},
    ),
    (
        Reshape478,
        [((256, 1152), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(4, 64, 1152)"}},
    ),
    (
        Reshape479,
        [((4, 64, 1152), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(4, 64, 3, 12, 32)"},
        },
    ),
    (
        Reshape480,
        [((1, 4, 12, 64, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(4, 12, 64, 32)"}},
    ),
    (
        Reshape481,
        [((4, 12, 64, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(48, 64, 32)"}},
    ),
    (
        Reshape482,
        [((48, 64, 64), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(4, 12, 64, 64)"}},
    ),
    (
        Reshape483,
        [((225, 12), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(225, 12)"}},
    ),
    (
        Reshape484,
        [((4096, 12), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(64, 64, 12)"}},
    ),
    (
        Reshape485,
        [((4, 12, 64, 64), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(48, 64, 64)"}},
    ),
    (
        Reshape486,
        [((4, 12, 64, 64), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(1, 4, 12, 64, 64)"},
        },
    ),
    (
        Reshape487,
        [((4, 12, 32, 64), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(48, 32, 64)"}},
    ),
    (
        Reshape480,
        [((48, 64, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(4, 12, 64, 32)"}},
    ),
    (
        Reshape477,
        [((4, 64, 12, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(256, 384)"}},
    ),
    (
        Reshape488,
        [((4, 64, 384), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(1, 2, 2, 8, 8, 384)"},
        },
    ),
    (
        Reshape474,
        [((1, 2, 8, 2, 8, 384), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(1, 16, 16, 384)"}},
    ),
    (
        Reshape489,
        [((256, 1536), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(1, 16, 16, 1536)"}},
    ),
    (
        Reshape490,
        [((1, 16, 16, 1536), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(256, 1536)"}},
    ),
    (
        Reshape482,
        [((1, 4, 12, 64, 64), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(4, 12, 64, 64)"}},
    ),
    (
        Reshape491,
        [((1, 8, 8, 1536), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(64, 1536)"}},
    ),
    (
        Reshape492,
        [((64, 768), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(1, 8, 8, 768)"}},
    ),
    (
        Reshape493,
        [((64, 768), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(1, 64, 768)"}},
    ),
    (
        Reshape494,
        [((1, 8, 8, 768), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(1, 1, 8, 1, 8, 768)"},
        },
    ),
    (
        Reshape495,
        [((1, 8, 8, 768), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(64, 768)"}},
    ),
    (
        Reshape495,
        [((1, 1, 1, 8, 8, 768), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(64, 768)"}},
    ),
    (
        Reshape496,
        [((64, 2304), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(1, 64, 2304)"}},
    ),
    (
        Reshape497,
        [((1, 64, 2304), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(1, 64, 3, 24, 32)"},
        },
    ),
    (
        Reshape498,
        [((1, 1, 24, 64, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(1, 24, 64, 32)"}},
    ),
    (
        Reshape499,
        [((1, 24, 64, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(24, 64, 32)"}},
    ),
    (
        Reshape500,
        [((24, 64, 64), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(1, 24, 64, 64)"}},
    ),
    (
        Reshape501,
        [((225, 24), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(225, 24)"}},
    ),
    (
        Reshape502,
        [((4096, 24), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(64, 64, 24)"}},
    ),
    (
        Reshape503,
        [((1, 24, 64, 64), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(24, 64, 64)"}},
    ),
    (
        Reshape504,
        [((1, 24, 32, 64), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(24, 32, 64)"}},
    ),
    (
        Reshape498,
        [((24, 64, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(1, 24, 64, 32)"}},
    ),
    (
        Reshape495,
        [((1, 64, 24, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(64, 768)"}},
    ),
    (
        Reshape505,
        [((1, 64, 768), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(1, 1, 1, 8, 8, 768)"},
        },
    ),
    (
        Reshape492,
        [((1, 1, 8, 1, 8, 768), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(1, 8, 8, 768)"}},
    ),
    (
        Reshape506,
        [((64, 3072), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(1, 8, 8, 3072)"}},
    ),
    (
        Reshape507,
        [((1, 8, 8, 3072), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(64, 3072)"}},
    ),
    (
        Reshape241,
        [((1, 768, 1, 1), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(1, 768, 1, 1)"}},
    ),
    (
        Reshape508,
        [((1, 513), torch.int64)],
        {
            "model_names": ["pt_t5_t5_base_text_gen_hf", "pt_t5_t5_small_text_gen_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 513)"},
        },
    ),
    (
        Reshape509,
        [((1, 513, 768), torch.float32)],
        {"model_names": ["pt_t5_t5_base_text_gen_hf"], "pcc": 0.99, "args": {"shape": "(513, 768)"}},
    ),
    (
        Reshape510,
        [((513, 768), torch.float32)],
        {"model_names": ["pt_t5_t5_base_text_gen_hf"], "pcc": 0.99, "args": {"shape": "(1, 513, 12, 64)"}},
    ),
    (
        Reshape511,
        [((513, 768), torch.float32)],
        {"model_names": ["pt_t5_t5_base_text_gen_hf"], "pcc": 0.99, "args": {"shape": "(1, 513, 768)"}},
    ),
    (
        Reshape512,
        [((1, 12, 513, 64), torch.float32)],
        {"model_names": ["pt_t5_t5_base_text_gen_hf"], "pcc": 0.99, "args": {"shape": "(12, 513, 64)"}},
    ),
    (
        Reshape513,
        [((12, 513, 513), torch.float32)],
        {"model_names": ["pt_t5_t5_base_text_gen_hf"], "pcc": 0.99, "args": {"shape": "(1, 12, 513, 513)"}},
    ),
    (
        Reshape514,
        [((1, 12, 513, 513), torch.float32)],
        {"model_names": ["pt_t5_t5_base_text_gen_hf"], "pcc": 0.99, "args": {"shape": "(12, 513, 513)"}},
    ),
    (
        Reshape515,
        [((1, 12, 64, 513), torch.float32)],
        {"model_names": ["pt_t5_t5_base_text_gen_hf"], "pcc": 0.99, "args": {"shape": "(12, 64, 513)"}},
    ),
    (
        Reshape516,
        [((12, 513, 64), torch.float32)],
        {"model_names": ["pt_t5_t5_base_text_gen_hf"], "pcc": 0.99, "args": {"shape": "(1, 12, 513, 64)"}},
    ),
    (
        Reshape509,
        [((1, 513, 12, 64), torch.float32)],
        {"model_names": ["pt_t5_t5_base_text_gen_hf"], "pcc": 0.99, "args": {"shape": "(513, 768)"}},
    ),
    (
        Reshape517,
        [((1, 61), torch.int64)],
        {
            "model_names": ["pt_t5_t5_base_text_gen_hf", "pt_t5_t5_small_text_gen_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 61)"},
        },
    ),
    (
        Reshape518,
        [((1, 61, 768), torch.float32)],
        {"model_names": ["pt_t5_t5_base_text_gen_hf"], "pcc": 0.99, "args": {"shape": "(61, 768)"}},
    ),
    (
        Reshape519,
        [((61, 768), torch.float32)],
        {"model_names": ["pt_t5_t5_base_text_gen_hf"], "pcc": 0.99, "args": {"shape": "(1, 61, 12, 64)"}},
    ),
    (
        Reshape520,
        [((61, 768), torch.float32)],
        {"model_names": ["pt_t5_t5_base_text_gen_hf"], "pcc": 0.99, "args": {"shape": "(1, 61, 768)"}},
    ),
    (
        Reshape521,
        [((1, 12, 61, 64), torch.float32)],
        {"model_names": ["pt_t5_t5_base_text_gen_hf"], "pcc": 0.99, "args": {"shape": "(12, 61, 64)"}},
    ),
    (
        Reshape522,
        [((12, 61, 61), torch.float32)],
        {"model_names": ["pt_t5_t5_base_text_gen_hf"], "pcc": 0.99, "args": {"shape": "(1, 12, 61, 61)"}},
    ),
    (
        Reshape523,
        [((1, 12, 61, 61), torch.float32)],
        {"model_names": ["pt_t5_t5_base_text_gen_hf"], "pcc": 0.99, "args": {"shape": "(12, 61, 61)"}},
    ),
    (
        Reshape524,
        [((1, 12, 64, 61), torch.float32)],
        {"model_names": ["pt_t5_t5_base_text_gen_hf"], "pcc": 0.99, "args": {"shape": "(12, 64, 61)"}},
    ),
    (
        Reshape525,
        [((12, 61, 64), torch.float32)],
        {"model_names": ["pt_t5_t5_base_text_gen_hf"], "pcc": 0.99, "args": {"shape": "(1, 12, 61, 64)"}},
    ),
    (
        Reshape518,
        [((1, 61, 12, 64), torch.float32)],
        {"model_names": ["pt_t5_t5_base_text_gen_hf"], "pcc": 0.99, "args": {"shape": "(61, 768)"}},
    ),
    (
        Reshape526,
        [((12, 513, 61), torch.float32)],
        {"model_names": ["pt_t5_t5_base_text_gen_hf"], "pcc": 0.99, "args": {"shape": "(1, 12, 513, 61)"}},
    ),
    (
        Reshape527,
        [((1, 12, 513, 61), torch.float32)],
        {"model_names": ["pt_t5_t5_base_text_gen_hf"], "pcc": 0.99, "args": {"shape": "(12, 513, 61)"}},
    ),
    (
        Reshape528,
        [((1, 3, 85, 10, 10), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5x_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_320x320",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 255, 10, 10)"},
        },
    ),
    (
        Reshape529,
        [((1, 255, 10, 10), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5x_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_320x320",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 1, 255, 100)"},
        },
    ),
    (
        Reshape530,
        [((1, 1, 255, 100), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5x_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_320x320",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 3, 85, 100)"},
        },
    ),
    (
        Reshape531,
        [((1, 3, 100, 85), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5x_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_320x320",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 300, 85)"},
        },
    ),
    (
        Reshape532,
        [((1, 128, 2048), torch.float32)],
        {"model_names": ["pt_albert_xlarge_v2_token_cls_hf"], "pcc": 0.99, "args": {"shape": "(128, 2048)"}},
    ),
    (
        Reshape533,
        [((1, 128, 2048), torch.float32)],
        {"model_names": ["pt_albert_xlarge_v2_token_cls_hf"], "pcc": 0.99, "args": {"shape": "(1, 128, 16, 128)"}},
    ),
    (
        Reshape534,
        [((128, 2048), torch.float32)],
        {"model_names": ["pt_albert_xlarge_v2_token_cls_hf"], "pcc": 0.99, "args": {"shape": "(1, 128, 2048)"}},
    ),
    (
        Reshape532,
        [((1, 128, 16, 128), torch.float32)],
        {"model_names": ["pt_albert_xlarge_v2_token_cls_hf"], "pcc": 0.99, "args": {"shape": "(128, 2048)"}},
    ),
    (
        Reshape535,
        [((729, 12), torch.bfloat16)],
        {
            "model_names": ["pt_beit_microsoft_beit_base_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 27, 27, 12)"},
        },
    ),
    (
        Reshape536,
        [((1, 27, 27, 12), torch.bfloat16)],
        {
            "model_names": ["pt_beit_microsoft_beit_base_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(729, 12)"},
        },
    ),
    (
        Reshape537,
        [((38809, 12), torch.bfloat16)],
        {
            "model_names": ["pt_beit_microsoft_beit_base_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(197, 197, 12)"},
        },
    ),
    (
        Reshape538,
        [((1, 192, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 192, 196, 1)"},
        },
    ),
    (
        Reshape539,
        [((1, 197, 192), torch.bfloat16)],
        {
            "model_names": ["pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(197, 192)"},
        },
    ),
    (
        Reshape540,
        [((1, 197, 192), torch.bfloat16)],
        {
            "model_names": ["pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 197, 3, 64)"},
        },
    ),
    (
        Reshape541,
        [((197, 192), torch.bfloat16)],
        {
            "model_names": ["pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 197, 192)"},
        },
    ),
    (
        Reshape542,
        [((1, 3, 197, 64), torch.bfloat16)],
        {
            "model_names": ["pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(3, 197, 64)"},
        },
    ),
    (
        Reshape543,
        [((3, 197, 197), torch.bfloat16)],
        {
            "model_names": ["pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 3, 197, 197)"},
        },
    ),
    (
        Reshape544,
        [((1, 3, 197, 197), torch.bfloat16)],
        {
            "model_names": ["pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(3, 197, 197)"},
        },
    ),
    (
        Reshape545,
        [((3, 197, 64), torch.bfloat16)],
        {
            "model_names": ["pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 3, 197, 64)"},
        },
    ),
    (
        Reshape539,
        [((1, 197, 3, 64), torch.bfloat16)],
        {
            "model_names": ["pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(197, 192)"},
        },
    ),
    (
        Reshape546,
        [((1, 1, 192), torch.bfloat16)],
        {
            "model_names": ["pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 192)"},
        },
    ),
    (
        Reshape547,
        [((384, 1, 5, 5), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(384, 1, 5, 5)"},
        },
    ),
    (
        Reshape548,
        [((768, 1, 5, 5), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(768, 1, 5, 5)"},
        },
    ),
    (
        Reshape549,
        [((1056, 1, 5, 5), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1056, 1, 5, 5)"},
        },
    ),
    (
        Reshape550,
        [((1824, 1, 5, 5), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1824, 1, 5, 5)"},
        },
    ),
    (
        Reshape551,
        [((1824, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1824, 1, 3, 3)"},
        },
    ),
    (
        Reshape552,
        [((3072, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(3072, 1, 3, 3)"},
        },
    ),
    (
        Reshape553,
        [((88, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(88, 1, 3, 3)"},
        },
    ),
    (
        Reshape554,
        [((96, 1, 5, 5), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(96, 1, 5, 5)"},
        },
    ),
    (
        Reshape555,
        [((144, 1, 5, 5), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(144, 1, 5, 5)"},
        },
    ),
    (
        Reshape556,
        [((1, 29, 896), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"], "pcc": 0.99, "args": {"shape": "(29, 896)"}},
    ),
    (
        Reshape557,
        [((1, 29, 896), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"], "pcc": 0.99, "args": {"shape": "(1, 29, 14, 64)"}},
    ),
    (
        Reshape558,
        [((29, 896), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"], "pcc": 0.99, "args": {"shape": "(1, 29, 896)"}},
    ),
    (
        Reshape559,
        [((1, 14, 29, 64), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"], "pcc": 0.99, "args": {"shape": "(14, 29, 64)"}},
    ),
    (
        Reshape560,
        [((29, 128), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"], "pcc": 0.99, "args": {"shape": "(1, 29, 128)"}},
    ),
    (
        Reshape561,
        [((1, 29, 128), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"], "pcc": 0.99, "args": {"shape": "(1, 29, 2, 64)"}},
    ),
    (
        Reshape559,
        [((1, 2, 7, 29, 64), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"], "pcc": 0.99, "args": {"shape": "(14, 29, 64)"}},
    ),
    (
        Reshape562,
        [((14, 29, 29), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"], "pcc": 0.99, "args": {"shape": "(1, 14, 29, 29)"}},
    ),
    (
        Reshape563,
        [((1, 14, 29, 29), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"], "pcc": 0.99, "args": {"shape": "(14, 29, 29)"}},
    ),
    (
        Reshape564,
        [((14, 29, 64), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"], "pcc": 0.99, "args": {"shape": "(1, 14, 29, 64)"}},
    ),
    (
        Reshape556,
        [((1, 29, 14, 64), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"], "pcc": 0.99, "args": {"shape": "(29, 896)"}},
    ),
    (
        Reshape565,
        [((29, 4864), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"], "pcc": 0.99, "args": {"shape": "(1, 29, 4864)"}},
    ),
    (
        Reshape566,
        [((1, 64, 128, 128), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 64, 16384, 1)"},
        },
    ),
    (
        Reshape300,
        [((1, 16384, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 16384, 1, 64)"},
        },
    ),
    (
        Reshape301,
        [((1, 16384, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 128, 128, 64)"},
        },
    ),
    (
        Reshape302,
        [((1, 64, 16384), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 64, 128, 128)"},
        },
    ),
    (
        Reshape131,
        [((1, 1, 64, 256), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 64, 256)"},
        },
    ),
    (
        Reshape303,
        [((1, 256, 16384), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 256, 128, 128)"},
        },
    ),
    (
        Reshape567,
        [((1, 256, 128, 128), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 256, 16384, 1)"},
        },
    ),
    (
        Reshape568,
        [((1, 128, 64, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 128, 4096, 1)"},
        },
    ),
    (
        Reshape306,
        [((1, 4096, 128), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 4096, 2, 64)"},
        },
    ),
    (
        Reshape307,
        [((1, 4096, 128), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 64, 64, 128)"},
        },
    ),
    (
        Reshape308,
        [((1, 2, 4096, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(2, 4096, 64)"},
        },
    ),
    (
        Reshape309,
        [((1, 128, 4096), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 128, 64, 64)"},
        },
    ),
    (
        Reshape310,
        [((1, 128, 16, 16), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 128, 256)"},
        },
    ),
    (
        Reshape311,
        [((1, 256, 128), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(256, 128)"},
        },
    ),
    (
        Reshape312,
        [((1, 256, 128), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 256, 2, 64)"},
        },
    ),
    (
        Reshape313,
        [((256, 128), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 256, 128)"},
        },
    ),
    (
        Reshape315,
        [((1, 2, 256, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(2, 256, 64)"},
        },
    ),
    (
        Reshape314,
        [((1, 2, 64, 256), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(2, 64, 256)"},
        },
    ),
    (
        Reshape316,
        [((2, 4096, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 2, 4096, 64)"},
        },
    ),
    (
        Reshape317,
        [((1, 4096, 2, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(4096, 128)"},
        },
    ),
    (
        Reshape318,
        [((4096, 128), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 4096, 128)"},
        },
    ),
    (
        Reshape319,
        [((1, 512, 4096), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 512, 64, 64)"},
        },
    ),
    (
        Reshape569,
        [((1, 512, 64, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 512, 4096, 1)"},
        },
    ),
    (
        Reshape570,
        [((1, 320, 32, 32), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 320, 1024, 1)"},
        },
    ),
    (
        Reshape322,
        [((1, 1024, 320), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 1024, 5, 64)"},
        },
    ),
    (
        Reshape323,
        [((1, 1024, 320), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 32, 32, 320)"},
        },
    ),
    (
        Reshape324,
        [((1, 5, 1024, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(5, 1024, 64)"},
        },
    ),
    (
        Reshape325,
        [((1, 320, 1024), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 320, 32, 32)"},
        },
    ),
    (
        Reshape326,
        [((1, 320, 16, 16), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 320, 256)"},
        },
    ),
    (
        Reshape327,
        [((1, 256, 320), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(256, 320)"},
        },
    ),
    (
        Reshape328,
        [((1, 256, 320), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 256, 5, 64)"},
        },
    ),
    (
        Reshape329,
        [((256, 320), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 256, 320)"},
        },
    ),
    (
        Reshape331,
        [((1, 5, 256, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(5, 256, 64)"},
        },
    ),
    (
        Reshape330,
        [((1, 5, 64, 256), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(5, 64, 256)"},
        },
    ),
    (
        Reshape332,
        [((5, 1024, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 5, 1024, 64)"},
        },
    ),
    (
        Reshape333,
        [((1, 1024, 5, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1024, 320)"},
        },
    ),
    (
        Reshape334,
        [((1024, 320), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 1024, 320)"},
        },
    ),
    (
        Reshape335,
        [((1, 1280, 1024), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 1280, 32, 32)"},
        },
    ),
    (
        Reshape571,
        [((1280, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1280, 1, 3, 3)"},
        },
    ),
    (
        Reshape572,
        [((1, 1280, 32, 32), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 1280, 1024, 1)"},
        },
    ),
    (
        Reshape573,
        [((1, 512, 16, 16), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 512, 256, 1)"},
        },
    ),
    (
        Reshape338,
        [((1, 8, 256, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(8, 256, 64)"},
        },
    ),
    (
        Reshape339,
        [((1, 8, 64, 256), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(8, 64, 256)"},
        },
    ),
    (
        Reshape340,
        [((8, 256, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 8, 256, 64)"},
        },
    ),
    (
        Reshape80,
        [((1, 256, 8, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(256, 512)"},
        },
    ),
    (
        Reshape341,
        [((1, 2048, 256), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 2048, 16, 16)"},
        },
    ),
    (
        Reshape574,
        [((2048, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(2048, 1, 3, 3)"},
        },
    ),
    (
        Reshape575,
        [((1, 2048, 16, 16), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 2048, 256, 1)"},
        },
    ),
    (
        Reshape343,
        [((1, 768, 256), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 768, 16, 16)"},
        },
    ),
    (
        Reshape344,
        [((1, 768, 1024), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 768, 32, 32)"},
        },
    ),
    (
        Reshape345,
        [((1, 768, 4096), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 768, 64, 64)"},
        },
    ),
    (
        Reshape346,
        [((1, 768, 16384), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 768, 128, 128)"},
        },
    ),
    (
        Reshape576,
        [((1, 768, 128), torch.float32)],
        {
            "model_names": ["pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 12, 64, 128)"},
        },
    ),
    (
        Reshape577,
        [((1, 768, 128), torch.float32)],
        {
            "model_names": ["pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(12, 64, 128)"},
        },
    ),
    (
        Reshape578,
        [((1, 12, 64, 128), torch.float32)],
        {
            "model_names": ["pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 768, 128, 1)"},
        },
    ),
    (
        Reshape579,
        [((768, 768, 1), torch.float32)],
        {
            "model_names": ["pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(768, 768, 1, 1)"},
        },
    ),
    (
        Reshape580,
        [((1, 768, 128, 1), torch.float32)],
        {
            "model_names": ["pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 768, 128)"},
        },
    ),
    (
        Reshape247,
        [((197, 1, 768), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_b_16_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(197, 768)"},
        },
    ),
    (
        Reshape581,
        [((197, 2304), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_b_16_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(197, 1, 2304)"},
        },
    ),
    (
        Reshape582,
        [((197, 1, 2304), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_b_16_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(197, 1, 3, 768)"},
        },
    ),
    (
        Reshape583,
        [((1, 197, 1, 768), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_b_16_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(197, 12, 64)"},
        },
    ),
    (
        Reshape247,
        [((197, 1, 12, 64), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_b_16_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(197, 768)"},
        },
    ),
    (
        Reshape357,
        [((197, 1, 1024), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_l_16_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(197, 1024)"},
        },
    ),
    (
        Reshape584,
        [((197, 3072), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_l_16_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(197, 1, 3072)"},
        },
    ),
    (
        Reshape585,
        [((197, 1, 3072), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_l_16_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(197, 1, 3, 1024)"},
        },
    ),
    (
        Reshape586,
        [((1, 197, 1, 1024), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_l_16_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(197, 16, 64)"},
        },
    ),
    (
        Reshape357,
        [((197, 1, 16, 64), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_l_16_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(197, 1024)"},
        },
    ),
    (
        Reshape587,
        [((160, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": ["pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(160, 1, 3, 3)"},
        },
    ),
    (
        Reshape588,
        [((224, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": ["pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(224, 1, 3, 3)"},
        },
    ),
    (
        Reshape589,
        [((1, 256), torch.int64)],
        {
            "model_names": [
                "pt_xglm_facebook_xglm_564m_clm_hf",
                "pt_gpt_gpt2_text_gen_hf",
                "pt_opt_facebook_opt_125m_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 256)"},
        },
    ),
    (
        Reshape590,
        [((256, 1024), torch.float32)],
        {
            "model_names": ["pt_xglm_facebook_xglm_564m_clm_hf", "pt_codegen_salesforce_codegen_350m_nl_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 256, 1024)"},
        },
    ),
    (
        Reshape591,
        [((256, 1024), torch.float32)],
        {
            "model_names": ["pt_codegen_salesforce_codegen_350m_nl_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 256, 4, 256)"},
        },
    ),
    (
        Reshape592,
        [((1, 256, 1024), torch.float32)],
        {
            "model_names": ["pt_xglm_facebook_xglm_564m_clm_hf", "pt_codegen_salesforce_codegen_350m_nl_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(256, 1024)"},
        },
    ),
    (
        Reshape593,
        [((1, 256, 1024), torch.float32)],
        {"model_names": ["pt_xglm_facebook_xglm_564m_clm_hf"], "pcc": 0.99, "args": {"shape": "(1, 256, 16, 64)"}},
    ),
    (
        Reshape594,
        [((1, 16, 256, 64), torch.float32)],
        {
            "model_names": ["pt_xglm_facebook_xglm_564m_clm_hf", "pt_codegen_salesforce_codegen_350m_nl_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(16, 256, 64)"},
        },
    ),
    (
        Reshape595,
        [((16, 256, 256), torch.float32)],
        {
            "model_names": ["pt_xglm_facebook_xglm_564m_clm_hf", "pt_codegen_salesforce_codegen_350m_nl_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 16, 256, 256)"},
        },
    ),
    (
        Reshape596,
        [((1, 16, 256, 256), torch.float32)],
        {
            "model_names": ["pt_xglm_facebook_xglm_564m_clm_hf", "pt_codegen_salesforce_codegen_350m_nl_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(16, 256, 256)"},
        },
    ),
    (
        Reshape597,
        [((16, 256, 64), torch.float32)],
        {
            "model_names": ["pt_xglm_facebook_xglm_564m_clm_hf", "pt_codegen_salesforce_codegen_350m_nl_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 16, 256, 64)"},
        },
    ),
    (
        Reshape592,
        [((1, 256, 16, 64), torch.float32)],
        {
            "model_names": ["pt_xglm_facebook_xglm_564m_clm_hf", "pt_codegen_salesforce_codegen_350m_nl_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(256, 1024)"},
        },
    ),
    (
        Reshape598,
        [((1, 68, 56, 80), torch.bfloat16)],
        {
            "model_names": ["pt_yolo_v6_yolov6l_obj_det_torchhub"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 4, 17, 4480)"},
        },
    ),
    (
        Reshape599,
        [((1, 1, 4, 4480), torch.bfloat16)],
        {
            "model_names": ["pt_yolo_v6_yolov6l_obj_det_torchhub"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 4, 4480)"},
        },
    ),
    (
        Reshape600,
        [((1, 68, 28, 40), torch.bfloat16)],
        {
            "model_names": ["pt_yolo_v6_yolov6l_obj_det_torchhub"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 4, 17, 1120)"},
        },
    ),
    (
        Reshape601,
        [((1, 1, 4, 1120), torch.bfloat16)],
        {
            "model_names": ["pt_yolo_v6_yolov6l_obj_det_torchhub"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 4, 1120)"},
        },
    ),
    (
        Reshape602,
        [((1, 68, 14, 20), torch.bfloat16)],
        {
            "model_names": ["pt_yolo_v6_yolov6l_obj_det_torchhub"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 4, 17, 280)"},
        },
    ),
    (
        Reshape603,
        [((1, 1, 4, 280), torch.bfloat16)],
        {
            "model_names": ["pt_yolo_v6_yolov6l_obj_det_torchhub"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 4, 280)"},
        },
    ),
    (
        Reshape604,
        [((1, 80, 56, 80), torch.bfloat16)],
        {
            "model_names": ["pt_yolo_v6_yolov6l_obj_det_torchhub"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 80, 4480)"},
        },
    ),
    (
        Reshape605,
        [((1, 80, 28, 40), torch.bfloat16)],
        {
            "model_names": ["pt_yolo_v6_yolov6l_obj_det_torchhub"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 80, 1120)"},
        },
    ),
    (
        Reshape606,
        [((1, 80, 14, 20), torch.bfloat16)],
        {
            "model_names": ["pt_yolo_v6_yolov6l_obj_det_torchhub"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 80, 280)"},
        },
    ),
    (
        Reshape607,
        [((64,), torch.float32)],
        {"model_names": ["tf_resnet_resnet50_img_cls_keras"], "pcc": 0.99, "args": {"shape": "(64,)"}},
    ),
    (
        Reshape608,
        [((256,), torch.float32)],
        {"model_names": ["tf_resnet_resnet50_img_cls_keras"], "pcc": 0.99, "args": {"shape": "(256,)"}},
    ),
    (
        Reshape609,
        [((512,), torch.float32)],
        {"model_names": ["tf_resnet_resnet50_img_cls_keras"], "pcc": 0.99, "args": {"shape": "(512,)"}},
    ),
    (
        Reshape610,
        [((128,), torch.float32)],
        {"model_names": ["tf_resnet_resnet50_img_cls_keras"], "pcc": 0.99, "args": {"shape": "(128,)"}},
    ),
    (
        Reshape611,
        [((1024,), torch.float32)],
        {"model_names": ["tf_resnet_resnet50_img_cls_keras"], "pcc": 0.99, "args": {"shape": "(1024,)"}},
    ),
    (
        Reshape612,
        [((2048,), torch.float32)],
        {"model_names": ["tf_resnet_resnet50_img_cls_keras"], "pcc": 0.99, "args": {"shape": "(2048,)"}},
    ),
    (
        Reshape613,
        [((1, 576, 1, 1), torch.float32)],
        {
            "model_names": ["onnx_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"],
            "pcc": 0.99,
            "args": {"shape": "(1, 576)"},
        },
    ),
    (
        Reshape96,
        [((1, 512, 1, 1), torch.float32)],
        {"model_names": ["onnx_vovnet_vovnet27s_obj_det_osmr"], "pcc": 0.99, "args": {"shape": "(1, 512)"}},
    ),
    (
        Reshape94,
        [((1, 1024, 1, 1), torch.float32)],
        {"model_names": ["onnx_vovnet_vovnet_v1_57_obj_det_torchhub"], "pcc": 0.99, "args": {"shape": "(1, 1024)"}},
    ),
    (
        Reshape614,
        [((1, 1280, 1, 1), torch.float32)],
        {
            "model_names": ["onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm"],
            "pcc": 0.99,
            "args": {"shape": "(1, 1280)"},
        },
    ),
    (
        Reshape615,
        [((1, 192, 14, 14), torch.float32)],
        {
            "model_names": ["onnx_deit_facebook_deit_tiny_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 192, 196)"},
        },
    ),
    (
        Reshape539,
        [((1, 197, 192), torch.float32)],
        {
            "model_names": ["onnx_deit_facebook_deit_tiny_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(197, 192)"},
        },
    ),
    (
        Reshape540,
        [((1, 197, 192), torch.float32)],
        {
            "model_names": ["onnx_deit_facebook_deit_tiny_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 197, 3, 64)"},
        },
    ),
    (
        Reshape541,
        [((197, 192), torch.float32)],
        {
            "model_names": ["onnx_deit_facebook_deit_tiny_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 197, 192)"},
        },
    ),
    (
        Reshape542,
        [((1, 3, 197, 64), torch.float32)],
        {
            "model_names": ["onnx_deit_facebook_deit_tiny_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(3, 197, 64)"},
        },
    ),
    (
        Reshape616,
        [((1, 3, 64, 197), torch.float32)],
        {
            "model_names": ["onnx_deit_facebook_deit_tiny_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(3, 64, 197)"},
        },
    ),
    (
        Reshape543,
        [((3, 197, 197), torch.float32)],
        {
            "model_names": ["onnx_deit_facebook_deit_tiny_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 3, 197, 197)"},
        },
    ),
    (
        Reshape544,
        [((1, 3, 197, 197), torch.float32)],
        {
            "model_names": ["onnx_deit_facebook_deit_tiny_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(3, 197, 197)"},
        },
    ),
    (
        Reshape545,
        [((3, 197, 64), torch.float32)],
        {
            "model_names": ["onnx_deit_facebook_deit_tiny_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 3, 197, 64)"},
        },
    ),
    (
        Reshape539,
        [((1, 197, 3, 64), torch.float32)],
        {
            "model_names": ["onnx_deit_facebook_deit_tiny_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(197, 192)"},
        },
    ),
    (
        Reshape546,
        [((1, 1, 192), torch.float32)],
        {
            "model_names": ["onnx_deit_facebook_deit_tiny_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 192)"},
        },
    ),
    (
        Reshape291,
        [((1, 144, 80, 80), torch.float32)],
        {"model_names": ["onnx_yolov8_default_obj_det_github"], "pcc": 0.99, "args": {"shape": "(1, 144, 6400)"}},
    ),
    (
        Reshape292,
        [((1, 144, 40, 40), torch.float32)],
        {"model_names": ["onnx_yolov8_default_obj_det_github"], "pcc": 0.99, "args": {"shape": "(1, 144, 1600)"}},
    ),
    (
        Reshape293,
        [((1, 144, 20, 20), torch.float32)],
        {"model_names": ["onnx_yolov8_default_obj_det_github"], "pcc": 0.99, "args": {"shape": "(1, 144, 400)"}},
    ),
    (
        Reshape294,
        [((1, 64, 8400), torch.float32)],
        {"model_names": ["onnx_yolov8_default_obj_det_github"], "pcc": 0.99, "args": {"shape": "(1, 4, 16, 8400)"}},
    ),
    (
        Reshape295,
        [((1, 1, 4, 8400), torch.float32)],
        {"model_names": ["onnx_yolov8_default_obj_det_github"], "pcc": 0.99, "args": {"shape": "(1, 4, 8400)"}},
    ),
    (
        Reshape617,
        [((1, 256, 6, 6), torch.float32)],
        {"model_names": ["pd_alexnet_base_img_cls_paddlemodels"], "pcc": 0.99, "args": {"shape": "(1, 9216, 1, 1)"}},
    ),
    (
        Reshape618,
        [((25, 1, 2, 48), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"shape": "(25, 1, 96)"},
        },
    ),
    (
        Reshape593,
        [((1, 256, 4, 256), torch.float32)],
        {
            "model_names": ["pt_codegen_salesforce_codegen_350m_nl_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 256, 16, 64)"},
        },
    ),
    (
        Reshape619,
        [((1, 256, 16, 16, 2), torch.float32)],
        {
            "model_names": ["pt_codegen_salesforce_codegen_350m_nl_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 256, 16, 32, 1)"},
        },
    ),
    (
        Reshape620,
        [((1, 16, 64, 256), torch.float32)],
        {
            "model_names": ["pt_codegen_salesforce_codegen_350m_nl_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(16, 64, 256)"},
        },
    ),
    (
        Reshape621,
        [((256, 4096), torch.float32)],
        {
            "model_names": ["pt_codegen_salesforce_codegen_350m_nl_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 256, 4096)"},
        },
    ),
    (
        Reshape622,
        [((1, 1920, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 1920, 1, 1)"},
        },
    ),
    (
        Reshape623,
        [((528, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b2_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(528, 1, 3, 3)"},
        },
    ),
    (
        Reshape624,
        [((528, 1, 5, 5), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b2_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(528, 1, 5, 5)"},
        },
    ),
    (
        Reshape625,
        [((720, 1, 5, 5), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b2_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(720, 1, 5, 5)"},
        },
    ),
    (
        Reshape626,
        [((1248, 1, 5, 5), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b2_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1248, 1, 5, 5)"},
        },
    ),
    (
        Reshape627,
        [((1248, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b2_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1248, 1, 3, 3)"},
        },
    ),
    (
        Reshape628,
        [((2112, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b2_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(2112, 1, 3, 3)"},
        },
    ),
    (
        Reshape629,
        [((1, 1408, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b2_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 1408, 1, 1)"},
        },
    ),
    (
        Reshape77,
        [((1, 256, 768), torch.float32)],
        {
            "model_names": ["pt_gpt_gpt2_text_gen_hf", "pt_opt_facebook_opt_125m_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(256, 768)"},
        },
    ),
    (
        Reshape630,
        [((1, 256, 768), torch.float32)],
        {
            "model_names": ["pt_gpt_gpt2_text_gen_hf", "pt_opt_facebook_opt_125m_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 256, 12, 64)"},
        },
    ),
    (
        Reshape631,
        [((256, 768), torch.float32)],
        {
            "model_names": ["pt_gpt_gpt2_text_gen_hf", "pt_opt_facebook_opt_125m_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 256, 768)"},
        },
    ),
    (
        Reshape632,
        [((1, 12, 256, 64), torch.float32)],
        {
            "model_names": ["pt_gpt_gpt2_text_gen_hf", "pt_opt_facebook_opt_125m_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(12, 256, 64)"},
        },
    ),
    (
        Reshape633,
        [((12, 256, 256), torch.float32)],
        {
            "model_names": ["pt_gpt_gpt2_text_gen_hf", "pt_opt_facebook_opt_125m_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 12, 256, 256)"},
        },
    ),
    (
        Reshape634,
        [((1, 12, 256, 256), torch.float32)],
        {
            "model_names": ["pt_gpt_gpt2_text_gen_hf", "pt_opt_facebook_opt_125m_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(12, 256, 256)"},
        },
    ),
    (
        Reshape635,
        [((12, 256, 64), torch.float32)],
        {
            "model_names": ["pt_gpt_gpt2_text_gen_hf", "pt_opt_facebook_opt_125m_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 12, 256, 64)"},
        },
    ),
    (
        Reshape77,
        [((1, 256, 12, 64), torch.float32)],
        {
            "model_names": ["pt_gpt_gpt2_text_gen_hf", "pt_opt_facebook_opt_125m_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(256, 768)"},
        },
    ),
    (
        Reshape636,
        [((256, 3072), torch.float32)],
        {"model_names": ["pt_gpt_gpt2_text_gen_hf"], "pcc": 0.99, "args": {"shape": "(1, 256, 3072)"}},
    ),
    (
        Reshape637,
        [((1, 256, 3072), torch.float32)],
        {"model_names": ["pt_gpt_gpt2_text_gen_hf"], "pcc": 0.99, "args": {"shape": "(256, 3072)"}},
    ),
    (
        Reshape638,
        [((1, 32), torch.int64)],
        {"model_names": ["pt_opt_facebook_opt_125m_seq_cls_hf"], "pcc": 0.99, "args": {"shape": "(1, 32)"}},
    ),
    (
        Reshape639,
        [((1, 32, 768), torch.float32)],
        {"model_names": ["pt_opt_facebook_opt_125m_seq_cls_hf"], "pcc": 0.99, "args": {"shape": "(32, 768)"}},
    ),
    (
        Reshape640,
        [((1, 32, 768), torch.float32)],
        {"model_names": ["pt_opt_facebook_opt_125m_seq_cls_hf"], "pcc": 0.99, "args": {"shape": "(1, 32, 12, 64)"}},
    ),
    (
        Reshape641,
        [((32, 768), torch.float32)],
        {"model_names": ["pt_opt_facebook_opt_125m_seq_cls_hf"], "pcc": 0.99, "args": {"shape": "(1, 32, 768)"}},
    ),
    (
        Reshape642,
        [((1, 12, 32, 64), torch.float32)],
        {"model_names": ["pt_opt_facebook_opt_125m_seq_cls_hf"], "pcc": 0.99, "args": {"shape": "(12, 32, 64)"}},
    ),
    (
        Reshape643,
        [((1, 12, 32, 64), torch.float32)],
        {"model_names": ["pt_phi1_microsoft_phi_1_token_cls_hf"], "pcc": 0.99, "args": {"shape": "(12, 2048)"}},
    ),
    (
        Reshape644,
        [((12, 32, 32), torch.float32)],
        {"model_names": ["pt_opt_facebook_opt_125m_seq_cls_hf"], "pcc": 0.99, "args": {"shape": "(1, 12, 32, 32)"}},
    ),
    (
        Reshape645,
        [((1, 12, 32, 32), torch.float32)],
        {"model_names": ["pt_opt_facebook_opt_125m_seq_cls_hf"], "pcc": 0.99, "args": {"shape": "(12, 32, 32)"}},
    ),
    (
        Reshape646,
        [((12, 32, 64), torch.float32)],
        {"model_names": ["pt_opt_facebook_opt_125m_seq_cls_hf"], "pcc": 0.99, "args": {"shape": "(1, 12, 32, 64)"}},
    ),
    (
        Reshape639,
        [((1, 32, 12, 64), torch.float32)],
        {"model_names": ["pt_opt_facebook_opt_125m_seq_cls_hf"], "pcc": 0.99, "args": {"shape": "(32, 768)"}},
    ),
    (
        Reshape647,
        [((1, 32, 12, 64), torch.float32)],
        {"model_names": ["pt_phi1_microsoft_phi_1_token_cls_hf"], "pcc": 0.99, "args": {"shape": "(32, 12, 64)"}},
    ),
    (
        Reshape648,
        [((32, 2), torch.float32)],
        {"model_names": ["pt_opt_facebook_opt_125m_seq_cls_hf"], "pcc": 0.99, "args": {"shape": "(32, 2)"}},
    ),
    (
        Reshape75,
        [((1, 2), torch.float32)],
        {"model_names": ["pt_opt_facebook_opt_125m_seq_cls_hf"], "pcc": 0.99, "args": {"shape": "(1, 2)"}},
    ),
    (
        Reshape643,
        [((1, 12, 2048), torch.float32)],
        {"model_names": ["pt_phi1_microsoft_phi_1_token_cls_hf"], "pcc": 0.99, "args": {"shape": "(12, 2048)"}},
    ),
    (
        Reshape646,
        [((1, 12, 2048), torch.float32)],
        {"model_names": ["pt_phi1_microsoft_phi_1_token_cls_hf"], "pcc": 0.99, "args": {"shape": "(1, 12, 32, 64)"}},
    ),
    (
        Reshape649,
        [((12, 2048), torch.float32)],
        {"model_names": ["pt_phi1_microsoft_phi_1_token_cls_hf"], "pcc": 0.99, "args": {"shape": "(1, 12, 2048)"}},
    ),
    (
        Reshape650,
        [((32, 12, 12), torch.float32)],
        {"model_names": ["pt_phi1_microsoft_phi_1_token_cls_hf"], "pcc": 0.99, "args": {"shape": "(1, 32, 12, 12)"}},
    ),
    (
        Reshape651,
        [((1, 32, 12, 12), torch.float32)],
        {"model_names": ["pt_phi1_microsoft_phi_1_token_cls_hf"], "pcc": 0.99, "args": {"shape": "(32, 12, 12)"}},
    ),
    (
        Reshape640,
        [((32, 12, 64), torch.float32)],
        {"model_names": ["pt_phi1_microsoft_phi_1_token_cls_hf"], "pcc": 0.99, "args": {"shape": "(1, 32, 12, 64)"}},
    ),
    (
        Reshape652,
        [((12, 8192), torch.float32)],
        {"model_names": ["pt_phi1_microsoft_phi_1_token_cls_hf"], "pcc": 0.99, "args": {"shape": "(1, 12, 8192)"}},
    ),
    (
        Reshape653,
        [((1, 513, 512), torch.float32)],
        {"model_names": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99, "args": {"shape": "(513, 512)"}},
    ),
    (
        Reshape654,
        [((513, 512), torch.float32)],
        {"model_names": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99, "args": {"shape": "(1, 513, 8, 64)"}},
    ),
    (
        Reshape655,
        [((513, 512), torch.float32)],
        {"model_names": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99, "args": {"shape": "(1, 513, 512)"}},
    ),
    (
        Reshape656,
        [((1, 8, 513, 64), torch.float32)],
        {"model_names": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99, "args": {"shape": "(8, 513, 64)"}},
    ),
    (
        Reshape657,
        [((8, 513, 513), torch.float32)],
        {"model_names": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99, "args": {"shape": "(1, 8, 513, 513)"}},
    ),
    (
        Reshape658,
        [((1, 8, 513, 513), torch.float32)],
        {"model_names": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99, "args": {"shape": "(8, 513, 513)"}},
    ),
    (
        Reshape659,
        [((1, 8, 64, 513), torch.float32)],
        {"model_names": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99, "args": {"shape": "(8, 64, 513)"}},
    ),
    (
        Reshape660,
        [((8, 513, 64), torch.float32)],
        {"model_names": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99, "args": {"shape": "(1, 8, 513, 64)"}},
    ),
    (
        Reshape653,
        [((1, 513, 8, 64), torch.float32)],
        {"model_names": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99, "args": {"shape": "(513, 512)"}},
    ),
    (
        Reshape661,
        [((1, 61, 512), torch.float32)],
        {"model_names": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99, "args": {"shape": "(61, 512)"}},
    ),
    (
        Reshape662,
        [((61, 512), torch.float32)],
        {"model_names": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99, "args": {"shape": "(1, 61, 8, 64)"}},
    ),
    (
        Reshape663,
        [((61, 512), torch.float32)],
        {"model_names": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99, "args": {"shape": "(1, 61, 512)"}},
    ),
    (
        Reshape664,
        [((1, 8, 61, 64), torch.float32)],
        {"model_names": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99, "args": {"shape": "(8, 61, 64)"}},
    ),
    (
        Reshape665,
        [((8, 61, 61), torch.float32)],
        {"model_names": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99, "args": {"shape": "(1, 8, 61, 61)"}},
    ),
    (
        Reshape666,
        [((1, 8, 61, 61), torch.float32)],
        {"model_names": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99, "args": {"shape": "(8, 61, 61)"}},
    ),
    (
        Reshape667,
        [((1, 8, 64, 61), torch.float32)],
        {"model_names": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99, "args": {"shape": "(8, 64, 61)"}},
    ),
    (
        Reshape668,
        [((8, 61, 64), torch.float32)],
        {"model_names": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99, "args": {"shape": "(1, 8, 61, 64)"}},
    ),
    (
        Reshape661,
        [((1, 61, 8, 64), torch.float32)],
        {"model_names": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99, "args": {"shape": "(61, 512)"}},
    ),
    (
        Reshape669,
        [((8, 513, 61), torch.float32)],
        {"model_names": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99, "args": {"shape": "(1, 8, 513, 61)"}},
    ),
    (
        Reshape670,
        [((1, 8, 513, 61), torch.float32)],
        {"model_names": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99, "args": {"shape": "(8, 513, 61)"}},
    ),
]


@pytest.mark.nightly_models_ops
@pytest.mark.parametrize("forge_module_and_shapes_dtypes", forge_modules_and_shapes_dtypes_list, ids=ids_func)
def test_module(forge_module_and_shapes_dtypes):

    record_forge_op_name("Reshape")

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
