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


class Reshape0(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 4, 1))
        return reshape_output_1


class Reshape1(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 1))
        return reshape_output_1


class Reshape2(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2048))
        return reshape_output_1


class Reshape3(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 2048))
        return reshape_output_1


class Reshape4(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 2048))
        return reshape_output_1


class Reshape5(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 1, 32, 64))
        return reshape_output_1


class Reshape6(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 1, 2048))
        return reshape_output_1


class Reshape7(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 1, 64))
        return reshape_output_1


class Reshape8(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 32, 1, 64))
        return reshape_output_1


class Reshape9(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 13))
        return reshape_output_1


class Reshape10(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(26, 768))
        return reshape_output_1


class Reshape11(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 13, 12, 64))
        return reshape_output_1


class Reshape12(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 13, 768))
        return reshape_output_1


class Reshape13(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(24, 13, 64))
        return reshape_output_1


class Reshape14(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 12, 13, 13))
        return reshape_output_1


class Reshape15(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(24, 13, 13))
        return reshape_output_1


class Reshape16(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(24, 64, 13))
        return reshape_output_1


class Reshape17(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 12, 13, 64))
        return reshape_output_1


class Reshape18(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 13, 3072))
        return reshape_output_1


class Reshape19(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(26, 3072))
        return reshape_output_1


class Reshape20(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 13, 2048))
        return reshape_output_1


class Reshape21(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 13, 32, 64))
        return reshape_output_1


class Reshape22(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(26, 2048))
        return reshape_output_1


class Reshape23(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 13, 64))
        return reshape_output_1


class Reshape24(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 32, 1, 13))
        return reshape_output_1


class Reshape25(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 1, 13))
        return reshape_output_1


class Reshape26(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 1, 8192))
        return reshape_output_1


class Reshape27(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 8192))
        return reshape_output_1


class Reshape28(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 1, 2048))
        return reshape_output_1


class Reshape29(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1024))
        return reshape_output_1


class Reshape30(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 1024))
        return reshape_output_1


class Reshape31(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 16, 64))
        return reshape_output_1


class Reshape32(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 1024))
        return reshape_output_1


class Reshape33(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 1, 16, 64))
        return reshape_output_1


class Reshape34(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 1, 1024))
        return reshape_output_1


class Reshape35(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 1, 64))
        return reshape_output_1


class Reshape36(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 16, 1, 64))
        return reshape_output_1


class Reshape37(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 13, 1024))
        return reshape_output_1


class Reshape38(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 13, 16, 64))
        return reshape_output_1


class Reshape39(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(26, 1024))
        return reshape_output_1


class Reshape40(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 13, 64))
        return reshape_output_1


class Reshape41(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 16, 1, 13))
        return reshape_output_1


class Reshape42(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 1, 13))
        return reshape_output_1


class Reshape43(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 1, 4096))
        return reshape_output_1


class Reshape44(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 4096))
        return reshape_output_1


class Reshape45(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1536))
        return reshape_output_1


class Reshape46(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 1536))
        return reshape_output_1


class Reshape47(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 1, 24, 64))
        return reshape_output_1


class Reshape48(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 1, 1536))
        return reshape_output_1


class Reshape49(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(48, 1, 64))
        return reshape_output_1


class Reshape50(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 24, 1, 64))
        return reshape_output_1


class Reshape51(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 13, 1536))
        return reshape_output_1


class Reshape52(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 13, 24, 64))
        return reshape_output_1


class Reshape53(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(26, 1536))
        return reshape_output_1


class Reshape54(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(48, 13, 64))
        return reshape_output_1


class Reshape55(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 24, 1, 13))
        return reshape_output_1


class Reshape56(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(48, 1, 13))
        return reshape_output_1


class Reshape57(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 1, 6144))
        return reshape_output_1


class Reshape58(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 6144))
        return reshape_output_1


class Reshape59(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1))
        return reshape_output_1


class Reshape60(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 384))
        return reshape_output_1


class Reshape61(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 6, 64))
        return reshape_output_1


class Reshape62(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 384))
        return reshape_output_1


class Reshape63(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(6, 1, 64))
        return reshape_output_1


class Reshape64(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 6, 1, 64))
        return reshape_output_1


class Reshape65(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 6, 1, 1))
        return reshape_output_1


class Reshape66(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(6, 1, 1))
        return reshape_output_1


class Reshape67(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(6, 64, 1))
        return reshape_output_1


class Reshape68(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 6, 64))
        return reshape_output_1


class Reshape69(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 80, 3000, 1))
        return reshape_output_1


class Reshape70(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(384, 80, 3, 1))
        return reshape_output_1


class Reshape71(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 384, 3000))
        return reshape_output_1


class Reshape72(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 384, 3000, 1))
        return reshape_output_1


class Reshape73(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(384, 384, 3, 1))
        return reshape_output_1


class Reshape74(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 384, 1500))
        return reshape_output_1


class Reshape75(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1500, 384))
        return reshape_output_1


class Reshape76(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1500, 6, 64))
        return reshape_output_1


class Reshape77(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1500, 384))
        return reshape_output_1


class Reshape78(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(6, 1500, 64))
        return reshape_output_1


class Reshape79(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 6, 1500, 1500))
        return reshape_output_1


class Reshape80(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(6, 1500, 1500))
        return reshape_output_1


class Reshape81(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(6, 64, 1500))
        return reshape_output_1


class Reshape82(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 6, 1500, 64))
        return reshape_output_1


class Reshape83(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 6, 1, 1500))
        return reshape_output_1


class Reshape84(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(6, 1, 1500))
        return reshape_output_1


class Reshape85(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 768))
        return reshape_output_1


class Reshape86(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 12, 64))
        return reshape_output_1


class Reshape87(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 768))
        return reshape_output_1


class Reshape88(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 1, 64))
        return reshape_output_1


class Reshape89(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 1, 1))
        return reshape_output_1


class Reshape90(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 1, 1))
        return reshape_output_1


class Reshape91(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 64, 1))
        return reshape_output_1


class Reshape92(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 1, 64))
        return reshape_output_1


class Reshape93(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(768, 80, 3, 1))
        return reshape_output_1


class Reshape94(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 768, 3000))
        return reshape_output_1


class Reshape95(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 768, 3000, 1))
        return reshape_output_1


class Reshape96(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(768, 768, 3, 1))
        return reshape_output_1


class Reshape97(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 768, 1500))
        return reshape_output_1


class Reshape98(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1500, 768))
        return reshape_output_1


class Reshape99(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1500, 12, 64))
        return reshape_output_1


class Reshape100(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1500, 768))
        return reshape_output_1


class Reshape101(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 1500, 64))
        return reshape_output_1


class Reshape102(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 1500, 1500))
        return reshape_output_1


class Reshape103(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 1500, 1500))
        return reshape_output_1


class Reshape104(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 64, 1500))
        return reshape_output_1


class Reshape105(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 1500, 64))
        return reshape_output_1


class Reshape106(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 1, 1500))
        return reshape_output_1


class Reshape107(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 1, 1500))
        return reshape_output_1


class Reshape108(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 1, 1024))
        return reshape_output_1


class Reshape109(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 1, 64))
        return reshape_output_1


class Reshape110(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 1, 1))
        return reshape_output_1


class Reshape111(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 1, 1))
        return reshape_output_1


class Reshape112(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 4))
        return reshape_output_1


class Reshape113(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 64, 1))
        return reshape_output_1


class Reshape114(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 1, 64))
        return reshape_output_1


class Reshape115(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1024, 80, 3, 1))
        return reshape_output_1


class Reshape116(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1024, 3000))
        return reshape_output_1


class Reshape117(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1024, 3000, 1))
        return reshape_output_1


class Reshape118(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1024, 1024, 3, 1))
        return reshape_output_1


class Reshape119(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1024, 1500))
        return reshape_output_1


class Reshape120(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1500, 1024))
        return reshape_output_1


class Reshape121(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1500, 16, 64))
        return reshape_output_1


class Reshape122(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1500, 1024))
        return reshape_output_1


class Reshape123(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 1500, 64))
        return reshape_output_1


class Reshape124(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 1500, 1500))
        return reshape_output_1


class Reshape125(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 1500, 1500))
        return reshape_output_1


class Reshape126(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 64, 1500))
        return reshape_output_1


class Reshape127(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 1500, 64))
        return reshape_output_1


class Reshape128(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 1, 1500))
        return reshape_output_1


class Reshape129(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 1, 1500))
        return reshape_output_1


class Reshape130(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1280))
        return reshape_output_1


class Reshape131(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 20, 64))
        return reshape_output_1


class Reshape132(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 1280))
        return reshape_output_1


class Reshape133(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(20, 1, 64))
        return reshape_output_1


class Reshape134(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 20, 1, 1))
        return reshape_output_1


class Reshape135(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(20, 1, 1))
        return reshape_output_1


class Reshape136(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(20, 64, 1))
        return reshape_output_1


class Reshape137(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 20, 1, 64))
        return reshape_output_1


class Reshape138(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1280, 80, 3, 1))
        return reshape_output_1


class Reshape139(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1280, 3000))
        return reshape_output_1


class Reshape140(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1280, 3000, 1))
        return reshape_output_1


class Reshape141(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1280, 1280, 3, 1))
        return reshape_output_1


class Reshape142(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1280, 1500))
        return reshape_output_1


class Reshape143(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1500, 1280))
        return reshape_output_1


class Reshape144(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1500, 20, 64))
        return reshape_output_1


class Reshape145(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1500, 1280))
        return reshape_output_1


class Reshape146(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(20, 1500, 64))
        return reshape_output_1


class Reshape147(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 20, 1500, 1500))
        return reshape_output_1


class Reshape148(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(20, 1500, 1500))
        return reshape_output_1


class Reshape149(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(20, 64, 1500))
        return reshape_output_1


class Reshape150(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 20, 1500, 64))
        return reshape_output_1


class Reshape151(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 20, 1, 1500))
        return reshape_output_1


class Reshape152(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(20, 1, 1500))
        return reshape_output_1


class Reshape153(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 512))
        return reshape_output_1


class Reshape154(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 8, 64))
        return reshape_output_1


class Reshape155(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 1, 512))
        return reshape_output_1


class Reshape156(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 512))
        return reshape_output_1


class Reshape157(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 1, 64))
        return reshape_output_1


class Reshape158(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 1, 1))
        return reshape_output_1


class Reshape159(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 1, 1))
        return reshape_output_1


class Reshape160(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 64, 1))
        return reshape_output_1


class Reshape161(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 1, 64))
        return reshape_output_1


class Reshape162(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(512, 80, 3, 1))
        return reshape_output_1


class Reshape163(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 512, 3000))
        return reshape_output_1


class Reshape164(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 512, 3000, 1))
        return reshape_output_1


class Reshape165(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(512, 512, 3, 1))
        return reshape_output_1


class Reshape166(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 512, 1500))
        return reshape_output_1


class Reshape167(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1500, 512))
        return reshape_output_1


class Reshape168(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1500, 8, 64))
        return reshape_output_1


class Reshape169(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1500, 512))
        return reshape_output_1


class Reshape170(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 1500, 64))
        return reshape_output_1


class Reshape171(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 1500, 1500))
        return reshape_output_1


class Reshape172(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 1500, 1500))
        return reshape_output_1


class Reshape173(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 64, 1500))
        return reshape_output_1


class Reshape174(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 1500, 64))
        return reshape_output_1


class Reshape175(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 1, 1500))
        return reshape_output_1


class Reshape176(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 1, 1500))
        return reshape_output_1


class Reshape177(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2))
        return reshape_output_1


class Reshape178(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 1280))
        return reshape_output_1


class Reshape179(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2, 20, 64))
        return reshape_output_1


class Reshape180(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2, 1280))
        return reshape_output_1


class Reshape181(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(20, 2, 64))
        return reshape_output_1


class Reshape182(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 20, 2, 2))
        return reshape_output_1


class Reshape183(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(20, 2, 2))
        return reshape_output_1


class Reshape184(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(20, 64, 2))
        return reshape_output_1


class Reshape185(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 20, 2, 64))
        return reshape_output_1


class Reshape186(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 20, 2, 1500))
        return reshape_output_1


class Reshape187(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(20, 2, 1500))
        return reshape_output_1


class Reshape188(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 7))
        return reshape_output_1


class Reshape189(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(14, 512))
        return reshape_output_1


class Reshape190(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 7, 8, 64))
        return reshape_output_1


class Reshape191(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 7, 512))
        return reshape_output_1


class Reshape192(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 7, 64))
        return reshape_output_1


class Reshape193(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 8, 7, 7))
        return reshape_output_1


class Reshape194(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 7, 7))
        return reshape_output_1


class Reshape195(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 8, 7, 64))
        return reshape_output_1


class Reshape196(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 7, 2048))
        return reshape_output_1


class Reshape197(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(14, 2048))
        return reshape_output_1


class Reshape198(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(588, 2048))
        return reshape_output_1


class Reshape199(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 588, 16, 128))
        return reshape_output_1


class Reshape200(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 588, 2048))
        return reshape_output_1


class Reshape201(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 588, 128))
        return reshape_output_1


class Reshape202(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 588, 588))
        return reshape_output_1


class Reshape203(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 588, 588))
        return reshape_output_1


class Reshape204(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 128, 588))
        return reshape_output_1


class Reshape205(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 588, 128))
        return reshape_output_1


class Reshape206(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 588, 5504))
        return reshape_output_1


class Reshape207(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(39, 4096))
        return reshape_output_1


class Reshape208(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 39, 32, 128))
        return reshape_output_1


class Reshape209(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 39, 4096))
        return reshape_output_1


class Reshape210(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 39, 128))
        return reshape_output_1


class Reshape211(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 39, 39))
        return reshape_output_1


class Reshape212(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 39, 39))
        return reshape_output_1


class Reshape213(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 128, 39))
        return reshape_output_1


class Reshape214(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 39, 128))
        return reshape_output_1


class Reshape215(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 39, 11008))
        return reshape_output_1


class Reshape216(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2441216,))
        return reshape_output_1


class Reshape217(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(596, 4096))
        return reshape_output_1


class Reshape218(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1024, 576, 1))
        return reshape_output_1


class Reshape219(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(577, 1024))
        return reshape_output_1


class Reshape220(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 577, 16, 64))
        return reshape_output_1


class Reshape221(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 577, 1024))
        return reshape_output_1


class Reshape222(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 577, 64))
        return reshape_output_1


class Reshape223(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 577, 64))
        return reshape_output_1


class Reshape224(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2359296,))
        return reshape_output_1


class Reshape225(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 596, 4096))
        return reshape_output_1


class Reshape226(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 596, 32, 128))
        return reshape_output_1


class Reshape227(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 596, 128))
        return reshape_output_1


class Reshape228(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 596, 596))
        return reshape_output_1


class Reshape229(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 596, 596))
        return reshape_output_1


class Reshape230(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 128, 596))
        return reshape_output_1


class Reshape231(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 596, 128))
        return reshape_output_1


class Reshape232(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 596, 11008))
        return reshape_output_1


class Reshape233(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(204, 768))
        return reshape_output_1


class Reshape234(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 204, 12, 64))
        return reshape_output_1


class Reshape235(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 204, 768))
        return reshape_output_1


class Reshape236(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 204, 64))
        return reshape_output_1


class Reshape237(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 204, 204))
        return reshape_output_1


class Reshape238(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 204, 204))
        return reshape_output_1


class Reshape239(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 64, 204))
        return reshape_output_1


class Reshape240(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 204, 64))
        return reshape_output_1


class Reshape241(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(201, 768))
        return reshape_output_1


class Reshape242(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 201, 12, 64))
        return reshape_output_1


class Reshape243(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 201, 768))
        return reshape_output_1


class Reshape244(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 201, 64))
        return reshape_output_1


class Reshape245(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 201, 201))
        return reshape_output_1


class Reshape246(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 201, 201))
        return reshape_output_1


class Reshape247(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 64, 201))
        return reshape_output_1


class Reshape248(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 201, 64))
        return reshape_output_1


class Reshape249(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(128, 2048))
        return reshape_output_1


class Reshape250(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 16, 128))
        return reshape_output_1


class Reshape251(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 2048))
        return reshape_output_1


class Reshape252(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 128, 128))
        return reshape_output_1


class Reshape253(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 128, 128))
        return reshape_output_1


class Reshape254(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 2048, 1))
        return reshape_output_1


class Reshape255(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(128, 1024))
        return reshape_output_1


class Reshape256(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 16, 64))
        return reshape_output_1


class Reshape257(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 1024))
        return reshape_output_1


class Reshape258(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 8, 128))
        return reshape_output_1


class Reshape259(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 128, 64))
        return reshape_output_1


class Reshape260(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 64, 128))
        return reshape_output_1


class Reshape261(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 128, 64))
        return reshape_output_1


class Reshape262(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 1024, 1))
        return reshape_output_1


class Reshape263(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(128, 4096))
        return reshape_output_1


class Reshape264(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 64, 64))
        return reshape_output_1


class Reshape265(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 4096))
        return reshape_output_1


class Reshape266(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 32, 128))
        return reshape_output_1


class Reshape267(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 128, 64))
        return reshape_output_1


class Reshape268(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 64, 128, 128))
        return reshape_output_1


class Reshape269(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 128, 128))
        return reshape_output_1


class Reshape270(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 64, 16384, 1))
        return reshape_output_1


class Reshape271(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 64, 128))
        return reshape_output_1


class Reshape272(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 64, 128, 64))
        return reshape_output_1


class Reshape273(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 4096, 1))
        return reshape_output_1


class Reshape274(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(128, 768))
        return reshape_output_1


class Reshape275(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 12, 64))
        return reshape_output_1


class Reshape276(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 768))
        return reshape_output_1


class Reshape277(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 128, 64))
        return reshape_output_1


class Reshape278(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 128, 128))
        return reshape_output_1


class Reshape279(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 128, 128))
        return reshape_output_1


class Reshape280(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 64, 128))
        return reshape_output_1


class Reshape281(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 768, 128, 1))
        return reshape_output_1


class Reshape282(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 128, 64))
        return reshape_output_1


class Reshape283(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 768, 1))
        return reshape_output_1


class Reshape284(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(9, 768))
        return reshape_output_1


class Reshape285(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 9, 12, 64))
        return reshape_output_1


class Reshape286(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 9, 768))
        return reshape_output_1


class Reshape287(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 9, 64))
        return reshape_output_1


class Reshape288(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 9, 9))
        return reshape_output_1


class Reshape289(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 9, 9))
        return reshape_output_1


class Reshape290(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 64, 9))
        return reshape_output_1


class Reshape291(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 9, 64))
        return reshape_output_1


class Reshape292(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 9, 768, 1))
        return reshape_output_1


class Reshape293(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(14, 768))
        return reshape_output_1


class Reshape294(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 14, 12, 64))
        return reshape_output_1


class Reshape295(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 14, 768))
        return reshape_output_1


class Reshape296(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 14, 64))
        return reshape_output_1


class Reshape297(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 14, 14))
        return reshape_output_1


class Reshape298(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 14, 14))
        return reshape_output_1


class Reshape299(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 64, 14))
        return reshape_output_1


class Reshape300(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 14, 64))
        return reshape_output_1


class Reshape301(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 14, 768, 1))
        return reshape_output_1


class Reshape302(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 14, 1))
        return reshape_output_1


class Reshape303(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(256, 1024))
        return reshape_output_1


class Reshape304(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 16, 64))
        return reshape_output_1


class Reshape305(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 32, 32))
        return reshape_output_1


class Reshape306(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 1024))
        return reshape_output_1


class Reshape307(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 4, 256))
        return reshape_output_1


class Reshape308(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 8, 128))
        return reshape_output_1


class Reshape309(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 256, 64))
        return reshape_output_1


class Reshape310(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 256, 256))
        return reshape_output_1


class Reshape311(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 256, 256))
        return reshape_output_1


class Reshape312(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 256, 64))
        return reshape_output_1


class Reshape313(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256))
        return reshape_output_1


class Reshape314(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(384, 1024))
        return reshape_output_1


class Reshape315(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 384, 16, 64))
        return reshape_output_1


class Reshape316(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 384, 1024))
        return reshape_output_1


class Reshape317(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 384, 64))
        return reshape_output_1


class Reshape318(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 384, 384))
        return reshape_output_1


class Reshape319(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 384, 384))
        return reshape_output_1


class Reshape320(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 64, 384))
        return reshape_output_1


class Reshape321(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 384, 64))
        return reshape_output_1


class Reshape322(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 384, 1))
        return reshape_output_1


class Reshape323(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(6, 768))
        return reshape_output_1


class Reshape324(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 6, 12, 64))
        return reshape_output_1


class Reshape325(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 6, 768))
        return reshape_output_1


class Reshape326(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 6, 64))
        return reshape_output_1


class Reshape327(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 6, 6))
        return reshape_output_1


class Reshape328(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 6, 6))
        return reshape_output_1


class Reshape329(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 64, 6))
        return reshape_output_1


class Reshape330(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 6, 64))
        return reshape_output_1


class Reshape331(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 16, 3, 96))
        return reshape_output_1


class Reshape332(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 16, 96))
        return reshape_output_1


class Reshape333(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 32, 96))
        return reshape_output_1


class Reshape334(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 1, 32))
        return reshape_output_1


class Reshape335(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 32, 32))
        return reshape_output_1


class Reshape336(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 32, 32))
        return reshape_output_1


class Reshape337(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 32, 96))
        return reshape_output_1


class Reshape338(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 1536))
        return reshape_output_1


class Reshape339(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 1536))
        return reshape_output_1


class Reshape340(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 16, 32, 1))
        return reshape_output_1


class Reshape341(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 64, 256))
        return reshape_output_1


class Reshape342(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 4096))
        return reshape_output_1


class Reshape343(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 32, 128))
        return reshape_output_1


class Reshape344(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 1, 128))
        return reshape_output_1


class Reshape345(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(384, 768))
        return reshape_output_1


class Reshape346(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 384, 12, 64))
        return reshape_output_1


class Reshape347(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 384, 768))
        return reshape_output_1


class Reshape348(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 384, 64))
        return reshape_output_1


class Reshape349(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 384, 384))
        return reshape_output_1


class Reshape350(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 1, 384))
        return reshape_output_1


class Reshape351(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 384, 384))
        return reshape_output_1


class Reshape352(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 64, 384))
        return reshape_output_1


class Reshape353(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 384, 64))
        return reshape_output_1


class Reshape354(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 1))
        return reshape_output_1


class Reshape355(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128))
        return reshape_output_1


class Reshape356(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1,))
        return reshape_output_1


class Reshape357(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(10, 3072))
        return reshape_output_1


class Reshape358(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 10, 12, 256))
        return reshape_output_1


class Reshape359(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 10, 3072))
        return reshape_output_1


class Reshape360(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 10, 256))
        return reshape_output_1


class Reshape361(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 10, 4, 256))
        return reshape_output_1


class Reshape362(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 10, 256))
        return reshape_output_1


class Reshape363(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 10, 10))
        return reshape_output_1


class Reshape364(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 10, 10))
        return reshape_output_1


class Reshape365(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 256, 10))
        return reshape_output_1


class Reshape366(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 10, 23040))
        return reshape_output_1


class Reshape367(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(10, 2048))
        return reshape_output_1


class Reshape368(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 10, 8, 256))
        return reshape_output_1


class Reshape369(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 10, 2048))
        return reshape_output_1


class Reshape370(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 10, 256))
        return reshape_output_1


class Reshape371(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 10, 256))
        return reshape_output_1


class Reshape372(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 10, 10))
        return reshape_output_1


class Reshape373(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 10, 10))
        return reshape_output_1


class Reshape374(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 256, 10))
        return reshape_output_1


class Reshape375(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 10, 8192))
        return reshape_output_1


class Reshape376(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 10, 9216))
        return reshape_output_1


class Reshape377(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(6, 4544))
        return reshape_output_1


class Reshape378(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 6, 18176))
        return reshape_output_1


class Reshape379(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 6, 73, 64))
        return reshape_output_1


class Reshape380(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 71, 6, 64))
        return reshape_output_1


class Reshape381(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(71, 6, 64))
        return reshape_output_1


class Reshape382(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 71, 6, 6))
        return reshape_output_1


class Reshape383(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(71, 6, 6))
        return reshape_output_1


class Reshape384(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 6, 4544))
        return reshape_output_1


class Reshape385(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 334, 64, 3, 64))
        return reshape_output_1


class Reshape386(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 334, 64, 64))
        return reshape_output_1


class Reshape387(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 334, 64))
        return reshape_output_1


class Reshape388(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 64, 334, 334))
        return reshape_output_1


class Reshape389(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 334, 334))
        return reshape_output_1


class Reshape390(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 64, 334))
        return reshape_output_1


class Reshape391(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 64, 334, 64))
        return reshape_output_1


class Reshape392(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(334, 4096))
        return reshape_output_1


class Reshape393(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 334, 4096))
        return reshape_output_1


class Reshape394(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(7, 2048))
        return reshape_output_1


class Reshape395(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 7, 8, 256))
        return reshape_output_1


class Reshape396(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 7, 2048))
        return reshape_output_1


class Reshape397(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 7, 256))
        return reshape_output_1


class Reshape398(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 7, 1, 256))
        return reshape_output_1


class Reshape399(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 7, 256))
        return reshape_output_1


class Reshape400(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 7, 7))
        return reshape_output_1


class Reshape401(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 7, 7))
        return reshape_output_1


class Reshape402(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 256, 7))
        return reshape_output_1


class Reshape403(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 7, 16384))
        return reshape_output_1


class Reshape404(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(207, 2304))
        return reshape_output_1


class Reshape405(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 207, 8, 256))
        return reshape_output_1


class Reshape406(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 207, 256))
        return reshape_output_1


class Reshape407(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 207, 4, 256))
        return reshape_output_1


class Reshape408(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 207, 256))
        return reshape_output_1


class Reshape409(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 207, 207))
        return reshape_output_1


class Reshape410(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 207, 207))
        return reshape_output_1


class Reshape411(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 256, 207))
        return reshape_output_1


class Reshape412(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(207, 2048))
        return reshape_output_1


class Reshape413(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 207, 2304))
        return reshape_output_1


class Reshape414(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 207, 9216))
        return reshape_output_1


class Reshape415(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(207, 3584))
        return reshape_output_1


class Reshape416(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 207, 16, 256))
        return reshape_output_1


class Reshape417(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 207, 256))
        return reshape_output_1


class Reshape418(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 207, 256))
        return reshape_output_1


class Reshape419(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 207, 207))
        return reshape_output_1


class Reshape420(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 207, 207))
        return reshape_output_1


class Reshape421(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 256, 207))
        return reshape_output_1


class Reshape422(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(207, 4096))
        return reshape_output_1


class Reshape423(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 207, 3584))
        return reshape_output_1


class Reshape424(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 207, 14336))
        return reshape_output_1


class Reshape425(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(107, 2048))
        return reshape_output_1


class Reshape426(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 107, 8, 256))
        return reshape_output_1


class Reshape427(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 107, 2048))
        return reshape_output_1


class Reshape428(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 107, 256))
        return reshape_output_1


class Reshape429(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 107, 1, 256))
        return reshape_output_1


class Reshape430(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 107, 256))
        return reshape_output_1


class Reshape431(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 107, 107))
        return reshape_output_1


class Reshape432(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 107, 107))
        return reshape_output_1


class Reshape433(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 256, 107))
        return reshape_output_1


class Reshape434(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 107, 16384))
        return reshape_output_1


class Reshape435(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(107, 3072))
        return reshape_output_1


class Reshape436(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 107, 16, 256))
        return reshape_output_1


class Reshape437(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 107, 256))
        return reshape_output_1


class Reshape438(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 107, 107))
        return reshape_output_1


class Reshape439(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 107, 107))
        return reshape_output_1


class Reshape440(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 256, 107))
        return reshape_output_1


class Reshape441(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 107, 256))
        return reshape_output_1


class Reshape442(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(107, 4096))
        return reshape_output_1


class Reshape443(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 107, 3072))
        return reshape_output_1


class Reshape444(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 107, 24576))
        return reshape_output_1


class Reshape445(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(256, 768))
        return reshape_output_1


class Reshape446(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 12, 64))
        return reshape_output_1


class Reshape447(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 8, 96))
        return reshape_output_1


class Reshape448(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 768))
        return reshape_output_1


class Reshape449(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 256, 64))
        return reshape_output_1


class Reshape450(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 256, 256))
        return reshape_output_1


class Reshape451(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 256, 256))
        return reshape_output_1


class Reshape452(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 64, 256))
        return reshape_output_1


class Reshape453(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 256, 64))
        return reshape_output_1


class Reshape454(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 3072))
        return reshape_output_1


class Reshape455(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(256, 3072))
        return reshape_output_1


class Reshape456(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 32, 96))
        return reshape_output_1


class Reshape457(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 7))
        return reshape_output_1


class Reshape458(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(7, 768))
        return reshape_output_1


class Reshape459(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 7, 12, 64))
        return reshape_output_1


class Reshape460(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 7, 768))
        return reshape_output_1


class Reshape461(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 7, 64))
        return reshape_output_1


class Reshape462(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 7, 7))
        return reshape_output_1


class Reshape463(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 7, 7))
        return reshape_output_1


class Reshape464(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 64, 7))
        return reshape_output_1


class Reshape465(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 7, 64))
        return reshape_output_1


class Reshape466(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 7, 3072))
        return reshape_output_1


class Reshape467(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(7, 3072))
        return reshape_output_1


class Reshape468(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(7, 2))
        return reshape_output_1


class Reshape469(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 2560))
        return reshape_output_1


class Reshape470(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 20, 128))
        return reshape_output_1


class Reshape471(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 2560))
        return reshape_output_1


class Reshape472(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(20, 32, 128))
        return reshape_output_1


class Reshape473(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 20, 32, 32))
        return reshape_output_1


class Reshape474(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(20, 32, 32))
        return reshape_output_1


class Reshape475(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(20, 128, 32))
        return reshape_output_1


class Reshape476(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 20, 32, 128))
        return reshape_output_1


class Reshape477(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 768))
        return reshape_output_1


class Reshape478(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 12, 64))
        return reshape_output_1


class Reshape479(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 768))
        return reshape_output_1


class Reshape480(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 32, 64))
        return reshape_output_1


class Reshape481(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 32, 32))
        return reshape_output_1


class Reshape482(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 32, 32))
        return reshape_output_1


class Reshape483(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 64, 32))
        return reshape_output_1


class Reshape484(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 32, 64))
        return reshape_output_1


class Reshape485(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(256, 2560))
        return reshape_output_1


class Reshape486(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 32, 80))
        return reshape_output_1


class Reshape487(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 20, 128))
        return reshape_output_1


class Reshape488(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 2560))
        return reshape_output_1


class Reshape489(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(20, 256, 128))
        return reshape_output_1


class Reshape490(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 20, 256, 256))
        return reshape_output_1


class Reshape491(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(20, 256, 256))
        return reshape_output_1


class Reshape492(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(20, 128, 256))
        return reshape_output_1


class Reshape493(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 20, 256, 128))
        return reshape_output_1


class Reshape494(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(256, 2048))
        return reshape_output_1


class Reshape495(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 32, 64))
        return reshape_output_1


class Reshape496(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 16, 128))
        return reshape_output_1


class Reshape497(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 2048))
        return reshape_output_1


class Reshape498(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 256, 128))
        return reshape_output_1


class Reshape499(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 128, 256))
        return reshape_output_1


class Reshape500(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 256, 128))
        return reshape_output_1


class Reshape501(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 2048))
        return reshape_output_1


class Reshape502(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 32, 64))
        return reshape_output_1


class Reshape503(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 16, 128))
        return reshape_output_1


class Reshape504(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 2048))
        return reshape_output_1


class Reshape505(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 32, 128))
        return reshape_output_1


class Reshape506(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 128, 32))
        return reshape_output_1


class Reshape507(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 32, 128))
        return reshape_output_1


class Reshape508(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(4, 4096))
        return reshape_output_1


class Reshape509(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 32, 128))
        return reshape_output_1


class Reshape510(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 4096))
        return reshape_output_1


class Reshape511(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 4, 128))
        return reshape_output_1


class Reshape512(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 8, 128))
        return reshape_output_1


class Reshape513(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 4, 128))
        return reshape_output_1


class Reshape514(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 4, 4))
        return reshape_output_1


class Reshape515(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 4, 4))
        return reshape_output_1


class Reshape516(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 128, 4))
        return reshape_output_1


class Reshape517(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 14336))
        return reshape_output_1


class Reshape518(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(4, 2048))
        return reshape_output_1


class Reshape519(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 32, 64))
        return reshape_output_1


class Reshape520(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 2048))
        return reshape_output_1


class Reshape521(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 4, 64))
        return reshape_output_1


class Reshape522(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 8, 64))
        return reshape_output_1


class Reshape523(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 4, 64))
        return reshape_output_1


class Reshape524(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 64, 4))
        return reshape_output_1


class Reshape525(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 8192))
        return reshape_output_1


class Reshape526(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 11008))
        return reshape_output_1


class Reshape527(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(4, 2))
        return reshape_output_1


class Reshape528(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(4, 3072))
        return reshape_output_1


class Reshape529(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 24, 128))
        return reshape_output_1


class Reshape530(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 3072))
        return reshape_output_1


class Reshape531(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(24, 4, 128))
        return reshape_output_1


class Reshape532(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 24, 4, 128))
        return reshape_output_1


class Reshape533(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 24, 4, 4))
        return reshape_output_1


class Reshape534(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(24, 4, 4))
        return reshape_output_1


class Reshape535(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(24, 128, 4))
        return reshape_output_1


class Reshape536(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(256, 4096))
        return reshape_output_1


class Reshape537(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 64, 64))
        return reshape_output_1


class Reshape538(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 256, 128))
        return reshape_output_1


class Reshape539(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 256, 128))
        return reshape_output_1


class Reshape540(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 256, 256))
        return reshape_output_1


class Reshape541(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 256, 256))
        return reshape_output_1


class Reshape542(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 128, 256))
        return reshape_output_1


class Reshape543(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 14336))
        return reshape_output_1


class Reshape544(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 256, 64))
        return reshape_output_1


class Reshape545(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 8, 64))
        return reshape_output_1


class Reshape546(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(256, 512))
        return reshape_output_1


class Reshape547(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 512))
        return reshape_output_1


class Reshape548(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 256, 64))
        return reshape_output_1


class Reshape549(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 64, 256))
        return reshape_output_1


class Reshape550(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 8192))
        return reshape_output_1


class Reshape551(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 3072))
        return reshape_output_1


class Reshape552(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 24, 128))
        return reshape_output_1


class Reshape553(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 3072))
        return reshape_output_1


class Reshape554(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(24, 32, 128))
        return reshape_output_1


class Reshape555(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 8, 128))
        return reshape_output_1


class Reshape556(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 1024))
        return reshape_output_1


class Reshape557(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 24, 32, 128))
        return reshape_output_1


class Reshape558(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 24, 32, 32))
        return reshape_output_1


class Reshape559(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(24, 32, 32))
        return reshape_output_1


class Reshape560(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(24, 128, 32))
        return reshape_output_1


class Reshape561(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 8192))
        return reshape_output_1


class Reshape562(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 4096))
        return reshape_output_1


class Reshape563(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 32, 128))
        return reshape_output_1


class Reshape564(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 4096))
        return reshape_output_1


class Reshape565(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 32, 128))
        return reshape_output_1


class Reshape566(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 32, 32))
        return reshape_output_1


class Reshape567(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 32, 32))
        return reshape_output_1


class Reshape568(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 128, 32))
        return reshape_output_1


class Reshape569(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 11008))
        return reshape_output_1


class Reshape570(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 128, 128))
        return reshape_output_1


class Reshape571(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 16384, 1))
        return reshape_output_1


class Reshape572(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 128, 128))
        return reshape_output_1


class Reshape573(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 14336))
        return reshape_output_1


class Reshape574(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32))
        return reshape_output_1


class Reshape575(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 2))
        return reshape_output_1


class Reshape576(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 32, 64))
        return reshape_output_1


class Reshape577(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 1024))
        return reshape_output_1


class Reshape578(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 16, 64))
        return reshape_output_1


class Reshape579(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 32, 64))
        return reshape_output_1


class Reshape580(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 32, 64))
        return reshape_output_1


class Reshape581(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 512))
        return reshape_output_1


class Reshape582(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 1))
        return reshape_output_1


class Reshape583(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 50272))
        return reshape_output_1


class Reshape584(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 512, 1, 261))
        return reshape_output_1


class Reshape585(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 50176, 3))
        return reshape_output_1


class Reshape586(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(50176, 261))
        return reshape_output_1


class Reshape587(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 50176, 1, 261))
        return reshape_output_1


class Reshape588(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 50176, 261))
        return reshape_output_1


class Reshape589(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 512, 50176))
        return reshape_output_1


class Reshape590(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 512, 50176))
        return reshape_output_1


class Reshape591(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 261, 50176))
        return reshape_output_1


class Reshape592(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(512, 1024))
        return reshape_output_1


class Reshape593(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 512, 8, 128))
        return reshape_output_1


class Reshape594(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 512, 1, 1024))
        return reshape_output_1


class Reshape595(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 512, 1024))
        return reshape_output_1


class Reshape596(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 512, 128))
        return reshape_output_1


class Reshape597(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 512, 512))
        return reshape_output_1


class Reshape598(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 512, 512))
        return reshape_output_1


class Reshape599(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 128, 512))
        return reshape_output_1


class Reshape600(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 512, 128))
        return reshape_output_1


class Reshape601(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1024, 512))
        return reshape_output_1


class Reshape602(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1000))
        return reshape_output_1


class Reshape603(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 512, 1, 322))
        return reshape_output_1


class Reshape604(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 3025, 64))
        return reshape_output_1


class Reshape605(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(3025, 322))
        return reshape_output_1


class Reshape606(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 3025, 1, 322))
        return reshape_output_1


class Reshape607(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 3025, 322))
        return reshape_output_1


class Reshape608(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 512, 3025))
        return reshape_output_1


class Reshape609(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 512, 3025))
        return reshape_output_1


class Reshape610(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 322, 3025))
        return reshape_output_1


class Reshape611(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 512, 1, 512))
        return reshape_output_1


class Reshape612(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 50176, 256))
        return reshape_output_1


class Reshape613(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(50176, 512))
        return reshape_output_1


class Reshape614(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 50176, 1, 512))
        return reshape_output_1


class Reshape615(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 50176, 512))
        return reshape_output_1


class Reshape616(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2048, 8, 32))
        return reshape_output_1


class Reshape617(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2048, 16, 16))
        return reshape_output_1


class Reshape618(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 2048, 32))
        return reshape_output_1


class Reshape619(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 8, 32))
        return reshape_output_1


class Reshape620(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(256, 256))
        return reshape_output_1


class Reshape621(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 256))
        return reshape_output_1


class Reshape622(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 16, 256))
        return reshape_output_1


class Reshape623(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 16, 16))
        return reshape_output_1


class Reshape624(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 256, 32))
        return reshape_output_1


class Reshape625(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2048, 768))
        return reshape_output_1


class Reshape626(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2048, 256))
        return reshape_output_1


class Reshape627(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 256, 2048))
        return reshape_output_1


class Reshape628(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 256, 2048))
        return reshape_output_1


class Reshape629(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2048, 1280))
        return reshape_output_1


class Reshape630(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2048, 8, 160))
        return reshape_output_1


class Reshape631(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 160, 2048))
        return reshape_output_1


class Reshape632(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 256, 160))
        return reshape_output_1


class Reshape633(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(256, 1280))
        return reshape_output_1


class Reshape634(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 1280))
        return reshape_output_1


class Reshape635(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 8, 160))
        return reshape_output_1


class Reshape636(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 256, 256))
        return reshape_output_1


class Reshape637(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 256, 256))
        return reshape_output_1


class Reshape638(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 160, 256))
        return reshape_output_1


class Reshape639(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 2048, 256))
        return reshape_output_1


class Reshape640(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 2048, 256))
        return reshape_output_1


class Reshape641(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 96, 256))
        return reshape_output_1


class Reshape642(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 2048, 96))
        return reshape_output_1


class Reshape643(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2048, 768))
        return reshape_output_1


class Reshape644(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2048, 262))
        return reshape_output_1


class Reshape645(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 256, 80))
        return reshape_output_1


class Reshape646(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 80, 256))
        return reshape_output_1


class Reshape647(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 256, 80))
        return reshape_output_1


class Reshape648(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 10240))
        return reshape_output_1


class Reshape649(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 2560))
        return reshape_output_1


class Reshape650(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 32, 80))
        return reshape_output_1


class Reshape651(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 2560))
        return reshape_output_1


class Reshape652(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 12, 80))
        return reshape_output_1


class Reshape653(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 12, 12))
        return reshape_output_1


class Reshape654(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 12, 12))
        return reshape_output_1


class Reshape655(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 80, 12))
        return reshape_output_1


class Reshape656(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 12, 80))
        return reshape_output_1


class Reshape657(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 10240))
        return reshape_output_1


class Reshape658(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(11, 2560))
        return reshape_output_1


class Reshape659(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 11, 32, 80))
        return reshape_output_1


class Reshape660(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 11, 2560))
        return reshape_output_1


class Reshape661(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 11, 80))
        return reshape_output_1


class Reshape662(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 11, 11))
        return reshape_output_1


class Reshape663(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 11, 11))
        return reshape_output_1


class Reshape664(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 80, 11))
        return reshape_output_1


class Reshape665(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 11, 80))
        return reshape_output_1


class Reshape666(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 11, 10240))
        return reshape_output_1


class Reshape667(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 256, 96))
        return reshape_output_1


class Reshape668(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 96, 256))
        return reshape_output_1


class Reshape669(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 256, 96))
        return reshape_output_1


class Reshape670(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 5, 32, 96))
        return reshape_output_1


class Reshape671(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(5, 3072))
        return reshape_output_1


class Reshape672(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 5, 96))
        return reshape_output_1


class Reshape673(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 5, 5))
        return reshape_output_1


class Reshape674(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 5, 5))
        return reshape_output_1


class Reshape675(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 96, 5))
        return reshape_output_1


class Reshape676(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 5, 96))
        return reshape_output_1


class Reshape677(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 5, 3072))
        return reshape_output_1


class Reshape678(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 5, 8192))
        return reshape_output_1


class Reshape679(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 13, 32, 96))
        return reshape_output_1


class Reshape680(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(13, 3072))
        return reshape_output_1


class Reshape681(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 13, 96))
        return reshape_output_1


class Reshape682(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 13, 13))
        return reshape_output_1


class Reshape683(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 13, 13))
        return reshape_output_1


class Reshape684(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 96, 13))
        return reshape_output_1


class Reshape685(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 13, 96))
        return reshape_output_1


class Reshape686(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 13, 3072))
        return reshape_output_1


class Reshape687(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 13, 8192))
        return reshape_output_1


class Reshape688(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(6, 1024))
        return reshape_output_1


class Reshape689(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 6, 16, 64))
        return reshape_output_1


class Reshape690(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 6, 1024))
        return reshape_output_1


class Reshape691(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 6, 64))
        return reshape_output_1


class Reshape692(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 6, 6))
        return reshape_output_1


class Reshape693(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 6, 6))
        return reshape_output_1


class Reshape694(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 64, 6))
        return reshape_output_1


class Reshape695(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 6, 64))
        return reshape_output_1


class Reshape696(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 6, 2816))
        return reshape_output_1


class Reshape697(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(29, 1024))
        return reshape_output_1


class Reshape698(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 29, 16, 64))
        return reshape_output_1


class Reshape699(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 29, 1024))
        return reshape_output_1


class Reshape700(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 29, 64))
        return reshape_output_1


class Reshape701(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 29, 29))
        return reshape_output_1


class Reshape702(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 29, 29))
        return reshape_output_1


class Reshape703(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 64, 29))
        return reshape_output_1


class Reshape704(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 29, 64))
        return reshape_output_1


class Reshape705(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 29, 2816))
        return reshape_output_1


class Reshape706(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(35, 2048))
        return reshape_output_1


class Reshape707(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 35, 16, 128))
        return reshape_output_1


class Reshape708(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 35, 2048))
        return reshape_output_1


class Reshape709(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 35, 128))
        return reshape_output_1


class Reshape710(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 35, 256))
        return reshape_output_1


class Reshape711(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 35, 2, 128))
        return reshape_output_1


class Reshape712(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 35, 128))
        return reshape_output_1


class Reshape713(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 35, 35))
        return reshape_output_1


class Reshape714(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 35, 35))
        return reshape_output_1


class Reshape715(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 128, 35))
        return reshape_output_1


class Reshape716(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 35, 11008))
        return reshape_output_1


class Reshape717(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(35, 896))
        return reshape_output_1


class Reshape718(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 35, 14, 64))
        return reshape_output_1


class Reshape719(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 35, 896))
        return reshape_output_1


class Reshape720(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(14, 35, 64))
        return reshape_output_1


class Reshape721(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 35, 128))
        return reshape_output_1


class Reshape722(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 35, 2, 64))
        return reshape_output_1


class Reshape723(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 14, 35, 64))
        return reshape_output_1


class Reshape724(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 14, 35, 35))
        return reshape_output_1


class Reshape725(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(14, 35, 35))
        return reshape_output_1


class Reshape726(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(14, 64, 35))
        return reshape_output_1


class Reshape727(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 35, 4864))
        return reshape_output_1


class Reshape728(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(35, 1536))
        return reshape_output_1


class Reshape729(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 35, 12, 128))
        return reshape_output_1


class Reshape730(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 35, 1536))
        return reshape_output_1


class Reshape731(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 35, 128))
        return reshape_output_1


class Reshape732(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 35, 128))
        return reshape_output_1


class Reshape733(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 35, 35))
        return reshape_output_1


class Reshape734(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 35, 35))
        return reshape_output_1


class Reshape735(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 128, 35))
        return reshape_output_1


class Reshape736(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 35, 8960))
        return reshape_output_1


class Reshape737(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(35, 3584))
        return reshape_output_1


class Reshape738(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 35, 28, 128))
        return reshape_output_1


class Reshape739(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 35, 3584))
        return reshape_output_1


class Reshape740(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(28, 35, 128))
        return reshape_output_1


class Reshape741(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 35, 512))
        return reshape_output_1


class Reshape742(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 35, 4, 128))
        return reshape_output_1


class Reshape743(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 28, 35, 128))
        return reshape_output_1


class Reshape744(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 28, 35, 35))
        return reshape_output_1


class Reshape745(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(28, 35, 35))
        return reshape_output_1


class Reshape746(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(28, 128, 35))
        return reshape_output_1


class Reshape747(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 35, 18944))
        return reshape_output_1


class Reshape748(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(29, 1536))
        return reshape_output_1


class Reshape749(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 29, 12, 128))
        return reshape_output_1


class Reshape750(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 29, 1536))
        return reshape_output_1


class Reshape751(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 29, 128))
        return reshape_output_1


class Reshape752(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 29, 256))
        return reshape_output_1


class Reshape753(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 29, 2, 128))
        return reshape_output_1


class Reshape754(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 29, 128))
        return reshape_output_1


class Reshape755(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 29, 29))
        return reshape_output_1


class Reshape756(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 29, 29))
        return reshape_output_1


class Reshape757(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 128, 29))
        return reshape_output_1


class Reshape758(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 29, 8960))
        return reshape_output_1


class Reshape759(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(39, 1536))
        return reshape_output_1


class Reshape760(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 39, 12, 128))
        return reshape_output_1


class Reshape761(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 39, 1536))
        return reshape_output_1


class Reshape762(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 39, 128))
        return reshape_output_1


class Reshape763(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 39, 256))
        return reshape_output_1


class Reshape764(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 39, 2, 128))
        return reshape_output_1


class Reshape765(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 39, 128))
        return reshape_output_1


class Reshape766(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 39, 39))
        return reshape_output_1


class Reshape767(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 39, 39))
        return reshape_output_1


class Reshape768(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 128, 39))
        return reshape_output_1


class Reshape769(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 39, 8960))
        return reshape_output_1


class Reshape770(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(13, 3584))
        return reshape_output_1


class Reshape771(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 13, 28, 128))
        return reshape_output_1


class Reshape772(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 13, 3584))
        return reshape_output_1


class Reshape773(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(28, 13, 128))
        return reshape_output_1


class Reshape774(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 13, 512))
        return reshape_output_1


class Reshape775(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 13, 4, 128))
        return reshape_output_1


class Reshape776(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 28, 13, 128))
        return reshape_output_1


class Reshape777(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 28, 13, 13))
        return reshape_output_1


class Reshape778(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(28, 13, 13))
        return reshape_output_1


class Reshape779(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(28, 128, 13))
        return reshape_output_1


class Reshape780(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 13, 18944))
        return reshape_output_1


class Reshape781(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(39, 3584))
        return reshape_output_1


class Reshape782(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 39, 28, 128))
        return reshape_output_1


class Reshape783(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 39, 3584))
        return reshape_output_1


class Reshape784(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(28, 39, 128))
        return reshape_output_1


class Reshape785(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 39, 512))
        return reshape_output_1


class Reshape786(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 39, 4, 128))
        return reshape_output_1


class Reshape787(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 28, 39, 128))
        return reshape_output_1


class Reshape788(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 28, 39, 39))
        return reshape_output_1


class Reshape789(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(28, 39, 39))
        return reshape_output_1


class Reshape790(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(28, 128, 39))
        return reshape_output_1


class Reshape791(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 39, 18944))
        return reshape_output_1


class Reshape792(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(39, 896))
        return reshape_output_1


class Reshape793(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 39, 14, 64))
        return reshape_output_1


class Reshape794(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 39, 896))
        return reshape_output_1


class Reshape795(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(14, 39, 64))
        return reshape_output_1


class Reshape796(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 39, 128))
        return reshape_output_1


class Reshape797(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 39, 2, 64))
        return reshape_output_1


class Reshape798(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 14, 39, 64))
        return reshape_output_1


class Reshape799(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 14, 39, 39))
        return reshape_output_1


class Reshape800(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(14, 39, 39))
        return reshape_output_1


class Reshape801(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(14, 64, 39))
        return reshape_output_1


class Reshape802(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 39, 4864))
        return reshape_output_1


class Reshape803(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(29, 896))
        return reshape_output_1


class Reshape804(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 29, 14, 64))
        return reshape_output_1


class Reshape805(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 29, 896))
        return reshape_output_1


class Reshape806(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(14, 29, 64))
        return reshape_output_1


class Reshape807(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 29, 128))
        return reshape_output_1


class Reshape808(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 29, 2, 64))
        return reshape_output_1


class Reshape809(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 14, 29, 64))
        return reshape_output_1


class Reshape810(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 14, 29, 29))
        return reshape_output_1


class Reshape811(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(14, 29, 29))
        return reshape_output_1


class Reshape812(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(14, 64, 29))
        return reshape_output_1


class Reshape813(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 29, 4864))
        return reshape_output_1


class Reshape814(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(29, 2048))
        return reshape_output_1


class Reshape815(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 29, 16, 128))
        return reshape_output_1


class Reshape816(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 29, 2048))
        return reshape_output_1


class Reshape817(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 29, 128))
        return reshape_output_1


class Reshape818(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 29, 128))
        return reshape_output_1


class Reshape819(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 128, 29))
        return reshape_output_1


class Reshape820(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 29, 11008))
        return reshape_output_1


class Reshape821(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(39, 2048))
        return reshape_output_1


class Reshape822(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 39, 16, 128))
        return reshape_output_1


class Reshape823(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 39, 2048))
        return reshape_output_1


class Reshape824(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 39, 128))
        return reshape_output_1


class Reshape825(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 39, 128))
        return reshape_output_1


class Reshape826(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 39, 39))
        return reshape_output_1


class Reshape827(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 39, 39))
        return reshape_output_1


class Reshape828(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 128, 39))
        return reshape_output_1


class Reshape829(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(29, 3584))
        return reshape_output_1


class Reshape830(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 29, 28, 128))
        return reshape_output_1


class Reshape831(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 29, 3584))
        return reshape_output_1


class Reshape832(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(28, 29, 128))
        return reshape_output_1


class Reshape833(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 29, 512))
        return reshape_output_1


class Reshape834(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 29, 4, 128))
        return reshape_output_1


class Reshape835(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 28, 29, 128))
        return reshape_output_1


class Reshape836(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 28, 29, 29))
        return reshape_output_1


class Reshape837(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(28, 29, 29))
        return reshape_output_1


class Reshape838(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(28, 128, 29))
        return reshape_output_1


class Reshape839(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 29, 18944))
        return reshape_output_1


class Reshape840(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 64, 128))
        return reshape_output_1


class Reshape841(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(768, 768, 1, 1))
        return reshape_output_1


class Reshape842(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 768, 128))
        return reshape_output_1


class Reshape843(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 61))
        return reshape_output_1


class Reshape844(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(61, 1024))
        return reshape_output_1


class Reshape845(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 61, 16, 64))
        return reshape_output_1


class Reshape846(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 61, 1024))
        return reshape_output_1


class Reshape847(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 61, 64))
        return reshape_output_1


class Reshape848(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 61, 61))
        return reshape_output_1


class Reshape849(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 61, 61))
        return reshape_output_1


class Reshape850(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 64, 61))
        return reshape_output_1


class Reshape851(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 61, 64))
        return reshape_output_1


class Reshape852(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 1, 61))
        return reshape_output_1


class Reshape853(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 1, 61))
        return reshape_output_1


class Reshape854(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(61, 512))
        return reshape_output_1


class Reshape855(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 61, 8, 64))
        return reshape_output_1


class Reshape856(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 61, 512))
        return reshape_output_1


class Reshape857(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 61, 64))
        return reshape_output_1


class Reshape858(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 61, 61))
        return reshape_output_1


class Reshape859(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 61, 61))
        return reshape_output_1


class Reshape860(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 64, 61))
        return reshape_output_1


class Reshape861(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 61, 64))
        return reshape_output_1


class Reshape862(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 1, 61))
        return reshape_output_1


class Reshape863(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 1, 61))
        return reshape_output_1


class Reshape864(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 61, 2816))
        return reshape_output_1


class Reshape865(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 2816))
        return reshape_output_1


class Reshape866(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(61, 768))
        return reshape_output_1


class Reshape867(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 61, 12, 64))
        return reshape_output_1


class Reshape868(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 61, 768))
        return reshape_output_1


class Reshape869(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 61, 64))
        return reshape_output_1


class Reshape870(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 61, 61))
        return reshape_output_1


class Reshape871(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 61, 61))
        return reshape_output_1


class Reshape872(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 64, 61))
        return reshape_output_1


class Reshape873(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 61, 64))
        return reshape_output_1


class Reshape874(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 1, 61))
        return reshape_output_1


class Reshape875(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 1, 61))
        return reshape_output_1


class Reshape876(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 61, 6, 64))
        return reshape_output_1


class Reshape877(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(6, 61, 64))
        return reshape_output_1


class Reshape878(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 6, 61, 61))
        return reshape_output_1


class Reshape879(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(6, 61, 61))
        return reshape_output_1


class Reshape880(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(6, 64, 61))
        return reshape_output_1


class Reshape881(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 6, 61, 64))
        return reshape_output_1


class Reshape882(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(61, 384))
        return reshape_output_1


class Reshape883(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 6, 1, 61))
        return reshape_output_1


class Reshape884(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(6, 1, 61))
        return reshape_output_1


class Reshape885(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 61, 2048))
        return reshape_output_1


class Reshape886(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 9216, 1, 1))
        return reshape_output_1


class Reshape887(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 9216))
        return reshape_output_1


class Reshape888(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 96, 54, 54))
        return reshape_output_1


class Reshape889(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 96, 54, 54))
        return reshape_output_1


class Reshape890(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 256, 27, 27))
        return reshape_output_1


class Reshape891(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 27, 27))
        return reshape_output_1


class Reshape892(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1024, 196, 1))
        return reshape_output_1


class Reshape893(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(197, 1024))
        return reshape_output_1


class Reshape894(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 197, 16, 64))
        return reshape_output_1


class Reshape895(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 197, 1024))
        return reshape_output_1


class Reshape896(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 197, 64))
        return reshape_output_1


class Reshape897(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 197, 197))
        return reshape_output_1


class Reshape898(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 27, 27, 16))
        return reshape_output_1


class Reshape899(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(729, 16))
        return reshape_output_1


class Reshape900(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(197, 197, 16))
        return reshape_output_1


class Reshape901(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 197, 197))
        return reshape_output_1


class Reshape902(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 64, 197))
        return reshape_output_1


class Reshape903(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 197, 64))
        return reshape_output_1


class Reshape904(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 768, 196, 1))
        return reshape_output_1


class Reshape905(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(197, 768))
        return reshape_output_1


class Reshape906(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 197, 12, 64))
        return reshape_output_1


class Reshape907(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 197, 768))
        return reshape_output_1


class Reshape908(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 197, 64))
        return reshape_output_1


class Reshape909(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 197, 197))
        return reshape_output_1


class Reshape910(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 27, 27, 12))
        return reshape_output_1


class Reshape911(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(729, 12))
        return reshape_output_1


class Reshape912(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(197, 197, 12))
        return reshape_output_1


class Reshape913(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 197, 197))
        return reshape_output_1


class Reshape914(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 64, 197))
        return reshape_output_1


class Reshape915(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 197, 64))
        return reshape_output_1


class Reshape916(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 192, 196, 1))
        return reshape_output_1


class Reshape917(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(197, 192))
        return reshape_output_1


class Reshape918(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 197, 3, 64))
        return reshape_output_1


class Reshape919(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 197, 192))
        return reshape_output_1


class Reshape920(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(3, 197, 64))
        return reshape_output_1


class Reshape921(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 3, 197, 197))
        return reshape_output_1


class Reshape922(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(3, 197, 197))
        return reshape_output_1


class Reshape923(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(3, 64, 197))
        return reshape_output_1


class Reshape924(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 3, 197, 64))
        return reshape_output_1


class Reshape925(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 192))
        return reshape_output_1


class Reshape926(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 384, 196, 1))
        return reshape_output_1


class Reshape927(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(197, 384))
        return reshape_output_1


class Reshape928(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 197, 6, 64))
        return reshape_output_1


class Reshape929(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 197, 384))
        return reshape_output_1


class Reshape930(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(6, 197, 64))
        return reshape_output_1


class Reshape931(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 6, 197, 197))
        return reshape_output_1


class Reshape932(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(6, 197, 197))
        return reshape_output_1


class Reshape933(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(6, 64, 197))
        return reshape_output_1


class Reshape934(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 6, 197, 64))
        return reshape_output_1


class Reshape935(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1920, 1, 1))
        return reshape_output_1


class Reshape936(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1024, 1, 1))
        return reshape_output_1


class Reshape937(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2208, 1, 1))
        return reshape_output_1


class Reshape938(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1664, 1, 1))
        return reshape_output_1


class Reshape939(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1000, 1, 1))
        return reshape_output_1


class Reshape940(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 1, 3, 3))
        return reshape_output_1


class Reshape941(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(96, 1, 3, 3))
        return reshape_output_1


class Reshape942(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(144, 1, 3, 3))
        return reshape_output_1


class Reshape943(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(144, 1, 5, 5))
        return reshape_output_1


class Reshape944(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(240, 1, 5, 5))
        return reshape_output_1


class Reshape945(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(240, 1, 3, 3))
        return reshape_output_1


class Reshape946(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(480, 1, 3, 3))
        return reshape_output_1


class Reshape947(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(480, 1, 5, 5))
        return reshape_output_1


class Reshape948(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(672, 1, 5, 5))
        return reshape_output_1


class Reshape949(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1152, 1, 5, 5))
        return reshape_output_1


class Reshape950(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1152, 1, 3, 3))
        return reshape_output_1


class Reshape951(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1280, 1, 1))
        return reshape_output_1


class Reshape952(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(48, 1, 3, 3))
        return reshape_output_1


class Reshape953(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(24, 1, 3, 3))
        return reshape_output_1


class Reshape954(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(192, 1, 3, 3))
        return reshape_output_1


class Reshape955(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(192, 1, 5, 5))
        return reshape_output_1


class Reshape956(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(336, 1, 5, 5))
        return reshape_output_1


class Reshape957(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(336, 1, 3, 3))
        return reshape_output_1


class Reshape958(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(672, 1, 3, 3))
        return reshape_output_1


class Reshape959(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(960, 1, 5, 5))
        return reshape_output_1


class Reshape960(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1632, 1, 5, 5))
        return reshape_output_1


class Reshape961(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1632, 1, 3, 3))
        return reshape_output_1


class Reshape962(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2688, 1, 3, 3))
        return reshape_output_1


class Reshape963(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1792, 1, 1))
        return reshape_output_1


class Reshape964(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(288, 1, 5, 5))
        return reshape_output_1


class Reshape965(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(288, 1, 3, 3))
        return reshape_output_1


class Reshape966(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(576, 1, 3, 3))
        return reshape_output_1


class Reshape967(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(576, 1, 5, 5))
        return reshape_output_1


class Reshape968(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(816, 1, 5, 5))
        return reshape_output_1


class Reshape969(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1392, 1, 5, 5))
        return reshape_output_1


class Reshape970(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1392, 1, 3, 3))
        return reshape_output_1


class Reshape971(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(528, 1, 3, 3))
        return reshape_output_1


class Reshape972(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(528, 1, 5, 5))
        return reshape_output_1


class Reshape973(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(720, 1, 5, 5))
        return reshape_output_1


class Reshape974(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1248, 1, 5, 5))
        return reshape_output_1


class Reshape975(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1248, 1, 3, 3))
        return reshape_output_1


class Reshape976(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 1, 3, 3))
        return reshape_output_1


class Reshape977(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 1, 3, 3))
        return reshape_output_1


class Reshape978(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 1, 3, 3))
        return reshape_output_1


class Reshape979(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(36, 1, 3, 3))
        return reshape_output_1


class Reshape980(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(72, 1, 5, 5))
        return reshape_output_1


class Reshape981(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(20, 1, 3, 3))
        return reshape_output_1


class Reshape982(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(24, 1, 5, 5))
        return reshape_output_1


class Reshape983(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(60, 1, 3, 3))
        return reshape_output_1


class Reshape984(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(120, 1, 3, 3))
        return reshape_output_1


class Reshape985(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(40, 1, 3, 3))
        return reshape_output_1


class Reshape986(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(100, 1, 3, 3))
        return reshape_output_1


class Reshape987(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(92, 1, 3, 3))
        return reshape_output_1


class Reshape988(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(56, 1, 3, 3))
        return reshape_output_1


class Reshape989(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(80, 1, 3, 3))
        return reshape_output_1


class Reshape990(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(112, 1, 5, 5))
        return reshape_output_1


class Reshape991(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(72, 1, 1, 5))
        return reshape_output_1


class Reshape992(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(72, 1, 5, 1))
        return reshape_output_1


class Reshape993(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(120, 1, 1, 5))
        return reshape_output_1


class Reshape994(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(120, 1, 5, 1))
        return reshape_output_1


class Reshape995(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(240, 1, 1, 5))
        return reshape_output_1


class Reshape996(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(240, 1, 5, 1))
        return reshape_output_1


class Reshape997(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(200, 1, 1, 5))
        return reshape_output_1


class Reshape998(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(200, 1, 5, 1))
        return reshape_output_1


class Reshape999(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(184, 1, 1, 5))
        return reshape_output_1


class Reshape1000(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(184, 1, 5, 1))
        return reshape_output_1


class Reshape1001(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(480, 1, 1, 5))
        return reshape_output_1


class Reshape1002(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(480, 1, 5, 1))
        return reshape_output_1


class Reshape1003(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(672, 1, 1, 5))
        return reshape_output_1


class Reshape1004(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(672, 1, 5, 1))
        return reshape_output_1


class Reshape1005(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(960, 1, 1, 5))
        return reshape_output_1


class Reshape1006(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(960, 1, 5, 1))
        return reshape_output_1


class Reshape1007(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 64, 19200, 1))
        return reshape_output_1


class Reshape1008(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 19200, 1, 64))
        return reshape_output_1


class Reshape1009(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 120, 160, 64))
        return reshape_output_1


class Reshape1010(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 64, 120, 160))
        return reshape_output_1


class Reshape1011(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 64, 300))
        return reshape_output_1


class Reshape1012(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(300, 64))
        return reshape_output_1


class Reshape1013(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 300, 1, 64))
        return reshape_output_1


class Reshape1014(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 300, 64))
        return reshape_output_1


class Reshape1015(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 19200, 300))
        return reshape_output_1


class Reshape1016(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 19200, 300))
        return reshape_output_1


class Reshape1017(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 120, 160))
        return reshape_output_1


class Reshape1018(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(256, 1, 3, 3))
        return reshape_output_1


class Reshape1019(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 19200, 1))
        return reshape_output_1


class Reshape1020(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 4800, 1))
        return reshape_output_1


class Reshape1021(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4800, 2, 64))
        return reshape_output_1


class Reshape1022(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 60, 80, 128))
        return reshape_output_1


class Reshape1023(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 4800, 64))
        return reshape_output_1


class Reshape1024(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 60, 80))
        return reshape_output_1


class Reshape1025(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 300))
        return reshape_output_1


class Reshape1026(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(300, 128))
        return reshape_output_1


class Reshape1027(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 300, 2, 64))
        return reshape_output_1


class Reshape1028(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 300, 128))
        return reshape_output_1


class Reshape1029(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 300, 64))
        return reshape_output_1


class Reshape1030(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2, 4800, 300))
        return reshape_output_1


class Reshape1031(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 4800, 300))
        return reshape_output_1


class Reshape1032(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 64, 300))
        return reshape_output_1


class Reshape1033(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2, 4800, 64))
        return reshape_output_1


class Reshape1034(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(4800, 128))
        return reshape_output_1


class Reshape1035(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4800, 128))
        return reshape_output_1


class Reshape1036(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 512, 60, 80))
        return reshape_output_1


class Reshape1037(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(512, 1, 3, 3))
        return reshape_output_1


class Reshape1038(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 512, 4800, 1))
        return reshape_output_1


class Reshape1039(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 320, 1200, 1))
        return reshape_output_1


class Reshape1040(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1200, 5, 64))
        return reshape_output_1


class Reshape1041(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 30, 40, 320))
        return reshape_output_1


class Reshape1042(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(5, 1200, 64))
        return reshape_output_1


class Reshape1043(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 320, 30, 40))
        return reshape_output_1


class Reshape1044(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 320, 300))
        return reshape_output_1


class Reshape1045(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(300, 320))
        return reshape_output_1


class Reshape1046(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 300, 5, 64))
        return reshape_output_1


class Reshape1047(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 300, 320))
        return reshape_output_1


class Reshape1048(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(5, 300, 64))
        return reshape_output_1


class Reshape1049(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 5, 1200, 300))
        return reshape_output_1


class Reshape1050(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(5, 1200, 300))
        return reshape_output_1


class Reshape1051(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(5, 64, 300))
        return reshape_output_1


class Reshape1052(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 5, 1200, 64))
        return reshape_output_1


class Reshape1053(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1200, 320))
        return reshape_output_1


class Reshape1054(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1200, 320))
        return reshape_output_1


class Reshape1055(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1280, 30, 40))
        return reshape_output_1


class Reshape1056(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1280, 1, 3, 3))
        return reshape_output_1


class Reshape1057(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1280, 1200, 1))
        return reshape_output_1


class Reshape1058(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 512, 300, 1))
        return reshape_output_1


class Reshape1059(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(300, 512))
        return reshape_output_1


class Reshape1060(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 300, 8, 64))
        return reshape_output_1


class Reshape1061(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 15, 20, 512))
        return reshape_output_1


class Reshape1062(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 300, 512))
        return reshape_output_1


class Reshape1063(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 300, 64))
        return reshape_output_1


class Reshape1064(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 300, 300))
        return reshape_output_1


class Reshape1065(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 300, 300))
        return reshape_output_1


class Reshape1066(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 64, 300))
        return reshape_output_1


class Reshape1067(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 300, 64))
        return reshape_output_1


class Reshape1068(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2048, 15, 20))
        return reshape_output_1


class Reshape1069(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2048, 1, 3, 3))
        return reshape_output_1


class Reshape1070(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2048, 300, 1))
        return reshape_output_1


class Reshape1071(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 30, 40))
        return reshape_output_1


class Reshape1072(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 60, 80))
        return reshape_output_1


class Reshape1073(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 120, 160))
        return reshape_output_1


class Reshape1074(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 224, 224))
        return reshape_output_1


class Reshape1075(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2048, 1, 1))
        return reshape_output_1


class Reshape1076(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1536, 1, 1))
        return reshape_output_1


class Reshape1077(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 3, 16, 16, 16, 16))
        return reshape_output_1


class Reshape1078(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 512, 1))
        return reshape_output_1


class Reshape1079(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 16, 512))
        return reshape_output_1


class Reshape1080(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1024, 256, 1, 1))
        return reshape_output_1


class Reshape1081(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1024, 512, 1))
        return reshape_output_1


class Reshape1082(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(256, 1024, 1, 1))
        return reshape_output_1


class Reshape1083(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 1, 3, 3))
        return reshape_output_1


class Reshape1084(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(128, 1, 3, 3))
        return reshape_output_1


class Reshape1085(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1024, 1, 3, 3))
        return reshape_output_1


class Reshape1086(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(384, 1, 3, 3))
        return reshape_output_1


class Reshape1087(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(768, 1, 3, 3))
        return reshape_output_1


class Reshape1088(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 768, 1, 1))
        return reshape_output_1


class Reshape1089(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(960, 1, 3, 3))
        return reshape_output_1


class Reshape1090(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(432, 1, 3, 3))
        return reshape_output_1


class Reshape1091(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(720, 1, 3, 3))
        return reshape_output_1


class Reshape1092(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(72, 1, 3, 3))
        return reshape_output_1


class Reshape1093(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(88, 1, 3, 3))
        return reshape_output_1


class Reshape1094(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(96, 1, 5, 5))
        return reshape_output_1


class Reshape1095(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(120, 1, 5, 5))
        return reshape_output_1


class Reshape1096(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(200, 1, 3, 3))
        return reshape_output_1


class Reshape1097(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(184, 1, 3, 3))
        return reshape_output_1


class Reshape1098(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 960, 1, 1))
        return reshape_output_1


class Reshape1099(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 576, 1, 1))
        return reshape_output_1


class Reshape1100(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 512, 1, 1))
        return reshape_output_1


class Reshape1101(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 3024, 1, 1))
        return reshape_output_1


class Reshape1102(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2520, 1, 1))
        return reshape_output_1


class Reshape1103(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 888, 1, 1))
        return reshape_output_1


class Reshape1104(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 3712, 1, 1))
        return reshape_output_1


class Reshape1105(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 784, 1, 1))
        return reshape_output_1


class Reshape1106(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 7392, 1, 1))
        return reshape_output_1


class Reshape1107(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 672, 1, 1))
        return reshape_output_1


class Reshape1108(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 400, 1, 1))
        return reshape_output_1


class Reshape1109(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1008, 1, 1))
        return reshape_output_1


class Reshape1110(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 912, 1, 1))
        return reshape_output_1


class Reshape1111(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 440, 1, 1))
        return reshape_output_1


class Reshape1112(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1088, 1, 1))
        return reshape_output_1


class Reshape1113(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2016, 1, 1))
        return reshape_output_1


class Reshape1114(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1512, 1, 1))
        return reshape_output_1


class Reshape1115(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16384, 1, 32))
        return reshape_output_1


class Reshape1116(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 128, 32))
        return reshape_output_1


class Reshape1117(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 256))
        return reshape_output_1


class Reshape1118(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(256, 32))
        return reshape_output_1


class Reshape1119(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 1, 32))
        return reshape_output_1


class Reshape1120(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 32))
        return reshape_output_1


class Reshape1121(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 16384, 256))
        return reshape_output_1


class Reshape1122(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16384, 256))
        return reshape_output_1


class Reshape1123(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 128, 128))
        return reshape_output_1


class Reshape1124(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 16384, 1))
        return reshape_output_1


class Reshape1125(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 64, 4096, 1))
        return reshape_output_1


class Reshape1126(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4096, 2, 32))
        return reshape_output_1


class Reshape1127(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 64, 64, 64))
        return reshape_output_1


class Reshape1128(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 4096, 32))
        return reshape_output_1


class Reshape1129(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 64, 256))
        return reshape_output_1


class Reshape1130(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(256, 64))
        return reshape_output_1


class Reshape1131(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 2, 32))
        return reshape_output_1


class Reshape1132(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 1, 64))
        return reshape_output_1


class Reshape1133(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 64))
        return reshape_output_1


class Reshape1134(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 256, 32))
        return reshape_output_1


class Reshape1135(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2, 4096, 256))
        return reshape_output_1


class Reshape1136(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 4096, 256))
        return reshape_output_1


class Reshape1137(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 32, 256))
        return reshape_output_1


class Reshape1138(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2, 4096, 32))
        return reshape_output_1


class Reshape1139(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(4096, 64))
        return reshape_output_1


class Reshape1140(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4096, 64))
        return reshape_output_1


class Reshape1141(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 4096, 1))
        return reshape_output_1


class Reshape1142(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 160, 1024, 1))
        return reshape_output_1


class Reshape1143(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1024, 5, 32))
        return reshape_output_1


class Reshape1144(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 32, 160))
        return reshape_output_1


class Reshape1145(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(5, 1024, 32))
        return reshape_output_1


class Reshape1146(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 160, 32, 32))
        return reshape_output_1


class Reshape1147(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 160, 256))
        return reshape_output_1


class Reshape1148(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(256, 160))
        return reshape_output_1


class Reshape1149(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 5, 32))
        return reshape_output_1


class Reshape1150(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 160))
        return reshape_output_1


class Reshape1151(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(5, 256, 32))
        return reshape_output_1


class Reshape1152(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 5, 1024, 256))
        return reshape_output_1


class Reshape1153(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(5, 1024, 256))
        return reshape_output_1


class Reshape1154(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(5, 32, 256))
        return reshape_output_1


class Reshape1155(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 5, 1024, 32))
        return reshape_output_1


class Reshape1156(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1024, 160))
        return reshape_output_1


class Reshape1157(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1024, 160))
        return reshape_output_1


class Reshape1158(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 640, 32, 32))
        return reshape_output_1


class Reshape1159(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(640, 1, 3, 3))
        return reshape_output_1


class Reshape1160(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 640, 1024, 1))
        return reshape_output_1


class Reshape1161(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 256, 1))
        return reshape_output_1


class Reshape1162(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 32, 256))
        return reshape_output_1


class Reshape1163(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 256, 32))
        return reshape_output_1


class Reshape1164(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1024, 16, 16))
        return reshape_output_1


class Reshape1165(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1024, 256, 1))
        return reshape_output_1


class Reshape1166(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16384, 1, 64))
        return reshape_output_1


class Reshape1167(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 128, 64))
        return reshape_output_1


class Reshape1168(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 128, 128))
        return reshape_output_1


class Reshape1169(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 16384, 1))
        return reshape_output_1


class Reshape1170(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4096, 2, 64))
        return reshape_output_1


class Reshape1171(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 64, 64, 128))
        return reshape_output_1


class Reshape1172(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 4096, 64))
        return reshape_output_1


class Reshape1173(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 256))
        return reshape_output_1


class Reshape1174(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(256, 128))
        return reshape_output_1


class Reshape1175(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 2, 64))
        return reshape_output_1


class Reshape1176(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 128))
        return reshape_output_1


class Reshape1177(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 256, 64))
        return reshape_output_1


class Reshape1178(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 64, 256))
        return reshape_output_1


class Reshape1179(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2, 4096, 64))
        return reshape_output_1


class Reshape1180(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(4096, 128))
        return reshape_output_1


class Reshape1181(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4096, 128))
        return reshape_output_1


class Reshape1182(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 512, 64, 64))
        return reshape_output_1


class Reshape1183(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 512, 4096, 1))
        return reshape_output_1


class Reshape1184(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 320, 1024, 1))
        return reshape_output_1


class Reshape1185(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1024, 5, 64))
        return reshape_output_1


class Reshape1186(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 32, 320))
        return reshape_output_1


class Reshape1187(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(5, 1024, 64))
        return reshape_output_1


class Reshape1188(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 320, 32, 32))
        return reshape_output_1


class Reshape1189(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 320, 256))
        return reshape_output_1


class Reshape1190(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(256, 320))
        return reshape_output_1


class Reshape1191(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 5, 64))
        return reshape_output_1


class Reshape1192(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 320))
        return reshape_output_1


class Reshape1193(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(5, 256, 64))
        return reshape_output_1


class Reshape1194(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(5, 64, 256))
        return reshape_output_1


class Reshape1195(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 5, 1024, 64))
        return reshape_output_1


class Reshape1196(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1024, 320))
        return reshape_output_1


class Reshape1197(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1024, 320))
        return reshape_output_1


class Reshape1198(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1280, 32, 32))
        return reshape_output_1


class Reshape1199(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1280, 1024, 1))
        return reshape_output_1


class Reshape1200(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 512, 256, 1))
        return reshape_output_1


class Reshape1201(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 256, 64))
        return reshape_output_1


class Reshape1202(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 64, 256))
        return reshape_output_1


class Reshape1203(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 256, 64))
        return reshape_output_1


class Reshape1204(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2048, 256, 1))
        return reshape_output_1


class Reshape1205(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 768, 16, 16))
        return reshape_output_1


class Reshape1206(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 768, 32, 32))
        return reshape_output_1


class Reshape1207(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 768, 64, 64))
        return reshape_output_1


class Reshape1208(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 768, 128, 128))
        return reshape_output_1


class Reshape1209(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 5776))
        return reshape_output_1


class Reshape1210(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 2166))
        return reshape_output_1


class Reshape1211(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 600))
        return reshape_output_1


class Reshape1212(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 150))
        return reshape_output_1


class Reshape1213(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 36))
        return reshape_output_1


class Reshape1214(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 81, 5776))
        return reshape_output_1


class Reshape1215(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 81, 2166))
        return reshape_output_1


class Reshape1216(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 81, 600))
        return reshape_output_1


class Reshape1217(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 81, 150))
        return reshape_output_1


class Reshape1218(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 81, 36))
        return reshape_output_1


class Reshape1219(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 81, 4))
        return reshape_output_1


class Reshape1220(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 7, 8, 7, 96))
        return reshape_output_1


class Reshape1221(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(3136, 96))
        return reshape_output_1


class Reshape1222(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 3136, 96))
        return reshape_output_1


class Reshape1223(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 49, 288))
        return reshape_output_1


class Reshape1224(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 49, 3, 3, 32))
        return reshape_output_1


class Reshape1225(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 3, 49, 32))
        return reshape_output_1


class Reshape1226(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(192, 49, 32))
        return reshape_output_1


class Reshape1227(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 3, 49, 49))
        return reshape_output_1


class Reshape1228(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(49, 49, 3))
        return reshape_output_1


class Reshape1229(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(192, 49, 49))
        return reshape_output_1


class Reshape1230(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 64, 3, 49, 49))
        return reshape_output_1


class Reshape1231(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(192, 32, 49))
        return reshape_output_1


class Reshape1232(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 49, 96))
        return reshape_output_1


class Reshape1233(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 56, 56, 96))
        return reshape_output_1


class Reshape1234(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 8, 7, 7, 96))
        return reshape_output_1


class Reshape1235(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 49, 3, 32))
        return reshape_output_1


class Reshape1236(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 56, 56, 384))
        return reshape_output_1


class Reshape1237(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 49, 384))
        return reshape_output_1


class Reshape1238(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(3136, 384))
        return reshape_output_1


class Reshape1239(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(784, 384))
        return reshape_output_1


class Reshape1240(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 784, 384))
        return reshape_output_1


class Reshape1241(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 28, 28, 192))
        return reshape_output_1


class Reshape1242(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 49, 192))
        return reshape_output_1


class Reshape1243(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 7, 4, 7, 192))
        return reshape_output_1


class Reshape1244(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(784, 192))
        return reshape_output_1


class Reshape1245(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 784, 192))
        return reshape_output_1


class Reshape1246(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 49, 576))
        return reshape_output_1


class Reshape1247(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 49, 3, 6, 32))
        return reshape_output_1


class Reshape1248(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 6, 49, 32))
        return reshape_output_1


class Reshape1249(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(96, 49, 32))
        return reshape_output_1


class Reshape1250(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 6, 49, 49))
        return reshape_output_1


class Reshape1251(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(49, 49, 6))
        return reshape_output_1


class Reshape1252(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(96, 49, 49))
        return reshape_output_1


class Reshape1253(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 6, 49, 49))
        return reshape_output_1


class Reshape1254(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(96, 32, 49))
        return reshape_output_1


class Reshape1255(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 4, 7, 7, 192))
        return reshape_output_1


class Reshape1256(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 49, 6, 32))
        return reshape_output_1


class Reshape1257(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 28, 28, 768))
        return reshape_output_1


class Reshape1258(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 49, 768))
        return reshape_output_1


class Reshape1259(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(784, 768))
        return reshape_output_1


class Reshape1260(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(196, 768))
        return reshape_output_1


class Reshape1261(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 196, 768))
        return reshape_output_1


class Reshape1262(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 14, 14, 384))
        return reshape_output_1


class Reshape1263(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(4, 49, 384))
        return reshape_output_1


class Reshape1264(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2, 7, 2, 7, 384))
        return reshape_output_1


class Reshape1265(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(196, 384))
        return reshape_output_1


class Reshape1266(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 196, 384))
        return reshape_output_1


class Reshape1267(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(4, 49, 1152))
        return reshape_output_1


class Reshape1268(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(4, 49, 3, 12, 32))
        return reshape_output_1


class Reshape1269(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(4, 12, 49, 32))
        return reshape_output_1


class Reshape1270(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(48, 49, 32))
        return reshape_output_1


class Reshape1271(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(4, 12, 49, 49))
        return reshape_output_1


class Reshape1272(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(49, 49, 12))
        return reshape_output_1


class Reshape1273(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(48, 49, 49))
        return reshape_output_1


class Reshape1274(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 12, 49, 49))
        return reshape_output_1


class Reshape1275(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(48, 32, 49))
        return reshape_output_1


class Reshape1276(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2, 2, 7, 7, 384))
        return reshape_output_1


class Reshape1277(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(4, 49, 12, 32))
        return reshape_output_1


class Reshape1278(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 14, 14, 1536))
        return reshape_output_1


class Reshape1279(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(4, 49, 1536))
        return reshape_output_1


class Reshape1280(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(196, 1536))
        return reshape_output_1


class Reshape1281(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(49, 1536))
        return reshape_output_1


class Reshape1282(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 49, 1536))
        return reshape_output_1


class Reshape1283(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 7, 7, 768))
        return reshape_output_1


class Reshape1284(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 49, 768))
        return reshape_output_1


class Reshape1285(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 7, 1, 7, 768))
        return reshape_output_1


class Reshape1286(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(49, 768))
        return reshape_output_1


class Reshape1287(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 49, 2304))
        return reshape_output_1


class Reshape1288(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 49, 3, 24, 32))
        return reshape_output_1


class Reshape1289(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 24, 49, 32))
        return reshape_output_1


class Reshape1290(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(24, 49, 32))
        return reshape_output_1


class Reshape1291(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 24, 49, 49))
        return reshape_output_1


class Reshape1292(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(49, 49, 24))
        return reshape_output_1


class Reshape1293(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(24, 49, 49))
        return reshape_output_1


class Reshape1294(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(24, 32, 49))
        return reshape_output_1


class Reshape1295(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 1, 7, 7, 768))
        return reshape_output_1


class Reshape1296(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 49, 24, 32))
        return reshape_output_1


class Reshape1297(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 7, 7, 3072))
        return reshape_output_1


class Reshape1298(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 49, 3072))
        return reshape_output_1


class Reshape1299(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(49, 3072))
        return reshape_output_1


class Reshape1300(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 96, 3136, 1))
        return reshape_output_1


class Reshape1301(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2401,))
        return reshape_output_1


class Reshape1302(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 768, 1))
        return reshape_output_1


class Reshape1303(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 7, 8, 7, 128))
        return reshape_output_1


class Reshape1304(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(3136, 128))
        return reshape_output_1


class Reshape1305(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 49, 3, 4, 32))
        return reshape_output_1


class Reshape1306(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 4, 49, 32))
        return reshape_output_1


class Reshape1307(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(256, 49, 32))
        return reshape_output_1


class Reshape1308(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 4, 49, 49))
        return reshape_output_1


class Reshape1309(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(49, 49, 4))
        return reshape_output_1


class Reshape1310(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(256, 49, 49))
        return reshape_output_1


class Reshape1311(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 64, 4, 49, 49))
        return reshape_output_1


class Reshape1312(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(256, 32, 49))
        return reshape_output_1


class Reshape1313(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 49, 128))
        return reshape_output_1


class Reshape1314(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 56, 56, 128))
        return reshape_output_1


class Reshape1315(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 8, 7, 7, 128))
        return reshape_output_1


class Reshape1316(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 56, 56, 512))
        return reshape_output_1


class Reshape1317(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(3136, 512))
        return reshape_output_1


class Reshape1318(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(784, 512))
        return reshape_output_1


class Reshape1319(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 28, 28, 256))
        return reshape_output_1


class Reshape1320(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 49, 256))
        return reshape_output_1


class Reshape1321(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 7, 4, 7, 256))
        return reshape_output_1


class Reshape1322(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(784, 256))
        return reshape_output_1


class Reshape1323(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 49, 3, 8, 32))
        return reshape_output_1


class Reshape1324(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 8, 49, 32))
        return reshape_output_1


class Reshape1325(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(128, 49, 32))
        return reshape_output_1


class Reshape1326(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 8, 49, 49))
        return reshape_output_1


class Reshape1327(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(49, 49, 8))
        return reshape_output_1


class Reshape1328(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(128, 49, 49))
        return reshape_output_1


class Reshape1329(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 8, 49, 49))
        return reshape_output_1


class Reshape1330(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(128, 32, 49))
        return reshape_output_1


class Reshape1331(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 4, 7, 7, 256))
        return reshape_output_1


class Reshape1332(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 28, 28, 1024))
        return reshape_output_1


class Reshape1333(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(784, 1024))
        return reshape_output_1


class Reshape1334(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(196, 1024))
        return reshape_output_1


class Reshape1335(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 14, 14, 512))
        return reshape_output_1


class Reshape1336(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(4, 49, 512))
        return reshape_output_1


class Reshape1337(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2, 7, 2, 7, 512))
        return reshape_output_1


class Reshape1338(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(196, 512))
        return reshape_output_1


class Reshape1339(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(4, 49, 3, 16, 32))
        return reshape_output_1


class Reshape1340(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(4, 16, 49, 32))
        return reshape_output_1


class Reshape1341(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 49, 32))
        return reshape_output_1


class Reshape1342(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(4, 16, 49, 49))
        return reshape_output_1


class Reshape1343(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(49, 49, 16))
        return reshape_output_1


class Reshape1344(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 49, 49))
        return reshape_output_1


class Reshape1345(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 16, 49, 49))
        return reshape_output_1


class Reshape1346(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 32, 49))
        return reshape_output_1


class Reshape1347(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2, 2, 7, 7, 512))
        return reshape_output_1


class Reshape1348(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 14, 14, 2048))
        return reshape_output_1


class Reshape1349(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(196, 2048))
        return reshape_output_1


class Reshape1350(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(49, 2048))
        return reshape_output_1


class Reshape1351(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 7, 7, 1024))
        return reshape_output_1


class Reshape1352(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 49, 1024))
        return reshape_output_1


class Reshape1353(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 7, 1, 7, 1024))
        return reshape_output_1


class Reshape1354(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(49, 1024))
        return reshape_output_1


class Reshape1355(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 49, 3, 32, 32))
        return reshape_output_1


class Reshape1356(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 49, 32))
        return reshape_output_1


class Reshape1357(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 49, 32))
        return reshape_output_1


class Reshape1358(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 49, 49))
        return reshape_output_1


class Reshape1359(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(49, 49, 32))
        return reshape_output_1


class Reshape1360(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 49, 49))
        return reshape_output_1


class Reshape1361(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 32, 49))
        return reshape_output_1


class Reshape1362(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 1, 7, 7, 1024))
        return reshape_output_1


class Reshape1363(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 7, 7, 4096))
        return reshape_output_1


class Reshape1364(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(49, 4096))
        return reshape_output_1


class Reshape1365(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 25088, 1, 1))
        return reshape_output_1


class Reshape1366(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 25088))
        return reshape_output_1


class Reshape1367(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4096, 1, 1))
        return reshape_output_1


class Reshape1368(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(160, 1, 3, 3))
        return reshape_output_1


class Reshape1369(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(224, 1, 3, 3))
        return reshape_output_1


class Reshape1370(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(728, 1, 3, 3))
        return reshape_output_1


class Reshape1371(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1536, 1, 3, 3))
        return reshape_output_1


class Reshape1372(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 255, 40, 40))
        return reshape_output_1


class Reshape1373(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 255, 1600))
        return reshape_output_1


class Reshape1374(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 3, 85, 1600))
        return reshape_output_1


class Reshape1375(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4800, 85))
        return reshape_output_1


class Reshape1376(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 255, 20, 20))
        return reshape_output_1


class Reshape1377(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 255, 400))
        return reshape_output_1


class Reshape1378(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 3, 85, 400))
        return reshape_output_1


class Reshape1379(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1200, 85))
        return reshape_output_1


class Reshape1380(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 255, 10, 10))
        return reshape_output_1


class Reshape1381(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 255, 100))
        return reshape_output_1


class Reshape1382(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 3, 85, 100))
        return reshape_output_1


class Reshape1383(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 300, 85))
        return reshape_output_1


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    pytest.param(
        (
            Reshape0,
            [((8, 1), torch.int64)],
            {
                "model_name": [
                    "pt_stereo_facebook_musicgen_large_music_generation_hf",
                    "pt_stereo_facebook_musicgen_small_music_generation_hf",
                    "pt_stereo_facebook_musicgen_medium_music_generation_hf",
                ],
                "pcc": 0.99,
                "op_params": {"shape": "(2, 4, 1)"},
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Reshape1,
        [((2, 1, 1), torch.int64)],
        {
            "model_name": [
                "pt_stereo_facebook_musicgen_large_music_generation_hf",
                "pt_stereo_facebook_musicgen_small_music_generation_hf",
                "pt_stereo_facebook_musicgen_medium_music_generation_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(2, 1)"},
        },
    ),
    (
        Reshape2,
        [((1, 2048), torch.float32)],
        {
            "model_name": ["pt_stereo_facebook_musicgen_large_music_generation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 2048)"},
        },
    ),
    (
        Reshape3,
        [((1, 2048), torch.float32)],
        {"model_name": ["pt_t5_google_flan_t5_base_text_gen_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 1, 2048)"}},
    ),
    (
        Reshape4,
        [((2, 1, 2048), torch.float32)],
        {
            "model_name": ["pt_stereo_facebook_musicgen_large_music_generation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(2, 2048)"},
        },
    ),
    (
        Reshape5,
        [((2, 1, 2048), torch.float32)],
        {
            "model_name": ["pt_stereo_facebook_musicgen_large_music_generation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(2, 1, 32, 64)"},
        },
    ),
    (
        Reshape6,
        [((2, 2048), torch.float32)],
        {
            "model_name": [
                "pt_stereo_facebook_musicgen_large_music_generation_hf",
                "pt_stereo_facebook_musicgen_small_music_generation_hf",
                "pt_stereo_facebook_musicgen_medium_music_generation_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(2, 1, 2048)"},
        },
    ),
    (
        Reshape5,
        [((2, 2048), torch.float32)],
        {
            "model_name": ["pt_stereo_facebook_musicgen_large_music_generation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(2, 1, 32, 64)"},
        },
    ),
    (
        Reshape7,
        [((2, 32, 1, 64), torch.float32)],
        {
            "model_name": ["pt_stereo_facebook_musicgen_large_music_generation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(64, 1, 64)"},
        },
    ),
    (
        Reshape8,
        [((64, 1, 64), torch.float32)],
        {
            "model_name": ["pt_stereo_facebook_musicgen_large_music_generation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(2, 32, 1, 64)"},
        },
    ),
    (
        Reshape4,
        [((2, 1, 32, 64), torch.float32)],
        {
            "model_name": ["pt_stereo_facebook_musicgen_large_music_generation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(2, 2048)"},
        },
    ),
    (
        Reshape9,
        [((2, 13), torch.int64)],
        {
            "model_name": [
                "pt_stereo_facebook_musicgen_large_music_generation_hf",
                "pt_stereo_facebook_musicgen_small_music_generation_hf",
                "pt_stereo_facebook_musicgen_medium_music_generation_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(2, 13)"},
        },
    ),
    (
        Reshape10,
        [((2, 13, 768), torch.float32)],
        {
            "model_name": [
                "pt_stereo_facebook_musicgen_large_music_generation_hf",
                "pt_stereo_facebook_musicgen_small_music_generation_hf",
                "pt_stereo_facebook_musicgen_medium_music_generation_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(26, 768)"},
        },
    ),
    (
        Reshape11,
        [((26, 768), torch.float32)],
        {
            "model_name": [
                "pt_stereo_facebook_musicgen_large_music_generation_hf",
                "pt_stereo_facebook_musicgen_small_music_generation_hf",
                "pt_stereo_facebook_musicgen_medium_music_generation_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(2, 13, 12, 64)"},
        },
    ),
    (
        Reshape12,
        [((26, 768), torch.float32)],
        {
            "model_name": [
                "pt_stereo_facebook_musicgen_large_music_generation_hf",
                "pt_stereo_facebook_musicgen_small_music_generation_hf",
                "pt_stereo_facebook_musicgen_medium_music_generation_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(2, 13, 768)"},
        },
    ),
    (
        Reshape13,
        [((2, 12, 13, 64), torch.float32)],
        {
            "model_name": [
                "pt_stereo_facebook_musicgen_large_music_generation_hf",
                "pt_stereo_facebook_musicgen_small_music_generation_hf",
                "pt_stereo_facebook_musicgen_medium_music_generation_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(24, 13, 64)"},
        },
    ),
    (
        Reshape14,
        [((24, 13, 13), torch.float32)],
        {
            "model_name": [
                "pt_stereo_facebook_musicgen_large_music_generation_hf",
                "pt_stereo_facebook_musicgen_small_music_generation_hf",
                "pt_stereo_facebook_musicgen_medium_music_generation_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(2, 12, 13, 13)"},
        },
    ),
    (
        Reshape15,
        [((2, 12, 13, 13), torch.float32)],
        {
            "model_name": [
                "pt_stereo_facebook_musicgen_large_music_generation_hf",
                "pt_stereo_facebook_musicgen_small_music_generation_hf",
                "pt_stereo_facebook_musicgen_medium_music_generation_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(24, 13, 13)"},
        },
    ),
    (
        Reshape16,
        [((2, 12, 64, 13), torch.float32)],
        {
            "model_name": [
                "pt_stereo_facebook_musicgen_large_music_generation_hf",
                "pt_stereo_facebook_musicgen_small_music_generation_hf",
                "pt_stereo_facebook_musicgen_medium_music_generation_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(24, 64, 13)"},
        },
    ),
    (
        Reshape17,
        [((24, 13, 64), torch.float32)],
        {
            "model_name": [
                "pt_stereo_facebook_musicgen_large_music_generation_hf",
                "pt_stereo_facebook_musicgen_small_music_generation_hf",
                "pt_stereo_facebook_musicgen_medium_music_generation_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(2, 12, 13, 64)"},
        },
    ),
    (
        Reshape10,
        [((2, 13, 12, 64), torch.float32)],
        {
            "model_name": [
                "pt_stereo_facebook_musicgen_large_music_generation_hf",
                "pt_stereo_facebook_musicgen_small_music_generation_hf",
                "pt_stereo_facebook_musicgen_medium_music_generation_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(26, 768)"},
        },
    ),
    (
        Reshape18,
        [((26, 3072), torch.float32)],
        {
            "model_name": [
                "pt_stereo_facebook_musicgen_large_music_generation_hf",
                "pt_stereo_facebook_musicgen_small_music_generation_hf",
                "pt_stereo_facebook_musicgen_medium_music_generation_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(2, 13, 3072)"},
        },
    ),
    (
        Reshape19,
        [((2, 13, 3072), torch.float32)],
        {
            "model_name": [
                "pt_stereo_facebook_musicgen_large_music_generation_hf",
                "pt_stereo_facebook_musicgen_small_music_generation_hf",
                "pt_stereo_facebook_musicgen_medium_music_generation_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(26, 3072)"},
        },
    ),
    (
        Reshape20,
        [((26, 2048), torch.float32)],
        {
            "model_name": ["pt_stereo_facebook_musicgen_large_music_generation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(2, 13, 2048)"},
        },
    ),
    (
        Reshape21,
        [((26, 2048), torch.float32)],
        {
            "model_name": ["pt_stereo_facebook_musicgen_large_music_generation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(2, 13, 32, 64)"},
        },
    ),
    (
        Reshape22,
        [((2, 13, 2048), torch.float32)],
        {
            "model_name": ["pt_stereo_facebook_musicgen_large_music_generation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(26, 2048)"},
        },
    ),
    (
        Reshape23,
        [((2, 32, 13, 64), torch.float32)],
        {
            "model_name": ["pt_stereo_facebook_musicgen_large_music_generation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(64, 13, 64)"},
        },
    ),
    (
        Reshape24,
        [((64, 1, 13), torch.float32)],
        {
            "model_name": ["pt_stereo_facebook_musicgen_large_music_generation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(2, 32, 1, 13)"},
        },
    ),
    (
        Reshape25,
        [((2, 32, 1, 13), torch.float32)],
        {
            "model_name": ["pt_stereo_facebook_musicgen_large_music_generation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(64, 1, 13)"},
        },
    ),
    (
        Reshape26,
        [((2, 8192), torch.float32)],
        {
            "model_name": ["pt_stereo_facebook_musicgen_large_music_generation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(2, 1, 8192)"},
        },
    ),
    (
        Reshape27,
        [((2, 1, 8192), torch.float32)],
        {
            "model_name": ["pt_stereo_facebook_musicgen_large_music_generation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(2, 8192)"},
        },
    ),
    (
        Reshape28,
        [((2, 4, 1, 2048), torch.float32)],
        {
            "model_name": [
                "pt_stereo_facebook_musicgen_large_music_generation_hf",
                "pt_stereo_facebook_musicgen_small_music_generation_hf",
                "pt_stereo_facebook_musicgen_medium_music_generation_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(8, 1, 2048)"},
        },
    ),
    (
        Reshape29,
        [((1, 1024), torch.float32)],
        {
            "model_name": ["pt_stereo_facebook_musicgen_small_music_generation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1024)"},
        },
    ),
    (
        Reshape30,
        [((1, 1024), torch.float32)],
        {
            "model_name": [
                "pt_whisper_openai_whisper_medium_speech_recognition_hf",
                "pt_t5_t5_large_text_gen_hf",
                "pt_t5_google_flan_t5_large_text_gen_hf",
                "pt_t5_google_flan_t5_small_text_gen_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1, 1024)"},
        },
    ),
    (
        Reshape31,
        [((1, 1024), torch.float32)],
        {
            "model_name": [
                "pt_whisper_openai_whisper_medium_speech_recognition_hf",
                "pt_t5_t5_large_text_gen_hf",
                "pt_t5_google_flan_t5_large_text_gen_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1, 16, 64)"},
        },
    ),
    (
        Reshape32,
        [((2, 1, 1024), torch.float32)],
        {
            "model_name": ["pt_stereo_facebook_musicgen_small_music_generation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(2, 1024)"},
        },
    ),
    (
        Reshape33,
        [((2, 1, 1024), torch.float32)],
        {
            "model_name": ["pt_stereo_facebook_musicgen_small_music_generation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(2, 1, 16, 64)"},
        },
    ),
    (
        Reshape34,
        [((2, 1024), torch.float32)],
        {
            "model_name": ["pt_stereo_facebook_musicgen_small_music_generation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(2, 1, 1024)"},
        },
    ),
    (
        Reshape33,
        [((2, 1024), torch.float32)],
        {
            "model_name": ["pt_stereo_facebook_musicgen_small_music_generation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(2, 1, 16, 64)"},
        },
    ),
    (
        Reshape35,
        [((2, 16, 1, 64), torch.float32)],
        {
            "model_name": ["pt_stereo_facebook_musicgen_small_music_generation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(32, 1, 64)"},
        },
    ),
    (
        Reshape36,
        [((32, 1, 64), torch.float32)],
        {
            "model_name": ["pt_stereo_facebook_musicgen_small_music_generation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(2, 16, 1, 64)"},
        },
    ),
    (
        Reshape32,
        [((2, 1, 16, 64), torch.float32)],
        {
            "model_name": ["pt_stereo_facebook_musicgen_small_music_generation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(2, 1024)"},
        },
    ),
    (
        Reshape37,
        [((26, 1024), torch.float32)],
        {
            "model_name": ["pt_stereo_facebook_musicgen_small_music_generation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(2, 13, 1024)"},
        },
    ),
    (
        Reshape38,
        [((26, 1024), torch.float32)],
        {
            "model_name": ["pt_stereo_facebook_musicgen_small_music_generation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(2, 13, 16, 64)"},
        },
    ),
    (
        Reshape39,
        [((2, 13, 1024), torch.float32)],
        {
            "model_name": ["pt_stereo_facebook_musicgen_small_music_generation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(26, 1024)"},
        },
    ),
    (
        Reshape40,
        [((2, 16, 13, 64), torch.float32)],
        {
            "model_name": ["pt_stereo_facebook_musicgen_small_music_generation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(32, 13, 64)"},
        },
    ),
    (
        Reshape41,
        [((32, 1, 13), torch.float32)],
        {
            "model_name": ["pt_stereo_facebook_musicgen_small_music_generation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(2, 16, 1, 13)"},
        },
    ),
    (
        Reshape42,
        [((2, 16, 1, 13), torch.float32)],
        {
            "model_name": ["pt_stereo_facebook_musicgen_small_music_generation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(32, 1, 13)"},
        },
    ),
    (
        Reshape43,
        [((2, 4096), torch.float32)],
        {
            "model_name": ["pt_stereo_facebook_musicgen_small_music_generation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(2, 1, 4096)"},
        },
    ),
    (
        Reshape44,
        [((2, 1, 4096), torch.float32)],
        {
            "model_name": ["pt_stereo_facebook_musicgen_small_music_generation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(2, 4096)"},
        },
    ),
    (
        Reshape45,
        [((1, 1536), torch.float32)],
        {
            "model_name": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1536)"},
        },
    ),
    (
        Reshape46,
        [((2, 1, 1536), torch.float32)],
        {
            "model_name": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(2, 1536)"},
        },
    ),
    (
        Reshape47,
        [((2, 1, 1536), torch.float32)],
        {
            "model_name": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(2, 1, 24, 64)"},
        },
    ),
    (
        Reshape48,
        [((2, 1536), torch.float32)],
        {
            "model_name": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(2, 1, 1536)"},
        },
    ),
    (
        Reshape47,
        [((2, 1536), torch.float32)],
        {
            "model_name": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(2, 1, 24, 64)"},
        },
    ),
    (
        Reshape49,
        [((2, 24, 1, 64), torch.float32)],
        {
            "model_name": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(48, 1, 64)"},
        },
    ),
    (
        Reshape50,
        [((48, 1, 64), torch.float32)],
        {
            "model_name": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(2, 24, 1, 64)"},
        },
    ),
    (
        Reshape46,
        [((2, 1, 24, 64), torch.float32)],
        {
            "model_name": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(2, 1536)"},
        },
    ),
    (
        Reshape51,
        [((26, 1536), torch.float32)],
        {
            "model_name": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(2, 13, 1536)"},
        },
    ),
    (
        Reshape52,
        [((26, 1536), torch.float32)],
        {
            "model_name": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(2, 13, 24, 64)"},
        },
    ),
    (
        Reshape53,
        [((2, 13, 1536), torch.float32)],
        {
            "model_name": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(26, 1536)"},
        },
    ),
    (
        Reshape54,
        [((2, 24, 13, 64), torch.float32)],
        {
            "model_name": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(48, 13, 64)"},
        },
    ),
    (
        Reshape55,
        [((48, 1, 13), torch.float32)],
        {
            "model_name": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(2, 24, 1, 13)"},
        },
    ),
    (
        Reshape56,
        [((2, 24, 1, 13), torch.float32)],
        {
            "model_name": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(48, 1, 13)"},
        },
    ),
    (
        Reshape57,
        [((2, 6144), torch.float32)],
        {
            "model_name": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(2, 1, 6144)"},
        },
    ),
    (
        Reshape58,
        [((2, 1, 6144), torch.float32)],
        {
            "model_name": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(2, 6144)"},
        },
    ),
    pytest.param(
        (
            Reshape59,
            [((1, 1), torch.int64)],
            {
                "model_name": [
                    "pt_whisper_openai_whisper_tiny_speech_recognition_hf",
                    "pt_whisper_openai_whisper_small_speech_recognition_hf",
                    "pt_whisper_openai_whisper_medium_speech_recognition_hf",
                    "pt_whisper_openai_whisper_large_speech_recognition_hf",
                    "pt_whisper_openai_whisper_base_speech_recognition_hf",
                    "pt_t5_t5_large_text_gen_hf",
                    "pt_t5_t5_small_text_gen_hf",
                    "pt_t5_google_flan_t5_large_text_gen_hf",
                    "pt_t5_t5_base_text_gen_hf",
                    "pt_t5_google_flan_t5_small_text_gen_hf",
                    "pt_t5_google_flan_t5_base_text_gen_hf",
                ],
                "pcc": 0.99,
                "op_params": {"shape": "(1, 1)"},
            },
        ),
        marks=[pytest.mark.xfail(reason="RuntimeError: Long did not match Int")],
    ),
    (
        Reshape60,
        [((1, 1, 384), torch.float32)],
        {
            "model_name": [
                "pt_whisper_openai_whisper_tiny_speech_recognition_hf",
                "pt_deit_facebook_deit_small_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 384)"},
        },
    ),
    (
        Reshape61,
        [((1, 1, 384), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1, 6, 64)"},
        },
    ),
    (
        Reshape62,
        [((1, 384), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1, 384)"},
        },
    ),
    (
        Reshape61,
        [((1, 384), torch.float32)],
        {
            "model_name": [
                "pt_whisper_openai_whisper_tiny_speech_recognition_hf",
                "pt_t5_google_flan_t5_small_text_gen_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1, 6, 64)"},
        },
    ),
    (
        Reshape63,
        [((1, 6, 1, 64), torch.float32)],
        {
            "model_name": [
                "pt_whisper_openai_whisper_tiny_speech_recognition_hf",
                "pt_t5_google_flan_t5_small_text_gen_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(6, 1, 64)"},
        },
    ),
    (
        Reshape64,
        [((1, 6, 1, 64), torch.float32)],
        {
            "model_name": ["pt_falcon_tiiuae_falcon_7b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 6, 1, 64)"},
        },
    ),
    (
        Reshape65,
        [((6, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_whisper_openai_whisper_tiny_speech_recognition_hf",
                "pt_t5_google_flan_t5_small_text_gen_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 6, 1, 1)"},
        },
    ),
    (
        Reshape66,
        [((1, 6, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_whisper_openai_whisper_tiny_speech_recognition_hf",
                "pt_t5_google_flan_t5_small_text_gen_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(6, 1, 1)"},
        },
    ),
    (
        Reshape67,
        [((1, 6, 64, 1), torch.float32)],
        {
            "model_name": [
                "pt_whisper_openai_whisper_tiny_speech_recognition_hf",
                "pt_t5_google_flan_t5_small_text_gen_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(6, 64, 1)"},
        },
    ),
    (
        Reshape64,
        [((6, 1, 64), torch.float32)],
        {
            "model_name": [
                "pt_whisper_openai_whisper_tiny_speech_recognition_hf",
                "pt_t5_google_flan_t5_small_text_gen_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 6, 1, 64)"},
        },
    ),
    (
        Reshape60,
        [((1, 1, 6, 64), torch.float32)],
        {
            "model_name": [
                "pt_whisper_openai_whisper_tiny_speech_recognition_hf",
                "pt_t5_google_flan_t5_small_text_gen_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 384)"},
        },
    ),
    (
        Reshape61,
        [((1, 1, 6, 64), torch.float32)],
        {
            "model_name": ["pt_falcon_tiiuae_falcon_7b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1, 6, 64)"},
        },
    ),
    (
        Reshape68,
        [((1, 1, 6, 64), torch.float32)],
        {
            "model_name": ["pt_falcon_tiiuae_falcon_7b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 6, 64)"},
        },
    ),
    (
        Reshape69,
        [((1, 80, 3000), torch.float32)],
        {
            "model_name": [
                "pt_whisper_openai_whisper_tiny_speech_recognition_hf",
                "pt_whisper_openai_whisper_small_speech_recognition_hf",
                "pt_whisper_openai_whisper_medium_speech_recognition_hf",
                "pt_whisper_openai_whisper_large_speech_recognition_hf",
                "pt_whisper_openai_whisper_base_speech_recognition_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 80, 3000, 1)"},
        },
    ),
    (
        Reshape70,
        [((384, 80, 3), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(384, 80, 3, 1)"},
        },
    ),
    (
        Reshape71,
        [((1, 384, 3000, 1), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 384, 3000)"},
        },
    ),
    (
        Reshape72,
        [((1, 384, 3000), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 384, 3000, 1)"},
        },
    ),
    (
        Reshape73,
        [((384, 384, 3), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(384, 384, 3, 1)"},
        },
    ),
    (
        Reshape74,
        [((1, 384, 1500, 1), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 384, 1500)"},
        },
    ),
    (
        Reshape75,
        [((1, 1500, 384), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1500, 384)"},
        },
    ),
    (
        Reshape76,
        [((1, 1500, 384), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1500, 6, 64)"},
        },
    ),
    (
        Reshape77,
        [((1500, 384), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1500, 384)"},
        },
    ),
    (
        Reshape76,
        [((1500, 384), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1500, 6, 64)"},
        },
    ),
    (
        Reshape78,
        [((1, 6, 1500, 64), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(6, 1500, 64)"},
        },
    ),
    (
        Reshape79,
        [((6, 1500, 1500), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 6, 1500, 1500)"},
        },
    ),
    (
        Reshape80,
        [((1, 6, 1500, 1500), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(6, 1500, 1500)"},
        },
    ),
    (
        Reshape81,
        [((1, 6, 64, 1500), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(6, 64, 1500)"},
        },
    ),
    (
        Reshape82,
        [((6, 1500, 64), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 6, 1500, 64)"},
        },
    ),
    (
        Reshape75,
        [((1, 1500, 6, 64), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1500, 384)"},
        },
    ),
    (
        Reshape83,
        [((6, 1, 1500), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 6, 1, 1500)"},
        },
    ),
    (
        Reshape84,
        [((1, 6, 1, 1500), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(6, 1, 1500)"},
        },
    ),
    (
        Reshape85,
        [((1, 1, 768), torch.float32)],
        {
            "model_name": [
                "pt_whisper_openai_whisper_small_speech_recognition_hf",
                "pt_vilt_dandelin_vilt_b32_mlm_mlm_hf",
                "pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf",
                "pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf",
                "pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf",
                "pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf",
                "pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf",
                "pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader",
                "pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf",
                "pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf",
                "pt_t5_t5_base_text_gen_hf",
                "pt_t5_google_flan_t5_base_text_gen_hf",
                "pt_deit_facebook_deit_base_patch16_224_img_cls_hf",
                "pt_deit_facebook_deit_base_distilled_patch16_224_img_cls_hf",
                "pt_vit_google_vit_base_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 768)"},
        },
    ),
    (
        Reshape86,
        [((1, 1, 768), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1, 12, 64)"},
        },
    ),
    (
        Reshape87,
        [((1, 768), torch.float32)],
        {
            "model_name": [
                "pt_whisper_openai_whisper_small_speech_recognition_hf",
                "pt_t5_t5_base_text_gen_hf",
                "pt_t5_google_flan_t5_base_text_gen_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1, 768)"},
        },
    ),
    (
        Reshape86,
        [((1, 768), torch.float32)],
        {
            "model_name": [
                "pt_whisper_openai_whisper_small_speech_recognition_hf",
                "pt_t5_t5_base_text_gen_hf",
                "pt_t5_google_flan_t5_base_text_gen_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1, 12, 64)"},
        },
    ),
    (
        Reshape88,
        [((1, 12, 1, 64), torch.float32)],
        {
            "model_name": [
                "pt_whisper_openai_whisper_small_speech_recognition_hf",
                "pt_t5_t5_base_text_gen_hf",
                "pt_t5_google_flan_t5_base_text_gen_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(12, 1, 64)"},
        },
    ),
    (
        Reshape89,
        [((12, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_whisper_openai_whisper_small_speech_recognition_hf",
                "pt_t5_t5_base_text_gen_hf",
                "pt_t5_google_flan_t5_base_text_gen_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 12, 1, 1)"},
        },
    ),
    (
        Reshape90,
        [((1, 12, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_whisper_openai_whisper_small_speech_recognition_hf",
                "pt_t5_t5_base_text_gen_hf",
                "pt_t5_google_flan_t5_base_text_gen_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(12, 1, 1)"},
        },
    ),
    (
        Reshape91,
        [((1, 12, 64, 1), torch.float32)],
        {
            "model_name": [
                "pt_whisper_openai_whisper_small_speech_recognition_hf",
                "pt_t5_t5_base_text_gen_hf",
                "pt_t5_google_flan_t5_base_text_gen_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(12, 64, 1)"},
        },
    ),
    (
        Reshape92,
        [((12, 1, 64), torch.float32)],
        {
            "model_name": [
                "pt_whisper_openai_whisper_small_speech_recognition_hf",
                "pt_t5_t5_base_text_gen_hf",
                "pt_t5_google_flan_t5_base_text_gen_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 12, 1, 64)"},
        },
    ),
    (
        Reshape85,
        [((1, 1, 12, 64), torch.float32)],
        {
            "model_name": [
                "pt_whisper_openai_whisper_small_speech_recognition_hf",
                "pt_t5_t5_base_text_gen_hf",
                "pt_t5_google_flan_t5_base_text_gen_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 768)"},
        },
    ),
    (
        Reshape93,
        [((768, 80, 3), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(768, 80, 3, 1)"},
        },
    ),
    (
        Reshape94,
        [((1, 768, 3000, 1), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 768, 3000)"},
        },
    ),
    (
        Reshape95,
        [((1, 768, 3000), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 768, 3000, 1)"},
        },
    ),
    (
        Reshape96,
        [((768, 768, 3), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(768, 768, 3, 1)"},
        },
    ),
    (
        Reshape97,
        [((1, 768, 1500, 1), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 768, 1500)"},
        },
    ),
    (
        Reshape98,
        [((1, 1500, 768), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1500, 768)"},
        },
    ),
    (
        Reshape99,
        [((1, 1500, 768), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1500, 12, 64)"},
        },
    ),
    (
        Reshape100,
        [((1500, 768), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1500, 768)"},
        },
    ),
    (
        Reshape99,
        [((1500, 768), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1500, 12, 64)"},
        },
    ),
    (
        Reshape101,
        [((1, 12, 1500, 64), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(12, 1500, 64)"},
        },
    ),
    (
        Reshape102,
        [((12, 1500, 1500), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 12, 1500, 1500)"},
        },
    ),
    (
        Reshape103,
        [((1, 12, 1500, 1500), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(12, 1500, 1500)"},
        },
    ),
    (
        Reshape104,
        [((1, 12, 64, 1500), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(12, 64, 1500)"},
        },
    ),
    (
        Reshape105,
        [((12, 1500, 64), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 12, 1500, 64)"},
        },
    ),
    (
        Reshape98,
        [((1, 1500, 12, 64), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1500, 768)"},
        },
    ),
    (
        Reshape106,
        [((12, 1, 1500), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 12, 1, 1500)"},
        },
    ),
    (
        Reshape107,
        [((1, 12, 1, 1500), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(12, 1, 1500)"},
        },
    ),
    (
        Reshape29,
        [((1, 1, 1024), torch.float32)],
        {
            "model_name": [
                "pt_whisper_openai_whisper_medium_speech_recognition_hf",
                "pt_t5_t5_large_text_gen_hf",
                "pt_t5_google_flan_t5_large_text_gen_hf",
                "pt_vit_google_vit_large_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1024)"},
        },
    ),
    (
        Reshape31,
        [((1, 1, 1024), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_medium_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1, 16, 64)"},
        },
    ),
    (
        Reshape108,
        [((1, 1, 1024), torch.float32)],
        {
            "model_name": [
                "pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1, 1, 1024)"},
        },
    ),
    (
        Reshape109,
        [((1, 16, 1, 64), torch.float32)],
        {
            "model_name": [
                "pt_whisper_openai_whisper_medium_speech_recognition_hf",
                "pt_t5_t5_large_text_gen_hf",
                "pt_t5_google_flan_t5_large_text_gen_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(16, 1, 64)"},
        },
    ),
    (
        Reshape110,
        [((16, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_whisper_openai_whisper_medium_speech_recognition_hf",
                "pt_t5_t5_large_text_gen_hf",
                "pt_t5_google_flan_t5_large_text_gen_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 16, 1, 1)"},
        },
    ),
    (
        Reshape111,
        [((1, 16, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_whisper_openai_whisper_medium_speech_recognition_hf",
                "pt_t5_t5_large_text_gen_hf",
                "pt_t5_google_flan_t5_large_text_gen_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(16, 1, 1)"},
        },
    ),
    (
        Reshape112,
        [((1, 16, 1, 1), torch.float32)],
        {"model_name": ["pt_ssd300_resnet50_base_img_cls_torchhub"], "pcc": 0.99, "op_params": {"shape": "(1, 4, 4)"}},
    ),
    (
        Reshape113,
        [((1, 16, 64, 1), torch.float32)],
        {
            "model_name": [
                "pt_whisper_openai_whisper_medium_speech_recognition_hf",
                "pt_t5_t5_large_text_gen_hf",
                "pt_t5_google_flan_t5_large_text_gen_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(16, 64, 1)"},
        },
    ),
    (
        Reshape114,
        [((16, 1, 64), torch.float32)],
        {
            "model_name": [
                "pt_whisper_openai_whisper_medium_speech_recognition_hf",
                "pt_t5_t5_large_text_gen_hf",
                "pt_t5_google_flan_t5_large_text_gen_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 16, 1, 64)"},
        },
    ),
    (
        Reshape29,
        [((1, 1, 16, 64), torch.float32)],
        {
            "model_name": [
                "pt_whisper_openai_whisper_medium_speech_recognition_hf",
                "pt_t5_t5_large_text_gen_hf",
                "pt_t5_google_flan_t5_large_text_gen_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1024)"},
        },
    ),
    (
        Reshape115,
        [((1024, 80, 3), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_medium_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1024, 80, 3, 1)"},
        },
    ),
    (
        Reshape116,
        [((1, 1024, 3000, 1), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_medium_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1024, 3000)"},
        },
    ),
    (
        Reshape117,
        [((1, 1024, 3000), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_medium_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1024, 3000, 1)"},
        },
    ),
    pytest.param(
        (
            Reshape118,
            [((1024, 1024, 3), torch.float32)],
            {
                "model_name": ["pt_whisper_openai_whisper_medium_speech_recognition_hf"],
                "pcc": 0.99,
                "op_params": {"shape": "(1024, 1024, 3, 1)"},
            },
        ),
        marks=[pytest.mark.skip(reason="Segmentation fault occurs while executing ttnn binary")],
    ),
    (
        Reshape119,
        [((1, 1024, 1500, 1), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_medium_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1024, 1500)"},
        },
    ),
    (
        Reshape120,
        [((1, 1500, 1024), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_medium_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1500, 1024)"},
        },
    ),
    (
        Reshape121,
        [((1, 1500, 1024), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_medium_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1500, 16, 64)"},
        },
    ),
    (
        Reshape122,
        [((1500, 1024), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_medium_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1500, 1024)"},
        },
    ),
    (
        Reshape121,
        [((1500, 1024), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_medium_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1500, 16, 64)"},
        },
    ),
    (
        Reshape123,
        [((1, 16, 1500, 64), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_medium_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(16, 1500, 64)"},
        },
    ),
    (
        Reshape124,
        [((16, 1500, 1500), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_medium_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 16, 1500, 1500)"},
        },
    ),
    (
        Reshape125,
        [((1, 16, 1500, 1500), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_medium_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(16, 1500, 1500)"},
        },
    ),
    (
        Reshape126,
        [((1, 16, 64, 1500), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_medium_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(16, 64, 1500)"},
        },
    ),
    (
        Reshape127,
        [((16, 1500, 64), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_medium_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 16, 1500, 64)"},
        },
    ),
    (
        Reshape120,
        [((1, 1500, 16, 64), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_medium_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1500, 1024)"},
        },
    ),
    (
        Reshape128,
        [((16, 1, 1500), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_medium_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 16, 1, 1500)"},
        },
    ),
    (
        Reshape129,
        [((1, 16, 1, 1500), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_medium_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(16, 1, 1500)"},
        },
    ),
    (
        Reshape130,
        [((1, 1, 1280), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_large_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1280)"},
        },
    ),
    (
        Reshape131,
        [((1, 1, 1280), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_large_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1, 20, 64)"},
        },
    ),
    (
        Reshape132,
        [((1, 1280), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_large_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1, 1280)"},
        },
    ),
    (
        Reshape131,
        [((1, 1280), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_large_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1, 20, 64)"},
        },
    ),
    (
        Reshape133,
        [((1, 20, 1, 64), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_large_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(20, 1, 64)"},
        },
    ),
    (
        Reshape134,
        [((20, 1, 1), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_large_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 20, 1, 1)"},
        },
    ),
    (
        Reshape135,
        [((1, 20, 1, 1), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_large_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(20, 1, 1)"},
        },
    ),
    (
        Reshape136,
        [((1, 20, 64, 1), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_large_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(20, 64, 1)"},
        },
    ),
    (
        Reshape137,
        [((20, 1, 64), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_large_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 20, 1, 64)"},
        },
    ),
    (
        Reshape130,
        [((1, 1, 20, 64), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_large_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1280)"},
        },
    ),
    (
        Reshape138,
        [((1280, 80, 3), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_large_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1280, 80, 3, 1)"},
        },
    ),
    (
        Reshape139,
        [((1, 1280, 3000, 1), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_large_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1280, 3000)"},
        },
    ),
    (
        Reshape140,
        [((1, 1280, 3000), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_large_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1280, 3000, 1)"},
        },
    ),
    pytest.param(
        (
            Reshape141,
            [((1280, 1280, 3), torch.float32)],
            {
                "model_name": ["pt_whisper_openai_whisper_large_speech_recognition_hf"],
                "pcc": 0.99,
                "op_params": {"shape": "(1280, 1280, 3, 1)"},
            },
        ),
        marks=[pytest.mark.skip(reason="Segmentation fault occurs while executing ttnn binary")],
    ),
    (
        Reshape142,
        [((1, 1280, 1500, 1), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_large_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1280, 1500)"},
        },
    ),
    (
        Reshape143,
        [((1, 1500, 1280), torch.float32)],
        {
            "model_name": [
                "pt_whisper_openai_whisper_large_speech_recognition_hf",
                "pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1500, 1280)"},
        },
    ),
    (
        Reshape144,
        [((1, 1500, 1280), torch.float32)],
        {
            "model_name": [
                "pt_whisper_openai_whisper_large_speech_recognition_hf",
                "pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1500, 20, 64)"},
        },
    ),
    (
        Reshape145,
        [((1500, 1280), torch.float32)],
        {
            "model_name": [
                "pt_whisper_openai_whisper_large_speech_recognition_hf",
                "pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1500, 1280)"},
        },
    ),
    (
        Reshape144,
        [((1500, 1280), torch.float32)],
        {
            "model_name": [
                "pt_whisper_openai_whisper_large_speech_recognition_hf",
                "pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1500, 20, 64)"},
        },
    ),
    (
        Reshape146,
        [((1, 20, 1500, 64), torch.float32)],
        {
            "model_name": [
                "pt_whisper_openai_whisper_large_speech_recognition_hf",
                "pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(20, 1500, 64)"},
        },
    ),
    (
        Reshape147,
        [((20, 1500, 1500), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_large_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 20, 1500, 1500)"},
        },
    ),
    (
        Reshape148,
        [((1, 20, 1500, 1500), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_large_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(20, 1500, 1500)"},
        },
    ),
    (
        Reshape149,
        [((1, 20, 64, 1500), torch.float32)],
        {
            "model_name": [
                "pt_whisper_openai_whisper_large_speech_recognition_hf",
                "pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(20, 64, 1500)"},
        },
    ),
    (
        Reshape150,
        [((20, 1500, 64), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_large_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 20, 1500, 64)"},
        },
    ),
    (
        Reshape143,
        [((1, 1500, 20, 64), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_large_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1500, 1280)"},
        },
    ),
    (
        Reshape151,
        [((20, 1, 1500), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_large_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 20, 1, 1500)"},
        },
    ),
    (
        Reshape152,
        [((1, 20, 1, 1500), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_large_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(20, 1, 1500)"},
        },
    ),
    (
        Reshape153,
        [((1, 1, 512), torch.float32)],
        {
            "model_name": [
                "pt_whisper_openai_whisper_base_speech_recognition_hf",
                "pt_t5_t5_small_text_gen_hf",
                "pt_t5_google_flan_t5_small_text_gen_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 512)"},
        },
    ),
    (
        Reshape154,
        [((1, 1, 512), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_base_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1, 8, 64)"},
        },
    ),
    (
        Reshape155,
        [((1, 1, 512), torch.float32)],
        {
            "model_name": [
                "pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1, 1, 512)"},
        },
    ),
    (
        Reshape156,
        [((1, 512), torch.float32)],
        {
            "model_name": [
                "pt_whisper_openai_whisper_base_speech_recognition_hf",
                "pt_t5_t5_small_text_gen_hf",
                "pt_t5_google_flan_t5_small_text_gen_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1, 512)"},
        },
    ),
    (
        Reshape154,
        [((1, 512), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_base_speech_recognition_hf", "pt_t5_t5_small_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1, 8, 64)"},
        },
    ),
    (
        Reshape157,
        [((1, 8, 1, 64), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_base_speech_recognition_hf", "pt_t5_t5_small_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(8, 1, 64)"},
        },
    ),
    (
        Reshape158,
        [((8, 1, 1), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_base_speech_recognition_hf", "pt_t5_t5_small_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 8, 1, 1)"},
        },
    ),
    (
        Reshape159,
        [((1, 8, 1, 1), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_base_speech_recognition_hf", "pt_t5_t5_small_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(8, 1, 1)"},
        },
    ),
    (
        Reshape160,
        [((1, 8, 64, 1), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_base_speech_recognition_hf", "pt_t5_t5_small_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(8, 64, 1)"},
        },
    ),
    (
        Reshape161,
        [((8, 1, 64), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_base_speech_recognition_hf", "pt_t5_t5_small_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 8, 1, 64)"},
        },
    ),
    (
        Reshape153,
        [((1, 1, 8, 64), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_base_speech_recognition_hf", "pt_t5_t5_small_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 512)"},
        },
    ),
    (
        Reshape162,
        [((512, 80, 3), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_base_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(512, 80, 3, 1)"},
        },
    ),
    (
        Reshape163,
        [((1, 512, 3000, 1), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_base_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 512, 3000)"},
        },
    ),
    (
        Reshape164,
        [((1, 512, 3000), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_base_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 512, 3000, 1)"},
        },
    ),
    (
        Reshape165,
        [((512, 512, 3), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_base_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(512, 512, 3, 1)"},
        },
    ),
    (
        Reshape166,
        [((1, 512, 1500, 1), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_base_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 512, 1500)"},
        },
    ),
    (
        Reshape167,
        [((1, 1500, 512), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_base_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1500, 512)"},
        },
    ),
    (
        Reshape168,
        [((1, 1500, 512), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_base_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1500, 8, 64)"},
        },
    ),
    (
        Reshape169,
        [((1500, 512), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_base_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1500, 512)"},
        },
    ),
    (
        Reshape168,
        [((1500, 512), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_base_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1500, 8, 64)"},
        },
    ),
    (
        Reshape170,
        [((1, 8, 1500, 64), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_base_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(8, 1500, 64)"},
        },
    ),
    (
        Reshape171,
        [((8, 1500, 1500), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_base_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 8, 1500, 1500)"},
        },
    ),
    (
        Reshape172,
        [((1, 8, 1500, 1500), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_base_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(8, 1500, 1500)"},
        },
    ),
    (
        Reshape173,
        [((1, 8, 64, 1500), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_base_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(8, 64, 1500)"},
        },
    ),
    (
        Reshape174,
        [((8, 1500, 64), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_base_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 8, 1500, 64)"},
        },
    ),
    (
        Reshape167,
        [((1, 1500, 8, 64), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_base_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1500, 512)"},
        },
    ),
    (
        Reshape175,
        [((8, 1, 1500), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_base_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 8, 1, 1500)"},
        },
    ),
    (
        Reshape176,
        [((1, 8, 1, 1500), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_base_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(8, 1, 1500)"},
        },
    ),
    (
        Reshape177,
        [((1, 2), torch.int64)],
        {
            "model_name": ["pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 2)"},
        },
    ),
    (
        Reshape178,
        [((1, 2, 1280), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(2, 1280)"},
        },
    ),
    (
        Reshape179,
        [((1, 2, 1280), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 2, 20, 64)"},
        },
    ),
    (
        Reshape180,
        [((2, 1280), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 2, 1280)"},
        },
    ),
    (
        Reshape179,
        [((2, 1280), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 2, 20, 64)"},
        },
    ),
    (
        Reshape181,
        [((1, 20, 2, 64), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(20, 2, 64)"},
        },
    ),
    (
        Reshape182,
        [((20, 2, 2), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 20, 2, 2)"},
        },
    ),
    (
        Reshape183,
        [((1, 20, 2, 2), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(20, 2, 2)"},
        },
    ),
    (
        Reshape184,
        [((1, 20, 64, 2), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(20, 64, 2)"},
        },
    ),
    (
        Reshape185,
        [((20, 2, 64), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 20, 2, 64)"},
        },
    ),
    (
        Reshape178,
        [((1, 2, 20, 64), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(2, 1280)"},
        },
    ),
    (
        Reshape186,
        [((20, 2, 1500), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 20, 2, 1500)"},
        },
    ),
    (
        Reshape187,
        [((1, 20, 2, 1500), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(20, 2, 1500)"},
        },
    ),
    (
        Reshape188,
        [((2, 7), torch.int64)],
        {
            "model_name": ["pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text"],
            "pcc": 0.99,
            "op_params": {"shape": "(2, 7)"},
        },
    ),
    (
        Reshape189,
        [((2, 7, 512), torch.float32)],
        {
            "model_name": ["pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text"],
            "pcc": 0.99,
            "op_params": {"shape": "(14, 512)"},
        },
    ),
    (
        Reshape190,
        [((2, 7, 512), torch.float32)],
        {
            "model_name": ["pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text"],
            "pcc": 0.99,
            "op_params": {"shape": "(2, 7, 8, 64)"},
        },
    ),
    (
        Reshape191,
        [((14, 512), torch.float32)],
        {
            "model_name": ["pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text"],
            "pcc": 0.99,
            "op_params": {"shape": "(2, 7, 512)"},
        },
    ),
    (
        Reshape192,
        [((2, 8, 7, 64), torch.float32)],
        {
            "model_name": ["pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text"],
            "pcc": 0.99,
            "op_params": {"shape": "(16, 7, 64)"},
        },
    ),
    (
        Reshape193,
        [((16, 7, 7), torch.float32)],
        {
            "model_name": ["pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text"],
            "pcc": 0.99,
            "op_params": {"shape": "(2, 8, 7, 7)"},
        },
    ),
    (
        Reshape193,
        [((2, 8, 7, 7), torch.float32)],
        {
            "model_name": ["pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text"],
            "pcc": 0.99,
            "op_params": {"shape": "(2, 8, 7, 7)"},
        },
    ),
    (
        Reshape194,
        [((2, 8, 7, 7), torch.float32)],
        {
            "model_name": ["pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text"],
            "pcc": 0.99,
            "op_params": {"shape": "(16, 7, 7)"},
        },
    ),
    (
        Reshape195,
        [((16, 7, 64), torch.float32)],
        {
            "model_name": ["pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text"],
            "pcc": 0.99,
            "op_params": {"shape": "(2, 8, 7, 64)"},
        },
    ),
    (
        Reshape189,
        [((2, 7, 8, 64), torch.float32)],
        {
            "model_name": ["pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text"],
            "pcc": 0.99,
            "op_params": {"shape": "(14, 512)"},
        },
    ),
    (
        Reshape196,
        [((14, 2048), torch.float32)],
        {
            "model_name": ["pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text"],
            "pcc": 0.99,
            "op_params": {"shape": "(2, 7, 2048)"},
        },
    ),
    (
        Reshape197,
        [((2, 7, 2048), torch.float32)],
        {
            "model_name": ["pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text"],
            "pcc": 0.99,
            "op_params": {"shape": "(14, 2048)"},
        },
    ),
    (
        Reshape198,
        [((1, 588, 2048), torch.float32)],
        {
            "model_name": ["pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(588, 2048)"},
        },
    ),
    (
        Reshape199,
        [((588, 2048), torch.float32)],
        {
            "model_name": ["pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 588, 16, 128)"},
        },
    ),
    (
        Reshape200,
        [((588, 2048), torch.float32)],
        {
            "model_name": ["pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 588, 2048)"},
        },
    ),
    (
        Reshape201,
        [((1, 16, 588, 128), torch.float32)],
        {
            "model_name": ["pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(16, 588, 128)"},
        },
    ),
    (
        Reshape202,
        [((16, 588, 588), torch.float32)],
        {
            "model_name": ["pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 16, 588, 588)"},
        },
    ),
    (
        Reshape203,
        [((1, 16, 588, 588), torch.float32)],
        {
            "model_name": ["pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(16, 588, 588)"},
        },
    ),
    (
        Reshape204,
        [((1, 16, 128, 588), torch.float32)],
        {
            "model_name": ["pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(16, 128, 588)"},
        },
    ),
    (
        Reshape205,
        [((16, 588, 128), torch.float32)],
        {
            "model_name": ["pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 16, 588, 128)"},
        },
    ),
    (
        Reshape198,
        [((1, 588, 16, 128), torch.float32)],
        {
            "model_name": ["pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(588, 2048)"},
        },
    ),
    (
        Reshape206,
        [((588, 5504), torch.float32)],
        {
            "model_name": ["pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 588, 5504)"},
        },
    ),
    (
        Reshape207,
        [((1, 39, 4096), torch.float32)],
        {
            "model_name": ["pt_deepseek_deepseek_math_7b_instruct_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(39, 4096)"},
        },
    ),
    (
        Reshape208,
        [((39, 4096), torch.float32)],
        {
            "model_name": ["pt_deepseek_deepseek_math_7b_instruct_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 39, 32, 128)"},
        },
    ),
    (
        Reshape209,
        [((39, 4096), torch.float32)],
        {
            "model_name": ["pt_deepseek_deepseek_math_7b_instruct_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 39, 4096)"},
        },
    ),
    (
        Reshape210,
        [((1, 32, 39, 128), torch.float32)],
        {
            "model_name": ["pt_deepseek_deepseek_math_7b_instruct_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(32, 39, 128)"},
        },
    ),
    (
        Reshape211,
        [((32, 39, 39), torch.float32)],
        {
            "model_name": ["pt_deepseek_deepseek_math_7b_instruct_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 32, 39, 39)"},
        },
    ),
    (
        Reshape212,
        [((1, 32, 39, 39), torch.float32)],
        {
            "model_name": ["pt_deepseek_deepseek_math_7b_instruct_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(32, 39, 39)"},
        },
    ),
    (
        Reshape213,
        [((1, 32, 128, 39), torch.float32)],
        {
            "model_name": ["pt_deepseek_deepseek_math_7b_instruct_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(32, 128, 39)"},
        },
    ),
    (
        Reshape214,
        [((32, 39, 128), torch.float32)],
        {
            "model_name": ["pt_deepseek_deepseek_math_7b_instruct_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 32, 39, 128)"},
        },
    ),
    (
        Reshape207,
        [((1, 39, 32, 128), torch.float32)],
        {
            "model_name": ["pt_deepseek_deepseek_math_7b_instruct_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(39, 4096)"},
        },
    ),
    (
        Reshape215,
        [((39, 11008), torch.float32)],
        {
            "model_name": ["pt_deepseek_deepseek_math_7b_instruct_qa_hf", "pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 39, 11008)"},
        },
    ),
    pytest.param(
        (
            Reshape216,
            [((1, 596, 4096), torch.float32)],
            {
                "model_name": ["pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf"],
                "pcc": 0.99,
                "op_params": {"shape": "(2441216,)"},
            },
        ),
        marks=[
            pytest.mark.xfail(
                reason="RuntimeError: TT_THROW @ /__w/tt-forge-fe/tt-forge-fe/third_party/tt-mlir/third_party/tt-metal/src/tt-metal/tt_metal/impl/program/program.cpp:903: tt::exception info: Statically allocated circular buffers on core range [(x=0,y=0) - (x=7,y=7)] grow to 9899888 B which is beyond max L1 size of 1499136 B"
            )
        ],
    ),
    (
        Reshape217,
        [((1, 596, 4096), torch.float32)],
        {
            "model_name": ["pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(596, 4096)"},
        },
    ),
    (
        Reshape218,
        [((1, 1024, 24, 24), torch.float32)],
        {
            "model_name": ["pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1024, 576, 1)"},
        },
    ),
    (
        Reshape219,
        [((1, 577, 1024), torch.float32)],
        {
            "model_name": ["pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(577, 1024)"},
        },
    ),
    (
        Reshape220,
        [((1, 577, 1024), torch.float32)],
        {
            "model_name": ["pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 577, 16, 64)"},
        },
    ),
    (
        Reshape221,
        [((577, 1024), torch.float32)],
        {
            "model_name": ["pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 577, 1024)"},
        },
    ),
    (
        Reshape222,
        [((1, 16, 577, 64), torch.float32)],
        {
            "model_name": ["pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(16, 577, 64)"},
        },
    ),
    (
        Reshape223,
        [((16, 577, 64), torch.float32)],
        {
            "model_name": ["pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 16, 577, 64)"},
        },
    ),
    (
        Reshape219,
        [((1, 577, 16, 64), torch.float32)],
        {
            "model_name": ["pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(577, 1024)"},
        },
    ),
    pytest.param(
        (
            Reshape224,
            [((1, 576, 4096), torch.float32)],
            {
                "model_name": ["pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf"],
                "pcc": 0.99,
                "op_params": {"shape": "(2359296,)"},
            },
        ),
        marks=[
            pytest.mark.xfail(
                reason="RuntimeError: TT_THROW @ /__w/tt-forge-fe/tt-forge-fe/third_party/tt-mlir/third_party/tt-metal/src/tt-metal/tt_metal/impl/program/program.cpp:903: tt::exception info: Statically allocated circular buffers on core range [(x=0,y=0) - (x=7,y=7)] grow to 9572208 B which is beyond max L1 size of 1499136 B"
            )
        ],
    ),
    (
        Reshape216,
        [((1, 2441216), torch.int32)],
        {
            "model_name": ["pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(2441216,)"},
        },
    ),
    (
        Reshape216,
        [((2441216,), torch.float32)],
        {
            "model_name": ["pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(2441216,)"},
        },
    ),
    pytest.param(
        (
            Reshape225,
            [((2441216,), torch.float32)],
            {
                "model_name": ["pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf"],
                "pcc": 0.99,
                "op_params": {"shape": "(1, 596, 4096)"},
            },
        ),
        marks=[
            pytest.mark.xfail(
                reason="RuntimeError: TT_THROW @ /__w/tt-forge-fe/tt-forge-fe/third_party/tt-mlir/third_party/tt-metal/src/tt-metal/tt_metal/impl/program/program.cpp:903: tt::exception info: Statically allocated circular buffers on core range [(x=6,y=7) - (x=6,y=7)] grow to 9981664 B which is beyond max L1 size of 1499136 B"
            )
        ],
    ),
    (
        Reshape226,
        [((596, 4096), torch.float32)],
        {
            "model_name": ["pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 596, 32, 128)"},
        },
    ),
    (
        Reshape225,
        [((596, 4096), torch.float32)],
        {
            "model_name": ["pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 596, 4096)"},
        },
    ),
    (
        Reshape227,
        [((1, 32, 596, 128), torch.float32)],
        {
            "model_name": ["pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(32, 596, 128)"},
        },
    ),
    (
        Reshape228,
        [((32, 596, 596), torch.float32)],
        {
            "model_name": ["pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 32, 596, 596)"},
        },
    ),
    (
        Reshape229,
        [((1, 32, 596, 596), torch.float32)],
        {
            "model_name": ["pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(32, 596, 596)"},
        },
    ),
    (
        Reshape230,
        [((1, 32, 128, 596), torch.float32)],
        {
            "model_name": ["pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(32, 128, 596)"},
        },
    ),
    (
        Reshape231,
        [((32, 596, 128), torch.float32)],
        {
            "model_name": ["pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 32, 596, 128)"},
        },
    ),
    (
        Reshape217,
        [((1, 596, 32, 128), torch.float32)],
        {
            "model_name": ["pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(596, 4096)"},
        },
    ),
    (
        Reshape232,
        [((596, 11008), torch.float32)],
        {
            "model_name": ["pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 596, 11008)"},
        },
    ),
    (
        Reshape233,
        [((1, 204, 768), torch.float32)],
        {"model_name": ["pt_vilt_dandelin_vilt_b32_mlm_mlm_hf"], "pcc": 0.99, "op_params": {"shape": "(204, 768)"}},
    ),
    (
        Reshape234,
        [((1, 204, 768), torch.float32)],
        {
            "model_name": ["pt_vilt_dandelin_vilt_b32_mlm_mlm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 204, 12, 64)"},
        },
    ),
    (
        Reshape235,
        [((204, 768), torch.float32)],
        {"model_name": ["pt_vilt_dandelin_vilt_b32_mlm_mlm_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 204, 768)"}},
    ),
    (
        Reshape236,
        [((1, 12, 204, 64), torch.float32)],
        {"model_name": ["pt_vilt_dandelin_vilt_b32_mlm_mlm_hf"], "pcc": 0.99, "op_params": {"shape": "(12, 204, 64)"}},
    ),
    (
        Reshape237,
        [((12, 204, 204), torch.float32)],
        {
            "model_name": ["pt_vilt_dandelin_vilt_b32_mlm_mlm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 12, 204, 204)"},
        },
    ),
    (
        Reshape238,
        [((1, 12, 204, 204), torch.float32)],
        {"model_name": ["pt_vilt_dandelin_vilt_b32_mlm_mlm_hf"], "pcc": 0.99, "op_params": {"shape": "(12, 204, 204)"}},
    ),
    (
        Reshape239,
        [((1, 12, 64, 204), torch.float32)],
        {"model_name": ["pt_vilt_dandelin_vilt_b32_mlm_mlm_hf"], "pcc": 0.99, "op_params": {"shape": "(12, 64, 204)"}},
    ),
    (
        Reshape240,
        [((12, 204, 64), torch.float32)],
        {
            "model_name": ["pt_vilt_dandelin_vilt_b32_mlm_mlm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 12, 204, 64)"},
        },
    ),
    (
        Reshape233,
        [((1, 204, 12, 64), torch.float32)],
        {"model_name": ["pt_vilt_dandelin_vilt_b32_mlm_mlm_hf"], "pcc": 0.99, "op_params": {"shape": "(204, 768)"}},
    ),
    (
        Reshape241,
        [((1, 201, 768), torch.float32)],
        {
            "model_name": ["pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(201, 768)"},
        },
    ),
    (
        Reshape242,
        [((1, 201, 768), torch.float32)],
        {
            "model_name": ["pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 201, 12, 64)"},
        },
    ),
    (
        Reshape243,
        [((201, 768), torch.float32)],
        {
            "model_name": ["pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 201, 768)"},
        },
    ),
    (
        Reshape244,
        [((1, 12, 201, 64), torch.float32)],
        {
            "model_name": ["pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(12, 201, 64)"},
        },
    ),
    (
        Reshape245,
        [((12, 201, 201), torch.float32)],
        {
            "model_name": ["pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 12, 201, 201)"},
        },
    ),
    (
        Reshape246,
        [((1, 12, 201, 201), torch.float32)],
        {
            "model_name": ["pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(12, 201, 201)"},
        },
    ),
    (
        Reshape247,
        [((1, 12, 64, 201), torch.float32)],
        {
            "model_name": ["pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(12, 64, 201)"},
        },
    ),
    (
        Reshape248,
        [((12, 201, 64), torch.float32)],
        {
            "model_name": ["pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 12, 201, 64)"},
        },
    ),
    (
        Reshape241,
        [((1, 201, 12, 64), torch.float32)],
        {
            "model_name": ["pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(201, 768)"},
        },
    ),
    (
        Reshape249,
        [((1, 128, 2048), torch.float32)],
        {
            "model_name": [
                "pt_albert_xlarge_v1_mlm_hf",
                "pt_albert_xlarge_v1_token_cls_hf",
                "pt_albert_xlarge_v2_mlm_hf",
                "pt_albert_xlarge_v2_token_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(128, 2048)"},
        },
    ),
    (
        Reshape250,
        [((1, 128, 2048), torch.float32)],
        {
            "model_name": [
                "pt_albert_xlarge_v1_mlm_hf",
                "pt_albert_xlarge_v1_token_cls_hf",
                "pt_albert_xlarge_v2_mlm_hf",
                "pt_albert_xlarge_v2_token_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 128, 16, 128)"},
        },
    ),
    (
        Reshape251,
        [((128, 2048), torch.float32)],
        {
            "model_name": [
                "pt_albert_xlarge_v1_mlm_hf",
                "pt_albert_xlarge_v1_token_cls_hf",
                "pt_albert_xlarge_v2_mlm_hf",
                "pt_albert_xlarge_v2_token_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 128, 2048)"},
        },
    ),
    (
        Reshape252,
        [((1, 16, 128, 128), torch.float32)],
        {
            "model_name": [
                "pt_albert_xlarge_v1_mlm_hf",
                "pt_albert_large_v2_token_cls_hf",
                "pt_albert_large_v2_mlm_hf",
                "pt_albert_large_v1_mlm_hf",
                "pt_albert_large_v1_token_cls_hf",
                "pt_albert_xlarge_v1_token_cls_hf",
                "pt_albert_xlarge_v2_mlm_hf",
                "pt_albert_xlarge_v2_token_cls_hf",
                "pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(16, 128, 128)"},
        },
    ),
    (
        Reshape253,
        [((16, 128, 128), torch.float32)],
        {
            "model_name": [
                "pt_albert_xlarge_v1_mlm_hf",
                "pt_albert_large_v2_token_cls_hf",
                "pt_albert_large_v2_mlm_hf",
                "pt_albert_large_v1_mlm_hf",
                "pt_albert_large_v1_token_cls_hf",
                "pt_albert_xlarge_v1_token_cls_hf",
                "pt_albert_xlarge_v2_mlm_hf",
                "pt_albert_xlarge_v2_token_cls_hf",
                "pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 16, 128, 128)"},
        },
    ),
    (
        Reshape254,
        [((1, 128, 16, 128), torch.float32)],
        {
            "model_name": [
                "pt_albert_xlarge_v1_mlm_hf",
                "pt_albert_xlarge_v1_token_cls_hf",
                "pt_albert_xlarge_v2_mlm_hf",
                "pt_albert_xlarge_v2_token_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 128, 2048, 1)"},
        },
    ),
    (
        Reshape255,
        [((1, 128, 1024), torch.float32)],
        {
            "model_name": [
                "pt_albert_large_v2_token_cls_hf",
                "pt_albert_large_v2_mlm_hf",
                "pt_albert_large_v1_mlm_hf",
                "pt_albert_large_v1_token_cls_hf",
                "pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(128, 1024)"},
        },
    ),
    (
        Reshape256,
        [((1, 128, 1024), torch.float32)],
        {
            "model_name": [
                "pt_albert_large_v2_token_cls_hf",
                "pt_albert_large_v2_mlm_hf",
                "pt_albert_large_v1_mlm_hf",
                "pt_albert_large_v1_token_cls_hf",
                "pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 128, 16, 64)"},
        },
    ),
    (
        Reshape257,
        [((128, 1024), torch.float32)],
        {
            "model_name": [
                "pt_albert_large_v2_token_cls_hf",
                "pt_albert_large_v2_mlm_hf",
                "pt_albert_large_v1_mlm_hf",
                "pt_albert_large_v1_token_cls_hf",
                "pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 128, 1024)"},
        },
    ),
    (
        Reshape258,
        [((128, 1024), torch.float32)],
        {
            "model_name": ["pt_mistral_mistralai_mistral_7b_v0_1_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 128, 8, 128)"},
        },
    ),
    (
        Reshape259,
        [((1, 16, 128, 64), torch.float32)],
        {
            "model_name": [
                "pt_albert_large_v2_token_cls_hf",
                "pt_albert_large_v2_mlm_hf",
                "pt_albert_large_v1_mlm_hf",
                "pt_albert_large_v1_token_cls_hf",
                "pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(16, 128, 64)"},
        },
    ),
    (
        Reshape260,
        [((1, 16, 64, 128), torch.float32)],
        {
            "model_name": [
                "pt_albert_large_v2_token_cls_hf",
                "pt_albert_large_v2_mlm_hf",
                "pt_albert_large_v1_mlm_hf",
                "pt_albert_large_v1_token_cls_hf",
                "pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(16, 64, 128)"},
        },
    ),
    (
        Reshape261,
        [((16, 128, 64), torch.float32)],
        {
            "model_name": [
                "pt_albert_large_v2_token_cls_hf",
                "pt_albert_large_v2_mlm_hf",
                "pt_albert_large_v1_mlm_hf",
                "pt_albert_large_v1_token_cls_hf",
                "pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 16, 128, 64)"},
        },
    ),
    (
        Reshape262,
        [((1, 128, 16, 64), torch.float32)],
        {
            "model_name": [
                "pt_albert_large_v2_token_cls_hf",
                "pt_albert_large_v2_mlm_hf",
                "pt_albert_large_v1_mlm_hf",
                "pt_albert_large_v1_token_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 128, 1024, 1)"},
        },
    ),
    (
        Reshape255,
        [((1, 128, 16, 64), torch.float32)],
        {
            "model_name": ["pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(128, 1024)"},
        },
    ),
    (
        Reshape263,
        [((1, 128, 4096), torch.float32)],
        {
            "model_name": [
                "pt_albert_xxlarge_v1_mlm_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_albert_xxlarge_v2_token_cls_hf",
                "pt_mistral_mistralai_mistral_7b_v0_1_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(128, 4096)"},
        },
    ),
    (
        Reshape264,
        [((1, 128, 4096), torch.float32)],
        {
            "model_name": [
                "pt_albert_xxlarge_v1_mlm_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_albert_xxlarge_v2_token_cls_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 128, 64, 64)"},
        },
    ),
    (
        Reshape265,
        [((128, 4096), torch.float32)],
        {
            "model_name": [
                "pt_albert_xxlarge_v1_mlm_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_albert_xxlarge_v2_token_cls_hf",
                "pt_mistral_mistralai_mistral_7b_v0_1_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 128, 4096)"},
        },
    ),
    (
        Reshape266,
        [((128, 4096), torch.float32)],
        {
            "model_name": ["pt_mistral_mistralai_mistral_7b_v0_1_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 128, 32, 128)"},
        },
    ),
    (
        Reshape267,
        [((1, 64, 128, 64), torch.float32)],
        {
            "model_name": [
                "pt_albert_xxlarge_v1_mlm_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_albert_xxlarge_v2_token_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(64, 128, 64)"},
        },
    ),
    (
        Reshape268,
        [((64, 128, 128), torch.float32)],
        {
            "model_name": [
                "pt_albert_xxlarge_v1_mlm_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_albert_xxlarge_v2_token_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 64, 128, 128)"},
        },
    ),
    (
        Reshape269,
        [((1, 64, 128, 128), torch.float32)],
        {
            "model_name": [
                "pt_albert_xxlarge_v1_mlm_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_albert_xxlarge_v2_token_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(64, 128, 128)"},
        },
    ),
    (
        Reshape270,
        [((1, 64, 128, 128), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 64, 16384, 1)"},
        },
    ),
    (
        Reshape271,
        [((1, 64, 64, 128), torch.float32)],
        {
            "model_name": [
                "pt_albert_xxlarge_v1_mlm_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_albert_xxlarge_v2_token_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(64, 64, 128)"},
        },
    ),
    (
        Reshape272,
        [((64, 128, 64), torch.float32)],
        {
            "model_name": [
                "pt_albert_xxlarge_v1_mlm_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_albert_xxlarge_v2_token_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 64, 128, 64)"},
        },
    ),
    (
        Reshape273,
        [((1, 128, 64, 64), torch.float32)],
        {
            "model_name": [
                "pt_albert_xxlarge_v1_mlm_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_albert_xxlarge_v2_token_cls_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 128, 4096, 1)"},
        },
    ),
    (
        Reshape274,
        [((1, 128, 768), torch.float32)],
        {
            "model_name": [
                "pt_albert_base_v2_token_cls_hf",
                "pt_albert_base_v1_mlm_hf",
                "pt_albert_base_v2_mlm_hf",
                "pt_albert_base_v1_token_cls_hf",
                "pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf",
                "pt_bert_bert_base_uncased_mlm_hf",
                "pt_distilbert_distilbert_base_uncased_mlm_hf",
                "pt_distilbert_distilbert_base_multilingual_cased_mlm_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
                "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf",
                "pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader",
                "pt_roberta_xlm_roberta_base_mlm_hf",
                "pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(128, 768)"},
        },
    ),
    (
        Reshape275,
        [((1, 128, 768), torch.float32)],
        {
            "model_name": [
                "pt_albert_base_v2_token_cls_hf",
                "pt_albert_base_v1_mlm_hf",
                "pt_albert_base_v2_mlm_hf",
                "pt_albert_base_v1_token_cls_hf",
                "pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf",
                "pt_bert_bert_base_uncased_mlm_hf",
                "pt_distilbert_distilbert_base_uncased_mlm_hf",
                "pt_distilbert_distilbert_base_multilingual_cased_mlm_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
                "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf",
                "pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader",
                "pt_roberta_xlm_roberta_base_mlm_hf",
                "pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 128, 12, 64)"},
        },
    ),
    (
        Reshape276,
        [((128, 768), torch.float32)],
        {
            "model_name": [
                "pt_albert_base_v2_token_cls_hf",
                "pt_albert_base_v1_mlm_hf",
                "pt_albert_base_v2_mlm_hf",
                "pt_albert_base_v1_token_cls_hf",
                "pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf",
                "pt_bert_bert_base_uncased_mlm_hf",
                "pt_distilbert_distilbert_base_uncased_mlm_hf",
                "pt_distilbert_distilbert_base_multilingual_cased_mlm_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
                "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf",
                "pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader",
                "pt_roberta_xlm_roberta_base_mlm_hf",
                "pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 128, 768)"},
        },
    ),
    (
        Reshape277,
        [((1, 12, 128, 64), torch.float32)],
        {
            "model_name": [
                "pt_albert_base_v2_token_cls_hf",
                "pt_albert_base_v1_mlm_hf",
                "pt_albert_base_v2_mlm_hf",
                "pt_albert_base_v1_token_cls_hf",
                "pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf",
                "pt_bert_bert_base_uncased_mlm_hf",
                "pt_distilbert_distilbert_base_uncased_mlm_hf",
                "pt_distilbert_distilbert_base_multilingual_cased_mlm_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
                "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf",
                "pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader",
                "pt_roberta_xlm_roberta_base_mlm_hf",
                "pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf",
                "pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(12, 128, 64)"},
        },
    ),
    (
        Reshape278,
        [((12, 128, 128), torch.float32)],
        {
            "model_name": [
                "pt_albert_base_v2_token_cls_hf",
                "pt_albert_base_v1_mlm_hf",
                "pt_albert_base_v2_mlm_hf",
                "pt_albert_base_v1_token_cls_hf",
                "pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf",
                "pt_bert_bert_base_uncased_mlm_hf",
                "pt_distilbert_distilbert_base_uncased_mlm_hf",
                "pt_distilbert_distilbert_base_multilingual_cased_mlm_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
                "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf",
                "pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader",
                "pt_roberta_xlm_roberta_base_mlm_hf",
                "pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf",
                "pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 12, 128, 128)"},
        },
    ),
    (
        Reshape279,
        [((1, 12, 128, 128), torch.float32)],
        {
            "model_name": [
                "pt_albert_base_v2_token_cls_hf",
                "pt_albert_base_v1_mlm_hf",
                "pt_albert_base_v2_mlm_hf",
                "pt_albert_base_v1_token_cls_hf",
                "pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf",
                "pt_bert_bert_base_uncased_mlm_hf",
                "pt_distilbert_distilbert_base_uncased_mlm_hf",
                "pt_distilbert_distilbert_base_multilingual_cased_mlm_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
                "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf",
                "pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader",
                "pt_roberta_xlm_roberta_base_mlm_hf",
                "pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf",
                "pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(12, 128, 128)"},
        },
    ),
    (
        Reshape280,
        [((1, 12, 64, 128), torch.float32)],
        {
            "model_name": [
                "pt_albert_base_v2_token_cls_hf",
                "pt_albert_base_v1_mlm_hf",
                "pt_albert_base_v2_mlm_hf",
                "pt_albert_base_v1_token_cls_hf",
                "pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf",
                "pt_bert_bert_base_uncased_mlm_hf",
                "pt_distilbert_distilbert_base_uncased_mlm_hf",
                "pt_distilbert_distilbert_base_multilingual_cased_mlm_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
                "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf",
                "pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader",
                "pt_roberta_xlm_roberta_base_mlm_hf",
                "pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(12, 64, 128)"},
        },
    ),
    (
        Reshape281,
        [((1, 12, 64, 128), torch.float32)],
        {
            "model_name": ["pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 768, 128, 1)"},
        },
    ),
    (
        Reshape282,
        [((12, 128, 64), torch.float32)],
        {
            "model_name": [
                "pt_albert_base_v2_token_cls_hf",
                "pt_albert_base_v1_mlm_hf",
                "pt_albert_base_v2_mlm_hf",
                "pt_albert_base_v1_token_cls_hf",
                "pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf",
                "pt_bert_bert_base_uncased_mlm_hf",
                "pt_distilbert_distilbert_base_uncased_mlm_hf",
                "pt_distilbert_distilbert_base_multilingual_cased_mlm_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
                "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf",
                "pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader",
                "pt_roberta_xlm_roberta_base_mlm_hf",
                "pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf",
                "pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 12, 128, 64)"},
        },
    ),
    (
        Reshape283,
        [((1, 128, 12, 64), torch.float32)],
        {
            "model_name": [
                "pt_albert_base_v2_token_cls_hf",
                "pt_albert_base_v1_mlm_hf",
                "pt_albert_base_v2_mlm_hf",
                "pt_albert_base_v1_token_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 128, 768, 1)"},
        },
    ),
    (
        Reshape274,
        [((1, 128, 12, 64), torch.float32)],
        {
            "model_name": [
                "pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf",
                "pt_bert_bert_base_uncased_mlm_hf",
                "pt_distilbert_distilbert_base_uncased_mlm_hf",
                "pt_distilbert_distilbert_base_multilingual_cased_mlm_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
                "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf",
                "pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader",
                "pt_roberta_xlm_roberta_base_mlm_hf",
                "pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(128, 768)"},
        },
    ),
    (
        Reshape284,
        [((1, 9, 768), torch.float32)],
        {
            "model_name": ["pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(9, 768)"},
        },
    ),
    (
        Reshape285,
        [((1, 9, 768), torch.float32)],
        {
            "model_name": ["pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 9, 12, 64)"},
        },
    ),
    (
        Reshape286,
        [((9, 768), torch.float32)],
        {
            "model_name": ["pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 9, 768)"},
        },
    ),
    (
        Reshape287,
        [((1, 12, 9, 64), torch.float32)],
        {
            "model_name": ["pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(12, 9, 64)"},
        },
    ),
    (
        Reshape288,
        [((12, 9, 9), torch.float32)],
        {
            "model_name": ["pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 12, 9, 9)"},
        },
    ),
    (
        Reshape289,
        [((1, 12, 9, 9), torch.float32)],
        {
            "model_name": ["pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(12, 9, 9)"},
        },
    ),
    (
        Reshape290,
        [((1, 12, 64, 9), torch.float32)],
        {
            "model_name": ["pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(12, 64, 9)"},
        },
    ),
    (
        Reshape291,
        [((12, 9, 64), torch.float32)],
        {
            "model_name": ["pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 12, 9, 64)"},
        },
    ),
    (
        Reshape292,
        [((1, 9, 12, 64), torch.float32)],
        {
            "model_name": ["pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 9, 768, 1)"},
        },
    ),
    (
        Reshape293,
        [((1, 14, 768), torch.float32)],
        {
            "model_name": ["pt_albert_twmkn9_albert_base_v2_squad2_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(14, 768)"},
        },
    ),
    (
        Reshape294,
        [((1, 14, 768), torch.float32)],
        {
            "model_name": ["pt_albert_twmkn9_albert_base_v2_squad2_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 14, 12, 64)"},
        },
    ),
    (
        Reshape295,
        [((14, 768), torch.float32)],
        {
            "model_name": ["pt_albert_twmkn9_albert_base_v2_squad2_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 14, 768)"},
        },
    ),
    (
        Reshape296,
        [((1, 12, 14, 64), torch.float32)],
        {
            "model_name": ["pt_albert_twmkn9_albert_base_v2_squad2_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(12, 14, 64)"},
        },
    ),
    (
        Reshape297,
        [((12, 14, 14), torch.float32)],
        {
            "model_name": ["pt_albert_twmkn9_albert_base_v2_squad2_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 12, 14, 14)"},
        },
    ),
    (
        Reshape298,
        [((1, 12, 14, 14), torch.float32)],
        {
            "model_name": ["pt_albert_twmkn9_albert_base_v2_squad2_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(12, 14, 14)"},
        },
    ),
    (
        Reshape299,
        [((1, 12, 64, 14), torch.float32)],
        {
            "model_name": ["pt_albert_twmkn9_albert_base_v2_squad2_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(12, 64, 14)"},
        },
    ),
    (
        Reshape300,
        [((12, 14, 64), torch.float32)],
        {
            "model_name": ["pt_albert_twmkn9_albert_base_v2_squad2_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 12, 14, 64)"},
        },
    ),
    (
        Reshape301,
        [((1, 14, 12, 64), torch.float32)],
        {
            "model_name": ["pt_albert_twmkn9_albert_base_v2_squad2_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 14, 768, 1)"},
        },
    ),
    (
        Reshape302,
        [((14, 1), torch.float32)],
        {
            "model_name": ["pt_albert_twmkn9_albert_base_v2_squad2_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 14, 1)"},
        },
    ),
    (
        Reshape303,
        [((1, 256, 1024), torch.float32)],
        {
            "model_name": [
                "pt_bart_facebook_bart_large_mnli_seq_cls_hf",
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
                "pt_codegen_salesforce_codegen_350m_mono_clm_hf",
                "pt_opt_facebook_opt_350m_clm_hf",
                "pt_xglm_facebook_xglm_564m_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(256, 1024)"},
        },
    ),
    (
        Reshape304,
        [((1, 256, 1024), torch.float32)],
        {
            "model_name": [
                "pt_bart_facebook_bart_large_mnli_seq_cls_hf",
                "pt_opt_facebook_opt_350m_clm_hf",
                "pt_xglm_facebook_xglm_564m_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 256, 16, 64)"},
        },
    ),
    (
        Reshape305,
        [((1, 256, 1024), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 256, 32, 32)"},
        },
    ),
    (
        Reshape306,
        [((256, 1024), torch.float32)],
        {
            "model_name": [
                "pt_bart_facebook_bart_large_mnli_seq_cls_hf",
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
                "pt_codegen_salesforce_codegen_350m_mono_clm_hf",
                "pt_opt_facebook_opt_350m_clm_hf",
                "pt_xglm_facebook_xglm_564m_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 256, 1024)"},
        },
    ),
    (
        Reshape307,
        [((256, 1024), torch.float32)],
        {
            "model_name": [
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
                "pt_codegen_salesforce_codegen_350m_mono_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 256, 4, 256)"},
        },
    ),
    (
        Reshape308,
        [((256, 1024), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_clm_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_clm_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_clm_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 256, 8, 128)"},
        },
    ),
    (
        Reshape309,
        [((1, 16, 256, 64), torch.float32)],
        {
            "model_name": [
                "pt_bart_facebook_bart_large_mnli_seq_cls_hf",
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
                "pt_codegen_salesforce_codegen_350m_mono_clm_hf",
                "pt_opt_facebook_opt_350m_clm_hf",
                "pt_xglm_facebook_xglm_564m_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(16, 256, 64)"},
        },
    ),
    (
        Reshape310,
        [((16, 256, 256), torch.float32)],
        {
            "model_name": [
                "pt_bart_facebook_bart_large_mnli_seq_cls_hf",
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
                "pt_codegen_salesforce_codegen_350m_mono_clm_hf",
                "pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf",
                "pt_opt_facebook_opt_350m_clm_hf",
                "pt_xglm_facebook_xglm_1_7b_clm_hf",
                "pt_xglm_facebook_xglm_564m_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 16, 256, 256)"},
        },
    ),
    (
        Reshape311,
        [((1, 16, 256, 256), torch.float32)],
        {
            "model_name": [
                "pt_bart_facebook_bart_large_mnli_seq_cls_hf",
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
                "pt_codegen_salesforce_codegen_350m_mono_clm_hf",
                "pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf",
                "pt_opt_facebook_opt_350m_clm_hf",
                "pt_xglm_facebook_xglm_1_7b_clm_hf",
                "pt_xglm_facebook_xglm_564m_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(16, 256, 256)"},
        },
    ),
    (
        Reshape312,
        [((16, 256, 64), torch.float32)],
        {
            "model_name": [
                "pt_bart_facebook_bart_large_mnli_seq_cls_hf",
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
                "pt_codegen_salesforce_codegen_350m_mono_clm_hf",
                "pt_opt_facebook_opt_350m_clm_hf",
                "pt_xglm_facebook_xglm_564m_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 16, 256, 64)"},
        },
    ),
    (
        Reshape303,
        [((1, 256, 16, 64), torch.float32)],
        {
            "model_name": [
                "pt_bart_facebook_bart_large_mnli_seq_cls_hf",
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
                "pt_codegen_salesforce_codegen_350m_mono_clm_hf",
                "pt_opt_facebook_opt_350m_clm_hf",
                "pt_xglm_facebook_xglm_564m_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(256, 1024)"},
        },
    ),
    (
        Reshape313,
        [((1, 256), torch.int64)],
        {
            "model_name": [
                "pt_bart_facebook_bart_large_mnli_seq_cls_hf",
                "pt_gpt2_gpt2_text_gen_hf",
                "pt_opt_facebook_opt_1_3b_clm_hf",
                "pt_opt_facebook_opt_125m_clm_hf",
                "pt_opt_facebook_opt_350m_clm_hf",
                "pt_xglm_facebook_xglm_1_7b_clm_hf",
                "pt_xglm_facebook_xglm_564m_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 256)"},
        },
    ),
    (
        Reshape314,
        [((1, 384, 1024), torch.float32)],
        {
            "model_name": [
                "pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf",
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(384, 1024)"},
        },
    ),
    (
        Reshape315,
        [((1, 384, 1024), torch.float32)],
        {
            "model_name": [
                "pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf",
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 384, 16, 64)"},
        },
    ),
    (
        Reshape316,
        [((384, 1024), torch.float32)],
        {
            "model_name": [
                "pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf",
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 384, 1024)"},
        },
    ),
    (
        Reshape317,
        [((1, 16, 384, 64), torch.float32)],
        {
            "model_name": [
                "pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf",
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(16, 384, 64)"},
        },
    ),
    (
        Reshape318,
        [((16, 384, 384), torch.float32)],
        {
            "model_name": [
                "pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf",
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 16, 384, 384)"},
        },
    ),
    (
        Reshape319,
        [((1, 16, 384, 384), torch.float32)],
        {
            "model_name": [
                "pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf",
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(16, 384, 384)"},
        },
    ),
    (
        Reshape320,
        [((1, 16, 64, 384), torch.float32)],
        {
            "model_name": [
                "pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf",
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(16, 64, 384)"},
        },
    ),
    (
        Reshape321,
        [((16, 384, 64), torch.float32)],
        {
            "model_name": [
                "pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf",
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 16, 384, 64)"},
        },
    ),
    (
        Reshape314,
        [((1, 384, 16, 64), torch.float32)],
        {
            "model_name": [
                "pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf",
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(384, 1024)"},
        },
    ),
    (
        Reshape322,
        [((384, 1), torch.float32)],
        {
            "model_name": [
                "pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf",
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                "pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 384, 1)"},
        },
    ),
    (
        Reshape323,
        [((1, 6, 768), torch.float32)],
        {
            "model_name": ["pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(6, 768)"},
        },
    ),
    (
        Reshape324,
        [((1, 6, 768), torch.float32)],
        {
            "model_name": ["pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 6, 12, 64)"},
        },
    ),
    (
        Reshape325,
        [((6, 768), torch.float32)],
        {
            "model_name": ["pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 6, 768)"},
        },
    ),
    (
        Reshape326,
        [((1, 12, 6, 64), torch.float32)],
        {
            "model_name": ["pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(12, 6, 64)"},
        },
    ),
    (
        Reshape327,
        [((12, 6, 6), torch.float32)],
        {
            "model_name": ["pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 12, 6, 6)"},
        },
    ),
    (
        Reshape328,
        [((1, 12, 6, 6), torch.float32)],
        {
            "model_name": ["pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(12, 6, 6)"},
        },
    ),
    (
        Reshape329,
        [((1, 12, 64, 6), torch.float32)],
        {
            "model_name": ["pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(12, 64, 6)"},
        },
    ),
    (
        Reshape330,
        [((12, 6, 64), torch.float32)],
        {
            "model_name": ["pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 12, 6, 64)"},
        },
    ),
    (
        Reshape323,
        [((1, 6, 12, 64), torch.float32)],
        {
            "model_name": ["pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(6, 768)"},
        },
    ),
    (
        Reshape331,
        [((1, 32, 4608), torch.float32)],
        {
            "model_name": ["pt_bloom_bigscience_bloom_1b1_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 32, 16, 3, 96)"},
        },
    ),
    (
        Reshape332,
        [((1, 32, 16, 1, 96), torch.float32)],
        {
            "model_name": ["pt_bloom_bigscience_bloom_1b1_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 32, 16, 96)"},
        },
    ),
    (
        Reshape333,
        [((1, 16, 32, 96), torch.float32)],
        {"model_name": ["pt_bloom_bigscience_bloom_1b1_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(16, 32, 96)"}},
    ),
    (
        Reshape334,
        [((1, 16, 32), torch.float32)],
        {"model_name": ["pt_bloom_bigscience_bloom_1b1_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(16, 1, 32)"}},
    ),
    (
        Reshape335,
        [((16, 32, 32), torch.float32)],
        {
            "model_name": [
                "pt_bloom_bigscience_bloom_1b1_clm_hf",
                "pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf",
                "pt_opt_facebook_opt_350m_seq_cls_hf",
                "pt_opt_facebook_opt_350m_qa_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 16, 32, 32)"},
        },
    ),
    (
        Reshape336,
        [((1, 16, 32, 32), torch.float32)],
        {
            "model_name": [
                "pt_bloom_bigscience_bloom_1b1_clm_hf",
                "pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf",
                "pt_opt_facebook_opt_350m_seq_cls_hf",
                "pt_opt_facebook_opt_350m_qa_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(16, 32, 32)"},
        },
    ),
    (
        Reshape337,
        [((16, 32, 96), torch.float32)],
        {
            "model_name": ["pt_bloom_bigscience_bloom_1b1_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 16, 32, 96)"},
        },
    ),
    (
        Reshape338,
        [((1, 32, 16, 96), torch.float32)],
        {"model_name": ["pt_bloom_bigscience_bloom_1b1_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(32, 1536)"}},
    ),
    (
        Reshape339,
        [((32, 1536), torch.float32)],
        {"model_name": ["pt_bloom_bigscience_bloom_1b1_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 32, 1536)"}},
    ),
    (
        Reshape304,
        [((1, 256, 4, 256), torch.float32)],
        {
            "model_name": [
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
                "pt_codegen_salesforce_codegen_350m_mono_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 256, 16, 64)"},
        },
    ),
    (
        Reshape340,
        [((1, 256, 16, 16, 2), torch.float32)],
        {
            "model_name": [
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
                "pt_codegen_salesforce_codegen_350m_mono_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 256, 16, 32, 1)"},
        },
    ),
    (
        Reshape341,
        [((1, 16, 64, 256), torch.float32)],
        {
            "model_name": [
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
                "pt_codegen_salesforce_codegen_350m_mono_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(16, 64, 256)"},
        },
    ),
    (
        Reshape342,
        [((256, 4096), torch.float32)],
        {
            "model_name": [
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
                "pt_codegen_salesforce_codegen_350m_mono_clm_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_clm_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_clm_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_clm_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 256, 4096)"},
        },
    ),
    (
        Reshape343,
        [((256, 4096), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_clm_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_clm_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_clm_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 256, 32, 128)"},
        },
    ),
    (
        Reshape344,
        [((1, 128), torch.bool)],
        {
            "model_name": [
                "pt_distilbert_distilbert_base_uncased_mlm_hf",
                "pt_distilbert_distilbert_base_multilingual_cased_mlm_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
                "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1, 1, 128)"},
        },
    ),
    (
        Reshape345,
        [((1, 384, 768), torch.float32)],
        {
            "model_name": ["pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(384, 768)"},
        },
    ),
    (
        Reshape346,
        [((1, 384, 768), torch.float32)],
        {
            "model_name": ["pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 384, 12, 64)"},
        },
    ),
    (
        Reshape347,
        [((384, 768), torch.float32)],
        {
            "model_name": ["pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 384, 768)"},
        },
    ),
    (
        Reshape348,
        [((1, 12, 384, 64), torch.float32)],
        {
            "model_name": ["pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(12, 384, 64)"},
        },
    ),
    (
        Reshape349,
        [((12, 384, 384), torch.float32)],
        {
            "model_name": ["pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 12, 384, 384)"},
        },
    ),
    (
        Reshape350,
        [((1, 384), torch.bool)],
        {
            "model_name": ["pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1, 1, 384)"},
        },
    ),
    (
        Reshape351,
        [((1, 12, 384, 384), torch.float32)],
        {
            "model_name": ["pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(12, 384, 384)"},
        },
    ),
    (
        Reshape352,
        [((1, 12, 64, 384), torch.float32)],
        {
            "model_name": ["pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(12, 64, 384)"},
        },
    ),
    (
        Reshape353,
        [((12, 384, 64), torch.float32)],
        {
            "model_name": ["pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 12, 384, 64)"},
        },
    ),
    (
        Reshape345,
        [((1, 384, 12, 64), torch.float32)],
        {
            "model_name": ["pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(384, 768)"},
        },
    ),
    (
        Reshape354,
        [((128, 1), torch.float32)],
        {
            "model_name": [
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 128, 1)"},
        },
    ),
    (
        Reshape355,
        [((1, 128), torch.float32)],
        {
            "model_name": [
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 128)"},
        },
    ),
    (
        Reshape356,
        [((1, 1), torch.float32)],
        {
            "model_name": [
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1,)"},
        },
    ),
    (
        Reshape357,
        [((1, 10, 3072), torch.float32)],
        {
            "model_name": ["pt_falcon3_tiiuae_falcon3_7b_base_clm_hf", "pt_falcon3_tiiuae_falcon3_3b_base_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(10, 3072)"},
        },
    ),
    (
        Reshape358,
        [((10, 3072), torch.float32)],
        {
            "model_name": ["pt_falcon3_tiiuae_falcon3_7b_base_clm_hf", "pt_falcon3_tiiuae_falcon3_3b_base_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 10, 12, 256)"},
        },
    ),
    (
        Reshape359,
        [((10, 3072), torch.float32)],
        {
            "model_name": ["pt_falcon3_tiiuae_falcon3_7b_base_clm_hf", "pt_falcon3_tiiuae_falcon3_3b_base_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 10, 3072)"},
        },
    ),
    (
        Reshape360,
        [((1, 12, 10, 256), torch.float32)],
        {
            "model_name": ["pt_falcon3_tiiuae_falcon3_7b_base_clm_hf", "pt_falcon3_tiiuae_falcon3_3b_base_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(12, 10, 256)"},
        },
    ),
    (
        Reshape361,
        [((10, 1024), torch.float32)],
        {
            "model_name": [
                "pt_falcon3_tiiuae_falcon3_7b_base_clm_hf",
                "pt_falcon3_tiiuae_falcon3_1b_base_clm_hf",
                "pt_falcon3_tiiuae_falcon3_3b_base_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 10, 4, 256)"},
        },
    ),
    (
        Reshape360,
        [((1, 4, 3, 10, 256), torch.float32)],
        {
            "model_name": ["pt_falcon3_tiiuae_falcon3_7b_base_clm_hf", "pt_falcon3_tiiuae_falcon3_3b_base_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(12, 10, 256)"},
        },
    ),
    (
        Reshape362,
        [((1, 4, 3, 10, 256), torch.float32)],
        {
            "model_name": ["pt_falcon3_tiiuae_falcon3_7b_base_clm_hf", "pt_falcon3_tiiuae_falcon3_3b_base_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 12, 10, 256)"},
        },
    ),
    (
        Reshape363,
        [((12, 10, 10), torch.float32)],
        {
            "model_name": ["pt_falcon3_tiiuae_falcon3_7b_base_clm_hf", "pt_falcon3_tiiuae_falcon3_3b_base_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 12, 10, 10)"},
        },
    ),
    (
        Reshape364,
        [((1, 12, 10, 10), torch.float32)],
        {
            "model_name": ["pt_falcon3_tiiuae_falcon3_7b_base_clm_hf", "pt_falcon3_tiiuae_falcon3_3b_base_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(12, 10, 10)"},
        },
    ),
    (
        Reshape365,
        [((1, 12, 256, 10), torch.float32)],
        {
            "model_name": ["pt_falcon3_tiiuae_falcon3_7b_base_clm_hf", "pt_falcon3_tiiuae_falcon3_3b_base_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(12, 256, 10)"},
        },
    ),
    (
        Reshape362,
        [((12, 10, 256), torch.float32)],
        {
            "model_name": ["pt_falcon3_tiiuae_falcon3_7b_base_clm_hf", "pt_falcon3_tiiuae_falcon3_3b_base_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 12, 10, 256)"},
        },
    ),
    (
        Reshape357,
        [((1, 10, 12, 256), torch.float32)],
        {
            "model_name": ["pt_falcon3_tiiuae_falcon3_7b_base_clm_hf", "pt_falcon3_tiiuae_falcon3_3b_base_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(10, 3072)"},
        },
    ),
    (
        Reshape366,
        [((10, 23040), torch.float32)],
        {
            "model_name": ["pt_falcon3_tiiuae_falcon3_7b_base_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 10, 23040)"},
        },
    ),
    (
        Reshape367,
        [((1, 10, 2048), torch.float32)],
        {"model_name": ["pt_falcon3_tiiuae_falcon3_1b_base_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(10, 2048)"}},
    ),
    (
        Reshape368,
        [((10, 2048), torch.float32)],
        {
            "model_name": ["pt_falcon3_tiiuae_falcon3_1b_base_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 10, 8, 256)"},
        },
    ),
    (
        Reshape369,
        [((10, 2048), torch.float32)],
        {
            "model_name": ["pt_falcon3_tiiuae_falcon3_1b_base_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 10, 2048)"},
        },
    ),
    (
        Reshape370,
        [((1, 8, 10, 256), torch.float32)],
        {
            "model_name": ["pt_falcon3_tiiuae_falcon3_1b_base_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(8, 10, 256)"},
        },
    ),
    (
        Reshape370,
        [((1, 4, 2, 10, 256), torch.float32)],
        {
            "model_name": ["pt_falcon3_tiiuae_falcon3_1b_base_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(8, 10, 256)"},
        },
    ),
    (
        Reshape371,
        [((1, 4, 2, 10, 256), torch.float32)],
        {
            "model_name": ["pt_falcon3_tiiuae_falcon3_1b_base_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 8, 10, 256)"},
        },
    ),
    (
        Reshape372,
        [((8, 10, 10), torch.float32)],
        {
            "model_name": ["pt_falcon3_tiiuae_falcon3_1b_base_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 8, 10, 10)"},
        },
    ),
    (
        Reshape373,
        [((1, 8, 10, 10), torch.float32)],
        {
            "model_name": ["pt_falcon3_tiiuae_falcon3_1b_base_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(8, 10, 10)"},
        },
    ),
    (
        Reshape374,
        [((1, 8, 256, 10), torch.float32)],
        {
            "model_name": ["pt_falcon3_tiiuae_falcon3_1b_base_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(8, 256, 10)"},
        },
    ),
    (
        Reshape371,
        [((8, 10, 256), torch.float32)],
        {
            "model_name": ["pt_falcon3_tiiuae_falcon3_1b_base_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 8, 10, 256)"},
        },
    ),
    (
        Reshape367,
        [((1, 10, 8, 256), torch.float32)],
        {"model_name": ["pt_falcon3_tiiuae_falcon3_1b_base_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(10, 2048)"}},
    ),
    (
        Reshape375,
        [((10, 8192), torch.float32)],
        {
            "model_name": ["pt_falcon3_tiiuae_falcon3_1b_base_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 10, 8192)"},
        },
    ),
    (
        Reshape376,
        [((10, 9216), torch.float32)],
        {
            "model_name": ["pt_falcon3_tiiuae_falcon3_3b_base_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 10, 9216)"},
        },
    ),
    (
        Reshape377,
        [((1, 6, 4544), torch.float32)],
        {
            "model_name": ["pt_falcon_tiiuae_falcon_7b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(6, 4544)"},
        },
    ),
    (
        Reshape378,
        [((6, 18176), torch.float32)],
        {
            "model_name": ["pt_falcon_tiiuae_falcon_7b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 6, 18176)"},
        },
    ),
    (
        Reshape379,
        [((6, 4672), torch.float32)],
        {
            "model_name": ["pt_falcon_tiiuae_falcon_7b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 6, 73, 64)"},
        },
    ),
    (
        Reshape380,
        [((1, 71, 6, 64), torch.float32)],
        {
            "model_name": ["pt_falcon_tiiuae_falcon_7b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 71, 6, 64)"},
        },
    ),
    (
        Reshape381,
        [((1, 71, 6, 64), torch.float32)],
        {
            "model_name": ["pt_falcon_tiiuae_falcon_7b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(71, 6, 64)"},
        },
    ),
    (
        Reshape382,
        [((71, 6, 6), torch.float32)],
        {
            "model_name": ["pt_falcon_tiiuae_falcon_7b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 71, 6, 6)"},
        },
    ),
    (
        Reshape383,
        [((1, 71, 6, 6), torch.float32)],
        {
            "model_name": ["pt_falcon_tiiuae_falcon_7b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(71, 6, 6)"},
        },
    ),
    (
        Reshape380,
        [((71, 6, 64), torch.float32)],
        {
            "model_name": ["pt_falcon_tiiuae_falcon_7b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 71, 6, 64)"},
        },
    ),
    (
        Reshape377,
        [((1, 6, 71, 64), torch.float32)],
        {
            "model_name": ["pt_falcon_tiiuae_falcon_7b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(6, 4544)"},
        },
    ),
    (
        Reshape384,
        [((6, 4544), torch.float32)],
        {
            "model_name": ["pt_falcon_tiiuae_falcon_7b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 6, 4544)"},
        },
    ),
    (
        Reshape385,
        [((1, 334, 12288), torch.float32)],
        {"model_name": ["pt_fuyu_adept_fuyu_8b_qa_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 334, 64, 3, 64)"}},
    ),
    (
        Reshape386,
        [((1, 334, 64, 1, 64), torch.float32)],
        {"model_name": ["pt_fuyu_adept_fuyu_8b_qa_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 334, 64, 64)"}},
    ),
    (
        Reshape387,
        [((1, 64, 334, 64), torch.float32)],
        {"model_name": ["pt_fuyu_adept_fuyu_8b_qa_hf"], "pcc": 0.99, "op_params": {"shape": "(64, 334, 64)"}},
    ),
    (
        Reshape388,
        [((64, 334, 334), torch.float32)],
        {"model_name": ["pt_fuyu_adept_fuyu_8b_qa_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 64, 334, 334)"}},
    ),
    (
        Reshape389,
        [((1, 64, 334, 334), torch.float32)],
        {"model_name": ["pt_fuyu_adept_fuyu_8b_qa_hf"], "pcc": 0.99, "op_params": {"shape": "(64, 334, 334)"}},
    ),
    (
        Reshape390,
        [((1, 64, 64, 334), torch.float32)],
        {"model_name": ["pt_fuyu_adept_fuyu_8b_qa_hf"], "pcc": 0.99, "op_params": {"shape": "(64, 64, 334)"}},
    ),
    (
        Reshape391,
        [((64, 334, 64), torch.float32)],
        {"model_name": ["pt_fuyu_adept_fuyu_8b_qa_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 64, 334, 64)"}},
    ),
    (
        Reshape392,
        [((1, 334, 64, 64), torch.float32)],
        {"model_name": ["pt_fuyu_adept_fuyu_8b_qa_hf"], "pcc": 0.99, "op_params": {"shape": "(334, 4096)"}},
    ),
    (
        Reshape393,
        [((334, 4096), torch.float32)],
        {"model_name": ["pt_fuyu_adept_fuyu_8b_qa_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 334, 4096)"}},
    ),
    (
        Reshape394,
        [((1, 7, 2048), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_2b_text_gen_hf"], "pcc": 0.99, "op_params": {"shape": "(7, 2048)"}},
    ),
    (
        Reshape395,
        [((7, 2048), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_2b_text_gen_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 7, 8, 256)"}},
    ),
    (
        Reshape396,
        [((7, 2048), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_2b_text_gen_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 7, 2048)"}},
    ),
    (
        Reshape397,
        [((1, 8, 7, 256), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_2b_text_gen_hf"], "pcc": 0.99, "op_params": {"shape": "(8, 7, 256)"}},
    ),
    (
        Reshape398,
        [((7, 256), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_2b_text_gen_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 7, 1, 256)"}},
    ),
    (
        Reshape397,
        [((1, 1, 8, 7, 256), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_2b_text_gen_hf"], "pcc": 0.99, "op_params": {"shape": "(8, 7, 256)"}},
    ),
    (
        Reshape399,
        [((1, 1, 8, 7, 256), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_2b_text_gen_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 8, 7, 256)"}},
    ),
    (
        Reshape400,
        [((8, 7, 7), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_2b_text_gen_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 8, 7, 7)"}},
    ),
    (
        Reshape401,
        [((1, 8, 7, 7), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_2b_text_gen_hf"], "pcc": 0.99, "op_params": {"shape": "(8, 7, 7)"}},
    ),
    (
        Reshape402,
        [((1, 8, 256, 7), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_2b_text_gen_hf"], "pcc": 0.99, "op_params": {"shape": "(8, 256, 7)"}},
    ),
    (
        Reshape399,
        [((8, 7, 256), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_2b_text_gen_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 8, 7, 256)"}},
    ),
    (
        Reshape394,
        [((1, 7, 8, 256), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_2b_text_gen_hf"], "pcc": 0.99, "op_params": {"shape": "(7, 2048)"}},
    ),
    (
        Reshape403,
        [((7, 16384), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_2b_text_gen_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 7, 16384)"}},
    ),
    (
        Reshape404,
        [((1, 207, 2304), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_2_2b_it_qa_hf"], "pcc": 0.99, "op_params": {"shape": "(207, 2304)"}},
    ),
    (
        Reshape405,
        [((207, 2048), torch.float32)],
        {
            "model_name": ["pt_gemma_google_gemma_2_2b_it_qa_hf", "pt_gemma_google_gemma_2_9b_it_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 207, 8, 256)"},
        },
    ),
    (
        Reshape406,
        [((1, 8, 207, 256), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_2_2b_it_qa_hf"], "pcc": 0.99, "op_params": {"shape": "(8, 207, 256)"}},
    ),
    (
        Reshape407,
        [((207, 1024), torch.float32)],
        {
            "model_name": ["pt_gemma_google_gemma_2_2b_it_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 207, 4, 256)"},
        },
    ),
    (
        Reshape406,
        [((1, 4, 2, 207, 256), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_2_2b_it_qa_hf"], "pcc": 0.99, "op_params": {"shape": "(8, 207, 256)"}},
    ),
    (
        Reshape408,
        [((1, 4, 2, 207, 256), torch.float32)],
        {
            "model_name": ["pt_gemma_google_gemma_2_2b_it_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 8, 207, 256)"},
        },
    ),
    (
        Reshape409,
        [((8, 207, 207), torch.float32)],
        {
            "model_name": ["pt_gemma_google_gemma_2_2b_it_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 8, 207, 207)"},
        },
    ),
    (
        Reshape410,
        [((1, 8, 207, 207), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_2_2b_it_qa_hf"], "pcc": 0.99, "op_params": {"shape": "(8, 207, 207)"}},
    ),
    (
        Reshape411,
        [((1, 8, 256, 207), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_2_2b_it_qa_hf"], "pcc": 0.99, "op_params": {"shape": "(8, 256, 207)"}},
    ),
    (
        Reshape408,
        [((8, 207, 256), torch.float32)],
        {
            "model_name": ["pt_gemma_google_gemma_2_2b_it_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 8, 207, 256)"},
        },
    ),
    (
        Reshape412,
        [((1, 207, 8, 256), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_2_2b_it_qa_hf"], "pcc": 0.99, "op_params": {"shape": "(207, 2048)"}},
    ),
    (
        Reshape413,
        [((207, 2304), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_2_2b_it_qa_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 207, 2304)"}},
    ),
    (
        Reshape414,
        [((207, 9216), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_2_2b_it_qa_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 207, 9216)"}},
    ),
    (
        Reshape415,
        [((1, 207, 3584), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_2_9b_it_qa_hf"], "pcc": 0.99, "op_params": {"shape": "(207, 3584)"}},
    ),
    (
        Reshape416,
        [((207, 4096), torch.float32)],
        {
            "model_name": ["pt_gemma_google_gemma_2_9b_it_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 207, 16, 256)"},
        },
    ),
    (
        Reshape417,
        [((1, 16, 207, 256), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_2_9b_it_qa_hf"], "pcc": 0.99, "op_params": {"shape": "(16, 207, 256)"}},
    ),
    (
        Reshape417,
        [((1, 8, 2, 207, 256), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_2_9b_it_qa_hf"], "pcc": 0.99, "op_params": {"shape": "(16, 207, 256)"}},
    ),
    (
        Reshape418,
        [((1, 8, 2, 207, 256), torch.float32)],
        {
            "model_name": ["pt_gemma_google_gemma_2_9b_it_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 16, 207, 256)"},
        },
    ),
    (
        Reshape419,
        [((16, 207, 207), torch.float32)],
        {
            "model_name": ["pt_gemma_google_gemma_2_9b_it_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 16, 207, 207)"},
        },
    ),
    (
        Reshape420,
        [((1, 16, 207, 207), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_2_9b_it_qa_hf"], "pcc": 0.99, "op_params": {"shape": "(16, 207, 207)"}},
    ),
    (
        Reshape421,
        [((1, 16, 256, 207), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_2_9b_it_qa_hf"], "pcc": 0.99, "op_params": {"shape": "(16, 256, 207)"}},
    ),
    (
        Reshape418,
        [((16, 207, 256), torch.float32)],
        {
            "model_name": ["pt_gemma_google_gemma_2_9b_it_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 16, 207, 256)"},
        },
    ),
    (
        Reshape422,
        [((1, 207, 16, 256), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_2_9b_it_qa_hf"], "pcc": 0.99, "op_params": {"shape": "(207, 4096)"}},
    ),
    (
        Reshape423,
        [((207, 3584), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_2_9b_it_qa_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 207, 3584)"}},
    ),
    (
        Reshape424,
        [((207, 14336), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_2_9b_it_qa_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 207, 14336)"}},
    ),
    (
        Reshape425,
        [((1, 107, 2048), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_1_1_2b_it_qa_hf"], "pcc": 0.99, "op_params": {"shape": "(107, 2048)"}},
    ),
    (
        Reshape426,
        [((107, 2048), torch.float32)],
        {
            "model_name": ["pt_gemma_google_gemma_1_1_2b_it_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 107, 8, 256)"},
        },
    ),
    (
        Reshape427,
        [((107, 2048), torch.float32)],
        {
            "model_name": ["pt_gemma_google_gemma_1_1_2b_it_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 107, 2048)"},
        },
    ),
    (
        Reshape428,
        [((1, 8, 107, 256), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_1_1_2b_it_qa_hf"], "pcc": 0.99, "op_params": {"shape": "(8, 107, 256)"}},
    ),
    (
        Reshape429,
        [((107, 256), torch.float32)],
        {
            "model_name": ["pt_gemma_google_gemma_1_1_2b_it_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 107, 1, 256)"},
        },
    ),
    (
        Reshape428,
        [((1, 1, 8, 107, 256), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_1_1_2b_it_qa_hf"], "pcc": 0.99, "op_params": {"shape": "(8, 107, 256)"}},
    ),
    (
        Reshape430,
        [((1, 1, 8, 107, 256), torch.float32)],
        {
            "model_name": ["pt_gemma_google_gemma_1_1_2b_it_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 8, 107, 256)"},
        },
    ),
    (
        Reshape431,
        [((8, 107, 107), torch.float32)],
        {
            "model_name": ["pt_gemma_google_gemma_1_1_2b_it_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 8, 107, 107)"},
        },
    ),
    (
        Reshape432,
        [((1, 8, 107, 107), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_1_1_2b_it_qa_hf"], "pcc": 0.99, "op_params": {"shape": "(8, 107, 107)"}},
    ),
    (
        Reshape433,
        [((1, 8, 256, 107), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_1_1_2b_it_qa_hf"], "pcc": 0.99, "op_params": {"shape": "(8, 256, 107)"}},
    ),
    (
        Reshape430,
        [((8, 107, 256), torch.float32)],
        {
            "model_name": ["pt_gemma_google_gemma_1_1_2b_it_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 8, 107, 256)"},
        },
    ),
    (
        Reshape425,
        [((1, 107, 8, 256), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_1_1_2b_it_qa_hf"], "pcc": 0.99, "op_params": {"shape": "(107, 2048)"}},
    ),
    (
        Reshape434,
        [((107, 16384), torch.float32)],
        {
            "model_name": ["pt_gemma_google_gemma_1_1_2b_it_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 107, 16384)"},
        },
    ),
    (
        Reshape435,
        [((1, 107, 3072), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_1_1_7b_it_qa_hf"], "pcc": 0.99, "op_params": {"shape": "(107, 3072)"}},
    ),
    (
        Reshape436,
        [((107, 4096), torch.float32)],
        {
            "model_name": ["pt_gemma_google_gemma_1_1_7b_it_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 107, 16, 256)"},
        },
    ),
    (
        Reshape437,
        [((1, 16, 107, 256), torch.float32)],
        {
            "model_name": ["pt_gemma_google_gemma_1_1_7b_it_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(16, 107, 256)"},
        },
    ),
    (
        Reshape438,
        [((16, 107, 107), torch.float32)],
        {
            "model_name": ["pt_gemma_google_gemma_1_1_7b_it_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 16, 107, 107)"},
        },
    ),
    (
        Reshape439,
        [((1, 16, 107, 107), torch.float32)],
        {
            "model_name": ["pt_gemma_google_gemma_1_1_7b_it_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(16, 107, 107)"},
        },
    ),
    (
        Reshape440,
        [((1, 16, 256, 107), torch.float32)],
        {
            "model_name": ["pt_gemma_google_gemma_1_1_7b_it_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(16, 256, 107)"},
        },
    ),
    (
        Reshape441,
        [((16, 107, 256), torch.float32)],
        {
            "model_name": ["pt_gemma_google_gemma_1_1_7b_it_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 16, 107, 256)"},
        },
    ),
    (
        Reshape442,
        [((1, 107, 16, 256), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_1_1_7b_it_qa_hf"], "pcc": 0.99, "op_params": {"shape": "(107, 4096)"}},
    ),
    (
        Reshape443,
        [((107, 3072), torch.float32)],
        {
            "model_name": ["pt_gemma_google_gemma_1_1_7b_it_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 107, 3072)"},
        },
    ),
    (
        Reshape444,
        [((107, 24576), torch.float32)],
        {
            "model_name": ["pt_gemma_google_gemma_1_1_7b_it_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 107, 24576)"},
        },
    ),
    (
        Reshape445,
        [((1, 256, 768), torch.float32)],
        {
            "model_name": [
                "pt_gpt2_gpt2_text_gen_hf",
                "pt_gptneo_eleutherai_gpt_neo_125m_clm_hf",
                "pt_opt_facebook_opt_125m_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(256, 768)"},
        },
    ),
    (
        Reshape446,
        [((1, 256, 768), torch.float32)],
        {
            "model_name": ["pt_gpt2_gpt2_text_gen_hf", "pt_opt_facebook_opt_125m_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 256, 12, 64)"},
        },
    ),
    (
        Reshape447,
        [((1, 256, 768), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 256, 8, 96)"},
        },
    ),
    (
        Reshape448,
        [((256, 768), torch.float32)],
        {
            "model_name": [
                "pt_gpt2_gpt2_text_gen_hf",
                "pt_gptneo_eleutherai_gpt_neo_125m_clm_hf",
                "pt_opt_facebook_opt_125m_clm_hf",
                "pt_perceiverio_deepmind_language_perceiver_mlm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 256, 768)"},
        },
    ),
    (
        Reshape446,
        [((256, 768), torch.float32)],
        {
            "model_name": ["pt_gptneo_eleutherai_gpt_neo_125m_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 256, 12, 64)"},
        },
    ),
    (
        Reshape449,
        [((1, 12, 256, 64), torch.float32)],
        {
            "model_name": [
                "pt_gpt2_gpt2_text_gen_hf",
                "pt_gptneo_eleutherai_gpt_neo_125m_clm_hf",
                "pt_opt_facebook_opt_125m_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(12, 256, 64)"},
        },
    ),
    (
        Reshape450,
        [((12, 256, 256), torch.float32)],
        {
            "model_name": [
                "pt_gpt2_gpt2_text_gen_hf",
                "pt_gptneo_eleutherai_gpt_neo_125m_clm_hf",
                "pt_opt_facebook_opt_125m_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 12, 256, 256)"},
        },
    ),
    (
        Reshape313,
        [((1, 256), torch.float32)],
        {"model_name": ["pt_gpt2_gpt2_text_gen_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 256)"}},
    ),
    (
        Reshape451,
        [((1, 12, 256, 256), torch.float32)],
        {
            "model_name": [
                "pt_gpt2_gpt2_text_gen_hf",
                "pt_gptneo_eleutherai_gpt_neo_125m_clm_hf",
                "pt_opt_facebook_opt_125m_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(12, 256, 256)"},
        },
    ),
    (
        Reshape452,
        [((1, 12, 64, 256), torch.float32)],
        {
            "model_name": ["pt_gpt2_gpt2_text_gen_hf", "pt_gptneo_eleutherai_gpt_neo_125m_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(12, 64, 256)"},
        },
    ),
    (
        Reshape453,
        [((12, 256, 64), torch.float32)],
        {
            "model_name": [
                "pt_gpt2_gpt2_text_gen_hf",
                "pt_gptneo_eleutherai_gpt_neo_125m_clm_hf",
                "pt_opt_facebook_opt_125m_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 12, 256, 64)"},
        },
    ),
    (
        Reshape445,
        [((1, 256, 12, 64), torch.float32)],
        {
            "model_name": [
                "pt_gpt2_gpt2_text_gen_hf",
                "pt_gptneo_eleutherai_gpt_neo_125m_clm_hf",
                "pt_opt_facebook_opt_125m_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(256, 768)"},
        },
    ),
    (
        Reshape454,
        [((256, 3072), torch.float32)],
        {
            "model_name": [
                "pt_gpt2_gpt2_text_gen_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf",
                "pt_phi3_5_microsoft_phi_3_5_mini_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 256, 3072)"},
        },
    ),
    (
        Reshape455,
        [((1, 256, 3072), torch.float32)],
        {
            "model_name": [
                "pt_gpt2_gpt2_text_gen_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf",
                "pt_phi3_5_microsoft_phi_3_5_mini_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(256, 3072)"},
        },
    ),
    (
        Reshape456,
        [((1, 256, 3072), torch.float32)],
        {
            "model_name": [
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf",
                "pt_phi3_5_microsoft_phi_3_5_mini_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 256, 32, 96)"},
        },
    ),
    (
        Reshape457,
        [((1, 7), torch.int64)],
        {
            "model_name": [
                "pt_gpt2_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 7)"},
        },
    ),
    (
        Reshape458,
        [((1, 7, 768), torch.float32)],
        {
            "model_name": [
                "pt_gpt2_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(7, 768)"},
        },
    ),
    (
        Reshape459,
        [((1, 7, 768), torch.float32)],
        {
            "model_name": [
                "pt_gpt2_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 7, 12, 64)"},
        },
    ),
    (
        Reshape460,
        [((1, 7, 768), torch.float32)],
        {
            "model_name": ["pt_nanogpt_financialsupport_nanogpt_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 7, 768)"},
        },
    ),
    (
        Reshape460,
        [((7, 768), torch.float32)],
        {
            "model_name": [
                "pt_gpt2_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 7, 768)"},
        },
    ),
    (
        Reshape461,
        [((1, 12, 7, 64), torch.float32)],
        {
            "model_name": [
                "pt_gpt2_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(12, 7, 64)"},
        },
    ),
    (
        Reshape462,
        [((12, 7, 7), torch.float32)],
        {
            "model_name": [
                "pt_gpt2_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 12, 7, 7)"},
        },
    ),
    (
        Reshape463,
        [((1, 12, 7, 7), torch.float32)],
        {
            "model_name": [
                "pt_gpt2_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(12, 7, 7)"},
        },
    ),
    (
        Reshape464,
        [((1, 12, 64, 7), torch.float32)],
        {
            "model_name": [
                "pt_gpt2_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(12, 64, 7)"},
        },
    ),
    (
        Reshape465,
        [((12, 7, 64), torch.float32)],
        {
            "model_name": [
                "pt_gpt2_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 12, 7, 64)"},
        },
    ),
    (
        Reshape458,
        [((1, 7, 12, 64), torch.float32)],
        {
            "model_name": [
                "pt_gpt2_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(7, 768)"},
        },
    ),
    (
        Reshape466,
        [((7, 3072), torch.float32)],
        {
            "model_name": [
                "pt_gpt2_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 7, 3072)"},
        },
    ),
    (
        Reshape467,
        [((1, 7, 3072), torch.float32)],
        {
            "model_name": [
                "pt_gpt2_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(7, 3072)"},
        },
    ),
    (
        Reshape468,
        [((7, 2), torch.float32)],
        {
            "model_name": ["pt_gpt2_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(7, 2)"},
        },
    ),
    (
        Reshape177,
        [((1, 2), torch.float32)],
        {
            "model_name": [
                "pt_gpt2_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
                "pt_llama3_huggyllama_llama_7b_seq_cls_hf",
                "pt_opt_facebook_opt_125m_seq_cls_hf",
                "pt_opt_facebook_opt_1_3b_seq_cls_hf",
                "pt_opt_facebook_opt_350m_seq_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 2)"},
        },
    ),
    (
        Reshape469,
        [((1, 32, 2560), torch.float32)],
        {
            "model_name": ["pt_gptneo_eleutherai_gpt_neo_2_7b_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(32, 2560)"},
        },
    ),
    (
        Reshape470,
        [((32, 2560), torch.float32)],
        {
            "model_name": ["pt_gptneo_eleutherai_gpt_neo_2_7b_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 32, 20, 128)"},
        },
    ),
    (
        Reshape471,
        [((32, 2560), torch.float32)],
        {
            "model_name": ["pt_gptneo_eleutherai_gpt_neo_2_7b_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 32, 2560)"},
        },
    ),
    (
        Reshape472,
        [((1, 20, 32, 128), torch.float32)],
        {
            "model_name": ["pt_gptneo_eleutherai_gpt_neo_2_7b_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(20, 32, 128)"},
        },
    ),
    (
        Reshape473,
        [((20, 32, 32), torch.float32)],
        {
            "model_name": ["pt_gptneo_eleutherai_gpt_neo_2_7b_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 20, 32, 32)"},
        },
    ),
    (
        Reshape474,
        [((1, 20, 32, 32), torch.float32)],
        {
            "model_name": ["pt_gptneo_eleutherai_gpt_neo_2_7b_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(20, 32, 32)"},
        },
    ),
    (
        Reshape475,
        [((1, 20, 128, 32), torch.float32)],
        {
            "model_name": ["pt_gptneo_eleutherai_gpt_neo_2_7b_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(20, 128, 32)"},
        },
    ),
    (
        Reshape476,
        [((20, 32, 128), torch.float32)],
        {
            "model_name": ["pt_gptneo_eleutherai_gpt_neo_2_7b_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 20, 32, 128)"},
        },
    ),
    (
        Reshape469,
        [((1, 32, 20, 128), torch.float32)],
        {
            "model_name": ["pt_gptneo_eleutherai_gpt_neo_2_7b_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(32, 2560)"},
        },
    ),
    (
        Reshape177,
        [((1, 1, 2), torch.float32)],
        {
            "model_name": [
                "pt_gptneo_eleutherai_gpt_neo_2_7b_seq_cls_hf",
                "pt_gptneo_eleutherai_gpt_neo_125m_seq_cls_hf",
                "pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_3b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf",
                "pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf",
                "pt_phi2_microsoft_phi_2_seq_cls_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 2)"},
        },
    ),
    (
        Reshape477,
        [((1, 32, 768), torch.float32)],
        {
            "model_name": [
                "pt_gptneo_eleutherai_gpt_neo_125m_seq_cls_hf",
                "pt_opt_facebook_opt_125m_seq_cls_hf",
                "pt_opt_facebook_opt_125m_qa_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(32, 768)"},
        },
    ),
    (
        Reshape478,
        [((1, 32, 768), torch.float32)],
        {
            "model_name": ["pt_opt_facebook_opt_125m_seq_cls_hf", "pt_opt_facebook_opt_125m_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 32, 12, 64)"},
        },
    ),
    (
        Reshape478,
        [((32, 768), torch.float32)],
        {
            "model_name": ["pt_gptneo_eleutherai_gpt_neo_125m_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 32, 12, 64)"},
        },
    ),
    (
        Reshape479,
        [((32, 768), torch.float32)],
        {
            "model_name": [
                "pt_gptneo_eleutherai_gpt_neo_125m_seq_cls_hf",
                "pt_opt_facebook_opt_125m_seq_cls_hf",
                "pt_opt_facebook_opt_125m_qa_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 32, 768)"},
        },
    ),
    (
        Reshape480,
        [((1, 12, 32, 64), torch.float32)],
        {
            "model_name": [
                "pt_gptneo_eleutherai_gpt_neo_125m_seq_cls_hf",
                "pt_opt_facebook_opt_125m_seq_cls_hf",
                "pt_opt_facebook_opt_125m_qa_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(12, 32, 64)"},
        },
    ),
    (
        Reshape481,
        [((12, 32, 32), torch.float32)],
        {
            "model_name": [
                "pt_gptneo_eleutherai_gpt_neo_125m_seq_cls_hf",
                "pt_opt_facebook_opt_125m_seq_cls_hf",
                "pt_opt_facebook_opt_125m_qa_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 12, 32, 32)"},
        },
    ),
    (
        Reshape482,
        [((1, 12, 32, 32), torch.float32)],
        {
            "model_name": [
                "pt_gptneo_eleutherai_gpt_neo_125m_seq_cls_hf",
                "pt_opt_facebook_opt_125m_seq_cls_hf",
                "pt_opt_facebook_opt_125m_qa_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(12, 32, 32)"},
        },
    ),
    (
        Reshape483,
        [((1, 12, 64, 32), torch.float32)],
        {
            "model_name": ["pt_gptneo_eleutherai_gpt_neo_125m_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(12, 64, 32)"},
        },
    ),
    (
        Reshape484,
        [((12, 32, 64), torch.float32)],
        {
            "model_name": [
                "pt_gptneo_eleutherai_gpt_neo_125m_seq_cls_hf",
                "pt_opt_facebook_opt_125m_seq_cls_hf",
                "pt_opt_facebook_opt_125m_qa_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 12, 32, 64)"},
        },
    ),
    (
        Reshape477,
        [((1, 32, 12, 64), torch.float32)],
        {
            "model_name": [
                "pt_gptneo_eleutherai_gpt_neo_125m_seq_cls_hf",
                "pt_opt_facebook_opt_125m_seq_cls_hf",
                "pt_opt_facebook_opt_125m_qa_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(32, 768)"},
        },
    ),
    (
        Reshape485,
        [((1, 256, 2560), torch.float32)],
        {
            "model_name": [
                "pt_gptneo_eleutherai_gpt_neo_2_7b_clm_hf",
                "pt_phi2_microsoft_phi_2_pytdml_clm_hf",
                "pt_phi2_microsoft_phi_2_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(256, 2560)"},
        },
    ),
    (
        Reshape486,
        [((1, 256, 2560), torch.float32)],
        {
            "model_name": ["pt_phi2_microsoft_phi_2_pytdml_clm_hf", "pt_phi2_microsoft_phi_2_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 256, 32, 80)"},
        },
    ),
    (
        Reshape487,
        [((256, 2560), torch.float32)],
        {
            "model_name": ["pt_gptneo_eleutherai_gpt_neo_2_7b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 256, 20, 128)"},
        },
    ),
    (
        Reshape488,
        [((256, 2560), torch.float32)],
        {
            "model_name": [
                "pt_gptneo_eleutherai_gpt_neo_2_7b_clm_hf",
                "pt_phi2_microsoft_phi_2_pytdml_clm_hf",
                "pt_phi2_microsoft_phi_2_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 256, 2560)"},
        },
    ),
    (
        Reshape489,
        [((1, 20, 256, 128), torch.float32)],
        {
            "model_name": ["pt_gptneo_eleutherai_gpt_neo_2_7b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(20, 256, 128)"},
        },
    ),
    (
        Reshape490,
        [((20, 256, 256), torch.float32)],
        {
            "model_name": ["pt_gptneo_eleutherai_gpt_neo_2_7b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 20, 256, 256)"},
        },
    ),
    (
        Reshape491,
        [((1, 20, 256, 256), torch.float32)],
        {
            "model_name": ["pt_gptneo_eleutherai_gpt_neo_2_7b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(20, 256, 256)"},
        },
    ),
    (
        Reshape492,
        [((1, 20, 128, 256), torch.float32)],
        {
            "model_name": ["pt_gptneo_eleutherai_gpt_neo_2_7b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(20, 128, 256)"},
        },
    ),
    (
        Reshape493,
        [((20, 256, 128), torch.float32)],
        {
            "model_name": ["pt_gptneo_eleutherai_gpt_neo_2_7b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 20, 256, 128)"},
        },
    ),
    (
        Reshape485,
        [((1, 256, 20, 128), torch.float32)],
        {
            "model_name": ["pt_gptneo_eleutherai_gpt_neo_2_7b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(256, 2560)"},
        },
    ),
    (
        Reshape494,
        [((1, 256, 2048), torch.float32)],
        {
            "model_name": [
                "pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_clm_hf",
                "pt_opt_facebook_opt_1_3b_clm_hf",
                "pt_xglm_facebook_xglm_1_7b_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(256, 2048)"},
        },
    ),
    (
        Reshape495,
        [((1, 256, 2048), torch.float32)],
        {"model_name": ["pt_opt_facebook_opt_1_3b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 256, 32, 64)"}},
    ),
    (
        Reshape496,
        [((1, 256, 2048), torch.float32)],
        {"model_name": ["pt_xglm_facebook_xglm_1_7b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 256, 16, 128)"}},
    ),
    (
        Reshape496,
        [((256, 2048), torch.float32)],
        {
            "model_name": ["pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 256, 16, 128)"},
        },
    ),
    (
        Reshape497,
        [((256, 2048), torch.float32)],
        {
            "model_name": [
                "pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_clm_hf",
                "pt_opt_facebook_opt_1_3b_clm_hf",
                "pt_xglm_facebook_xglm_1_7b_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 256, 2048)"},
        },
    ),
    (
        Reshape495,
        [((256, 2048), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 256, 32, 64)"},
        },
    ),
    (
        Reshape498,
        [((1, 16, 256, 128), torch.float32)],
        {
            "model_name": ["pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf", "pt_xglm_facebook_xglm_1_7b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(16, 256, 128)"},
        },
    ),
    (
        Reshape499,
        [((1, 16, 128, 256), torch.float32)],
        {
            "model_name": ["pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(16, 128, 256)"},
        },
    ),
    (
        Reshape500,
        [((16, 256, 128), torch.float32)],
        {
            "model_name": ["pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf", "pt_xglm_facebook_xglm_1_7b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 16, 256, 128)"},
        },
    ),
    (
        Reshape494,
        [((1, 256, 16, 128), torch.float32)],
        {
            "model_name": ["pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf", "pt_xglm_facebook_xglm_1_7b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(256, 2048)"},
        },
    ),
    (
        Reshape501,
        [((1, 32, 2048), torch.float32)],
        {
            "model_name": [
                "pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf",
                "pt_opt_facebook_opt_1_3b_seq_cls_hf",
                "pt_opt_facebook_opt_1_3b_qa_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(32, 2048)"},
        },
    ),
    (
        Reshape502,
        [((1, 32, 2048), torch.float32)],
        {
            "model_name": ["pt_opt_facebook_opt_1_3b_seq_cls_hf", "pt_opt_facebook_opt_1_3b_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 32, 32, 64)"},
        },
    ),
    (
        Reshape503,
        [((32, 2048), torch.float32)],
        {
            "model_name": ["pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 32, 16, 128)"},
        },
    ),
    (
        Reshape504,
        [((32, 2048), torch.float32)],
        {
            "model_name": [
                "pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf",
                "pt_opt_facebook_opt_1_3b_seq_cls_hf",
                "pt_opt_facebook_opt_1_3b_qa_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 32, 2048)"},
        },
    ),
    (
        Reshape505,
        [((1, 16, 32, 128), torch.float32)],
        {
            "model_name": ["pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(16, 32, 128)"},
        },
    ),
    (
        Reshape506,
        [((1, 16, 128, 32), torch.float32)],
        {
            "model_name": ["pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(16, 128, 32)"},
        },
    ),
    (
        Reshape507,
        [((16, 32, 128), torch.float32)],
        {
            "model_name": ["pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 16, 32, 128)"},
        },
    ),
    (
        Reshape501,
        [((1, 32, 16, 128), torch.float32)],
        {
            "model_name": ["pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(32, 2048)"},
        },
    ),
    (
        Reshape508,
        [((1, 4, 4096), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf",
                "pt_llama3_huggyllama_llama_7b_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(4, 4096)"},
        },
    ),
    (
        Reshape509,
        [((4, 4096), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf",
                "pt_llama3_huggyllama_llama_7b_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 4, 32, 128)"},
        },
    ),
    (
        Reshape510,
        [((4, 4096), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf",
                "pt_llama3_huggyllama_llama_7b_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 4, 4096)"},
        },
    ),
    (
        Reshape511,
        [((1, 32, 4, 128), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf",
                "pt_llama3_huggyllama_llama_7b_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(32, 4, 128)"},
        },
    ),
    (
        Reshape512,
        [((4, 1024), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_3b_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 4, 8, 128)"},
        },
    ),
    (
        Reshape511,
        [((1, 8, 4, 4, 128), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(32, 4, 128)"},
        },
    ),
    (
        Reshape513,
        [((1, 8, 4, 4, 128), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 32, 4, 128)"},
        },
    ),
    (
        Reshape514,
        [((32, 4, 4), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_seq_cls_hf",
                "pt_llama3_huggyllama_llama_7b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 32, 4, 4)"},
        },
    ),
    (
        Reshape515,
        [((1, 32, 4, 4), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_seq_cls_hf",
                "pt_llama3_huggyllama_llama_7b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(32, 4, 4)"},
        },
    ),
    (
        Reshape516,
        [((1, 32, 128, 4), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf",
                "pt_llama3_huggyllama_llama_7b_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(32, 128, 4)"},
        },
    ),
    (
        Reshape513,
        [((32, 4, 128), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf",
                "pt_llama3_huggyllama_llama_7b_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 32, 4, 128)"},
        },
    ),
    (
        Reshape508,
        [((1, 4, 32, 128), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf",
                "pt_llama3_huggyllama_llama_7b_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(4, 4096)"},
        },
    ),
    (
        Reshape517,
        [((4, 14336), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 4, 14336)"},
        },
    ),
    (
        Reshape518,
        [((1, 4, 2048), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_2_1b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(4, 2048)"},
        },
    ),
    (
        Reshape519,
        [((4, 2048), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_2_1b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 4, 32, 64)"},
        },
    ),
    (
        Reshape520,
        [((4, 2048), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_2_1b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 4, 2048)"},
        },
    ),
    (
        Reshape521,
        [((1, 32, 4, 64), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_2_1b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(32, 4, 64)"},
        },
    ),
    (
        Reshape522,
        [((4, 512), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_2_1b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 4, 8, 64)"},
        },
    ),
    (
        Reshape521,
        [((1, 8, 4, 4, 64), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_2_1b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(32, 4, 64)"},
        },
    ),
    (
        Reshape523,
        [((1, 8, 4, 4, 64), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_2_1b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 32, 4, 64)"},
        },
    ),
    (
        Reshape524,
        [((1, 32, 64, 4), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_2_1b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(32, 64, 4)"},
        },
    ),
    (
        Reshape523,
        [((32, 4, 64), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_2_1b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 32, 4, 64)"},
        },
    ),
    (
        Reshape518,
        [((1, 4, 32, 64), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_2_1b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(4, 2048)"},
        },
    ),
    (
        Reshape525,
        [((4, 8192), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_2_1b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_3b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 4, 8192)"},
        },
    ),
    (
        Reshape526,
        [((4, 11008), torch.float32)],
        {
            "model_name": ["pt_llama3_huggyllama_llama_7b_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 4, 11008)"},
        },
    ),
    (
        Reshape527,
        [((4, 2), torch.float32)],
        {"model_name": ["pt_llama3_huggyllama_llama_7b_seq_cls_hf"], "pcc": 0.99, "op_params": {"shape": "(4, 2)"}},
    ),
    (
        Reshape528,
        [((1, 4, 3072), torch.float32)],
        {
            "model_name": ["pt_llama3_meta_llama_llama_3_2_3b_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(4, 3072)"},
        },
    ),
    (
        Reshape529,
        [((4, 3072), torch.float32)],
        {
            "model_name": ["pt_llama3_meta_llama_llama_3_2_3b_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 4, 24, 128)"},
        },
    ),
    (
        Reshape530,
        [((4, 3072), torch.float32)],
        {
            "model_name": ["pt_llama3_meta_llama_llama_3_2_3b_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 4, 3072)"},
        },
    ),
    (
        Reshape531,
        [((1, 24, 4, 128), torch.float32)],
        {
            "model_name": ["pt_llama3_meta_llama_llama_3_2_3b_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(24, 4, 128)"},
        },
    ),
    (
        Reshape531,
        [((1, 8, 3, 4, 128), torch.float32)],
        {
            "model_name": ["pt_llama3_meta_llama_llama_3_2_3b_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(24, 4, 128)"},
        },
    ),
    (
        Reshape532,
        [((1, 8, 3, 4, 128), torch.float32)],
        {
            "model_name": ["pt_llama3_meta_llama_llama_3_2_3b_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 24, 4, 128)"},
        },
    ),
    (
        Reshape533,
        [((24, 4, 4), torch.float32)],
        {
            "model_name": ["pt_llama3_meta_llama_llama_3_2_3b_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 24, 4, 4)"},
        },
    ),
    (
        Reshape534,
        [((1, 24, 4, 4), torch.float32)],
        {
            "model_name": ["pt_llama3_meta_llama_llama_3_2_3b_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(24, 4, 4)"},
        },
    ),
    (
        Reshape535,
        [((1, 24, 128, 4), torch.float32)],
        {
            "model_name": ["pt_llama3_meta_llama_llama_3_2_3b_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(24, 128, 4)"},
        },
    ),
    (
        Reshape532,
        [((24, 4, 128), torch.float32)],
        {
            "model_name": ["pt_llama3_meta_llama_llama_3_2_3b_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 24, 4, 128)"},
        },
    ),
    (
        Reshape528,
        [((1, 4, 24, 128), torch.float32)],
        {
            "model_name": ["pt_llama3_meta_llama_llama_3_2_3b_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(4, 3072)"},
        },
    ),
    (
        Reshape536,
        [((1, 256, 4096), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_clm_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_clm_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_clm_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(256, 4096)"},
        },
    ),
    (
        Reshape537,
        [((1, 256, 4096), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 256, 64, 64)"},
        },
    ),
    (
        Reshape538,
        [((1, 32, 256, 128), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_clm_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_clm_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_clm_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(32, 256, 128)"},
        },
    ),
    (
        Reshape538,
        [((1, 8, 4, 256, 128), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_clm_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_clm_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_clm_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(32, 256, 128)"},
        },
    ),
    (
        Reshape539,
        [((1, 8, 4, 256, 128), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_clm_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_clm_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_clm_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 32, 256, 128)"},
        },
    ),
    (
        Reshape540,
        [((32, 256, 256), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_clm_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_clm_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_clm_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_clm_hf",
                "pt_opt_facebook_opt_1_3b_clm_hf",
                "pt_phi2_microsoft_phi_2_pytdml_clm_hf",
                "pt_phi2_microsoft_phi_2_clm_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf",
                "pt_phi3_5_microsoft_phi_3_5_mini_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 32, 256, 256)"},
        },
    ),
    (
        Reshape541,
        [((1, 32, 256, 256), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_clm_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_clm_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_clm_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_clm_hf",
                "pt_opt_facebook_opt_1_3b_clm_hf",
                "pt_phi2_microsoft_phi_2_pytdml_clm_hf",
                "pt_phi2_microsoft_phi_2_clm_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf",
                "pt_phi3_5_microsoft_phi_3_5_mini_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(32, 256, 256)"},
        },
    ),
    (
        Reshape542,
        [((1, 32, 128, 256), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_clm_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_clm_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_clm_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(32, 128, 256)"},
        },
    ),
    (
        Reshape539,
        [((32, 256, 128), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_clm_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_clm_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_clm_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 32, 256, 128)"},
        },
    ),
    (
        Reshape536,
        [((1, 256, 32, 128), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_clm_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_clm_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_clm_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(256, 4096)"},
        },
    ),
    (
        Reshape543,
        [((256, 14336), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_clm_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_clm_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_clm_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 256, 14336)"},
        },
    ),
    (
        Reshape544,
        [((1, 32, 256, 64), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_clm_hf",
                "pt_opt_facebook_opt_1_3b_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(32, 256, 64)"},
        },
    ),
    (
        Reshape545,
        [((256, 512), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 256, 8, 64)"},
        },
    ),
    (
        Reshape546,
        [((256, 512), torch.float32)],
        {"model_name": ["pt_opt_facebook_opt_350m_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(256, 512)"}},
    ),
    (
        Reshape547,
        [((256, 512), torch.float32)],
        {
            "model_name": [
                "pt_mlp_mixer_base_img_cls_github",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 256, 512)"},
        },
    ),
    (
        Reshape544,
        [((1, 8, 4, 256, 64), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(32, 256, 64)"},
        },
    ),
    (
        Reshape548,
        [((1, 8, 4, 256, 64), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 32, 256, 64)"},
        },
    ),
    (
        Reshape549,
        [((1, 32, 64, 256), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(32, 64, 256)"},
        },
    ),
    (
        Reshape548,
        [((32, 256, 64), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_clm_hf",
                "pt_opt_facebook_opt_1_3b_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 32, 256, 64)"},
        },
    ),
    (
        Reshape494,
        [((1, 256, 32, 64), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_clm_hf",
                "pt_opt_facebook_opt_1_3b_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(256, 2048)"},
        },
    ),
    (
        Reshape550,
        [((256, 8192), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_clm_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf",
                "pt_phi3_5_microsoft_phi_3_5_mini_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 256, 8192)"},
        },
    ),
    (
        Reshape551,
        [((1, 32, 3072), torch.float32)],
        {"model_name": ["pt_llama3_meta_llama_llama_3_2_3b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(32, 3072)"}},
    ),
    (
        Reshape552,
        [((32, 3072), torch.float32)],
        {
            "model_name": ["pt_llama3_meta_llama_llama_3_2_3b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 32, 24, 128)"},
        },
    ),
    (
        Reshape553,
        [((32, 3072), torch.float32)],
        {
            "model_name": ["pt_llama3_meta_llama_llama_3_2_3b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 32, 3072)"},
        },
    ),
    (
        Reshape554,
        [((1, 24, 32, 128), torch.float32)],
        {
            "model_name": ["pt_llama3_meta_llama_llama_3_2_3b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(24, 32, 128)"},
        },
    ),
    (
        Reshape555,
        [((32, 1024), torch.float32)],
        {
            "model_name": ["pt_llama3_meta_llama_llama_3_2_3b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 32, 8, 128)"},
        },
    ),
    (
        Reshape556,
        [((32, 1024), torch.float32)],
        {
            "model_name": ["pt_opt_facebook_opt_350m_seq_cls_hf", "pt_opt_facebook_opt_350m_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 32, 1024)"},
        },
    ),
    (
        Reshape554,
        [((1, 8, 3, 32, 128), torch.float32)],
        {
            "model_name": ["pt_llama3_meta_llama_llama_3_2_3b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(24, 32, 128)"},
        },
    ),
    (
        Reshape557,
        [((1, 8, 3, 32, 128), torch.float32)],
        {
            "model_name": ["pt_llama3_meta_llama_llama_3_2_3b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 24, 32, 128)"},
        },
    ),
    (
        Reshape558,
        [((24, 32, 32), torch.float32)],
        {
            "model_name": ["pt_llama3_meta_llama_llama_3_2_3b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 24, 32, 32)"},
        },
    ),
    (
        Reshape559,
        [((1, 24, 32, 32), torch.float32)],
        {
            "model_name": ["pt_llama3_meta_llama_llama_3_2_3b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(24, 32, 32)"},
        },
    ),
    (
        Reshape560,
        [((1, 24, 128, 32), torch.float32)],
        {
            "model_name": ["pt_llama3_meta_llama_llama_3_2_3b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(24, 128, 32)"},
        },
    ),
    (
        Reshape557,
        [((24, 32, 128), torch.float32)],
        {
            "model_name": ["pt_llama3_meta_llama_llama_3_2_3b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 24, 32, 128)"},
        },
    ),
    (
        Reshape551,
        [((1, 32, 24, 128), torch.float32)],
        {"model_name": ["pt_llama3_meta_llama_llama_3_2_3b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(32, 3072)"}},
    ),
    (
        Reshape561,
        [((32, 8192), torch.float32)],
        {
            "model_name": ["pt_llama3_meta_llama_llama_3_2_3b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 32, 8192)"},
        },
    ),
    (
        Reshape562,
        [((1, 32, 4096), torch.float32)],
        {"model_name": ["pt_llama3_huggyllama_llama_7b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(32, 4096)"}},
    ),
    (
        Reshape563,
        [((32, 4096), torch.float32)],
        {
            "model_name": ["pt_llama3_huggyllama_llama_7b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 32, 32, 128)"},
        },
    ),
    (
        Reshape564,
        [((32, 4096), torch.float32)],
        {"model_name": ["pt_llama3_huggyllama_llama_7b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 32, 4096)"}},
    ),
    (
        Reshape565,
        [((1, 32, 32, 128), torch.float32)],
        {"model_name": ["pt_llama3_huggyllama_llama_7b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(32, 32, 128)"}},
    ),
    (
        Reshape562,
        [((1, 32, 32, 128), torch.float32)],
        {"model_name": ["pt_llama3_huggyllama_llama_7b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(32, 4096)"}},
    ),
    (
        Reshape566,
        [((32, 32, 32), torch.float32)],
        {
            "model_name": [
                "pt_llama3_huggyllama_llama_7b_clm_hf",
                "pt_opt_facebook_opt_1_3b_seq_cls_hf",
                "pt_opt_facebook_opt_1_3b_qa_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 32, 32, 32)"},
        },
    ),
    (
        Reshape567,
        [((1, 32, 32, 32), torch.float32)],
        {
            "model_name": [
                "pt_llama3_huggyllama_llama_7b_clm_hf",
                "pt_opt_facebook_opt_1_3b_seq_cls_hf",
                "pt_opt_facebook_opt_1_3b_qa_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(32, 32, 32)"},
        },
    ),
    (
        Reshape568,
        [((1, 32, 128, 32), torch.float32)],
        {"model_name": ["pt_llama3_huggyllama_llama_7b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(32, 128, 32)"}},
    ),
    (
        Reshape563,
        [((32, 32, 128), torch.float32)],
        {
            "model_name": ["pt_llama3_huggyllama_llama_7b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 32, 32, 128)"},
        },
    ),
    (
        Reshape569,
        [((32, 11008), torch.float32)],
        {"model_name": ["pt_llama3_huggyllama_llama_7b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 32, 11008)"}},
    ),
    (
        Reshape570,
        [((1, 32, 128, 128), torch.float32)],
        {
            "model_name": ["pt_mistral_mistralai_mistral_7b_v0_1_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(32, 128, 128)"},
        },
    ),
    (
        Reshape571,
        [((1, 32, 128, 128), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 32, 16384, 1)"},
        },
    ),
    (
        Reshape570,
        [((1, 8, 4, 128, 128), torch.float32)],
        {
            "model_name": ["pt_mistral_mistralai_mistral_7b_v0_1_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(32, 128, 128)"},
        },
    ),
    (
        Reshape572,
        [((1, 8, 4, 128, 128), torch.float32)],
        {
            "model_name": ["pt_mistral_mistralai_mistral_7b_v0_1_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 32, 128, 128)"},
        },
    ),
    (
        Reshape572,
        [((32, 128, 128), torch.float32)],
        {
            "model_name": ["pt_mistral_mistralai_mistral_7b_v0_1_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 32, 128, 128)"},
        },
    ),
    (
        Reshape263,
        [((1, 128, 32, 128), torch.float32)],
        {
            "model_name": ["pt_mistral_mistralai_mistral_7b_v0_1_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(128, 4096)"},
        },
    ),
    (
        Reshape573,
        [((128, 14336), torch.float32)],
        {
            "model_name": ["pt_mistral_mistralai_mistral_7b_v0_1_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 128, 14336)"},
        },
    ),
    (
        Reshape574,
        [((1, 32), torch.int64)],
        {
            "model_name": [
                "pt_opt_facebook_opt_125m_seq_cls_hf",
                "pt_opt_facebook_opt_1_3b_seq_cls_hf",
                "pt_opt_facebook_opt_350m_seq_cls_hf",
                "pt_opt_facebook_opt_125m_qa_hf",
                "pt_opt_facebook_opt_1_3b_qa_hf",
                "pt_opt_facebook_opt_350m_qa_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 32)"},
        },
    ),
    (
        Reshape575,
        [((32, 2), torch.float32)],
        {
            "model_name": [
                "pt_opt_facebook_opt_125m_seq_cls_hf",
                "pt_opt_facebook_opt_1_3b_seq_cls_hf",
                "pt_opt_facebook_opt_350m_seq_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(32, 2)"},
        },
    ),
    (
        Reshape576,
        [((1, 32, 32, 64), torch.float32)],
        {
            "model_name": ["pt_opt_facebook_opt_1_3b_seq_cls_hf", "pt_opt_facebook_opt_1_3b_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(32, 32, 64)"},
        },
    ),
    (
        Reshape501,
        [((1, 32, 32, 64), torch.float32)],
        {
            "model_name": ["pt_opt_facebook_opt_1_3b_seq_cls_hf", "pt_opt_facebook_opt_1_3b_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(32, 2048)"},
        },
    ),
    (
        Reshape502,
        [((32, 32, 64), torch.float32)],
        {
            "model_name": ["pt_opt_facebook_opt_1_3b_seq_cls_hf", "pt_opt_facebook_opt_1_3b_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 32, 32, 64)"},
        },
    ),
    (
        Reshape577,
        [((1, 32, 1024), torch.float32)],
        {
            "model_name": ["pt_opt_facebook_opt_350m_seq_cls_hf", "pt_opt_facebook_opt_350m_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(32, 1024)"},
        },
    ),
    (
        Reshape578,
        [((1, 32, 1024), torch.float32)],
        {
            "model_name": ["pt_opt_facebook_opt_350m_seq_cls_hf", "pt_opt_facebook_opt_350m_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 32, 16, 64)"},
        },
    ),
    (
        Reshape579,
        [((1, 16, 32, 64), torch.float32)],
        {
            "model_name": ["pt_opt_facebook_opt_350m_seq_cls_hf", "pt_opt_facebook_opt_350m_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(16, 32, 64)"},
        },
    ),
    (
        Reshape580,
        [((16, 32, 64), torch.float32)],
        {
            "model_name": ["pt_opt_facebook_opt_350m_seq_cls_hf", "pt_opt_facebook_opt_350m_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 16, 32, 64)"},
        },
    ),
    (
        Reshape577,
        [((1, 32, 16, 64), torch.float32)],
        {
            "model_name": ["pt_opt_facebook_opt_350m_seq_cls_hf", "pt_opt_facebook_opt_350m_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(32, 1024)"},
        },
    ),
    (
        Reshape581,
        [((32, 512), torch.float32)],
        {
            "model_name": ["pt_opt_facebook_opt_350m_seq_cls_hf", "pt_opt_facebook_opt_350m_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(32, 512)"},
        },
    ),
    (
        Reshape582,
        [((32, 1), torch.float32)],
        {
            "model_name": [
                "pt_opt_facebook_opt_125m_qa_hf",
                "pt_opt_facebook_opt_1_3b_qa_hf",
                "pt_opt_facebook_opt_350m_qa_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 32, 1)"},
        },
    ),
    (
        Reshape583,
        [((256, 50272), torch.float32)],
        {"model_name": ["pt_opt_facebook_opt_350m_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 256, 50272)"}},
    ),
    (
        Reshape584,
        [((1, 512, 261), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 512, 1, 261)"},
        },
    ),
    (
        Reshape585,
        [((1, 224, 224, 3), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 50176, 3)"},
        },
    ),
    (
        Reshape586,
        [((1, 50176, 261), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(50176, 261)"},
        },
    ),
    (
        Reshape587,
        [((1, 50176, 261), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 50176, 1, 261)"},
        },
    ),
    (
        Reshape588,
        [((50176, 261), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 50176, 261)"},
        },
    ),
    (
        Reshape589,
        [((1, 512, 50176), torch.float32)],
        {
            "model_name": [
                "pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1, 512, 50176)"},
        },
    ),
    (
        Reshape590,
        [((1, 1, 512, 50176), torch.float32)],
        {
            "model_name": [
                "pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 512, 50176)"},
        },
    ),
    (
        Reshape591,
        [((1, 1, 261, 50176), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 261, 50176)"},
        },
    ),
    (
        Reshape592,
        [((1, 512, 1024), torch.float32)],
        {
            "model_name": [
                "pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(512, 1024)"},
        },
    ),
    (
        Reshape593,
        [((1, 512, 1024), torch.float32)],
        {
            "model_name": [
                "pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 512, 8, 128)"},
        },
    ),
    (
        Reshape594,
        [((1, 512, 1024), torch.float32)],
        {
            "model_name": [
                "pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 512, 1, 1024)"},
        },
    ),
    (
        Reshape595,
        [((512, 1024), torch.float32)],
        {
            "model_name": [
                "pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 512, 1024)"},
        },
    ),
    (
        Reshape596,
        [((1, 8, 512, 128), torch.float32)],
        {
            "model_name": [
                "pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(8, 512, 128)"},
        },
    ),
    (
        Reshape597,
        [((8, 512, 512), torch.float32)],
        {
            "model_name": [
                "pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 8, 512, 512)"},
        },
    ),
    (
        Reshape598,
        [((1, 8, 512, 512), torch.float32)],
        {
            "model_name": [
                "pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(8, 512, 512)"},
        },
    ),
    (
        Reshape599,
        [((1, 8, 128, 512), torch.float32)],
        {
            "model_name": [
                "pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(8, 128, 512)"},
        },
    ),
    (
        Reshape600,
        [((8, 512, 128), torch.float32)],
        {
            "model_name": [
                "pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 8, 512, 128)"},
        },
    ),
    (
        Reshape592,
        [((1, 512, 8, 128), torch.float32)],
        {
            "model_name": [
                "pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(512, 1024)"},
        },
    ),
    (
        Reshape156,
        [((1, 1, 1, 512), torch.float32)],
        {
            "model_name": [
                "pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1, 512)"},
        },
    ),
    (
        Reshape601,
        [((1, 1, 1024, 512), torch.float32)],
        {
            "model_name": [
                "pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1024, 512)"},
        },
    ),
    (
        Reshape602,
        [((1, 1, 1000), torch.float32)],
        {
            "model_name": [
                "pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1000)"},
        },
    ),
    (
        Reshape603,
        [((1, 512, 322), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 512, 1, 322)"},
        },
    ),
    (
        Reshape604,
        [((1, 55, 55, 64), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 3025, 64)"},
        },
    ),
    (
        Reshape605,
        [((1, 3025, 322), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(3025, 322)"},
        },
    ),
    (
        Reshape606,
        [((1, 3025, 322), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 3025, 1, 322)"},
        },
    ),
    (
        Reshape607,
        [((3025, 322), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 3025, 322)"},
        },
    ),
    (
        Reshape608,
        [((1, 512, 3025), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1, 512, 3025)"},
        },
    ),
    (
        Reshape609,
        [((1, 1, 512, 3025), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 512, 3025)"},
        },
    ),
    (
        Reshape610,
        [((1, 1, 322, 3025), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 322, 3025)"},
        },
    ),
    (
        Reshape611,
        [((1, 512, 512), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 512, 1, 512)"},
        },
    ),
    (
        Reshape612,
        [((1, 224, 224, 256), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 50176, 256)"},
        },
    ),
    (
        Reshape613,
        [((1, 50176, 512), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(50176, 512)"},
        },
    ),
    (
        Reshape614,
        [((1, 50176, 512), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 50176, 1, 512)"},
        },
    ),
    (
        Reshape615,
        [((50176, 512), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 50176, 512)"},
        },
    ),
    (
        Reshape616,
        [((1, 2048, 256), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 2048, 8, 32)"},
        },
    ),
    (
        Reshape617,
        [((1, 2048, 256), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 2048, 16, 16)"},
        },
    ),
    (
        Reshape618,
        [((1, 8, 2048, 32), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(8, 2048, 32)"},
        },
    ),
    (
        Reshape619,
        [((1, 256, 256), torch.float32)],
        {
            "model_name": [
                "pt_perceiverio_deepmind_language_perceiver_mlm_hf",
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 256, 8, 32)"},
        },
    ),
    (
        Reshape620,
        [((1, 256, 256), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(256, 256)"},
        },
    ),
    (
        Reshape621,
        [((1, 256, 256), torch.float32)],
        {"model_name": ["pt_segformer_nvidia_mit_b0_img_cls_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 256, 256)"}},
    ),
    (
        Reshape622,
        [((1, 256, 256), torch.float32)],
        {
            "model_name": ["pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 16, 16, 256)"},
        },
    ),
    (
        Reshape623,
        [((1, 256, 256), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 256, 16, 16)"},
        },
    ),
    (
        Reshape624,
        [((1, 8, 256, 32), torch.float32)],
        {
            "model_name": [
                "pt_perceiverio_deepmind_language_perceiver_mlm_hf",
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(8, 256, 32)"},
        },
    ),
    (
        Reshape625,
        [((1, 2048, 768), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(2048, 768)"},
        },
    ),
    (
        Reshape626,
        [((2048, 256), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 2048, 256)"},
        },
    ),
    (
        Reshape627,
        [((8, 256, 2048), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 8, 256, 2048)"},
        },
    ),
    (
        Reshape628,
        [((1, 8, 256, 2048), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(8, 256, 2048)"},
        },
    ),
    (
        Reshape629,
        [((2048, 1280), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 2048, 1280)"},
        },
    ),
    (
        Reshape630,
        [((1, 2048, 1280), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 2048, 8, 160)"},
        },
    ),
    (
        Reshape631,
        [((1, 8, 160, 2048), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(8, 160, 2048)"},
        },
    ),
    (
        Reshape632,
        [((8, 256, 160), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 8, 256, 160)"},
        },
    ),
    (
        Reshape633,
        [((1, 256, 8, 160), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(256, 1280)"},
        },
    ),
    (
        Reshape634,
        [((256, 1280), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 256, 1280)"},
        },
    ),
    (
        Reshape633,
        [((1, 256, 1280), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(256, 1280)"},
        },
    ),
    (
        Reshape635,
        [((1, 256, 1280), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 256, 8, 160)"},
        },
    ),
    (
        Reshape621,
        [((256, 256), torch.float32)],
        {
            "model_name": [
                "pt_perceiverio_deepmind_language_perceiver_mlm_hf",
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 256, 256)"},
        },
    ),
    (
        Reshape636,
        [((8, 256, 256), torch.float32)],
        {
            "model_name": [
                "pt_perceiverio_deepmind_language_perceiver_mlm_hf",
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 8, 256, 256)"},
        },
    ),
    (
        Reshape637,
        [((1, 8, 256, 256), torch.float32)],
        {
            "model_name": [
                "pt_perceiverio_deepmind_language_perceiver_mlm_hf",
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(8, 256, 256)"},
        },
    ),
    (
        Reshape638,
        [((1, 8, 160, 256), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(8, 160, 256)"},
        },
    ),
    (
        Reshape639,
        [((8, 2048, 256), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 8, 2048, 256)"},
        },
    ),
    (
        Reshape640,
        [((1, 8, 2048, 256), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(8, 2048, 256)"},
        },
    ),
    (
        Reshape641,
        [((1, 8, 96, 256), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(8, 96, 256)"},
        },
    ),
    (
        Reshape642,
        [((8, 2048, 96), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 8, 2048, 96)"},
        },
    ),
    (
        Reshape625,
        [((1, 2048, 8, 96), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(2048, 768)"},
        },
    ),
    (
        Reshape643,
        [((2048, 768), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 2048, 768)"},
        },
    ),
    (
        Reshape644,
        [((2048, 262), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 2048, 262)"},
        },
    ),
    (
        Reshape645,
        [((1, 32, 256, 80), torch.float32)],
        {
            "model_name": ["pt_phi2_microsoft_phi_2_pytdml_clm_hf", "pt_phi2_microsoft_phi_2_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(32, 256, 80)"},
        },
    ),
    (
        Reshape646,
        [((1, 32, 80, 256), torch.float32)],
        {
            "model_name": ["pt_phi2_microsoft_phi_2_pytdml_clm_hf", "pt_phi2_microsoft_phi_2_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(32, 80, 256)"},
        },
    ),
    (
        Reshape647,
        [((32, 256, 80), torch.float32)],
        {
            "model_name": ["pt_phi2_microsoft_phi_2_pytdml_clm_hf", "pt_phi2_microsoft_phi_2_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 32, 256, 80)"},
        },
    ),
    (
        Reshape485,
        [((1, 256, 32, 80), torch.float32)],
        {
            "model_name": ["pt_phi2_microsoft_phi_2_pytdml_clm_hf", "pt_phi2_microsoft_phi_2_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(256, 2560)"},
        },
    ),
    (
        Reshape648,
        [((256, 10240), torch.float32)],
        {
            "model_name": ["pt_phi2_microsoft_phi_2_pytdml_clm_hf", "pt_phi2_microsoft_phi_2_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 256, 10240)"},
        },
    ),
    (
        Reshape649,
        [((1, 12, 2560), torch.float32)],
        {
            "model_name": ["pt_phi2_microsoft_phi_2_pytdml_token_cls_hf", "pt_phi2_microsoft_phi_2_token_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(12, 2560)"},
        },
    ),
    (
        Reshape650,
        [((1, 12, 2560), torch.float32)],
        {
            "model_name": ["pt_phi2_microsoft_phi_2_pytdml_token_cls_hf", "pt_phi2_microsoft_phi_2_token_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 12, 32, 80)"},
        },
    ),
    (
        Reshape651,
        [((12, 2560), torch.float32)],
        {
            "model_name": ["pt_phi2_microsoft_phi_2_pytdml_token_cls_hf", "pt_phi2_microsoft_phi_2_token_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 12, 2560)"},
        },
    ),
    (
        Reshape652,
        [((1, 32, 12, 80), torch.float32)],
        {
            "model_name": ["pt_phi2_microsoft_phi_2_pytdml_token_cls_hf", "pt_phi2_microsoft_phi_2_token_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(32, 12, 80)"},
        },
    ),
    (
        Reshape653,
        [((32, 12, 12), torch.float32)],
        {
            "model_name": ["pt_phi2_microsoft_phi_2_pytdml_token_cls_hf", "pt_phi2_microsoft_phi_2_token_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 32, 12, 12)"},
        },
    ),
    (
        Reshape654,
        [((1, 32, 12, 12), torch.float32)],
        {
            "model_name": ["pt_phi2_microsoft_phi_2_pytdml_token_cls_hf", "pt_phi2_microsoft_phi_2_token_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(32, 12, 12)"},
        },
    ),
    (
        Reshape655,
        [((1, 32, 80, 12), torch.float32)],
        {
            "model_name": ["pt_phi2_microsoft_phi_2_pytdml_token_cls_hf", "pt_phi2_microsoft_phi_2_token_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(32, 80, 12)"},
        },
    ),
    (
        Reshape656,
        [((32, 12, 80), torch.float32)],
        {
            "model_name": ["pt_phi2_microsoft_phi_2_pytdml_token_cls_hf", "pt_phi2_microsoft_phi_2_token_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 32, 12, 80)"},
        },
    ),
    (
        Reshape649,
        [((1, 12, 32, 80), torch.float32)],
        {
            "model_name": ["pt_phi2_microsoft_phi_2_pytdml_token_cls_hf", "pt_phi2_microsoft_phi_2_token_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(12, 2560)"},
        },
    ),
    (
        Reshape657,
        [((12, 10240), torch.float32)],
        {
            "model_name": ["pt_phi2_microsoft_phi_2_pytdml_token_cls_hf", "pt_phi2_microsoft_phi_2_token_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 12, 10240)"},
        },
    ),
    (
        Reshape658,
        [((1, 11, 2560), torch.float32)],
        {
            "model_name": ["pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf", "pt_phi2_microsoft_phi_2_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(11, 2560)"},
        },
    ),
    (
        Reshape659,
        [((1, 11, 2560), torch.float32)],
        {
            "model_name": ["pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf", "pt_phi2_microsoft_phi_2_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 11, 32, 80)"},
        },
    ),
    (
        Reshape660,
        [((11, 2560), torch.float32)],
        {
            "model_name": ["pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf", "pt_phi2_microsoft_phi_2_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 11, 2560)"},
        },
    ),
    (
        Reshape661,
        [((1, 32, 11, 80), torch.float32)],
        {
            "model_name": ["pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf", "pt_phi2_microsoft_phi_2_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(32, 11, 80)"},
        },
    ),
    (
        Reshape662,
        [((32, 11, 11), torch.float32)],
        {
            "model_name": ["pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf", "pt_phi2_microsoft_phi_2_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 32, 11, 11)"},
        },
    ),
    (
        Reshape663,
        [((1, 32, 11, 11), torch.float32)],
        {
            "model_name": ["pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf", "pt_phi2_microsoft_phi_2_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(32, 11, 11)"},
        },
    ),
    (
        Reshape664,
        [((1, 32, 80, 11), torch.float32)],
        {
            "model_name": ["pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf", "pt_phi2_microsoft_phi_2_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(32, 80, 11)"},
        },
    ),
    (
        Reshape665,
        [((32, 11, 80), torch.float32)],
        {
            "model_name": ["pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf", "pt_phi2_microsoft_phi_2_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 32, 11, 80)"},
        },
    ),
    (
        Reshape658,
        [((1, 11, 32, 80), torch.float32)],
        {
            "model_name": ["pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf", "pt_phi2_microsoft_phi_2_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(11, 2560)"},
        },
    ),
    (
        Reshape666,
        [((11, 10240), torch.float32)],
        {
            "model_name": ["pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf", "pt_phi2_microsoft_phi_2_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 11, 10240)"},
        },
    ),
    (
        Reshape667,
        [((1, 32, 256, 96), torch.float32)],
        {
            "model_name": [
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf",
                "pt_phi3_5_microsoft_phi_3_5_mini_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(32, 256, 96)"},
        },
    ),
    (
        Reshape668,
        [((1, 32, 96, 256), torch.float32)],
        {
            "model_name": [
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf",
                "pt_phi3_5_microsoft_phi_3_5_mini_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(32, 96, 256)"},
        },
    ),
    (
        Reshape669,
        [((32, 256, 96), torch.float32)],
        {
            "model_name": [
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf",
                "pt_phi3_5_microsoft_phi_3_5_mini_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 32, 256, 96)"},
        },
    ),
    (
        Reshape455,
        [((1, 256, 32, 96), torch.float32)],
        {
            "model_name": [
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf",
                "pt_phi3_5_microsoft_phi_3_5_mini_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(256, 3072)"},
        },
    ),
    (
        Reshape670,
        [((1, 5, 3072), torch.float32)],
        {
            "model_name": ["pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 5, 32, 96)"},
        },
    ),
    (
        Reshape671,
        [((1, 5, 3072), torch.float32)],
        {
            "model_name": ["pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(5, 3072)"},
        },
    ),
    (
        Reshape672,
        [((1, 32, 5, 96), torch.float32)],
        {
            "model_name": ["pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(32, 5, 96)"},
        },
    ),
    (
        Reshape673,
        [((32, 5, 5), torch.float32)],
        {
            "model_name": ["pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 32, 5, 5)"},
        },
    ),
    (
        Reshape674,
        [((1, 32, 5, 5), torch.float32)],
        {
            "model_name": ["pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(32, 5, 5)"},
        },
    ),
    (
        Reshape675,
        [((1, 32, 96, 5), torch.float32)],
        {
            "model_name": ["pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(32, 96, 5)"},
        },
    ),
    (
        Reshape676,
        [((32, 5, 96), torch.float32)],
        {
            "model_name": ["pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 32, 5, 96)"},
        },
    ),
    (
        Reshape671,
        [((1, 5, 32, 96), torch.float32)],
        {
            "model_name": ["pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(5, 3072)"},
        },
    ),
    (
        Reshape677,
        [((5, 3072), torch.float32)],
        {
            "model_name": ["pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 5, 3072)"},
        },
    ),
    (
        Reshape678,
        [((5, 8192), torch.float32)],
        {
            "model_name": ["pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 5, 8192)"},
        },
    ),
    (
        Reshape679,
        [((1, 13, 3072), torch.float32)],
        {
            "model_name": ["pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 13, 32, 96)"},
        },
    ),
    (
        Reshape680,
        [((1, 13, 3072), torch.float32)],
        {
            "model_name": ["pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(13, 3072)"},
        },
    ),
    (
        Reshape681,
        [((1, 32, 13, 96), torch.float32)],
        {
            "model_name": ["pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(32, 13, 96)"},
        },
    ),
    (
        Reshape682,
        [((32, 13, 13), torch.float32)],
        {
            "model_name": ["pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 32, 13, 13)"},
        },
    ),
    (
        Reshape683,
        [((1, 32, 13, 13), torch.float32)],
        {
            "model_name": ["pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(32, 13, 13)"},
        },
    ),
    (
        Reshape684,
        [((1, 32, 96, 13), torch.float32)],
        {
            "model_name": ["pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(32, 96, 13)"},
        },
    ),
    (
        Reshape685,
        [((32, 13, 96), torch.float32)],
        {
            "model_name": ["pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 32, 13, 96)"},
        },
    ),
    (
        Reshape680,
        [((1, 13, 32, 96), torch.float32)],
        {
            "model_name": ["pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(13, 3072)"},
        },
    ),
    (
        Reshape686,
        [((13, 3072), torch.float32)],
        {
            "model_name": ["pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 13, 3072)"},
        },
    ),
    (
        Reshape687,
        [((13, 8192), torch.float32)],
        {
            "model_name": ["pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 13, 8192)"},
        },
    ),
    (
        Reshape688,
        [((1, 6, 1024), torch.float32)],
        {"model_name": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(6, 1024)"}},
    ),
    (
        Reshape689,
        [((1, 6, 1024), torch.float32)],
        {"model_name": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 6, 16, 64)"}},
    ),
    (
        Reshape690,
        [((6, 1024), torch.float32)],
        {"model_name": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 6, 1024)"}},
    ),
    (
        Reshape691,
        [((1, 16, 6, 64), torch.float32)],
        {"model_name": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(16, 6, 64)"}},
    ),
    (
        Reshape692,
        [((16, 6, 6), torch.float32)],
        {"model_name": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 16, 6, 6)"}},
    ),
    (
        Reshape693,
        [((1, 16, 6, 6), torch.float32)],
        {"model_name": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(16, 6, 6)"}},
    ),
    (
        Reshape694,
        [((1, 16, 64, 6), torch.float32)],
        {"model_name": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(16, 64, 6)"}},
    ),
    (
        Reshape695,
        [((16, 6, 64), torch.float32)],
        {"model_name": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 16, 6, 64)"}},
    ),
    (
        Reshape688,
        [((1, 6, 16, 64), torch.float32)],
        {"model_name": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(6, 1024)"}},
    ),
    (
        Reshape696,
        [((6, 2816), torch.float32)],
        {"model_name": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 6, 2816)"}},
    ),
    (
        Reshape697,
        [((1, 29, 1024), torch.float32)],
        {"model_name": ["pt_qwen1_5_qwen_qwen1_5_0_5b_chat_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(29, 1024)"}},
    ),
    (
        Reshape698,
        [((1, 29, 1024), torch.float32)],
        {
            "model_name": ["pt_qwen1_5_qwen_qwen1_5_0_5b_chat_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 29, 16, 64)"},
        },
    ),
    (
        Reshape699,
        [((29, 1024), torch.float32)],
        {
            "model_name": ["pt_qwen1_5_qwen_qwen1_5_0_5b_chat_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 29, 1024)"},
        },
    ),
    (
        Reshape700,
        [((1, 16, 29, 64), torch.float32)],
        {
            "model_name": ["pt_qwen1_5_qwen_qwen1_5_0_5b_chat_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(16, 29, 64)"},
        },
    ),
    (
        Reshape701,
        [((16, 29, 29), torch.float32)],
        {
            "model_name": ["pt_qwen1_5_qwen_qwen1_5_0_5b_chat_clm_hf", "pt_qwen_v2_qwen_qwen2_5_3b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 16, 29, 29)"},
        },
    ),
    (
        Reshape702,
        [((1, 16, 29, 29), torch.float32)],
        {
            "model_name": ["pt_qwen1_5_qwen_qwen1_5_0_5b_chat_clm_hf", "pt_qwen_v2_qwen_qwen2_5_3b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(16, 29, 29)"},
        },
    ),
    (
        Reshape703,
        [((1, 16, 64, 29), torch.float32)],
        {
            "model_name": ["pt_qwen1_5_qwen_qwen1_5_0_5b_chat_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(16, 64, 29)"},
        },
    ),
    (
        Reshape704,
        [((16, 29, 64), torch.float32)],
        {
            "model_name": ["pt_qwen1_5_qwen_qwen1_5_0_5b_chat_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 16, 29, 64)"},
        },
    ),
    (
        Reshape697,
        [((1, 29, 16, 64), torch.float32)],
        {"model_name": ["pt_qwen1_5_qwen_qwen1_5_0_5b_chat_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(29, 1024)"}},
    ),
    (
        Reshape705,
        [((29, 2816), torch.float32)],
        {
            "model_name": ["pt_qwen1_5_qwen_qwen1_5_0_5b_chat_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 29, 2816)"},
        },
    ),
    (
        Reshape706,
        [((1, 35, 2048), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(35, 2048)"},
        },
    ),
    (
        Reshape707,
        [((1, 35, 2048), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 35, 16, 128)"},
        },
    ),
    (
        Reshape708,
        [((35, 2048), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 35, 2048)"},
        },
    ),
    (
        Reshape709,
        [((1, 16, 35, 128), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(16, 35, 128)"},
        },
    ),
    (
        Reshape710,
        [((35, 256), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 35, 256)"},
        },
    ),
    (
        Reshape711,
        [((1, 35, 256), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 35, 2, 128)"},
        },
    ),
    (
        Reshape709,
        [((1, 2, 8, 35, 128), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(16, 35, 128)"},
        },
    ),
    (
        Reshape712,
        [((1, 2, 8, 35, 128), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 16, 35, 128)"},
        },
    ),
    (
        Reshape713,
        [((16, 35, 35), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 16, 35, 35)"},
        },
    ),
    (
        Reshape714,
        [((1, 16, 35, 35), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(16, 35, 35)"},
        },
    ),
    (
        Reshape715,
        [((1, 16, 128, 35), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(16, 128, 35)"},
        },
    ),
    (
        Reshape712,
        [((16, 35, 128), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 16, 35, 128)"},
        },
    ),
    (
        Reshape706,
        [((1, 35, 16, 128), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(35, 2048)"},
        },
    ),
    (
        Reshape716,
        [((35, 11008), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 35, 11008)"},
        },
    ),
    (
        Reshape717,
        [((1, 35, 896), torch.float32)],
        {
            "model_name": ["pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(35, 896)"},
        },
    ),
    (
        Reshape718,
        [((1, 35, 896), torch.float32)],
        {
            "model_name": ["pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 35, 14, 64)"},
        },
    ),
    (
        Reshape719,
        [((35, 896), torch.float32)],
        {
            "model_name": ["pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 35, 896)"},
        },
    ),
    (
        Reshape720,
        [((1, 14, 35, 64), torch.float32)],
        {
            "model_name": ["pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(14, 35, 64)"},
        },
    ),
    (
        Reshape721,
        [((35, 128), torch.float32)],
        {
            "model_name": ["pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 35, 128)"},
        },
    ),
    (
        Reshape722,
        [((1, 35, 128), torch.float32)],
        {
            "model_name": ["pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 35, 2, 64)"},
        },
    ),
    (
        Reshape720,
        [((1, 2, 7, 35, 64), torch.float32)],
        {
            "model_name": ["pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(14, 35, 64)"},
        },
    ),
    (
        Reshape723,
        [((1, 2, 7, 35, 64), torch.float32)],
        {
            "model_name": ["pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 14, 35, 64)"},
        },
    ),
    (
        Reshape724,
        [((14, 35, 35), torch.float32)],
        {
            "model_name": ["pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 14, 35, 35)"},
        },
    ),
    (
        Reshape725,
        [((1, 14, 35, 35), torch.float32)],
        {
            "model_name": ["pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(14, 35, 35)"},
        },
    ),
    (
        Reshape726,
        [((1, 14, 64, 35), torch.float32)],
        {
            "model_name": ["pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(14, 64, 35)"},
        },
    ),
    (
        Reshape723,
        [((14, 35, 64), torch.float32)],
        {
            "model_name": ["pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 14, 35, 64)"},
        },
    ),
    (
        Reshape717,
        [((1, 35, 14, 64), torch.float32)],
        {
            "model_name": ["pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(35, 896)"},
        },
    ),
    (
        Reshape727,
        [((35, 4864), torch.float32)],
        {
            "model_name": ["pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 35, 4864)"},
        },
    ),
    (
        Reshape728,
        [((1, 35, 1536), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(35, 1536)"},
        },
    ),
    (
        Reshape729,
        [((1, 35, 1536), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 35, 12, 128)"},
        },
    ),
    (
        Reshape730,
        [((35, 1536), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 35, 1536)"},
        },
    ),
    (
        Reshape731,
        [((1, 12, 35, 128), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(12, 35, 128)"},
        },
    ),
    (
        Reshape731,
        [((1, 2, 6, 35, 128), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(12, 35, 128)"},
        },
    ),
    (
        Reshape732,
        [((1, 2, 6, 35, 128), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 12, 35, 128)"},
        },
    ),
    (
        Reshape733,
        [((12, 35, 35), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 12, 35, 35)"},
        },
    ),
    (
        Reshape734,
        [((1, 12, 35, 35), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(12, 35, 35)"},
        },
    ),
    (
        Reshape735,
        [((1, 12, 128, 35), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(12, 128, 35)"},
        },
    ),
    (
        Reshape732,
        [((12, 35, 128), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 12, 35, 128)"},
        },
    ),
    (
        Reshape728,
        [((1, 35, 12, 128), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(35, 1536)"},
        },
    ),
    (
        Reshape736,
        [((35, 8960), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 35, 8960)"},
        },
    ),
    (
        Reshape737,
        [((1, 35, 3584), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(35, 3584)"},
        },
    ),
    (
        Reshape738,
        [((1, 35, 3584), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 35, 28, 128)"},
        },
    ),
    (
        Reshape739,
        [((35, 3584), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 35, 3584)"},
        },
    ),
    (
        Reshape740,
        [((1, 28, 35, 128), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(28, 35, 128)"},
        },
    ),
    (
        Reshape741,
        [((35, 512), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 35, 512)"},
        },
    ),
    (
        Reshape742,
        [((1, 35, 512), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 35, 4, 128)"},
        },
    ),
    (
        Reshape740,
        [((1, 4, 7, 35, 128), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(28, 35, 128)"},
        },
    ),
    (
        Reshape743,
        [((1, 4, 7, 35, 128), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 28, 35, 128)"},
        },
    ),
    (
        Reshape744,
        [((28, 35, 35), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 28, 35, 35)"},
        },
    ),
    (
        Reshape745,
        [((1, 28, 35, 35), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(28, 35, 35)"},
        },
    ),
    (
        Reshape746,
        [((1, 28, 128, 35), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(28, 128, 35)"},
        },
    ),
    (
        Reshape743,
        [((28, 35, 128), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 28, 35, 128)"},
        },
    ),
    (
        Reshape737,
        [((1, 35, 28, 128), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(35, 3584)"},
        },
    ),
    (
        Reshape747,
        [((35, 18944), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 35, 18944)"},
        },
    ),
    (
        Reshape748,
        [((1, 29, 1536), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(29, 1536)"}},
    ),
    (
        Reshape749,
        [((1, 29, 1536), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 29, 12, 128)"},
        },
    ),
    (
        Reshape750,
        [((29, 1536), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 29, 1536)"}},
    ),
    (
        Reshape751,
        [((1, 12, 29, 128), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(12, 29, 128)"}},
    ),
    (
        Reshape752,
        [((29, 256), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf", "pt_qwen_v2_qwen_qwen2_5_3b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 29, 256)"},
        },
    ),
    (
        Reshape753,
        [((1, 29, 256), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf", "pt_qwen_v2_qwen_qwen2_5_3b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 29, 2, 128)"},
        },
    ),
    (
        Reshape751,
        [((1, 2, 6, 29, 128), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(12, 29, 128)"}},
    ),
    (
        Reshape754,
        [((1, 2, 6, 29, 128), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 12, 29, 128)"},
        },
    ),
    (
        Reshape755,
        [((12, 29, 29), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 12, 29, 29)"}},
    ),
    (
        Reshape756,
        [((1, 12, 29, 29), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(12, 29, 29)"}},
    ),
    (
        Reshape757,
        [((1, 12, 128, 29), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(12, 128, 29)"}},
    ),
    (
        Reshape754,
        [((12, 29, 128), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 12, 29, 128)"},
        },
    ),
    (
        Reshape748,
        [((1, 29, 12, 128), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(29, 1536)"}},
    ),
    (
        Reshape758,
        [((29, 8960), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 29, 8960)"}},
    ),
    (
        Reshape759,
        [((1, 39, 1536), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(39, 1536)"},
        },
    ),
    (
        Reshape760,
        [((1, 39, 1536), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 39, 12, 128)"},
        },
    ),
    (
        Reshape761,
        [((39, 1536), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 39, 1536)"},
        },
    ),
    (
        Reshape762,
        [((1, 12, 39, 128), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(12, 39, 128)"},
        },
    ),
    (
        Reshape763,
        [((39, 256), torch.float32)],
        {
            "model_name": [
                "pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf",
                "pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 39, 256)"},
        },
    ),
    (
        Reshape764,
        [((1, 39, 256), torch.float32)],
        {
            "model_name": [
                "pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf",
                "pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 39, 2, 128)"},
        },
    ),
    (
        Reshape762,
        [((1, 2, 6, 39, 128), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(12, 39, 128)"},
        },
    ),
    (
        Reshape765,
        [((1, 2, 6, 39, 128), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 12, 39, 128)"},
        },
    ),
    (
        Reshape766,
        [((12, 39, 39), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 12, 39, 39)"},
        },
    ),
    (
        Reshape767,
        [((1, 12, 39, 39), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(12, 39, 39)"},
        },
    ),
    (
        Reshape768,
        [((1, 12, 128, 39), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(12, 128, 39)"},
        },
    ),
    (
        Reshape765,
        [((12, 39, 128), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 12, 39, 128)"},
        },
    ),
    (
        Reshape759,
        [((1, 39, 12, 128), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(39, 1536)"},
        },
    ),
    (
        Reshape769,
        [((39, 8960), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 39, 8960)"},
        },
    ),
    (
        Reshape770,
        [((1, 13, 3584), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_7b_token_cls_hf"], "pcc": 0.99, "op_params": {"shape": "(13, 3584)"}},
    ),
    (
        Reshape771,
        [((1, 13, 3584), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_7b_token_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 13, 28, 128)"},
        },
    ),
    (
        Reshape772,
        [((13, 3584), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_7b_token_cls_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 13, 3584)"}},
    ),
    (
        Reshape773,
        [((1, 28, 13, 128), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_7b_token_cls_hf"], "pcc": 0.99, "op_params": {"shape": "(28, 13, 128)"}},
    ),
    (
        Reshape774,
        [((13, 512), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_7b_token_cls_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 13, 512)"}},
    ),
    (
        Reshape775,
        [((1, 13, 512), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_7b_token_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 13, 4, 128)"},
        },
    ),
    (
        Reshape773,
        [((1, 4, 7, 13, 128), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_7b_token_cls_hf"], "pcc": 0.99, "op_params": {"shape": "(28, 13, 128)"}},
    ),
    (
        Reshape776,
        [((1, 4, 7, 13, 128), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_7b_token_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 28, 13, 128)"},
        },
    ),
    (
        Reshape777,
        [((28, 13, 13), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_7b_token_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 28, 13, 13)"},
        },
    ),
    (
        Reshape778,
        [((1, 28, 13, 13), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_7b_token_cls_hf"], "pcc": 0.99, "op_params": {"shape": "(28, 13, 13)"}},
    ),
    (
        Reshape779,
        [((1, 28, 128, 13), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_7b_token_cls_hf"], "pcc": 0.99, "op_params": {"shape": "(28, 128, 13)"}},
    ),
    (
        Reshape776,
        [((28, 13, 128), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_7b_token_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 28, 13, 128)"},
        },
    ),
    (
        Reshape770,
        [((1, 13, 28, 128), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_7b_token_cls_hf"], "pcc": 0.99, "op_params": {"shape": "(13, 3584)"}},
    ),
    (
        Reshape780,
        [((13, 18944), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_7b_token_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 13, 18944)"},
        },
    ),
    (
        Reshape781,
        [((1, 39, 3584), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(39, 3584)"},
        },
    ),
    (
        Reshape782,
        [((1, 39, 3584), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 39, 28, 128)"},
        },
    ),
    (
        Reshape783,
        [((39, 3584), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 39, 3584)"},
        },
    ),
    (
        Reshape784,
        [((1, 28, 39, 128), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(28, 39, 128)"},
        },
    ),
    (
        Reshape785,
        [((39, 512), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 39, 512)"},
        },
    ),
    (
        Reshape786,
        [((1, 39, 512), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 39, 4, 128)"},
        },
    ),
    (
        Reshape784,
        [((1, 4, 7, 39, 128), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(28, 39, 128)"},
        },
    ),
    (
        Reshape787,
        [((1, 4, 7, 39, 128), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 28, 39, 128)"},
        },
    ),
    (
        Reshape788,
        [((28, 39, 39), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 28, 39, 39)"},
        },
    ),
    (
        Reshape789,
        [((1, 28, 39, 39), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(28, 39, 39)"},
        },
    ),
    (
        Reshape790,
        [((1, 28, 128, 39), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(28, 128, 39)"},
        },
    ),
    (
        Reshape787,
        [((28, 39, 128), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 28, 39, 128)"},
        },
    ),
    (
        Reshape781,
        [((1, 39, 28, 128), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(39, 3584)"},
        },
    ),
    (
        Reshape791,
        [((39, 18944), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 39, 18944)"},
        },
    ),
    (
        Reshape792,
        [((1, 39, 896), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(39, 896)"},
        },
    ),
    (
        Reshape793,
        [((1, 39, 896), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 39, 14, 64)"},
        },
    ),
    (
        Reshape794,
        [((39, 896), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 39, 896)"},
        },
    ),
    (
        Reshape795,
        [((1, 14, 39, 64), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(14, 39, 64)"},
        },
    ),
    (
        Reshape796,
        [((39, 128), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 39, 128)"},
        },
    ),
    (
        Reshape797,
        [((1, 39, 128), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 39, 2, 64)"},
        },
    ),
    (
        Reshape795,
        [((1, 2, 7, 39, 64), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(14, 39, 64)"},
        },
    ),
    (
        Reshape798,
        [((1, 2, 7, 39, 64), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 14, 39, 64)"},
        },
    ),
    (
        Reshape799,
        [((14, 39, 39), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 14, 39, 39)"},
        },
    ),
    (
        Reshape800,
        [((1, 14, 39, 39), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(14, 39, 39)"},
        },
    ),
    (
        Reshape801,
        [((1, 14, 64, 39), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(14, 64, 39)"},
        },
    ),
    (
        Reshape798,
        [((14, 39, 64), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 14, 39, 64)"},
        },
    ),
    (
        Reshape792,
        [((1, 39, 14, 64), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(39, 896)"},
        },
    ),
    (
        Reshape802,
        [((39, 4864), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 39, 4864)"},
        },
    ),
    (
        Reshape803,
        [((1, 29, 896), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(29, 896)"}},
    ),
    (
        Reshape804,
        [((1, 29, 896), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 29, 14, 64)"}},
    ),
    (
        Reshape805,
        [((29, 896), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 29, 896)"}},
    ),
    (
        Reshape806,
        [((1, 14, 29, 64), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(14, 29, 64)"}},
    ),
    (
        Reshape807,
        [((29, 128), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 29, 128)"}},
    ),
    (
        Reshape808,
        [((1, 29, 128), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 29, 2, 64)"}},
    ),
    (
        Reshape806,
        [((1, 2, 7, 29, 64), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(14, 29, 64)"}},
    ),
    (
        Reshape809,
        [((1, 2, 7, 29, 64), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 14, 29, 64)"}},
    ),
    (
        Reshape810,
        [((14, 29, 29), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 14, 29, 29)"}},
    ),
    (
        Reshape811,
        [((1, 14, 29, 29), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(14, 29, 29)"}},
    ),
    (
        Reshape812,
        [((1, 14, 64, 29), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(14, 64, 29)"}},
    ),
    (
        Reshape809,
        [((14, 29, 64), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 14, 29, 64)"}},
    ),
    (
        Reshape803,
        [((1, 29, 14, 64), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(29, 896)"}},
    ),
    (
        Reshape813,
        [((29, 4864), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 29, 4864)"}},
    ),
    (
        Reshape814,
        [((1, 29, 2048), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_3b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(29, 2048)"}},
    ),
    (
        Reshape815,
        [((1, 29, 2048), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_3b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 29, 16, 128)"}},
    ),
    (
        Reshape816,
        [((29, 2048), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_3b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 29, 2048)"}},
    ),
    (
        Reshape817,
        [((1, 16, 29, 128), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_3b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(16, 29, 128)"}},
    ),
    (
        Reshape817,
        [((1, 2, 8, 29, 128), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_3b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(16, 29, 128)"}},
    ),
    (
        Reshape818,
        [((1, 2, 8, 29, 128), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_3b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 16, 29, 128)"}},
    ),
    (
        Reshape819,
        [((1, 16, 128, 29), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_3b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(16, 128, 29)"}},
    ),
    (
        Reshape818,
        [((16, 29, 128), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_3b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 16, 29, 128)"}},
    ),
    (
        Reshape814,
        [((1, 29, 16, 128), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_3b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(29, 2048)"}},
    ),
    (
        Reshape820,
        [((29, 11008), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_3b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 29, 11008)"}},
    ),
    (
        Reshape821,
        [((1, 39, 2048), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(39, 2048)"},
        },
    ),
    (
        Reshape822,
        [((1, 39, 2048), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 39, 16, 128)"},
        },
    ),
    (
        Reshape823,
        [((39, 2048), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 39, 2048)"},
        },
    ),
    (
        Reshape824,
        [((1, 16, 39, 128), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(16, 39, 128)"},
        },
    ),
    (
        Reshape824,
        [((1, 2, 8, 39, 128), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(16, 39, 128)"},
        },
    ),
    (
        Reshape825,
        [((1, 2, 8, 39, 128), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 16, 39, 128)"},
        },
    ),
    (
        Reshape826,
        [((16, 39, 39), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 16, 39, 39)"},
        },
    ),
    (
        Reshape827,
        [((1, 16, 39, 39), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(16, 39, 39)"},
        },
    ),
    (
        Reshape828,
        [((1, 16, 128, 39), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(16, 128, 39)"},
        },
    ),
    (
        Reshape825,
        [((16, 39, 128), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 16, 39, 128)"},
        },
    ),
    (
        Reshape821,
        [((1, 39, 16, 128), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(39, 2048)"},
        },
    ),
    (
        Reshape829,
        [((1, 29, 3584), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_7b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(29, 3584)"}},
    ),
    (
        Reshape830,
        [((1, 29, 3584), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_7b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 29, 28, 128)"}},
    ),
    (
        Reshape831,
        [((29, 3584), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_7b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 29, 3584)"}},
    ),
    (
        Reshape832,
        [((1, 28, 29, 128), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_7b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(28, 29, 128)"}},
    ),
    (
        Reshape833,
        [((29, 512), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_7b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 29, 512)"}},
    ),
    (
        Reshape834,
        [((1, 29, 512), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_7b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 29, 4, 128)"}},
    ),
    (
        Reshape832,
        [((1, 4, 7, 29, 128), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_7b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(28, 29, 128)"}},
    ),
    (
        Reshape835,
        [((1, 4, 7, 29, 128), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_7b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 28, 29, 128)"}},
    ),
    (
        Reshape836,
        [((28, 29, 29), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_7b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 28, 29, 29)"}},
    ),
    (
        Reshape837,
        [((1, 28, 29, 29), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_7b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(28, 29, 29)"}},
    ),
    (
        Reshape838,
        [((1, 28, 128, 29), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_7b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(28, 128, 29)"}},
    ),
    (
        Reshape835,
        [((28, 29, 128), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_7b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 28, 29, 128)"}},
    ),
    (
        Reshape829,
        [((1, 29, 28, 128), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_7b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(29, 3584)"}},
    ),
    (
        Reshape839,
        [((29, 18944), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_7b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 29, 18944)"}},
    ),
    (
        Reshape840,
        [((1, 768, 128), torch.float32)],
        {
            "model_name": ["pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 12, 64, 128)"},
        },
    ),
    (
        Reshape280,
        [((1, 768, 128), torch.float32)],
        {
            "model_name": ["pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(12, 64, 128)"},
        },
    ),
    (
        Reshape841,
        [((768, 768, 1), torch.float32)],
        {
            "model_name": ["pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(768, 768, 1, 1)"},
        },
    ),
    (
        Reshape842,
        [((1, 768, 128, 1), torch.float32)],
        {
            "model_name": ["pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 768, 128)"},
        },
    ),
    (
        Reshape843,
        [((1, 61), torch.int64)],
        {
            "model_name": [
                "pt_t5_t5_large_text_gen_hf",
                "pt_t5_t5_small_text_gen_hf",
                "pt_t5_google_flan_t5_large_text_gen_hf",
                "pt_t5_t5_base_text_gen_hf",
                "pt_t5_google_flan_t5_small_text_gen_hf",
                "pt_t5_google_flan_t5_base_text_gen_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 61)"},
        },
    ),
    (
        Reshape844,
        [((1, 61, 1024), torch.float32)],
        {
            "model_name": ["pt_t5_t5_large_text_gen_hf", "pt_t5_google_flan_t5_large_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(61, 1024)"},
        },
    ),
    (
        Reshape845,
        [((61, 1024), torch.float32)],
        {
            "model_name": ["pt_t5_t5_large_text_gen_hf", "pt_t5_google_flan_t5_large_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 61, 16, 64)"},
        },
    ),
    (
        Reshape846,
        [((61, 1024), torch.float32)],
        {
            "model_name": [
                "pt_t5_t5_large_text_gen_hf",
                "pt_t5_google_flan_t5_large_text_gen_hf",
                "pt_t5_google_flan_t5_small_text_gen_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 61, 1024)"},
        },
    ),
    (
        Reshape847,
        [((1, 16, 61, 64), torch.float32)],
        {
            "model_name": ["pt_t5_t5_large_text_gen_hf", "pt_t5_google_flan_t5_large_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(16, 61, 64)"},
        },
    ),
    (
        Reshape848,
        [((16, 61, 61), torch.float32)],
        {
            "model_name": ["pt_t5_t5_large_text_gen_hf", "pt_t5_google_flan_t5_large_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 16, 61, 61)"},
        },
    ),
    (
        Reshape849,
        [((1, 16, 61, 61), torch.float32)],
        {
            "model_name": ["pt_t5_t5_large_text_gen_hf", "pt_t5_google_flan_t5_large_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(16, 61, 61)"},
        },
    ),
    (
        Reshape850,
        [((1, 16, 64, 61), torch.float32)],
        {
            "model_name": ["pt_t5_t5_large_text_gen_hf", "pt_t5_google_flan_t5_large_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(16, 64, 61)"},
        },
    ),
    (
        Reshape851,
        [((16, 61, 64), torch.float32)],
        {
            "model_name": ["pt_t5_t5_large_text_gen_hf", "pt_t5_google_flan_t5_large_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 16, 61, 64)"},
        },
    ),
    (
        Reshape844,
        [((1, 61, 16, 64), torch.float32)],
        {
            "model_name": ["pt_t5_t5_large_text_gen_hf", "pt_t5_google_flan_t5_large_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(61, 1024)"},
        },
    ),
    (
        Reshape852,
        [((16, 1, 61), torch.float32)],
        {
            "model_name": ["pt_t5_t5_large_text_gen_hf", "pt_t5_google_flan_t5_large_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 16, 1, 61)"},
        },
    ),
    (
        Reshape853,
        [((1, 16, 1, 61), torch.float32)],
        {
            "model_name": ["pt_t5_t5_large_text_gen_hf", "pt_t5_google_flan_t5_large_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(16, 1, 61)"},
        },
    ),
    (
        Reshape854,
        [((1, 61, 512), torch.float32)],
        {
            "model_name": ["pt_t5_t5_small_text_gen_hf", "pt_t5_google_flan_t5_small_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(61, 512)"},
        },
    ),
    (
        Reshape855,
        [((61, 512), torch.float32)],
        {"model_name": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 61, 8, 64)"}},
    ),
    (
        Reshape856,
        [((61, 512), torch.float32)],
        {
            "model_name": ["pt_t5_t5_small_text_gen_hf", "pt_t5_google_flan_t5_small_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 61, 512)"},
        },
    ),
    (
        Reshape857,
        [((1, 8, 61, 64), torch.float32)],
        {"model_name": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99, "op_params": {"shape": "(8, 61, 64)"}},
    ),
    (
        Reshape858,
        [((8, 61, 61), torch.float32)],
        {"model_name": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 8, 61, 61)"}},
    ),
    (
        Reshape859,
        [((1, 8, 61, 61), torch.float32)],
        {"model_name": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99, "op_params": {"shape": "(8, 61, 61)"}},
    ),
    (
        Reshape860,
        [((1, 8, 64, 61), torch.float32)],
        {"model_name": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99, "op_params": {"shape": "(8, 64, 61)"}},
    ),
    (
        Reshape861,
        [((8, 61, 64), torch.float32)],
        {"model_name": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 8, 61, 64)"}},
    ),
    (
        Reshape854,
        [((1, 61, 8, 64), torch.float32)],
        {"model_name": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99, "op_params": {"shape": "(61, 512)"}},
    ),
    (
        Reshape862,
        [((8, 1, 61), torch.float32)],
        {"model_name": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 8, 1, 61)"}},
    ),
    (
        Reshape863,
        [((1, 8, 1, 61), torch.float32)],
        {"model_name": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99, "op_params": {"shape": "(8, 1, 61)"}},
    ),
    (
        Reshape864,
        [((61, 2816), torch.float32)],
        {
            "model_name": ["pt_t5_google_flan_t5_large_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 61, 2816)"},
        },
    ),
    (
        Reshape865,
        [((1, 2816), torch.float32)],
        {"model_name": ["pt_t5_google_flan_t5_large_text_gen_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 1, 2816)"}},
    ),
    (
        Reshape866,
        [((1, 61, 768), torch.float32)],
        {
            "model_name": ["pt_t5_t5_base_text_gen_hf", "pt_t5_google_flan_t5_base_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(61, 768)"},
        },
    ),
    (
        Reshape867,
        [((61, 768), torch.float32)],
        {
            "model_name": ["pt_t5_t5_base_text_gen_hf", "pt_t5_google_flan_t5_base_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 61, 12, 64)"},
        },
    ),
    (
        Reshape868,
        [((61, 768), torch.float32)],
        {
            "model_name": ["pt_t5_t5_base_text_gen_hf", "pt_t5_google_flan_t5_base_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 61, 768)"},
        },
    ),
    (
        Reshape869,
        [((1, 12, 61, 64), torch.float32)],
        {
            "model_name": ["pt_t5_t5_base_text_gen_hf", "pt_t5_google_flan_t5_base_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(12, 61, 64)"},
        },
    ),
    (
        Reshape870,
        [((12, 61, 61), torch.float32)],
        {
            "model_name": ["pt_t5_t5_base_text_gen_hf", "pt_t5_google_flan_t5_base_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 12, 61, 61)"},
        },
    ),
    (
        Reshape871,
        [((1, 12, 61, 61), torch.float32)],
        {
            "model_name": ["pt_t5_t5_base_text_gen_hf", "pt_t5_google_flan_t5_base_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(12, 61, 61)"},
        },
    ),
    (
        Reshape872,
        [((1, 12, 64, 61), torch.float32)],
        {
            "model_name": ["pt_t5_t5_base_text_gen_hf", "pt_t5_google_flan_t5_base_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(12, 64, 61)"},
        },
    ),
    (
        Reshape873,
        [((12, 61, 64), torch.float32)],
        {
            "model_name": ["pt_t5_t5_base_text_gen_hf", "pt_t5_google_flan_t5_base_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 12, 61, 64)"},
        },
    ),
    (
        Reshape866,
        [((1, 61, 12, 64), torch.float32)],
        {
            "model_name": ["pt_t5_t5_base_text_gen_hf", "pt_t5_google_flan_t5_base_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(61, 768)"},
        },
    ),
    (
        Reshape874,
        [((12, 1, 61), torch.float32)],
        {
            "model_name": ["pt_t5_t5_base_text_gen_hf", "pt_t5_google_flan_t5_base_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 12, 1, 61)"},
        },
    ),
    (
        Reshape875,
        [((1, 12, 1, 61), torch.float32)],
        {
            "model_name": ["pt_t5_t5_base_text_gen_hf", "pt_t5_google_flan_t5_base_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(12, 1, 61)"},
        },
    ),
    (
        Reshape876,
        [((61, 384), torch.float32)],
        {
            "model_name": ["pt_t5_google_flan_t5_small_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 61, 6, 64)"},
        },
    ),
    (
        Reshape877,
        [((1, 6, 61, 64), torch.float32)],
        {"model_name": ["pt_t5_google_flan_t5_small_text_gen_hf"], "pcc": 0.99, "op_params": {"shape": "(6, 61, 64)"}},
    ),
    (
        Reshape878,
        [((6, 61, 61), torch.float32)],
        {
            "model_name": ["pt_t5_google_flan_t5_small_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 6, 61, 61)"},
        },
    ),
    (
        Reshape879,
        [((1, 6, 61, 61), torch.float32)],
        {"model_name": ["pt_t5_google_flan_t5_small_text_gen_hf"], "pcc": 0.99, "op_params": {"shape": "(6, 61, 61)"}},
    ),
    (
        Reshape880,
        [((1, 6, 64, 61), torch.float32)],
        {"model_name": ["pt_t5_google_flan_t5_small_text_gen_hf"], "pcc": 0.99, "op_params": {"shape": "(6, 64, 61)"}},
    ),
    (
        Reshape881,
        [((6, 61, 64), torch.float32)],
        {
            "model_name": ["pt_t5_google_flan_t5_small_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 6, 61, 64)"},
        },
    ),
    (
        Reshape882,
        [((1, 61, 6, 64), torch.float32)],
        {"model_name": ["pt_t5_google_flan_t5_small_text_gen_hf"], "pcc": 0.99, "op_params": {"shape": "(61, 384)"}},
    ),
    (
        Reshape883,
        [((6, 1, 61), torch.float32)],
        {
            "model_name": ["pt_t5_google_flan_t5_small_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 6, 1, 61)"},
        },
    ),
    (
        Reshape884,
        [((1, 6, 1, 61), torch.float32)],
        {"model_name": ["pt_t5_google_flan_t5_small_text_gen_hf"], "pcc": 0.99, "op_params": {"shape": "(6, 1, 61)"}},
    ),
    (
        Reshape885,
        [((61, 2048), torch.float32)],
        {"model_name": ["pt_t5_google_flan_t5_base_text_gen_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 61, 2048)"}},
    ),
    (
        Reshape886,
        [((1, 256, 6, 6), torch.float32)],
        {
            "model_name": ["pt_alexnet_alexnet_img_cls_torchhub", "pt_rcnn_base_obj_det_torchvision_rect_0"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 9216, 1, 1)"},
        },
    ),
    (
        Reshape887,
        [((1, 256, 6, 6), torch.float32)],
        {"model_name": ["pt_alexnet_base_img_cls_osmr"], "pcc": 0.99, "op_params": {"shape": "(1, 9216)"}},
    ),
    (
        Reshape888,
        [((1, 96, 54, 54), torch.float32)],
        {"model_name": ["pt_alexnet_base_img_cls_osmr"], "pcc": 0.99, "op_params": {"shape": "(1, 1, 96, 54, 54)"}},
    ),
    (
        Reshape889,
        [((1, 96, 54, 54), torch.float32)],
        {"model_name": ["pt_alexnet_base_img_cls_osmr"], "pcc": 0.99, "op_params": {"shape": "(1, 96, 54, 54)"}},
    ),
    (
        Reshape890,
        [((1, 256, 27, 27), torch.float32)],
        {"model_name": ["pt_alexnet_base_img_cls_osmr"], "pcc": 0.99, "op_params": {"shape": "(1, 1, 256, 27, 27)"}},
    ),
    (
        Reshape891,
        [((1, 256, 27, 27), torch.float32)],
        {"model_name": ["pt_alexnet_base_img_cls_osmr"], "pcc": 0.99, "op_params": {"shape": "(1, 256, 27, 27)"}},
    ),
    (
        Reshape892,
        [((1, 1024, 14, 14), torch.float32)],
        {
            "model_name": [
                "pt_beit_microsoft_beit_large_patch16_224_img_cls_hf",
                "pt_mlp_mixer_mixer_l16_224_img_cls_timm",
                "pt_mlp_mixer_mixer_l16_224_in21k_img_cls_timm",
                "pt_vit_google_vit_large_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1024, 196, 1)"},
        },
    ),
    (
        Reshape893,
        [((1, 197, 1024), torch.float32)],
        {
            "model_name": [
                "pt_beit_microsoft_beit_large_patch16_224_img_cls_hf",
                "pt_vit_google_vit_large_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(197, 1024)"},
        },
    ),
    (
        Reshape894,
        [((1, 197, 1024), torch.float32)],
        {
            "model_name": [
                "pt_beit_microsoft_beit_large_patch16_224_img_cls_hf",
                "pt_vit_google_vit_large_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 197, 16, 64)"},
        },
    ),
    (
        Reshape895,
        [((197, 1024), torch.float32)],
        {
            "model_name": [
                "pt_beit_microsoft_beit_large_patch16_224_img_cls_hf",
                "pt_vit_google_vit_large_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 197, 1024)"},
        },
    ),
    (
        Reshape894,
        [((197, 1024), torch.float32)],
        {
            "model_name": ["pt_beit_microsoft_beit_large_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 197, 16, 64)"},
        },
    ),
    (
        Reshape896,
        [((1, 16, 197, 64), torch.float32)],
        {
            "model_name": [
                "pt_beit_microsoft_beit_large_patch16_224_img_cls_hf",
                "pt_vit_google_vit_large_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(16, 197, 64)"},
        },
    ),
    (
        Reshape897,
        [((16, 197, 197), torch.float32)],
        {
            "model_name": [
                "pt_beit_microsoft_beit_large_patch16_224_img_cls_hf",
                "pt_vit_google_vit_large_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 16, 197, 197)"},
        },
    ),
    (
        Reshape898,
        [((729, 16), torch.float32)],
        {
            "model_name": ["pt_beit_microsoft_beit_large_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 27, 27, 16)"},
        },
    ),
    (
        Reshape899,
        [((1, 27, 27, 16), torch.float32)],
        {
            "model_name": ["pt_beit_microsoft_beit_large_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(729, 16)"},
        },
    ),
    (
        Reshape900,
        [((38809, 16), torch.float32)],
        {
            "model_name": ["pt_beit_microsoft_beit_large_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(197, 197, 16)"},
        },
    ),
    (
        Reshape901,
        [((1, 16, 197, 197), torch.float32)],
        {
            "model_name": [
                "pt_beit_microsoft_beit_large_patch16_224_img_cls_hf",
                "pt_vit_google_vit_large_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(16, 197, 197)"},
        },
    ),
    (
        Reshape902,
        [((1, 16, 64, 197), torch.float32)],
        {
            "model_name": [
                "pt_beit_microsoft_beit_large_patch16_224_img_cls_hf",
                "pt_vit_google_vit_large_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(16, 64, 197)"},
        },
    ),
    (
        Reshape903,
        [((16, 197, 64), torch.float32)],
        {
            "model_name": [
                "pt_beit_microsoft_beit_large_patch16_224_img_cls_hf",
                "pt_vit_google_vit_large_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 16, 197, 64)"},
        },
    ),
    (
        Reshape893,
        [((1, 197, 16, 64), torch.float32)],
        {
            "model_name": [
                "pt_beit_microsoft_beit_large_patch16_224_img_cls_hf",
                "pt_vit_google_vit_large_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(197, 1024)"},
        },
    ),
    (
        Reshape904,
        [((1, 768, 14, 14), torch.float32)],
        {
            "model_name": [
                "pt_beit_microsoft_beit_base_patch16_224_img_cls_hf",
                "pt_deit_facebook_deit_base_patch16_224_img_cls_hf",
                "pt_deit_facebook_deit_base_distilled_patch16_224_img_cls_hf",
                "pt_mlp_mixer_mixer_b16_224_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_miil_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_goog_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_miil_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_img_cls_timm",
                "pt_vit_google_vit_base_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 768, 196, 1)"},
        },
    ),
    (
        Reshape905,
        [((1, 197, 768), torch.float32)],
        {
            "model_name": [
                "pt_beit_microsoft_beit_base_patch16_224_img_cls_hf",
                "pt_deit_facebook_deit_base_patch16_224_img_cls_hf",
                "pt_deit_facebook_deit_base_distilled_patch16_224_img_cls_hf",
                "pt_vit_google_vit_base_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(197, 768)"},
        },
    ),
    (
        Reshape906,
        [((1, 197, 768), torch.float32)],
        {
            "model_name": [
                "pt_beit_microsoft_beit_base_patch16_224_img_cls_hf",
                "pt_deit_facebook_deit_base_patch16_224_img_cls_hf",
                "pt_deit_facebook_deit_base_distilled_patch16_224_img_cls_hf",
                "pt_vit_google_vit_base_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 197, 12, 64)"},
        },
    ),
    (
        Reshape907,
        [((197, 768), torch.float32)],
        {
            "model_name": [
                "pt_beit_microsoft_beit_base_patch16_224_img_cls_hf",
                "pt_deit_facebook_deit_base_patch16_224_img_cls_hf",
                "pt_deit_facebook_deit_base_distilled_patch16_224_img_cls_hf",
                "pt_vit_google_vit_base_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 197, 768)"},
        },
    ),
    (
        Reshape906,
        [((197, 768), torch.float32)],
        {
            "model_name": ["pt_beit_microsoft_beit_base_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 197, 12, 64)"},
        },
    ),
    (
        Reshape908,
        [((1, 12, 197, 64), torch.float32)],
        {
            "model_name": [
                "pt_beit_microsoft_beit_base_patch16_224_img_cls_hf",
                "pt_deit_facebook_deit_base_patch16_224_img_cls_hf",
                "pt_deit_facebook_deit_base_distilled_patch16_224_img_cls_hf",
                "pt_vit_google_vit_base_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(12, 197, 64)"},
        },
    ),
    (
        Reshape909,
        [((12, 197, 197), torch.float32)],
        {
            "model_name": [
                "pt_beit_microsoft_beit_base_patch16_224_img_cls_hf",
                "pt_deit_facebook_deit_base_patch16_224_img_cls_hf",
                "pt_deit_facebook_deit_base_distilled_patch16_224_img_cls_hf",
                "pt_vit_google_vit_base_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 12, 197, 197)"},
        },
    ),
    (
        Reshape910,
        [((729, 12), torch.float32)],
        {
            "model_name": ["pt_beit_microsoft_beit_base_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 27, 27, 12)"},
        },
    ),
    (
        Reshape911,
        [((1, 27, 27, 12), torch.float32)],
        {
            "model_name": ["pt_beit_microsoft_beit_base_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(729, 12)"},
        },
    ),
    (
        Reshape912,
        [((38809, 12), torch.float32)],
        {
            "model_name": ["pt_beit_microsoft_beit_base_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(197, 197, 12)"},
        },
    ),
    (
        Reshape913,
        [((1, 12, 197, 197), torch.float32)],
        {
            "model_name": [
                "pt_beit_microsoft_beit_base_patch16_224_img_cls_hf",
                "pt_deit_facebook_deit_base_patch16_224_img_cls_hf",
                "pt_deit_facebook_deit_base_distilled_patch16_224_img_cls_hf",
                "pt_vit_google_vit_base_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(12, 197, 197)"},
        },
    ),
    (
        Reshape914,
        [((1, 12, 64, 197), torch.float32)],
        {
            "model_name": [
                "pt_beit_microsoft_beit_base_patch16_224_img_cls_hf",
                "pt_deit_facebook_deit_base_patch16_224_img_cls_hf",
                "pt_deit_facebook_deit_base_distilled_patch16_224_img_cls_hf",
                "pt_vit_google_vit_base_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(12, 64, 197)"},
        },
    ),
    (
        Reshape915,
        [((12, 197, 64), torch.float32)],
        {
            "model_name": [
                "pt_beit_microsoft_beit_base_patch16_224_img_cls_hf",
                "pt_deit_facebook_deit_base_patch16_224_img_cls_hf",
                "pt_deit_facebook_deit_base_distilled_patch16_224_img_cls_hf",
                "pt_vit_google_vit_base_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 12, 197, 64)"},
        },
    ),
    (
        Reshape905,
        [((1, 197, 12, 64), torch.float32)],
        {
            "model_name": [
                "pt_beit_microsoft_beit_base_patch16_224_img_cls_hf",
                "pt_deit_facebook_deit_base_patch16_224_img_cls_hf",
                "pt_deit_facebook_deit_base_distilled_patch16_224_img_cls_hf",
                "pt_vit_google_vit_base_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(197, 768)"},
        },
    ),
    (
        Reshape916,
        [((1, 192, 14, 14), torch.float32)],
        {
            "model_name": ["pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 192, 196, 1)"},
        },
    ),
    (
        Reshape917,
        [((1, 197, 192), torch.float32)],
        {
            "model_name": ["pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(197, 192)"},
        },
    ),
    (
        Reshape918,
        [((1, 197, 192), torch.float32)],
        {
            "model_name": ["pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 197, 3, 64)"},
        },
    ),
    (
        Reshape919,
        [((197, 192), torch.float32)],
        {
            "model_name": ["pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 197, 192)"},
        },
    ),
    (
        Reshape920,
        [((1, 3, 197, 64), torch.float32)],
        {
            "model_name": ["pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(3, 197, 64)"},
        },
    ),
    (
        Reshape921,
        [((3, 197, 197), torch.float32)],
        {
            "model_name": ["pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 3, 197, 197)"},
        },
    ),
    (
        Reshape922,
        [((1, 3, 197, 197), torch.float32)],
        {
            "model_name": ["pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(3, 197, 197)"},
        },
    ),
    (
        Reshape923,
        [((1, 3, 64, 197), torch.float32)],
        {
            "model_name": ["pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(3, 64, 197)"},
        },
    ),
    (
        Reshape924,
        [((3, 197, 64), torch.float32)],
        {
            "model_name": ["pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 3, 197, 64)"},
        },
    ),
    (
        Reshape917,
        [((1, 197, 3, 64), torch.float32)],
        {
            "model_name": ["pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(197, 192)"},
        },
    ),
    (
        Reshape925,
        [((1, 1, 192), torch.float32)],
        {
            "model_name": ["pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 192)"},
        },
    ),
    (
        Reshape926,
        [((1, 384, 14, 14), torch.float32)],
        {
            "model_name": ["pt_deit_facebook_deit_small_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 384, 196, 1)"},
        },
    ),
    (
        Reshape927,
        [((1, 197, 384), torch.float32)],
        {
            "model_name": ["pt_deit_facebook_deit_small_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(197, 384)"},
        },
    ),
    (
        Reshape928,
        [((1, 197, 384), torch.float32)],
        {
            "model_name": ["pt_deit_facebook_deit_small_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 197, 6, 64)"},
        },
    ),
    (
        Reshape929,
        [((197, 384), torch.float32)],
        {
            "model_name": ["pt_deit_facebook_deit_small_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 197, 384)"},
        },
    ),
    (
        Reshape930,
        [((1, 6, 197, 64), torch.float32)],
        {
            "model_name": ["pt_deit_facebook_deit_small_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(6, 197, 64)"},
        },
    ),
    (
        Reshape931,
        [((6, 197, 197), torch.float32)],
        {
            "model_name": ["pt_deit_facebook_deit_small_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 6, 197, 197)"},
        },
    ),
    (
        Reshape932,
        [((1, 6, 197, 197), torch.float32)],
        {
            "model_name": ["pt_deit_facebook_deit_small_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(6, 197, 197)"},
        },
    ),
    (
        Reshape933,
        [((1, 6, 64, 197), torch.float32)],
        {
            "model_name": ["pt_deit_facebook_deit_small_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(6, 64, 197)"},
        },
    ),
    (
        Reshape934,
        [((6, 197, 64), torch.float32)],
        {
            "model_name": ["pt_deit_facebook_deit_small_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 6, 197, 64)"},
        },
    ),
    (
        Reshape927,
        [((1, 197, 6, 64), torch.float32)],
        {
            "model_name": ["pt_deit_facebook_deit_small_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(197, 384)"},
        },
    ),
    (
        Reshape935,
        [((1, 1920, 1, 1), torch.float32)],
        {
            "model_name": ["pt_densenet_densenet201_img_cls_torchvision", "pt_regnet_regnet_x_8gf_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1920, 1, 1)"},
        },
    ),
    (
        Reshape936,
        [((1, 1024, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_googlenet_base_img_cls_torchvision",
                "pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf",
                "pt_mobilenet_v1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_swin_swin_b_img_cls_torchvision",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_vovnet_ese_vovnet39b_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1024, 1, 1)"},
        },
    ),
    (
        Reshape29,
        [((1, 1024, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_mobilenet_v1_basic_img_cls_torchvision",
                "pt_vovnet_vovnet_v1_57_obj_det_torchhub",
                "pt_vovnet_vovnet57_obj_det_osmr",
                "pt_vovnet_vovnet39_obj_det_osmr",
                "pt_vovnet_v1_vovnet39_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1024)"},
        },
    ),
    (
        Reshape937,
        [((1, 2208, 1, 1), torch.float32)],
        {
            "model_name": ["pt_densenet_densenet161_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 2208, 1, 1)"},
        },
    ),
    (
        Reshape938,
        [((1, 1664, 1, 1), torch.float32)],
        {
            "model_name": ["pt_densenet_densenet169_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1664, 1, 1)"},
        },
    ),
    (
        Reshape602,
        [((1, 1000, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_dla_dla60x_visual_bb_torchvision",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_dla_dla46x_c_visual_bb_torchvision",
                "pt_dla_dla60x_c_visual_bb_torchvision",
                "pt_dla_dla102x_visual_bb_torchvision",
                "pt_dla_dla46_c_visual_bb_torchvision",
                "pt_dla_dla169_visual_bb_torchvision",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_dla_dla34_visual_bb_torchvision",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1000)"},
        },
    ),
    (
        Reshape939,
        [((1, 1000, 1, 1), torch.float32)],
        {"model_name": ["pt_dla_dla34_in1k_img_cls_timm"], "pcc": 0.99, "op_params": {"shape": "(1, 1000, 1, 1)"}},
    ),
    (
        Reshape940,
        [((32, 1, 3, 3), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm",
                "pt_mobilenet_v1_basic_img_cls_torchvision",
                "pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf",
                "pt_mobilenet_v1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(32, 1, 3, 3)"},
        },
    ),
    (
        Reshape941,
        [((96, 1, 3, 3), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm",
                "pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(96, 1, 3, 3)"},
        },
    ),
    (
        Reshape942,
        [((144, 1, 3, 3), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(144, 1, 3, 3)"},
        },
    ),
    (
        Reshape943,
        [((144, 1, 5, 5), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(144, 1, 5, 5)"},
        },
    ),
    (
        Reshape944,
        [((240, 1, 5, 5), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(240, 1, 5, 5)"},
        },
    ),
    (
        Reshape945,
        [((240, 1, 3, 3), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(240, 1, 3, 3)"},
        },
    ),
    (
        Reshape946,
        [((480, 1, 3, 3), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(480, 1, 3, 3)"},
        },
    ),
    (
        Reshape947,
        [((480, 1, 5, 5), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(480, 1, 5, 5)"},
        },
    ),
    (
        Reshape948,
        [((672, 1, 5, 5), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(672, 1, 5, 5)"},
        },
    ),
    (
        Reshape949,
        [((1152, 1, 5, 5), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1152, 1, 5, 5)"},
        },
    ),
    (
        Reshape950,
        [((1152, 1, 3, 3), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1152, 1, 3, 3)"},
        },
    ),
    (
        Reshape951,
        [((1, 1280, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1280, 1, 1)"},
        },
    ),
    (
        Reshape952,
        [((48, 1, 3, 3), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf",
                "pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(48, 1, 3, 3)"},
        },
    ),
    (
        Reshape953,
        [((24, 1, 3, 3), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(24, 1, 3, 3)"},
        },
    ),
    (
        Reshape954,
        [((192, 1, 3, 3), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm",
                "pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(192, 1, 3, 3)"},
        },
    ),
    (
        Reshape955,
        [((192, 1, 5, 5), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(192, 1, 5, 5)"},
        },
    ),
    (
        Reshape956,
        [((336, 1, 5, 5), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(336, 1, 5, 5)"},
        },
    ),
    (
        Reshape957,
        [((336, 1, 3, 3), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(336, 1, 3, 3)"},
        },
    ),
    (
        Reshape958,
        [((672, 1, 3, 3), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(672, 1, 3, 3)"},
        },
    ),
    (
        Reshape959,
        [((960, 1, 5, 5), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(960, 1, 5, 5)"},
        },
    ),
    (
        Reshape960,
        [((1632, 1, 5, 5), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1632, 1, 5, 5)"},
        },
    ),
    (
        Reshape961,
        [((1632, 1, 3, 3), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1632, 1, 3, 3)"},
        },
    ),
    (
        Reshape962,
        [((2688, 1, 3, 3), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(2688, 1, 3, 3)"},
        },
    ),
    (
        Reshape963,
        [((1, 1792, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1792, 1, 1)"},
        },
    ),
    (
        Reshape964,
        [((288, 1, 5, 5), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(288, 1, 5, 5)"},
        },
    ),
    (
        Reshape965,
        [((288, 1, 3, 3), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(288, 1, 3, 3)"},
        },
    ),
    (
        Reshape966,
        [((576, 1, 3, 3), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(576, 1, 3, 3)"},
        },
    ),
    (
        Reshape967,
        [((576, 1, 5, 5), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(576, 1, 5, 5)"},
        },
    ),
    (
        Reshape968,
        [((816, 1, 5, 5), torch.float32)],
        {
            "model_name": ["pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {"shape": "(816, 1, 5, 5)"},
        },
    ),
    (
        Reshape969,
        [((1392, 1, 5, 5), torch.float32)],
        {
            "model_name": ["pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {"shape": "(1392, 1, 5, 5)"},
        },
    ),
    (
        Reshape970,
        [((1392, 1, 3, 3), torch.float32)],
        {
            "model_name": ["pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {"shape": "(1392, 1, 3, 3)"},
        },
    ),
    (
        Reshape971,
        [((528, 1, 3, 3), torch.float32)],
        {
            "model_name": ["pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {"shape": "(528, 1, 3, 3)"},
        },
    ),
    (
        Reshape972,
        [((528, 1, 5, 5), torch.float32)],
        {
            "model_name": ["pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {"shape": "(528, 1, 5, 5)"},
        },
    ),
    (
        Reshape973,
        [((720, 1, 5, 5), torch.float32)],
        {
            "model_name": ["pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {"shape": "(720, 1, 5, 5)"},
        },
    ),
    (
        Reshape974,
        [((1248, 1, 5, 5), torch.float32)],
        {
            "model_name": ["pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {"shape": "(1248, 1, 5, 5)"},
        },
    ),
    (
        Reshape975,
        [((1248, 1, 3, 3), torch.float32)],
        {
            "model_name": ["pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {"shape": "(1248, 1, 3, 3)"},
        },
    ),
    (
        Reshape976,
        [((8, 1, 3, 3), torch.float32)],
        {
            "model_name": [
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(8, 1, 3, 3)"},
        },
    ),
    (
        Reshape977,
        [((12, 1, 3, 3), torch.float32)],
        {
            "model_name": [
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(12, 1, 3, 3)"},
        },
    ),
    (
        Reshape978,
        [((16, 1, 3, 3), torch.float32)],
        {
            "model_name": [
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(16, 1, 3, 3)"},
        },
    ),
    (
        Reshape979,
        [((36, 1, 3, 3), torch.float32)],
        {
            "model_name": [
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(36, 1, 3, 3)"},
        },
    ),
    (
        Reshape980,
        [((72, 1, 5, 5), torch.float32)],
        {
            "model_name": [
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(72, 1, 5, 5)"},
        },
    ),
    (
        Reshape981,
        [((20, 1, 3, 3), torch.float32)],
        {
            "model_name": [
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(20, 1, 3, 3)"},
        },
    ),
    (
        Reshape982,
        [((24, 1, 5, 5), torch.float32)],
        {
            "model_name": [
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(24, 1, 5, 5)"},
        },
    ),
    (
        Reshape983,
        [((60, 1, 3, 3), torch.float32)],
        {
            "model_name": [
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(60, 1, 3, 3)"},
        },
    ),
    (
        Reshape984,
        [((120, 1, 3, 3), torch.float32)],
        {
            "model_name": [
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(120, 1, 3, 3)"},
        },
    ),
    (
        Reshape985,
        [((40, 1, 3, 3), torch.float32)],
        {
            "model_name": [
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(40, 1, 3, 3)"},
        },
    ),
    (
        Reshape986,
        [((100, 1, 3, 3), torch.float32)],
        {
            "model_name": [
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(100, 1, 3, 3)"},
        },
    ),
    (
        Reshape987,
        [((92, 1, 3, 3), torch.float32)],
        {
            "model_name": [
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(92, 1, 3, 3)"},
        },
    ),
    (
        Reshape988,
        [((56, 1, 3, 3), torch.float32)],
        {
            "model_name": [
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(56, 1, 3, 3)"},
        },
    ),
    (
        Reshape989,
        [((80, 1, 3, 3), torch.float32)],
        {
            "model_name": [
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(80, 1, 3, 3)"},
        },
    ),
    (
        Reshape990,
        [((112, 1, 5, 5), torch.float32)],
        {
            "model_name": [
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(112, 1, 5, 5)"},
        },
    ),
    (
        Reshape991,
        [((72, 1, 1, 5), torch.float32)],
        {
            "model_name": ["pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {"shape": "(72, 1, 1, 5)"},
        },
    ),
    (
        Reshape992,
        [((72, 1, 5, 1), torch.float32)],
        {
            "model_name": ["pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {"shape": "(72, 1, 5, 1)"},
        },
    ),
    (
        Reshape993,
        [((120, 1, 1, 5), torch.float32)],
        {
            "model_name": ["pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {"shape": "(120, 1, 1, 5)"},
        },
    ),
    (
        Reshape994,
        [((120, 1, 5, 1), torch.float32)],
        {
            "model_name": ["pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {"shape": "(120, 1, 5, 1)"},
        },
    ),
    (
        Reshape995,
        [((240, 1, 1, 5), torch.float32)],
        {
            "model_name": ["pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {"shape": "(240, 1, 1, 5)"},
        },
    ),
    (
        Reshape996,
        [((240, 1, 5, 1), torch.float32)],
        {
            "model_name": ["pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {"shape": "(240, 1, 5, 1)"},
        },
    ),
    (
        Reshape997,
        [((200, 1, 1, 5), torch.float32)],
        {
            "model_name": ["pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {"shape": "(200, 1, 1, 5)"},
        },
    ),
    (
        Reshape998,
        [((200, 1, 5, 1), torch.float32)],
        {
            "model_name": ["pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {"shape": "(200, 1, 5, 1)"},
        },
    ),
    (
        Reshape999,
        [((184, 1, 1, 5), torch.float32)],
        {
            "model_name": ["pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {"shape": "(184, 1, 1, 5)"},
        },
    ),
    (
        Reshape1000,
        [((184, 1, 5, 1), torch.float32)],
        {
            "model_name": ["pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {"shape": "(184, 1, 5, 1)"},
        },
    ),
    (
        Reshape1001,
        [((480, 1, 1, 5), torch.float32)],
        {
            "model_name": ["pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {"shape": "(480, 1, 1, 5)"},
        },
    ),
    (
        Reshape1002,
        [((480, 1, 5, 1), torch.float32)],
        {
            "model_name": ["pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {"shape": "(480, 1, 5, 1)"},
        },
    ),
    (
        Reshape1003,
        [((672, 1, 1, 5), torch.float32)],
        {
            "model_name": ["pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {"shape": "(672, 1, 1, 5)"},
        },
    ),
    (
        Reshape1004,
        [((672, 1, 5, 1), torch.float32)],
        {
            "model_name": ["pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {"shape": "(672, 1, 5, 1)"},
        },
    ),
    (
        Reshape1005,
        [((960, 1, 1, 5), torch.float32)],
        {
            "model_name": ["pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {"shape": "(960, 1, 1, 5)"},
        },
    ),
    (
        Reshape1006,
        [((960, 1, 5, 1), torch.float32)],
        {
            "model_name": ["pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {"shape": "(960, 1, 5, 1)"},
        },
    ),
    (
        Reshape1007,
        [((1, 64, 120, 160), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 64, 19200, 1)"},
        },
    ),
    (
        Reshape1008,
        [((1, 19200, 64), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 19200, 1, 64)"},
        },
    ),
    (
        Reshape1009,
        [((1, 19200, 64), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 120, 160, 64)"},
        },
    ),
    (
        Reshape1010,
        [((1, 64, 19200), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 64, 120, 160)"},
        },
    ),
    (
        Reshape1011,
        [((1, 64, 15, 20), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 64, 300)"},
        },
    ),
    (
        Reshape1012,
        [((1, 300, 64), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(300, 64)"},
        },
    ),
    (
        Reshape1013,
        [((1, 300, 64), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 300, 1, 64)"},
        },
    ),
    (
        Reshape1014,
        [((300, 64), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 300, 64)"},
        },
    ),
    (
        Reshape1015,
        [((1, 19200, 300), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1, 19200, 300)"},
        },
    ),
    (
        Reshape1016,
        [((1, 1, 19200, 300), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 19200, 300)"},
        },
    ),
    (
        Reshape1011,
        [((1, 1, 64, 300), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 64, 300)"},
        },
    ),
    (
        Reshape1017,
        [((1, 256, 19200), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 256, 120, 160)"},
        },
    ),
    (
        Reshape1018,
        [((256, 1, 3, 3), torch.float32)],
        {
            "model_name": [
                "pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf",
                "pt_mobilenet_v1_basic_img_cls_torchvision",
                "pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf",
                "pt_mobilenet_v1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_xception_xception71_tf_in1k_img_cls_timm",
                "pt_xception_xception65_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
                "pt_xception_xception_img_cls_timm",
                "pt_xception_xception41_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(256, 1, 3, 3)"},
        },
    ),
    (
        Reshape1019,
        [((1, 256, 120, 160), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 256, 19200, 1)"},
        },
    ),
    (
        Reshape1020,
        [((1, 128, 60, 80), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 128, 4800, 1)"},
        },
    ),
    (
        Reshape1021,
        [((1, 4800, 128), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 4800, 2, 64)"},
        },
    ),
    (
        Reshape1022,
        [((1, 4800, 128), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 60, 80, 128)"},
        },
    ),
    (
        Reshape1023,
        [((1, 2, 4800, 64), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(2, 4800, 64)"},
        },
    ),
    (
        Reshape1024,
        [((1, 128, 4800), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 128, 60, 80)"},
        },
    ),
    (
        Reshape1025,
        [((1, 128, 15, 20), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 128, 300)"},
        },
    ),
    (
        Reshape1026,
        [((1, 300, 128), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(300, 128)"},
        },
    ),
    (
        Reshape1027,
        [((1, 300, 128), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 300, 2, 64)"},
        },
    ),
    (
        Reshape1028,
        [((300, 128), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 300, 128)"},
        },
    ),
    (
        Reshape1029,
        [((1, 2, 300, 64), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(2, 300, 64)"},
        },
    ),
    (
        Reshape1030,
        [((2, 4800, 300), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 2, 4800, 300)"},
        },
    ),
    (
        Reshape1031,
        [((1, 2, 4800, 300), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(2, 4800, 300)"},
        },
    ),
    (
        Reshape1032,
        [((1, 2, 64, 300), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(2, 64, 300)"},
        },
    ),
    (
        Reshape1033,
        [((2, 4800, 64), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 2, 4800, 64)"},
        },
    ),
    (
        Reshape1034,
        [((1, 4800, 2, 64), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(4800, 128)"},
        },
    ),
    (
        Reshape1035,
        [((4800, 128), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 4800, 128)"},
        },
    ),
    (
        Reshape1036,
        [((1, 512, 4800), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 512, 60, 80)"},
        },
    ),
    (
        Reshape1037,
        [((512, 1, 3, 3), torch.float32)],
        {
            "model_name": [
                "pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf",
                "pt_mobilenet_v1_basic_img_cls_torchvision",
                "pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf",
                "pt_mobilenet_v1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(512, 1, 3, 3)"},
        },
    ),
    (
        Reshape1038,
        [((1, 512, 60, 80), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 512, 4800, 1)"},
        },
    ),
    (
        Reshape1039,
        [((1, 320, 30, 40), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 320, 1200, 1)"},
        },
    ),
    (
        Reshape1040,
        [((1, 1200, 320), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1200, 5, 64)"},
        },
    ),
    (
        Reshape1041,
        [((1, 1200, 320), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 30, 40, 320)"},
        },
    ),
    (
        Reshape1042,
        [((1, 5, 1200, 64), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(5, 1200, 64)"},
        },
    ),
    (
        Reshape1043,
        [((1, 320, 1200), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 320, 30, 40)"},
        },
    ),
    (
        Reshape1044,
        [((1, 320, 15, 20), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 320, 300)"},
        },
    ),
    (
        Reshape1045,
        [((1, 300, 320), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(300, 320)"},
        },
    ),
    (
        Reshape1046,
        [((1, 300, 320), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 300, 5, 64)"},
        },
    ),
    (
        Reshape1047,
        [((300, 320), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 300, 320)"},
        },
    ),
    (
        Reshape1048,
        [((1, 5, 300, 64), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(5, 300, 64)"},
        },
    ),
    (
        Reshape1049,
        [((5, 1200, 300), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 5, 1200, 300)"},
        },
    ),
    (
        Reshape1050,
        [((1, 5, 1200, 300), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(5, 1200, 300)"},
        },
    ),
    (
        Reshape1051,
        [((1, 5, 64, 300), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(5, 64, 300)"},
        },
    ),
    (
        Reshape1052,
        [((5, 1200, 64), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 5, 1200, 64)"},
        },
    ),
    (
        Reshape1053,
        [((1, 1200, 5, 64), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1200, 320)"},
        },
    ),
    (
        Reshape1054,
        [((1200, 320), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1200, 320)"},
        },
    ),
    (
        Reshape1055,
        [((1, 1280, 1200), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1280, 30, 40)"},
        },
    ),
    (
        Reshape1056,
        [((1280, 1, 3, 3), torch.float32)],
        {
            "model_name": [
                "pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1280, 1, 3, 3)"},
        },
    ),
    (
        Reshape1057,
        [((1, 1280, 30, 40), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1280, 1200, 1)"},
        },
    ),
    (
        Reshape1058,
        [((1, 512, 15, 20), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 512, 300, 1)"},
        },
    ),
    (
        Reshape1059,
        [((1, 300, 512), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(300, 512)"},
        },
    ),
    (
        Reshape1060,
        [((1, 300, 512), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 300, 8, 64)"},
        },
    ),
    (
        Reshape1061,
        [((1, 300, 512), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 15, 20, 512)"},
        },
    ),
    (
        Reshape1062,
        [((300, 512), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 300, 512)"},
        },
    ),
    (
        Reshape1063,
        [((1, 8, 300, 64), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(8, 300, 64)"},
        },
    ),
    (
        Reshape1064,
        [((8, 300, 300), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 8, 300, 300)"},
        },
    ),
    (
        Reshape1065,
        [((1, 8, 300, 300), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(8, 300, 300)"},
        },
    ),
    (
        Reshape1066,
        [((1, 8, 64, 300), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(8, 64, 300)"},
        },
    ),
    (
        Reshape1067,
        [((8, 300, 64), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 8, 300, 64)"},
        },
    ),
    (
        Reshape1059,
        [((1, 300, 8, 64), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(300, 512)"},
        },
    ),
    (
        Reshape1068,
        [((1, 2048, 300), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 2048, 15, 20)"},
        },
    ),
    (
        Reshape1069,
        [((2048, 1, 3, 3), torch.float32)],
        {
            "model_name": [
                "pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(2048, 1, 3, 3)"},
        },
    ),
    (
        Reshape1070,
        [((1, 2048, 15, 20), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 2048, 300, 1)"},
        },
    ),
    (
        Reshape1071,
        [((1, 1, 30, 40), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 30, 40)"},
        },
    ),
    (
        Reshape1072,
        [((1, 1, 60, 80), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 60, 80)"},
        },
    ),
    (
        Reshape1073,
        [((1, 1, 120, 160), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 120, 160)"},
        },
    ),
    (
        Reshape1074,
        [((1, 1, 224, 224), torch.float32)],
        {"model_name": ["pt_googlenet_base_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(1, 224, 224)"}},
    ),
    (
        Reshape1075,
        [((1, 2048, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_hrnet_hrnet_w48_pose_estimation_timm",
                "pt_hrnet_hrnet_w40_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_pose_estimation_timm",
                "pt_hrnet_hrnet_w64_pose_estimation_timm",
                "pt_hrnet_hrnet_w32_pose_estimation_timm",
                "pt_hrnet_hrnet_w44_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_mobilenetv3_ssd_resnet101_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet152_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet50_img_cls_torchvision",
                "pt_regnet_regnet_x_16gf_img_cls_torchvision",
                "pt_resnet_resnet50_img_cls_torchvision",
                "pt_resnet_resnet101_img_cls_torchvision",
                "pt_resnet_50_img_cls_timm",
                "pt_resnet_resnet152_img_cls_torchvision",
                "ResNetForImageClassification",
                "pt_resnext_resnext101_32x8d_img_cls_torchhub",
                "pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub",
                "pt_resnext_resnext50_32x4d_img_cls_torchhub",
                "pt_wideresnet_wide_resnet50_2_img_cls_torchvision",
                "pt_wideresnet_wide_resnet50_2_img_cls_timm",
                "pt_wideresnet_wide_resnet101_2_img_cls_torchvision",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
                "pt_xception_xception71_tf_in1k_img_cls_timm",
                "pt_xception_xception65_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
                "pt_xception_xception_img_cls_timm",
                "pt_xception_xception41_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 2048, 1, 1)"},
        },
    ),
    (
        Reshape2,
        [((1, 2048, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_hrnet_hrnetv2_w40_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w44_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w32_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w30_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w48_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
                "pt_resnext_resnext101_64x4d_img_cls_osmr",
                "pt_resnext_resnext50_32x4d_img_cls_osmr",
                "pt_resnext_resnext14_32x4d_img_cls_osmr",
                "pt_resnext_resnext26_32x4d_img_cls_osmr",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 2048)"},
        },
    ),
    (
        Reshape1076,
        [((1, 1536, 1, 1), torch.float32)],
        {
            "model_name": ["pt_inception_inception_v4_tf_in1k_img_cls_timm", "pt_inception_inception_v4_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1536, 1, 1)"},
        },
    ),
    (
        Reshape45,
        [((1, 1536, 1, 1), torch.float32)],
        {"model_name": ["pt_inception_v4_img_cls_osmr"], "pcc": 0.99, "op_params": {"shape": "(1, 1536)"}},
    ),
    (
        Reshape1077,
        [((1, 3, 256, 256), torch.float32)],
        {
            "model_name": ["pt_mlp_mixer_base_img_cls_github"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 3, 16, 16, 16, 16)"},
        },
    ),
    (
        Reshape445,
        [((1, 16, 16, 16, 16, 3), torch.float32)],
        {"model_name": ["pt_mlp_mixer_base_img_cls_github"], "pcc": 0.99, "op_params": {"shape": "(256, 768)"}},
    ),
    (
        Reshape1078,
        [((1, 256, 512), torch.float32)],
        {"model_name": ["pt_mlp_mixer_base_img_cls_github"], "pcc": 0.99, "op_params": {"shape": "(1, 256, 512, 1)"}},
    ),
    (
        Reshape546,
        [((1, 256, 512), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(256, 512)"},
        },
    ),
    (
        Reshape545,
        [((1, 256, 512), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 256, 8, 64)"},
        },
    ),
    (
        Reshape547,
        [((1, 256, 512), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 256, 512)"},
        },
    ),
    (
        Reshape1079,
        [((1, 256, 512), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 16, 16, 512)"},
        },
    ),
    (
        Reshape1080,
        [((1024, 256, 1), torch.float32)],
        {"model_name": ["pt_mlp_mixer_base_img_cls_github"], "pcc": 0.99, "op_params": {"shape": "(1024, 256, 1, 1)"}},
    ),
    (
        Reshape601,
        [((1, 1024, 512, 1), torch.float32)],
        {"model_name": ["pt_mlp_mixer_base_img_cls_github"], "pcc": 0.99, "op_params": {"shape": "(1, 1024, 512)"}},
    ),
    (
        Reshape1081,
        [((1, 1024, 512), torch.float32)],
        {"model_name": ["pt_mlp_mixer_base_img_cls_github"], "pcc": 0.99, "op_params": {"shape": "(1, 1024, 512, 1)"}},
    ),
    (
        Reshape1082,
        [((256, 1024, 1), torch.float32)],
        {"model_name": ["pt_mlp_mixer_base_img_cls_github"], "pcc": 0.99, "op_params": {"shape": "(256, 1024, 1, 1)"}},
    ),
    (
        Reshape547,
        [((1, 256, 512, 1), torch.float32)],
        {"model_name": ["pt_mlp_mixer_base_img_cls_github"], "pcc": 0.99, "op_params": {"shape": "(1, 256, 512)"}},
    ),
    (
        Reshape886,
        [((1, 64, 12, 12), torch.float32)],
        {"model_name": ["pt_mnist_base_img_cls_github"], "pcc": 0.99, "op_params": {"shape": "(1, 9216, 1, 1)"}},
    ),
    (
        Reshape1083,
        [((64, 1, 3, 3), torch.float32)],
        {
            "model_name": [
                "pt_mobilenet_v1_basic_img_cls_torchvision",
                "pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf",
                "pt_mobilenet_v1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_xception_xception71_tf_in1k_img_cls_timm",
                "pt_xception_xception65_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
                "pt_xception_xception_img_cls_timm",
                "pt_xception_xception41_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(64, 1, 3, 3)"},
        },
    ),
    (
        Reshape1084,
        [((128, 1, 3, 3), torch.float32)],
        {
            "model_name": [
                "pt_mobilenet_v1_basic_img_cls_torchvision",
                "pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf",
                "pt_mobilenet_v1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_xception_xception71_tf_in1k_img_cls_timm",
                "pt_xception_xception65_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
                "pt_xception_xception_img_cls_timm",
                "pt_xception_xception41_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(128, 1, 3, 3)"},
        },
    ),
    (
        Reshape1085,
        [((1024, 1, 3, 3), torch.float32)],
        {
            "model_name": [
                "pt_mobilenet_v1_basic_img_cls_torchvision",
                "pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf",
                "pt_mobilenet_v1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pt_xception_xception71_tf_in1k_img_cls_timm",
                "pt_xception_xception65_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
                "pt_xception_xception_img_cls_timm",
                "pt_xception_xception41_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1024, 1, 3, 3)"},
        },
    ),
    (
        Reshape1086,
        [((384, 1, 3, 3), torch.float32)],
        {
            "model_name": [
                "pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(384, 1, 3, 3)"},
        },
    ),
    (
        Reshape1087,
        [((768, 1, 3, 3), torch.float32)],
        {
            "model_name": ["pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(768, 1, 3, 3)"},
        },
    ),
    (
        Reshape1088,
        [((1, 768, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 768, 1, 1)"},
        },
    ),
    (
        Reshape1089,
        [((960, 1, 3, 3), torch.float32)],
        {
            "model_name": [
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(960, 1, 3, 3)"},
        },
    ),
    (
        Reshape1090,
        [((432, 1, 3, 3), torch.float32)],
        {
            "model_name": ["pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(432, 1, 3, 3)"},
        },
    ),
    (
        Reshape1091,
        [((720, 1, 3, 3), torch.float32)],
        {
            "model_name": ["pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(720, 1, 3, 3)"},
        },
    ),
    (
        Reshape1092,
        [((72, 1, 3, 3), torch.float32)],
        {
            "model_name": [
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(72, 1, 3, 3)"},
        },
    ),
    (
        Reshape1093,
        [((88, 1, 3, 3), torch.float32)],
        {
            "model_name": [
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(88, 1, 3, 3)"},
        },
    ),
    (
        Reshape1094,
        [((96, 1, 5, 5), torch.float32)],
        {
            "model_name": [
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(96, 1, 5, 5)"},
        },
    ),
    (
        Reshape1095,
        [((120, 1, 5, 5), torch.float32)],
        {
            "model_name": [
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(120, 1, 5, 5)"},
        },
    ),
    (
        Reshape1096,
        [((200, 1, 3, 3), torch.float32)],
        {
            "model_name": [
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(200, 1, 3, 3)"},
        },
    ),
    (
        Reshape1097,
        [((184, 1, 3, 3), torch.float32)],
        {
            "model_name": [
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(184, 1, 3, 3)"},
        },
    ),
    (
        Reshape1098,
        [((1, 960, 1, 1), torch.float32)],
        {
            "model_name": ["pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 960, 1, 1)"},
        },
    ),
    (
        Reshape1099,
        [((1, 576, 1, 1), torch.float32)],
        {
            "model_name": ["pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 576, 1, 1)"},
        },
    ),
    (
        Reshape1100,
        [((1, 512, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_mobilenetv3_ssd_resnet18_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet34_img_cls_torchvision",
                "pt_resnet_resnet34_img_cls_torchvision",
                "pt_resnet_resnet18_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 512, 1, 1)"},
        },
    ),
    (
        Reshape153,
        [((1, 512, 1, 1), torch.float32)],
        {"model_name": ["pt_vovnet_vovnet27s_obj_det_osmr"], "pcc": 0.99, "op_params": {"shape": "(1, 512)"}},
    ),
    (
        Reshape1101,
        [((1, 3024, 1, 1), torch.float32)],
        {
            "model_name": ["pt_regnet_regnet_y_16gf_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 3024, 1, 1)"},
        },
    ),
    (
        Reshape1102,
        [((1, 2520, 1, 1), torch.float32)],
        {
            "model_name": ["pt_regnet_regnet_x_32gf_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 2520, 1, 1)"},
        },
    ),
    (
        Reshape1103,
        [((1, 888, 1, 1), torch.float32)],
        {
            "model_name": ["pt_regnet_regnet_y_1_6gf_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 888, 1, 1)"},
        },
    ),
    (
        Reshape1104,
        [((1, 3712, 1, 1), torch.float32)],
        {
            "model_name": ["pt_regnet_regnet_y_32gf_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 3712, 1, 1)"},
        },
    ),
    (
        Reshape1105,
        [((1, 784, 1, 1), torch.float32)],
        {
            "model_name": ["pt_regnet_regnet_y_800mf_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 784, 1, 1)"},
        },
    ),
    (
        Reshape1106,
        [((1, 7392, 1, 1), torch.float32)],
        {
            "model_name": ["pt_regnet_regnet_y_128gf_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 7392, 1, 1)"},
        },
    ),
    (
        Reshape1107,
        [((1, 672, 1, 1), torch.float32)],
        {
            "model_name": ["pt_regnet_regnet_x_800mf_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 672, 1, 1)"},
        },
    ),
    (
        Reshape1108,
        [((1, 400, 1, 1), torch.float32)],
        {
            "model_name": ["pt_regnet_regnet_x_400mf_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 400, 1, 1)"},
        },
    ),
    (
        Reshape1109,
        [((1, 1008, 1, 1), torch.float32)],
        {
            "model_name": ["pt_regnet_regnet_x_3_2gf_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1008, 1, 1)"},
        },
    ),
    (
        Reshape1110,
        [((1, 912, 1, 1), torch.float32)],
        {
            "model_name": ["pt_regnet_regnet_x_1_6gf_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 912, 1, 1)"},
        },
    ),
    (
        Reshape1111,
        [((1, 440, 1, 1), torch.float32)],
        {
            "model_name": ["pt_regnet_regnet_y_400mf_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 440, 1, 1)"},
        },
    ),
    (
        Reshape1112,
        [((1, 1088, 1, 1), torch.float32)],
        {
            "model_name": ["pt_regnet_facebook_regnet_y_040_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1088, 1, 1)"},
        },
    ),
    (
        Reshape1113,
        [((1, 2016, 1, 1), torch.float32)],
        {
            "model_name": ["pt_regnet_regnet_y_8gf_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 2016, 1, 1)"},
        },
    ),
    (
        Reshape1114,
        [((1, 1512, 1, 1), torch.float32)],
        {
            "model_name": ["pt_regnet_regnet_y_3_2gf_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1512, 1, 1)"},
        },
    ),
    (
        Reshape1115,
        [((1, 16384, 32), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 16384, 1, 32)"},
        },
    ),
    (
        Reshape1116,
        [((1, 16384, 32), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 128, 128, 32)"},
        },
    ),
    (
        Reshape572,
        [((1, 32, 16384), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 32, 128, 128)"},
        },
    ),
    (
        Reshape1117,
        [((1, 32, 16, 16), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 32, 256)"},
        },
    ),
    (
        Reshape1118,
        [((1, 256, 32), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(256, 32)"},
        },
    ),
    (
        Reshape1119,
        [((1, 256, 32), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 256, 1, 32)"},
        },
    ),
    (
        Reshape1120,
        [((256, 32), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 256, 32)"},
        },
    ),
    (
        Reshape1121,
        [((1, 16384, 256), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1, 16384, 256)"},
        },
    ),
    (
        Reshape1122,
        [((1, 1, 16384, 256), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 16384, 256)"},
        },
    ),
    (
        Reshape1117,
        [((1, 1, 32, 256), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 32, 256)"},
        },
    ),
    (
        Reshape1123,
        [((1, 128, 16384), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 128, 128, 128)"},
        },
    ),
    (
        Reshape1124,
        [((1, 128, 128, 128), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 128, 16384, 1)"},
        },
    ),
    (
        Reshape1125,
        [((1, 64, 64, 64), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 64, 4096, 1)"},
        },
    ),
    (
        Reshape1126,
        [((1, 4096, 64), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 4096, 2, 32)"},
        },
    ),
    (
        Reshape1127,
        [((1, 4096, 64), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 64, 64, 64)"},
        },
    ),
    (
        Reshape1128,
        [((1, 2, 4096, 32), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(2, 4096, 32)"},
        },
    ),
    (
        Reshape1127,
        [((1, 64, 4096), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 64, 64, 64)"},
        },
    ),
    (
        Reshape1129,
        [((1, 64, 16, 16), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 64, 256)"},
        },
    ),
    (
        Reshape1130,
        [((1, 256, 64), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(256, 64)"},
        },
    ),
    (
        Reshape1131,
        [((1, 256, 64), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 256, 2, 32)"},
        },
    ),
    (
        Reshape1132,
        [((1, 256, 64), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 256, 1, 64)"},
        },
    ),
    (
        Reshape1133,
        [((256, 64), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 256, 64)"},
        },
    ),
    (
        Reshape1134,
        [((1, 2, 256, 32), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(2, 256, 32)"},
        },
    ),
    (
        Reshape1135,
        [((2, 4096, 256), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 2, 4096, 256)"},
        },
    ),
    (
        Reshape1136,
        [((1, 2, 4096, 256), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(2, 4096, 256)"},
        },
    ),
    (
        Reshape1137,
        [((1, 2, 32, 256), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(2, 32, 256)"},
        },
    ),
    (
        Reshape1138,
        [((2, 4096, 32), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 2, 4096, 32)"},
        },
    ),
    (
        Reshape1139,
        [((1, 4096, 2, 32), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(4096, 64)"},
        },
    ),
    (
        Reshape1140,
        [((4096, 64), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 4096, 64)"},
        },
    ),
    (
        Reshape1141,
        [((1, 256, 64, 64), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 256, 4096, 1)"},
        },
    ),
    (
        Reshape1142,
        [((1, 160, 32, 32), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 160, 1024, 1)"},
        },
    ),
    (
        Reshape1143,
        [((1, 1024, 160), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1024, 5, 32)"},
        },
    ),
    (
        Reshape1144,
        [((1, 1024, 160), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 32, 32, 160)"},
        },
    ),
    (
        Reshape1145,
        [((1, 5, 1024, 32), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(5, 1024, 32)"},
        },
    ),
    (
        Reshape1146,
        [((1, 160, 1024), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 160, 32, 32)"},
        },
    ),
    (
        Reshape1147,
        [((1, 160, 16, 16), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 160, 256)"},
        },
    ),
    (
        Reshape1148,
        [((1, 256, 160), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(256, 160)"},
        },
    ),
    (
        Reshape1149,
        [((1, 256, 160), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 256, 5, 32)"},
        },
    ),
    (
        Reshape1150,
        [((256, 160), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 256, 160)"},
        },
    ),
    (
        Reshape1151,
        [((1, 5, 256, 32), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(5, 256, 32)"},
        },
    ),
    (
        Reshape1152,
        [((5, 1024, 256), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 5, 1024, 256)"},
        },
    ),
    (
        Reshape1153,
        [((1, 5, 1024, 256), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(5, 1024, 256)"},
        },
    ),
    (
        Reshape1154,
        [((1, 5, 32, 256), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(5, 32, 256)"},
        },
    ),
    (
        Reshape1155,
        [((5, 1024, 32), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 5, 1024, 32)"},
        },
    ),
    (
        Reshape1156,
        [((1, 1024, 5, 32), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1024, 160)"},
        },
    ),
    (
        Reshape1157,
        [((1024, 160), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1024, 160)"},
        },
    ),
    (
        Reshape1158,
        [((1, 640, 1024), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 640, 32, 32)"},
        },
    ),
    (
        Reshape1159,
        [((640, 1, 3, 3), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(640, 1, 3, 3)"},
        },
    ),
    (
        Reshape1160,
        [((1, 640, 32, 32), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 640, 1024, 1)"},
        },
    ),
    (
        Reshape1161,
        [((1, 256, 16, 16), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 256, 256, 1)"},
        },
    ),
    (
        Reshape1162,
        [((1, 8, 32, 256), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(8, 32, 256)"},
        },
    ),
    (
        Reshape1163,
        [((8, 256, 32), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 8, 256, 32)"},
        },
    ),
    (
        Reshape620,
        [((1, 256, 8, 32), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(256, 256)"},
        },
    ),
    (
        Reshape1164,
        [((1, 1024, 256), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1024, 16, 16)"},
        },
    ),
    (
        Reshape1165,
        [((1, 1024, 16, 16), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1024, 256, 1)"},
        },
    ),
    (
        Reshape1166,
        [((1, 16384, 64), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 16384, 1, 64)"},
        },
    ),
    (
        Reshape1167,
        [((1, 16384, 64), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 128, 128, 64)"},
        },
    ),
    (
        Reshape268,
        [((1, 64, 16384), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 64, 128, 128)"},
        },
    ),
    (
        Reshape1129,
        [((1, 1, 64, 256), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 64, 256)"},
        },
    ),
    (
        Reshape1168,
        [((1, 256, 16384), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 256, 128, 128)"},
        },
    ),
    (
        Reshape1169,
        [((1, 256, 128, 128), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 256, 16384, 1)"},
        },
    ),
    (
        Reshape1170,
        [((1, 4096, 128), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 4096, 2, 64)"},
        },
    ),
    (
        Reshape1171,
        [((1, 4096, 128), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 64, 64, 128)"},
        },
    ),
    (
        Reshape1172,
        [((1, 2, 4096, 64), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(2, 4096, 64)"},
        },
    ),
    (
        Reshape1173,
        [((1, 128, 16, 16), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 128, 256)"},
        },
    ),
    (
        Reshape1174,
        [((1, 256, 128), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(256, 128)"},
        },
    ),
    (
        Reshape1175,
        [((1, 256, 128), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 256, 2, 64)"},
        },
    ),
    (
        Reshape1176,
        [((256, 128), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 256, 128)"},
        },
    ),
    (
        Reshape1177,
        [((1, 2, 256, 64), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(2, 256, 64)"},
        },
    ),
    (
        Reshape1178,
        [((1, 2, 64, 256), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(2, 64, 256)"},
        },
    ),
    (
        Reshape1179,
        [((2, 4096, 64), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 2, 4096, 64)"},
        },
    ),
    (
        Reshape1180,
        [((1, 4096, 2, 64), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(4096, 128)"},
        },
    ),
    (
        Reshape1181,
        [((4096, 128), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 4096, 128)"},
        },
    ),
    (
        Reshape1182,
        [((1, 512, 4096), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 512, 64, 64)"},
        },
    ),
    (
        Reshape1183,
        [((1, 512, 64, 64), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 512, 4096, 1)"},
        },
    ),
    (
        Reshape1184,
        [((1, 320, 32, 32), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 320, 1024, 1)"},
        },
    ),
    (
        Reshape1185,
        [((1, 1024, 320), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1024, 5, 64)"},
        },
    ),
    (
        Reshape1186,
        [((1, 1024, 320), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 32, 32, 320)"},
        },
    ),
    (
        Reshape1187,
        [((1, 5, 1024, 64), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(5, 1024, 64)"},
        },
    ),
    (
        Reshape1188,
        [((1, 320, 1024), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 320, 32, 32)"},
        },
    ),
    (
        Reshape1189,
        [((1, 320, 16, 16), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 320, 256)"},
        },
    ),
    (
        Reshape1190,
        [((1, 256, 320), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(256, 320)"},
        },
    ),
    (
        Reshape1191,
        [((1, 256, 320), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 256, 5, 64)"},
        },
    ),
    (
        Reshape1192,
        [((256, 320), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 256, 320)"},
        },
    ),
    (
        Reshape1193,
        [((1, 5, 256, 64), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(5, 256, 64)"},
        },
    ),
    (
        Reshape1194,
        [((1, 5, 64, 256), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(5, 64, 256)"},
        },
    ),
    (
        Reshape1195,
        [((5, 1024, 64), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 5, 1024, 64)"},
        },
    ),
    (
        Reshape1196,
        [((1, 1024, 5, 64), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1024, 320)"},
        },
    ),
    (
        Reshape1197,
        [((1024, 320), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1024, 320)"},
        },
    ),
    (
        Reshape1198,
        [((1, 1280, 1024), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1280, 32, 32)"},
        },
    ),
    (
        Reshape1199,
        [((1, 1280, 32, 32), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1280, 1024, 1)"},
        },
    ),
    (
        Reshape1200,
        [((1, 512, 16, 16), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 512, 256, 1)"},
        },
    ),
    (
        Reshape1201,
        [((1, 8, 256, 64), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(8, 256, 64)"},
        },
    ),
    (
        Reshape1202,
        [((1, 8, 64, 256), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(8, 64, 256)"},
        },
    ),
    (
        Reshape1203,
        [((8, 256, 64), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 8, 256, 64)"},
        },
    ),
    (
        Reshape546,
        [((1, 256, 8, 64), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(256, 512)"},
        },
    ),
    (
        Reshape1204,
        [((1, 2048, 16, 16), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 2048, 256, 1)"},
        },
    ),
    (
        Reshape1205,
        [((1, 768, 256), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 768, 16, 16)"},
        },
    ),
    (
        Reshape1206,
        [((1, 768, 1024), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 768, 32, 32)"},
        },
    ),
    (
        Reshape1207,
        [((1, 768, 4096), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 768, 64, 64)"},
        },
    ),
    (
        Reshape1208,
        [((1, 768, 16384), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 768, 128, 128)"},
        },
    ),
    (
        Reshape1209,
        [((1, 16, 38, 38), torch.float32)],
        {
            "model_name": ["pt_ssd300_resnet50_base_img_cls_torchhub"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 4, 5776)"},
        },
    ),
    (
        Reshape1210,
        [((1, 24, 19, 19), torch.float32)],
        {
            "model_name": ["pt_ssd300_resnet50_base_img_cls_torchhub"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 4, 2166)"},
        },
    ),
    (
        Reshape1211,
        [((1, 24, 10, 10), torch.float32)],
        {
            "model_name": ["pt_ssd300_resnet50_base_img_cls_torchhub"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 4, 600)"},
        },
    ),
    (
        Reshape1212,
        [((1, 24, 5, 5), torch.float32)],
        {
            "model_name": ["pt_ssd300_resnet50_base_img_cls_torchhub"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 4, 150)"},
        },
    ),
    (
        Reshape1213,
        [((1, 16, 3, 3), torch.float32)],
        {"model_name": ["pt_ssd300_resnet50_base_img_cls_torchhub"], "pcc": 0.99, "op_params": {"shape": "(1, 4, 36)"}},
    ),
    (
        Reshape1214,
        [((1, 324, 38, 38), torch.float32)],
        {
            "model_name": ["pt_ssd300_resnet50_base_img_cls_torchhub"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 81, 5776)"},
        },
    ),
    (
        Reshape1215,
        [((1, 486, 19, 19), torch.float32)],
        {
            "model_name": ["pt_ssd300_resnet50_base_img_cls_torchhub"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 81, 2166)"},
        },
    ),
    (
        Reshape1216,
        [((1, 486, 10, 10), torch.float32)],
        {
            "model_name": ["pt_ssd300_resnet50_base_img_cls_torchhub"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 81, 600)"},
        },
    ),
    (
        Reshape1217,
        [((1, 486, 5, 5), torch.float32)],
        {
            "model_name": ["pt_ssd300_resnet50_base_img_cls_torchhub"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 81, 150)"},
        },
    ),
    (
        Reshape1218,
        [((1, 324, 3, 3), torch.float32)],
        {
            "model_name": ["pt_ssd300_resnet50_base_img_cls_torchhub"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 81, 36)"},
        },
    ),
    (
        Reshape1219,
        [((1, 324, 1, 1), torch.float32)],
        {"model_name": ["pt_ssd300_resnet50_base_img_cls_torchhub"], "pcc": 0.99, "op_params": {"shape": "(1, 81, 4)"}},
    ),
    (
        Reshape1220,
        [((1, 56, 56, 96), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 8, 7, 8, 7, 96)"},
        },
    ),
    (
        Reshape1221,
        [((1, 56, 56, 96), torch.float32)],
        {
            "model_name": ["pt_swin_swin_t_img_cls_torchvision", "pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(3136, 96)"},
        },
    ),
    (
        Reshape1222,
        [((1, 56, 56, 96), torch.float32)],
        {
            "model_name": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 3136, 96)"},
        },
    ),
    (
        Reshape1221,
        [((1, 8, 8, 7, 7, 96), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(3136, 96)"},
        },
    ),
    (
        Reshape1223,
        [((3136, 288), torch.float32)],
        {
            "model_name": ["pt_swin_swin_t_img_cls_torchvision", "pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(64, 49, 288)"},
        },
    ),
    (
        Reshape1224,
        [((64, 49, 288), torch.float32)],
        {
            "model_name": ["pt_swin_swin_t_img_cls_torchvision", "pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(64, 49, 3, 3, 32)"},
        },
    ),
    (
        Reshape1225,
        [((1, 64, 3, 49, 32), torch.float32)],
        {
            "model_name": ["pt_swin_swin_t_img_cls_torchvision", "pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(64, 3, 49, 32)"},
        },
    ),
    (
        Reshape1226,
        [((1, 64, 3, 49, 32), torch.float32)],
        {
            "model_name": ["pt_swin_swin_t_img_cls_torchvision", "pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(192, 49, 32)"},
        },
    ),
    (
        Reshape1226,
        [((64, 3, 49, 32), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(192, 49, 32)"},
        },
    ),
    (
        Reshape1227,
        [((192, 49, 49), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(64, 3, 49, 49)"},
        },
    ),
    (
        Reshape1228,
        [((2401, 3), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(49, 49, 3)"},
        },
    ),
    (
        Reshape1229,
        [((64, 3, 49, 49), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(192, 49, 49)"},
        },
    ),
    (
        Reshape1230,
        [((64, 3, 49, 49), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 64, 3, 49, 49)"},
        },
    ),
    (
        Reshape1231,
        [((64, 3, 32, 49), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(192, 32, 49)"},
        },
    ),
    (
        Reshape1225,
        [((192, 49, 32), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(64, 3, 49, 32)"},
        },
    ),
    (
        Reshape1221,
        [((64, 49, 3, 32), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(3136, 96)"},
        },
    ),
    (
        Reshape1232,
        [((3136, 96), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(64, 49, 96)"},
        },
    ),
    (
        Reshape1233,
        [((3136, 96), torch.float32)],
        {
            "model_name": ["pt_swin_swin_t_img_cls_torchvision", "pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 56, 56, 96)"},
        },
    ),
    (
        Reshape1234,
        [((64, 49, 96), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 8, 8, 7, 7, 96)"},
        },
    ),
    (
        Reshape1235,
        [((64, 49, 96), torch.float32)],
        {
            "model_name": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(64, 49, 3, 32)"},
        },
    ),
    (
        Reshape1233,
        [((1, 8, 7, 8, 7, 96), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 56, 56, 96)"},
        },
    ),
    (
        Reshape1222,
        [((1, 8, 7, 8, 7, 96), torch.float32)],
        {
            "model_name": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 3136, 96)"},
        },
    ),
    (
        Reshape1236,
        [((3136, 384), torch.float32)],
        {
            "model_name": ["pt_swin_swin_t_img_cls_torchvision", "pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 56, 56, 384)"},
        },
    ),
    (
        Reshape1237,
        [((3136, 384), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(64, 49, 384)"}},
    ),
    (
        Reshape1238,
        [((1, 56, 56, 384), torch.float32)],
        {
            "model_name": ["pt_swin_swin_t_img_cls_torchvision", "pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(3136, 384)"},
        },
    ),
    (
        Reshape1227,
        [((1, 64, 3, 49, 49), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(64, 3, 49, 49)"},
        },
    ),
    (
        Reshape1239,
        [((1, 28, 28, 384), torch.float32)],
        {
            "model_name": ["pt_swin_swin_t_img_cls_torchvision", "pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(784, 384)"},
        },
    ),
    (
        Reshape1240,
        [((1, 28, 28, 384), torch.float32)],
        {
            "model_name": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 784, 384)"},
        },
    ),
    (
        Reshape1241,
        [((784, 192), torch.float32)],
        {
            "model_name": ["pt_swin_swin_t_img_cls_torchvision", "pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 28, 28, 192)"},
        },
    ),
    (
        Reshape1242,
        [((784, 192), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(16, 49, 192)"},
        },
    ),
    (
        Reshape1243,
        [((1, 28, 28, 192), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 4, 7, 4, 7, 192)"},
        },
    ),
    (
        Reshape1244,
        [((1, 28, 28, 192), torch.float32)],
        {
            "model_name": ["pt_swin_swin_t_img_cls_torchvision", "pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(784, 192)"},
        },
    ),
    (
        Reshape1245,
        [((1, 28, 28, 192), torch.float32)],
        {
            "model_name": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 784, 192)"},
        },
    ),
    (
        Reshape1244,
        [((1, 4, 4, 7, 7, 192), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(784, 192)"},
        },
    ),
    (
        Reshape1246,
        [((784, 576), torch.float32)],
        {
            "model_name": ["pt_swin_swin_t_img_cls_torchvision", "pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(16, 49, 576)"},
        },
    ),
    (
        Reshape1247,
        [((16, 49, 576), torch.float32)],
        {
            "model_name": ["pt_swin_swin_t_img_cls_torchvision", "pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(16, 49, 3, 6, 32)"},
        },
    ),
    (
        Reshape1248,
        [((1, 16, 6, 49, 32), torch.float32)],
        {
            "model_name": ["pt_swin_swin_t_img_cls_torchvision", "pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(16, 6, 49, 32)"},
        },
    ),
    (
        Reshape1249,
        [((1, 16, 6, 49, 32), torch.float32)],
        {
            "model_name": ["pt_swin_swin_t_img_cls_torchvision", "pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(96, 49, 32)"},
        },
    ),
    (
        Reshape1249,
        [((16, 6, 49, 32), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(96, 49, 32)"},
        },
    ),
    (
        Reshape1250,
        [((96, 49, 49), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(16, 6, 49, 49)"},
        },
    ),
    (
        Reshape1251,
        [((2401, 6), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(49, 49, 6)"},
        },
    ),
    (
        Reshape1252,
        [((16, 6, 49, 49), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(96, 49, 49)"},
        },
    ),
    (
        Reshape1253,
        [((16, 6, 49, 49), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 16, 6, 49, 49)"},
        },
    ),
    (
        Reshape1254,
        [((16, 6, 32, 49), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(96, 32, 49)"},
        },
    ),
    (
        Reshape1248,
        [((96, 49, 32), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(16, 6, 49, 32)"},
        },
    ),
    (
        Reshape1244,
        [((16, 49, 6, 32), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(784, 192)"},
        },
    ),
    (
        Reshape1255,
        [((16, 49, 192), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 4, 4, 7, 7, 192)"},
        },
    ),
    (
        Reshape1256,
        [((16, 49, 192), torch.float32)],
        {
            "model_name": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(16, 49, 6, 32)"},
        },
    ),
    (
        Reshape1241,
        [((1, 4, 7, 4, 7, 192), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 28, 28, 192)"},
        },
    ),
    (
        Reshape1245,
        [((1, 4, 7, 4, 7, 192), torch.float32)],
        {
            "model_name": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 784, 192)"},
        },
    ),
    (
        Reshape1257,
        [((784, 768), torch.float32)],
        {
            "model_name": ["pt_swin_swin_t_img_cls_torchvision", "pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 28, 28, 768)"},
        },
    ),
    (
        Reshape1258,
        [((784, 768), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(16, 49, 768)"}},
    ),
    (
        Reshape1259,
        [((1, 28, 28, 768), torch.float32)],
        {
            "model_name": ["pt_swin_swin_t_img_cls_torchvision", "pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(784, 768)"},
        },
    ),
    (
        Reshape1250,
        [((1, 16, 6, 49, 49), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(16, 6, 49, 49)"},
        },
    ),
    (
        Reshape1260,
        [((1, 14, 14, 768), torch.float32)],
        {
            "model_name": ["pt_swin_swin_t_img_cls_torchvision", "pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(196, 768)"},
        },
    ),
    (
        Reshape1261,
        [((1, 14, 14, 768), torch.float32)],
        {
            "model_name": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 196, 768)"},
        },
    ),
    (
        Reshape1262,
        [((196, 384), torch.float32)],
        {
            "model_name": ["pt_swin_swin_t_img_cls_torchvision", "pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 14, 14, 384)"},
        },
    ),
    (
        Reshape1263,
        [((196, 384), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(4, 49, 384)"},
        },
    ),
    (
        Reshape1264,
        [((1, 14, 14, 384), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 2, 7, 2, 7, 384)"},
        },
    ),
    (
        Reshape1265,
        [((1, 14, 14, 384), torch.float32)],
        {
            "model_name": ["pt_swin_swin_t_img_cls_torchvision", "pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(196, 384)"},
        },
    ),
    (
        Reshape1266,
        [((1, 14, 14, 384), torch.float32)],
        {
            "model_name": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 196, 384)"},
        },
    ),
    (
        Reshape1265,
        [((1, 2, 2, 7, 7, 384), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(196, 384)"},
        },
    ),
    (
        Reshape1267,
        [((196, 1152), torch.float32)],
        {
            "model_name": ["pt_swin_swin_t_img_cls_torchvision", "pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(4, 49, 1152)"},
        },
    ),
    (
        Reshape1268,
        [((4, 49, 1152), torch.float32)],
        {
            "model_name": ["pt_swin_swin_t_img_cls_torchvision", "pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(4, 49, 3, 12, 32)"},
        },
    ),
    (
        Reshape1269,
        [((1, 4, 12, 49, 32), torch.float32)],
        {
            "model_name": ["pt_swin_swin_t_img_cls_torchvision", "pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(4, 12, 49, 32)"},
        },
    ),
    (
        Reshape1270,
        [((1, 4, 12, 49, 32), torch.float32)],
        {
            "model_name": ["pt_swin_swin_t_img_cls_torchvision", "pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(48, 49, 32)"},
        },
    ),
    (
        Reshape1270,
        [((4, 12, 49, 32), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(48, 49, 32)"},
        },
    ),
    (
        Reshape1271,
        [((48, 49, 49), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(4, 12, 49, 49)"},
        },
    ),
    (
        Reshape1272,
        [((2401, 12), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(49, 49, 12)"},
        },
    ),
    (
        Reshape1273,
        [((4, 12, 49, 49), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(48, 49, 49)"},
        },
    ),
    (
        Reshape1274,
        [((4, 12, 49, 49), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 4, 12, 49, 49)"},
        },
    ),
    (
        Reshape1275,
        [((4, 12, 32, 49), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(48, 32, 49)"},
        },
    ),
    (
        Reshape1269,
        [((48, 49, 32), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(4, 12, 49, 32)"},
        },
    ),
    (
        Reshape1265,
        [((4, 49, 12, 32), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(196, 384)"},
        },
    ),
    (
        Reshape1276,
        [((4, 49, 384), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 2, 2, 7, 7, 384)"},
        },
    ),
    (
        Reshape1277,
        [((4, 49, 384), torch.float32)],
        {
            "model_name": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(4, 49, 12, 32)"},
        },
    ),
    (
        Reshape1262,
        [((1, 2, 7, 2, 7, 384), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 14, 14, 384)"},
        },
    ),
    (
        Reshape1266,
        [((1, 2, 7, 2, 7, 384), torch.float32)],
        {
            "model_name": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 196, 384)"},
        },
    ),
    (
        Reshape1278,
        [((196, 1536), torch.float32)],
        {
            "model_name": ["pt_swin_swin_t_img_cls_torchvision", "pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 14, 14, 1536)"},
        },
    ),
    (
        Reshape1279,
        [((196, 1536), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(4, 49, 1536)"}},
    ),
    (
        Reshape1280,
        [((1, 14, 14, 1536), torch.float32)],
        {
            "model_name": ["pt_swin_swin_t_img_cls_torchvision", "pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(196, 1536)"},
        },
    ),
    (
        Reshape1271,
        [((1, 4, 12, 49, 49), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(4, 12, 49, 49)"},
        },
    ),
    (
        Reshape1281,
        [((1, 7, 7, 1536), torch.float32)],
        {
            "model_name": ["pt_swin_swin_t_img_cls_torchvision", "pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(49, 1536)"},
        },
    ),
    (
        Reshape1282,
        [((1, 7, 7, 1536), torch.float32)],
        {
            "model_name": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 49, 1536)"},
        },
    ),
    (
        Reshape1283,
        [((49, 768), torch.float32)],
        {
            "model_name": ["pt_swin_swin_t_img_cls_torchvision", "pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 7, 7, 768)"},
        },
    ),
    (
        Reshape1284,
        [((49, 768), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 49, 768)"},
        },
    ),
    (
        Reshape1285,
        [((1, 7, 7, 768), torch.float32)],
        {
            "model_name": ["pt_swin_swin_t_img_cls_torchvision", "pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1, 7, 1, 7, 768)"},
        },
    ),
    (
        Reshape1286,
        [((1, 7, 7, 768), torch.float32)],
        {
            "model_name": ["pt_swin_swin_t_img_cls_torchvision", "pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(49, 768)"},
        },
    ),
    (
        Reshape1286,
        [((1, 1, 1, 7, 7, 768), torch.float32)],
        {
            "model_name": ["pt_swin_swin_t_img_cls_torchvision", "pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(49, 768)"},
        },
    ),
    (
        Reshape1287,
        [((49, 2304), torch.float32)],
        {
            "model_name": ["pt_swin_swin_t_img_cls_torchvision", "pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 49, 2304)"},
        },
    ),
    (
        Reshape1288,
        [((1, 49, 2304), torch.float32)],
        {
            "model_name": ["pt_swin_swin_t_img_cls_torchvision", "pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 49, 3, 24, 32)"},
        },
    ),
    (
        Reshape1289,
        [((1, 1, 24, 49, 32), torch.float32)],
        {
            "model_name": ["pt_swin_swin_t_img_cls_torchvision", "pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 24, 49, 32)"},
        },
    ),
    (
        Reshape1290,
        [((1, 1, 24, 49, 32), torch.float32)],
        {
            "model_name": ["pt_swin_swin_t_img_cls_torchvision", "pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(24, 49, 32)"},
        },
    ),
    (
        Reshape1290,
        [((1, 24, 49, 32), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(24, 49, 32)"},
        },
    ),
    (
        Reshape1291,
        [((24, 49, 49), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 24, 49, 49)"},
        },
    ),
    (
        Reshape1292,
        [((2401, 24), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(49, 49, 24)"},
        },
    ),
    (
        Reshape1293,
        [((1, 24, 49, 49), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(24, 49, 49)"},
        },
    ),
    (
        Reshape1294,
        [((1, 24, 32, 49), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(24, 32, 49)"},
        },
    ),
    (
        Reshape1289,
        [((24, 49, 32), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 24, 49, 32)"},
        },
    ),
    (
        Reshape1286,
        [((1, 49, 24, 32), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(49, 768)"},
        },
    ),
    (
        Reshape1295,
        [((1, 49, 768), torch.float32)],
        {
            "model_name": ["pt_swin_swin_t_img_cls_torchvision", "pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1, 1, 7, 7, 768)"},
        },
    ),
    (
        Reshape1286,
        [((1, 49, 768), torch.float32)],
        {
            "model_name": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(49, 768)"},
        },
    ),
    (
        Reshape1296,
        [((1, 49, 768), torch.float32)],
        {
            "model_name": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 49, 24, 32)"},
        },
    ),
    (
        Reshape1284,
        [((1, 49, 768), torch.float32)],
        {
            "model_name": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 49, 768)"},
        },
    ),
    (
        Reshape1283,
        [((1, 1, 7, 1, 7, 768), torch.float32)],
        {
            "model_name": ["pt_swin_swin_t_img_cls_torchvision", "pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 7, 7, 768)"},
        },
    ),
    (
        Reshape1297,
        [((49, 3072), torch.float32)],
        {
            "model_name": ["pt_swin_swin_t_img_cls_torchvision", "pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 7, 7, 3072)"},
        },
    ),
    (
        Reshape1298,
        [((49, 3072), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(1, 49, 3072)"}},
    ),
    (
        Reshape1299,
        [((1, 7, 7, 3072), torch.float32)],
        {
            "model_name": ["pt_swin_swin_t_img_cls_torchvision", "pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(49, 3072)"},
        },
    ),
    (
        Reshape1300,
        [((1, 96, 56, 56), torch.float32)],
        {
            "model_name": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 96, 3136, 1)"},
        },
    ),
    (
        Reshape1220,
        [((1, 3136, 96), torch.float32)],
        {
            "model_name": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 8, 7, 8, 7, 96)"},
        },
    ),
    (
        Reshape1233,
        [((1, 3136, 96), torch.float32)],
        {
            "model_name": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 56, 56, 96)"},
        },
    ),
    pytest.param(
        (
            Reshape1301,
            [((49, 49), torch.int64)],
            {
                "model_name": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"],
                "pcc": 0.99,
                "op_params": {"shape": "(2401,)"},
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Reshape1243,
        [((1, 784, 192), torch.float32)],
        {
            "model_name": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 4, 7, 4, 7, 192)"},
        },
    ),
    (
        Reshape1241,
        [((1, 784, 192), torch.float32)],
        {
            "model_name": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 28, 28, 192)"},
        },
    ),
    (
        Reshape1264,
        [((1, 196, 384), torch.float32)],
        {
            "model_name": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 2, 7, 2, 7, 384)"},
        },
    ),
    (
        Reshape1262,
        [((1, 196, 384), torch.float32)],
        {
            "model_name": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 14, 14, 384)"},
        },
    ),
    (
        Reshape1302,
        [((1, 768, 1), torch.float32)],
        {
            "model_name": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 768, 1)"},
        },
    ),
    (
        Reshape1303,
        [((1, 56, 56, 128), torch.float32)],
        {
            "model_name": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 8, 7, 8, 7, 128)"},
        },
    ),
    (
        Reshape1304,
        [((1, 56, 56, 128), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(3136, 128)"}},
    ),
    (
        Reshape1304,
        [((1, 8, 8, 7, 7, 128), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(3136, 128)"}},
    ),
    (
        Reshape1305,
        [((64, 49, 384), torch.float32)],
        {
            "model_name": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(64, 49, 3, 4, 32)"},
        },
    ),
    (
        Reshape1306,
        [((1, 64, 4, 49, 32), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(64, 4, 49, 32)"}},
    ),
    (
        Reshape1307,
        [((1, 64, 4, 49, 32), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(256, 49, 32)"}},
    ),
    (
        Reshape1307,
        [((64, 4, 49, 32), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(256, 49, 32)"}},
    ),
    (
        Reshape1308,
        [((256, 49, 49), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(64, 4, 49, 49)"}},
    ),
    (
        Reshape1309,
        [((2401, 4), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(49, 49, 4)"}},
    ),
    (
        Reshape1310,
        [((64, 4, 49, 49), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(256, 49, 49)"}},
    ),
    (
        Reshape1311,
        [((64, 4, 49, 49), torch.float32)],
        {
            "model_name": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 64, 4, 49, 49)"},
        },
    ),
    (
        Reshape1312,
        [((64, 4, 32, 49), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(256, 32, 49)"}},
    ),
    (
        Reshape1306,
        [((256, 49, 32), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(64, 4, 49, 32)"}},
    ),
    (
        Reshape1304,
        [((64, 49, 4, 32), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(3136, 128)"}},
    ),
    (
        Reshape1313,
        [((3136, 128), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(64, 49, 128)"}},
    ),
    (
        Reshape1314,
        [((3136, 128), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(1, 56, 56, 128)"}},
    ),
    (
        Reshape1315,
        [((64, 49, 128), torch.float32)],
        {
            "model_name": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 8, 8, 7, 7, 128)"},
        },
    ),
    (
        Reshape1314,
        [((1, 8, 7, 8, 7, 128), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(1, 56, 56, 128)"}},
    ),
    (
        Reshape1316,
        [((3136, 512), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(1, 56, 56, 512)"}},
    ),
    (
        Reshape1317,
        [((1, 56, 56, 512), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(3136, 512)"}},
    ),
    (
        Reshape1308,
        [((1, 64, 4, 49, 49), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(64, 4, 49, 49)"}},
    ),
    (
        Reshape1318,
        [((1, 28, 28, 512), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(784, 512)"}},
    ),
    (
        Reshape1319,
        [((784, 256), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(1, 28, 28, 256)"}},
    ),
    (
        Reshape1320,
        [((784, 256), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(16, 49, 256)"}},
    ),
    (
        Reshape1321,
        [((1, 28, 28, 256), torch.float32)],
        {
            "model_name": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 4, 7, 4, 7, 256)"},
        },
    ),
    (
        Reshape1322,
        [((1, 28, 28, 256), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(784, 256)"}},
    ),
    (
        Reshape1322,
        [((1, 4, 4, 7, 7, 256), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(784, 256)"}},
    ),
    (
        Reshape1323,
        [((16, 49, 768), torch.float32)],
        {
            "model_name": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(16, 49, 3, 8, 32)"},
        },
    ),
    (
        Reshape1324,
        [((1, 16, 8, 49, 32), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(16, 8, 49, 32)"}},
    ),
    (
        Reshape1325,
        [((1, 16, 8, 49, 32), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(128, 49, 32)"}},
    ),
    (
        Reshape1325,
        [((16, 8, 49, 32), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(128, 49, 32)"}},
    ),
    (
        Reshape1326,
        [((128, 49, 49), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(16, 8, 49, 49)"}},
    ),
    (
        Reshape1327,
        [((2401, 8), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(49, 49, 8)"}},
    ),
    (
        Reshape1328,
        [((16, 8, 49, 49), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(128, 49, 49)"}},
    ),
    (
        Reshape1329,
        [((16, 8, 49, 49), torch.float32)],
        {
            "model_name": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 16, 8, 49, 49)"},
        },
    ),
    (
        Reshape1330,
        [((16, 8, 32, 49), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(128, 32, 49)"}},
    ),
    (
        Reshape1324,
        [((128, 49, 32), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(16, 8, 49, 32)"}},
    ),
    (
        Reshape1322,
        [((16, 49, 8, 32), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(784, 256)"}},
    ),
    (
        Reshape1331,
        [((16, 49, 256), torch.float32)],
        {
            "model_name": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 4, 4, 7, 7, 256)"},
        },
    ),
    (
        Reshape1319,
        [((1, 4, 7, 4, 7, 256), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(1, 28, 28, 256)"}},
    ),
    (
        Reshape1332,
        [((784, 1024), torch.float32)],
        {
            "model_name": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 28, 28, 1024)"},
        },
    ),
    (
        Reshape1333,
        [((1, 28, 28, 1024), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(784, 1024)"}},
    ),
    (
        Reshape1326,
        [((1, 16, 8, 49, 49), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(16, 8, 49, 49)"}},
    ),
    (
        Reshape1334,
        [((1, 14, 14, 1024), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(196, 1024)"}},
    ),
    (
        Reshape1335,
        [((196, 512), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(1, 14, 14, 512)"}},
    ),
    (
        Reshape1336,
        [((196, 512), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(4, 49, 512)"}},
    ),
    (
        Reshape1337,
        [((1, 14, 14, 512), torch.float32)],
        {
            "model_name": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 2, 7, 2, 7, 512)"},
        },
    ),
    (
        Reshape1338,
        [((1, 14, 14, 512), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(196, 512)"}},
    ),
    (
        Reshape1338,
        [((1, 2, 2, 7, 7, 512), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(196, 512)"}},
    ),
    (
        Reshape1339,
        [((4, 49, 1536), torch.float32)],
        {
            "model_name": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(4, 49, 3, 16, 32)"},
        },
    ),
    (
        Reshape1340,
        [((1, 4, 16, 49, 32), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(4, 16, 49, 32)"}},
    ),
    (
        Reshape1341,
        [((1, 4, 16, 49, 32), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(64, 49, 32)"}},
    ),
    (
        Reshape1341,
        [((4, 16, 49, 32), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(64, 49, 32)"}},
    ),
    (
        Reshape1342,
        [((64, 49, 49), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(4, 16, 49, 49)"}},
    ),
    (
        Reshape1343,
        [((2401, 16), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(49, 49, 16)"}},
    ),
    (
        Reshape1344,
        [((4, 16, 49, 49), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(64, 49, 49)"}},
    ),
    (
        Reshape1345,
        [((4, 16, 49, 49), torch.float32)],
        {
            "model_name": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 4, 16, 49, 49)"},
        },
    ),
    (
        Reshape1346,
        [((4, 16, 32, 49), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(64, 32, 49)"}},
    ),
    (
        Reshape1340,
        [((64, 49, 32), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(4, 16, 49, 32)"}},
    ),
    (
        Reshape1338,
        [((4, 49, 16, 32), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(196, 512)"}},
    ),
    (
        Reshape1347,
        [((4, 49, 512), torch.float32)],
        {
            "model_name": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 2, 2, 7, 7, 512)"},
        },
    ),
    (
        Reshape1335,
        [((1, 2, 7, 2, 7, 512), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(1, 14, 14, 512)"}},
    ),
    (
        Reshape1348,
        [((196, 2048), torch.float32)],
        {
            "model_name": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 14, 14, 2048)"},
        },
    ),
    (
        Reshape1349,
        [((1, 14, 14, 2048), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(196, 2048)"}},
    ),
    (
        Reshape1342,
        [((1, 4, 16, 49, 49), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(4, 16, 49, 49)"}},
    ),
    (
        Reshape1350,
        [((1, 7, 7, 2048), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(49, 2048)"}},
    ),
    (
        Reshape1351,
        [((49, 1024), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(1, 7, 7, 1024)"}},
    ),
    (
        Reshape1352,
        [((49, 1024), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(1, 49, 1024)"}},
    ),
    (
        Reshape1353,
        [((1, 7, 7, 1024), torch.float32)],
        {
            "model_name": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1, 7, 1, 7, 1024)"},
        },
    ),
    (
        Reshape1354,
        [((1, 7, 7, 1024), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(49, 1024)"}},
    ),
    (
        Reshape1354,
        [((1, 1, 1, 7, 7, 1024), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(49, 1024)"}},
    ),
    (
        Reshape1355,
        [((1, 49, 3072), torch.float32)],
        {
            "model_name": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 49, 3, 32, 32)"},
        },
    ),
    (
        Reshape1356,
        [((1, 1, 32, 49, 32), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(1, 32, 49, 32)"}},
    ),
    (
        Reshape1357,
        [((1, 1, 32, 49, 32), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(32, 49, 32)"}},
    ),
    (
        Reshape1357,
        [((1, 32, 49, 32), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(32, 49, 32)"}},
    ),
    (
        Reshape1358,
        [((32, 49, 49), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(1, 32, 49, 49)"}},
    ),
    (
        Reshape1359,
        [((2401, 32), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(49, 49, 32)"}},
    ),
    (
        Reshape1360,
        [((1, 32, 49, 49), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(32, 49, 49)"}},
    ),
    (
        Reshape1361,
        [((1, 32, 32, 49), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(32, 32, 49)"}},
    ),
    (
        Reshape1356,
        [((32, 49, 32), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(1, 32, 49, 32)"}},
    ),
    (
        Reshape1354,
        [((1, 49, 32, 32), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(49, 1024)"}},
    ),
    (
        Reshape1362,
        [((1, 49, 1024), torch.float32)],
        {
            "model_name": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1, 1, 7, 7, 1024)"},
        },
    ),
    (
        Reshape1351,
        [((1, 1, 7, 1, 7, 1024), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(1, 7, 7, 1024)"}},
    ),
    (
        Reshape1363,
        [((49, 4096), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(1, 7, 7, 4096)"}},
    ),
    (
        Reshape1364,
        [((1, 7, 7, 4096), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(49, 4096)"}},
    ),
    (
        Reshape1365,
        [((1, 512, 7, 7), torch.float32)],
        {
            "model_name": [
                "pt_vgg_vgg16_bn_img_cls_torchvision",
                "pt_vgg_19_obj_det_hf",
                "pt_vgg_vgg11_bn_img_cls_torchvision",
                "pt_vgg_vgg13_img_cls_torchvision",
                "pt_vgg_vgg11_img_cls_torchvision",
                "pt_vgg_vgg16_img_cls_torchvision",
                "pt_vgg_vgg19_bn_obj_det_torchhub",
                "pt_vgg_vgg13_bn_img_cls_torchvision",
                "pt_vgg_vgg19_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 25088, 1, 1)"},
        },
    ),
    (
        Reshape1366,
        [((1, 512, 7, 7), torch.float32)],
        {
            "model_name": [
                "pt_vgg_vgg11_obj_det_osmr",
                "pt_vgg_vgg13_obj_det_osmr",
                "pt_vgg_bn_vgg19b_obj_det_osmr",
                "pt_vgg_vgg16_obj_det_osmr",
                "pt_vgg_vgg19_obj_det_osmr",
                "pt_vgg_bn_vgg19_obj_det_osmr",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 25088)"},
        },
    ),
    (
        Reshape1367,
        [((1, 4096, 1, 1), torch.float32)],
        {"model_name": ["pt_vgg_vgg19_bn_obj_det_timm"], "pcc": 0.99, "op_params": {"shape": "(1, 4096, 1, 1)"}},
    ),
    (
        Reshape1368,
        [((160, 1, 3, 3), torch.float32)],
        {
            "model_name": [
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(160, 1, 3, 3)"},
        },
    ),
    (
        Reshape1369,
        [((224, 1, 3, 3), torch.float32)],
        {
            "model_name": [
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(224, 1, 3, 3)"},
        },
    ),
    (
        Reshape1370,
        [((728, 1, 3, 3), torch.float32)],
        {
            "model_name": [
                "pt_xception_xception71_tf_in1k_img_cls_timm",
                "pt_xception_xception65_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
                "pt_xception_xception_img_cls_timm",
                "pt_xception_xception41_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(728, 1, 3, 3)"},
        },
    ),
    (
        Reshape1371,
        [((1536, 1, 3, 3), torch.float32)],
        {
            "model_name": [
                "pt_xception_xception71_tf_in1k_img_cls_timm",
                "pt_xception_xception65_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
                "pt_xception_xception_img_cls_timm",
                "pt_xception_xception41_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1536, 1, 3, 3)"},
        },
    ),
    (
        Reshape1372,
        [((1, 3, 85, 40, 40), torch.float32)],
        {
            "model_name": ["pt_yolo_v5_yolov5n_imgcls_torchhub_320x320"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 255, 40, 40)"},
        },
    ),
    (
        Reshape1373,
        [((1, 255, 40, 40), torch.float32)],
        {
            "model_name": ["pt_yolo_v5_yolov5n_imgcls_torchhub_320x320"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1, 255, 1600)"},
        },
    ),
    (
        Reshape1374,
        [((1, 1, 255, 1600), torch.float32)],
        {
            "model_name": ["pt_yolo_v5_yolov5n_imgcls_torchhub_320x320"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 3, 85, 1600)"},
        },
    ),
    (
        Reshape1375,
        [((1, 3, 1600, 85), torch.float32)],
        {
            "model_name": ["pt_yolo_v5_yolov5n_imgcls_torchhub_320x320"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 4800, 85)"},
        },
    ),
    (
        Reshape1376,
        [((1, 3, 85, 20, 20), torch.float32)],
        {
            "model_name": ["pt_yolo_v5_yolov5n_imgcls_torchhub_320x320"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 255, 20, 20)"},
        },
    ),
    (
        Reshape1377,
        [((1, 255, 20, 20), torch.float32)],
        {
            "model_name": ["pt_yolo_v5_yolov5n_imgcls_torchhub_320x320"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1, 255, 400)"},
        },
    ),
    (
        Reshape1378,
        [((1, 1, 255, 400), torch.float32)],
        {
            "model_name": ["pt_yolo_v5_yolov5n_imgcls_torchhub_320x320"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 3, 85, 400)"},
        },
    ),
    (
        Reshape1379,
        [((1, 3, 400, 85), torch.float32)],
        {
            "model_name": ["pt_yolo_v5_yolov5n_imgcls_torchhub_320x320"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1200, 85)"},
        },
    ),
    (
        Reshape1380,
        [((1, 3, 85, 10, 10), torch.float32)],
        {
            "model_name": ["pt_yolo_v5_yolov5n_imgcls_torchhub_320x320"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 255, 10, 10)"},
        },
    ),
    (
        Reshape1381,
        [((1, 255, 10, 10), torch.float32)],
        {
            "model_name": ["pt_yolo_v5_yolov5n_imgcls_torchhub_320x320"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1, 255, 100)"},
        },
    ),
    (
        Reshape1382,
        [((1, 1, 255, 100), torch.float32)],
        {
            "model_name": ["pt_yolo_v5_yolov5n_imgcls_torchhub_320x320"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 3, 85, 100)"},
        },
    ),
    (
        Reshape1383,
        [((1, 3, 100, 85), torch.float32)],
        {
            "model_name": ["pt_yolo_v5_yolov5n_imgcls_torchhub_320x320"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 300, 85)"},
        },
    ),
]


@pytest.mark.nightly_models_ops
@pytest.mark.parametrize("forge_module_and_shapes_dtypes", forge_modules_and_shapes_dtypes_list, ids=ids_func)
def test_module(forge_module_and_shapes_dtypes, forge_property_recorder):
    forge_property_recorder("tags.op_name", "Reshape")

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
