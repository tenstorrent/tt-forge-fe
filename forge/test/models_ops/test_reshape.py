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
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(6, 768))
        return reshape_output_1


class Reshape1(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 6, 12, 64))
        return reshape_output_1


class Reshape2(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 6, 768))
        return reshape_output_1


class Reshape3(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 6, 64))
        return reshape_output_1


class Reshape4(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 64, 6))
        return reshape_output_1


class Reshape5(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 6, 6))
        return reshape_output_1


class Reshape6(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 6, 6))
        return reshape_output_1


class Reshape7(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 6, 64))
        return reshape_output_1


class Reshape8(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 768))
        return reshape_output_1


class Reshape9(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 12, 64))
        return reshape_output_1


class Reshape10(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(384, 1024))
        return reshape_output_1


class Reshape11(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 384, 16, 64))
        return reshape_output_1


class Reshape12(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 384, 1024))
        return reshape_output_1


class Reshape13(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 384, 64))
        return reshape_output_1


class Reshape14(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 64, 384))
        return reshape_output_1


class Reshape15(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 384, 384))
        return reshape_output_1


class Reshape16(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 384, 384))
        return reshape_output_1


class Reshape17(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 384, 64))
        return reshape_output_1


class Reshape18(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 384, 1))
        return reshape_output_1


class Reshape19(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(128, 768))
        return reshape_output_1


class Reshape20(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 12, 64))
        return reshape_output_1


class Reshape21(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 768))
        return reshape_output_1


class Reshape22(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 128, 64))
        return reshape_output_1


class Reshape23(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 64, 128))
        return reshape_output_1


class Reshape24(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 768, 128, 1))
        return reshape_output_1


class Reshape25(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 128, 128))
        return reshape_output_1


class Reshape26(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 128, 128))
        return reshape_output_1


class Reshape27(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 128, 64))
        return reshape_output_1


class Reshape28(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 768, 1))
        return reshape_output_1


class Reshape29(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(13, 384))
        return reshape_output_1


class Reshape30(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 13, 12, 32))
        return reshape_output_1


class Reshape31(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 13, 384))
        return reshape_output_1


class Reshape32(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 13, 32))
        return reshape_output_1


class Reshape33(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 32, 13))
        return reshape_output_1


class Reshape34(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 13, 13))
        return reshape_output_1


class Reshape35(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 13, 13))
        return reshape_output_1


class Reshape36(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 13, 32))
        return reshape_output_1


class Reshape37(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 384))
        return reshape_output_1


class Reshape38(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 6, 64))
        return reshape_output_1


class Reshape39(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(100, 256))
        return reshape_output_1


class Reshape40(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 100, 8, 32))
        return reshape_output_1


class Reshape41(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 100, 256))
        return reshape_output_1


class Reshape42(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 100, 32))
        return reshape_output_1


class Reshape43(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 100, 100))
        return reshape_output_1


class Reshape44(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 100, 32))
        return reshape_output_1


class Reshape45(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 280))
        return reshape_output_1


class Reshape46(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 32, 280))
        return reshape_output_1


class Reshape47(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(280, 256))
        return reshape_output_1


class Reshape48(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 280, 8, 32))
        return reshape_output_1


class Reshape49(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 280, 256))
        return reshape_output_1


class Reshape50(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 280, 32))
        return reshape_output_1


class Reshape51(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 280, 280))
        return reshape_output_1


class Reshape52(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 280, 280))
        return reshape_output_1


class Reshape53(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 280, 32))
        return reshape_output_1


class Reshape54(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 100, 280))
        return reshape_output_1


class Reshape55(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 100, 280))
        return reshape_output_1


class Reshape56(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 100, 92))
        return reshape_output_1


class Reshape57(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 100, 251))
        return reshape_output_1


class Reshape58(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(100, 32, 107, 160))
        return reshape_output_1


class Reshape59(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(100, 64, 54, 80))
        return reshape_output_1


class Reshape60(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(100, 128, 27, 40))
        return reshape_output_1


class Reshape61(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(100, 256, 14, 20))
        return reshape_output_1


class Reshape62(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 14, 20))
        return reshape_output_1


class Reshape63(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 100, 8, 14, 20))
        return reshape_output_1


class Reshape64(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 100, 2240))
        return reshape_output_1


class Reshape65(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(100, 8, 14, 20))
        return reshape_output_1


class Reshape66(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(100, 8, 9240))
        return reshape_output_1


class Reshape67(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(100, 264, 14, 20))
        return reshape_output_1


class Reshape68(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(100, 8, 4480))
        return reshape_output_1


class Reshape69(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(100, 128, 14, 20))
        return reshape_output_1


class Reshape70(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(100, 8, 8640))
        return reshape_output_1


class Reshape71(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(100, 64, 27, 40))
        return reshape_output_1


class Reshape72(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(100, 8, 17280))
        return reshape_output_1


class Reshape73(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(100, 32, 54, 80))
        return reshape_output_1


class Reshape74(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(100, 8, 34240))
        return reshape_output_1


class Reshape75(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(100, 16, 107, 160))
        return reshape_output_1


class Reshape76(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 100, 107, 160))
        return reshape_output_1


class Reshape77(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2048))
        return reshape_output_1


class Reshape78(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2048, 1, 1))
        return reshape_output_1


class Reshape79(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 768, 196))
        return reshape_output_1


class Reshape80(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 768, 196, 1))
        return reshape_output_1


class Reshape81(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(197, 768))
        return reshape_output_1


class Reshape82(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 197, 12, 64))
        return reshape_output_1


class Reshape83(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 197, 768))
        return reshape_output_1


class Reshape84(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 197, 64))
        return reshape_output_1


class Reshape85(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 64, 197))
        return reshape_output_1


class Reshape86(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 197, 197))
        return reshape_output_1


class Reshape87(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 197, 197))
        return reshape_output_1


class Reshape88(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 197, 64))
        return reshape_output_1


class Reshape89(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1024, 196))
        return reshape_output_1


class Reshape90(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1024, 196, 1))
        return reshape_output_1


class Reshape91(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(197, 1024))
        return reshape_output_1


class Reshape92(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 197, 16, 64))
        return reshape_output_1


class Reshape93(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 197, 1024))
        return reshape_output_1


class Reshape94(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 197, 64))
        return reshape_output_1


class Reshape95(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 64, 197))
        return reshape_output_1


class Reshape96(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 197, 197))
        return reshape_output_1


class Reshape97(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 197, 197))
        return reshape_output_1


class Reshape98(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 197, 64))
        return reshape_output_1


class Reshape99(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1024))
        return reshape_output_1


class Reshape100(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 16, 64))
        return reshape_output_1


class Reshape101(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 1, 1024))
        return reshape_output_1


class Reshape102(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 9216, 1, 1))
        return reshape_output_1


class Reshape103(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 9216))
        return reshape_output_1


class Reshape104(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1024, 1, 1))
        return reshape_output_1


class Reshape105(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1152, 1, 1))
        return reshape_output_1


class Reshape106(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1280, 1, 1))
        return reshape_output_1


class Reshape107(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 512, 1, 1))
        return reshape_output_1


class Reshape108(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 512))
        return reshape_output_1


class Reshape109(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 4, 1))
        return reshape_output_1


class Reshape110(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 1))
        return reshape_output_1


class Reshape111(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 2048))
        return reshape_output_1


class Reshape112(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 2048))
        return reshape_output_1


class Reshape113(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 1, 32, 64))
        return reshape_output_1


class Reshape114(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 1, 2048))
        return reshape_output_1


class Reshape115(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 1, 64))
        return reshape_output_1


class Reshape116(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 32, 1, 64))
        return reshape_output_1


class Reshape117(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 13))
        return reshape_output_1


class Reshape118(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(26, 768))
        return reshape_output_1


class Reshape119(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 13, 12, 64))
        return reshape_output_1


class Reshape120(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 13, 768))
        return reshape_output_1


class Reshape121(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(24, 13, 64))
        return reshape_output_1


class Reshape122(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 12, 13, 13))
        return reshape_output_1


class Reshape123(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(24, 13, 13))
        return reshape_output_1


class Reshape124(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(24, 64, 13))
        return reshape_output_1


class Reshape125(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 12, 13, 64))
        return reshape_output_1


class Reshape126(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 13, 3072))
        return reshape_output_1


class Reshape127(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(26, 3072))
        return reshape_output_1


class Reshape128(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 13, 2048))
        return reshape_output_1


class Reshape129(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 13, 32, 64))
        return reshape_output_1


class Reshape130(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(26, 2048))
        return reshape_output_1


class Reshape131(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 13, 64))
        return reshape_output_1


class Reshape132(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 32, 1, 13))
        return reshape_output_1


class Reshape133(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 1, 13))
        return reshape_output_1


class Reshape134(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 1, 8192))
        return reshape_output_1


class Reshape135(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 8192))
        return reshape_output_1


class Reshape136(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 1, 2048))
        return reshape_output_1


class Reshape137(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 1024))
        return reshape_output_1


class Reshape138(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 1024))
        return reshape_output_1


class Reshape139(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 1, 16, 64))
        return reshape_output_1


class Reshape140(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 1, 1024))
        return reshape_output_1


class Reshape141(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 1, 64))
        return reshape_output_1


class Reshape142(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 16, 1, 64))
        return reshape_output_1


class Reshape143(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 13, 1024))
        return reshape_output_1


class Reshape144(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 13, 16, 64))
        return reshape_output_1


class Reshape145(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(26, 1024))
        return reshape_output_1


class Reshape146(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 13, 64))
        return reshape_output_1


class Reshape147(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 16, 1, 13))
        return reshape_output_1


class Reshape148(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 1, 13))
        return reshape_output_1


class Reshape149(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 1, 4096))
        return reshape_output_1


class Reshape150(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 4096))
        return reshape_output_1


class Reshape151(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1536))
        return reshape_output_1


class Reshape152(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 1536))
        return reshape_output_1


class Reshape153(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 1, 24, 64))
        return reshape_output_1


class Reshape154(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 1, 1536))
        return reshape_output_1


class Reshape155(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(48, 1, 64))
        return reshape_output_1


class Reshape156(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 24, 1, 64))
        return reshape_output_1


class Reshape157(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 13, 1536))
        return reshape_output_1


class Reshape158(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 13, 24, 64))
        return reshape_output_1


class Reshape159(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(26, 1536))
        return reshape_output_1


class Reshape160(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(48, 13, 64))
        return reshape_output_1


class Reshape161(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 24, 1, 13))
        return reshape_output_1


class Reshape162(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(48, 1, 13))
        return reshape_output_1


class Reshape163(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 1, 6144))
        return reshape_output_1


class Reshape164(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 6144))
        return reshape_output_1


class Reshape165(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1))
        return reshape_output_1


class Reshape166(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 384))
        return reshape_output_1


class Reshape167(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(6, 1, 64))
        return reshape_output_1


class Reshape168(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 6, 1, 64))
        return reshape_output_1


class Reshape169(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 6, 1, 1))
        return reshape_output_1


class Reshape170(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(6, 1, 1))
        return reshape_output_1


class Reshape171(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(6, 64, 1))
        return reshape_output_1


class Reshape172(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 6, 64))
        return reshape_output_1


class Reshape173(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 80, 3000, 1))
        return reshape_output_1


class Reshape174(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(384, 80, 3, 1))
        return reshape_output_1


class Reshape175(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 384, 3000))
        return reshape_output_1


class Reshape176(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 384, 3000, 1))
        return reshape_output_1


class Reshape177(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(384, 384, 3, 1))
        return reshape_output_1


class Reshape178(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 384, 1500))
        return reshape_output_1


class Reshape179(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1500, 384))
        return reshape_output_1


class Reshape180(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1500, 6, 64))
        return reshape_output_1


class Reshape181(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1500, 384))
        return reshape_output_1


class Reshape182(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(6, 1500, 64))
        return reshape_output_1


class Reshape183(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 6, 1500, 1500))
        return reshape_output_1


class Reshape184(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(6, 1500, 1500))
        return reshape_output_1


class Reshape185(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(6, 64, 1500))
        return reshape_output_1


class Reshape186(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 6, 1500, 64))
        return reshape_output_1


class Reshape187(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 6, 1, 1500))
        return reshape_output_1


class Reshape188(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(6, 1, 1500))
        return reshape_output_1


class Reshape189(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1280))
        return reshape_output_1


class Reshape190(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 20, 64))
        return reshape_output_1


class Reshape191(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 1280))
        return reshape_output_1


class Reshape192(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(20, 1, 64))
        return reshape_output_1


class Reshape193(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 20, 1, 1))
        return reshape_output_1


class Reshape194(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(20, 1, 1))
        return reshape_output_1


class Reshape195(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(20, 64, 1))
        return reshape_output_1


class Reshape196(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 20, 1, 64))
        return reshape_output_1


class Reshape197(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1280, 80, 3, 1))
        return reshape_output_1


class Reshape198(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1280, 3000))
        return reshape_output_1


class Reshape199(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1280, 3000, 1))
        return reshape_output_1


class Reshape200(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1280, 1280, 3, 1))
        return reshape_output_1


class Reshape201(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1280, 1500))
        return reshape_output_1


class Reshape202(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1500, 1280))
        return reshape_output_1


class Reshape203(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1500, 20, 64))
        return reshape_output_1


class Reshape204(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1500, 1280))
        return reshape_output_1


class Reshape205(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(20, 1500, 64))
        return reshape_output_1


class Reshape206(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 20, 1500, 1500))
        return reshape_output_1


class Reshape207(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(20, 1500, 1500))
        return reshape_output_1


class Reshape208(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(20, 64, 1500))
        return reshape_output_1


class Reshape209(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 20, 1500, 64))
        return reshape_output_1


class Reshape210(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 20, 1, 1500))
        return reshape_output_1


class Reshape211(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(20, 1, 1500))
        return reshape_output_1


class Reshape212(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 768))
        return reshape_output_1


class Reshape213(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 1, 64))
        return reshape_output_1


class Reshape214(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 1, 1))
        return reshape_output_1


class Reshape215(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 1, 1))
        return reshape_output_1


class Reshape216(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 64, 1))
        return reshape_output_1


class Reshape217(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 1, 64))
        return reshape_output_1


class Reshape218(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(768, 80, 3, 1))
        return reshape_output_1


class Reshape219(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 768, 3000))
        return reshape_output_1


class Reshape220(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 768, 3000, 1))
        return reshape_output_1


class Reshape221(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(768, 768, 3, 1))
        return reshape_output_1


class Reshape222(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 768, 1500))
        return reshape_output_1


class Reshape223(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1500, 768))
        return reshape_output_1


class Reshape224(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1500, 12, 64))
        return reshape_output_1


class Reshape225(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1500, 768))
        return reshape_output_1


class Reshape226(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 1500, 64))
        return reshape_output_1


class Reshape227(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 1500, 1500))
        return reshape_output_1


class Reshape228(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 1500, 1500))
        return reshape_output_1


class Reshape229(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 64, 1500))
        return reshape_output_1


class Reshape230(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 1500, 64))
        return reshape_output_1


class Reshape231(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 1, 1500))
        return reshape_output_1


class Reshape232(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 1, 1500))
        return reshape_output_1


class Reshape233(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 1, 64))
        return reshape_output_1


class Reshape234(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 1, 1))
        return reshape_output_1


class Reshape235(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 1, 1))
        return reshape_output_1


class Reshape236(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 4))
        return reshape_output_1


class Reshape237(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 64, 1))
        return reshape_output_1


class Reshape238(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 1, 64))
        return reshape_output_1


class Reshape239(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1024, 80, 3, 1))
        return reshape_output_1


class Reshape240(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1024, 3000))
        return reshape_output_1


class Reshape241(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1024, 3000, 1))
        return reshape_output_1


class Reshape242(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1024, 1024, 3, 1))
        return reshape_output_1


class Reshape243(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1024, 1500))
        return reshape_output_1


class Reshape244(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1500, 1024))
        return reshape_output_1


class Reshape245(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1500, 16, 64))
        return reshape_output_1


class Reshape246(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1500, 1024))
        return reshape_output_1


class Reshape247(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 1500, 64))
        return reshape_output_1


class Reshape248(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 1500, 1500))
        return reshape_output_1


class Reshape249(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 1500, 1500))
        return reshape_output_1


class Reshape250(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 64, 1500))
        return reshape_output_1


class Reshape251(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 1500, 64))
        return reshape_output_1


class Reshape252(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 1, 1500))
        return reshape_output_1


class Reshape253(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 1, 1500))
        return reshape_output_1


class Reshape254(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 8, 64))
        return reshape_output_1


class Reshape255(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 1, 512))
        return reshape_output_1


class Reshape256(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 512))
        return reshape_output_1


class Reshape257(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 1, 64))
        return reshape_output_1


class Reshape258(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 1, 1))
        return reshape_output_1


class Reshape259(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 1, 1))
        return reshape_output_1


class Reshape260(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 64, 1))
        return reshape_output_1


class Reshape261(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 1, 64))
        return reshape_output_1


class Reshape262(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(512, 80, 3, 1))
        return reshape_output_1


class Reshape263(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 512, 3000))
        return reshape_output_1


class Reshape264(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 512, 3000, 1))
        return reshape_output_1


class Reshape265(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(512, 512, 3, 1))
        return reshape_output_1


class Reshape266(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 512, 1500))
        return reshape_output_1


class Reshape267(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1500, 512))
        return reshape_output_1


class Reshape268(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1500, 8, 64))
        return reshape_output_1


class Reshape269(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1500, 512))
        return reshape_output_1


class Reshape270(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 1500, 64))
        return reshape_output_1


class Reshape271(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 1500, 1500))
        return reshape_output_1


class Reshape272(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 1500, 1500))
        return reshape_output_1


class Reshape273(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 64, 1500))
        return reshape_output_1


class Reshape274(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 1500, 64))
        return reshape_output_1


class Reshape275(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 1, 1500))
        return reshape_output_1


class Reshape276(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 1, 1500))
        return reshape_output_1


class Reshape277(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2))
        return reshape_output_1


class Reshape278(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 1280))
        return reshape_output_1


class Reshape279(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2, 20, 64))
        return reshape_output_1


class Reshape280(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2, 1280))
        return reshape_output_1


class Reshape281(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(20, 2, 64))
        return reshape_output_1


class Reshape282(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 20, 2, 2))
        return reshape_output_1


class Reshape283(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(20, 2, 2))
        return reshape_output_1


class Reshape284(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(20, 64, 2))
        return reshape_output_1


class Reshape285(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 20, 2, 64))
        return reshape_output_1


class Reshape286(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 20, 2, 1500))
        return reshape_output_1


class Reshape287(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(20, 2, 1500))
        return reshape_output_1


class Reshape288(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 7))
        return reshape_output_1


class Reshape289(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(14, 512))
        return reshape_output_1


class Reshape290(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 7, 8, 64))
        return reshape_output_1


class Reshape291(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 7, 512))
        return reshape_output_1


class Reshape292(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 7, 64))
        return reshape_output_1


class Reshape293(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 8, 7, 7))
        return reshape_output_1


class Reshape294(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 7, 7))
        return reshape_output_1


class Reshape295(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 8, 7, 64))
        return reshape_output_1


class Reshape296(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 7, 2048))
        return reshape_output_1


class Reshape297(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(14, 2048))
        return reshape_output_1


class Reshape298(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(588, 2048))
        return reshape_output_1


class Reshape299(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 588, 16, 128))
        return reshape_output_1


class Reshape300(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 588, 2048))
        return reshape_output_1


class Reshape301(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 588, 128))
        return reshape_output_1


class Reshape302(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 588, 588))
        return reshape_output_1


class Reshape303(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 588, 588))
        return reshape_output_1


class Reshape304(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 128, 588))
        return reshape_output_1


class Reshape305(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 588, 128))
        return reshape_output_1


class Reshape306(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 588, 5504))
        return reshape_output_1


class Reshape307(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(39, 4096))
        return reshape_output_1


class Reshape308(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 39, 32, 128))
        return reshape_output_1


class Reshape309(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 39, 4096))
        return reshape_output_1


class Reshape310(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 39, 128))
        return reshape_output_1


class Reshape311(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 39, 39))
        return reshape_output_1


class Reshape312(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 39, 39))
        return reshape_output_1


class Reshape313(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 128, 39))
        return reshape_output_1


class Reshape314(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 39, 128))
        return reshape_output_1


class Reshape315(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 39, 11008))
        return reshape_output_1


class Reshape316(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2441216,))
        return reshape_output_1


class Reshape317(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(596, 4096))
        return reshape_output_1


class Reshape318(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1024, 576, 1))
        return reshape_output_1


class Reshape319(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(577, 1024))
        return reshape_output_1


class Reshape320(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 577, 16, 64))
        return reshape_output_1


class Reshape321(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 577, 1024))
        return reshape_output_1


class Reshape322(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 577, 64))
        return reshape_output_1


class Reshape323(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 577, 64))
        return reshape_output_1


class Reshape324(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2359296,))
        return reshape_output_1


class Reshape325(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 596, 4096))
        return reshape_output_1


class Reshape326(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 596, 32, 128))
        return reshape_output_1


class Reshape327(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 596, 128))
        return reshape_output_1


class Reshape328(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 596, 596))
        return reshape_output_1


class Reshape329(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 596, 596))
        return reshape_output_1


class Reshape330(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 128, 596))
        return reshape_output_1


class Reshape331(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 596, 128))
        return reshape_output_1


class Reshape332(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 596, 11008))
        return reshape_output_1


class Reshape333(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(204, 768))
        return reshape_output_1


class Reshape334(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 204, 12, 64))
        return reshape_output_1


class Reshape335(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 204, 768))
        return reshape_output_1


class Reshape336(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 204, 64))
        return reshape_output_1


class Reshape337(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 204, 204))
        return reshape_output_1


class Reshape338(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 204, 204))
        return reshape_output_1


class Reshape339(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 64, 204))
        return reshape_output_1


class Reshape340(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 204, 64))
        return reshape_output_1


class Reshape341(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(201, 768))
        return reshape_output_1


class Reshape342(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 201, 12, 64))
        return reshape_output_1


class Reshape343(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 201, 768))
        return reshape_output_1


class Reshape344(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 201, 64))
        return reshape_output_1


class Reshape345(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 201, 201))
        return reshape_output_1


class Reshape346(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 201, 201))
        return reshape_output_1


class Reshape347(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 64, 201))
        return reshape_output_1


class Reshape348(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 201, 64))
        return reshape_output_1


class Reshape349(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(128, 2048))
        return reshape_output_1


class Reshape350(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 16, 128))
        return reshape_output_1


class Reshape351(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 2048))
        return reshape_output_1


class Reshape352(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 128, 128))
        return reshape_output_1


class Reshape353(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 128, 128))
        return reshape_output_1


class Reshape354(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 2048, 1))
        return reshape_output_1


class Reshape355(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(128, 1024))
        return reshape_output_1


class Reshape356(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 16, 64))
        return reshape_output_1


class Reshape357(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 1024))
        return reshape_output_1


class Reshape358(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 8, 128))
        return reshape_output_1


class Reshape359(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 128, 64))
        return reshape_output_1


class Reshape360(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 64, 128))
        return reshape_output_1


class Reshape361(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 128, 64))
        return reshape_output_1


class Reshape362(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 1024, 1))
        return reshape_output_1


class Reshape363(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(14, 768))
        return reshape_output_1


class Reshape364(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 14, 12, 64))
        return reshape_output_1


class Reshape365(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 14, 768))
        return reshape_output_1


class Reshape366(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 14, 64))
        return reshape_output_1


class Reshape367(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 14, 14))
        return reshape_output_1


class Reshape368(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 14, 14))
        return reshape_output_1


class Reshape369(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 64, 14))
        return reshape_output_1


class Reshape370(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 14, 64))
        return reshape_output_1


class Reshape371(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 14, 768, 1))
        return reshape_output_1


class Reshape372(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 14, 1))
        return reshape_output_1


class Reshape373(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(9, 768))
        return reshape_output_1


class Reshape374(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 9, 12, 64))
        return reshape_output_1


class Reshape375(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 9, 768))
        return reshape_output_1


class Reshape376(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 9, 64))
        return reshape_output_1


class Reshape377(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 9, 9))
        return reshape_output_1


class Reshape378(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 9, 9))
        return reshape_output_1


class Reshape379(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 64, 9))
        return reshape_output_1


class Reshape380(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 9, 64))
        return reshape_output_1


class Reshape381(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 9, 768, 1))
        return reshape_output_1


class Reshape382(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(128, 4096))
        return reshape_output_1


class Reshape383(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 64, 64))
        return reshape_output_1


class Reshape384(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 4096))
        return reshape_output_1


class Reshape385(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 32, 128))
        return reshape_output_1


class Reshape386(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 128, 64))
        return reshape_output_1


class Reshape387(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 64, 128, 128))
        return reshape_output_1


class Reshape388(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 128, 128))
        return reshape_output_1


class Reshape389(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 64, 16384, 1))
        return reshape_output_1


class Reshape390(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 64, 128))
        return reshape_output_1


class Reshape391(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 64, 128, 64))
        return reshape_output_1


class Reshape392(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 4096, 1))
        return reshape_output_1


class Reshape393(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(256, 1024))
        return reshape_output_1


class Reshape394(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 16, 64))
        return reshape_output_1


class Reshape395(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 32, 32))
        return reshape_output_1


class Reshape396(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 1024))
        return reshape_output_1


class Reshape397(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 4, 256))
        return reshape_output_1


class Reshape398(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 8, 128))
        return reshape_output_1


class Reshape399(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 256, 64))
        return reshape_output_1


class Reshape400(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 256, 256))
        return reshape_output_1


class Reshape401(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 256, 256))
        return reshape_output_1


class Reshape402(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 256, 64))
        return reshape_output_1


class Reshape403(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256))
        return reshape_output_1


class Reshape404(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 16, 3, 96))
        return reshape_output_1


class Reshape405(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 16, 96))
        return reshape_output_1


class Reshape406(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 32, 96))
        return reshape_output_1


class Reshape407(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 1, 32))
        return reshape_output_1


class Reshape408(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 32, 32))
        return reshape_output_1


class Reshape409(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 32, 32))
        return reshape_output_1


class Reshape410(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 32, 96))
        return reshape_output_1


class Reshape411(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 1536))
        return reshape_output_1


class Reshape412(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 1536))
        return reshape_output_1


class Reshape413(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 16, 32, 1))
        return reshape_output_1


class Reshape414(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 64, 256))
        return reshape_output_1


class Reshape415(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 4096))
        return reshape_output_1


class Reshape416(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 32, 128))
        return reshape_output_1


class Reshape417(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(384, 768))
        return reshape_output_1


class Reshape418(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 384, 12, 64))
        return reshape_output_1


class Reshape419(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 384, 768))
        return reshape_output_1


class Reshape420(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 384, 64))
        return reshape_output_1


class Reshape421(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 384, 384))
        return reshape_output_1


class Reshape422(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 1, 384))
        return reshape_output_1


class Reshape423(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 384, 384))
        return reshape_output_1


class Reshape424(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 64, 384))
        return reshape_output_1


class Reshape425(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 384, 64))
        return reshape_output_1


class Reshape426(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 1, 128))
        return reshape_output_1


class Reshape427(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 1))
        return reshape_output_1


class Reshape428(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128))
        return reshape_output_1


class Reshape429(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1,))
        return reshape_output_1


class Reshape430(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(10, 2048))
        return reshape_output_1


class Reshape431(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 10, 8, 256))
        return reshape_output_1


class Reshape432(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 10, 2048))
        return reshape_output_1


class Reshape433(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 10, 256))
        return reshape_output_1


class Reshape434(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 10, 4, 256))
        return reshape_output_1


class Reshape435(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 10, 256))
        return reshape_output_1


class Reshape436(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 10, 10))
        return reshape_output_1


class Reshape437(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 10, 10))
        return reshape_output_1


class Reshape438(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 256, 10))
        return reshape_output_1


class Reshape439(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 10, 8192))
        return reshape_output_1


class Reshape440(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(6, 4544))
        return reshape_output_1


class Reshape441(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 6, 18176))
        return reshape_output_1


class Reshape442(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 6, 73, 64))
        return reshape_output_1


class Reshape443(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 71, 6, 64))
        return reshape_output_1


class Reshape444(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(71, 6, 64))
        return reshape_output_1


class Reshape445(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 71, 6, 6))
        return reshape_output_1


class Reshape446(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(71, 6, 6))
        return reshape_output_1


class Reshape447(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 6, 4544))
        return reshape_output_1


class Reshape448(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(10, 3072))
        return reshape_output_1


class Reshape449(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 10, 12, 256))
        return reshape_output_1


class Reshape450(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 10, 3072))
        return reshape_output_1


class Reshape451(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 10, 256))
        return reshape_output_1


class Reshape452(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 10, 256))
        return reshape_output_1


class Reshape453(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 10, 10))
        return reshape_output_1


class Reshape454(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 10, 10))
        return reshape_output_1


class Reshape455(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 256, 10))
        return reshape_output_1


class Reshape456(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 10, 9216))
        return reshape_output_1


class Reshape457(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 10, 23040))
        return reshape_output_1


class Reshape458(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 334, 64, 3, 64))
        return reshape_output_1


class Reshape459(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 334, 64, 64))
        return reshape_output_1


class Reshape460(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 334, 64))
        return reshape_output_1


class Reshape461(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 64, 334, 334))
        return reshape_output_1


class Reshape462(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 334, 334))
        return reshape_output_1


class Reshape463(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 64, 334))
        return reshape_output_1


class Reshape464(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 64, 334, 64))
        return reshape_output_1


class Reshape465(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(334, 4096))
        return reshape_output_1


class Reshape466(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 334, 4096))
        return reshape_output_1


class Reshape467(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(207, 3584))
        return reshape_output_1


class Reshape468(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 207, 16, 256))
        return reshape_output_1


class Reshape469(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 207, 256))
        return reshape_output_1


class Reshape470(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 207, 8, 256))
        return reshape_output_1


class Reshape471(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 207, 256))
        return reshape_output_1


class Reshape472(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 207, 207))
        return reshape_output_1


class Reshape473(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 207, 207))
        return reshape_output_1


class Reshape474(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 256, 207))
        return reshape_output_1


class Reshape475(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(207, 4096))
        return reshape_output_1


class Reshape476(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 207, 3584))
        return reshape_output_1


class Reshape477(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 207, 14336))
        return reshape_output_1


class Reshape478(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(7, 2048))
        return reshape_output_1


class Reshape479(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 7, 8, 256))
        return reshape_output_1


class Reshape480(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 7, 2048))
        return reshape_output_1


class Reshape481(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 7, 256))
        return reshape_output_1


class Reshape482(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 7, 1, 256))
        return reshape_output_1


class Reshape483(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 7, 256))
        return reshape_output_1


class Reshape484(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 7, 7))
        return reshape_output_1


class Reshape485(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 7, 7))
        return reshape_output_1


class Reshape486(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 256, 7))
        return reshape_output_1


class Reshape487(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 7, 16384))
        return reshape_output_1


class Reshape488(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(207, 2304))
        return reshape_output_1


class Reshape489(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 207, 256))
        return reshape_output_1


class Reshape490(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 207, 4, 256))
        return reshape_output_1


class Reshape491(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 207, 256))
        return reshape_output_1


class Reshape492(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 207, 207))
        return reshape_output_1


class Reshape493(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 207, 207))
        return reshape_output_1


class Reshape494(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 256, 207))
        return reshape_output_1


class Reshape495(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(207, 2048))
        return reshape_output_1


class Reshape496(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 207, 2304))
        return reshape_output_1


class Reshape497(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 207, 9216))
        return reshape_output_1


class Reshape498(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(107, 2048))
        return reshape_output_1


class Reshape499(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 107, 8, 256))
        return reshape_output_1


class Reshape500(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 107, 2048))
        return reshape_output_1


class Reshape501(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 107, 256))
        return reshape_output_1


class Reshape502(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 107, 1, 256))
        return reshape_output_1


class Reshape503(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 107, 256))
        return reshape_output_1


class Reshape504(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 107, 107))
        return reshape_output_1


class Reshape505(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 107, 107))
        return reshape_output_1


class Reshape506(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 256, 107))
        return reshape_output_1


class Reshape507(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 107, 16384))
        return reshape_output_1


class Reshape508(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(107, 3072))
        return reshape_output_1


class Reshape509(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 107, 16, 256))
        return reshape_output_1


class Reshape510(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 107, 256))
        return reshape_output_1


class Reshape511(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 107, 107))
        return reshape_output_1


class Reshape512(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 107, 107))
        return reshape_output_1


class Reshape513(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 256, 107))
        return reshape_output_1


class Reshape514(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 107, 256))
        return reshape_output_1


class Reshape515(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(107, 4096))
        return reshape_output_1


class Reshape516(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 107, 3072))
        return reshape_output_1


class Reshape517(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 107, 24576))
        return reshape_output_1


class Reshape518(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 7))
        return reshape_output_1


class Reshape519(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(7, 768))
        return reshape_output_1


class Reshape520(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 7, 12, 64))
        return reshape_output_1


class Reshape521(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 7, 768))
        return reshape_output_1


class Reshape522(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 7, 64))
        return reshape_output_1


class Reshape523(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 7, 7))
        return reshape_output_1


class Reshape524(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 7, 7))
        return reshape_output_1


class Reshape525(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 64, 7))
        return reshape_output_1


class Reshape526(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 7, 64))
        return reshape_output_1


class Reshape527(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 7, 3072))
        return reshape_output_1


class Reshape528(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(7, 3072))
        return reshape_output_1


class Reshape529(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(7, 2))
        return reshape_output_1


class Reshape530(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(256, 768))
        return reshape_output_1


class Reshape531(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 12, 64))
        return reshape_output_1


class Reshape532(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 8, 96))
        return reshape_output_1


class Reshape533(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 768))
        return reshape_output_1


class Reshape534(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 256, 64))
        return reshape_output_1


class Reshape535(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 256, 256))
        return reshape_output_1


class Reshape536(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 256, 256))
        return reshape_output_1


class Reshape537(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 64, 256))
        return reshape_output_1


class Reshape538(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 256, 64))
        return reshape_output_1


class Reshape539(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 3072))
        return reshape_output_1


class Reshape540(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(256, 3072))
        return reshape_output_1


class Reshape541(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 32, 96))
        return reshape_output_1


class Reshape542(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 2048))
        return reshape_output_1


class Reshape543(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 32, 64))
        return reshape_output_1


class Reshape544(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 16, 128))
        return reshape_output_1


class Reshape545(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 2048))
        return reshape_output_1


class Reshape546(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 32, 128))
        return reshape_output_1


class Reshape547(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 128, 32))
        return reshape_output_1


class Reshape548(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 32, 128))
        return reshape_output_1


class Reshape549(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(256, 2560))
        return reshape_output_1


class Reshape550(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 32, 80))
        return reshape_output_1


class Reshape551(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 20, 128))
        return reshape_output_1


class Reshape552(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 2560))
        return reshape_output_1


class Reshape553(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(20, 256, 128))
        return reshape_output_1


class Reshape554(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 20, 256, 256))
        return reshape_output_1


class Reshape555(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(20, 256, 256))
        return reshape_output_1


class Reshape556(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(20, 128, 256))
        return reshape_output_1


class Reshape557(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 20, 256, 128))
        return reshape_output_1


class Reshape558(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 768))
        return reshape_output_1


class Reshape559(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 12, 64))
        return reshape_output_1


class Reshape560(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 768))
        return reshape_output_1


class Reshape561(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 32, 64))
        return reshape_output_1


class Reshape562(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 32, 32))
        return reshape_output_1


class Reshape563(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 32, 32))
        return reshape_output_1


class Reshape564(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 64, 32))
        return reshape_output_1


class Reshape565(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 32, 64))
        return reshape_output_1


class Reshape566(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 2560))
        return reshape_output_1


class Reshape567(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 20, 128))
        return reshape_output_1


class Reshape568(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 2560))
        return reshape_output_1


class Reshape569(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(20, 32, 128))
        return reshape_output_1


class Reshape570(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 20, 32, 32))
        return reshape_output_1


class Reshape571(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(20, 32, 32))
        return reshape_output_1


class Reshape572(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(20, 128, 32))
        return reshape_output_1


class Reshape573(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 20, 32, 128))
        return reshape_output_1


class Reshape574(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(256, 2048))
        return reshape_output_1


class Reshape575(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 32, 64))
        return reshape_output_1


class Reshape576(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 16, 128))
        return reshape_output_1


class Reshape577(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 2048))
        return reshape_output_1


class Reshape578(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 256, 128))
        return reshape_output_1


class Reshape579(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 128, 256))
        return reshape_output_1


class Reshape580(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 256, 128))
        return reshape_output_1


class Reshape581(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(256, 4096))
        return reshape_output_1


class Reshape582(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 64, 64))
        return reshape_output_1


class Reshape583(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 256, 128))
        return reshape_output_1


class Reshape584(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 256, 128))
        return reshape_output_1


class Reshape585(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 256, 256))
        return reshape_output_1


class Reshape586(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 256, 256))
        return reshape_output_1


class Reshape587(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 128, 256))
        return reshape_output_1


class Reshape588(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 14336))
        return reshape_output_1


class Reshape589(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 256, 64))
        return reshape_output_1


class Reshape590(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 8, 64))
        return reshape_output_1


class Reshape591(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(256, 512))
        return reshape_output_1


class Reshape592(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 512))
        return reshape_output_1


class Reshape593(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 256, 64))
        return reshape_output_1


class Reshape594(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 64, 256))
        return reshape_output_1


class Reshape595(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 8192))
        return reshape_output_1


class Reshape596(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(4, 4096))
        return reshape_output_1


class Reshape597(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 32, 128))
        return reshape_output_1


class Reshape598(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 4096))
        return reshape_output_1


class Reshape599(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 4, 128))
        return reshape_output_1


class Reshape600(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 8, 128))
        return reshape_output_1


class Reshape601(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 4, 128))
        return reshape_output_1


class Reshape602(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 4, 4))
        return reshape_output_1


class Reshape603(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 4, 4))
        return reshape_output_1


class Reshape604(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 128, 4))
        return reshape_output_1


class Reshape605(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 14336))
        return reshape_output_1


class Reshape606(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 11008))
        return reshape_output_1


class Reshape607(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(4, 2))
        return reshape_output_1


class Reshape608(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(4, 3072))
        return reshape_output_1


class Reshape609(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 24, 128))
        return reshape_output_1


class Reshape610(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 3072))
        return reshape_output_1


class Reshape611(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(24, 4, 128))
        return reshape_output_1


class Reshape612(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 24, 4, 128))
        return reshape_output_1


class Reshape613(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 24, 4, 4))
        return reshape_output_1


class Reshape614(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(24, 4, 4))
        return reshape_output_1


class Reshape615(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(24, 128, 4))
        return reshape_output_1


class Reshape616(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 8192))
        return reshape_output_1


class Reshape617(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(4, 2048))
        return reshape_output_1


class Reshape618(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 32, 64))
        return reshape_output_1


class Reshape619(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 2048))
        return reshape_output_1


class Reshape620(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 4, 64))
        return reshape_output_1


class Reshape621(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 8, 64))
        return reshape_output_1


class Reshape622(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 4, 64))
        return reshape_output_1


class Reshape623(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 64, 4))
        return reshape_output_1


class Reshape624(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 3072))
        return reshape_output_1


class Reshape625(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 24, 128))
        return reshape_output_1


class Reshape626(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 3072))
        return reshape_output_1


class Reshape627(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(24, 32, 128))
        return reshape_output_1


class Reshape628(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 8, 128))
        return reshape_output_1


class Reshape629(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 1024))
        return reshape_output_1


class Reshape630(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 24, 32, 128))
        return reshape_output_1


class Reshape631(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 24, 32, 32))
        return reshape_output_1


class Reshape632(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(24, 32, 32))
        return reshape_output_1


class Reshape633(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(24, 128, 32))
        return reshape_output_1


class Reshape634(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 8192))
        return reshape_output_1


class Reshape635(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 4096))
        return reshape_output_1


class Reshape636(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 32, 128))
        return reshape_output_1


class Reshape637(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 4096))
        return reshape_output_1


class Reshape638(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 32, 128))
        return reshape_output_1


class Reshape639(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 32, 32))
        return reshape_output_1


class Reshape640(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 32, 32))
        return reshape_output_1


class Reshape641(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 128, 32))
        return reshape_output_1


class Reshape642(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 11008))
        return reshape_output_1


class Reshape643(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 128, 128))
        return reshape_output_1


class Reshape644(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 16384, 1))
        return reshape_output_1


class Reshape645(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 128, 128))
        return reshape_output_1


class Reshape646(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 14336))
        return reshape_output_1


class Reshape647(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32))
        return reshape_output_1


class Reshape648(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 32, 64))
        return reshape_output_1


class Reshape649(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 1))
        return reshape_output_1


class Reshape650(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 1024))
        return reshape_output_1


class Reshape651(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 16, 64))
        return reshape_output_1


class Reshape652(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 32, 64))
        return reshape_output_1


class Reshape653(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 32, 64))
        return reshape_output_1


class Reshape654(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 512))
        return reshape_output_1


class Reshape655(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 2))
        return reshape_output_1


class Reshape656(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 50272))
        return reshape_output_1


class Reshape657(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 512, 1, 512))
        return reshape_output_1


class Reshape658(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 50176, 256))
        return reshape_output_1


class Reshape659(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(50176, 512))
        return reshape_output_1


class Reshape660(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 50176, 1, 512))
        return reshape_output_1


class Reshape661(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 50176, 512))
        return reshape_output_1


class Reshape662(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 512, 50176))
        return reshape_output_1


class Reshape663(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 512, 50176))
        return reshape_output_1


class Reshape664(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(512, 1024))
        return reshape_output_1


class Reshape665(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 512, 8, 128))
        return reshape_output_1


class Reshape666(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 512, 1, 1024))
        return reshape_output_1


class Reshape667(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 512, 1024))
        return reshape_output_1


class Reshape668(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 512, 128))
        return reshape_output_1


class Reshape669(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 512, 512))
        return reshape_output_1


class Reshape670(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 512, 512))
        return reshape_output_1


class Reshape671(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 128, 512))
        return reshape_output_1


class Reshape672(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 512, 128))
        return reshape_output_1


class Reshape673(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1024, 512))
        return reshape_output_1


class Reshape674(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1000))
        return reshape_output_1


class Reshape675(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 512, 1, 322))
        return reshape_output_1


class Reshape676(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 3025, 64))
        return reshape_output_1


class Reshape677(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(3025, 322))
        return reshape_output_1


class Reshape678(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 3025, 1, 322))
        return reshape_output_1


class Reshape679(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 3025, 322))
        return reshape_output_1


class Reshape680(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 512, 3025))
        return reshape_output_1


class Reshape681(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 512, 3025))
        return reshape_output_1


class Reshape682(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 322, 3025))
        return reshape_output_1


class Reshape683(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 512, 1, 261))
        return reshape_output_1


class Reshape684(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 50176, 3))
        return reshape_output_1


class Reshape685(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(50176, 261))
        return reshape_output_1


class Reshape686(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 50176, 1, 261))
        return reshape_output_1


class Reshape687(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 50176, 261))
        return reshape_output_1


class Reshape688(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 261, 50176))
        return reshape_output_1


class Reshape689(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2048, 8, 32))
        return reshape_output_1


class Reshape690(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2048, 16, 16))
        return reshape_output_1


class Reshape691(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 2048, 32))
        return reshape_output_1


class Reshape692(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 8, 32))
        return reshape_output_1


class Reshape693(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 16, 16))
        return reshape_output_1


class Reshape694(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(256, 256))
        return reshape_output_1


class Reshape695(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 256))
        return reshape_output_1


class Reshape696(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 16, 256))
        return reshape_output_1


class Reshape697(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 256, 32))
        return reshape_output_1


class Reshape698(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2048, 768))
        return reshape_output_1


class Reshape699(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2048, 256))
        return reshape_output_1


class Reshape700(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 256, 2048))
        return reshape_output_1


class Reshape701(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 256, 2048))
        return reshape_output_1


class Reshape702(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2048, 1280))
        return reshape_output_1


class Reshape703(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2048, 8, 160))
        return reshape_output_1


class Reshape704(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 160, 2048))
        return reshape_output_1


class Reshape705(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 256, 160))
        return reshape_output_1


class Reshape706(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(256, 1280))
        return reshape_output_1


class Reshape707(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 1280))
        return reshape_output_1


class Reshape708(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 8, 160))
        return reshape_output_1


class Reshape709(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 256, 256))
        return reshape_output_1


class Reshape710(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 256, 256))
        return reshape_output_1


class Reshape711(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 160, 256))
        return reshape_output_1


class Reshape712(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 2048, 256))
        return reshape_output_1


class Reshape713(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 2048, 256))
        return reshape_output_1


class Reshape714(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 96, 256))
        return reshape_output_1


class Reshape715(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 2048, 96))
        return reshape_output_1


class Reshape716(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2048, 768))
        return reshape_output_1


class Reshape717(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2048, 262))
        return reshape_output_1


class Reshape718(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 256, 80))
        return reshape_output_1


class Reshape719(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 80, 256))
        return reshape_output_1


class Reshape720(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 256, 80))
        return reshape_output_1


class Reshape721(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 10240))
        return reshape_output_1


class Reshape722(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(11, 2560))
        return reshape_output_1


class Reshape723(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 11, 32, 80))
        return reshape_output_1


class Reshape724(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 11, 2560))
        return reshape_output_1


class Reshape725(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 11, 80))
        return reshape_output_1


class Reshape726(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 11, 11))
        return reshape_output_1


class Reshape727(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 11, 11))
        return reshape_output_1


class Reshape728(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 80, 11))
        return reshape_output_1


class Reshape729(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 11, 80))
        return reshape_output_1


class Reshape730(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 11, 10240))
        return reshape_output_1


class Reshape731(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 2560))
        return reshape_output_1


class Reshape732(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 32, 80))
        return reshape_output_1


class Reshape733(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 2560))
        return reshape_output_1


class Reshape734(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 12, 80))
        return reshape_output_1


class Reshape735(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 12, 12))
        return reshape_output_1


class Reshape736(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 12, 12))
        return reshape_output_1


class Reshape737(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 80, 12))
        return reshape_output_1


class Reshape738(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 12, 80))
        return reshape_output_1


class Reshape739(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 10240))
        return reshape_output_1


class Reshape740(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 5, 32, 96))
        return reshape_output_1


class Reshape741(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(5, 3072))
        return reshape_output_1


class Reshape742(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 5, 96))
        return reshape_output_1


class Reshape743(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 5, 5))
        return reshape_output_1


class Reshape744(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 5, 5))
        return reshape_output_1


class Reshape745(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 96, 5))
        return reshape_output_1


class Reshape746(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 5, 96))
        return reshape_output_1


class Reshape747(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 5, 3072))
        return reshape_output_1


class Reshape748(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 5, 8192))
        return reshape_output_1


class Reshape749(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 13, 32, 96))
        return reshape_output_1


class Reshape750(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(13, 3072))
        return reshape_output_1


class Reshape751(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 13, 96))
        return reshape_output_1


class Reshape752(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 13, 13))
        return reshape_output_1


class Reshape753(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 13, 13))
        return reshape_output_1


class Reshape754(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 96, 13))
        return reshape_output_1


class Reshape755(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 13, 96))
        return reshape_output_1


class Reshape756(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 13, 3072))
        return reshape_output_1


class Reshape757(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 13, 8192))
        return reshape_output_1


class Reshape758(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 256, 96))
        return reshape_output_1


class Reshape759(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 96, 256))
        return reshape_output_1


class Reshape760(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 256, 96))
        return reshape_output_1


class Reshape761(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(29, 1024))
        return reshape_output_1


class Reshape762(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 29, 16, 64))
        return reshape_output_1


class Reshape763(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 29, 1024))
        return reshape_output_1


class Reshape764(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 29, 64))
        return reshape_output_1


class Reshape765(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 29, 29))
        return reshape_output_1


class Reshape766(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 29, 29))
        return reshape_output_1


class Reshape767(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 64, 29))
        return reshape_output_1


class Reshape768(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 29, 64))
        return reshape_output_1


class Reshape769(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 29, 2816))
        return reshape_output_1


class Reshape770(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(6, 1024))
        return reshape_output_1


class Reshape771(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 6, 16, 64))
        return reshape_output_1


class Reshape772(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 6, 1024))
        return reshape_output_1


class Reshape773(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 6, 64))
        return reshape_output_1


class Reshape774(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 6, 6))
        return reshape_output_1


class Reshape775(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 6, 6))
        return reshape_output_1


class Reshape776(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 64, 6))
        return reshape_output_1


class Reshape777(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 6, 64))
        return reshape_output_1


class Reshape778(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 6, 2816))
        return reshape_output_1


class Reshape779(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(35, 1536))
        return reshape_output_1


class Reshape780(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 35, 12, 128))
        return reshape_output_1


class Reshape781(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 35, 1536))
        return reshape_output_1


class Reshape782(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 35, 128))
        return reshape_output_1


class Reshape783(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 35, 256))
        return reshape_output_1


class Reshape784(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 35, 2, 128))
        return reshape_output_1


class Reshape785(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 35, 128))
        return reshape_output_1


class Reshape786(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 35, 35))
        return reshape_output_1


class Reshape787(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 35, 35))
        return reshape_output_1


class Reshape788(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 128, 35))
        return reshape_output_1


class Reshape789(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 35, 8960))
        return reshape_output_1


class Reshape790(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(35, 3584))
        return reshape_output_1


class Reshape791(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 35, 28, 128))
        return reshape_output_1


class Reshape792(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 35, 3584))
        return reshape_output_1


class Reshape793(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(28, 35, 128))
        return reshape_output_1


class Reshape794(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 35, 512))
        return reshape_output_1


class Reshape795(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 35, 4, 128))
        return reshape_output_1


class Reshape796(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 28, 35, 128))
        return reshape_output_1


class Reshape797(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 28, 35, 35))
        return reshape_output_1


class Reshape798(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(28, 35, 35))
        return reshape_output_1


class Reshape799(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(28, 128, 35))
        return reshape_output_1


class Reshape800(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 35, 18944))
        return reshape_output_1


class Reshape801(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(35, 896))
        return reshape_output_1


class Reshape802(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 35, 14, 64))
        return reshape_output_1


class Reshape803(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 35, 896))
        return reshape_output_1


class Reshape804(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(14, 35, 64))
        return reshape_output_1


class Reshape805(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 35, 128))
        return reshape_output_1


class Reshape806(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 35, 2, 64))
        return reshape_output_1


class Reshape807(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 14, 35, 64))
        return reshape_output_1


class Reshape808(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 14, 35, 35))
        return reshape_output_1


class Reshape809(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(14, 35, 35))
        return reshape_output_1


class Reshape810(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(14, 64, 35))
        return reshape_output_1


class Reshape811(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 35, 4864))
        return reshape_output_1


class Reshape812(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(35, 2048))
        return reshape_output_1


class Reshape813(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 35, 16, 128))
        return reshape_output_1


class Reshape814(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 35, 2048))
        return reshape_output_1


class Reshape815(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 35, 128))
        return reshape_output_1


class Reshape816(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 35, 128))
        return reshape_output_1


class Reshape817(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 35, 35))
        return reshape_output_1


class Reshape818(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 35, 35))
        return reshape_output_1


class Reshape819(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 128, 35))
        return reshape_output_1


class Reshape820(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 35, 11008))
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
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 39, 256))
        return reshape_output_1


class Reshape826(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 39, 2, 128))
        return reshape_output_1


class Reshape827(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 39, 128))
        return reshape_output_1


class Reshape828(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 39, 39))
        return reshape_output_1


class Reshape829(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 39, 39))
        return reshape_output_1


class Reshape830(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 128, 39))
        return reshape_output_1


class Reshape831(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(29, 2048))
        return reshape_output_1


class Reshape832(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 29, 16, 128))
        return reshape_output_1


class Reshape833(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 29, 2048))
        return reshape_output_1


class Reshape834(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 29, 128))
        return reshape_output_1


class Reshape835(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 29, 256))
        return reshape_output_1


class Reshape836(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 29, 2, 128))
        return reshape_output_1


class Reshape837(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 29, 128))
        return reshape_output_1


class Reshape838(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 128, 29))
        return reshape_output_1


class Reshape839(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 29, 11008))
        return reshape_output_1


class Reshape840(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(29, 1536))
        return reshape_output_1


class Reshape841(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 29, 12, 128))
        return reshape_output_1


class Reshape842(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 29, 1536))
        return reshape_output_1


class Reshape843(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 29, 128))
        return reshape_output_1


class Reshape844(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 29, 128))
        return reshape_output_1


class Reshape845(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 29, 29))
        return reshape_output_1


class Reshape846(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 29, 29))
        return reshape_output_1


class Reshape847(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 128, 29))
        return reshape_output_1


class Reshape848(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 29, 8960))
        return reshape_output_1


class Reshape849(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(39, 1536))
        return reshape_output_1


class Reshape850(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 39, 12, 128))
        return reshape_output_1


class Reshape851(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 39, 1536))
        return reshape_output_1


class Reshape852(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 39, 128))
        return reshape_output_1


class Reshape853(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 39, 128))
        return reshape_output_1


class Reshape854(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 39, 39))
        return reshape_output_1


class Reshape855(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 39, 39))
        return reshape_output_1


class Reshape856(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 128, 39))
        return reshape_output_1


class Reshape857(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 39, 8960))
        return reshape_output_1


class Reshape858(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(39, 896))
        return reshape_output_1


class Reshape859(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 39, 14, 64))
        return reshape_output_1


class Reshape860(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 39, 896))
        return reshape_output_1


class Reshape861(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(14, 39, 64))
        return reshape_output_1


class Reshape862(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 39, 128))
        return reshape_output_1


class Reshape863(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 39, 2, 64))
        return reshape_output_1


class Reshape864(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 14, 39, 64))
        return reshape_output_1


class Reshape865(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 14, 39, 39))
        return reshape_output_1


class Reshape866(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(14, 39, 39))
        return reshape_output_1


class Reshape867(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(14, 64, 39))
        return reshape_output_1


class Reshape868(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 39, 4864))
        return reshape_output_1


class Reshape869(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(39, 3584))
        return reshape_output_1


class Reshape870(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 39, 28, 128))
        return reshape_output_1


class Reshape871(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 39, 3584))
        return reshape_output_1


class Reshape872(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(28, 39, 128))
        return reshape_output_1


class Reshape873(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 39, 512))
        return reshape_output_1


class Reshape874(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 39, 4, 128))
        return reshape_output_1


class Reshape875(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 28, 39, 128))
        return reshape_output_1


class Reshape876(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 28, 39, 39))
        return reshape_output_1


class Reshape877(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(28, 39, 39))
        return reshape_output_1


class Reshape878(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(28, 128, 39))
        return reshape_output_1


class Reshape879(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 39, 18944))
        return reshape_output_1


class Reshape880(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(29, 896))
        return reshape_output_1


class Reshape881(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 29, 14, 64))
        return reshape_output_1


class Reshape882(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 29, 896))
        return reshape_output_1


class Reshape883(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(14, 29, 64))
        return reshape_output_1


class Reshape884(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 29, 128))
        return reshape_output_1


class Reshape885(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 29, 2, 64))
        return reshape_output_1


class Reshape886(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 14, 29, 64))
        return reshape_output_1


class Reshape887(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 14, 29, 29))
        return reshape_output_1


class Reshape888(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(14, 29, 29))
        return reshape_output_1


class Reshape889(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(14, 64, 29))
        return reshape_output_1


class Reshape890(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 29, 4864))
        return reshape_output_1


class Reshape891(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(13, 3584))
        return reshape_output_1


class Reshape892(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 13, 28, 128))
        return reshape_output_1


class Reshape893(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 13, 3584))
        return reshape_output_1


class Reshape894(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(28, 13, 128))
        return reshape_output_1


class Reshape895(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 13, 512))
        return reshape_output_1


class Reshape896(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 13, 4, 128))
        return reshape_output_1


class Reshape897(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 28, 13, 128))
        return reshape_output_1


class Reshape898(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 28, 13, 13))
        return reshape_output_1


class Reshape899(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(28, 13, 13))
        return reshape_output_1


class Reshape900(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(28, 128, 13))
        return reshape_output_1


class Reshape901(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 13, 18944))
        return reshape_output_1


class Reshape902(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(29, 3584))
        return reshape_output_1


class Reshape903(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 29, 28, 128))
        return reshape_output_1


class Reshape904(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 29, 3584))
        return reshape_output_1


class Reshape905(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(28, 29, 128))
        return reshape_output_1


class Reshape906(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 29, 512))
        return reshape_output_1


class Reshape907(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 29, 4, 128))
        return reshape_output_1


class Reshape908(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 28, 29, 128))
        return reshape_output_1


class Reshape909(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 28, 29, 29))
        return reshape_output_1


class Reshape910(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(28, 29, 29))
        return reshape_output_1


class Reshape911(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(28, 128, 29))
        return reshape_output_1


class Reshape912(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 29, 18944))
        return reshape_output_1


class Reshape913(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 64, 128))
        return reshape_output_1


class Reshape914(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(768, 768, 1, 1))
        return reshape_output_1


class Reshape915(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 768, 128))
        return reshape_output_1


class Reshape916(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 61))
        return reshape_output_1


class Reshape917(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(61, 768))
        return reshape_output_1


class Reshape918(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 61, 12, 64))
        return reshape_output_1


class Reshape919(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 61, 768))
        return reshape_output_1


class Reshape920(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 61, 64))
        return reshape_output_1


class Reshape921(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 61, 61))
        return reshape_output_1


class Reshape922(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 61, 61))
        return reshape_output_1


class Reshape923(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 64, 61))
        return reshape_output_1


class Reshape924(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 61, 64))
        return reshape_output_1


class Reshape925(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 61, 2048))
        return reshape_output_1


class Reshape926(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 1, 61))
        return reshape_output_1


class Reshape927(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 1, 61))
        return reshape_output_1


class Reshape928(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(61, 512))
        return reshape_output_1


class Reshape929(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 61, 8, 64))
        return reshape_output_1


class Reshape930(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 61, 512))
        return reshape_output_1


class Reshape931(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 61, 64))
        return reshape_output_1


class Reshape932(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 61, 61))
        return reshape_output_1


class Reshape933(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 61, 61))
        return reshape_output_1


class Reshape934(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 64, 61))
        return reshape_output_1


class Reshape935(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 61, 64))
        return reshape_output_1


class Reshape936(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 1, 61))
        return reshape_output_1


class Reshape937(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 1, 61))
        return reshape_output_1


class Reshape938(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(61, 1024))
        return reshape_output_1


class Reshape939(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 61, 16, 64))
        return reshape_output_1


class Reshape940(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 61, 1024))
        return reshape_output_1


class Reshape941(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 61, 64))
        return reshape_output_1


class Reshape942(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 61, 61))
        return reshape_output_1


class Reshape943(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 61, 61))
        return reshape_output_1


class Reshape944(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 64, 61))
        return reshape_output_1


class Reshape945(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 61, 64))
        return reshape_output_1


class Reshape946(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 1, 61))
        return reshape_output_1


class Reshape947(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 1, 61))
        return reshape_output_1


class Reshape948(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 61, 2816))
        return reshape_output_1


class Reshape949(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 2816))
        return reshape_output_1


class Reshape950(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 61, 6, 64))
        return reshape_output_1


class Reshape951(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(6, 61, 64))
        return reshape_output_1


class Reshape952(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 6, 61, 61))
        return reshape_output_1


class Reshape953(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(6, 61, 61))
        return reshape_output_1


class Reshape954(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(6, 64, 61))
        return reshape_output_1


class Reshape955(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 6, 61, 64))
        return reshape_output_1


class Reshape956(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(61, 384))
        return reshape_output_1


class Reshape957(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 6, 1, 61))
        return reshape_output_1


class Reshape958(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(6, 1, 61))
        return reshape_output_1


class Reshape959(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 96, 54, 54))
        return reshape_output_1


class Reshape960(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 96, 54, 54))
        return reshape_output_1


class Reshape961(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 256, 27, 27))
        return reshape_output_1


class Reshape962(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 27, 27))
        return reshape_output_1


class Reshape963(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 27, 27, 12))
        return reshape_output_1


class Reshape964(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(729, 12))
        return reshape_output_1


class Reshape965(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(197, 197, 12))
        return reshape_output_1


class Reshape966(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 27, 27, 16))
        return reshape_output_1


class Reshape967(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(729, 16))
        return reshape_output_1


class Reshape968(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(197, 197, 16))
        return reshape_output_1


class Reshape969(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 384, 196, 1))
        return reshape_output_1


class Reshape970(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(197, 384))
        return reshape_output_1


class Reshape971(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 197, 6, 64))
        return reshape_output_1


class Reshape972(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 197, 384))
        return reshape_output_1


class Reshape973(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(6, 197, 64))
        return reshape_output_1


class Reshape974(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 6, 197, 197))
        return reshape_output_1


class Reshape975(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(6, 197, 197))
        return reshape_output_1


class Reshape976(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(6, 64, 197))
        return reshape_output_1


class Reshape977(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 6, 197, 64))
        return reshape_output_1


class Reshape978(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 192, 196, 1))
        return reshape_output_1


class Reshape979(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(197, 192))
        return reshape_output_1


class Reshape980(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 197, 3, 64))
        return reshape_output_1


class Reshape981(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 197, 192))
        return reshape_output_1


class Reshape982(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(3, 197, 64))
        return reshape_output_1


class Reshape983(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 3, 197, 197))
        return reshape_output_1


class Reshape984(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(3, 197, 197))
        return reshape_output_1


class Reshape985(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(3, 64, 197))
        return reshape_output_1


class Reshape986(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 3, 197, 64))
        return reshape_output_1


class Reshape987(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 192))
        return reshape_output_1


class Reshape988(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2208, 1, 1))
        return reshape_output_1


class Reshape989(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1920, 1, 1))
        return reshape_output_1


class Reshape990(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1664, 1, 1))
        return reshape_output_1


class Reshape991(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1000, 1, 1))
        return reshape_output_1


class Reshape992(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 1, 3, 3))
        return reshape_output_1


class Reshape993(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(96, 1, 3, 3))
        return reshape_output_1


class Reshape994(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(144, 1, 3, 3))
        return reshape_output_1


class Reshape995(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(144, 1, 5, 5))
        return reshape_output_1


class Reshape996(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(240, 1, 5, 5))
        return reshape_output_1


class Reshape997(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(240, 1, 3, 3))
        return reshape_output_1


class Reshape998(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(480, 1, 3, 3))
        return reshape_output_1


class Reshape999(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(480, 1, 5, 5))
        return reshape_output_1


class Reshape1000(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(672, 1, 5, 5))
        return reshape_output_1


class Reshape1001(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1152, 1, 5, 5))
        return reshape_output_1


class Reshape1002(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1152, 1, 3, 3))
        return reshape_output_1


class Reshape1003(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(48, 1, 3, 3))
        return reshape_output_1


class Reshape1004(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(24, 1, 3, 3))
        return reshape_output_1


class Reshape1005(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(192, 1, 3, 3))
        return reshape_output_1


class Reshape1006(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(192, 1, 5, 5))
        return reshape_output_1


class Reshape1007(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(336, 1, 5, 5))
        return reshape_output_1


class Reshape1008(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(336, 1, 3, 3))
        return reshape_output_1


class Reshape1009(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(672, 1, 3, 3))
        return reshape_output_1


class Reshape1010(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(960, 1, 5, 5))
        return reshape_output_1


class Reshape1011(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1632, 1, 5, 5))
        return reshape_output_1


class Reshape1012(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1632, 1, 3, 3))
        return reshape_output_1


class Reshape1013(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2688, 1, 3, 3))
        return reshape_output_1


class Reshape1014(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1792, 1, 1))
        return reshape_output_1


class Reshape1015(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(288, 1, 5, 5))
        return reshape_output_1


class Reshape1016(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(288, 1, 3, 3))
        return reshape_output_1


class Reshape1017(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(528, 1, 3, 3))
        return reshape_output_1


class Reshape1018(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(528, 1, 5, 5))
        return reshape_output_1


class Reshape1019(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(720, 1, 5, 5))
        return reshape_output_1


class Reshape1020(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1248, 1, 5, 5))
        return reshape_output_1


class Reshape1021(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1248, 1, 3, 3))
        return reshape_output_1


class Reshape1022(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(576, 1, 3, 3))
        return reshape_output_1


class Reshape1023(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(576, 1, 5, 5))
        return reshape_output_1


class Reshape1024(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(816, 1, 5, 5))
        return reshape_output_1


class Reshape1025(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1392, 1, 5, 5))
        return reshape_output_1


class Reshape1026(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1392, 1, 3, 3))
        return reshape_output_1


class Reshape1027(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 1, 3, 3))
        return reshape_output_1


class Reshape1028(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 1, 3, 3))
        return reshape_output_1


class Reshape1029(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 1, 3, 3))
        return reshape_output_1


class Reshape1030(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(36, 1, 3, 3))
        return reshape_output_1


class Reshape1031(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(72, 1, 5, 5))
        return reshape_output_1


class Reshape1032(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(20, 1, 3, 3))
        return reshape_output_1


class Reshape1033(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(24, 1, 5, 5))
        return reshape_output_1


class Reshape1034(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(60, 1, 3, 3))
        return reshape_output_1


class Reshape1035(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(120, 1, 3, 3))
        return reshape_output_1


class Reshape1036(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(40, 1, 3, 3))
        return reshape_output_1


class Reshape1037(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(100, 1, 3, 3))
        return reshape_output_1


class Reshape1038(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(92, 1, 3, 3))
        return reshape_output_1


class Reshape1039(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(56, 1, 3, 3))
        return reshape_output_1


class Reshape1040(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(80, 1, 3, 3))
        return reshape_output_1


class Reshape1041(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(112, 1, 5, 5))
        return reshape_output_1


class Reshape1042(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(72, 1, 1, 5))
        return reshape_output_1


class Reshape1043(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(72, 1, 5, 1))
        return reshape_output_1


class Reshape1044(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(120, 1, 1, 5))
        return reshape_output_1


class Reshape1045(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(120, 1, 5, 1))
        return reshape_output_1


class Reshape1046(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(240, 1, 1, 5))
        return reshape_output_1


class Reshape1047(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(240, 1, 5, 1))
        return reshape_output_1


class Reshape1048(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(200, 1, 1, 5))
        return reshape_output_1


class Reshape1049(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(200, 1, 5, 1))
        return reshape_output_1


class Reshape1050(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(184, 1, 1, 5))
        return reshape_output_1


class Reshape1051(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(184, 1, 5, 1))
        return reshape_output_1


class Reshape1052(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(480, 1, 1, 5))
        return reshape_output_1


class Reshape1053(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(480, 1, 5, 1))
        return reshape_output_1


class Reshape1054(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(672, 1, 1, 5))
        return reshape_output_1


class Reshape1055(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(672, 1, 5, 1))
        return reshape_output_1


class Reshape1056(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(960, 1, 1, 5))
        return reshape_output_1


class Reshape1057(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(960, 1, 5, 1))
        return reshape_output_1


class Reshape1058(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 64, 19200, 1))
        return reshape_output_1


class Reshape1059(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 19200, 1, 64))
        return reshape_output_1


class Reshape1060(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 120, 160, 64))
        return reshape_output_1


class Reshape1061(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 64, 120, 160))
        return reshape_output_1


class Reshape1062(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 64, 300))
        return reshape_output_1


class Reshape1063(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(300, 64))
        return reshape_output_1


class Reshape1064(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 300, 1, 64))
        return reshape_output_1


class Reshape1065(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 300, 64))
        return reshape_output_1


class Reshape1066(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 19200, 300))
        return reshape_output_1


class Reshape1067(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 19200, 300))
        return reshape_output_1


class Reshape1068(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 120, 160))
        return reshape_output_1


class Reshape1069(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(256, 1, 3, 3))
        return reshape_output_1


class Reshape1070(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 19200, 1))
        return reshape_output_1


class Reshape1071(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 4800, 1))
        return reshape_output_1


class Reshape1072(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4800, 2, 64))
        return reshape_output_1


class Reshape1073(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 60, 80, 128))
        return reshape_output_1


class Reshape1074(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 4800, 64))
        return reshape_output_1


class Reshape1075(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 60, 80))
        return reshape_output_1


class Reshape1076(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 300))
        return reshape_output_1


class Reshape1077(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(300, 128))
        return reshape_output_1


class Reshape1078(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 300, 2, 64))
        return reshape_output_1


class Reshape1079(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 300, 128))
        return reshape_output_1


class Reshape1080(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 300, 64))
        return reshape_output_1


class Reshape1081(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2, 4800, 300))
        return reshape_output_1


class Reshape1082(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 4800, 300))
        return reshape_output_1


class Reshape1083(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 64, 300))
        return reshape_output_1


class Reshape1084(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2, 4800, 64))
        return reshape_output_1


class Reshape1085(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(4800, 128))
        return reshape_output_1


class Reshape1086(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4800, 128))
        return reshape_output_1


class Reshape1087(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 512, 60, 80))
        return reshape_output_1


class Reshape1088(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(512, 1, 3, 3))
        return reshape_output_1


class Reshape1089(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 512, 4800, 1))
        return reshape_output_1


class Reshape1090(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 320, 1200, 1))
        return reshape_output_1


class Reshape1091(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1200, 5, 64))
        return reshape_output_1


class Reshape1092(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 30, 40, 320))
        return reshape_output_1


class Reshape1093(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(5, 1200, 64))
        return reshape_output_1


class Reshape1094(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 320, 30, 40))
        return reshape_output_1


class Reshape1095(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 320, 300))
        return reshape_output_1


class Reshape1096(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(300, 320))
        return reshape_output_1


class Reshape1097(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 300, 5, 64))
        return reshape_output_1


class Reshape1098(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 300, 320))
        return reshape_output_1


class Reshape1099(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(5, 300, 64))
        return reshape_output_1


class Reshape1100(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 5, 1200, 300))
        return reshape_output_1


class Reshape1101(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(5, 1200, 300))
        return reshape_output_1


class Reshape1102(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(5, 64, 300))
        return reshape_output_1


class Reshape1103(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 5, 1200, 64))
        return reshape_output_1


class Reshape1104(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1200, 320))
        return reshape_output_1


class Reshape1105(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1200, 320))
        return reshape_output_1


class Reshape1106(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1280, 30, 40))
        return reshape_output_1


class Reshape1107(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1280, 1, 3, 3))
        return reshape_output_1


class Reshape1108(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1280, 1200, 1))
        return reshape_output_1


class Reshape1109(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 512, 300, 1))
        return reshape_output_1


class Reshape1110(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(300, 512))
        return reshape_output_1


class Reshape1111(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 300, 8, 64))
        return reshape_output_1


class Reshape1112(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 15, 20, 512))
        return reshape_output_1


class Reshape1113(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 300, 512))
        return reshape_output_1


class Reshape1114(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 300, 64))
        return reshape_output_1


class Reshape1115(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 300, 300))
        return reshape_output_1


class Reshape1116(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 300, 300))
        return reshape_output_1


class Reshape1117(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 64, 300))
        return reshape_output_1


class Reshape1118(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 300, 64))
        return reshape_output_1


class Reshape1119(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2048, 15, 20))
        return reshape_output_1


class Reshape1120(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2048, 1, 3, 3))
        return reshape_output_1


class Reshape1121(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2048, 300, 1))
        return reshape_output_1


class Reshape1122(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 30, 40))
        return reshape_output_1


class Reshape1123(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 60, 80))
        return reshape_output_1


class Reshape1124(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 120, 160))
        return reshape_output_1


class Reshape1125(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 224, 224))
        return reshape_output_1


class Reshape1126(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1536, 1, 1))
        return reshape_output_1


class Reshape1127(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 3, 16, 16, 16, 16))
        return reshape_output_1


class Reshape1128(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 512, 1))
        return reshape_output_1


class Reshape1129(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 16, 512))
        return reshape_output_1


class Reshape1130(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1024, 256, 1, 1))
        return reshape_output_1


class Reshape1131(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1024, 512, 1))
        return reshape_output_1


class Reshape1132(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(256, 1024, 1, 1))
        return reshape_output_1


class Reshape1133(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 1, 3, 3))
        return reshape_output_1


class Reshape1134(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(128, 1, 3, 3))
        return reshape_output_1


class Reshape1135(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1024, 1, 3, 3))
        return reshape_output_1


class Reshape1136(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(384, 1, 3, 3))
        return reshape_output_1


class Reshape1137(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(768, 1, 3, 3))
        return reshape_output_1


class Reshape1138(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 768, 1, 1))
        return reshape_output_1


class Reshape1139(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(960, 1, 3, 3))
        return reshape_output_1


class Reshape1140(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(432, 1, 3, 3))
        return reshape_output_1


class Reshape1141(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(720, 1, 3, 3))
        return reshape_output_1


class Reshape1142(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(72, 1, 3, 3))
        return reshape_output_1


class Reshape1143(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(88, 1, 3, 3))
        return reshape_output_1


class Reshape1144(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(96, 1, 5, 5))
        return reshape_output_1


class Reshape1145(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(120, 1, 5, 5))
        return reshape_output_1


class Reshape1146(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 576, 1, 1))
        return reshape_output_1


class Reshape1147(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(200, 1, 3, 3))
        return reshape_output_1


class Reshape1148(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(184, 1, 3, 3))
        return reshape_output_1


class Reshape1149(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 960, 1, 1))
        return reshape_output_1


class Reshape1150(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1088, 1, 1))
        return reshape_output_1


class Reshape1151(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 7392, 1, 1))
        return reshape_output_1


class Reshape1152(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 888, 1, 1))
        return reshape_output_1


class Reshape1153(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 3712, 1, 1))
        return reshape_output_1


class Reshape1154(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 440, 1, 1))
        return reshape_output_1


class Reshape1155(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2520, 1, 1))
        return reshape_output_1


class Reshape1156(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1008, 1, 1))
        return reshape_output_1


class Reshape1157(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 912, 1, 1))
        return reshape_output_1


class Reshape1158(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 672, 1, 1))
        return reshape_output_1


class Reshape1159(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2016, 1, 1))
        return reshape_output_1


class Reshape1160(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 784, 1, 1))
        return reshape_output_1


class Reshape1161(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1512, 1, 1))
        return reshape_output_1


class Reshape1162(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 400, 1, 1))
        return reshape_output_1


class Reshape1163(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 3024, 1, 1))
        return reshape_output_1


class Reshape1164(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16384, 1, 64))
        return reshape_output_1


class Reshape1165(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 128, 64))
        return reshape_output_1


class Reshape1166(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 64, 256))
        return reshape_output_1


class Reshape1167(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(256, 64))
        return reshape_output_1


class Reshape1168(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 1, 64))
        return reshape_output_1


class Reshape1169(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 2, 32))
        return reshape_output_1


class Reshape1170(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 64))
        return reshape_output_1


class Reshape1171(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 16384, 256))
        return reshape_output_1


class Reshape1172(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16384, 256))
        return reshape_output_1


class Reshape1173(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 128, 128))
        return reshape_output_1


class Reshape1174(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 16384, 1))
        return reshape_output_1


class Reshape1175(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4096, 2, 64))
        return reshape_output_1


class Reshape1176(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 64, 64, 128))
        return reshape_output_1


class Reshape1177(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 4096, 64))
        return reshape_output_1


class Reshape1178(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 256))
        return reshape_output_1


class Reshape1179(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(256, 128))
        return reshape_output_1


class Reshape1180(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 2, 64))
        return reshape_output_1


class Reshape1181(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 128))
        return reshape_output_1


class Reshape1182(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 256, 64))
        return reshape_output_1


class Reshape1183(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2, 4096, 256))
        return reshape_output_1


class Reshape1184(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 4096, 256))
        return reshape_output_1


class Reshape1185(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 64, 256))
        return reshape_output_1


class Reshape1186(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2, 4096, 64))
        return reshape_output_1


class Reshape1187(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(4096, 128))
        return reshape_output_1


class Reshape1188(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4096, 128))
        return reshape_output_1


class Reshape1189(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 512, 64, 64))
        return reshape_output_1


class Reshape1190(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 512, 4096, 1))
        return reshape_output_1


class Reshape1191(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 320, 1024, 1))
        return reshape_output_1


class Reshape1192(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1024, 5, 64))
        return reshape_output_1


class Reshape1193(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 32, 320))
        return reshape_output_1


class Reshape1194(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(5, 1024, 64))
        return reshape_output_1


class Reshape1195(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 320, 32, 32))
        return reshape_output_1


class Reshape1196(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 320, 256))
        return reshape_output_1


class Reshape1197(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(256, 320))
        return reshape_output_1


class Reshape1198(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 5, 64))
        return reshape_output_1


class Reshape1199(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 320))
        return reshape_output_1


class Reshape1200(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(5, 256, 64))
        return reshape_output_1


class Reshape1201(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 5, 1024, 256))
        return reshape_output_1


class Reshape1202(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(5, 1024, 256))
        return reshape_output_1


class Reshape1203(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(5, 64, 256))
        return reshape_output_1


class Reshape1204(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 5, 1024, 64))
        return reshape_output_1


class Reshape1205(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1024, 320))
        return reshape_output_1


class Reshape1206(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1024, 320))
        return reshape_output_1


class Reshape1207(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1280, 32, 32))
        return reshape_output_1


class Reshape1208(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1280, 1024, 1))
        return reshape_output_1


class Reshape1209(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 512, 256, 1))
        return reshape_output_1


class Reshape1210(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 256, 64))
        return reshape_output_1


class Reshape1211(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 64, 256))
        return reshape_output_1


class Reshape1212(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 256, 64))
        return reshape_output_1


class Reshape1213(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2048, 256, 1))
        return reshape_output_1


class Reshape1214(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 768, 16, 16))
        return reshape_output_1


class Reshape1215(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 768, 32, 32))
        return reshape_output_1


class Reshape1216(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 768, 64, 64))
        return reshape_output_1


class Reshape1217(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 768, 128, 128))
        return reshape_output_1


class Reshape1218(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16384, 1, 32))
        return reshape_output_1


class Reshape1219(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 128, 32))
        return reshape_output_1


class Reshape1220(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 256))
        return reshape_output_1


class Reshape1221(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(256, 32))
        return reshape_output_1


class Reshape1222(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 1, 32))
        return reshape_output_1


class Reshape1223(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 32))
        return reshape_output_1


class Reshape1224(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 128, 128))
        return reshape_output_1


class Reshape1225(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 16384, 1))
        return reshape_output_1


class Reshape1226(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 64, 4096, 1))
        return reshape_output_1


class Reshape1227(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4096, 2, 32))
        return reshape_output_1


class Reshape1228(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 64, 64, 64))
        return reshape_output_1


class Reshape1229(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 4096, 32))
        return reshape_output_1


class Reshape1230(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 256, 32))
        return reshape_output_1


class Reshape1231(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 32, 256))
        return reshape_output_1


class Reshape1232(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2, 4096, 32))
        return reshape_output_1


class Reshape1233(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(4096, 64))
        return reshape_output_1


class Reshape1234(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4096, 64))
        return reshape_output_1


class Reshape1235(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 4096, 1))
        return reshape_output_1


class Reshape1236(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 160, 1024, 1))
        return reshape_output_1


class Reshape1237(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1024, 5, 32))
        return reshape_output_1


class Reshape1238(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 32, 160))
        return reshape_output_1


class Reshape1239(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(5, 1024, 32))
        return reshape_output_1


class Reshape1240(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 160, 32, 32))
        return reshape_output_1


class Reshape1241(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 160, 256))
        return reshape_output_1


class Reshape1242(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(256, 160))
        return reshape_output_1


class Reshape1243(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 5, 32))
        return reshape_output_1


class Reshape1244(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 160))
        return reshape_output_1


class Reshape1245(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(5, 256, 32))
        return reshape_output_1


class Reshape1246(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(5, 32, 256))
        return reshape_output_1


class Reshape1247(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 5, 1024, 32))
        return reshape_output_1


class Reshape1248(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1024, 160))
        return reshape_output_1


class Reshape1249(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1024, 160))
        return reshape_output_1


class Reshape1250(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 640, 32, 32))
        return reshape_output_1


class Reshape1251(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(640, 1, 3, 3))
        return reshape_output_1


class Reshape1252(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 640, 1024, 1))
        return reshape_output_1


class Reshape1253(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 256, 1))
        return reshape_output_1


class Reshape1254(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 32, 256))
        return reshape_output_1


class Reshape1255(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 256, 32))
        return reshape_output_1


class Reshape1256(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1024, 16, 16))
        return reshape_output_1


class Reshape1257(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1024, 256, 1))
        return reshape_output_1


class Reshape1258(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 5776))
        return reshape_output_1


class Reshape1259(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 2166))
        return reshape_output_1


class Reshape1260(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 600))
        return reshape_output_1


class Reshape1261(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 150))
        return reshape_output_1


class Reshape1262(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 36))
        return reshape_output_1


class Reshape1263(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 81, 5776))
        return reshape_output_1


class Reshape1264(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 81, 2166))
        return reshape_output_1


class Reshape1265(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 81, 600))
        return reshape_output_1


class Reshape1266(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 81, 150))
        return reshape_output_1


class Reshape1267(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 81, 36))
        return reshape_output_1


class Reshape1268(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 81, 4))
        return reshape_output_1


class Reshape1269(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 7, 8, 7, 96))
        return reshape_output_1


class Reshape1270(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(3136, 96))
        return reshape_output_1


class Reshape1271(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 3136, 96))
        return reshape_output_1


class Reshape1272(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 49, 288))
        return reshape_output_1


class Reshape1273(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 49, 3, 3, 32))
        return reshape_output_1


class Reshape1274(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 3, 49, 32))
        return reshape_output_1


class Reshape1275(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(192, 49, 32))
        return reshape_output_1


class Reshape1276(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 3, 49, 49))
        return reshape_output_1


class Reshape1277(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(49, 49, 3))
        return reshape_output_1


class Reshape1278(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(192, 49, 49))
        return reshape_output_1


class Reshape1279(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 64, 3, 49, 49))
        return reshape_output_1


class Reshape1280(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(192, 32, 49))
        return reshape_output_1


class Reshape1281(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 49, 96))
        return reshape_output_1


class Reshape1282(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 56, 56, 96))
        return reshape_output_1


class Reshape1283(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 8, 7, 7, 96))
        return reshape_output_1


class Reshape1284(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 49, 3, 32))
        return reshape_output_1


class Reshape1285(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 56, 56, 384))
        return reshape_output_1


class Reshape1286(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 49, 384))
        return reshape_output_1


class Reshape1287(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(3136, 384))
        return reshape_output_1


class Reshape1288(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(784, 384))
        return reshape_output_1


class Reshape1289(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 784, 384))
        return reshape_output_1


class Reshape1290(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 28, 28, 192))
        return reshape_output_1


class Reshape1291(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 49, 192))
        return reshape_output_1


class Reshape1292(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 7, 4, 7, 192))
        return reshape_output_1


class Reshape1293(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(784, 192))
        return reshape_output_1


class Reshape1294(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 784, 192))
        return reshape_output_1


class Reshape1295(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 49, 576))
        return reshape_output_1


class Reshape1296(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 49, 3, 6, 32))
        return reshape_output_1


class Reshape1297(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 6, 49, 32))
        return reshape_output_1


class Reshape1298(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(96, 49, 32))
        return reshape_output_1


class Reshape1299(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 6, 49, 49))
        return reshape_output_1


class Reshape1300(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(49, 49, 6))
        return reshape_output_1


class Reshape1301(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(96, 49, 49))
        return reshape_output_1


class Reshape1302(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 6, 49, 49))
        return reshape_output_1


class Reshape1303(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(96, 32, 49))
        return reshape_output_1


class Reshape1304(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 4, 7, 7, 192))
        return reshape_output_1


class Reshape1305(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 49, 6, 32))
        return reshape_output_1


class Reshape1306(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 28, 28, 768))
        return reshape_output_1


class Reshape1307(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 49, 768))
        return reshape_output_1


class Reshape1308(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(784, 768))
        return reshape_output_1


class Reshape1309(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(196, 768))
        return reshape_output_1


class Reshape1310(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 196, 768))
        return reshape_output_1


class Reshape1311(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 14, 14, 384))
        return reshape_output_1


class Reshape1312(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(4, 49, 384))
        return reshape_output_1


class Reshape1313(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2, 7, 2, 7, 384))
        return reshape_output_1


class Reshape1314(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(196, 384))
        return reshape_output_1


class Reshape1315(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 196, 384))
        return reshape_output_1


class Reshape1316(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(4, 49, 1152))
        return reshape_output_1


class Reshape1317(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(4, 49, 3, 12, 32))
        return reshape_output_1


class Reshape1318(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(4, 12, 49, 32))
        return reshape_output_1


class Reshape1319(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(48, 49, 32))
        return reshape_output_1


class Reshape1320(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(4, 12, 49, 49))
        return reshape_output_1


class Reshape1321(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(49, 49, 12))
        return reshape_output_1


class Reshape1322(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(48, 49, 49))
        return reshape_output_1


class Reshape1323(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 12, 49, 49))
        return reshape_output_1


class Reshape1324(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(48, 32, 49))
        return reshape_output_1


class Reshape1325(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2, 2, 7, 7, 384))
        return reshape_output_1


class Reshape1326(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(4, 49, 12, 32))
        return reshape_output_1


class Reshape1327(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 14, 14, 1536))
        return reshape_output_1


class Reshape1328(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(4, 49, 1536))
        return reshape_output_1


class Reshape1329(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(196, 1536))
        return reshape_output_1


class Reshape1330(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(49, 1536))
        return reshape_output_1


class Reshape1331(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 49, 1536))
        return reshape_output_1


class Reshape1332(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 7, 7, 768))
        return reshape_output_1


class Reshape1333(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 49, 768))
        return reshape_output_1


class Reshape1334(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 7, 1, 7, 768))
        return reshape_output_1


class Reshape1335(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(49, 768))
        return reshape_output_1


class Reshape1336(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 49, 2304))
        return reshape_output_1


class Reshape1337(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 49, 3, 24, 32))
        return reshape_output_1


class Reshape1338(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 24, 49, 32))
        return reshape_output_1


class Reshape1339(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(24, 49, 32))
        return reshape_output_1


class Reshape1340(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 24, 49, 49))
        return reshape_output_1


class Reshape1341(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(49, 49, 24))
        return reshape_output_1


class Reshape1342(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(24, 49, 49))
        return reshape_output_1


class Reshape1343(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(24, 32, 49))
        return reshape_output_1


class Reshape1344(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 1, 7, 7, 768))
        return reshape_output_1


class Reshape1345(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 49, 24, 32))
        return reshape_output_1


class Reshape1346(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 7, 7, 3072))
        return reshape_output_1


class Reshape1347(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 49, 3072))
        return reshape_output_1


class Reshape1348(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(49, 3072))
        return reshape_output_1


class Reshape1349(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 7, 8, 7, 128))
        return reshape_output_1


class Reshape1350(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(3136, 128))
        return reshape_output_1


class Reshape1351(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 49, 3, 4, 32))
        return reshape_output_1


class Reshape1352(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 4, 49, 32))
        return reshape_output_1


class Reshape1353(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(256, 49, 32))
        return reshape_output_1


class Reshape1354(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 4, 49, 49))
        return reshape_output_1


class Reshape1355(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(49, 49, 4))
        return reshape_output_1


class Reshape1356(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(256, 49, 49))
        return reshape_output_1


class Reshape1357(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 64, 4, 49, 49))
        return reshape_output_1


class Reshape1358(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(256, 32, 49))
        return reshape_output_1


class Reshape1359(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 49, 128))
        return reshape_output_1


class Reshape1360(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 56, 56, 128))
        return reshape_output_1


class Reshape1361(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 8, 7, 7, 128))
        return reshape_output_1


class Reshape1362(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 56, 56, 512))
        return reshape_output_1


class Reshape1363(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(3136, 512))
        return reshape_output_1


class Reshape1364(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(784, 512))
        return reshape_output_1


class Reshape1365(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 28, 28, 256))
        return reshape_output_1


class Reshape1366(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 49, 256))
        return reshape_output_1


class Reshape1367(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 7, 4, 7, 256))
        return reshape_output_1


class Reshape1368(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(784, 256))
        return reshape_output_1


class Reshape1369(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 49, 3, 8, 32))
        return reshape_output_1


class Reshape1370(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 8, 49, 32))
        return reshape_output_1


class Reshape1371(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(128, 49, 32))
        return reshape_output_1


class Reshape1372(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 8, 49, 49))
        return reshape_output_1


class Reshape1373(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(49, 49, 8))
        return reshape_output_1


class Reshape1374(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(128, 49, 49))
        return reshape_output_1


class Reshape1375(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 8, 49, 49))
        return reshape_output_1


class Reshape1376(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(128, 32, 49))
        return reshape_output_1


class Reshape1377(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 4, 7, 7, 256))
        return reshape_output_1


class Reshape1378(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 28, 28, 1024))
        return reshape_output_1


class Reshape1379(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(784, 1024))
        return reshape_output_1


class Reshape1380(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(196, 1024))
        return reshape_output_1


class Reshape1381(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 14, 14, 512))
        return reshape_output_1


class Reshape1382(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(4, 49, 512))
        return reshape_output_1


class Reshape1383(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2, 7, 2, 7, 512))
        return reshape_output_1


class Reshape1384(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(196, 512))
        return reshape_output_1


class Reshape1385(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(4, 49, 3, 16, 32))
        return reshape_output_1


class Reshape1386(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(4, 16, 49, 32))
        return reshape_output_1


class Reshape1387(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 49, 32))
        return reshape_output_1


class Reshape1388(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(4, 16, 49, 49))
        return reshape_output_1


class Reshape1389(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(49, 49, 16))
        return reshape_output_1


class Reshape1390(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 49, 49))
        return reshape_output_1


class Reshape1391(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 16, 49, 49))
        return reshape_output_1


class Reshape1392(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 32, 49))
        return reshape_output_1


class Reshape1393(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2, 2, 7, 7, 512))
        return reshape_output_1


class Reshape1394(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 14, 14, 2048))
        return reshape_output_1


class Reshape1395(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(196, 2048))
        return reshape_output_1


class Reshape1396(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(49, 2048))
        return reshape_output_1


class Reshape1397(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 7, 7, 1024))
        return reshape_output_1


class Reshape1398(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 49, 1024))
        return reshape_output_1


class Reshape1399(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 7, 1, 7, 1024))
        return reshape_output_1


class Reshape1400(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(49, 1024))
        return reshape_output_1


class Reshape1401(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 49, 3, 32, 32))
        return reshape_output_1


class Reshape1402(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 49, 32))
        return reshape_output_1


class Reshape1403(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 49, 32))
        return reshape_output_1


class Reshape1404(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 49, 49))
        return reshape_output_1


class Reshape1405(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(49, 49, 32))
        return reshape_output_1


class Reshape1406(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 49, 49))
        return reshape_output_1


class Reshape1407(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 32, 49))
        return reshape_output_1


class Reshape1408(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 1, 7, 7, 1024))
        return reshape_output_1


class Reshape1409(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 7, 7, 4096))
        return reshape_output_1


class Reshape1410(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(49, 4096))
        return reshape_output_1


class Reshape1411(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 96, 3136, 1))
        return reshape_output_1


class Reshape1412(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2401,))
        return reshape_output_1


class Reshape1413(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 768, 1))
        return reshape_output_1


class Reshape1414(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 25088, 1, 1))
        return reshape_output_1


class Reshape1415(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 25088))
        return reshape_output_1


class Reshape1416(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4096, 1, 1))
        return reshape_output_1


class Reshape1417(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(160, 1, 3, 3))
        return reshape_output_1


class Reshape1418(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(224, 1, 3, 3))
        return reshape_output_1


class Reshape1419(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(728, 1, 3, 3))
        return reshape_output_1


class Reshape1420(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1536, 1, 3, 3))
        return reshape_output_1


class Reshape1421(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 255, 60, 60))
        return reshape_output_1


class Reshape1422(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 255, 3600))
        return reshape_output_1


class Reshape1423(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 3, 85, 3600))
        return reshape_output_1


class Reshape1424(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 10800, 85))
        return reshape_output_1


class Reshape1425(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 255, 30, 30))
        return reshape_output_1


class Reshape1426(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 255, 900))
        return reshape_output_1


class Reshape1427(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 3, 85, 900))
        return reshape_output_1


class Reshape1428(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2700, 85))
        return reshape_output_1


class Reshape1429(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 255, 15, 15))
        return reshape_output_1


class Reshape1430(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 255, 225))
        return reshape_output_1


class Reshape1431(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 3, 85, 225))
        return reshape_output_1


class Reshape1432(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 675, 85))
        return reshape_output_1


class Reshape1433(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 255, 40, 40))
        return reshape_output_1


class Reshape1434(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 255, 1600))
        return reshape_output_1


class Reshape1435(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 3, 85, 1600))
        return reshape_output_1


class Reshape1436(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4800, 85))
        return reshape_output_1


class Reshape1437(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 255, 20, 20))
        return reshape_output_1


class Reshape1438(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 255, 400))
        return reshape_output_1


class Reshape1439(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 3, 85, 400))
        return reshape_output_1


class Reshape1440(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1200, 85))
        return reshape_output_1


class Reshape1441(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 255, 10, 10))
        return reshape_output_1


class Reshape1442(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 255, 100))
        return reshape_output_1


class Reshape1443(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 3, 85, 100))
        return reshape_output_1


class Reshape1444(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 300, 85))
        return reshape_output_1


class Reshape1445(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 255, 80, 80))
        return reshape_output_1


class Reshape1446(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 255, 6400))
        return reshape_output_1


class Reshape1447(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 3, 85, 6400))
        return reshape_output_1


class Reshape1448(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 19200, 85))
        return reshape_output_1


class Reshape1449(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 255, 160, 160))
        return reshape_output_1


class Reshape1450(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 255, 25600))
        return reshape_output_1


class Reshape1451(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 3, 85, 25600))
        return reshape_output_1


class Reshape1452(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 76800, 85))
        return reshape_output_1


class Reshape1453(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 4480))
        return reshape_output_1


class Reshape1454(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 1120))
        return reshape_output_1


class Reshape1455(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 280))
        return reshape_output_1


class Reshape1456(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 80, 4480))
        return reshape_output_1


class Reshape1457(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 80, 1120))
        return reshape_output_1


class Reshape1458(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 80, 280))
        return reshape_output_1


class Reshape1459(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 17, 4480))
        return reshape_output_1


class Reshape1460(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 17, 1120))
        return reshape_output_1


class Reshape1461(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 17, 280))
        return reshape_output_1


class Reshape1462(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 85, 6400, 1))
        return reshape_output_1


class Reshape1463(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 85, 1600, 1))
        return reshape_output_1


class Reshape1464(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 85, 400, 1))
        return reshape_output_1


class Reshape1465(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 85, 2704, 1))
        return reshape_output_1


class Reshape1466(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 85, 676, 1))
        return reshape_output_1


class Reshape1467(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 85, 169, 1))
        return reshape_output_1


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (
        Reshape0,
        [((1, 6, 768), torch.float32)],
        {
            "model_name": [
                "onnx_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf",
                "pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(6, 768)"},
        },
    ),
    (
        Reshape1,
        [((1, 6, 768), torch.float32)],
        {
            "model_name": [
                "onnx_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf",
                "pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 6, 12, 64)"},
        },
    ),
    (
        Reshape2,
        [((6, 768), torch.float32)],
        {
            "model_name": [
                "onnx_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf",
                "pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 6, 768)"},
        },
    ),
    (
        Reshape3,
        [((1, 12, 6, 64), torch.float32)],
        {
            "model_name": [
                "onnx_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf",
                "pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(12, 6, 64)"},
        },
    ),
    (
        Reshape4,
        [((1, 12, 64, 6), torch.float32)],
        {
            "model_name": [
                "onnx_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf",
                "pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(12, 64, 6)"},
        },
    ),
    (
        Reshape5,
        [((12, 6, 6), torch.float32)],
        {
            "model_name": [
                "onnx_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf",
                "pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 12, 6, 6)"},
        },
    ),
    (
        Reshape6,
        [((1, 12, 6, 6), torch.float32)],
        {
            "model_name": [
                "onnx_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf",
                "pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(12, 6, 6)"},
        },
    ),
    (
        Reshape7,
        [((12, 6, 64), torch.float32)],
        {
            "model_name": [
                "onnx_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf",
                "pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 12, 6, 64)"},
        },
    ),
    (
        Reshape0,
        [((1, 6, 12, 64), torch.float32)],
        {
            "model_name": [
                "onnx_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf",
                "pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(6, 768)"},
        },
    ),
    (
        Reshape8,
        [((1, 1, 768), torch.float32)],
        {
            "model_name": [
                "onnx_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf",
                "onnx_vit_base_google_vit_base_patch16_224_img_cls_hf",
                "pt_whisper_openai_whisper_small_speech_recognition_hf",
                "pt_vilt_dandelin_vilt_b32_mlm_mlm_hf",
                "pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf",
                "pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf",
                "pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf",
                "pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf",
                "pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf",
                "pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder",
                "pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf",
                "pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf",
                "pt_t5_google_flan_t5_base_text_gen_hf",
                "pt_t5_t5_base_text_gen_hf",
                "pt_deit_facebook_deit_base_patch16_224_img_cls_hf",
                "pt_deit_facebook_deit_base_distilled_patch16_224_img_cls_hf",
                "pt_vit_google_vit_base_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 768)"},
        },
    ),
    (
        Reshape9,
        [((1, 1, 768), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1, 12, 64)"},
        },
    ),
    (
        Reshape10,
        [((1, 384, 1024), torch.float32)],
        {
            "model_name": [
                "onnx_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                "pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf",
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(384, 1024)"},
        },
    ),
    (
        Reshape11,
        [((1, 384, 1024), torch.float32)],
        {
            "model_name": [
                "onnx_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                "pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf",
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 384, 16, 64)"},
        },
    ),
    (
        Reshape12,
        [((384, 1024), torch.float32)],
        {
            "model_name": [
                "onnx_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                "pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf",
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 384, 1024)"},
        },
    ),
    (
        Reshape13,
        [((1, 16, 384, 64), torch.float32)],
        {
            "model_name": [
                "onnx_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                "pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf",
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(16, 384, 64)"},
        },
    ),
    (
        Reshape14,
        [((1, 16, 64, 384), torch.float32)],
        {
            "model_name": [
                "onnx_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                "pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf",
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(16, 64, 384)"},
        },
    ),
    (
        Reshape15,
        [((16, 384, 384), torch.float32)],
        {
            "model_name": [
                "onnx_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                "pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf",
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 16, 384, 384)"},
        },
    ),
    (
        Reshape16,
        [((1, 16, 384, 384), torch.float32)],
        {
            "model_name": [
                "onnx_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                "pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf",
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(16, 384, 384)"},
        },
    ),
    (
        Reshape17,
        [((16, 384, 64), torch.float32)],
        {
            "model_name": [
                "onnx_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                "pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf",
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 16, 384, 64)"},
        },
    ),
    (
        Reshape10,
        [((1, 384, 16, 64), torch.float32)],
        {
            "model_name": [
                "onnx_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                "pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf",
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(384, 1024)"},
        },
    ),
    (
        Reshape18,
        [((384, 1), torch.float32)],
        {
            "model_name": [
                "onnx_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                "pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf",
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                "pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 384, 1)"},
        },
    ),
    (
        Reshape19,
        [((1, 128, 768), torch.float32)],
        {
            "model_name": [
                "onnx_bert_bert_base_uncased_mlm_hf",
                "pt_albert_base_v1_token_cls_hf",
                "pt_albert_base_v2_token_cls_hf",
                "pt_albert_base_v1_mlm_hf",
                "pt_albert_base_v2_mlm_hf",
                "pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf",
                "pt_bert_bert_base_uncased_mlm_hf",
                "pt_distilbert_distilbert_base_multilingual_cased_mlm_hf",
                "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
                "pt_distilbert_distilbert_base_uncased_mlm_hf",
                "pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder",
                "pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_roberta_xlm_roberta_base_mlm_hf",
                "pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(128, 768)"},
        },
    ),
    (
        Reshape20,
        [((1, 128, 768), torch.float32)],
        {
            "model_name": [
                "onnx_bert_bert_base_uncased_mlm_hf",
                "pt_albert_base_v1_token_cls_hf",
                "pt_albert_base_v2_token_cls_hf",
                "pt_albert_base_v1_mlm_hf",
                "pt_albert_base_v2_mlm_hf",
                "pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf",
                "pt_bert_bert_base_uncased_mlm_hf",
                "pt_distilbert_distilbert_base_multilingual_cased_mlm_hf",
                "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
                "pt_distilbert_distilbert_base_uncased_mlm_hf",
                "pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder",
                "pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_roberta_xlm_roberta_base_mlm_hf",
                "pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 128, 12, 64)"},
        },
    ),
    (
        Reshape21,
        [((128, 768), torch.float32)],
        {
            "model_name": [
                "onnx_bert_bert_base_uncased_mlm_hf",
                "pt_albert_base_v1_token_cls_hf",
                "pt_albert_base_v2_token_cls_hf",
                "pt_albert_base_v1_mlm_hf",
                "pt_albert_base_v2_mlm_hf",
                "pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf",
                "pt_bert_bert_base_uncased_mlm_hf",
                "pt_distilbert_distilbert_base_multilingual_cased_mlm_hf",
                "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
                "pt_distilbert_distilbert_base_uncased_mlm_hf",
                "pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder",
                "pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_roberta_xlm_roberta_base_mlm_hf",
                "pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 128, 768)"},
        },
    ),
    (
        Reshape22,
        [((1, 12, 128, 64), torch.float32)],
        {
            "model_name": [
                "onnx_bert_bert_base_uncased_mlm_hf",
                "pt_albert_base_v1_token_cls_hf",
                "pt_albert_base_v2_token_cls_hf",
                "pt_albert_base_v1_mlm_hf",
                "pt_albert_base_v2_mlm_hf",
                "pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf",
                "pt_bert_bert_base_uncased_mlm_hf",
                "pt_distilbert_distilbert_base_multilingual_cased_mlm_hf",
                "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
                "pt_distilbert_distilbert_base_uncased_mlm_hf",
                "pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder",
                "pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_roberta_xlm_roberta_base_mlm_hf",
                "pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf",
                "pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(12, 128, 64)"},
        },
    ),
    (
        Reshape23,
        [((1, 12, 64, 128), torch.float32)],
        {
            "model_name": [
                "onnx_bert_bert_base_uncased_mlm_hf",
                "pt_albert_base_v1_token_cls_hf",
                "pt_albert_base_v2_token_cls_hf",
                "pt_albert_base_v1_mlm_hf",
                "pt_albert_base_v2_mlm_hf",
                "pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf",
                "pt_bert_bert_base_uncased_mlm_hf",
                "pt_distilbert_distilbert_base_multilingual_cased_mlm_hf",
                "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
                "pt_distilbert_distilbert_base_uncased_mlm_hf",
                "pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder",
                "pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_roberta_xlm_roberta_base_mlm_hf",
                "pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(12, 64, 128)"},
        },
    ),
    (
        Reshape24,
        [((1, 12, 64, 128), torch.float32)],
        {
            "model_name": ["pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 768, 128, 1)"},
        },
    ),
    (
        Reshape25,
        [((12, 128, 128), torch.float32)],
        {
            "model_name": [
                "onnx_bert_bert_base_uncased_mlm_hf",
                "pt_albert_base_v1_token_cls_hf",
                "pt_albert_base_v2_token_cls_hf",
                "pt_albert_base_v1_mlm_hf",
                "pt_albert_base_v2_mlm_hf",
                "pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf",
                "pt_bert_bert_base_uncased_mlm_hf",
                "pt_distilbert_distilbert_base_multilingual_cased_mlm_hf",
                "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
                "pt_distilbert_distilbert_base_uncased_mlm_hf",
                "pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder",
                "pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_roberta_xlm_roberta_base_mlm_hf",
                "pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf",
                "pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 12, 128, 128)"},
        },
    ),
    (
        Reshape26,
        [((1, 12, 128, 128), torch.float32)],
        {
            "model_name": [
                "onnx_bert_bert_base_uncased_mlm_hf",
                "pt_albert_base_v1_token_cls_hf",
                "pt_albert_base_v2_token_cls_hf",
                "pt_albert_base_v1_mlm_hf",
                "pt_albert_base_v2_mlm_hf",
                "pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf",
                "pt_bert_bert_base_uncased_mlm_hf",
                "pt_distilbert_distilbert_base_multilingual_cased_mlm_hf",
                "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
                "pt_distilbert_distilbert_base_uncased_mlm_hf",
                "pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder",
                "pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_roberta_xlm_roberta_base_mlm_hf",
                "pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf",
                "pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(12, 128, 128)"},
        },
    ),
    (
        Reshape27,
        [((12, 128, 64), torch.float32)],
        {
            "model_name": [
                "onnx_bert_bert_base_uncased_mlm_hf",
                "pt_albert_base_v1_token_cls_hf",
                "pt_albert_base_v2_token_cls_hf",
                "pt_albert_base_v1_mlm_hf",
                "pt_albert_base_v2_mlm_hf",
                "pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf",
                "pt_bert_bert_base_uncased_mlm_hf",
                "pt_distilbert_distilbert_base_multilingual_cased_mlm_hf",
                "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
                "pt_distilbert_distilbert_base_uncased_mlm_hf",
                "pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder",
                "pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_roberta_xlm_roberta_base_mlm_hf",
                "pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf",
                "pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 12, 128, 64)"},
        },
    ),
    (
        Reshape19,
        [((1, 128, 12, 64), torch.float32)],
        {
            "model_name": [
                "onnx_bert_bert_base_uncased_mlm_hf",
                "pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf",
                "pt_bert_bert_base_uncased_mlm_hf",
                "pt_distilbert_distilbert_base_multilingual_cased_mlm_hf",
                "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
                "pt_distilbert_distilbert_base_uncased_mlm_hf",
                "pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder",
                "pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_roberta_xlm_roberta_base_mlm_hf",
                "pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(128, 768)"},
        },
    ),
    (
        Reshape28,
        [((1, 128, 12, 64), torch.float32)],
        {
            "model_name": [
                "pt_albert_base_v1_token_cls_hf",
                "pt_albert_base_v2_token_cls_hf",
                "pt_albert_base_v1_mlm_hf",
                "pt_albert_base_v2_mlm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 128, 768, 1)"},
        },
    ),
    (
        Reshape29,
        [((1, 13, 384), torch.float32)],
        {
            "model_name": ["onnx_minilm_sentence_transformers_all_minilm_l6_v2_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(13, 384)"},
        },
    ),
    (
        Reshape30,
        [((1, 13, 384), torch.float32)],
        {
            "model_name": ["onnx_minilm_sentence_transformers_all_minilm_l6_v2_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 13, 12, 32)"},
        },
    ),
    (
        Reshape31,
        [((13, 384), torch.float32)],
        {
            "model_name": ["onnx_minilm_sentence_transformers_all_minilm_l6_v2_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 13, 384)"},
        },
    ),
    (
        Reshape32,
        [((1, 12, 13, 32), torch.float32)],
        {
            "model_name": ["onnx_minilm_sentence_transformers_all_minilm_l6_v2_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(12, 13, 32)"},
        },
    ),
    (
        Reshape33,
        [((1, 12, 32, 13), torch.float32)],
        {
            "model_name": ["onnx_minilm_sentence_transformers_all_minilm_l6_v2_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(12, 32, 13)"},
        },
    ),
    (
        Reshape34,
        [((12, 13, 13), torch.float32)],
        {
            "model_name": ["onnx_minilm_sentence_transformers_all_minilm_l6_v2_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 12, 13, 13)"},
        },
    ),
    (
        Reshape35,
        [((1, 12, 13, 13), torch.float32)],
        {
            "model_name": ["onnx_minilm_sentence_transformers_all_minilm_l6_v2_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(12, 13, 13)"},
        },
    ),
    (
        Reshape36,
        [((12, 13, 32), torch.float32)],
        {
            "model_name": ["onnx_minilm_sentence_transformers_all_minilm_l6_v2_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 12, 13, 32)"},
        },
    ),
    (
        Reshape29,
        [((1, 13, 12, 32), torch.float32)],
        {
            "model_name": ["onnx_minilm_sentence_transformers_all_minilm_l6_v2_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(13, 384)"},
        },
    ),
    (
        Reshape37,
        [((1, 1, 384), torch.float32)],
        {
            "model_name": [
                "onnx_minilm_sentence_transformers_all_minilm_l6_v2_seq_cls_hf",
                "pt_whisper_openai_whisper_tiny_speech_recognition_hf",
                "pt_deit_facebook_deit_small_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 384)"},
        },
    ),
    (
        Reshape38,
        [((1, 1, 384), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1, 6, 64)"},
        },
    ),
    (
        Reshape39,
        [((1, 100, 256), torch.float32)],
        {
            "model_name": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(100, 256)"},
        },
    ),
    (
        Reshape40,
        [((1, 100, 256), torch.float32)],
        {
            "model_name": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 100, 8, 32)"},
        },
    ),
    (
        Reshape41,
        [((100, 256), torch.float32)],
        {
            "model_name": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 100, 256)"},
        },
    ),
    (
        Reshape42,
        [((1, 8, 100, 32), torch.float32)],
        {
            "model_name": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(8, 100, 32)"},
        },
    ),
    (
        Reshape43,
        [((8, 100, 100), torch.float32)],
        {
            "model_name": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(8, 100, 100)"},
        },
    ),
    (
        Reshape44,
        [((8, 100, 32), torch.float32)],
        {
            "model_name": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 8, 100, 32)"},
        },
    ),
    (
        Reshape39,
        [((1, 100, 8, 32), torch.float32)],
        {
            "model_name": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(100, 256)"},
        },
    ),
    (
        Reshape45,
        [((1, 256, 14, 20), torch.float32)],
        {
            "model_name": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 256, 280)"},
        },
    ),
    (
        Reshape46,
        [((1, 256, 14, 20), torch.float32)],
        {
            "model_name": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 8, 32, 280)"},
        },
    ),
    (
        Reshape47,
        [((1, 280, 256), torch.float32)],
        {
            "model_name": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(280, 256)"},
        },
    ),
    (
        Reshape48,
        [((1, 280, 256), torch.float32)],
        {
            "model_name": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 280, 8, 32)"},
        },
    ),
    (
        Reshape49,
        [((280, 256), torch.float32)],
        {
            "model_name": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 280, 256)"},
        },
    ),
    (
        Reshape50,
        [((1, 8, 280, 32), torch.float32)],
        {
            "model_name": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(8, 280, 32)"},
        },
    ),
    (
        Reshape51,
        [((8, 280, 280), torch.float32)],
        {
            "model_name": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 8, 280, 280)"},
        },
    ),
    (
        Reshape52,
        [((1, 8, 280, 280), torch.float32)],
        {
            "model_name": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(8, 280, 280)"},
        },
    ),
    (
        Reshape53,
        [((8, 280, 32), torch.float32)],
        {
            "model_name": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 8, 280, 32)"},
        },
    ),
    (
        Reshape47,
        [((1, 280, 8, 32), torch.float32)],
        {
            "model_name": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(280, 256)"},
        },
    ),
    (
        Reshape54,
        [((8, 100, 280), torch.float32)],
        {
            "model_name": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 8, 100, 280)"},
        },
    ),
    (
        Reshape55,
        [((1, 8, 100, 280), torch.float32)],
        {
            "model_name": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(8, 100, 280)"},
        },
    ),
    (
        Reshape56,
        [((100, 92), torch.float32)],
        {
            "model_name": ["onnx_detr_facebook_detr_resnet_50_obj_det_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 100, 92)"},
        },
    ),
    (
        Reshape57,
        [((100, 251), torch.float32)],
        {
            "model_name": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 100, 251)"},
        },
    ),
    (
        Reshape58,
        [((1, 100, 32, 107, 160), torch.float32)],
        {
            "model_name": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(100, 32, 107, 160)"},
        },
    ),
    (
        Reshape59,
        [((1, 100, 64, 54, 80), torch.float32)],
        {
            "model_name": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(100, 64, 54, 80)"},
        },
    ),
    (
        Reshape60,
        [((1, 100, 128, 27, 40), torch.float32)],
        {
            "model_name": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(100, 128, 27, 40)"},
        },
    ),
    (
        Reshape61,
        [((1, 100, 256, 14, 20), torch.float32)],
        {
            "model_name": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(100, 256, 14, 20)"},
        },
    ),
    (
        Reshape62,
        [((1, 256, 280), torch.float32)],
        {
            "model_name": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 256, 14, 20)"},
        },
    ),
    (
        Reshape63,
        [((1, 100, 8, 280), torch.float32)],
        {
            "model_name": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 100, 8, 14, 20)"},
        },
    ),
    (
        Reshape64,
        [((1, 100, 8, 14, 20), torch.float32)],
        {
            "model_name": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 100, 2240)"},
        },
    ),
    (
        Reshape65,
        [((1, 100, 2240), torch.float32)],
        {
            "model_name": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(100, 8, 14, 20)"},
        },
    ),
    (
        Reshape66,
        [((100, 264, 14, 20), torch.float32)],
        {
            "model_name": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(100, 8, 9240)"},
        },
    ),
    (
        Reshape67,
        [((100, 8, 9240), torch.float32)],
        {
            "model_name": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(100, 264, 14, 20)"},
        },
    ),
    (
        Reshape68,
        [((100, 128, 14, 20), torch.float32)],
        {
            "model_name": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(100, 8, 4480)"},
        },
    ),
    (
        Reshape69,
        [((100, 8, 4480), torch.float32)],
        {
            "model_name": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(100, 128, 14, 20)"},
        },
    ),
    (
        Reshape70,
        [((100, 64, 27, 40), torch.float32)],
        {
            "model_name": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(100, 8, 8640)"},
        },
    ),
    (
        Reshape71,
        [((100, 8, 8640), torch.float32)],
        {
            "model_name": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(100, 64, 27, 40)"},
        },
    ),
    (
        Reshape72,
        [((100, 32, 54, 80), torch.float32)],
        {
            "model_name": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(100, 8, 17280)"},
        },
    ),
    (
        Reshape73,
        [((100, 8, 17280), torch.float32)],
        {
            "model_name": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(100, 32, 54, 80)"},
        },
    ),
    (
        Reshape74,
        [((100, 16, 107, 160), torch.float32)],
        {
            "model_name": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(100, 8, 34240)"},
        },
    ),
    (
        Reshape75,
        [((100, 8, 34240), torch.float32)],
        {
            "model_name": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(100, 16, 107, 160)"},
        },
    ),
    (
        Reshape76,
        [((100, 1, 107, 160), torch.float32)],
        {
            "model_name": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 100, 107, 160)"},
        },
    ),
    (
        Reshape77,
        [((1, 2048, 1, 1), torch.float32)],
        {
            "model_name": [
                "onnx_resnet_50_img_cls_hf",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w48_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w32_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w40_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w44_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w30_pose_estimation_osmr",
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
        Reshape78,
        [((1, 2048, 1, 1), torch.float32)],
        {
            "model_name": [
                "pd_resnet_101_img_cls_paddlemodels",
                "pd_resnet_152_img_cls_paddlemodels",
                "pd_resnet_50_img_cls_paddlemodels",
                "pt_hrnet_hrnet_w18_pose_estimation_timm",
                "pt_hrnet_hrnet_w40_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_hrnet_hrnet_w48_pose_estimation_timm",
                "pt_hrnet_hrnet_w44_pose_estimation_timm",
                "pt_hrnet_hrnet_w64_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnet_w32_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm",
                "pt_mobilenetv3_ssd_resnet101_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet152_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet50_img_cls_torchvision",
                "pt_regnet_regnet_x_16gf_img_cls_torchvision",
                "pt_resnet_resnet101_img_cls_torchvision",
                "pt_resnet_resnet50_img_cls_torchvision",
                "pt_resnet_resnet152_img_cls_torchvision",
                "pt_resnet_50_img_cls_hf",
                "pt_resnet_50_img_cls_timm",
                "pt_resnext_resnext50_32x4d_img_cls_torchhub",
                "pt_resnext_resnext101_32x8d_img_cls_torchhub",
                "pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub",
                "pt_wideresnet_wide_resnet50_2_img_cls_timm",
                "pt_wideresnet_wide_resnet50_2_img_cls_torchvision",
                "pt_wideresnet_wide_resnet101_2_img_cls_torchvision",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
                "pt_xception_xception71_tf_in1k_img_cls_timm",
                "pt_xception_xception65_img_cls_timm",
                "pt_xception_xception_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 2048, 1, 1)"},
        },
    ),
    (
        Reshape79,
        [((1, 768, 14, 14), torch.float32)],
        {
            "model_name": ["onnx_vit_base_google_vit_base_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 768, 196)"},
        },
    ),
    (
        Reshape80,
        [((1, 768, 14, 14), torch.float32)],
        {
            "model_name": [
                "pt_beit_microsoft_beit_base_patch16_224_img_cls_hf",
                "pt_deit_facebook_deit_base_patch16_224_img_cls_hf",
                "pt_deit_facebook_deit_base_distilled_patch16_224_img_cls_hf",
                "pt_mlp_mixer_mixer_b16_224_miil_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_goog_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_miil_img_cls_timm",
                "pt_vit_google_vit_base_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 768, 196, 1)"},
        },
    ),
    (
        Reshape81,
        [((1, 197, 768), torch.float32)],
        {
            "model_name": [
                "onnx_vit_base_google_vit_base_patch16_224_img_cls_hf",
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
        Reshape82,
        [((1, 197, 768), torch.float32)],
        {
            "model_name": [
                "onnx_vit_base_google_vit_base_patch16_224_img_cls_hf",
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
        Reshape83,
        [((197, 768), torch.float32)],
        {
            "model_name": [
                "onnx_vit_base_google_vit_base_patch16_224_img_cls_hf",
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
        Reshape82,
        [((197, 768), torch.float32)],
        {
            "model_name": ["pt_beit_microsoft_beit_base_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 197, 12, 64)"},
        },
    ),
    (
        Reshape84,
        [((1, 12, 197, 64), torch.float32)],
        {
            "model_name": [
                "onnx_vit_base_google_vit_base_patch16_224_img_cls_hf",
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
        Reshape85,
        [((1, 12, 64, 197), torch.float32)],
        {
            "model_name": [
                "onnx_vit_base_google_vit_base_patch16_224_img_cls_hf",
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
        Reshape86,
        [((12, 197, 197), torch.float32)],
        {
            "model_name": [
                "onnx_vit_base_google_vit_base_patch16_224_img_cls_hf",
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
        Reshape87,
        [((1, 12, 197, 197), torch.float32)],
        {
            "model_name": [
                "onnx_vit_base_google_vit_base_patch16_224_img_cls_hf",
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
        Reshape88,
        [((12, 197, 64), torch.float32)],
        {
            "model_name": [
                "onnx_vit_base_google_vit_base_patch16_224_img_cls_hf",
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
        Reshape81,
        [((1, 197, 12, 64), torch.float32)],
        {
            "model_name": [
                "onnx_vit_base_google_vit_base_patch16_224_img_cls_hf",
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
        Reshape89,
        [((1, 1024, 14, 14), torch.float32)],
        {
            "model_name": ["onnx_vit_base_google_vit_large_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1024, 196)"},
        },
    ),
    (
        Reshape90,
        [((1, 1024, 14, 14), torch.float32)],
        {
            "model_name": [
                "pt_beit_microsoft_beit_large_patch16_224_img_cls_hf",
                "pt_mlp_mixer_mixer_l16_224_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_l16_224_img_cls_timm",
                "pt_vit_google_vit_large_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1024, 196, 1)"},
        },
    ),
    (
        Reshape91,
        [((1, 197, 1024), torch.float32)],
        {
            "model_name": [
                "onnx_vit_base_google_vit_large_patch16_224_img_cls_hf",
                "pt_beit_microsoft_beit_large_patch16_224_img_cls_hf",
                "pt_vit_google_vit_large_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(197, 1024)"},
        },
    ),
    (
        Reshape92,
        [((1, 197, 1024), torch.float32)],
        {
            "model_name": [
                "onnx_vit_base_google_vit_large_patch16_224_img_cls_hf",
                "pt_beit_microsoft_beit_large_patch16_224_img_cls_hf",
                "pt_vit_google_vit_large_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 197, 16, 64)"},
        },
    ),
    (
        Reshape93,
        [((197, 1024), torch.float32)],
        {
            "model_name": [
                "onnx_vit_base_google_vit_large_patch16_224_img_cls_hf",
                "pt_beit_microsoft_beit_large_patch16_224_img_cls_hf",
                "pt_vit_google_vit_large_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 197, 1024)"},
        },
    ),
    (
        Reshape92,
        [((197, 1024), torch.float32)],
        {
            "model_name": ["pt_beit_microsoft_beit_large_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 197, 16, 64)"},
        },
    ),
    (
        Reshape94,
        [((1, 16, 197, 64), torch.float32)],
        {
            "model_name": [
                "onnx_vit_base_google_vit_large_patch16_224_img_cls_hf",
                "pt_beit_microsoft_beit_large_patch16_224_img_cls_hf",
                "pt_vit_google_vit_large_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(16, 197, 64)"},
        },
    ),
    (
        Reshape95,
        [((1, 16, 64, 197), torch.float32)],
        {
            "model_name": [
                "onnx_vit_base_google_vit_large_patch16_224_img_cls_hf",
                "pt_beit_microsoft_beit_large_patch16_224_img_cls_hf",
                "pt_vit_google_vit_large_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(16, 64, 197)"},
        },
    ),
    (
        Reshape96,
        [((16, 197, 197), torch.float32)],
        {
            "model_name": [
                "onnx_vit_base_google_vit_large_patch16_224_img_cls_hf",
                "pt_beit_microsoft_beit_large_patch16_224_img_cls_hf",
                "pt_vit_google_vit_large_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 16, 197, 197)"},
        },
    ),
    (
        Reshape97,
        [((1, 16, 197, 197), torch.float32)],
        {
            "model_name": [
                "onnx_vit_base_google_vit_large_patch16_224_img_cls_hf",
                "pt_beit_microsoft_beit_large_patch16_224_img_cls_hf",
                "pt_vit_google_vit_large_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(16, 197, 197)"},
        },
    ),
    (
        Reshape98,
        [((16, 197, 64), torch.float32)],
        {
            "model_name": [
                "onnx_vit_base_google_vit_large_patch16_224_img_cls_hf",
                "pt_beit_microsoft_beit_large_patch16_224_img_cls_hf",
                "pt_vit_google_vit_large_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 16, 197, 64)"},
        },
    ),
    (
        Reshape91,
        [((1, 197, 16, 64), torch.float32)],
        {
            "model_name": [
                "onnx_vit_base_google_vit_large_patch16_224_img_cls_hf",
                "pt_beit_microsoft_beit_large_patch16_224_img_cls_hf",
                "pt_vit_google_vit_large_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(197, 1024)"},
        },
    ),
    (
        Reshape99,
        [((1, 1, 1024), torch.float32)],
        {
            "model_name": [
                "onnx_vit_base_google_vit_large_patch16_224_img_cls_hf",
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
        Reshape100,
        [((1, 1, 1024), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_medium_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1, 16, 64)"},
        },
    ),
    (
        Reshape101,
        [((1, 1, 1024), torch.float32)],
        {
            "model_name": [
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1, 1, 1024)"},
        },
    ),
    (
        Reshape102,
        [((1, 256, 6, 6), torch.float32)],
        {
            "model_name": [
                "pd_alexnet_base_img_cls_paddlemodels",
                "pt_alexnet_alexnet_img_cls_torchhub",
                "pt_rcnn_base_obj_det_torchvision_rect_0",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 9216, 1, 1)"},
        },
    ),
    (
        Reshape103,
        [((1, 256, 6, 6), torch.float32)],
        {"model_name": ["pt_alexnet_base_img_cls_osmr"], "pcc": 0.99, "op_params": {"shape": "(1, 9216)"}},
    ),
    (
        Reshape104,
        [((1, 1024, 1, 1), torch.float32)],
        {
            "model_name": [
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_mobilenetv1_basic_img_cls_paddlemodels",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_googlenet_base_img_cls_torchvision",
                "pt_mobilenet_v1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
                "pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_swin_swin_b_img_cls_torchvision",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_vovnet_ese_vovnet39b_obj_det_torchhub",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1024, 1, 1)"},
        },
    ),
    (
        Reshape99,
        [((1, 1024, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_mobilenet_v1_basic_img_cls_torchvision",
                "pt_vovnet_vovnet57_obj_det_osmr",
                "pt_vovnet_vovnet39_obj_det_osmr",
                "pt_vovnet_vovnet_v1_57_obj_det_torchhub",
                "pt_vovnet_v1_vovnet39_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1024)"},
        },
    ),
    (
        Reshape105,
        [((1, 128, 3, 3), torch.float32)],
        {
            "model_name": ["pd_googlenet_base_img_cls_paddlemodels"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1152, 1, 1)"},
        },
    ),
    (
        Reshape106,
        [((1, 1280, 1, 1), torch.float32)],
        {
            "model_name": [
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1280, 1, 1)"},
        },
    ),
    (
        Reshape107,
        [((1, 512, 1, 1), torch.float32)],
        {
            "model_name": [
                "pd_resnet_18_img_cls_paddlemodels",
                "pd_resnet_34_img_cls_paddlemodels",
                "pt_mobilenetv3_ssd_resnet18_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet34_img_cls_torchvision",
                "pt_resnet_resnet18_img_cls_torchvision",
                "pt_resnet_resnet34_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 512, 1, 1)"},
        },
    ),
    (
        Reshape108,
        [((1, 512, 1, 1), torch.float32)],
        {"model_name": ["pt_vovnet_vovnet27s_obj_det_osmr"], "pcc": 0.99, "op_params": {"shape": "(1, 512)"}},
    ),
    pytest.param(
        (
            Reshape109,
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
        Reshape110,
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
        Reshape77,
        [((1, 2048), torch.float32)],
        {
            "model_name": ["pt_stereo_facebook_musicgen_large_music_generation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 2048)"},
        },
    ),
    (
        Reshape111,
        [((1, 2048), torch.float32)],
        {"model_name": ["pt_t5_google_flan_t5_base_text_gen_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 1, 2048)"}},
    ),
    (
        Reshape112,
        [((2, 1, 2048), torch.float32)],
        {
            "model_name": ["pt_stereo_facebook_musicgen_large_music_generation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(2, 2048)"},
        },
    ),
    (
        Reshape113,
        [((2, 1, 2048), torch.float32)],
        {
            "model_name": ["pt_stereo_facebook_musicgen_large_music_generation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(2, 1, 32, 64)"},
        },
    ),
    (
        Reshape114,
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
        Reshape113,
        [((2, 2048), torch.float32)],
        {
            "model_name": ["pt_stereo_facebook_musicgen_large_music_generation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(2, 1, 32, 64)"},
        },
    ),
    (
        Reshape115,
        [((2, 32, 1, 64), torch.float32)],
        {
            "model_name": ["pt_stereo_facebook_musicgen_large_music_generation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(64, 1, 64)"},
        },
    ),
    (
        Reshape116,
        [((64, 1, 64), torch.float32)],
        {
            "model_name": ["pt_stereo_facebook_musicgen_large_music_generation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(2, 32, 1, 64)"},
        },
    ),
    (
        Reshape112,
        [((2, 1, 32, 64), torch.float32)],
        {
            "model_name": ["pt_stereo_facebook_musicgen_large_music_generation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(2, 2048)"},
        },
    ),
    (
        Reshape117,
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
        Reshape118,
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
        Reshape119,
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
        Reshape120,
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
        Reshape121,
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
        Reshape122,
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
        Reshape123,
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
        Reshape124,
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
        Reshape125,
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
        Reshape118,
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
        Reshape126,
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
        Reshape127,
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
        Reshape128,
        [((26, 2048), torch.float32)],
        {
            "model_name": ["pt_stereo_facebook_musicgen_large_music_generation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(2, 13, 2048)"},
        },
    ),
    (
        Reshape129,
        [((26, 2048), torch.float32)],
        {
            "model_name": ["pt_stereo_facebook_musicgen_large_music_generation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(2, 13, 32, 64)"},
        },
    ),
    (
        Reshape130,
        [((2, 13, 2048), torch.float32)],
        {
            "model_name": ["pt_stereo_facebook_musicgen_large_music_generation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(26, 2048)"},
        },
    ),
    (
        Reshape131,
        [((2, 32, 13, 64), torch.float32)],
        {
            "model_name": ["pt_stereo_facebook_musicgen_large_music_generation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(64, 13, 64)"},
        },
    ),
    (
        Reshape132,
        [((64, 1, 13), torch.float32)],
        {
            "model_name": ["pt_stereo_facebook_musicgen_large_music_generation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(2, 32, 1, 13)"},
        },
    ),
    (
        Reshape133,
        [((2, 32, 1, 13), torch.float32)],
        {
            "model_name": ["pt_stereo_facebook_musicgen_large_music_generation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(64, 1, 13)"},
        },
    ),
    (
        Reshape134,
        [((2, 8192), torch.float32)],
        {
            "model_name": ["pt_stereo_facebook_musicgen_large_music_generation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(2, 1, 8192)"},
        },
    ),
    (
        Reshape135,
        [((2, 1, 8192), torch.float32)],
        {
            "model_name": ["pt_stereo_facebook_musicgen_large_music_generation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(2, 8192)"},
        },
    ),
    (
        Reshape136,
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
        Reshape99,
        [((1, 1024), torch.float32)],
        {
            "model_name": ["pt_stereo_facebook_musicgen_small_music_generation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1024)"},
        },
    ),
    (
        Reshape137,
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
        Reshape100,
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
        Reshape138,
        [((2, 1, 1024), torch.float32)],
        {
            "model_name": ["pt_stereo_facebook_musicgen_small_music_generation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(2, 1024)"},
        },
    ),
    (
        Reshape139,
        [((2, 1, 1024), torch.float32)],
        {
            "model_name": ["pt_stereo_facebook_musicgen_small_music_generation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(2, 1, 16, 64)"},
        },
    ),
    (
        Reshape140,
        [((2, 1024), torch.float32)],
        {
            "model_name": ["pt_stereo_facebook_musicgen_small_music_generation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(2, 1, 1024)"},
        },
    ),
    (
        Reshape139,
        [((2, 1024), torch.float32)],
        {
            "model_name": ["pt_stereo_facebook_musicgen_small_music_generation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(2, 1, 16, 64)"},
        },
    ),
    (
        Reshape141,
        [((2, 16, 1, 64), torch.float32)],
        {
            "model_name": ["pt_stereo_facebook_musicgen_small_music_generation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(32, 1, 64)"},
        },
    ),
    (
        Reshape142,
        [((32, 1, 64), torch.float32)],
        {
            "model_name": ["pt_stereo_facebook_musicgen_small_music_generation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(2, 16, 1, 64)"},
        },
    ),
    (
        Reshape138,
        [((2, 1, 16, 64), torch.float32)],
        {
            "model_name": ["pt_stereo_facebook_musicgen_small_music_generation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(2, 1024)"},
        },
    ),
    (
        Reshape143,
        [((26, 1024), torch.float32)],
        {
            "model_name": ["pt_stereo_facebook_musicgen_small_music_generation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(2, 13, 1024)"},
        },
    ),
    (
        Reshape144,
        [((26, 1024), torch.float32)],
        {
            "model_name": ["pt_stereo_facebook_musicgen_small_music_generation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(2, 13, 16, 64)"},
        },
    ),
    (
        Reshape145,
        [((2, 13, 1024), torch.float32)],
        {
            "model_name": ["pt_stereo_facebook_musicgen_small_music_generation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(26, 1024)"},
        },
    ),
    (
        Reshape146,
        [((2, 16, 13, 64), torch.float32)],
        {
            "model_name": ["pt_stereo_facebook_musicgen_small_music_generation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(32, 13, 64)"},
        },
    ),
    (
        Reshape147,
        [((32, 1, 13), torch.float32)],
        {
            "model_name": ["pt_stereo_facebook_musicgen_small_music_generation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(2, 16, 1, 13)"},
        },
    ),
    (
        Reshape148,
        [((2, 16, 1, 13), torch.float32)],
        {
            "model_name": ["pt_stereo_facebook_musicgen_small_music_generation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(32, 1, 13)"},
        },
    ),
    (
        Reshape149,
        [((2, 4096), torch.float32)],
        {
            "model_name": ["pt_stereo_facebook_musicgen_small_music_generation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(2, 1, 4096)"},
        },
    ),
    (
        Reshape150,
        [((2, 1, 4096), torch.float32)],
        {
            "model_name": ["pt_stereo_facebook_musicgen_small_music_generation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(2, 4096)"},
        },
    ),
    (
        Reshape151,
        [((1, 1536), torch.float32)],
        {
            "model_name": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1536)"},
        },
    ),
    (
        Reshape152,
        [((2, 1, 1536), torch.float32)],
        {
            "model_name": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(2, 1536)"},
        },
    ),
    (
        Reshape153,
        [((2, 1, 1536), torch.float32)],
        {
            "model_name": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(2, 1, 24, 64)"},
        },
    ),
    (
        Reshape154,
        [((2, 1536), torch.float32)],
        {
            "model_name": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(2, 1, 1536)"},
        },
    ),
    (
        Reshape153,
        [((2, 1536), torch.float32)],
        {
            "model_name": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(2, 1, 24, 64)"},
        },
    ),
    (
        Reshape155,
        [((2, 24, 1, 64), torch.float32)],
        {
            "model_name": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(48, 1, 64)"},
        },
    ),
    (
        Reshape156,
        [((48, 1, 64), torch.float32)],
        {
            "model_name": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(2, 24, 1, 64)"},
        },
    ),
    (
        Reshape152,
        [((2, 1, 24, 64), torch.float32)],
        {
            "model_name": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(2, 1536)"},
        },
    ),
    (
        Reshape157,
        [((26, 1536), torch.float32)],
        {
            "model_name": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(2, 13, 1536)"},
        },
    ),
    (
        Reshape158,
        [((26, 1536), torch.float32)],
        {
            "model_name": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(2, 13, 24, 64)"},
        },
    ),
    (
        Reshape159,
        [((2, 13, 1536), torch.float32)],
        {
            "model_name": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(26, 1536)"},
        },
    ),
    (
        Reshape160,
        [((2, 24, 13, 64), torch.float32)],
        {
            "model_name": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(48, 13, 64)"},
        },
    ),
    (
        Reshape161,
        [((48, 1, 13), torch.float32)],
        {
            "model_name": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(2, 24, 1, 13)"},
        },
    ),
    (
        Reshape162,
        [((2, 24, 1, 13), torch.float32)],
        {
            "model_name": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(48, 1, 13)"},
        },
    ),
    (
        Reshape163,
        [((2, 6144), torch.float32)],
        {
            "model_name": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(2, 1, 6144)"},
        },
    ),
    (
        Reshape164,
        [((2, 1, 6144), torch.float32)],
        {
            "model_name": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(2, 6144)"},
        },
    ),
    pytest.param(
        (
            Reshape165,
            [((1, 1), torch.int64)],
            {
                "model_name": [
                    "pt_whisper_openai_whisper_tiny_speech_recognition_hf",
                    "pt_whisper_openai_whisper_large_speech_recognition_hf",
                    "pt_whisper_openai_whisper_small_speech_recognition_hf",
                    "pt_whisper_openai_whisper_medium_speech_recognition_hf",
                    "pt_whisper_openai_whisper_base_speech_recognition_hf",
                    "pt_t5_google_flan_t5_base_text_gen_hf",
                    "pt_t5_t5_small_text_gen_hf",
                    "pt_t5_t5_large_text_gen_hf",
                    "pt_t5_google_flan_t5_large_text_gen_hf",
                    "pt_t5_t5_base_text_gen_hf",
                    "pt_t5_google_flan_t5_small_text_gen_hf",
                ],
                "pcc": 0.99,
                "op_params": {"shape": "(1, 1)"},
            },
        ),
        marks=[pytest.mark.xfail(reason="RuntimeError: Long did not match Int")],
    ),
    (
        Reshape166,
        [((1, 384), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1, 384)"},
        },
    ),
    (
        Reshape38,
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
        Reshape167,
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
        Reshape168,
        [((1, 6, 1, 64), torch.float32)],
        {
            "model_name": ["pt_falcon_tiiuae_falcon_7b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 6, 1, 64)"},
        },
    ),
    (
        Reshape169,
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
        Reshape170,
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
        Reshape171,
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
        Reshape168,
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
        Reshape37,
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
        Reshape38,
        [((1, 1, 6, 64), torch.float32)],
        {
            "model_name": ["pt_falcon_tiiuae_falcon_7b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1, 6, 64)"},
        },
    ),
    (
        Reshape172,
        [((1, 1, 6, 64), torch.float32)],
        {
            "model_name": ["pt_falcon_tiiuae_falcon_7b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 6, 64)"},
        },
    ),
    (
        Reshape173,
        [((1, 80, 3000), torch.float32)],
        {
            "model_name": [
                "pt_whisper_openai_whisper_tiny_speech_recognition_hf",
                "pt_whisper_openai_whisper_large_speech_recognition_hf",
                "pt_whisper_openai_whisper_small_speech_recognition_hf",
                "pt_whisper_openai_whisper_medium_speech_recognition_hf",
                "pt_whisper_openai_whisper_base_speech_recognition_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 80, 3000, 1)"},
        },
    ),
    (
        Reshape174,
        [((384, 80, 3), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(384, 80, 3, 1)"},
        },
    ),
    (
        Reshape175,
        [((1, 384, 3000, 1), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 384, 3000)"},
        },
    ),
    (
        Reshape176,
        [((1, 384, 3000), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 384, 3000, 1)"},
        },
    ),
    (
        Reshape177,
        [((384, 384, 3), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(384, 384, 3, 1)"},
        },
    ),
    (
        Reshape178,
        [((1, 384, 1500, 1), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 384, 1500)"},
        },
    ),
    (
        Reshape179,
        [((1, 1500, 384), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1500, 384)"},
        },
    ),
    (
        Reshape180,
        [((1, 1500, 384), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1500, 6, 64)"},
        },
    ),
    (
        Reshape181,
        [((1500, 384), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1500, 384)"},
        },
    ),
    (
        Reshape180,
        [((1500, 384), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1500, 6, 64)"},
        },
    ),
    (
        Reshape182,
        [((1, 6, 1500, 64), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(6, 1500, 64)"},
        },
    ),
    (
        Reshape183,
        [((6, 1500, 1500), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 6, 1500, 1500)"},
        },
    ),
    (
        Reshape184,
        [((1, 6, 1500, 1500), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(6, 1500, 1500)"},
        },
    ),
    (
        Reshape185,
        [((1, 6, 64, 1500), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(6, 64, 1500)"},
        },
    ),
    (
        Reshape186,
        [((6, 1500, 64), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 6, 1500, 64)"},
        },
    ),
    (
        Reshape179,
        [((1, 1500, 6, 64), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1500, 384)"},
        },
    ),
    (
        Reshape187,
        [((6, 1, 1500), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 6, 1, 1500)"},
        },
    ),
    (
        Reshape188,
        [((1, 6, 1, 1500), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(6, 1, 1500)"},
        },
    ),
    (
        Reshape189,
        [((1, 1, 1280), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_large_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1280)"},
        },
    ),
    (
        Reshape190,
        [((1, 1, 1280), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_large_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1, 20, 64)"},
        },
    ),
    (
        Reshape191,
        [((1, 1280), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_large_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1, 1280)"},
        },
    ),
    (
        Reshape190,
        [((1, 1280), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_large_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1, 20, 64)"},
        },
    ),
    (
        Reshape192,
        [((1, 20, 1, 64), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_large_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(20, 1, 64)"},
        },
    ),
    (
        Reshape193,
        [((20, 1, 1), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_large_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 20, 1, 1)"},
        },
    ),
    (
        Reshape194,
        [((1, 20, 1, 1), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_large_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(20, 1, 1)"},
        },
    ),
    (
        Reshape195,
        [((1, 20, 64, 1), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_large_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(20, 64, 1)"},
        },
    ),
    (
        Reshape196,
        [((20, 1, 64), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_large_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 20, 1, 64)"},
        },
    ),
    (
        Reshape189,
        [((1, 1, 20, 64), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_large_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1280)"},
        },
    ),
    (
        Reshape197,
        [((1280, 80, 3), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_large_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1280, 80, 3, 1)"},
        },
    ),
    (
        Reshape198,
        [((1, 1280, 3000, 1), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_large_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1280, 3000)"},
        },
    ),
    (
        Reshape199,
        [((1, 1280, 3000), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_large_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1280, 3000, 1)"},
        },
    ),
    pytest.param(
        (
            Reshape200,
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
        Reshape201,
        [((1, 1280, 1500, 1), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_large_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1280, 1500)"},
        },
    ),
    (
        Reshape202,
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
        Reshape203,
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
        Reshape204,
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
        Reshape203,
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
        Reshape205,
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
        Reshape206,
        [((20, 1500, 1500), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_large_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 20, 1500, 1500)"},
        },
    ),
    (
        Reshape207,
        [((1, 20, 1500, 1500), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_large_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(20, 1500, 1500)"},
        },
    ),
    (
        Reshape208,
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
        Reshape209,
        [((20, 1500, 64), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_large_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 20, 1500, 64)"},
        },
    ),
    (
        Reshape202,
        [((1, 1500, 20, 64), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_large_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1500, 1280)"},
        },
    ),
    (
        Reshape210,
        [((20, 1, 1500), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_large_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 20, 1, 1500)"},
        },
    ),
    (
        Reshape211,
        [((1, 20, 1, 1500), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_large_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(20, 1, 1500)"},
        },
    ),
    (
        Reshape212,
        [((1, 768), torch.float32)],
        {
            "model_name": [
                "pt_whisper_openai_whisper_small_speech_recognition_hf",
                "pt_t5_google_flan_t5_base_text_gen_hf",
                "pt_t5_t5_base_text_gen_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1, 768)"},
        },
    ),
    (
        Reshape9,
        [((1, 768), torch.float32)],
        {
            "model_name": [
                "pt_whisper_openai_whisper_small_speech_recognition_hf",
                "pt_t5_google_flan_t5_base_text_gen_hf",
                "pt_t5_t5_base_text_gen_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1, 12, 64)"},
        },
    ),
    (
        Reshape213,
        [((1, 12, 1, 64), torch.float32)],
        {
            "model_name": [
                "pt_whisper_openai_whisper_small_speech_recognition_hf",
                "pt_t5_google_flan_t5_base_text_gen_hf",
                "pt_t5_t5_base_text_gen_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(12, 1, 64)"},
        },
    ),
    (
        Reshape214,
        [((12, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_whisper_openai_whisper_small_speech_recognition_hf",
                "pt_t5_google_flan_t5_base_text_gen_hf",
                "pt_t5_t5_base_text_gen_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 12, 1, 1)"},
        },
    ),
    (
        Reshape215,
        [((1, 12, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_whisper_openai_whisper_small_speech_recognition_hf",
                "pt_t5_google_flan_t5_base_text_gen_hf",
                "pt_t5_t5_base_text_gen_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(12, 1, 1)"},
        },
    ),
    (
        Reshape216,
        [((1, 12, 64, 1), torch.float32)],
        {
            "model_name": [
                "pt_whisper_openai_whisper_small_speech_recognition_hf",
                "pt_t5_google_flan_t5_base_text_gen_hf",
                "pt_t5_t5_base_text_gen_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(12, 64, 1)"},
        },
    ),
    (
        Reshape217,
        [((12, 1, 64), torch.float32)],
        {
            "model_name": [
                "pt_whisper_openai_whisper_small_speech_recognition_hf",
                "pt_t5_google_flan_t5_base_text_gen_hf",
                "pt_t5_t5_base_text_gen_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 12, 1, 64)"},
        },
    ),
    (
        Reshape8,
        [((1, 1, 12, 64), torch.float32)],
        {
            "model_name": [
                "pt_whisper_openai_whisper_small_speech_recognition_hf",
                "pt_t5_google_flan_t5_base_text_gen_hf",
                "pt_t5_t5_base_text_gen_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 768)"},
        },
    ),
    (
        Reshape218,
        [((768, 80, 3), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(768, 80, 3, 1)"},
        },
    ),
    (
        Reshape219,
        [((1, 768, 3000, 1), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 768, 3000)"},
        },
    ),
    (
        Reshape220,
        [((1, 768, 3000), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 768, 3000, 1)"},
        },
    ),
    (
        Reshape221,
        [((768, 768, 3), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(768, 768, 3, 1)"},
        },
    ),
    (
        Reshape222,
        [((1, 768, 1500, 1), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 768, 1500)"},
        },
    ),
    (
        Reshape223,
        [((1, 1500, 768), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1500, 768)"},
        },
    ),
    (
        Reshape224,
        [((1, 1500, 768), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1500, 12, 64)"},
        },
    ),
    (
        Reshape225,
        [((1500, 768), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1500, 768)"},
        },
    ),
    (
        Reshape224,
        [((1500, 768), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1500, 12, 64)"},
        },
    ),
    (
        Reshape226,
        [((1, 12, 1500, 64), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(12, 1500, 64)"},
        },
    ),
    (
        Reshape227,
        [((12, 1500, 1500), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 12, 1500, 1500)"},
        },
    ),
    (
        Reshape228,
        [((1, 12, 1500, 1500), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(12, 1500, 1500)"},
        },
    ),
    (
        Reshape229,
        [((1, 12, 64, 1500), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(12, 64, 1500)"},
        },
    ),
    (
        Reshape230,
        [((12, 1500, 64), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 12, 1500, 64)"},
        },
    ),
    (
        Reshape223,
        [((1, 1500, 12, 64), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1500, 768)"},
        },
    ),
    (
        Reshape231,
        [((12, 1, 1500), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 12, 1, 1500)"},
        },
    ),
    (
        Reshape232,
        [((1, 12, 1, 1500), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(12, 1, 1500)"},
        },
    ),
    (
        Reshape233,
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
        Reshape234,
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
        Reshape235,
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
        Reshape236,
        [((1, 16, 1, 1), torch.float32)],
        {"model_name": ["pt_ssd300_resnet50_base_img_cls_torchhub"], "pcc": 0.99, "op_params": {"shape": "(1, 4, 4)"}},
    ),
    (
        Reshape237,
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
        Reshape238,
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
        Reshape99,
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
        Reshape239,
        [((1024, 80, 3), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_medium_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1024, 80, 3, 1)"},
        },
    ),
    (
        Reshape240,
        [((1, 1024, 3000, 1), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_medium_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1024, 3000)"},
        },
    ),
    (
        Reshape241,
        [((1, 1024, 3000), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_medium_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1024, 3000, 1)"},
        },
    ),
    pytest.param(
        (
            Reshape242,
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
        Reshape243,
        [((1, 1024, 1500, 1), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_medium_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1024, 1500)"},
        },
    ),
    (
        Reshape244,
        [((1, 1500, 1024), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_medium_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1500, 1024)"},
        },
    ),
    (
        Reshape245,
        [((1, 1500, 1024), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_medium_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1500, 16, 64)"},
        },
    ),
    (
        Reshape246,
        [((1500, 1024), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_medium_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1500, 1024)"},
        },
    ),
    (
        Reshape245,
        [((1500, 1024), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_medium_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1500, 16, 64)"},
        },
    ),
    (
        Reshape247,
        [((1, 16, 1500, 64), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_medium_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(16, 1500, 64)"},
        },
    ),
    (
        Reshape248,
        [((16, 1500, 1500), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_medium_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 16, 1500, 1500)"},
        },
    ),
    (
        Reshape249,
        [((1, 16, 1500, 1500), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_medium_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(16, 1500, 1500)"},
        },
    ),
    (
        Reshape250,
        [((1, 16, 64, 1500), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_medium_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(16, 64, 1500)"},
        },
    ),
    (
        Reshape251,
        [((16, 1500, 64), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_medium_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 16, 1500, 64)"},
        },
    ),
    (
        Reshape244,
        [((1, 1500, 16, 64), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_medium_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1500, 1024)"},
        },
    ),
    (
        Reshape252,
        [((16, 1, 1500), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_medium_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 16, 1, 1500)"},
        },
    ),
    (
        Reshape253,
        [((1, 16, 1, 1500), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_medium_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(16, 1, 1500)"},
        },
    ),
    (
        Reshape108,
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
        Reshape254,
        [((1, 1, 512), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_base_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1, 8, 64)"},
        },
    ),
    (
        Reshape255,
        [((1, 1, 512), torch.float32)],
        {
            "model_name": [
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1, 1, 512)"},
        },
    ),
    (
        Reshape256,
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
        Reshape254,
        [((1, 512), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_base_speech_recognition_hf", "pt_t5_t5_small_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1, 8, 64)"},
        },
    ),
    (
        Reshape257,
        [((1, 8, 1, 64), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_base_speech_recognition_hf", "pt_t5_t5_small_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(8, 1, 64)"},
        },
    ),
    (
        Reshape258,
        [((8, 1, 1), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_base_speech_recognition_hf", "pt_t5_t5_small_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 8, 1, 1)"},
        },
    ),
    (
        Reshape259,
        [((1, 8, 1, 1), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_base_speech_recognition_hf", "pt_t5_t5_small_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(8, 1, 1)"},
        },
    ),
    (
        Reshape260,
        [((1, 8, 64, 1), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_base_speech_recognition_hf", "pt_t5_t5_small_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(8, 64, 1)"},
        },
    ),
    (
        Reshape261,
        [((8, 1, 64), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_base_speech_recognition_hf", "pt_t5_t5_small_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 8, 1, 64)"},
        },
    ),
    (
        Reshape108,
        [((1, 1, 8, 64), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_base_speech_recognition_hf", "pt_t5_t5_small_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 512)"},
        },
    ),
    (
        Reshape262,
        [((512, 80, 3), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_base_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(512, 80, 3, 1)"},
        },
    ),
    (
        Reshape263,
        [((1, 512, 3000, 1), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_base_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 512, 3000)"},
        },
    ),
    (
        Reshape264,
        [((1, 512, 3000), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_base_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 512, 3000, 1)"},
        },
    ),
    (
        Reshape265,
        [((512, 512, 3), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_base_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(512, 512, 3, 1)"},
        },
    ),
    (
        Reshape266,
        [((1, 512, 1500, 1), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_base_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 512, 1500)"},
        },
    ),
    (
        Reshape267,
        [((1, 1500, 512), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_base_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1500, 512)"},
        },
    ),
    (
        Reshape268,
        [((1, 1500, 512), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_base_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1500, 8, 64)"},
        },
    ),
    (
        Reshape269,
        [((1500, 512), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_base_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1500, 512)"},
        },
    ),
    (
        Reshape268,
        [((1500, 512), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_base_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1500, 8, 64)"},
        },
    ),
    (
        Reshape270,
        [((1, 8, 1500, 64), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_base_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(8, 1500, 64)"},
        },
    ),
    (
        Reshape271,
        [((8, 1500, 1500), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_base_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 8, 1500, 1500)"},
        },
    ),
    (
        Reshape272,
        [((1, 8, 1500, 1500), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_base_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(8, 1500, 1500)"},
        },
    ),
    (
        Reshape273,
        [((1, 8, 64, 1500), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_base_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(8, 64, 1500)"},
        },
    ),
    (
        Reshape274,
        [((8, 1500, 64), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_base_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 8, 1500, 64)"},
        },
    ),
    (
        Reshape267,
        [((1, 1500, 8, 64), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_base_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1500, 512)"},
        },
    ),
    (
        Reshape275,
        [((8, 1, 1500), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_base_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 8, 1, 1500)"},
        },
    ),
    (
        Reshape276,
        [((1, 8, 1, 1500), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_base_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(8, 1, 1500)"},
        },
    ),
    (
        Reshape277,
        [((1, 2), torch.int64)],
        {
            "model_name": ["pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 2)"},
        },
    ),
    (
        Reshape278,
        [((1, 2, 1280), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(2, 1280)"},
        },
    ),
    (
        Reshape279,
        [((1, 2, 1280), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 2, 20, 64)"},
        },
    ),
    (
        Reshape280,
        [((2, 1280), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 2, 1280)"},
        },
    ),
    (
        Reshape279,
        [((2, 1280), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 2, 20, 64)"},
        },
    ),
    (
        Reshape281,
        [((1, 20, 2, 64), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(20, 2, 64)"},
        },
    ),
    (
        Reshape282,
        [((20, 2, 2), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 20, 2, 2)"},
        },
    ),
    (
        Reshape283,
        [((1, 20, 2, 2), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(20, 2, 2)"},
        },
    ),
    (
        Reshape284,
        [((1, 20, 64, 2), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(20, 64, 2)"},
        },
    ),
    (
        Reshape285,
        [((20, 2, 64), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 20, 2, 64)"},
        },
    ),
    (
        Reshape278,
        [((1, 2, 20, 64), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(2, 1280)"},
        },
    ),
    (
        Reshape286,
        [((20, 2, 1500), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 20, 2, 1500)"},
        },
    ),
    (
        Reshape287,
        [((1, 20, 2, 1500), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(20, 2, 1500)"},
        },
    ),
    (
        Reshape288,
        [((2, 7), torch.int64)],
        {
            "model_name": ["pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text"],
            "pcc": 0.99,
            "op_params": {"shape": "(2, 7)"},
        },
    ),
    (
        Reshape289,
        [((2, 7, 512), torch.float32)],
        {
            "model_name": ["pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text"],
            "pcc": 0.99,
            "op_params": {"shape": "(14, 512)"},
        },
    ),
    (
        Reshape290,
        [((2, 7, 512), torch.float32)],
        {
            "model_name": ["pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text"],
            "pcc": 0.99,
            "op_params": {"shape": "(2, 7, 8, 64)"},
        },
    ),
    (
        Reshape291,
        [((14, 512), torch.float32)],
        {
            "model_name": ["pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text"],
            "pcc": 0.99,
            "op_params": {"shape": "(2, 7, 512)"},
        },
    ),
    (
        Reshape292,
        [((2, 8, 7, 64), torch.float32)],
        {
            "model_name": ["pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text"],
            "pcc": 0.99,
            "op_params": {"shape": "(16, 7, 64)"},
        },
    ),
    (
        Reshape293,
        [((16, 7, 7), torch.float32)],
        {
            "model_name": ["pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text"],
            "pcc": 0.99,
            "op_params": {"shape": "(2, 8, 7, 7)"},
        },
    ),
    (
        Reshape293,
        [((2, 8, 7, 7), torch.float32)],
        {
            "model_name": ["pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text"],
            "pcc": 0.99,
            "op_params": {"shape": "(2, 8, 7, 7)"},
        },
    ),
    (
        Reshape294,
        [((2, 8, 7, 7), torch.float32)],
        {
            "model_name": ["pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text"],
            "pcc": 0.99,
            "op_params": {"shape": "(16, 7, 7)"},
        },
    ),
    (
        Reshape295,
        [((16, 7, 64), torch.float32)],
        {
            "model_name": ["pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text"],
            "pcc": 0.99,
            "op_params": {"shape": "(2, 8, 7, 64)"},
        },
    ),
    (
        Reshape289,
        [((2, 7, 8, 64), torch.float32)],
        {
            "model_name": ["pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text"],
            "pcc": 0.99,
            "op_params": {"shape": "(14, 512)"},
        },
    ),
    (
        Reshape296,
        [((14, 2048), torch.float32)],
        {
            "model_name": ["pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text"],
            "pcc": 0.99,
            "op_params": {"shape": "(2, 7, 2048)"},
        },
    ),
    (
        Reshape297,
        [((2, 7, 2048), torch.float32)],
        {
            "model_name": ["pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text"],
            "pcc": 0.99,
            "op_params": {"shape": "(14, 2048)"},
        },
    ),
    (
        Reshape298,
        [((1, 588, 2048), torch.float32)],
        {
            "model_name": ["pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(588, 2048)"},
        },
    ),
    (
        Reshape299,
        [((588, 2048), torch.float32)],
        {
            "model_name": ["pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 588, 16, 128)"},
        },
    ),
    (
        Reshape300,
        [((588, 2048), torch.float32)],
        {
            "model_name": ["pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 588, 2048)"},
        },
    ),
    (
        Reshape301,
        [((1, 16, 588, 128), torch.float32)],
        {
            "model_name": ["pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(16, 588, 128)"},
        },
    ),
    (
        Reshape302,
        [((16, 588, 588), torch.float32)],
        {
            "model_name": ["pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 16, 588, 588)"},
        },
    ),
    (
        Reshape303,
        [((1, 16, 588, 588), torch.float32)],
        {
            "model_name": ["pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(16, 588, 588)"},
        },
    ),
    (
        Reshape304,
        [((1, 16, 128, 588), torch.float32)],
        {
            "model_name": ["pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(16, 128, 588)"},
        },
    ),
    (
        Reshape305,
        [((16, 588, 128), torch.float32)],
        {
            "model_name": ["pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 16, 588, 128)"},
        },
    ),
    (
        Reshape298,
        [((1, 588, 16, 128), torch.float32)],
        {
            "model_name": ["pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(588, 2048)"},
        },
    ),
    (
        Reshape306,
        [((588, 5504), torch.float32)],
        {
            "model_name": ["pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 588, 5504)"},
        },
    ),
    (
        Reshape307,
        [((1, 39, 4096), torch.float32)],
        {
            "model_name": ["pt_deepseek_deepseek_math_7b_instruct_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(39, 4096)"},
        },
    ),
    (
        Reshape308,
        [((39, 4096), torch.float32)],
        {
            "model_name": ["pt_deepseek_deepseek_math_7b_instruct_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 39, 32, 128)"},
        },
    ),
    (
        Reshape309,
        [((39, 4096), torch.float32)],
        {
            "model_name": ["pt_deepseek_deepseek_math_7b_instruct_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 39, 4096)"},
        },
    ),
    (
        Reshape310,
        [((1, 32, 39, 128), torch.float32)],
        {
            "model_name": ["pt_deepseek_deepseek_math_7b_instruct_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(32, 39, 128)"},
        },
    ),
    (
        Reshape311,
        [((32, 39, 39), torch.float32)],
        {
            "model_name": ["pt_deepseek_deepseek_math_7b_instruct_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 32, 39, 39)"},
        },
    ),
    (
        Reshape312,
        [((1, 32, 39, 39), torch.float32)],
        {
            "model_name": ["pt_deepseek_deepseek_math_7b_instruct_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(32, 39, 39)"},
        },
    ),
    (
        Reshape313,
        [((1, 32, 128, 39), torch.float32)],
        {
            "model_name": ["pt_deepseek_deepseek_math_7b_instruct_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(32, 128, 39)"},
        },
    ),
    (
        Reshape314,
        [((32, 39, 128), torch.float32)],
        {
            "model_name": ["pt_deepseek_deepseek_math_7b_instruct_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 32, 39, 128)"},
        },
    ),
    (
        Reshape307,
        [((1, 39, 32, 128), torch.float32)],
        {
            "model_name": ["pt_deepseek_deepseek_math_7b_instruct_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(39, 4096)"},
        },
    ),
    (
        Reshape315,
        [((39, 11008), torch.float32)],
        {
            "model_name": ["pt_deepseek_deepseek_math_7b_instruct_qa_hf", "pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 39, 11008)"},
        },
    ),
    pytest.param(
        (
            Reshape316,
            [((1, 596, 4096), torch.float32)],
            {
                "model_name": ["pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf"],
                "pcc": 0.99,
                "op_params": {"shape": "(2441216,)"},
            },
        ),
        marks=[
            pytest.mark.xfail(
                reason="RuntimeError: TT_THROW @ /__w/tt-forge-fe/tt-forge-fe/third_party/tt-mlir/third_party/tt-metal/src/tt-metal/tt_metal/impl/program/program.cpp:971: tt::exception info: Statically allocated circular buffers on core range [(x=0,y=0) - (x=7,y=7)] grow to 9895024 B which is beyond max L1 size of 1499136 B"
            )
        ],
    ),
    (
        Reshape317,
        [((1, 596, 4096), torch.float32)],
        {
            "model_name": ["pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(596, 4096)"},
        },
    ),
    (
        Reshape318,
        [((1, 1024, 24, 24), torch.float32)],
        {
            "model_name": ["pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1024, 576, 1)"},
        },
    ),
    (
        Reshape319,
        [((1, 577, 1024), torch.float32)],
        {
            "model_name": ["pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(577, 1024)"},
        },
    ),
    (
        Reshape320,
        [((1, 577, 1024), torch.float32)],
        {
            "model_name": ["pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 577, 16, 64)"},
        },
    ),
    (
        Reshape321,
        [((577, 1024), torch.float32)],
        {
            "model_name": ["pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 577, 1024)"},
        },
    ),
    (
        Reshape322,
        [((1, 16, 577, 64), torch.float32)],
        {
            "model_name": ["pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(16, 577, 64)"},
        },
    ),
    (
        Reshape323,
        [((16, 577, 64), torch.float32)],
        {
            "model_name": ["pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 16, 577, 64)"},
        },
    ),
    (
        Reshape319,
        [((1, 577, 16, 64), torch.float32)],
        {
            "model_name": ["pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(577, 1024)"},
        },
    ),
    pytest.param(
        (
            Reshape324,
            [((1, 576, 4096), torch.float32)],
            {
                "model_name": ["pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf"],
                "pcc": 0.99,
                "op_params": {"shape": "(2359296,)"},
            },
        ),
        marks=[
            pytest.mark.xfail(
                reason="RuntimeError: TT_THROW @ /__w/tt-forge-fe/tt-forge-fe/third_party/tt-mlir/third_party/tt-metal/src/tt-metal/tt_metal/impl/program/program.cpp:971: tt::exception info: Statically allocated circular buffers on core range [(x=0,y=0) - (x=7,y=7)] grow to 9567344 B which is beyond max L1 size of 1499136 B"
            )
        ],
    ),
    (
        Reshape316,
        [((1, 2441216), torch.int32)],
        {
            "model_name": ["pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(2441216,)"},
        },
    ),
    (
        Reshape316,
        [((2441216,), torch.float32)],
        {
            "model_name": ["pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(2441216,)"},
        },
    ),
    pytest.param(
        (
            Reshape325,
            [((2441216,), torch.float32)],
            {
                "model_name": ["pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf"],
                "pcc": 0.99,
                "op_params": {"shape": "(1, 596, 4096)"},
            },
        ),
        marks=[
            pytest.mark.xfail(
                reason="RuntimeError: TT_THROW @ /__w/tt-forge-fe/tt-forge-fe/third_party/tt-mlir/third_party/tt-metal/src/tt-metal/tt_metal/impl/program/program.cpp:971: tt::exception info: Statically allocated circular buffers on core range [(x=6,y=7) - (x=6,y=7)] grow to 9976800 B which is beyond max L1 size of 1499136 B"
            )
        ],
    ),
    (
        Reshape326,
        [((596, 4096), torch.float32)],
        {
            "model_name": ["pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 596, 32, 128)"},
        },
    ),
    (
        Reshape325,
        [((596, 4096), torch.float32)],
        {
            "model_name": ["pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 596, 4096)"},
        },
    ),
    (
        Reshape327,
        [((1, 32, 596, 128), torch.float32)],
        {
            "model_name": ["pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(32, 596, 128)"},
        },
    ),
    (
        Reshape328,
        [((32, 596, 596), torch.float32)],
        {
            "model_name": ["pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 32, 596, 596)"},
        },
    ),
    (
        Reshape329,
        [((1, 32, 596, 596), torch.float32)],
        {
            "model_name": ["pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(32, 596, 596)"},
        },
    ),
    (
        Reshape330,
        [((1, 32, 128, 596), torch.float32)],
        {
            "model_name": ["pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(32, 128, 596)"},
        },
    ),
    (
        Reshape331,
        [((32, 596, 128), torch.float32)],
        {
            "model_name": ["pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 32, 596, 128)"},
        },
    ),
    (
        Reshape317,
        [((1, 596, 32, 128), torch.float32)],
        {
            "model_name": ["pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(596, 4096)"},
        },
    ),
    (
        Reshape332,
        [((596, 11008), torch.float32)],
        {
            "model_name": ["pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 596, 11008)"},
        },
    ),
    (
        Reshape333,
        [((1, 204, 768), torch.float32)],
        {"model_name": ["pt_vilt_dandelin_vilt_b32_mlm_mlm_hf"], "pcc": 0.99, "op_params": {"shape": "(204, 768)"}},
    ),
    (
        Reshape334,
        [((1, 204, 768), torch.float32)],
        {
            "model_name": ["pt_vilt_dandelin_vilt_b32_mlm_mlm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 204, 12, 64)"},
        },
    ),
    (
        Reshape335,
        [((204, 768), torch.float32)],
        {"model_name": ["pt_vilt_dandelin_vilt_b32_mlm_mlm_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 204, 768)"}},
    ),
    (
        Reshape336,
        [((1, 12, 204, 64), torch.float32)],
        {"model_name": ["pt_vilt_dandelin_vilt_b32_mlm_mlm_hf"], "pcc": 0.99, "op_params": {"shape": "(12, 204, 64)"}},
    ),
    (
        Reshape337,
        [((12, 204, 204), torch.float32)],
        {
            "model_name": ["pt_vilt_dandelin_vilt_b32_mlm_mlm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 12, 204, 204)"},
        },
    ),
    (
        Reshape338,
        [((1, 12, 204, 204), torch.float32)],
        {"model_name": ["pt_vilt_dandelin_vilt_b32_mlm_mlm_hf"], "pcc": 0.99, "op_params": {"shape": "(12, 204, 204)"}},
    ),
    (
        Reshape339,
        [((1, 12, 64, 204), torch.float32)],
        {"model_name": ["pt_vilt_dandelin_vilt_b32_mlm_mlm_hf"], "pcc": 0.99, "op_params": {"shape": "(12, 64, 204)"}},
    ),
    (
        Reshape340,
        [((12, 204, 64), torch.float32)],
        {
            "model_name": ["pt_vilt_dandelin_vilt_b32_mlm_mlm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 12, 204, 64)"},
        },
    ),
    (
        Reshape333,
        [((1, 204, 12, 64), torch.float32)],
        {"model_name": ["pt_vilt_dandelin_vilt_b32_mlm_mlm_hf"], "pcc": 0.99, "op_params": {"shape": "(204, 768)"}},
    ),
    (
        Reshape341,
        [((1, 201, 768), torch.float32)],
        {
            "model_name": ["pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(201, 768)"},
        },
    ),
    (
        Reshape342,
        [((1, 201, 768), torch.float32)],
        {
            "model_name": ["pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 201, 12, 64)"},
        },
    ),
    (
        Reshape343,
        [((201, 768), torch.float32)],
        {
            "model_name": ["pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 201, 768)"},
        },
    ),
    (
        Reshape344,
        [((1, 12, 201, 64), torch.float32)],
        {
            "model_name": ["pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(12, 201, 64)"},
        },
    ),
    (
        Reshape345,
        [((12, 201, 201), torch.float32)],
        {
            "model_name": ["pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 12, 201, 201)"},
        },
    ),
    (
        Reshape346,
        [((1, 12, 201, 201), torch.float32)],
        {
            "model_name": ["pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(12, 201, 201)"},
        },
    ),
    (
        Reshape347,
        [((1, 12, 64, 201), torch.float32)],
        {
            "model_name": ["pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(12, 64, 201)"},
        },
    ),
    (
        Reshape348,
        [((12, 201, 64), torch.float32)],
        {
            "model_name": ["pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 12, 201, 64)"},
        },
    ),
    (
        Reshape341,
        [((1, 201, 12, 64), torch.float32)],
        {
            "model_name": ["pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(201, 768)"},
        },
    ),
    (
        Reshape349,
        [((1, 128, 2048), torch.float32)],
        {
            "model_name": [
                "pt_albert_xlarge_v1_mlm_hf",
                "pt_albert_xlarge_v2_token_cls_hf",
                "pt_albert_xlarge_v1_token_cls_hf",
                "pt_albert_xlarge_v2_mlm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(128, 2048)"},
        },
    ),
    (
        Reshape350,
        [((1, 128, 2048), torch.float32)],
        {
            "model_name": [
                "pt_albert_xlarge_v1_mlm_hf",
                "pt_albert_xlarge_v2_token_cls_hf",
                "pt_albert_xlarge_v1_token_cls_hf",
                "pt_albert_xlarge_v2_mlm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 128, 16, 128)"},
        },
    ),
    (
        Reshape351,
        [((128, 2048), torch.float32)],
        {
            "model_name": [
                "pt_albert_xlarge_v1_mlm_hf",
                "pt_albert_xlarge_v2_token_cls_hf",
                "pt_albert_xlarge_v1_token_cls_hf",
                "pt_albert_xlarge_v2_mlm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 128, 2048)"},
        },
    ),
    (
        Reshape352,
        [((1, 16, 128, 128), torch.float32)],
        {
            "model_name": [
                "pt_albert_xlarge_v1_mlm_hf",
                "pt_albert_large_v2_mlm_hf",
                "pt_albert_large_v1_mlm_hf",
                "pt_albert_large_v2_token_cls_hf",
                "pt_albert_xlarge_v2_token_cls_hf",
                "pt_albert_xlarge_v1_token_cls_hf",
                "pt_albert_large_v1_token_cls_hf",
                "pt_albert_xlarge_v2_mlm_hf",
                "pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(16, 128, 128)"},
        },
    ),
    (
        Reshape353,
        [((16, 128, 128), torch.float32)],
        {
            "model_name": [
                "pt_albert_xlarge_v1_mlm_hf",
                "pt_albert_large_v2_mlm_hf",
                "pt_albert_large_v1_mlm_hf",
                "pt_albert_large_v2_token_cls_hf",
                "pt_albert_xlarge_v2_token_cls_hf",
                "pt_albert_xlarge_v1_token_cls_hf",
                "pt_albert_large_v1_token_cls_hf",
                "pt_albert_xlarge_v2_mlm_hf",
                "pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 16, 128, 128)"},
        },
    ),
    (
        Reshape354,
        [((1, 128, 16, 128), torch.float32)],
        {
            "model_name": [
                "pt_albert_xlarge_v1_mlm_hf",
                "pt_albert_xlarge_v2_token_cls_hf",
                "pt_albert_xlarge_v1_token_cls_hf",
                "pt_albert_xlarge_v2_mlm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 128, 2048, 1)"},
        },
    ),
    (
        Reshape355,
        [((1, 128, 1024), torch.float32)],
        {
            "model_name": [
                "pt_albert_large_v2_mlm_hf",
                "pt_albert_large_v1_mlm_hf",
                "pt_albert_large_v2_token_cls_hf",
                "pt_albert_large_v1_token_cls_hf",
                "pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(128, 1024)"},
        },
    ),
    (
        Reshape356,
        [((1, 128, 1024), torch.float32)],
        {
            "model_name": [
                "pt_albert_large_v2_mlm_hf",
                "pt_albert_large_v1_mlm_hf",
                "pt_albert_large_v2_token_cls_hf",
                "pt_albert_large_v1_token_cls_hf",
                "pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 128, 16, 64)"},
        },
    ),
    (
        Reshape357,
        [((128, 1024), torch.float32)],
        {
            "model_name": [
                "pt_albert_large_v2_mlm_hf",
                "pt_albert_large_v1_mlm_hf",
                "pt_albert_large_v2_token_cls_hf",
                "pt_albert_large_v1_token_cls_hf",
                "pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 128, 1024)"},
        },
    ),
    (
        Reshape358,
        [((128, 1024), torch.float32)],
        {
            "model_name": ["pt_mistral_mistralai_mistral_7b_v0_1_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 128, 8, 128)"},
        },
    ),
    (
        Reshape359,
        [((1, 16, 128, 64), torch.float32)],
        {
            "model_name": [
                "pt_albert_large_v2_mlm_hf",
                "pt_albert_large_v1_mlm_hf",
                "pt_albert_large_v2_token_cls_hf",
                "pt_albert_large_v1_token_cls_hf",
                "pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(16, 128, 64)"},
        },
    ),
    (
        Reshape360,
        [((1, 16, 64, 128), torch.float32)],
        {
            "model_name": [
                "pt_albert_large_v2_mlm_hf",
                "pt_albert_large_v1_mlm_hf",
                "pt_albert_large_v2_token_cls_hf",
                "pt_albert_large_v1_token_cls_hf",
                "pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(16, 64, 128)"},
        },
    ),
    (
        Reshape361,
        [((16, 128, 64), torch.float32)],
        {
            "model_name": [
                "pt_albert_large_v2_mlm_hf",
                "pt_albert_large_v1_mlm_hf",
                "pt_albert_large_v2_token_cls_hf",
                "pt_albert_large_v1_token_cls_hf",
                "pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 16, 128, 64)"},
        },
    ),
    (
        Reshape362,
        [((1, 128, 16, 64), torch.float32)],
        {
            "model_name": [
                "pt_albert_large_v2_mlm_hf",
                "pt_albert_large_v1_mlm_hf",
                "pt_albert_large_v2_token_cls_hf",
                "pt_albert_large_v1_token_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 128, 1024, 1)"},
        },
    ),
    (
        Reshape355,
        [((1, 128, 16, 64), torch.float32)],
        {
            "model_name": ["pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(128, 1024)"},
        },
    ),
    (
        Reshape363,
        [((1, 14, 768), torch.float32)],
        {
            "model_name": ["pt_albert_twmkn9_albert_base_v2_squad2_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(14, 768)"},
        },
    ),
    (
        Reshape364,
        [((1, 14, 768), torch.float32)],
        {
            "model_name": ["pt_albert_twmkn9_albert_base_v2_squad2_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 14, 12, 64)"},
        },
    ),
    (
        Reshape365,
        [((14, 768), torch.float32)],
        {
            "model_name": ["pt_albert_twmkn9_albert_base_v2_squad2_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 14, 768)"},
        },
    ),
    (
        Reshape366,
        [((1, 12, 14, 64), torch.float32)],
        {
            "model_name": ["pt_albert_twmkn9_albert_base_v2_squad2_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(12, 14, 64)"},
        },
    ),
    (
        Reshape367,
        [((12, 14, 14), torch.float32)],
        {
            "model_name": ["pt_albert_twmkn9_albert_base_v2_squad2_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 12, 14, 14)"},
        },
    ),
    (
        Reshape368,
        [((1, 12, 14, 14), torch.float32)],
        {
            "model_name": ["pt_albert_twmkn9_albert_base_v2_squad2_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(12, 14, 14)"},
        },
    ),
    (
        Reshape369,
        [((1, 12, 64, 14), torch.float32)],
        {
            "model_name": ["pt_albert_twmkn9_albert_base_v2_squad2_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(12, 64, 14)"},
        },
    ),
    (
        Reshape370,
        [((12, 14, 64), torch.float32)],
        {
            "model_name": ["pt_albert_twmkn9_albert_base_v2_squad2_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 12, 14, 64)"},
        },
    ),
    (
        Reshape371,
        [((1, 14, 12, 64), torch.float32)],
        {
            "model_name": ["pt_albert_twmkn9_albert_base_v2_squad2_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 14, 768, 1)"},
        },
    ),
    (
        Reshape372,
        [((14, 1), torch.float32)],
        {
            "model_name": ["pt_albert_twmkn9_albert_base_v2_squad2_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 14, 1)"},
        },
    ),
    (
        Reshape373,
        [((1, 9, 768), torch.float32)],
        {
            "model_name": ["pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(9, 768)"},
        },
    ),
    (
        Reshape374,
        [((1, 9, 768), torch.float32)],
        {
            "model_name": ["pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 9, 12, 64)"},
        },
    ),
    (
        Reshape375,
        [((9, 768), torch.float32)],
        {
            "model_name": ["pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 9, 768)"},
        },
    ),
    (
        Reshape376,
        [((1, 12, 9, 64), torch.float32)],
        {
            "model_name": ["pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(12, 9, 64)"},
        },
    ),
    (
        Reshape377,
        [((12, 9, 9), torch.float32)],
        {
            "model_name": ["pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 12, 9, 9)"},
        },
    ),
    (
        Reshape378,
        [((1, 12, 9, 9), torch.float32)],
        {
            "model_name": ["pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(12, 9, 9)"},
        },
    ),
    (
        Reshape379,
        [((1, 12, 64, 9), torch.float32)],
        {
            "model_name": ["pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(12, 64, 9)"},
        },
    ),
    (
        Reshape380,
        [((12, 9, 64), torch.float32)],
        {
            "model_name": ["pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 12, 9, 64)"},
        },
    ),
    (
        Reshape381,
        [((1, 9, 12, 64), torch.float32)],
        {
            "model_name": ["pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 9, 768, 1)"},
        },
    ),
    (
        Reshape382,
        [((1, 128, 4096), torch.float32)],
        {
            "model_name": [
                "pt_albert_xxlarge_v2_token_cls_hf",
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_albert_xxlarge_v1_mlm_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
                "pt_mistral_mistralai_mistral_7b_v0_1_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(128, 4096)"},
        },
    ),
    (
        Reshape383,
        [((1, 128, 4096), torch.float32)],
        {
            "model_name": [
                "pt_albert_xxlarge_v2_token_cls_hf",
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_albert_xxlarge_v1_mlm_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 128, 64, 64)"},
        },
    ),
    (
        Reshape384,
        [((128, 4096), torch.float32)],
        {
            "model_name": [
                "pt_albert_xxlarge_v2_token_cls_hf",
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_albert_xxlarge_v1_mlm_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
                "pt_mistral_mistralai_mistral_7b_v0_1_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 128, 4096)"},
        },
    ),
    (
        Reshape385,
        [((128, 4096), torch.float32)],
        {
            "model_name": ["pt_mistral_mistralai_mistral_7b_v0_1_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 128, 32, 128)"},
        },
    ),
    (
        Reshape386,
        [((1, 64, 128, 64), torch.float32)],
        {
            "model_name": [
                "pt_albert_xxlarge_v2_token_cls_hf",
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_albert_xxlarge_v1_mlm_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(64, 128, 64)"},
        },
    ),
    (
        Reshape387,
        [((64, 128, 128), torch.float32)],
        {
            "model_name": [
                "pt_albert_xxlarge_v2_token_cls_hf",
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_albert_xxlarge_v1_mlm_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 64, 128, 128)"},
        },
    ),
    (
        Reshape388,
        [((1, 64, 128, 128), torch.float32)],
        {
            "model_name": [
                "pt_albert_xxlarge_v2_token_cls_hf",
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_albert_xxlarge_v1_mlm_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(64, 128, 128)"},
        },
    ),
    (
        Reshape389,
        [((1, 64, 128, 128), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 64, 16384, 1)"},
        },
    ),
    (
        Reshape390,
        [((1, 64, 64, 128), torch.float32)],
        {
            "model_name": [
                "pt_albert_xxlarge_v2_token_cls_hf",
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_albert_xxlarge_v1_mlm_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(64, 64, 128)"},
        },
    ),
    (
        Reshape391,
        [((64, 128, 64), torch.float32)],
        {
            "model_name": [
                "pt_albert_xxlarge_v2_token_cls_hf",
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_albert_xxlarge_v1_mlm_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 64, 128, 64)"},
        },
    ),
    (
        Reshape392,
        [((1, 128, 64, 64), torch.float32)],
        {
            "model_name": [
                "pt_albert_xxlarge_v2_token_cls_hf",
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_albert_xxlarge_v1_mlm_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 128, 4096, 1)"},
        },
    ),
    (
        Reshape393,
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
        Reshape394,
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
        Reshape395,
        [((1, 256, 1024), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 256, 32, 32)"},
        },
    ),
    (
        Reshape396,
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
        Reshape397,
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
        Reshape398,
        [((256, 1024), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_1_8b_clm_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_clm_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_clm_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 256, 8, 128)"},
        },
    ),
    (
        Reshape399,
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
        Reshape400,
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
        Reshape401,
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
        Reshape402,
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
        Reshape393,
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
        Reshape403,
        [((1, 256), torch.int64)],
        {
            "model_name": [
                "pt_bart_facebook_bart_large_mnli_seq_cls_hf",
                "pt_gpt2_gpt2_text_gen_hf",
                "pt_opt_facebook_opt_1_3b_clm_hf",
                "pt_opt_facebook_opt_350m_clm_hf",
                "pt_opt_facebook_opt_125m_clm_hf",
                "pt_xglm_facebook_xglm_1_7b_clm_hf",
                "pt_xglm_facebook_xglm_564m_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 256)"},
        },
    ),
    (
        Reshape404,
        [((1, 32, 4608), torch.float32)],
        {
            "model_name": ["pt_bloom_bigscience_bloom_1b1_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 32, 16, 3, 96)"},
        },
    ),
    (
        Reshape405,
        [((1, 32, 16, 1, 96), torch.float32)],
        {
            "model_name": ["pt_bloom_bigscience_bloom_1b1_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 32, 16, 96)"},
        },
    ),
    (
        Reshape406,
        [((1, 16, 32, 96), torch.float32)],
        {"model_name": ["pt_bloom_bigscience_bloom_1b1_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(16, 32, 96)"}},
    ),
    (
        Reshape407,
        [((1, 16, 32), torch.float32)],
        {"model_name": ["pt_bloom_bigscience_bloom_1b1_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(16, 1, 32)"}},
    ),
    (
        Reshape408,
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
        Reshape409,
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
        Reshape410,
        [((16, 32, 96), torch.float32)],
        {
            "model_name": ["pt_bloom_bigscience_bloom_1b1_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 16, 32, 96)"},
        },
    ),
    (
        Reshape411,
        [((1, 32, 16, 96), torch.float32)],
        {"model_name": ["pt_bloom_bigscience_bloom_1b1_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(32, 1536)"}},
    ),
    (
        Reshape412,
        [((32, 1536), torch.float32)],
        {"model_name": ["pt_bloom_bigscience_bloom_1b1_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 32, 1536)"}},
    ),
    (
        Reshape394,
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
        Reshape413,
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
        Reshape414,
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
        Reshape415,
        [((256, 4096), torch.float32)],
        {
            "model_name": [
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
                "pt_codegen_salesforce_codegen_350m_mono_clm_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_clm_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_clm_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_clm_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 256, 4096)"},
        },
    ),
    (
        Reshape416,
        [((256, 4096), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_1_8b_clm_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_clm_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_clm_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 256, 32, 128)"},
        },
    ),
    (
        Reshape417,
        [((1, 384, 768), torch.float32)],
        {
            "model_name": ["pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(384, 768)"},
        },
    ),
    (
        Reshape418,
        [((1, 384, 768), torch.float32)],
        {
            "model_name": ["pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 384, 12, 64)"},
        },
    ),
    (
        Reshape419,
        [((384, 768), torch.float32)],
        {
            "model_name": ["pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 384, 768)"},
        },
    ),
    (
        Reshape420,
        [((1, 12, 384, 64), torch.float32)],
        {
            "model_name": ["pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(12, 384, 64)"},
        },
    ),
    (
        Reshape421,
        [((12, 384, 384), torch.float32)],
        {
            "model_name": ["pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 12, 384, 384)"},
        },
    ),
    (
        Reshape422,
        [((1, 384), torch.bool)],
        {
            "model_name": ["pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1, 1, 384)"},
        },
    ),
    (
        Reshape423,
        [((1, 12, 384, 384), torch.float32)],
        {
            "model_name": ["pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(12, 384, 384)"},
        },
    ),
    (
        Reshape424,
        [((1, 12, 64, 384), torch.float32)],
        {
            "model_name": ["pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(12, 64, 384)"},
        },
    ),
    (
        Reshape425,
        [((12, 384, 64), torch.float32)],
        {
            "model_name": ["pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 12, 384, 64)"},
        },
    ),
    (
        Reshape417,
        [((1, 384, 12, 64), torch.float32)],
        {
            "model_name": ["pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(384, 768)"},
        },
    ),
    (
        Reshape426,
        [((1, 128), torch.bool)],
        {
            "model_name": [
                "pt_distilbert_distilbert_base_multilingual_cased_mlm_hf",
                "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
                "pt_distilbert_distilbert_base_uncased_mlm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1, 1, 128)"},
        },
    ),
    (
        Reshape427,
        [((128, 1), torch.float32)],
        {
            "model_name": [
                "pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 128, 1)"},
        },
    ),
    (
        Reshape428,
        [((1, 128), torch.float32)],
        {
            "model_name": [
                "pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 128)"},
        },
    ),
    (
        Reshape429,
        [((1, 1), torch.float32)],
        {
            "model_name": [
                "pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1,)"},
        },
    ),
    (
        Reshape430,
        [((1, 10, 2048), torch.float32)],
        {"model_name": ["pt_falcon3_tiiuae_falcon3_1b_base_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(10, 2048)"}},
    ),
    (
        Reshape431,
        [((10, 2048), torch.float32)],
        {
            "model_name": ["pt_falcon3_tiiuae_falcon3_1b_base_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 10, 8, 256)"},
        },
    ),
    (
        Reshape432,
        [((10, 2048), torch.float32)],
        {
            "model_name": ["pt_falcon3_tiiuae_falcon3_1b_base_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 10, 2048)"},
        },
    ),
    (
        Reshape433,
        [((1, 8, 10, 256), torch.float32)],
        {
            "model_name": ["pt_falcon3_tiiuae_falcon3_1b_base_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(8, 10, 256)"},
        },
    ),
    (
        Reshape434,
        [((10, 1024), torch.float32)],
        {
            "model_name": [
                "pt_falcon3_tiiuae_falcon3_1b_base_clm_hf",
                "pt_falcon3_tiiuae_falcon3_3b_base_clm_hf",
                "pt_falcon3_tiiuae_falcon3_7b_base_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 10, 4, 256)"},
        },
    ),
    (
        Reshape433,
        [((1, 4, 2, 10, 256), torch.float32)],
        {
            "model_name": ["pt_falcon3_tiiuae_falcon3_1b_base_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(8, 10, 256)"},
        },
    ),
    (
        Reshape435,
        [((1, 4, 2, 10, 256), torch.float32)],
        {
            "model_name": ["pt_falcon3_tiiuae_falcon3_1b_base_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 8, 10, 256)"},
        },
    ),
    (
        Reshape436,
        [((8, 10, 10), torch.float32)],
        {
            "model_name": ["pt_falcon3_tiiuae_falcon3_1b_base_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 8, 10, 10)"},
        },
    ),
    (
        Reshape437,
        [((1, 8, 10, 10), torch.float32)],
        {
            "model_name": ["pt_falcon3_tiiuae_falcon3_1b_base_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(8, 10, 10)"},
        },
    ),
    (
        Reshape438,
        [((1, 8, 256, 10), torch.float32)],
        {
            "model_name": ["pt_falcon3_tiiuae_falcon3_1b_base_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(8, 256, 10)"},
        },
    ),
    (
        Reshape435,
        [((8, 10, 256), torch.float32)],
        {
            "model_name": ["pt_falcon3_tiiuae_falcon3_1b_base_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 8, 10, 256)"},
        },
    ),
    (
        Reshape430,
        [((1, 10, 8, 256), torch.float32)],
        {"model_name": ["pt_falcon3_tiiuae_falcon3_1b_base_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(10, 2048)"}},
    ),
    (
        Reshape439,
        [((10, 8192), torch.float32)],
        {
            "model_name": ["pt_falcon3_tiiuae_falcon3_1b_base_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 10, 8192)"},
        },
    ),
    (
        Reshape440,
        [((1, 6, 4544), torch.float32)],
        {
            "model_name": ["pt_falcon_tiiuae_falcon_7b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(6, 4544)"},
        },
    ),
    (
        Reshape441,
        [((6, 18176), torch.float32)],
        {
            "model_name": ["pt_falcon_tiiuae_falcon_7b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 6, 18176)"},
        },
    ),
    (
        Reshape442,
        [((6, 4672), torch.float32)],
        {
            "model_name": ["pt_falcon_tiiuae_falcon_7b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 6, 73, 64)"},
        },
    ),
    (
        Reshape443,
        [((1, 71, 6, 64), torch.float32)],
        {
            "model_name": ["pt_falcon_tiiuae_falcon_7b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 71, 6, 64)"},
        },
    ),
    (
        Reshape444,
        [((1, 71, 6, 64), torch.float32)],
        {
            "model_name": ["pt_falcon_tiiuae_falcon_7b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(71, 6, 64)"},
        },
    ),
    (
        Reshape445,
        [((71, 6, 6), torch.float32)],
        {
            "model_name": ["pt_falcon_tiiuae_falcon_7b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 71, 6, 6)"},
        },
    ),
    (
        Reshape446,
        [((1, 71, 6, 6), torch.float32)],
        {
            "model_name": ["pt_falcon_tiiuae_falcon_7b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(71, 6, 6)"},
        },
    ),
    (
        Reshape443,
        [((71, 6, 64), torch.float32)],
        {
            "model_name": ["pt_falcon_tiiuae_falcon_7b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 71, 6, 64)"},
        },
    ),
    (
        Reshape440,
        [((1, 6, 71, 64), torch.float32)],
        {
            "model_name": ["pt_falcon_tiiuae_falcon_7b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(6, 4544)"},
        },
    ),
    (
        Reshape447,
        [((6, 4544), torch.float32)],
        {
            "model_name": ["pt_falcon_tiiuae_falcon_7b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 6, 4544)"},
        },
    ),
    (
        Reshape448,
        [((1, 10, 3072), torch.float32)],
        {
            "model_name": ["pt_falcon3_tiiuae_falcon3_3b_base_clm_hf", "pt_falcon3_tiiuae_falcon3_7b_base_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(10, 3072)"},
        },
    ),
    (
        Reshape449,
        [((10, 3072), torch.float32)],
        {
            "model_name": ["pt_falcon3_tiiuae_falcon3_3b_base_clm_hf", "pt_falcon3_tiiuae_falcon3_7b_base_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 10, 12, 256)"},
        },
    ),
    (
        Reshape450,
        [((10, 3072), torch.float32)],
        {
            "model_name": ["pt_falcon3_tiiuae_falcon3_3b_base_clm_hf", "pt_falcon3_tiiuae_falcon3_7b_base_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 10, 3072)"},
        },
    ),
    (
        Reshape451,
        [((1, 12, 10, 256), torch.float32)],
        {
            "model_name": ["pt_falcon3_tiiuae_falcon3_3b_base_clm_hf", "pt_falcon3_tiiuae_falcon3_7b_base_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(12, 10, 256)"},
        },
    ),
    (
        Reshape451,
        [((1, 4, 3, 10, 256), torch.float32)],
        {
            "model_name": ["pt_falcon3_tiiuae_falcon3_3b_base_clm_hf", "pt_falcon3_tiiuae_falcon3_7b_base_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(12, 10, 256)"},
        },
    ),
    (
        Reshape452,
        [((1, 4, 3, 10, 256), torch.float32)],
        {
            "model_name": ["pt_falcon3_tiiuae_falcon3_3b_base_clm_hf", "pt_falcon3_tiiuae_falcon3_7b_base_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 12, 10, 256)"},
        },
    ),
    (
        Reshape453,
        [((12, 10, 10), torch.float32)],
        {
            "model_name": ["pt_falcon3_tiiuae_falcon3_3b_base_clm_hf", "pt_falcon3_tiiuae_falcon3_7b_base_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 12, 10, 10)"},
        },
    ),
    (
        Reshape454,
        [((1, 12, 10, 10), torch.float32)],
        {
            "model_name": ["pt_falcon3_tiiuae_falcon3_3b_base_clm_hf", "pt_falcon3_tiiuae_falcon3_7b_base_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(12, 10, 10)"},
        },
    ),
    (
        Reshape455,
        [((1, 12, 256, 10), torch.float32)],
        {
            "model_name": ["pt_falcon3_tiiuae_falcon3_3b_base_clm_hf", "pt_falcon3_tiiuae_falcon3_7b_base_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(12, 256, 10)"},
        },
    ),
    (
        Reshape452,
        [((12, 10, 256), torch.float32)],
        {
            "model_name": ["pt_falcon3_tiiuae_falcon3_3b_base_clm_hf", "pt_falcon3_tiiuae_falcon3_7b_base_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 12, 10, 256)"},
        },
    ),
    (
        Reshape448,
        [((1, 10, 12, 256), torch.float32)],
        {
            "model_name": ["pt_falcon3_tiiuae_falcon3_3b_base_clm_hf", "pt_falcon3_tiiuae_falcon3_7b_base_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(10, 3072)"},
        },
    ),
    (
        Reshape456,
        [((10, 9216), torch.float32)],
        {
            "model_name": ["pt_falcon3_tiiuae_falcon3_3b_base_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 10, 9216)"},
        },
    ),
    (
        Reshape457,
        [((10, 23040), torch.float32)],
        {
            "model_name": ["pt_falcon3_tiiuae_falcon3_7b_base_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 10, 23040)"},
        },
    ),
    (
        Reshape458,
        [((1, 334, 12288), torch.float32)],
        {"model_name": ["pt_fuyu_adept_fuyu_8b_qa_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 334, 64, 3, 64)"}},
    ),
    (
        Reshape459,
        [((1, 334, 64, 1, 64), torch.float32)],
        {"model_name": ["pt_fuyu_adept_fuyu_8b_qa_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 334, 64, 64)"}},
    ),
    (
        Reshape460,
        [((1, 64, 334, 64), torch.float32)],
        {"model_name": ["pt_fuyu_adept_fuyu_8b_qa_hf"], "pcc": 0.99, "op_params": {"shape": "(64, 334, 64)"}},
    ),
    (
        Reshape461,
        [((64, 334, 334), torch.float32)],
        {"model_name": ["pt_fuyu_adept_fuyu_8b_qa_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 64, 334, 334)"}},
    ),
    (
        Reshape462,
        [((1, 64, 334, 334), torch.float32)],
        {"model_name": ["pt_fuyu_adept_fuyu_8b_qa_hf"], "pcc": 0.99, "op_params": {"shape": "(64, 334, 334)"}},
    ),
    (
        Reshape463,
        [((1, 64, 64, 334), torch.float32)],
        {"model_name": ["pt_fuyu_adept_fuyu_8b_qa_hf"], "pcc": 0.99, "op_params": {"shape": "(64, 64, 334)"}},
    ),
    (
        Reshape464,
        [((64, 334, 64), torch.float32)],
        {"model_name": ["pt_fuyu_adept_fuyu_8b_qa_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 64, 334, 64)"}},
    ),
    (
        Reshape465,
        [((1, 334, 64, 64), torch.float32)],
        {"model_name": ["pt_fuyu_adept_fuyu_8b_qa_hf"], "pcc": 0.99, "op_params": {"shape": "(334, 4096)"}},
    ),
    (
        Reshape466,
        [((334, 4096), torch.float32)],
        {"model_name": ["pt_fuyu_adept_fuyu_8b_qa_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 334, 4096)"}},
    ),
    (
        Reshape467,
        [((1, 207, 3584), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_2_9b_it_qa_hf"], "pcc": 0.99, "op_params": {"shape": "(207, 3584)"}},
    ),
    (
        Reshape468,
        [((207, 4096), torch.float32)],
        {
            "model_name": ["pt_gemma_google_gemma_2_9b_it_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 207, 16, 256)"},
        },
    ),
    (
        Reshape469,
        [((1, 16, 207, 256), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_2_9b_it_qa_hf"], "pcc": 0.99, "op_params": {"shape": "(16, 207, 256)"}},
    ),
    (
        Reshape470,
        [((207, 2048), torch.float32)],
        {
            "model_name": ["pt_gemma_google_gemma_2_9b_it_qa_hf", "pt_gemma_google_gemma_2_2b_it_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 207, 8, 256)"},
        },
    ),
    (
        Reshape469,
        [((1, 8, 2, 207, 256), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_2_9b_it_qa_hf"], "pcc": 0.99, "op_params": {"shape": "(16, 207, 256)"}},
    ),
    (
        Reshape471,
        [((1, 8, 2, 207, 256), torch.float32)],
        {
            "model_name": ["pt_gemma_google_gemma_2_9b_it_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 16, 207, 256)"},
        },
    ),
    (
        Reshape472,
        [((16, 207, 207), torch.float32)],
        {
            "model_name": ["pt_gemma_google_gemma_2_9b_it_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 16, 207, 207)"},
        },
    ),
    (
        Reshape473,
        [((1, 16, 207, 207), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_2_9b_it_qa_hf"], "pcc": 0.99, "op_params": {"shape": "(16, 207, 207)"}},
    ),
    (
        Reshape474,
        [((1, 16, 256, 207), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_2_9b_it_qa_hf"], "pcc": 0.99, "op_params": {"shape": "(16, 256, 207)"}},
    ),
    (
        Reshape471,
        [((16, 207, 256), torch.float32)],
        {
            "model_name": ["pt_gemma_google_gemma_2_9b_it_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 16, 207, 256)"},
        },
    ),
    (
        Reshape475,
        [((1, 207, 16, 256), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_2_9b_it_qa_hf"], "pcc": 0.99, "op_params": {"shape": "(207, 4096)"}},
    ),
    (
        Reshape476,
        [((207, 3584), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_2_9b_it_qa_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 207, 3584)"}},
    ),
    (
        Reshape477,
        [((207, 14336), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_2_9b_it_qa_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 207, 14336)"}},
    ),
    (
        Reshape478,
        [((1, 7, 2048), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_2b_text_gen_hf"], "pcc": 0.99, "op_params": {"shape": "(7, 2048)"}},
    ),
    (
        Reshape479,
        [((7, 2048), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_2b_text_gen_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 7, 8, 256)"}},
    ),
    (
        Reshape480,
        [((7, 2048), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_2b_text_gen_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 7, 2048)"}},
    ),
    (
        Reshape481,
        [((1, 8, 7, 256), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_2b_text_gen_hf"], "pcc": 0.99, "op_params": {"shape": "(8, 7, 256)"}},
    ),
    (
        Reshape482,
        [((7, 256), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_2b_text_gen_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 7, 1, 256)"}},
    ),
    (
        Reshape481,
        [((1, 1, 8, 7, 256), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_2b_text_gen_hf"], "pcc": 0.99, "op_params": {"shape": "(8, 7, 256)"}},
    ),
    (
        Reshape483,
        [((1, 1, 8, 7, 256), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_2b_text_gen_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 8, 7, 256)"}},
    ),
    (
        Reshape484,
        [((8, 7, 7), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_2b_text_gen_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 8, 7, 7)"}},
    ),
    (
        Reshape485,
        [((1, 8, 7, 7), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_2b_text_gen_hf"], "pcc": 0.99, "op_params": {"shape": "(8, 7, 7)"}},
    ),
    (
        Reshape486,
        [((1, 8, 256, 7), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_2b_text_gen_hf"], "pcc": 0.99, "op_params": {"shape": "(8, 256, 7)"}},
    ),
    (
        Reshape483,
        [((8, 7, 256), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_2b_text_gen_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 8, 7, 256)"}},
    ),
    (
        Reshape478,
        [((1, 7, 8, 256), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_2b_text_gen_hf"], "pcc": 0.99, "op_params": {"shape": "(7, 2048)"}},
    ),
    (
        Reshape487,
        [((7, 16384), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_2b_text_gen_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 7, 16384)"}},
    ),
    (
        Reshape488,
        [((1, 207, 2304), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_2_2b_it_qa_hf"], "pcc": 0.99, "op_params": {"shape": "(207, 2304)"}},
    ),
    (
        Reshape489,
        [((1, 8, 207, 256), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_2_2b_it_qa_hf"], "pcc": 0.99, "op_params": {"shape": "(8, 207, 256)"}},
    ),
    (
        Reshape490,
        [((207, 1024), torch.float32)],
        {
            "model_name": ["pt_gemma_google_gemma_2_2b_it_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 207, 4, 256)"},
        },
    ),
    (
        Reshape489,
        [((1, 4, 2, 207, 256), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_2_2b_it_qa_hf"], "pcc": 0.99, "op_params": {"shape": "(8, 207, 256)"}},
    ),
    (
        Reshape491,
        [((1, 4, 2, 207, 256), torch.float32)],
        {
            "model_name": ["pt_gemma_google_gemma_2_2b_it_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 8, 207, 256)"},
        },
    ),
    (
        Reshape492,
        [((8, 207, 207), torch.float32)],
        {
            "model_name": ["pt_gemma_google_gemma_2_2b_it_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 8, 207, 207)"},
        },
    ),
    (
        Reshape493,
        [((1, 8, 207, 207), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_2_2b_it_qa_hf"], "pcc": 0.99, "op_params": {"shape": "(8, 207, 207)"}},
    ),
    (
        Reshape494,
        [((1, 8, 256, 207), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_2_2b_it_qa_hf"], "pcc": 0.99, "op_params": {"shape": "(8, 256, 207)"}},
    ),
    (
        Reshape491,
        [((8, 207, 256), torch.float32)],
        {
            "model_name": ["pt_gemma_google_gemma_2_2b_it_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 8, 207, 256)"},
        },
    ),
    (
        Reshape495,
        [((1, 207, 8, 256), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_2_2b_it_qa_hf"], "pcc": 0.99, "op_params": {"shape": "(207, 2048)"}},
    ),
    (
        Reshape496,
        [((207, 2304), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_2_2b_it_qa_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 207, 2304)"}},
    ),
    (
        Reshape497,
        [((207, 9216), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_2_2b_it_qa_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 207, 9216)"}},
    ),
    (
        Reshape498,
        [((1, 107, 2048), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_1_1_2b_it_qa_hf"], "pcc": 0.99, "op_params": {"shape": "(107, 2048)"}},
    ),
    (
        Reshape499,
        [((107, 2048), torch.float32)],
        {
            "model_name": ["pt_gemma_google_gemma_1_1_2b_it_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 107, 8, 256)"},
        },
    ),
    (
        Reshape500,
        [((107, 2048), torch.float32)],
        {
            "model_name": ["pt_gemma_google_gemma_1_1_2b_it_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 107, 2048)"},
        },
    ),
    (
        Reshape501,
        [((1, 8, 107, 256), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_1_1_2b_it_qa_hf"], "pcc": 0.99, "op_params": {"shape": "(8, 107, 256)"}},
    ),
    (
        Reshape502,
        [((107, 256), torch.float32)],
        {
            "model_name": ["pt_gemma_google_gemma_1_1_2b_it_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 107, 1, 256)"},
        },
    ),
    (
        Reshape501,
        [((1, 1, 8, 107, 256), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_1_1_2b_it_qa_hf"], "pcc": 0.99, "op_params": {"shape": "(8, 107, 256)"}},
    ),
    (
        Reshape503,
        [((1, 1, 8, 107, 256), torch.float32)],
        {
            "model_name": ["pt_gemma_google_gemma_1_1_2b_it_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 8, 107, 256)"},
        },
    ),
    (
        Reshape504,
        [((8, 107, 107), torch.float32)],
        {
            "model_name": ["pt_gemma_google_gemma_1_1_2b_it_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 8, 107, 107)"},
        },
    ),
    (
        Reshape505,
        [((1, 8, 107, 107), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_1_1_2b_it_qa_hf"], "pcc": 0.99, "op_params": {"shape": "(8, 107, 107)"}},
    ),
    (
        Reshape506,
        [((1, 8, 256, 107), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_1_1_2b_it_qa_hf"], "pcc": 0.99, "op_params": {"shape": "(8, 256, 107)"}},
    ),
    (
        Reshape503,
        [((8, 107, 256), torch.float32)],
        {
            "model_name": ["pt_gemma_google_gemma_1_1_2b_it_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 8, 107, 256)"},
        },
    ),
    (
        Reshape498,
        [((1, 107, 8, 256), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_1_1_2b_it_qa_hf"], "pcc": 0.99, "op_params": {"shape": "(107, 2048)"}},
    ),
    (
        Reshape507,
        [((107, 16384), torch.float32)],
        {
            "model_name": ["pt_gemma_google_gemma_1_1_2b_it_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 107, 16384)"},
        },
    ),
    (
        Reshape508,
        [((1, 107, 3072), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_1_1_7b_it_qa_hf"], "pcc": 0.99, "op_params": {"shape": "(107, 3072)"}},
    ),
    (
        Reshape509,
        [((107, 4096), torch.float32)],
        {
            "model_name": ["pt_gemma_google_gemma_1_1_7b_it_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 107, 16, 256)"},
        },
    ),
    (
        Reshape510,
        [((1, 16, 107, 256), torch.float32)],
        {
            "model_name": ["pt_gemma_google_gemma_1_1_7b_it_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(16, 107, 256)"},
        },
    ),
    (
        Reshape511,
        [((16, 107, 107), torch.float32)],
        {
            "model_name": ["pt_gemma_google_gemma_1_1_7b_it_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 16, 107, 107)"},
        },
    ),
    (
        Reshape512,
        [((1, 16, 107, 107), torch.float32)],
        {
            "model_name": ["pt_gemma_google_gemma_1_1_7b_it_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(16, 107, 107)"},
        },
    ),
    (
        Reshape513,
        [((1, 16, 256, 107), torch.float32)],
        {
            "model_name": ["pt_gemma_google_gemma_1_1_7b_it_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(16, 256, 107)"},
        },
    ),
    (
        Reshape514,
        [((16, 107, 256), torch.float32)],
        {
            "model_name": ["pt_gemma_google_gemma_1_1_7b_it_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 16, 107, 256)"},
        },
    ),
    (
        Reshape515,
        [((1, 107, 16, 256), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_1_1_7b_it_qa_hf"], "pcc": 0.99, "op_params": {"shape": "(107, 4096)"}},
    ),
    (
        Reshape516,
        [((107, 3072), torch.float32)],
        {
            "model_name": ["pt_gemma_google_gemma_1_1_7b_it_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 107, 3072)"},
        },
    ),
    (
        Reshape517,
        [((107, 24576), torch.float32)],
        {
            "model_name": ["pt_gemma_google_gemma_1_1_7b_it_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 107, 24576)"},
        },
    ),
    (
        Reshape518,
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
        Reshape519,
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
        Reshape520,
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
        Reshape521,
        [((1, 7, 768), torch.float32)],
        {
            "model_name": ["pt_nanogpt_financialsupport_nanogpt_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 7, 768)"},
        },
    ),
    (
        Reshape521,
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
        Reshape522,
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
        Reshape523,
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
        Reshape524,
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
        Reshape525,
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
        Reshape526,
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
        Reshape519,
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
        Reshape527,
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
        Reshape528,
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
        Reshape529,
        [((7, 2), torch.float32)],
        {
            "model_name": ["pt_gpt2_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(7, 2)"},
        },
    ),
    (
        Reshape277,
        [((1, 2), torch.float32)],
        {
            "model_name": [
                "pt_gpt2_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
                "pt_llama3_huggyllama_llama_7b_seq_cls_hf",
                "pt_opt_facebook_opt_350m_seq_cls_hf",
                "pt_opt_facebook_opt_125m_seq_cls_hf",
                "pt_opt_facebook_opt_1_3b_seq_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 2)"},
        },
    ),
    (
        Reshape530,
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
        Reshape531,
        [((1, 256, 768), torch.float32)],
        {
            "model_name": ["pt_gpt2_gpt2_text_gen_hf", "pt_opt_facebook_opt_125m_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 256, 12, 64)"},
        },
    ),
    (
        Reshape532,
        [((1, 256, 768), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 256, 8, 96)"},
        },
    ),
    (
        Reshape533,
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
        Reshape531,
        [((256, 768), torch.float32)],
        {
            "model_name": ["pt_gptneo_eleutherai_gpt_neo_125m_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 256, 12, 64)"},
        },
    ),
    (
        Reshape534,
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
        Reshape535,
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
        Reshape403,
        [((1, 256), torch.float32)],
        {"model_name": ["pt_gpt2_gpt2_text_gen_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 256)"}},
    ),
    (
        Reshape536,
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
        Reshape537,
        [((1, 12, 64, 256), torch.float32)],
        {
            "model_name": ["pt_gpt2_gpt2_text_gen_hf", "pt_gptneo_eleutherai_gpt_neo_125m_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(12, 64, 256)"},
        },
    ),
    (
        Reshape538,
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
        Reshape530,
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
        Reshape539,
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
        Reshape540,
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
        Reshape541,
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
        Reshape542,
        [((1, 32, 2048), torch.float32)],
        {
            "model_name": [
                "pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf",
                "pt_opt_facebook_opt_1_3b_qa_hf",
                "pt_opt_facebook_opt_1_3b_seq_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(32, 2048)"},
        },
    ),
    (
        Reshape543,
        [((1, 32, 2048), torch.float32)],
        {
            "model_name": ["pt_opt_facebook_opt_1_3b_qa_hf", "pt_opt_facebook_opt_1_3b_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 32, 32, 64)"},
        },
    ),
    (
        Reshape544,
        [((32, 2048), torch.float32)],
        {
            "model_name": ["pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 32, 16, 128)"},
        },
    ),
    (
        Reshape545,
        [((32, 2048), torch.float32)],
        {
            "model_name": [
                "pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf",
                "pt_opt_facebook_opt_1_3b_qa_hf",
                "pt_opt_facebook_opt_1_3b_seq_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 32, 2048)"},
        },
    ),
    (
        Reshape546,
        [((1, 16, 32, 128), torch.float32)],
        {
            "model_name": ["pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(16, 32, 128)"},
        },
    ),
    (
        Reshape547,
        [((1, 16, 128, 32), torch.float32)],
        {
            "model_name": ["pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(16, 128, 32)"},
        },
    ),
    (
        Reshape548,
        [((16, 32, 128), torch.float32)],
        {
            "model_name": ["pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 16, 32, 128)"},
        },
    ),
    (
        Reshape542,
        [((1, 32, 16, 128), torch.float32)],
        {
            "model_name": ["pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(32, 2048)"},
        },
    ),
    (
        Reshape277,
        [((1, 1, 2), torch.float32)],
        {
            "model_name": [
                "pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf",
                "pt_gptneo_eleutherai_gpt_neo_125m_seq_cls_hf",
                "pt_gptneo_eleutherai_gpt_neo_2_7b_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_3b_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf",
                "pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf",
                "pt_phi2_microsoft_phi_2_seq_cls_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 2)"},
        },
    ),
    (
        Reshape549,
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
        Reshape550,
        [((1, 256, 2560), torch.float32)],
        {
            "model_name": ["pt_phi2_microsoft_phi_2_pytdml_clm_hf", "pt_phi2_microsoft_phi_2_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 256, 32, 80)"},
        },
    ),
    (
        Reshape551,
        [((256, 2560), torch.float32)],
        {
            "model_name": ["pt_gptneo_eleutherai_gpt_neo_2_7b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 256, 20, 128)"},
        },
    ),
    (
        Reshape552,
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
        Reshape553,
        [((1, 20, 256, 128), torch.float32)],
        {
            "model_name": ["pt_gptneo_eleutherai_gpt_neo_2_7b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(20, 256, 128)"},
        },
    ),
    (
        Reshape554,
        [((20, 256, 256), torch.float32)],
        {
            "model_name": ["pt_gptneo_eleutherai_gpt_neo_2_7b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 20, 256, 256)"},
        },
    ),
    (
        Reshape555,
        [((1, 20, 256, 256), torch.float32)],
        {
            "model_name": ["pt_gptneo_eleutherai_gpt_neo_2_7b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(20, 256, 256)"},
        },
    ),
    (
        Reshape556,
        [((1, 20, 128, 256), torch.float32)],
        {
            "model_name": ["pt_gptneo_eleutherai_gpt_neo_2_7b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(20, 128, 256)"},
        },
    ),
    (
        Reshape557,
        [((20, 256, 128), torch.float32)],
        {
            "model_name": ["pt_gptneo_eleutherai_gpt_neo_2_7b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 20, 256, 128)"},
        },
    ),
    (
        Reshape549,
        [((1, 256, 20, 128), torch.float32)],
        {
            "model_name": ["pt_gptneo_eleutherai_gpt_neo_2_7b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(256, 2560)"},
        },
    ),
    (
        Reshape558,
        [((1, 32, 768), torch.float32)],
        {
            "model_name": [
                "pt_gptneo_eleutherai_gpt_neo_125m_seq_cls_hf",
                "pt_opt_facebook_opt_125m_qa_hf",
                "pt_opt_facebook_opt_125m_seq_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(32, 768)"},
        },
    ),
    (
        Reshape559,
        [((1, 32, 768), torch.float32)],
        {
            "model_name": ["pt_opt_facebook_opt_125m_qa_hf", "pt_opt_facebook_opt_125m_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 32, 12, 64)"},
        },
    ),
    (
        Reshape559,
        [((32, 768), torch.float32)],
        {
            "model_name": ["pt_gptneo_eleutherai_gpt_neo_125m_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 32, 12, 64)"},
        },
    ),
    (
        Reshape560,
        [((32, 768), torch.float32)],
        {
            "model_name": [
                "pt_gptneo_eleutherai_gpt_neo_125m_seq_cls_hf",
                "pt_opt_facebook_opt_125m_qa_hf",
                "pt_opt_facebook_opt_125m_seq_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 32, 768)"},
        },
    ),
    (
        Reshape561,
        [((1, 12, 32, 64), torch.float32)],
        {
            "model_name": [
                "pt_gptneo_eleutherai_gpt_neo_125m_seq_cls_hf",
                "pt_opt_facebook_opt_125m_qa_hf",
                "pt_opt_facebook_opt_125m_seq_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(12, 32, 64)"},
        },
    ),
    (
        Reshape562,
        [((12, 32, 32), torch.float32)],
        {
            "model_name": [
                "pt_gptneo_eleutherai_gpt_neo_125m_seq_cls_hf",
                "pt_opt_facebook_opt_125m_qa_hf",
                "pt_opt_facebook_opt_125m_seq_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 12, 32, 32)"},
        },
    ),
    (
        Reshape563,
        [((1, 12, 32, 32), torch.float32)],
        {
            "model_name": [
                "pt_gptneo_eleutherai_gpt_neo_125m_seq_cls_hf",
                "pt_opt_facebook_opt_125m_qa_hf",
                "pt_opt_facebook_opt_125m_seq_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(12, 32, 32)"},
        },
    ),
    (
        Reshape564,
        [((1, 12, 64, 32), torch.float32)],
        {
            "model_name": ["pt_gptneo_eleutherai_gpt_neo_125m_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(12, 64, 32)"},
        },
    ),
    (
        Reshape565,
        [((12, 32, 64), torch.float32)],
        {
            "model_name": [
                "pt_gptneo_eleutherai_gpt_neo_125m_seq_cls_hf",
                "pt_opt_facebook_opt_125m_qa_hf",
                "pt_opt_facebook_opt_125m_seq_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 12, 32, 64)"},
        },
    ),
    (
        Reshape558,
        [((1, 32, 12, 64), torch.float32)],
        {
            "model_name": [
                "pt_gptneo_eleutherai_gpt_neo_125m_seq_cls_hf",
                "pt_opt_facebook_opt_125m_qa_hf",
                "pt_opt_facebook_opt_125m_seq_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(32, 768)"},
        },
    ),
    (
        Reshape566,
        [((1, 32, 2560), torch.float32)],
        {
            "model_name": ["pt_gptneo_eleutherai_gpt_neo_2_7b_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(32, 2560)"},
        },
    ),
    (
        Reshape567,
        [((32, 2560), torch.float32)],
        {
            "model_name": ["pt_gptneo_eleutherai_gpt_neo_2_7b_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 32, 20, 128)"},
        },
    ),
    (
        Reshape568,
        [((32, 2560), torch.float32)],
        {
            "model_name": ["pt_gptneo_eleutherai_gpt_neo_2_7b_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 32, 2560)"},
        },
    ),
    (
        Reshape569,
        [((1, 20, 32, 128), torch.float32)],
        {
            "model_name": ["pt_gptneo_eleutherai_gpt_neo_2_7b_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(20, 32, 128)"},
        },
    ),
    (
        Reshape570,
        [((20, 32, 32), torch.float32)],
        {
            "model_name": ["pt_gptneo_eleutherai_gpt_neo_2_7b_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 20, 32, 32)"},
        },
    ),
    (
        Reshape571,
        [((1, 20, 32, 32), torch.float32)],
        {
            "model_name": ["pt_gptneo_eleutherai_gpt_neo_2_7b_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(20, 32, 32)"},
        },
    ),
    (
        Reshape572,
        [((1, 20, 128, 32), torch.float32)],
        {
            "model_name": ["pt_gptneo_eleutherai_gpt_neo_2_7b_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(20, 128, 32)"},
        },
    ),
    (
        Reshape573,
        [((20, 32, 128), torch.float32)],
        {
            "model_name": ["pt_gptneo_eleutherai_gpt_neo_2_7b_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 20, 32, 128)"},
        },
    ),
    (
        Reshape566,
        [((1, 32, 20, 128), torch.float32)],
        {
            "model_name": ["pt_gptneo_eleutherai_gpt_neo_2_7b_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(32, 2560)"},
        },
    ),
    (
        Reshape574,
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
        Reshape575,
        [((1, 256, 2048), torch.float32)],
        {"model_name": ["pt_opt_facebook_opt_1_3b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 256, 32, 64)"}},
    ),
    (
        Reshape576,
        [((1, 256, 2048), torch.float32)],
        {"model_name": ["pt_xglm_facebook_xglm_1_7b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 256, 16, 128)"}},
    ),
    (
        Reshape576,
        [((256, 2048), torch.float32)],
        {
            "model_name": ["pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 256, 16, 128)"},
        },
    ),
    (
        Reshape577,
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
        Reshape575,
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
        Reshape578,
        [((1, 16, 256, 128), torch.float32)],
        {
            "model_name": ["pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf", "pt_xglm_facebook_xglm_1_7b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(16, 256, 128)"},
        },
    ),
    (
        Reshape579,
        [((1, 16, 128, 256), torch.float32)],
        {
            "model_name": ["pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(16, 128, 256)"},
        },
    ),
    (
        Reshape580,
        [((16, 256, 128), torch.float32)],
        {
            "model_name": ["pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf", "pt_xglm_facebook_xglm_1_7b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 16, 256, 128)"},
        },
    ),
    (
        Reshape574,
        [((1, 256, 16, 128), torch.float32)],
        {
            "model_name": ["pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf", "pt_xglm_facebook_xglm_1_7b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(256, 2048)"},
        },
    ),
    (
        Reshape581,
        [((1, 256, 4096), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_1_8b_clm_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_clm_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_clm_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(256, 4096)"},
        },
    ),
    (
        Reshape582,
        [((1, 256, 4096), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 256, 64, 64)"},
        },
    ),
    (
        Reshape583,
        [((1, 32, 256, 128), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_1_8b_clm_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_clm_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_clm_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(32, 256, 128)"},
        },
    ),
    (
        Reshape583,
        [((1, 8, 4, 256, 128), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_1_8b_clm_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_clm_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_clm_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(32, 256, 128)"},
        },
    ),
    (
        Reshape584,
        [((1, 8, 4, 256, 128), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_1_8b_clm_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_clm_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_clm_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 32, 256, 128)"},
        },
    ),
    (
        Reshape585,
        [((32, 256, 256), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_1_8b_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_clm_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_clm_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_clm_hf",
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
        Reshape586,
        [((1, 32, 256, 256), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_1_8b_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_clm_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_clm_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_clm_hf",
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
        Reshape587,
        [((1, 32, 128, 256), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_1_8b_clm_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_clm_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_clm_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(32, 128, 256)"},
        },
    ),
    (
        Reshape584,
        [((32, 256, 128), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_1_8b_clm_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_clm_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_clm_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 32, 256, 128)"},
        },
    ),
    (
        Reshape581,
        [((1, 256, 32, 128), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_1_8b_clm_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_clm_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_clm_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(256, 4096)"},
        },
    ),
    (
        Reshape588,
        [((256, 14336), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_1_8b_clm_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_clm_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_clm_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 256, 14336)"},
        },
    ),
    (
        Reshape589,
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
        Reshape590,
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
        Reshape591,
        [((256, 512), torch.float32)],
        {"model_name": ["pt_opt_facebook_opt_350m_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(256, 512)"}},
    ),
    (
        Reshape592,
        [((256, 512), torch.float32)],
        {
            "model_name": [
                "pt_mlp_mixer_base_img_cls_github",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 256, 512)"},
        },
    ),
    (
        Reshape589,
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
        Reshape593,
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
        Reshape594,
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
        Reshape593,
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
        Reshape574,
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
        Reshape595,
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
        Reshape596,
        [((1, 4, 4096), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf",
                "pt_llama3_huggyllama_llama_7b_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(4, 4096)"},
        },
    ),
    (
        Reshape597,
        [((4, 4096), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf",
                "pt_llama3_huggyllama_llama_7b_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 4, 32, 128)"},
        },
    ),
    (
        Reshape598,
        [((4, 4096), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf",
                "pt_llama3_huggyllama_llama_7b_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 4, 4096)"},
        },
    ),
    (
        Reshape599,
        [((1, 32, 4, 128), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf",
                "pt_llama3_huggyllama_llama_7b_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(32, 4, 128)"},
        },
    ),
    (
        Reshape600,
        [((4, 1024), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_3b_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 4, 8, 128)"},
        },
    ),
    (
        Reshape599,
        [((1, 8, 4, 4, 128), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(32, 4, 128)"},
        },
    ),
    (
        Reshape601,
        [((1, 8, 4, 4, 128), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 32, 4, 128)"},
        },
    ),
    (
        Reshape602,
        [((32, 4, 4), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf",
                "pt_llama3_huggyllama_llama_7b_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 32, 4, 4)"},
        },
    ),
    (
        Reshape603,
        [((1, 32, 4, 4), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf",
                "pt_llama3_huggyllama_llama_7b_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(32, 4, 4)"},
        },
    ),
    (
        Reshape604,
        [((1, 32, 128, 4), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf",
                "pt_llama3_huggyllama_llama_7b_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(32, 128, 4)"},
        },
    ),
    (
        Reshape601,
        [((32, 4, 128), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf",
                "pt_llama3_huggyllama_llama_7b_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 32, 4, 128)"},
        },
    ),
    (
        Reshape596,
        [((1, 4, 32, 128), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf",
                "pt_llama3_huggyllama_llama_7b_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(4, 4096)"},
        },
    ),
    (
        Reshape605,
        [((4, 14336), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 4, 14336)"},
        },
    ),
    (
        Reshape606,
        [((4, 11008), torch.float32)],
        {
            "model_name": ["pt_llama3_huggyllama_llama_7b_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 4, 11008)"},
        },
    ),
    (
        Reshape607,
        [((4, 2), torch.float32)],
        {"model_name": ["pt_llama3_huggyllama_llama_7b_seq_cls_hf"], "pcc": 0.99, "op_params": {"shape": "(4, 2)"}},
    ),
    (
        Reshape608,
        [((1, 4, 3072), torch.float32)],
        {
            "model_name": ["pt_llama3_meta_llama_llama_3_2_3b_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(4, 3072)"},
        },
    ),
    (
        Reshape609,
        [((4, 3072), torch.float32)],
        {
            "model_name": ["pt_llama3_meta_llama_llama_3_2_3b_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 4, 24, 128)"},
        },
    ),
    (
        Reshape610,
        [((4, 3072), torch.float32)],
        {
            "model_name": ["pt_llama3_meta_llama_llama_3_2_3b_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 4, 3072)"},
        },
    ),
    (
        Reshape611,
        [((1, 24, 4, 128), torch.float32)],
        {
            "model_name": ["pt_llama3_meta_llama_llama_3_2_3b_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(24, 4, 128)"},
        },
    ),
    (
        Reshape611,
        [((1, 8, 3, 4, 128), torch.float32)],
        {
            "model_name": ["pt_llama3_meta_llama_llama_3_2_3b_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(24, 4, 128)"},
        },
    ),
    (
        Reshape612,
        [((1, 8, 3, 4, 128), torch.float32)],
        {
            "model_name": ["pt_llama3_meta_llama_llama_3_2_3b_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 24, 4, 128)"},
        },
    ),
    (
        Reshape613,
        [((24, 4, 4), torch.float32)],
        {
            "model_name": ["pt_llama3_meta_llama_llama_3_2_3b_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 24, 4, 4)"},
        },
    ),
    (
        Reshape614,
        [((1, 24, 4, 4), torch.float32)],
        {
            "model_name": ["pt_llama3_meta_llama_llama_3_2_3b_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(24, 4, 4)"},
        },
    ),
    (
        Reshape615,
        [((1, 24, 128, 4), torch.float32)],
        {
            "model_name": ["pt_llama3_meta_llama_llama_3_2_3b_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(24, 128, 4)"},
        },
    ),
    (
        Reshape612,
        [((24, 4, 128), torch.float32)],
        {
            "model_name": ["pt_llama3_meta_llama_llama_3_2_3b_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 24, 4, 128)"},
        },
    ),
    (
        Reshape608,
        [((1, 4, 24, 128), torch.float32)],
        {
            "model_name": ["pt_llama3_meta_llama_llama_3_2_3b_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(4, 3072)"},
        },
    ),
    (
        Reshape616,
        [((4, 8192), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_2_3b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 4, 8192)"},
        },
    ),
    (
        Reshape617,
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
        Reshape618,
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
        Reshape619,
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
        Reshape620,
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
        Reshape621,
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
        Reshape620,
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
        Reshape622,
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
        Reshape623,
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
        Reshape622,
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
        Reshape617,
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
        Reshape624,
        [((1, 32, 3072), torch.float32)],
        {"model_name": ["pt_llama3_meta_llama_llama_3_2_3b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(32, 3072)"}},
    ),
    (
        Reshape625,
        [((32, 3072), torch.float32)],
        {
            "model_name": ["pt_llama3_meta_llama_llama_3_2_3b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 32, 24, 128)"},
        },
    ),
    (
        Reshape626,
        [((32, 3072), torch.float32)],
        {
            "model_name": ["pt_llama3_meta_llama_llama_3_2_3b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 32, 3072)"},
        },
    ),
    (
        Reshape627,
        [((1, 24, 32, 128), torch.float32)],
        {
            "model_name": ["pt_llama3_meta_llama_llama_3_2_3b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(24, 32, 128)"},
        },
    ),
    (
        Reshape628,
        [((32, 1024), torch.float32)],
        {
            "model_name": ["pt_llama3_meta_llama_llama_3_2_3b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 32, 8, 128)"},
        },
    ),
    (
        Reshape629,
        [((32, 1024), torch.float32)],
        {
            "model_name": ["pt_opt_facebook_opt_350m_seq_cls_hf", "pt_opt_facebook_opt_350m_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 32, 1024)"},
        },
    ),
    (
        Reshape627,
        [((1, 8, 3, 32, 128), torch.float32)],
        {
            "model_name": ["pt_llama3_meta_llama_llama_3_2_3b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(24, 32, 128)"},
        },
    ),
    (
        Reshape630,
        [((1, 8, 3, 32, 128), torch.float32)],
        {
            "model_name": ["pt_llama3_meta_llama_llama_3_2_3b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 24, 32, 128)"},
        },
    ),
    (
        Reshape631,
        [((24, 32, 32), torch.float32)],
        {
            "model_name": ["pt_llama3_meta_llama_llama_3_2_3b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 24, 32, 32)"},
        },
    ),
    (
        Reshape632,
        [((1, 24, 32, 32), torch.float32)],
        {
            "model_name": ["pt_llama3_meta_llama_llama_3_2_3b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(24, 32, 32)"},
        },
    ),
    (
        Reshape633,
        [((1, 24, 128, 32), torch.float32)],
        {
            "model_name": ["pt_llama3_meta_llama_llama_3_2_3b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(24, 128, 32)"},
        },
    ),
    (
        Reshape630,
        [((24, 32, 128), torch.float32)],
        {
            "model_name": ["pt_llama3_meta_llama_llama_3_2_3b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 24, 32, 128)"},
        },
    ),
    (
        Reshape624,
        [((1, 32, 24, 128), torch.float32)],
        {"model_name": ["pt_llama3_meta_llama_llama_3_2_3b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(32, 3072)"}},
    ),
    (
        Reshape634,
        [((32, 8192), torch.float32)],
        {
            "model_name": ["pt_llama3_meta_llama_llama_3_2_3b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 32, 8192)"},
        },
    ),
    (
        Reshape635,
        [((1, 32, 4096), torch.float32)],
        {"model_name": ["pt_llama3_huggyllama_llama_7b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(32, 4096)"}},
    ),
    (
        Reshape636,
        [((32, 4096), torch.float32)],
        {
            "model_name": ["pt_llama3_huggyllama_llama_7b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 32, 32, 128)"},
        },
    ),
    (
        Reshape637,
        [((32, 4096), torch.float32)],
        {"model_name": ["pt_llama3_huggyllama_llama_7b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 32, 4096)"}},
    ),
    (
        Reshape638,
        [((1, 32, 32, 128), torch.float32)],
        {"model_name": ["pt_llama3_huggyllama_llama_7b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(32, 32, 128)"}},
    ),
    (
        Reshape635,
        [((1, 32, 32, 128), torch.float32)],
        {"model_name": ["pt_llama3_huggyllama_llama_7b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(32, 4096)"}},
    ),
    (
        Reshape639,
        [((32, 32, 32), torch.float32)],
        {
            "model_name": [
                "pt_llama3_huggyllama_llama_7b_clm_hf",
                "pt_opt_facebook_opt_1_3b_qa_hf",
                "pt_opt_facebook_opt_1_3b_seq_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 32, 32, 32)"},
        },
    ),
    (
        Reshape640,
        [((1, 32, 32, 32), torch.float32)],
        {
            "model_name": [
                "pt_llama3_huggyllama_llama_7b_clm_hf",
                "pt_opt_facebook_opt_1_3b_qa_hf",
                "pt_opt_facebook_opt_1_3b_seq_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(32, 32, 32)"},
        },
    ),
    (
        Reshape641,
        [((1, 32, 128, 32), torch.float32)],
        {"model_name": ["pt_llama3_huggyllama_llama_7b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(32, 128, 32)"}},
    ),
    (
        Reshape636,
        [((32, 32, 128), torch.float32)],
        {
            "model_name": ["pt_llama3_huggyllama_llama_7b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 32, 32, 128)"},
        },
    ),
    (
        Reshape642,
        [((32, 11008), torch.float32)],
        {"model_name": ["pt_llama3_huggyllama_llama_7b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 32, 11008)"}},
    ),
    (
        Reshape643,
        [((1, 32, 128, 128), torch.float32)],
        {
            "model_name": ["pt_mistral_mistralai_mistral_7b_v0_1_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(32, 128, 128)"},
        },
    ),
    (
        Reshape644,
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
        Reshape643,
        [((1, 8, 4, 128, 128), torch.float32)],
        {
            "model_name": ["pt_mistral_mistralai_mistral_7b_v0_1_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(32, 128, 128)"},
        },
    ),
    (
        Reshape645,
        [((1, 8, 4, 128, 128), torch.float32)],
        {
            "model_name": ["pt_mistral_mistralai_mistral_7b_v0_1_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 32, 128, 128)"},
        },
    ),
    (
        Reshape645,
        [((32, 128, 128), torch.float32)],
        {
            "model_name": ["pt_mistral_mistralai_mistral_7b_v0_1_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 32, 128, 128)"},
        },
    ),
    (
        Reshape382,
        [((1, 128, 32, 128), torch.float32)],
        {
            "model_name": ["pt_mistral_mistralai_mistral_7b_v0_1_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(128, 4096)"},
        },
    ),
    (
        Reshape646,
        [((128, 14336), torch.float32)],
        {
            "model_name": ["pt_mistral_mistralai_mistral_7b_v0_1_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 128, 14336)"},
        },
    ),
    (
        Reshape647,
        [((1, 32), torch.int64)],
        {
            "model_name": [
                "pt_opt_facebook_opt_1_3b_qa_hf",
                "pt_opt_facebook_opt_350m_seq_cls_hf",
                "pt_opt_facebook_opt_350m_qa_hf",
                "pt_opt_facebook_opt_125m_qa_hf",
                "pt_opt_facebook_opt_125m_seq_cls_hf",
                "pt_opt_facebook_opt_1_3b_seq_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 32)"},
        },
    ),
    (
        Reshape648,
        [((1, 32, 32, 64), torch.float32)],
        {
            "model_name": ["pt_opt_facebook_opt_1_3b_qa_hf", "pt_opt_facebook_opt_1_3b_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(32, 32, 64)"},
        },
    ),
    (
        Reshape542,
        [((1, 32, 32, 64), torch.float32)],
        {
            "model_name": ["pt_opt_facebook_opt_1_3b_qa_hf", "pt_opt_facebook_opt_1_3b_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(32, 2048)"},
        },
    ),
    (
        Reshape543,
        [((32, 32, 64), torch.float32)],
        {
            "model_name": ["pt_opt_facebook_opt_1_3b_qa_hf", "pt_opt_facebook_opt_1_3b_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 32, 32, 64)"},
        },
    ),
    (
        Reshape649,
        [((32, 1), torch.float32)],
        {
            "model_name": [
                "pt_opt_facebook_opt_1_3b_qa_hf",
                "pt_opt_facebook_opt_350m_qa_hf",
                "pt_opt_facebook_opt_125m_qa_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 32, 1)"},
        },
    ),
    (
        Reshape650,
        [((1, 32, 1024), torch.float32)],
        {
            "model_name": ["pt_opt_facebook_opt_350m_seq_cls_hf", "pt_opt_facebook_opt_350m_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(32, 1024)"},
        },
    ),
    (
        Reshape651,
        [((1, 32, 1024), torch.float32)],
        {
            "model_name": ["pt_opt_facebook_opt_350m_seq_cls_hf", "pt_opt_facebook_opt_350m_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 32, 16, 64)"},
        },
    ),
    (
        Reshape652,
        [((1, 16, 32, 64), torch.float32)],
        {
            "model_name": ["pt_opt_facebook_opt_350m_seq_cls_hf", "pt_opt_facebook_opt_350m_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(16, 32, 64)"},
        },
    ),
    (
        Reshape653,
        [((16, 32, 64), torch.float32)],
        {
            "model_name": ["pt_opt_facebook_opt_350m_seq_cls_hf", "pt_opt_facebook_opt_350m_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 16, 32, 64)"},
        },
    ),
    (
        Reshape650,
        [((1, 32, 16, 64), torch.float32)],
        {
            "model_name": ["pt_opt_facebook_opt_350m_seq_cls_hf", "pt_opt_facebook_opt_350m_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(32, 1024)"},
        },
    ),
    (
        Reshape654,
        [((32, 512), torch.float32)],
        {
            "model_name": ["pt_opt_facebook_opt_350m_seq_cls_hf", "pt_opt_facebook_opt_350m_qa_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(32, 512)"},
        },
    ),
    (
        Reshape655,
        [((32, 2), torch.float32)],
        {
            "model_name": [
                "pt_opt_facebook_opt_350m_seq_cls_hf",
                "pt_opt_facebook_opt_125m_seq_cls_hf",
                "pt_opt_facebook_opt_1_3b_seq_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(32, 2)"},
        },
    ),
    (
        Reshape656,
        [((256, 50272), torch.float32)],
        {"model_name": ["pt_opt_facebook_opt_350m_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 256, 50272)"}},
    ),
    (
        Reshape657,
        [((1, 512, 512), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 512, 1, 512)"},
        },
    ),
    (
        Reshape658,
        [((1, 224, 224, 256), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 50176, 256)"},
        },
    ),
    (
        Reshape659,
        [((1, 50176, 512), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(50176, 512)"},
        },
    ),
    (
        Reshape660,
        [((1, 50176, 512), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 50176, 1, 512)"},
        },
    ),
    (
        Reshape661,
        [((50176, 512), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 50176, 512)"},
        },
    ),
    (
        Reshape662,
        [((1, 512, 50176), torch.float32)],
        {
            "model_name": [
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1, 512, 50176)"},
        },
    ),
    (
        Reshape663,
        [((1, 1, 512, 50176), torch.float32)],
        {
            "model_name": [
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 512, 50176)"},
        },
    ),
    (
        Reshape664,
        [((1, 512, 1024), torch.float32)],
        {
            "model_name": [
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(512, 1024)"},
        },
    ),
    (
        Reshape665,
        [((1, 512, 1024), torch.float32)],
        {
            "model_name": [
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 512, 8, 128)"},
        },
    ),
    (
        Reshape666,
        [((1, 512, 1024), torch.float32)],
        {
            "model_name": [
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 512, 1, 1024)"},
        },
    ),
    (
        Reshape667,
        [((512, 1024), torch.float32)],
        {
            "model_name": [
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 512, 1024)"},
        },
    ),
    (
        Reshape668,
        [((1, 8, 512, 128), torch.float32)],
        {
            "model_name": [
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(8, 512, 128)"},
        },
    ),
    (
        Reshape669,
        [((8, 512, 512), torch.float32)],
        {
            "model_name": [
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 8, 512, 512)"},
        },
    ),
    (
        Reshape670,
        [((1, 8, 512, 512), torch.float32)],
        {
            "model_name": [
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(8, 512, 512)"},
        },
    ),
    (
        Reshape671,
        [((1, 8, 128, 512), torch.float32)],
        {
            "model_name": [
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(8, 128, 512)"},
        },
    ),
    (
        Reshape672,
        [((8, 512, 128), torch.float32)],
        {
            "model_name": [
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 8, 512, 128)"},
        },
    ),
    (
        Reshape664,
        [((1, 512, 8, 128), torch.float32)],
        {
            "model_name": [
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(512, 1024)"},
        },
    ),
    (
        Reshape256,
        [((1, 1, 1, 512), torch.float32)],
        {
            "model_name": [
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1, 512)"},
        },
    ),
    (
        Reshape673,
        [((1, 1, 1024, 512), torch.float32)],
        {
            "model_name": [
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1024, 512)"},
        },
    ),
    (
        Reshape674,
        [((1, 1, 1000), torch.float32)],
        {
            "model_name": [
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1000)"},
        },
    ),
    (
        Reshape675,
        [((1, 512, 322), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 512, 1, 322)"},
        },
    ),
    (
        Reshape676,
        [((1, 55, 55, 64), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 3025, 64)"},
        },
    ),
    (
        Reshape677,
        [((1, 3025, 322), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(3025, 322)"},
        },
    ),
    (
        Reshape678,
        [((1, 3025, 322), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 3025, 1, 322)"},
        },
    ),
    (
        Reshape679,
        [((3025, 322), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 3025, 322)"},
        },
    ),
    (
        Reshape680,
        [((1, 512, 3025), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1, 512, 3025)"},
        },
    ),
    (
        Reshape681,
        [((1, 1, 512, 3025), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 512, 3025)"},
        },
    ),
    (
        Reshape682,
        [((1, 1, 322, 3025), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 322, 3025)"},
        },
    ),
    (
        Reshape683,
        [((1, 512, 261), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 512, 1, 261)"},
        },
    ),
    (
        Reshape684,
        [((1, 224, 224, 3), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 50176, 3)"},
        },
    ),
    (
        Reshape685,
        [((1, 50176, 261), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(50176, 261)"},
        },
    ),
    (
        Reshape686,
        [((1, 50176, 261), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 50176, 1, 261)"},
        },
    ),
    (
        Reshape687,
        [((50176, 261), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 50176, 261)"},
        },
    ),
    (
        Reshape688,
        [((1, 1, 261, 50176), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 261, 50176)"},
        },
    ),
    (
        Reshape689,
        [((1, 2048, 256), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 2048, 8, 32)"},
        },
    ),
    (
        Reshape690,
        [((1, 2048, 256), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 2048, 16, 16)"},
        },
    ),
    (
        Reshape691,
        [((1, 8, 2048, 32), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(8, 2048, 32)"},
        },
    ),
    (
        Reshape692,
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
        Reshape693,
        [((1, 256, 256), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 256, 16, 16)"},
        },
    ),
    (
        Reshape694,
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
        Reshape695,
        [((1, 256, 256), torch.float32)],
        {"model_name": ["pt_segformer_nvidia_mit_b0_img_cls_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 256, 256)"}},
    ),
    (
        Reshape696,
        [((1, 256, 256), torch.float32)],
        {
            "model_name": ["pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 16, 16, 256)"},
        },
    ),
    (
        Reshape697,
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
        Reshape698,
        [((1, 2048, 768), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(2048, 768)"},
        },
    ),
    (
        Reshape699,
        [((2048, 256), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 2048, 256)"},
        },
    ),
    (
        Reshape700,
        [((8, 256, 2048), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 8, 256, 2048)"},
        },
    ),
    (
        Reshape701,
        [((1, 8, 256, 2048), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(8, 256, 2048)"},
        },
    ),
    (
        Reshape702,
        [((2048, 1280), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 2048, 1280)"},
        },
    ),
    (
        Reshape703,
        [((1, 2048, 1280), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 2048, 8, 160)"},
        },
    ),
    (
        Reshape704,
        [((1, 8, 160, 2048), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(8, 160, 2048)"},
        },
    ),
    (
        Reshape705,
        [((8, 256, 160), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 8, 256, 160)"},
        },
    ),
    (
        Reshape706,
        [((1, 256, 8, 160), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(256, 1280)"},
        },
    ),
    (
        Reshape707,
        [((256, 1280), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 256, 1280)"},
        },
    ),
    (
        Reshape706,
        [((1, 256, 1280), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(256, 1280)"},
        },
    ),
    (
        Reshape708,
        [((1, 256, 1280), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 256, 8, 160)"},
        },
    ),
    (
        Reshape695,
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
        Reshape709,
        [((8, 256, 256), torch.float32)],
        {
            "model_name": [
                "pt_perceiverio_deepmind_language_perceiver_mlm_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 8, 256, 256)"},
        },
    ),
    (
        Reshape710,
        [((1, 8, 256, 256), torch.float32)],
        {
            "model_name": [
                "pt_perceiverio_deepmind_language_perceiver_mlm_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(8, 256, 256)"},
        },
    ),
    (
        Reshape711,
        [((1, 8, 160, 256), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(8, 160, 256)"},
        },
    ),
    (
        Reshape712,
        [((8, 2048, 256), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 8, 2048, 256)"},
        },
    ),
    (
        Reshape713,
        [((1, 8, 2048, 256), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(8, 2048, 256)"},
        },
    ),
    (
        Reshape714,
        [((1, 8, 96, 256), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(8, 96, 256)"},
        },
    ),
    (
        Reshape715,
        [((8, 2048, 96), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 8, 2048, 96)"},
        },
    ),
    (
        Reshape698,
        [((1, 2048, 8, 96), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(2048, 768)"},
        },
    ),
    (
        Reshape716,
        [((2048, 768), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 2048, 768)"},
        },
    ),
    (
        Reshape717,
        [((2048, 262), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 2048, 262)"},
        },
    ),
    (
        Reshape718,
        [((1, 32, 256, 80), torch.float32)],
        {
            "model_name": ["pt_phi2_microsoft_phi_2_pytdml_clm_hf", "pt_phi2_microsoft_phi_2_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(32, 256, 80)"},
        },
    ),
    (
        Reshape719,
        [((1, 32, 80, 256), torch.float32)],
        {
            "model_name": ["pt_phi2_microsoft_phi_2_pytdml_clm_hf", "pt_phi2_microsoft_phi_2_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(32, 80, 256)"},
        },
    ),
    (
        Reshape720,
        [((32, 256, 80), torch.float32)],
        {
            "model_name": ["pt_phi2_microsoft_phi_2_pytdml_clm_hf", "pt_phi2_microsoft_phi_2_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 32, 256, 80)"},
        },
    ),
    (
        Reshape549,
        [((1, 256, 32, 80), torch.float32)],
        {
            "model_name": ["pt_phi2_microsoft_phi_2_pytdml_clm_hf", "pt_phi2_microsoft_phi_2_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(256, 2560)"},
        },
    ),
    (
        Reshape721,
        [((256, 10240), torch.float32)],
        {
            "model_name": ["pt_phi2_microsoft_phi_2_pytdml_clm_hf", "pt_phi2_microsoft_phi_2_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 256, 10240)"},
        },
    ),
    (
        Reshape722,
        [((1, 11, 2560), torch.float32)],
        {
            "model_name": ["pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf", "pt_phi2_microsoft_phi_2_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(11, 2560)"},
        },
    ),
    (
        Reshape723,
        [((1, 11, 2560), torch.float32)],
        {
            "model_name": ["pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf", "pt_phi2_microsoft_phi_2_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 11, 32, 80)"},
        },
    ),
    (
        Reshape724,
        [((11, 2560), torch.float32)],
        {
            "model_name": ["pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf", "pt_phi2_microsoft_phi_2_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 11, 2560)"},
        },
    ),
    (
        Reshape725,
        [((1, 32, 11, 80), torch.float32)],
        {
            "model_name": ["pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf", "pt_phi2_microsoft_phi_2_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(32, 11, 80)"},
        },
    ),
    (
        Reshape726,
        [((32, 11, 11), torch.float32)],
        {
            "model_name": ["pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf", "pt_phi2_microsoft_phi_2_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 32, 11, 11)"},
        },
    ),
    (
        Reshape727,
        [((1, 32, 11, 11), torch.float32)],
        {
            "model_name": ["pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf", "pt_phi2_microsoft_phi_2_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(32, 11, 11)"},
        },
    ),
    (
        Reshape728,
        [((1, 32, 80, 11), torch.float32)],
        {
            "model_name": ["pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf", "pt_phi2_microsoft_phi_2_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(32, 80, 11)"},
        },
    ),
    (
        Reshape729,
        [((32, 11, 80), torch.float32)],
        {
            "model_name": ["pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf", "pt_phi2_microsoft_phi_2_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 32, 11, 80)"},
        },
    ),
    (
        Reshape722,
        [((1, 11, 32, 80), torch.float32)],
        {
            "model_name": ["pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf", "pt_phi2_microsoft_phi_2_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(11, 2560)"},
        },
    ),
    (
        Reshape730,
        [((11, 10240), torch.float32)],
        {
            "model_name": ["pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf", "pt_phi2_microsoft_phi_2_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 11, 10240)"},
        },
    ),
    (
        Reshape731,
        [((1, 12, 2560), torch.float32)],
        {
            "model_name": ["pt_phi2_microsoft_phi_2_pytdml_token_cls_hf", "pt_phi2_microsoft_phi_2_token_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(12, 2560)"},
        },
    ),
    (
        Reshape732,
        [((1, 12, 2560), torch.float32)],
        {
            "model_name": ["pt_phi2_microsoft_phi_2_pytdml_token_cls_hf", "pt_phi2_microsoft_phi_2_token_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 12, 32, 80)"},
        },
    ),
    (
        Reshape733,
        [((12, 2560), torch.float32)],
        {
            "model_name": ["pt_phi2_microsoft_phi_2_pytdml_token_cls_hf", "pt_phi2_microsoft_phi_2_token_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 12, 2560)"},
        },
    ),
    (
        Reshape734,
        [((1, 32, 12, 80), torch.float32)],
        {
            "model_name": ["pt_phi2_microsoft_phi_2_pytdml_token_cls_hf", "pt_phi2_microsoft_phi_2_token_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(32, 12, 80)"},
        },
    ),
    (
        Reshape735,
        [((32, 12, 12), torch.float32)],
        {
            "model_name": ["pt_phi2_microsoft_phi_2_pytdml_token_cls_hf", "pt_phi2_microsoft_phi_2_token_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 32, 12, 12)"},
        },
    ),
    (
        Reshape736,
        [((1, 32, 12, 12), torch.float32)],
        {
            "model_name": ["pt_phi2_microsoft_phi_2_pytdml_token_cls_hf", "pt_phi2_microsoft_phi_2_token_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(32, 12, 12)"},
        },
    ),
    (
        Reshape737,
        [((1, 32, 80, 12), torch.float32)],
        {
            "model_name": ["pt_phi2_microsoft_phi_2_pytdml_token_cls_hf", "pt_phi2_microsoft_phi_2_token_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(32, 80, 12)"},
        },
    ),
    (
        Reshape738,
        [((32, 12, 80), torch.float32)],
        {
            "model_name": ["pt_phi2_microsoft_phi_2_pytdml_token_cls_hf", "pt_phi2_microsoft_phi_2_token_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 32, 12, 80)"},
        },
    ),
    (
        Reshape731,
        [((1, 12, 32, 80), torch.float32)],
        {
            "model_name": ["pt_phi2_microsoft_phi_2_pytdml_token_cls_hf", "pt_phi2_microsoft_phi_2_token_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(12, 2560)"},
        },
    ),
    (
        Reshape739,
        [((12, 10240), torch.float32)],
        {
            "model_name": ["pt_phi2_microsoft_phi_2_pytdml_token_cls_hf", "pt_phi2_microsoft_phi_2_token_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 12, 10240)"},
        },
    ),
    (
        Reshape740,
        [((1, 5, 3072), torch.float32)],
        {
            "model_name": ["pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 5, 32, 96)"},
        },
    ),
    (
        Reshape741,
        [((1, 5, 3072), torch.float32)],
        {
            "model_name": ["pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(5, 3072)"},
        },
    ),
    (
        Reshape742,
        [((1, 32, 5, 96), torch.float32)],
        {
            "model_name": ["pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(32, 5, 96)"},
        },
    ),
    (
        Reshape743,
        [((32, 5, 5), torch.float32)],
        {
            "model_name": ["pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 32, 5, 5)"},
        },
    ),
    (
        Reshape744,
        [((1, 32, 5, 5), torch.float32)],
        {
            "model_name": ["pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(32, 5, 5)"},
        },
    ),
    (
        Reshape745,
        [((1, 32, 96, 5), torch.float32)],
        {
            "model_name": ["pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(32, 96, 5)"},
        },
    ),
    (
        Reshape746,
        [((32, 5, 96), torch.float32)],
        {
            "model_name": ["pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 32, 5, 96)"},
        },
    ),
    (
        Reshape741,
        [((1, 5, 32, 96), torch.float32)],
        {
            "model_name": ["pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(5, 3072)"},
        },
    ),
    (
        Reshape747,
        [((5, 3072), torch.float32)],
        {
            "model_name": ["pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 5, 3072)"},
        },
    ),
    (
        Reshape748,
        [((5, 8192), torch.float32)],
        {
            "model_name": ["pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 5, 8192)"},
        },
    ),
    (
        Reshape749,
        [((1, 13, 3072), torch.float32)],
        {
            "model_name": ["pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 13, 32, 96)"},
        },
    ),
    (
        Reshape750,
        [((1, 13, 3072), torch.float32)],
        {
            "model_name": ["pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(13, 3072)"},
        },
    ),
    (
        Reshape751,
        [((1, 32, 13, 96), torch.float32)],
        {
            "model_name": ["pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(32, 13, 96)"},
        },
    ),
    (
        Reshape752,
        [((32, 13, 13), torch.float32)],
        {
            "model_name": ["pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 32, 13, 13)"},
        },
    ),
    (
        Reshape753,
        [((1, 32, 13, 13), torch.float32)],
        {
            "model_name": ["pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(32, 13, 13)"},
        },
    ),
    (
        Reshape754,
        [((1, 32, 96, 13), torch.float32)],
        {
            "model_name": ["pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(32, 96, 13)"},
        },
    ),
    (
        Reshape755,
        [((32, 13, 96), torch.float32)],
        {
            "model_name": ["pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 32, 13, 96)"},
        },
    ),
    (
        Reshape750,
        [((1, 13, 32, 96), torch.float32)],
        {
            "model_name": ["pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(13, 3072)"},
        },
    ),
    (
        Reshape756,
        [((13, 3072), torch.float32)],
        {
            "model_name": ["pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 13, 3072)"},
        },
    ),
    (
        Reshape757,
        [((13, 8192), torch.float32)],
        {
            "model_name": ["pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 13, 8192)"},
        },
    ),
    (
        Reshape758,
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
        Reshape759,
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
        Reshape760,
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
        Reshape540,
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
        Reshape761,
        [((1, 29, 1024), torch.float32)],
        {"model_name": ["pt_qwen1_5_qwen_qwen1_5_0_5b_chat_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(29, 1024)"}},
    ),
    (
        Reshape762,
        [((1, 29, 1024), torch.float32)],
        {
            "model_name": ["pt_qwen1_5_qwen_qwen1_5_0_5b_chat_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 29, 16, 64)"},
        },
    ),
    (
        Reshape763,
        [((29, 1024), torch.float32)],
        {
            "model_name": ["pt_qwen1_5_qwen_qwen1_5_0_5b_chat_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 29, 1024)"},
        },
    ),
    (
        Reshape764,
        [((1, 16, 29, 64), torch.float32)],
        {
            "model_name": ["pt_qwen1_5_qwen_qwen1_5_0_5b_chat_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(16, 29, 64)"},
        },
    ),
    (
        Reshape765,
        [((16, 29, 29), torch.float32)],
        {
            "model_name": ["pt_qwen1_5_qwen_qwen1_5_0_5b_chat_clm_hf", "pt_qwen_v2_qwen_qwen2_5_3b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 16, 29, 29)"},
        },
    ),
    (
        Reshape766,
        [((1, 16, 29, 29), torch.float32)],
        {
            "model_name": ["pt_qwen1_5_qwen_qwen1_5_0_5b_chat_clm_hf", "pt_qwen_v2_qwen_qwen2_5_3b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(16, 29, 29)"},
        },
    ),
    (
        Reshape767,
        [((1, 16, 64, 29), torch.float32)],
        {
            "model_name": ["pt_qwen1_5_qwen_qwen1_5_0_5b_chat_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(16, 64, 29)"},
        },
    ),
    (
        Reshape768,
        [((16, 29, 64), torch.float32)],
        {
            "model_name": ["pt_qwen1_5_qwen_qwen1_5_0_5b_chat_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 16, 29, 64)"},
        },
    ),
    (
        Reshape761,
        [((1, 29, 16, 64), torch.float32)],
        {"model_name": ["pt_qwen1_5_qwen_qwen1_5_0_5b_chat_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(29, 1024)"}},
    ),
    (
        Reshape769,
        [((29, 2816), torch.float32)],
        {
            "model_name": ["pt_qwen1_5_qwen_qwen1_5_0_5b_chat_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 29, 2816)"},
        },
    ),
    (
        Reshape770,
        [((1, 6, 1024), torch.float32)],
        {"model_name": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(6, 1024)"}},
    ),
    (
        Reshape771,
        [((1, 6, 1024), torch.float32)],
        {"model_name": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 6, 16, 64)"}},
    ),
    (
        Reshape772,
        [((6, 1024), torch.float32)],
        {"model_name": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 6, 1024)"}},
    ),
    (
        Reshape773,
        [((1, 16, 6, 64), torch.float32)],
        {"model_name": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(16, 6, 64)"}},
    ),
    (
        Reshape774,
        [((16, 6, 6), torch.float32)],
        {"model_name": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 16, 6, 6)"}},
    ),
    (
        Reshape775,
        [((1, 16, 6, 6), torch.float32)],
        {"model_name": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(16, 6, 6)"}},
    ),
    (
        Reshape776,
        [((1, 16, 64, 6), torch.float32)],
        {"model_name": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(16, 64, 6)"}},
    ),
    (
        Reshape777,
        [((16, 6, 64), torch.float32)],
        {"model_name": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 16, 6, 64)"}},
    ),
    (
        Reshape770,
        [((1, 6, 16, 64), torch.float32)],
        {"model_name": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(6, 1024)"}},
    ),
    (
        Reshape778,
        [((6, 2816), torch.float32)],
        {"model_name": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 6, 2816)"}},
    ),
    (
        Reshape779,
        [((1, 35, 1536), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(35, 1536)"},
        },
    ),
    (
        Reshape780,
        [((1, 35, 1536), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 35, 12, 128)"},
        },
    ),
    (
        Reshape781,
        [((35, 1536), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 35, 1536)"},
        },
    ),
    (
        Reshape782,
        [((1, 12, 35, 128), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(12, 35, 128)"},
        },
    ),
    (
        Reshape783,
        [((35, 256), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 35, 256)"},
        },
    ),
    (
        Reshape784,
        [((1, 35, 256), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 35, 2, 128)"},
        },
    ),
    (
        Reshape782,
        [((1, 2, 6, 35, 128), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(12, 35, 128)"},
        },
    ),
    (
        Reshape785,
        [((1, 2, 6, 35, 128), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 12, 35, 128)"},
        },
    ),
    (
        Reshape786,
        [((12, 35, 35), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 12, 35, 35)"},
        },
    ),
    (
        Reshape787,
        [((1, 12, 35, 35), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(12, 35, 35)"},
        },
    ),
    (
        Reshape788,
        [((1, 12, 128, 35), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(12, 128, 35)"},
        },
    ),
    (
        Reshape785,
        [((12, 35, 128), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 12, 35, 128)"},
        },
    ),
    (
        Reshape779,
        [((1, 35, 12, 128), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(35, 1536)"},
        },
    ),
    (
        Reshape789,
        [((35, 8960), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 35, 8960)"},
        },
    ),
    (
        Reshape790,
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
        Reshape791,
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
        Reshape792,
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
        Reshape793,
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
        Reshape794,
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
        Reshape795,
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
        Reshape793,
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
        Reshape796,
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
        Reshape797,
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
        Reshape798,
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
        Reshape799,
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
        Reshape796,
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
        Reshape790,
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
        Reshape800,
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
        Reshape801,
        [((1, 35, 896), torch.float32)],
        {
            "model_name": ["pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(35, 896)"},
        },
    ),
    (
        Reshape802,
        [((1, 35, 896), torch.float32)],
        {
            "model_name": ["pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 35, 14, 64)"},
        },
    ),
    (
        Reshape803,
        [((35, 896), torch.float32)],
        {
            "model_name": ["pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 35, 896)"},
        },
    ),
    (
        Reshape804,
        [((1, 14, 35, 64), torch.float32)],
        {
            "model_name": ["pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(14, 35, 64)"},
        },
    ),
    (
        Reshape805,
        [((35, 128), torch.float32)],
        {
            "model_name": ["pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 35, 128)"},
        },
    ),
    (
        Reshape806,
        [((1, 35, 128), torch.float32)],
        {
            "model_name": ["pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 35, 2, 64)"},
        },
    ),
    (
        Reshape804,
        [((1, 2, 7, 35, 64), torch.float32)],
        {
            "model_name": ["pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(14, 35, 64)"},
        },
    ),
    (
        Reshape807,
        [((1, 2, 7, 35, 64), torch.float32)],
        {
            "model_name": ["pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 14, 35, 64)"},
        },
    ),
    (
        Reshape808,
        [((14, 35, 35), torch.float32)],
        {
            "model_name": ["pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 14, 35, 35)"},
        },
    ),
    (
        Reshape809,
        [((1, 14, 35, 35), torch.float32)],
        {
            "model_name": ["pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(14, 35, 35)"},
        },
    ),
    (
        Reshape810,
        [((1, 14, 64, 35), torch.float32)],
        {
            "model_name": ["pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(14, 64, 35)"},
        },
    ),
    (
        Reshape807,
        [((14, 35, 64), torch.float32)],
        {
            "model_name": ["pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 14, 35, 64)"},
        },
    ),
    (
        Reshape801,
        [((1, 35, 14, 64), torch.float32)],
        {
            "model_name": ["pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(35, 896)"},
        },
    ),
    (
        Reshape811,
        [((35, 4864), torch.float32)],
        {
            "model_name": ["pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 35, 4864)"},
        },
    ),
    (
        Reshape812,
        [((1, 35, 2048), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(35, 2048)"},
        },
    ),
    (
        Reshape813,
        [((1, 35, 2048), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 35, 16, 128)"},
        },
    ),
    (
        Reshape814,
        [((35, 2048), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 35, 2048)"},
        },
    ),
    (
        Reshape815,
        [((1, 16, 35, 128), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(16, 35, 128)"},
        },
    ),
    (
        Reshape815,
        [((1, 2, 8, 35, 128), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(16, 35, 128)"},
        },
    ),
    (
        Reshape816,
        [((1, 2, 8, 35, 128), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 16, 35, 128)"},
        },
    ),
    (
        Reshape817,
        [((16, 35, 35), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 16, 35, 35)"},
        },
    ),
    (
        Reshape818,
        [((1, 16, 35, 35), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(16, 35, 35)"},
        },
    ),
    (
        Reshape819,
        [((1, 16, 128, 35), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(16, 128, 35)"},
        },
    ),
    (
        Reshape816,
        [((16, 35, 128), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 16, 35, 128)"},
        },
    ),
    (
        Reshape812,
        [((1, 35, 16, 128), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(35, 2048)"},
        },
    ),
    (
        Reshape820,
        [((35, 11008), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 35, 11008)"},
        },
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
        Reshape825,
        [((39, 256), torch.float32)],
        {
            "model_name": [
                "pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm_hf",
                "pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 39, 256)"},
        },
    ),
    (
        Reshape826,
        [((1, 39, 256), torch.float32)],
        {
            "model_name": [
                "pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm_hf",
                "pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 39, 2, 128)"},
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
        Reshape827,
        [((1, 2, 8, 39, 128), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 16, 39, 128)"},
        },
    ),
    (
        Reshape828,
        [((16, 39, 39), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 16, 39, 39)"},
        },
    ),
    (
        Reshape829,
        [((1, 16, 39, 39), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(16, 39, 39)"},
        },
    ),
    (
        Reshape830,
        [((1, 16, 128, 39), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(16, 128, 39)"},
        },
    ),
    (
        Reshape827,
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
        Reshape831,
        [((1, 29, 2048), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_3b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(29, 2048)"}},
    ),
    (
        Reshape832,
        [((1, 29, 2048), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_3b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 29, 16, 128)"}},
    ),
    (
        Reshape833,
        [((29, 2048), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_3b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 29, 2048)"}},
    ),
    (
        Reshape834,
        [((1, 16, 29, 128), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_3b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(16, 29, 128)"}},
    ),
    (
        Reshape835,
        [((29, 256), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_3b_clm_hf", "pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 29, 256)"},
        },
    ),
    (
        Reshape836,
        [((1, 29, 256), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_3b_clm_hf", "pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 29, 2, 128)"},
        },
    ),
    (
        Reshape834,
        [((1, 2, 8, 29, 128), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_3b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(16, 29, 128)"}},
    ),
    (
        Reshape837,
        [((1, 2, 8, 29, 128), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_3b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 16, 29, 128)"}},
    ),
    (
        Reshape838,
        [((1, 16, 128, 29), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_3b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(16, 128, 29)"}},
    ),
    (
        Reshape837,
        [((16, 29, 128), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_3b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 16, 29, 128)"}},
    ),
    (
        Reshape831,
        [((1, 29, 16, 128), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_3b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(29, 2048)"}},
    ),
    (
        Reshape839,
        [((29, 11008), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_3b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 29, 11008)"}},
    ),
    (
        Reshape840,
        [((1, 29, 1536), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(29, 1536)"}},
    ),
    (
        Reshape841,
        [((1, 29, 1536), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 29, 12, 128)"},
        },
    ),
    (
        Reshape842,
        [((29, 1536), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 29, 1536)"}},
    ),
    (
        Reshape843,
        [((1, 12, 29, 128), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(12, 29, 128)"}},
    ),
    (
        Reshape843,
        [((1, 2, 6, 29, 128), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(12, 29, 128)"}},
    ),
    (
        Reshape844,
        [((1, 2, 6, 29, 128), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 12, 29, 128)"},
        },
    ),
    (
        Reshape845,
        [((12, 29, 29), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 12, 29, 29)"}},
    ),
    (
        Reshape846,
        [((1, 12, 29, 29), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(12, 29, 29)"}},
    ),
    (
        Reshape847,
        [((1, 12, 128, 29), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(12, 128, 29)"}},
    ),
    (
        Reshape844,
        [((12, 29, 128), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 12, 29, 128)"},
        },
    ),
    (
        Reshape840,
        [((1, 29, 12, 128), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(29, 1536)"}},
    ),
    (
        Reshape848,
        [((29, 8960), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 29, 8960)"}},
    ),
    (
        Reshape849,
        [((1, 39, 1536), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(39, 1536)"},
        },
    ),
    (
        Reshape850,
        [((1, 39, 1536), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 39, 12, 128)"},
        },
    ),
    (
        Reshape851,
        [((39, 1536), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 39, 1536)"},
        },
    ),
    (
        Reshape852,
        [((1, 12, 39, 128), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(12, 39, 128)"},
        },
    ),
    (
        Reshape852,
        [((1, 2, 6, 39, 128), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(12, 39, 128)"},
        },
    ),
    (
        Reshape853,
        [((1, 2, 6, 39, 128), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 12, 39, 128)"},
        },
    ),
    (
        Reshape854,
        [((12, 39, 39), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 12, 39, 39)"},
        },
    ),
    (
        Reshape855,
        [((1, 12, 39, 39), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(12, 39, 39)"},
        },
    ),
    (
        Reshape856,
        [((1, 12, 128, 39), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(12, 128, 39)"},
        },
    ),
    (
        Reshape853,
        [((12, 39, 128), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 12, 39, 128)"},
        },
    ),
    (
        Reshape849,
        [((1, 39, 12, 128), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(39, 1536)"},
        },
    ),
    (
        Reshape857,
        [((39, 8960), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 39, 8960)"},
        },
    ),
    (
        Reshape858,
        [((1, 39, 896), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(39, 896)"},
        },
    ),
    (
        Reshape859,
        [((1, 39, 896), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 39, 14, 64)"},
        },
    ),
    (
        Reshape860,
        [((39, 896), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 39, 896)"},
        },
    ),
    (
        Reshape861,
        [((1, 14, 39, 64), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(14, 39, 64)"},
        },
    ),
    (
        Reshape862,
        [((39, 128), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 39, 128)"},
        },
    ),
    (
        Reshape863,
        [((1, 39, 128), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 39, 2, 64)"},
        },
    ),
    (
        Reshape861,
        [((1, 2, 7, 39, 64), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(14, 39, 64)"},
        },
    ),
    (
        Reshape864,
        [((1, 2, 7, 39, 64), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 14, 39, 64)"},
        },
    ),
    (
        Reshape865,
        [((14, 39, 39), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 14, 39, 39)"},
        },
    ),
    (
        Reshape866,
        [((1, 14, 39, 39), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(14, 39, 39)"},
        },
    ),
    (
        Reshape867,
        [((1, 14, 64, 39), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(14, 64, 39)"},
        },
    ),
    (
        Reshape864,
        [((14, 39, 64), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 14, 39, 64)"},
        },
    ),
    (
        Reshape858,
        [((1, 39, 14, 64), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(39, 896)"},
        },
    ),
    (
        Reshape868,
        [((39, 4864), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 39, 4864)"},
        },
    ),
    (
        Reshape869,
        [((1, 39, 3584), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(39, 3584)"},
        },
    ),
    (
        Reshape870,
        [((1, 39, 3584), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 39, 28, 128)"},
        },
    ),
    (
        Reshape871,
        [((39, 3584), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 39, 3584)"},
        },
    ),
    (
        Reshape872,
        [((1, 28, 39, 128), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(28, 39, 128)"},
        },
    ),
    (
        Reshape873,
        [((39, 512), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 39, 512)"},
        },
    ),
    (
        Reshape874,
        [((1, 39, 512), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 39, 4, 128)"},
        },
    ),
    (
        Reshape872,
        [((1, 4, 7, 39, 128), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(28, 39, 128)"},
        },
    ),
    (
        Reshape875,
        [((1, 4, 7, 39, 128), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 28, 39, 128)"},
        },
    ),
    (
        Reshape876,
        [((28, 39, 39), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 28, 39, 39)"},
        },
    ),
    (
        Reshape877,
        [((1, 28, 39, 39), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(28, 39, 39)"},
        },
    ),
    (
        Reshape878,
        [((1, 28, 128, 39), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(28, 128, 39)"},
        },
    ),
    (
        Reshape875,
        [((28, 39, 128), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 28, 39, 128)"},
        },
    ),
    (
        Reshape869,
        [((1, 39, 28, 128), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(39, 3584)"},
        },
    ),
    (
        Reshape879,
        [((39, 18944), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 39, 18944)"},
        },
    ),
    (
        Reshape880,
        [((1, 29, 896), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(29, 896)"}},
    ),
    (
        Reshape881,
        [((1, 29, 896), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 29, 14, 64)"}},
    ),
    (
        Reshape882,
        [((29, 896), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 29, 896)"}},
    ),
    (
        Reshape883,
        [((1, 14, 29, 64), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(14, 29, 64)"}},
    ),
    (
        Reshape884,
        [((29, 128), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 29, 128)"}},
    ),
    (
        Reshape885,
        [((1, 29, 128), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 29, 2, 64)"}},
    ),
    (
        Reshape883,
        [((1, 2, 7, 29, 64), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(14, 29, 64)"}},
    ),
    (
        Reshape886,
        [((1, 2, 7, 29, 64), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 14, 29, 64)"}},
    ),
    (
        Reshape887,
        [((14, 29, 29), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 14, 29, 29)"}},
    ),
    (
        Reshape888,
        [((1, 14, 29, 29), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(14, 29, 29)"}},
    ),
    (
        Reshape889,
        [((1, 14, 64, 29), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(14, 64, 29)"}},
    ),
    (
        Reshape886,
        [((14, 29, 64), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 14, 29, 64)"}},
    ),
    (
        Reshape880,
        [((1, 29, 14, 64), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(29, 896)"}},
    ),
    (
        Reshape890,
        [((29, 4864), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 29, 4864)"}},
    ),
    (
        Reshape891,
        [((1, 13, 3584), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_7b_token_cls_hf"], "pcc": 0.99, "op_params": {"shape": "(13, 3584)"}},
    ),
    (
        Reshape892,
        [((1, 13, 3584), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_7b_token_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 13, 28, 128)"},
        },
    ),
    (
        Reshape893,
        [((13, 3584), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_7b_token_cls_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 13, 3584)"}},
    ),
    (
        Reshape894,
        [((1, 28, 13, 128), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_7b_token_cls_hf"], "pcc": 0.99, "op_params": {"shape": "(28, 13, 128)"}},
    ),
    (
        Reshape895,
        [((13, 512), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_7b_token_cls_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 13, 512)"}},
    ),
    (
        Reshape896,
        [((1, 13, 512), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_7b_token_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 13, 4, 128)"},
        },
    ),
    (
        Reshape894,
        [((1, 4, 7, 13, 128), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_7b_token_cls_hf"], "pcc": 0.99, "op_params": {"shape": "(28, 13, 128)"}},
    ),
    (
        Reshape897,
        [((1, 4, 7, 13, 128), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_7b_token_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 28, 13, 128)"},
        },
    ),
    (
        Reshape898,
        [((28, 13, 13), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_7b_token_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 28, 13, 13)"},
        },
    ),
    (
        Reshape899,
        [((1, 28, 13, 13), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_7b_token_cls_hf"], "pcc": 0.99, "op_params": {"shape": "(28, 13, 13)"}},
    ),
    (
        Reshape900,
        [((1, 28, 128, 13), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_7b_token_cls_hf"], "pcc": 0.99, "op_params": {"shape": "(28, 128, 13)"}},
    ),
    (
        Reshape897,
        [((28, 13, 128), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_7b_token_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 28, 13, 128)"},
        },
    ),
    (
        Reshape891,
        [((1, 13, 28, 128), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_7b_token_cls_hf"], "pcc": 0.99, "op_params": {"shape": "(13, 3584)"}},
    ),
    (
        Reshape901,
        [((13, 18944), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_7b_token_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 13, 18944)"},
        },
    ),
    (
        Reshape902,
        [((1, 29, 3584), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_7b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(29, 3584)"}},
    ),
    (
        Reshape903,
        [((1, 29, 3584), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_7b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 29, 28, 128)"}},
    ),
    (
        Reshape904,
        [((29, 3584), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_7b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 29, 3584)"}},
    ),
    (
        Reshape905,
        [((1, 28, 29, 128), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_7b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(28, 29, 128)"}},
    ),
    (
        Reshape906,
        [((29, 512), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_7b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 29, 512)"}},
    ),
    (
        Reshape907,
        [((1, 29, 512), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_7b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 29, 4, 128)"}},
    ),
    (
        Reshape905,
        [((1, 4, 7, 29, 128), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_7b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(28, 29, 128)"}},
    ),
    (
        Reshape908,
        [((1, 4, 7, 29, 128), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_7b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 28, 29, 128)"}},
    ),
    (
        Reshape909,
        [((28, 29, 29), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_7b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 28, 29, 29)"}},
    ),
    (
        Reshape910,
        [((1, 28, 29, 29), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_7b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(28, 29, 29)"}},
    ),
    (
        Reshape911,
        [((1, 28, 128, 29), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_7b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(28, 128, 29)"}},
    ),
    (
        Reshape908,
        [((28, 29, 128), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_7b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 28, 29, 128)"}},
    ),
    (
        Reshape902,
        [((1, 29, 28, 128), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_7b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(29, 3584)"}},
    ),
    (
        Reshape912,
        [((29, 18944), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_7b_clm_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 29, 18944)"}},
    ),
    (
        Reshape913,
        [((1, 768, 128), torch.float32)],
        {
            "model_name": ["pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 12, 64, 128)"},
        },
    ),
    (
        Reshape23,
        [((1, 768, 128), torch.float32)],
        {
            "model_name": ["pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(12, 64, 128)"},
        },
    ),
    (
        Reshape914,
        [((768, 768, 1), torch.float32)],
        {
            "model_name": ["pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(768, 768, 1, 1)"},
        },
    ),
    (
        Reshape915,
        [((1, 768, 128, 1), torch.float32)],
        {
            "model_name": ["pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 768, 128)"},
        },
    ),
    (
        Reshape916,
        [((1, 61), torch.int64)],
        {
            "model_name": [
                "pt_t5_google_flan_t5_base_text_gen_hf",
                "pt_t5_t5_small_text_gen_hf",
                "pt_t5_t5_large_text_gen_hf",
                "pt_t5_google_flan_t5_large_text_gen_hf",
                "pt_t5_t5_base_text_gen_hf",
                "pt_t5_google_flan_t5_small_text_gen_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 61)"},
        },
    ),
    (
        Reshape917,
        [((1, 61, 768), torch.float32)],
        {
            "model_name": ["pt_t5_google_flan_t5_base_text_gen_hf", "pt_t5_t5_base_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(61, 768)"},
        },
    ),
    (
        Reshape918,
        [((61, 768), torch.float32)],
        {
            "model_name": ["pt_t5_google_flan_t5_base_text_gen_hf", "pt_t5_t5_base_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 61, 12, 64)"},
        },
    ),
    (
        Reshape919,
        [((61, 768), torch.float32)],
        {
            "model_name": ["pt_t5_google_flan_t5_base_text_gen_hf", "pt_t5_t5_base_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 61, 768)"},
        },
    ),
    (
        Reshape920,
        [((1, 12, 61, 64), torch.float32)],
        {
            "model_name": ["pt_t5_google_flan_t5_base_text_gen_hf", "pt_t5_t5_base_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(12, 61, 64)"},
        },
    ),
    (
        Reshape921,
        [((12, 61, 61), torch.float32)],
        {
            "model_name": ["pt_t5_google_flan_t5_base_text_gen_hf", "pt_t5_t5_base_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 12, 61, 61)"},
        },
    ),
    (
        Reshape922,
        [((1, 12, 61, 61), torch.float32)],
        {
            "model_name": ["pt_t5_google_flan_t5_base_text_gen_hf", "pt_t5_t5_base_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(12, 61, 61)"},
        },
    ),
    (
        Reshape923,
        [((1, 12, 64, 61), torch.float32)],
        {
            "model_name": ["pt_t5_google_flan_t5_base_text_gen_hf", "pt_t5_t5_base_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(12, 64, 61)"},
        },
    ),
    (
        Reshape924,
        [((12, 61, 64), torch.float32)],
        {
            "model_name": ["pt_t5_google_flan_t5_base_text_gen_hf", "pt_t5_t5_base_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 12, 61, 64)"},
        },
    ),
    (
        Reshape917,
        [((1, 61, 12, 64), torch.float32)],
        {
            "model_name": ["pt_t5_google_flan_t5_base_text_gen_hf", "pt_t5_t5_base_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(61, 768)"},
        },
    ),
    (
        Reshape925,
        [((61, 2048), torch.float32)],
        {"model_name": ["pt_t5_google_flan_t5_base_text_gen_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 61, 2048)"}},
    ),
    (
        Reshape926,
        [((12, 1, 61), torch.float32)],
        {
            "model_name": ["pt_t5_google_flan_t5_base_text_gen_hf", "pt_t5_t5_base_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 12, 1, 61)"},
        },
    ),
    (
        Reshape927,
        [((1, 12, 1, 61), torch.float32)],
        {
            "model_name": ["pt_t5_google_flan_t5_base_text_gen_hf", "pt_t5_t5_base_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(12, 1, 61)"},
        },
    ),
    (
        Reshape928,
        [((1, 61, 512), torch.float32)],
        {
            "model_name": ["pt_t5_t5_small_text_gen_hf", "pt_t5_google_flan_t5_small_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(61, 512)"},
        },
    ),
    (
        Reshape929,
        [((61, 512), torch.float32)],
        {"model_name": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 61, 8, 64)"}},
    ),
    (
        Reshape930,
        [((61, 512), torch.float32)],
        {
            "model_name": ["pt_t5_t5_small_text_gen_hf", "pt_t5_google_flan_t5_small_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 61, 512)"},
        },
    ),
    (
        Reshape931,
        [((1, 8, 61, 64), torch.float32)],
        {"model_name": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99, "op_params": {"shape": "(8, 61, 64)"}},
    ),
    (
        Reshape932,
        [((8, 61, 61), torch.float32)],
        {"model_name": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 8, 61, 61)"}},
    ),
    (
        Reshape933,
        [((1, 8, 61, 61), torch.float32)],
        {"model_name": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99, "op_params": {"shape": "(8, 61, 61)"}},
    ),
    (
        Reshape934,
        [((1, 8, 64, 61), torch.float32)],
        {"model_name": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99, "op_params": {"shape": "(8, 64, 61)"}},
    ),
    (
        Reshape935,
        [((8, 61, 64), torch.float32)],
        {"model_name": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 8, 61, 64)"}},
    ),
    (
        Reshape928,
        [((1, 61, 8, 64), torch.float32)],
        {"model_name": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99, "op_params": {"shape": "(61, 512)"}},
    ),
    (
        Reshape936,
        [((8, 1, 61), torch.float32)],
        {"model_name": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 8, 1, 61)"}},
    ),
    (
        Reshape937,
        [((1, 8, 1, 61), torch.float32)],
        {"model_name": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99, "op_params": {"shape": "(8, 1, 61)"}},
    ),
    (
        Reshape938,
        [((1, 61, 1024), torch.float32)],
        {
            "model_name": ["pt_t5_t5_large_text_gen_hf", "pt_t5_google_flan_t5_large_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(61, 1024)"},
        },
    ),
    (
        Reshape939,
        [((61, 1024), torch.float32)],
        {
            "model_name": ["pt_t5_t5_large_text_gen_hf", "pt_t5_google_flan_t5_large_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 61, 16, 64)"},
        },
    ),
    (
        Reshape940,
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
        Reshape941,
        [((1, 16, 61, 64), torch.float32)],
        {
            "model_name": ["pt_t5_t5_large_text_gen_hf", "pt_t5_google_flan_t5_large_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(16, 61, 64)"},
        },
    ),
    (
        Reshape942,
        [((16, 61, 61), torch.float32)],
        {
            "model_name": ["pt_t5_t5_large_text_gen_hf", "pt_t5_google_flan_t5_large_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 16, 61, 61)"},
        },
    ),
    (
        Reshape943,
        [((1, 16, 61, 61), torch.float32)],
        {
            "model_name": ["pt_t5_t5_large_text_gen_hf", "pt_t5_google_flan_t5_large_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(16, 61, 61)"},
        },
    ),
    (
        Reshape944,
        [((1, 16, 64, 61), torch.float32)],
        {
            "model_name": ["pt_t5_t5_large_text_gen_hf", "pt_t5_google_flan_t5_large_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(16, 64, 61)"},
        },
    ),
    (
        Reshape945,
        [((16, 61, 64), torch.float32)],
        {
            "model_name": ["pt_t5_t5_large_text_gen_hf", "pt_t5_google_flan_t5_large_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 16, 61, 64)"},
        },
    ),
    (
        Reshape938,
        [((1, 61, 16, 64), torch.float32)],
        {
            "model_name": ["pt_t5_t5_large_text_gen_hf", "pt_t5_google_flan_t5_large_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(61, 1024)"},
        },
    ),
    (
        Reshape946,
        [((16, 1, 61), torch.float32)],
        {
            "model_name": ["pt_t5_t5_large_text_gen_hf", "pt_t5_google_flan_t5_large_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 16, 1, 61)"},
        },
    ),
    (
        Reshape947,
        [((1, 16, 1, 61), torch.float32)],
        {
            "model_name": ["pt_t5_t5_large_text_gen_hf", "pt_t5_google_flan_t5_large_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(16, 1, 61)"},
        },
    ),
    (
        Reshape948,
        [((61, 2816), torch.float32)],
        {
            "model_name": ["pt_t5_google_flan_t5_large_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 61, 2816)"},
        },
    ),
    (
        Reshape949,
        [((1, 2816), torch.float32)],
        {"model_name": ["pt_t5_google_flan_t5_large_text_gen_hf"], "pcc": 0.99, "op_params": {"shape": "(1, 1, 2816)"}},
    ),
    (
        Reshape950,
        [((61, 384), torch.float32)],
        {
            "model_name": ["pt_t5_google_flan_t5_small_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 61, 6, 64)"},
        },
    ),
    (
        Reshape951,
        [((1, 6, 61, 64), torch.float32)],
        {"model_name": ["pt_t5_google_flan_t5_small_text_gen_hf"], "pcc": 0.99, "op_params": {"shape": "(6, 61, 64)"}},
    ),
    (
        Reshape952,
        [((6, 61, 61), torch.float32)],
        {
            "model_name": ["pt_t5_google_flan_t5_small_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 6, 61, 61)"},
        },
    ),
    (
        Reshape953,
        [((1, 6, 61, 61), torch.float32)],
        {"model_name": ["pt_t5_google_flan_t5_small_text_gen_hf"], "pcc": 0.99, "op_params": {"shape": "(6, 61, 61)"}},
    ),
    (
        Reshape954,
        [((1, 6, 64, 61), torch.float32)],
        {"model_name": ["pt_t5_google_flan_t5_small_text_gen_hf"], "pcc": 0.99, "op_params": {"shape": "(6, 64, 61)"}},
    ),
    (
        Reshape955,
        [((6, 61, 64), torch.float32)],
        {
            "model_name": ["pt_t5_google_flan_t5_small_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 6, 61, 64)"},
        },
    ),
    (
        Reshape956,
        [((1, 61, 6, 64), torch.float32)],
        {"model_name": ["pt_t5_google_flan_t5_small_text_gen_hf"], "pcc": 0.99, "op_params": {"shape": "(61, 384)"}},
    ),
    (
        Reshape957,
        [((6, 1, 61), torch.float32)],
        {
            "model_name": ["pt_t5_google_flan_t5_small_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 6, 1, 61)"},
        },
    ),
    (
        Reshape958,
        [((1, 6, 1, 61), torch.float32)],
        {"model_name": ["pt_t5_google_flan_t5_small_text_gen_hf"], "pcc": 0.99, "op_params": {"shape": "(6, 1, 61)"}},
    ),
    (
        Reshape959,
        [((1, 96, 54, 54), torch.float32)],
        {"model_name": ["pt_alexnet_base_img_cls_osmr"], "pcc": 0.99, "op_params": {"shape": "(1, 1, 96, 54, 54)"}},
    ),
    (
        Reshape960,
        [((1, 96, 54, 54), torch.float32)],
        {"model_name": ["pt_alexnet_base_img_cls_osmr"], "pcc": 0.99, "op_params": {"shape": "(1, 96, 54, 54)"}},
    ),
    (
        Reshape961,
        [((1, 256, 27, 27), torch.float32)],
        {"model_name": ["pt_alexnet_base_img_cls_osmr"], "pcc": 0.99, "op_params": {"shape": "(1, 1, 256, 27, 27)"}},
    ),
    (
        Reshape962,
        [((1, 256, 27, 27), torch.float32)],
        {"model_name": ["pt_alexnet_base_img_cls_osmr"], "pcc": 0.99, "op_params": {"shape": "(1, 256, 27, 27)"}},
    ),
    (
        Reshape963,
        [((729, 12), torch.float32)],
        {
            "model_name": ["pt_beit_microsoft_beit_base_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 27, 27, 12)"},
        },
    ),
    (
        Reshape964,
        [((1, 27, 27, 12), torch.float32)],
        {
            "model_name": ["pt_beit_microsoft_beit_base_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(729, 12)"},
        },
    ),
    (
        Reshape965,
        [((38809, 12), torch.float32)],
        {
            "model_name": ["pt_beit_microsoft_beit_base_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(197, 197, 12)"},
        },
    ),
    (
        Reshape966,
        [((729, 16), torch.float32)],
        {
            "model_name": ["pt_beit_microsoft_beit_large_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 27, 27, 16)"},
        },
    ),
    (
        Reshape967,
        [((1, 27, 27, 16), torch.float32)],
        {
            "model_name": ["pt_beit_microsoft_beit_large_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(729, 16)"},
        },
    ),
    (
        Reshape968,
        [((38809, 16), torch.float32)],
        {
            "model_name": ["pt_beit_microsoft_beit_large_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(197, 197, 16)"},
        },
    ),
    (
        Reshape969,
        [((1, 384, 14, 14), torch.float32)],
        {
            "model_name": ["pt_deit_facebook_deit_small_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 384, 196, 1)"},
        },
    ),
    (
        Reshape970,
        [((1, 197, 384), torch.float32)],
        {
            "model_name": ["pt_deit_facebook_deit_small_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(197, 384)"},
        },
    ),
    (
        Reshape971,
        [((1, 197, 384), torch.float32)],
        {
            "model_name": ["pt_deit_facebook_deit_small_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 197, 6, 64)"},
        },
    ),
    (
        Reshape972,
        [((197, 384), torch.float32)],
        {
            "model_name": ["pt_deit_facebook_deit_small_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 197, 384)"},
        },
    ),
    (
        Reshape973,
        [((1, 6, 197, 64), torch.float32)],
        {
            "model_name": ["pt_deit_facebook_deit_small_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(6, 197, 64)"},
        },
    ),
    (
        Reshape974,
        [((6, 197, 197), torch.float32)],
        {
            "model_name": ["pt_deit_facebook_deit_small_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 6, 197, 197)"},
        },
    ),
    (
        Reshape975,
        [((1, 6, 197, 197), torch.float32)],
        {
            "model_name": ["pt_deit_facebook_deit_small_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(6, 197, 197)"},
        },
    ),
    (
        Reshape976,
        [((1, 6, 64, 197), torch.float32)],
        {
            "model_name": ["pt_deit_facebook_deit_small_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(6, 64, 197)"},
        },
    ),
    (
        Reshape977,
        [((6, 197, 64), torch.float32)],
        {
            "model_name": ["pt_deit_facebook_deit_small_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 6, 197, 64)"},
        },
    ),
    (
        Reshape970,
        [((1, 197, 6, 64), torch.float32)],
        {
            "model_name": ["pt_deit_facebook_deit_small_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(197, 384)"},
        },
    ),
    (
        Reshape978,
        [((1, 192, 14, 14), torch.float32)],
        {
            "model_name": ["pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 192, 196, 1)"},
        },
    ),
    (
        Reshape979,
        [((1, 197, 192), torch.float32)],
        {
            "model_name": ["pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(197, 192)"},
        },
    ),
    (
        Reshape980,
        [((1, 197, 192), torch.float32)],
        {
            "model_name": ["pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 197, 3, 64)"},
        },
    ),
    (
        Reshape981,
        [((197, 192), torch.float32)],
        {
            "model_name": ["pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 197, 192)"},
        },
    ),
    (
        Reshape982,
        [((1, 3, 197, 64), torch.float32)],
        {
            "model_name": ["pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(3, 197, 64)"},
        },
    ),
    (
        Reshape983,
        [((3, 197, 197), torch.float32)],
        {
            "model_name": ["pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 3, 197, 197)"},
        },
    ),
    (
        Reshape984,
        [((1, 3, 197, 197), torch.float32)],
        {
            "model_name": ["pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(3, 197, 197)"},
        },
    ),
    (
        Reshape985,
        [((1, 3, 64, 197), torch.float32)],
        {
            "model_name": ["pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(3, 64, 197)"},
        },
    ),
    (
        Reshape986,
        [((3, 197, 64), torch.float32)],
        {
            "model_name": ["pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 3, 197, 64)"},
        },
    ),
    (
        Reshape979,
        [((1, 197, 3, 64), torch.float32)],
        {
            "model_name": ["pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(197, 192)"},
        },
    ),
    (
        Reshape987,
        [((1, 1, 192), torch.float32)],
        {
            "model_name": ["pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 192)"},
        },
    ),
    (
        Reshape988,
        [((1, 2208, 1, 1), torch.float32)],
        {
            "model_name": ["pt_densenet_densenet161_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 2208, 1, 1)"},
        },
    ),
    (
        Reshape989,
        [((1, 1920, 1, 1), torch.float32)],
        {
            "model_name": ["pt_densenet_densenet201_img_cls_torchvision", "pt_regnet_regnet_x_8gf_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1920, 1, 1)"},
        },
    ),
    (
        Reshape990,
        [((1, 1664, 1, 1), torch.float32)],
        {
            "model_name": ["pt_densenet_densenet169_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1664, 1, 1)"},
        },
    ),
    (
        Reshape674,
        [((1, 1000, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_dla_dla169_visual_bb_torchvision",
                "pt_dla_dla60x_visual_bb_torchvision",
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_dla_dla60x_c_visual_bb_torchvision",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_dla_dla102x_visual_bb_torchvision",
                "pt_dla_dla34_visual_bb_torchvision",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_dla_dla46x_c_visual_bb_torchvision",
                "pt_dla_dla46_c_visual_bb_torchvision",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1000)"},
        },
    ),
    (
        Reshape991,
        [((1, 1000, 1, 1), torch.float32)],
        {"model_name": ["pt_dla_dla34_in1k_img_cls_timm"], "pcc": 0.99, "op_params": {"shape": "(1, 1000, 1, 1)"}},
    ),
    (
        Reshape992,
        [((32, 1, 3, 3), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                "pt_mobilenet_v1_basic_img_cls_torchvision",
                "pt_mobilenet_v1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
                "pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf",
                "pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_yolox_yolox_nano_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(32, 1, 3, 3)"},
        },
    ),
    (
        Reshape993,
        [((96, 1, 3, 3), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                "pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf",
                "pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(96, 1, 3, 3)"},
        },
    ),
    (
        Reshape994,
        [((144, 1, 3, 3), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                "pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(144, 1, 3, 3)"},
        },
    ),
    (
        Reshape995,
        [((144, 1, 5, 5), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(144, 1, 5, 5)"},
        },
    ),
    (
        Reshape996,
        [((240, 1, 5, 5), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(240, 1, 5, 5)"},
        },
    ),
    (
        Reshape997,
        [((240, 1, 3, 3), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(240, 1, 3, 3)"},
        },
    ),
    (
        Reshape998,
        [((480, 1, 3, 3), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(480, 1, 3, 3)"},
        },
    ),
    (
        Reshape999,
        [((480, 1, 5, 5), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(480, 1, 5, 5)"},
        },
    ),
    (
        Reshape1000,
        [((672, 1, 5, 5), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(672, 1, 5, 5)"},
        },
    ),
    (
        Reshape1001,
        [((1152, 1, 5, 5), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1152, 1, 5, 5)"},
        },
    ),
    (
        Reshape1002,
        [((1152, 1, 3, 3), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1152, 1, 3, 3)"},
        },
    ),
    (
        Reshape1003,
        [((48, 1, 3, 3), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf",
                "pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(48, 1, 3, 3)"},
        },
    ),
    (
        Reshape1004,
        [((24, 1, 3, 3), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(24, 1, 3, 3)"},
        },
    ),
    (
        Reshape1005,
        [((192, 1, 3, 3), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                "pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf",
                "pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(192, 1, 3, 3)"},
        },
    ),
    (
        Reshape1006,
        [((192, 1, 5, 5), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(192, 1, 5, 5)"},
        },
    ),
    (
        Reshape1007,
        [((336, 1, 5, 5), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(336, 1, 5, 5)"},
        },
    ),
    (
        Reshape1008,
        [((336, 1, 3, 3), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(336, 1, 3, 3)"},
        },
    ),
    (
        Reshape1009,
        [((672, 1, 3, 3), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(672, 1, 3, 3)"},
        },
    ),
    (
        Reshape1010,
        [((960, 1, 5, 5), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(960, 1, 5, 5)"},
        },
    ),
    (
        Reshape1011,
        [((1632, 1, 5, 5), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1632, 1, 5, 5)"},
        },
    ),
    (
        Reshape1012,
        [((1632, 1, 3, 3), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1632, 1, 3, 3)"},
        },
    ),
    (
        Reshape1013,
        [((2688, 1, 3, 3), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(2688, 1, 3, 3)"},
        },
    ),
    (
        Reshape1014,
        [((1, 1792, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1792, 1, 1)"},
        },
    ),
    (
        Reshape1015,
        [((288, 1, 5, 5), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(288, 1, 5, 5)"},
        },
    ),
    (
        Reshape1016,
        [((288, 1, 3, 3), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(288, 1, 3, 3)"},
        },
    ),
    (
        Reshape1017,
        [((528, 1, 3, 3), torch.float32)],
        {
            "model_name": ["pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {"shape": "(528, 1, 3, 3)"},
        },
    ),
    (
        Reshape1018,
        [((528, 1, 5, 5), torch.float32)],
        {
            "model_name": ["pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {"shape": "(528, 1, 5, 5)"},
        },
    ),
    (
        Reshape1019,
        [((720, 1, 5, 5), torch.float32)],
        {
            "model_name": ["pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {"shape": "(720, 1, 5, 5)"},
        },
    ),
    (
        Reshape1020,
        [((1248, 1, 5, 5), torch.float32)],
        {
            "model_name": ["pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {"shape": "(1248, 1, 5, 5)"},
        },
    ),
    (
        Reshape1021,
        [((1248, 1, 3, 3), torch.float32)],
        {
            "model_name": ["pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {"shape": "(1248, 1, 3, 3)"},
        },
    ),
    (
        Reshape1022,
        [((576, 1, 3, 3), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                "pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(576, 1, 3, 3)"},
        },
    ),
    (
        Reshape1023,
        [((576, 1, 5, 5), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(576, 1, 5, 5)"},
        },
    ),
    (
        Reshape1024,
        [((816, 1, 5, 5), torch.float32)],
        {
            "model_name": ["pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {"shape": "(816, 1, 5, 5)"},
        },
    ),
    (
        Reshape1025,
        [((1392, 1, 5, 5), torch.float32)],
        {
            "model_name": ["pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {"shape": "(1392, 1, 5, 5)"},
        },
    ),
    (
        Reshape1026,
        [((1392, 1, 3, 3), torch.float32)],
        {
            "model_name": ["pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {"shape": "(1392, 1, 3, 3)"},
        },
    ),
    (
        Reshape1027,
        [((8, 1, 3, 3), torch.float32)],
        {
            "model_name": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(8, 1, 3, 3)"},
        },
    ),
    (
        Reshape1028,
        [((12, 1, 3, 3), torch.float32)],
        {
            "model_name": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(12, 1, 3, 3)"},
        },
    ),
    (
        Reshape1029,
        [((16, 1, 3, 3), torch.float32)],
        {
            "model_name": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_yolox_yolox_nano_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(16, 1, 3, 3)"},
        },
    ),
    (
        Reshape1030,
        [((36, 1, 3, 3), torch.float32)],
        {
            "model_name": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(36, 1, 3, 3)"},
        },
    ),
    (
        Reshape1031,
        [((72, 1, 5, 5), torch.float32)],
        {
            "model_name": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(72, 1, 5, 5)"},
        },
    ),
    (
        Reshape1032,
        [((20, 1, 3, 3), torch.float32)],
        {
            "model_name": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(20, 1, 3, 3)"},
        },
    ),
    (
        Reshape1033,
        [((24, 1, 5, 5), torch.float32)],
        {
            "model_name": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(24, 1, 5, 5)"},
        },
    ),
    (
        Reshape1034,
        [((60, 1, 3, 3), torch.float32)],
        {
            "model_name": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(60, 1, 3, 3)"},
        },
    ),
    (
        Reshape1035,
        [((120, 1, 3, 3), torch.float32)],
        {
            "model_name": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(120, 1, 3, 3)"},
        },
    ),
    (
        Reshape1036,
        [((40, 1, 3, 3), torch.float32)],
        {
            "model_name": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(40, 1, 3, 3)"},
        },
    ),
    (
        Reshape1037,
        [((100, 1, 3, 3), torch.float32)],
        {
            "model_name": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(100, 1, 3, 3)"},
        },
    ),
    (
        Reshape1038,
        [((92, 1, 3, 3), torch.float32)],
        {
            "model_name": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(92, 1, 3, 3)"},
        },
    ),
    (
        Reshape1039,
        [((56, 1, 3, 3), torch.float32)],
        {
            "model_name": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(56, 1, 3, 3)"},
        },
    ),
    (
        Reshape1040,
        [((80, 1, 3, 3), torch.float32)],
        {
            "model_name": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(80, 1, 3, 3)"},
        },
    ),
    (
        Reshape1041,
        [((112, 1, 5, 5), torch.float32)],
        {
            "model_name": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(112, 1, 5, 5)"},
        },
    ),
    (
        Reshape1042,
        [((72, 1, 1, 5), torch.float32)],
        {
            "model_name": ["pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {"shape": "(72, 1, 1, 5)"},
        },
    ),
    (
        Reshape1043,
        [((72, 1, 5, 1), torch.float32)],
        {
            "model_name": ["pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {"shape": "(72, 1, 5, 1)"},
        },
    ),
    (
        Reshape1044,
        [((120, 1, 1, 5), torch.float32)],
        {
            "model_name": ["pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {"shape": "(120, 1, 1, 5)"},
        },
    ),
    (
        Reshape1045,
        [((120, 1, 5, 1), torch.float32)],
        {
            "model_name": ["pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {"shape": "(120, 1, 5, 1)"},
        },
    ),
    (
        Reshape1046,
        [((240, 1, 1, 5), torch.float32)],
        {
            "model_name": ["pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {"shape": "(240, 1, 1, 5)"},
        },
    ),
    (
        Reshape1047,
        [((240, 1, 5, 1), torch.float32)],
        {
            "model_name": ["pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {"shape": "(240, 1, 5, 1)"},
        },
    ),
    (
        Reshape1048,
        [((200, 1, 1, 5), torch.float32)],
        {
            "model_name": ["pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {"shape": "(200, 1, 1, 5)"},
        },
    ),
    (
        Reshape1049,
        [((200, 1, 5, 1), torch.float32)],
        {
            "model_name": ["pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {"shape": "(200, 1, 5, 1)"},
        },
    ),
    (
        Reshape1050,
        [((184, 1, 1, 5), torch.float32)],
        {
            "model_name": ["pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {"shape": "(184, 1, 1, 5)"},
        },
    ),
    (
        Reshape1051,
        [((184, 1, 5, 1), torch.float32)],
        {
            "model_name": ["pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {"shape": "(184, 1, 5, 1)"},
        },
    ),
    (
        Reshape1052,
        [((480, 1, 1, 5), torch.float32)],
        {
            "model_name": ["pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {"shape": "(480, 1, 1, 5)"},
        },
    ),
    (
        Reshape1053,
        [((480, 1, 5, 1), torch.float32)],
        {
            "model_name": ["pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {"shape": "(480, 1, 5, 1)"},
        },
    ),
    (
        Reshape1054,
        [((672, 1, 1, 5), torch.float32)],
        {
            "model_name": ["pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {"shape": "(672, 1, 1, 5)"},
        },
    ),
    (
        Reshape1055,
        [((672, 1, 5, 1), torch.float32)],
        {
            "model_name": ["pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {"shape": "(672, 1, 5, 1)"},
        },
    ),
    (
        Reshape1056,
        [((960, 1, 1, 5), torch.float32)],
        {
            "model_name": ["pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {"shape": "(960, 1, 1, 5)"},
        },
    ),
    (
        Reshape1057,
        [((960, 1, 5, 1), torch.float32)],
        {
            "model_name": ["pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {"shape": "(960, 1, 5, 1)"},
        },
    ),
    (
        Reshape1058,
        [((1, 64, 120, 160), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 64, 19200, 1)"},
        },
    ),
    (
        Reshape1059,
        [((1, 19200, 64), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 19200, 1, 64)"},
        },
    ),
    (
        Reshape1060,
        [((1, 19200, 64), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 120, 160, 64)"},
        },
    ),
    (
        Reshape1061,
        [((1, 64, 19200), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 64, 120, 160)"},
        },
    ),
    (
        Reshape1062,
        [((1, 64, 15, 20), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 64, 300)"},
        },
    ),
    (
        Reshape1063,
        [((1, 300, 64), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(300, 64)"},
        },
    ),
    (
        Reshape1064,
        [((1, 300, 64), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 300, 1, 64)"},
        },
    ),
    (
        Reshape1065,
        [((300, 64), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 300, 64)"},
        },
    ),
    (
        Reshape1066,
        [((1, 19200, 300), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1, 19200, 300)"},
        },
    ),
    (
        Reshape1067,
        [((1, 1, 19200, 300), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 19200, 300)"},
        },
    ),
    (
        Reshape1062,
        [((1, 1, 64, 300), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 64, 300)"},
        },
    ),
    (
        Reshape1068,
        [((1, 256, 19200), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 256, 120, 160)"},
        },
    ),
    (
        Reshape1069,
        [((256, 1, 3, 3), torch.float32)],
        {
            "model_name": [
                "pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf",
                "pt_mobilenet_v1_basic_img_cls_torchvision",
                "pt_mobilenet_v1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
                "pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
                "pt_xception_xception71_tf_in1k_img_cls_timm",
                "pt_xception_xception65_img_cls_timm",
                "pt_xception_xception_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(256, 1, 3, 3)"},
        },
    ),
    (
        Reshape1070,
        [((1, 256, 120, 160), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 256, 19200, 1)"},
        },
    ),
    (
        Reshape1071,
        [((1, 128, 60, 80), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 128, 4800, 1)"},
        },
    ),
    (
        Reshape1072,
        [((1, 4800, 128), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 4800, 2, 64)"},
        },
    ),
    (
        Reshape1073,
        [((1, 4800, 128), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 60, 80, 128)"},
        },
    ),
    (
        Reshape1074,
        [((1, 2, 4800, 64), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(2, 4800, 64)"},
        },
    ),
    (
        Reshape1075,
        [((1, 128, 4800), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 128, 60, 80)"},
        },
    ),
    (
        Reshape1076,
        [((1, 128, 15, 20), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 128, 300)"},
        },
    ),
    (
        Reshape1077,
        [((1, 300, 128), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(300, 128)"},
        },
    ),
    (
        Reshape1078,
        [((1, 300, 128), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 300, 2, 64)"},
        },
    ),
    (
        Reshape1079,
        [((300, 128), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 300, 128)"},
        },
    ),
    (
        Reshape1080,
        [((1, 2, 300, 64), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(2, 300, 64)"},
        },
    ),
    (
        Reshape1081,
        [((2, 4800, 300), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 2, 4800, 300)"},
        },
    ),
    (
        Reshape1082,
        [((1, 2, 4800, 300), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(2, 4800, 300)"},
        },
    ),
    (
        Reshape1083,
        [((1, 2, 64, 300), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(2, 64, 300)"},
        },
    ),
    (
        Reshape1084,
        [((2, 4800, 64), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 2, 4800, 64)"},
        },
    ),
    (
        Reshape1085,
        [((1, 4800, 2, 64), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(4800, 128)"},
        },
    ),
    (
        Reshape1086,
        [((4800, 128), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 4800, 128)"},
        },
    ),
    (
        Reshape1087,
        [((1, 512, 4800), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 512, 60, 80)"},
        },
    ),
    (
        Reshape1088,
        [((512, 1, 3, 3), torch.float32)],
        {
            "model_name": [
                "pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf",
                "pt_mobilenet_v1_basic_img_cls_torchvision",
                "pt_mobilenet_v1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
                "pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(512, 1, 3, 3)"},
        },
    ),
    (
        Reshape1089,
        [((1, 512, 60, 80), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 512, 4800, 1)"},
        },
    ),
    (
        Reshape1090,
        [((1, 320, 30, 40), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 320, 1200, 1)"},
        },
    ),
    (
        Reshape1091,
        [((1, 1200, 320), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1200, 5, 64)"},
        },
    ),
    (
        Reshape1092,
        [((1, 1200, 320), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 30, 40, 320)"},
        },
    ),
    (
        Reshape1093,
        [((1, 5, 1200, 64), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(5, 1200, 64)"},
        },
    ),
    (
        Reshape1094,
        [((1, 320, 1200), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 320, 30, 40)"},
        },
    ),
    (
        Reshape1095,
        [((1, 320, 15, 20), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 320, 300)"},
        },
    ),
    (
        Reshape1096,
        [((1, 300, 320), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(300, 320)"},
        },
    ),
    (
        Reshape1097,
        [((1, 300, 320), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 300, 5, 64)"},
        },
    ),
    (
        Reshape1098,
        [((300, 320), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 300, 320)"},
        },
    ),
    (
        Reshape1099,
        [((1, 5, 300, 64), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(5, 300, 64)"},
        },
    ),
    (
        Reshape1100,
        [((5, 1200, 300), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 5, 1200, 300)"},
        },
    ),
    (
        Reshape1101,
        [((1, 5, 1200, 300), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(5, 1200, 300)"},
        },
    ),
    (
        Reshape1102,
        [((1, 5, 64, 300), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(5, 64, 300)"},
        },
    ),
    (
        Reshape1103,
        [((5, 1200, 64), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 5, 1200, 64)"},
        },
    ),
    (
        Reshape1104,
        [((1, 1200, 5, 64), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1200, 320)"},
        },
    ),
    (
        Reshape1105,
        [((1200, 320), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1200, 320)"},
        },
    ),
    (
        Reshape1106,
        [((1, 1280, 1200), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1280, 30, 40)"},
        },
    ),
    (
        Reshape1107,
        [((1280, 1, 3, 3), torch.float32)],
        {
            "model_name": [
                "pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1280, 1, 3, 3)"},
        },
    ),
    (
        Reshape1108,
        [((1, 1280, 30, 40), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1280, 1200, 1)"},
        },
    ),
    (
        Reshape1109,
        [((1, 512, 15, 20), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 512, 300, 1)"},
        },
    ),
    (
        Reshape1110,
        [((1, 300, 512), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(300, 512)"},
        },
    ),
    (
        Reshape1111,
        [((1, 300, 512), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 300, 8, 64)"},
        },
    ),
    (
        Reshape1112,
        [((1, 300, 512), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 15, 20, 512)"},
        },
    ),
    (
        Reshape1113,
        [((300, 512), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 300, 512)"},
        },
    ),
    (
        Reshape1114,
        [((1, 8, 300, 64), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(8, 300, 64)"},
        },
    ),
    (
        Reshape1115,
        [((8, 300, 300), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 8, 300, 300)"},
        },
    ),
    (
        Reshape1116,
        [((1, 8, 300, 300), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(8, 300, 300)"},
        },
    ),
    (
        Reshape1117,
        [((1, 8, 64, 300), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(8, 64, 300)"},
        },
    ),
    (
        Reshape1118,
        [((8, 300, 64), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 8, 300, 64)"},
        },
    ),
    (
        Reshape1110,
        [((1, 300, 8, 64), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(300, 512)"},
        },
    ),
    (
        Reshape1119,
        [((1, 2048, 300), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 2048, 15, 20)"},
        },
    ),
    (
        Reshape1120,
        [((2048, 1, 3, 3), torch.float32)],
        {
            "model_name": [
                "pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(2048, 1, 3, 3)"},
        },
    ),
    (
        Reshape1121,
        [((1, 2048, 15, 20), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 2048, 300, 1)"},
        },
    ),
    (
        Reshape1122,
        [((1, 1, 30, 40), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 30, 40)"},
        },
    ),
    (
        Reshape1123,
        [((1, 1, 60, 80), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 60, 80)"},
        },
    ),
    (
        Reshape1124,
        [((1, 1, 120, 160), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 120, 160)"},
        },
    ),
    (
        Reshape1125,
        [((1, 1, 224, 224), torch.float32)],
        {"model_name": ["pt_googlenet_base_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(1, 224, 224)"}},
    ),
    (
        Reshape151,
        [((1, 1536, 1, 1), torch.float32)],
        {"model_name": ["pt_inception_v4_img_cls_osmr"], "pcc": 0.99, "op_params": {"shape": "(1, 1536)"}},
    ),
    (
        Reshape1126,
        [((1, 1536, 1, 1), torch.float32)],
        {
            "model_name": ["pt_inception_inception_v4_img_cls_timm", "pt_inception_inception_v4_tf_in1k_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1536, 1, 1)"},
        },
    ),
    (
        Reshape1127,
        [((1, 3, 256, 256), torch.float32)],
        {
            "model_name": ["pt_mlp_mixer_base_img_cls_github"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 3, 16, 16, 16, 16)"},
        },
    ),
    (
        Reshape530,
        [((1, 16, 16, 16, 16, 3), torch.float32)],
        {"model_name": ["pt_mlp_mixer_base_img_cls_github"], "pcc": 0.99, "op_params": {"shape": "(256, 768)"}},
    ),
    (
        Reshape1128,
        [((1, 256, 512), torch.float32)],
        {"model_name": ["pt_mlp_mixer_base_img_cls_github"], "pcc": 0.99, "op_params": {"shape": "(1, 256, 512, 1)"}},
    ),
    (
        Reshape591,
        [((1, 256, 512), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(256, 512)"},
        },
    ),
    (
        Reshape590,
        [((1, 256, 512), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 256, 8, 64)"},
        },
    ),
    (
        Reshape1129,
        [((1, 256, 512), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 16, 16, 512)"},
        },
    ),
    (
        Reshape592,
        [((1, 256, 512), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 256, 512)"},
        },
    ),
    (
        Reshape1130,
        [((1024, 256, 1), torch.float32)],
        {"model_name": ["pt_mlp_mixer_base_img_cls_github"], "pcc": 0.99, "op_params": {"shape": "(1024, 256, 1, 1)"}},
    ),
    (
        Reshape673,
        [((1, 1024, 512, 1), torch.float32)],
        {"model_name": ["pt_mlp_mixer_base_img_cls_github"], "pcc": 0.99, "op_params": {"shape": "(1, 1024, 512)"}},
    ),
    (
        Reshape1131,
        [((1, 1024, 512), torch.float32)],
        {"model_name": ["pt_mlp_mixer_base_img_cls_github"], "pcc": 0.99, "op_params": {"shape": "(1, 1024, 512, 1)"}},
    ),
    (
        Reshape1132,
        [((256, 1024, 1), torch.float32)],
        {"model_name": ["pt_mlp_mixer_base_img_cls_github"], "pcc": 0.99, "op_params": {"shape": "(256, 1024, 1, 1)"}},
    ),
    (
        Reshape592,
        [((1, 256, 512, 1), torch.float32)],
        {"model_name": ["pt_mlp_mixer_base_img_cls_github"], "pcc": 0.99, "op_params": {"shape": "(1, 256, 512)"}},
    ),
    (
        Reshape102,
        [((1, 64, 12, 12), torch.float32)],
        {"model_name": ["pt_mnist_base_img_cls_github"], "pcc": 0.99, "op_params": {"shape": "(1, 9216, 1, 1)"}},
    ),
    (
        Reshape1133,
        [((64, 1, 3, 3), torch.float32)],
        {
            "model_name": [
                "pt_mobilenet_v1_basic_img_cls_torchvision",
                "pt_mobilenet_v1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
                "pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
                "pt_xception_xception71_tf_in1k_img_cls_timm",
                "pt_xception_xception65_img_cls_timm",
                "pt_xception_xception_img_cls_timm",
                "pt_yolox_yolox_nano_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(64, 1, 3, 3)"},
        },
    ),
    (
        Reshape1134,
        [((128, 1, 3, 3), torch.float32)],
        {
            "model_name": [
                "pt_mobilenet_v1_basic_img_cls_torchvision",
                "pt_mobilenet_v1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
                "pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf",
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
                "pt_xception_xception71_tf_in1k_img_cls_timm",
                "pt_xception_xception65_img_cls_timm",
                "pt_xception_xception_img_cls_timm",
                "pt_yolox_yolox_nano_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(128, 1, 3, 3)"},
        },
    ),
    (
        Reshape1135,
        [((1024, 1, 3, 3), torch.float32)],
        {
            "model_name": [
                "pt_mobilenet_v1_basic_img_cls_torchvision",
                "pt_mobilenet_v1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
                "pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf",
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
                "pt_xception_xception71_tf_in1k_img_cls_timm",
                "pt_xception_xception65_img_cls_timm",
                "pt_xception_xception_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1024, 1, 3, 3)"},
        },
    ),
    (
        Reshape1136,
        [((384, 1, 3, 3), torch.float32)],
        {
            "model_name": [
                "pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf",
                "pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(384, 1, 3, 3)"},
        },
    ),
    (
        Reshape1137,
        [((768, 1, 3, 3), torch.float32)],
        {
            "model_name": ["pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(768, 1, 3, 3)"},
        },
    ),
    (
        Reshape1138,
        [((1, 768, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 768, 1, 1)"},
        },
    ),
    (
        Reshape1139,
        [((960, 1, 3, 3), torch.float32)],
        {
            "model_name": [
                "pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(960, 1, 3, 3)"},
        },
    ),
    (
        Reshape1140,
        [((432, 1, 3, 3), torch.float32)],
        {
            "model_name": ["pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(432, 1, 3, 3)"},
        },
    ),
    (
        Reshape1141,
        [((720, 1, 3, 3), torch.float32)],
        {
            "model_name": ["pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(720, 1, 3, 3)"},
        },
    ),
    (
        Reshape1142,
        [((72, 1, 3, 3), torch.float32)],
        {
            "model_name": [
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(72, 1, 3, 3)"},
        },
    ),
    (
        Reshape1143,
        [((88, 1, 3, 3), torch.float32)],
        {
            "model_name": [
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(88, 1, 3, 3)"},
        },
    ),
    (
        Reshape1144,
        [((96, 1, 5, 5), torch.float32)],
        {
            "model_name": [
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(96, 1, 5, 5)"},
        },
    ),
    (
        Reshape1145,
        [((120, 1, 5, 5), torch.float32)],
        {
            "model_name": [
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(120, 1, 5, 5)"},
        },
    ),
    (
        Reshape1146,
        [((1, 576, 1, 1), torch.float32)],
        {
            "model_name": ["pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 576, 1, 1)"},
        },
    ),
    (
        Reshape1147,
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
        Reshape1148,
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
        Reshape1149,
        [((1, 960, 1, 1), torch.float32)],
        {
            "model_name": ["pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 960, 1, 1)"},
        },
    ),
    (
        Reshape1150,
        [((1, 1088, 1, 1), torch.float32)],
        {
            "model_name": ["pt_regnet_facebook_regnet_y_040_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1088, 1, 1)"},
        },
    ),
    (
        Reshape1151,
        [((1, 7392, 1, 1), torch.float32)],
        {
            "model_name": ["pt_regnet_regnet_y_128gf_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 7392, 1, 1)"},
        },
    ),
    (
        Reshape1152,
        [((1, 888, 1, 1), torch.float32)],
        {
            "model_name": ["pt_regnet_regnet_y_1_6gf_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 888, 1, 1)"},
        },
    ),
    (
        Reshape1153,
        [((1, 3712, 1, 1), torch.float32)],
        {
            "model_name": ["pt_regnet_regnet_y_32gf_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 3712, 1, 1)"},
        },
    ),
    (
        Reshape1154,
        [((1, 440, 1, 1), torch.float32)],
        {
            "model_name": ["pt_regnet_regnet_y_400mf_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 440, 1, 1)"},
        },
    ),
    (
        Reshape1155,
        [((1, 2520, 1, 1), torch.float32)],
        {
            "model_name": ["pt_regnet_regnet_x_32gf_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 2520, 1, 1)"},
        },
    ),
    (
        Reshape1156,
        [((1, 1008, 1, 1), torch.float32)],
        {
            "model_name": ["pt_regnet_regnet_x_3_2gf_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1008, 1, 1)"},
        },
    ),
    (
        Reshape1157,
        [((1, 912, 1, 1), torch.float32)],
        {
            "model_name": ["pt_regnet_regnet_x_1_6gf_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 912, 1, 1)"},
        },
    ),
    (
        Reshape1158,
        [((1, 672, 1, 1), torch.float32)],
        {
            "model_name": ["pt_regnet_regnet_x_800mf_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 672, 1, 1)"},
        },
    ),
    (
        Reshape1159,
        [((1, 2016, 1, 1), torch.float32)],
        {
            "model_name": ["pt_regnet_regnet_y_8gf_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 2016, 1, 1)"},
        },
    ),
    (
        Reshape1160,
        [((1, 784, 1, 1), torch.float32)],
        {
            "model_name": ["pt_regnet_regnet_y_800mf_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 784, 1, 1)"},
        },
    ),
    (
        Reshape1161,
        [((1, 1512, 1, 1), torch.float32)],
        {
            "model_name": ["pt_regnet_regnet_y_3_2gf_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1512, 1, 1)"},
        },
    ),
    (
        Reshape1162,
        [((1, 400, 1, 1), torch.float32)],
        {
            "model_name": ["pt_regnet_regnet_x_400mf_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 400, 1, 1)"},
        },
    ),
    (
        Reshape1163,
        [((1, 3024, 1, 1), torch.float32)],
        {
            "model_name": ["pt_regnet_regnet_y_16gf_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 3024, 1, 1)"},
        },
    ),
    (
        Reshape1164,
        [((1, 16384, 64), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 16384, 1, 64)"},
        },
    ),
    (
        Reshape1165,
        [((1, 16384, 64), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 128, 128, 64)"},
        },
    ),
    (
        Reshape387,
        [((1, 64, 16384), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 64, 128, 128)"},
        },
    ),
    (
        Reshape1166,
        [((1, 64, 16, 16), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 64, 256)"},
        },
    ),
    (
        Reshape1167,
        [((1, 256, 64), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(256, 64)"},
        },
    ),
    (
        Reshape1168,
        [((1, 256, 64), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 256, 1, 64)"},
        },
    ),
    (
        Reshape1169,
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
        Reshape1170,
        [((256, 64), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 256, 64)"},
        },
    ),
    (
        Reshape1171,
        [((1, 16384, 256), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1, 16384, 256)"},
        },
    ),
    (
        Reshape1172,
        [((1, 1, 16384, 256), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 16384, 256)"},
        },
    ),
    (
        Reshape1166,
        [((1, 1, 64, 256), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 64, 256)"},
        },
    ),
    (
        Reshape1173,
        [((1, 256, 16384), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 256, 128, 128)"},
        },
    ),
    (
        Reshape1174,
        [((1, 256, 128, 128), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 256, 16384, 1)"},
        },
    ),
    (
        Reshape1175,
        [((1, 4096, 128), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 4096, 2, 64)"},
        },
    ),
    (
        Reshape1176,
        [((1, 4096, 128), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 64, 64, 128)"},
        },
    ),
    (
        Reshape1177,
        [((1, 2, 4096, 64), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(2, 4096, 64)"},
        },
    ),
    (
        Reshape1178,
        [((1, 128, 16, 16), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 128, 256)"},
        },
    ),
    (
        Reshape1179,
        [((1, 256, 128), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(256, 128)"},
        },
    ),
    (
        Reshape1180,
        [((1, 256, 128), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 256, 2, 64)"},
        },
    ),
    (
        Reshape1181,
        [((256, 128), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 256, 128)"},
        },
    ),
    (
        Reshape1182,
        [((1, 2, 256, 64), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(2, 256, 64)"},
        },
    ),
    (
        Reshape1183,
        [((2, 4096, 256), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 2, 4096, 256)"},
        },
    ),
    (
        Reshape1184,
        [((1, 2, 4096, 256), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(2, 4096, 256)"},
        },
    ),
    (
        Reshape1185,
        [((1, 2, 64, 256), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(2, 64, 256)"},
        },
    ),
    (
        Reshape1186,
        [((2, 4096, 64), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 2, 4096, 64)"},
        },
    ),
    (
        Reshape1187,
        [((1, 4096, 2, 64), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(4096, 128)"},
        },
    ),
    (
        Reshape1188,
        [((4096, 128), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 4096, 128)"},
        },
    ),
    (
        Reshape1189,
        [((1, 512, 4096), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 512, 64, 64)"},
        },
    ),
    (
        Reshape1190,
        [((1, 512, 64, 64), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 512, 4096, 1)"},
        },
    ),
    (
        Reshape1191,
        [((1, 320, 32, 32), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 320, 1024, 1)"},
        },
    ),
    (
        Reshape1192,
        [((1, 1024, 320), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1024, 5, 64)"},
        },
    ),
    (
        Reshape1193,
        [((1, 1024, 320), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 32, 32, 320)"},
        },
    ),
    (
        Reshape1194,
        [((1, 5, 1024, 64), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(5, 1024, 64)"},
        },
    ),
    (
        Reshape1195,
        [((1, 320, 1024), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 320, 32, 32)"},
        },
    ),
    (
        Reshape1196,
        [((1, 320, 16, 16), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 320, 256)"},
        },
    ),
    (
        Reshape1197,
        [((1, 256, 320), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(256, 320)"},
        },
    ),
    (
        Reshape1198,
        [((1, 256, 320), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 256, 5, 64)"},
        },
    ),
    (
        Reshape1199,
        [((256, 320), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 256, 320)"},
        },
    ),
    (
        Reshape1200,
        [((1, 5, 256, 64), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(5, 256, 64)"},
        },
    ),
    (
        Reshape1201,
        [((5, 1024, 256), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 5, 1024, 256)"},
        },
    ),
    (
        Reshape1202,
        [((1, 5, 1024, 256), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(5, 1024, 256)"},
        },
    ),
    (
        Reshape1203,
        [((1, 5, 64, 256), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(5, 64, 256)"},
        },
    ),
    (
        Reshape1204,
        [((5, 1024, 64), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 5, 1024, 64)"},
        },
    ),
    (
        Reshape1205,
        [((1, 1024, 5, 64), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1024, 320)"},
        },
    ),
    (
        Reshape1206,
        [((1024, 320), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1024, 320)"},
        },
    ),
    (
        Reshape1207,
        [((1, 1280, 1024), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1280, 32, 32)"},
        },
    ),
    (
        Reshape1208,
        [((1, 1280, 32, 32), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1280, 1024, 1)"},
        },
    ),
    (
        Reshape1209,
        [((1, 512, 16, 16), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 512, 256, 1)"},
        },
    ),
    (
        Reshape1210,
        [((1, 8, 256, 64), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(8, 256, 64)"},
        },
    ),
    (
        Reshape1211,
        [((1, 8, 64, 256), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(8, 64, 256)"},
        },
    ),
    (
        Reshape1212,
        [((8, 256, 64), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 8, 256, 64)"},
        },
    ),
    (
        Reshape591,
        [((1, 256, 8, 64), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(256, 512)"},
        },
    ),
    (
        Reshape1213,
        [((1, 2048, 16, 16), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 2048, 256, 1)"},
        },
    ),
    (
        Reshape1214,
        [((1, 768, 256), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 768, 16, 16)"},
        },
    ),
    (
        Reshape1215,
        [((1, 768, 1024), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 768, 32, 32)"},
        },
    ),
    (
        Reshape1216,
        [((1, 768, 4096), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 768, 64, 64)"},
        },
    ),
    (
        Reshape1217,
        [((1, 768, 16384), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 768, 128, 128)"},
        },
    ),
    (
        Reshape1218,
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
        Reshape1219,
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
        Reshape645,
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
        Reshape1220,
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
        Reshape1221,
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
        Reshape1222,
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
        Reshape1223,
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
        Reshape1220,
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
        Reshape1224,
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
        Reshape1225,
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
        Reshape1226,
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
        Reshape1227,
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
        Reshape1228,
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
        Reshape1229,
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
        Reshape1228,
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
        Reshape1230,
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
        Reshape1231,
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
        Reshape1232,
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
        Reshape1233,
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
        Reshape1234,
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
        Reshape1235,
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
        Reshape1236,
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
        Reshape1237,
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
        Reshape1238,
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
        Reshape1239,
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
        Reshape1240,
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
        Reshape1241,
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
        Reshape1242,
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
        Reshape1243,
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
        Reshape1244,
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
        Reshape1245,
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
        Reshape1246,
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
        Reshape1247,
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
        Reshape1248,
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
        Reshape1249,
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
        Reshape1250,
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
        Reshape1251,
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
        Reshape1252,
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
        Reshape1253,
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
        Reshape1254,
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
        Reshape1255,
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
        Reshape694,
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
        Reshape1256,
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
        Reshape1257,
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
        Reshape1258,
        [((1, 16, 38, 38), torch.float32)],
        {
            "model_name": ["pt_ssd300_resnet50_base_img_cls_torchhub"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 4, 5776)"},
        },
    ),
    (
        Reshape1259,
        [((1, 24, 19, 19), torch.float32)],
        {
            "model_name": ["pt_ssd300_resnet50_base_img_cls_torchhub"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 4, 2166)"},
        },
    ),
    (
        Reshape1260,
        [((1, 24, 10, 10), torch.float32)],
        {
            "model_name": ["pt_ssd300_resnet50_base_img_cls_torchhub"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 4, 600)"},
        },
    ),
    (
        Reshape1261,
        [((1, 24, 5, 5), torch.float32)],
        {
            "model_name": ["pt_ssd300_resnet50_base_img_cls_torchhub"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 4, 150)"},
        },
    ),
    (
        Reshape1262,
        [((1, 16, 3, 3), torch.float32)],
        {"model_name": ["pt_ssd300_resnet50_base_img_cls_torchhub"], "pcc": 0.99, "op_params": {"shape": "(1, 4, 36)"}},
    ),
    (
        Reshape1263,
        [((1, 324, 38, 38), torch.float32)],
        {
            "model_name": ["pt_ssd300_resnet50_base_img_cls_torchhub"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 81, 5776)"},
        },
    ),
    (
        Reshape1264,
        [((1, 486, 19, 19), torch.float32)],
        {
            "model_name": ["pt_ssd300_resnet50_base_img_cls_torchhub"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 81, 2166)"},
        },
    ),
    (
        Reshape1265,
        [((1, 486, 10, 10), torch.float32)],
        {
            "model_name": ["pt_ssd300_resnet50_base_img_cls_torchhub"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 81, 600)"},
        },
    ),
    (
        Reshape1266,
        [((1, 486, 5, 5), torch.float32)],
        {
            "model_name": ["pt_ssd300_resnet50_base_img_cls_torchhub"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 81, 150)"},
        },
    ),
    (
        Reshape1267,
        [((1, 324, 3, 3), torch.float32)],
        {
            "model_name": ["pt_ssd300_resnet50_base_img_cls_torchhub"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 81, 36)"},
        },
    ),
    (
        Reshape1268,
        [((1, 324, 1, 1), torch.float32)],
        {"model_name": ["pt_ssd300_resnet50_base_img_cls_torchhub"], "pcc": 0.99, "op_params": {"shape": "(1, 81, 4)"}},
    ),
    (
        Reshape1269,
        [((1, 56, 56, 96), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 8, 7, 8, 7, 96)"},
        },
    ),
    (
        Reshape1270,
        [((1, 56, 56, 96), torch.float32)],
        {
            "model_name": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(3136, 96)"},
        },
    ),
    (
        Reshape1271,
        [((1, 56, 56, 96), torch.float32)],
        {
            "model_name": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 3136, 96)"},
        },
    ),
    (
        Reshape1270,
        [((1, 8, 8, 7, 7, 96), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(3136, 96)"},
        },
    ),
    (
        Reshape1272,
        [((3136, 288), torch.float32)],
        {
            "model_name": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(64, 49, 288)"},
        },
    ),
    (
        Reshape1273,
        [((64, 49, 288), torch.float32)],
        {
            "model_name": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(64, 49, 3, 3, 32)"},
        },
    ),
    (
        Reshape1274,
        [((1, 64, 3, 49, 32), torch.float32)],
        {
            "model_name": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(64, 3, 49, 32)"},
        },
    ),
    (
        Reshape1275,
        [((1, 64, 3, 49, 32), torch.float32)],
        {
            "model_name": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(192, 49, 32)"},
        },
    ),
    (
        Reshape1275,
        [((64, 3, 49, 32), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(192, 49, 32)"},
        },
    ),
    (
        Reshape1276,
        [((192, 49, 49), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(64, 3, 49, 49)"},
        },
    ),
    (
        Reshape1277,
        [((2401, 3), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(49, 49, 3)"},
        },
    ),
    (
        Reshape1278,
        [((64, 3, 49, 49), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(192, 49, 49)"},
        },
    ),
    (
        Reshape1279,
        [((64, 3, 49, 49), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 64, 3, 49, 49)"},
        },
    ),
    (
        Reshape1280,
        [((64, 3, 32, 49), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(192, 32, 49)"},
        },
    ),
    (
        Reshape1274,
        [((192, 49, 32), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(64, 3, 49, 32)"},
        },
    ),
    (
        Reshape1270,
        [((64, 49, 3, 32), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(3136, 96)"},
        },
    ),
    (
        Reshape1281,
        [((3136, 96), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(64, 49, 96)"},
        },
    ),
    (
        Reshape1282,
        [((3136, 96), torch.float32)],
        {
            "model_name": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 56, 56, 96)"},
        },
    ),
    (
        Reshape1283,
        [((64, 49, 96), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 8, 8, 7, 7, 96)"},
        },
    ),
    (
        Reshape1284,
        [((64, 49, 96), torch.float32)],
        {
            "model_name": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(64, 49, 3, 32)"},
        },
    ),
    (
        Reshape1282,
        [((1, 8, 7, 8, 7, 96), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 56, 56, 96)"},
        },
    ),
    (
        Reshape1271,
        [((1, 8, 7, 8, 7, 96), torch.float32)],
        {
            "model_name": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 3136, 96)"},
        },
    ),
    (
        Reshape1285,
        [((3136, 384), torch.float32)],
        {
            "model_name": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 56, 56, 384)"},
        },
    ),
    (
        Reshape1286,
        [((3136, 384), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(64, 49, 384)"}},
    ),
    (
        Reshape1287,
        [((1, 56, 56, 384), torch.float32)],
        {
            "model_name": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(3136, 384)"},
        },
    ),
    (
        Reshape1276,
        [((1, 64, 3, 49, 49), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(64, 3, 49, 49)"},
        },
    ),
    (
        Reshape1288,
        [((1, 28, 28, 384), torch.float32)],
        {
            "model_name": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(784, 384)"},
        },
    ),
    (
        Reshape1289,
        [((1, 28, 28, 384), torch.float32)],
        {
            "model_name": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 784, 384)"},
        },
    ),
    (
        Reshape1290,
        [((784, 192), torch.float32)],
        {
            "model_name": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 28, 28, 192)"},
        },
    ),
    (
        Reshape1291,
        [((784, 192), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(16, 49, 192)"},
        },
    ),
    (
        Reshape1292,
        [((1, 28, 28, 192), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 4, 7, 4, 7, 192)"},
        },
    ),
    (
        Reshape1293,
        [((1, 28, 28, 192), torch.float32)],
        {
            "model_name": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(784, 192)"},
        },
    ),
    (
        Reshape1294,
        [((1, 28, 28, 192), torch.float32)],
        {
            "model_name": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 784, 192)"},
        },
    ),
    (
        Reshape1293,
        [((1, 4, 4, 7, 7, 192), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(784, 192)"},
        },
    ),
    (
        Reshape1295,
        [((784, 576), torch.float32)],
        {
            "model_name": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(16, 49, 576)"},
        },
    ),
    (
        Reshape1296,
        [((16, 49, 576), torch.float32)],
        {
            "model_name": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(16, 49, 3, 6, 32)"},
        },
    ),
    (
        Reshape1297,
        [((1, 16, 6, 49, 32), torch.float32)],
        {
            "model_name": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(16, 6, 49, 32)"},
        },
    ),
    (
        Reshape1298,
        [((1, 16, 6, 49, 32), torch.float32)],
        {
            "model_name": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(96, 49, 32)"},
        },
    ),
    (
        Reshape1298,
        [((16, 6, 49, 32), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(96, 49, 32)"},
        },
    ),
    (
        Reshape1299,
        [((96, 49, 49), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(16, 6, 49, 49)"},
        },
    ),
    (
        Reshape1300,
        [((2401, 6), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(49, 49, 6)"},
        },
    ),
    (
        Reshape1301,
        [((16, 6, 49, 49), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(96, 49, 49)"},
        },
    ),
    (
        Reshape1302,
        [((16, 6, 49, 49), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 16, 6, 49, 49)"},
        },
    ),
    (
        Reshape1303,
        [((16, 6, 32, 49), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(96, 32, 49)"},
        },
    ),
    (
        Reshape1297,
        [((96, 49, 32), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(16, 6, 49, 32)"},
        },
    ),
    (
        Reshape1293,
        [((16, 49, 6, 32), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(784, 192)"},
        },
    ),
    (
        Reshape1304,
        [((16, 49, 192), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 4, 4, 7, 7, 192)"},
        },
    ),
    (
        Reshape1305,
        [((16, 49, 192), torch.float32)],
        {
            "model_name": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(16, 49, 6, 32)"},
        },
    ),
    (
        Reshape1290,
        [((1, 4, 7, 4, 7, 192), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 28, 28, 192)"},
        },
    ),
    (
        Reshape1294,
        [((1, 4, 7, 4, 7, 192), torch.float32)],
        {
            "model_name": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 784, 192)"},
        },
    ),
    (
        Reshape1306,
        [((784, 768), torch.float32)],
        {
            "model_name": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 28, 28, 768)"},
        },
    ),
    (
        Reshape1307,
        [((784, 768), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(16, 49, 768)"}},
    ),
    (
        Reshape1308,
        [((1, 28, 28, 768), torch.float32)],
        {
            "model_name": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(784, 768)"},
        },
    ),
    (
        Reshape1299,
        [((1, 16, 6, 49, 49), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(16, 6, 49, 49)"},
        },
    ),
    (
        Reshape1309,
        [((1, 14, 14, 768), torch.float32)],
        {
            "model_name": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(196, 768)"},
        },
    ),
    (
        Reshape1310,
        [((1, 14, 14, 768), torch.float32)],
        {
            "model_name": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 196, 768)"},
        },
    ),
    (
        Reshape1311,
        [((196, 384), torch.float32)],
        {
            "model_name": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 14, 14, 384)"},
        },
    ),
    (
        Reshape1312,
        [((196, 384), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(4, 49, 384)"},
        },
    ),
    (
        Reshape1313,
        [((1, 14, 14, 384), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 2, 7, 2, 7, 384)"},
        },
    ),
    (
        Reshape1314,
        [((1, 14, 14, 384), torch.float32)],
        {
            "model_name": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(196, 384)"},
        },
    ),
    (
        Reshape1315,
        [((1, 14, 14, 384), torch.float32)],
        {
            "model_name": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 196, 384)"},
        },
    ),
    (
        Reshape1314,
        [((1, 2, 2, 7, 7, 384), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(196, 384)"},
        },
    ),
    (
        Reshape1316,
        [((196, 1152), torch.float32)],
        {
            "model_name": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(4, 49, 1152)"},
        },
    ),
    (
        Reshape1317,
        [((4, 49, 1152), torch.float32)],
        {
            "model_name": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(4, 49, 3, 12, 32)"},
        },
    ),
    (
        Reshape1318,
        [((1, 4, 12, 49, 32), torch.float32)],
        {
            "model_name": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(4, 12, 49, 32)"},
        },
    ),
    (
        Reshape1319,
        [((1, 4, 12, 49, 32), torch.float32)],
        {
            "model_name": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(48, 49, 32)"},
        },
    ),
    (
        Reshape1319,
        [((4, 12, 49, 32), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(48, 49, 32)"},
        },
    ),
    (
        Reshape1320,
        [((48, 49, 49), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(4, 12, 49, 49)"},
        },
    ),
    (
        Reshape1321,
        [((2401, 12), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(49, 49, 12)"},
        },
    ),
    (
        Reshape1322,
        [((4, 12, 49, 49), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(48, 49, 49)"},
        },
    ),
    (
        Reshape1323,
        [((4, 12, 49, 49), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 4, 12, 49, 49)"},
        },
    ),
    (
        Reshape1324,
        [((4, 12, 32, 49), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(48, 32, 49)"},
        },
    ),
    (
        Reshape1318,
        [((48, 49, 32), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(4, 12, 49, 32)"},
        },
    ),
    (
        Reshape1314,
        [((4, 49, 12, 32), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(196, 384)"},
        },
    ),
    (
        Reshape1325,
        [((4, 49, 384), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 2, 2, 7, 7, 384)"},
        },
    ),
    (
        Reshape1326,
        [((4, 49, 384), torch.float32)],
        {
            "model_name": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(4, 49, 12, 32)"},
        },
    ),
    (
        Reshape1311,
        [((1, 2, 7, 2, 7, 384), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 14, 14, 384)"},
        },
    ),
    (
        Reshape1315,
        [((1, 2, 7, 2, 7, 384), torch.float32)],
        {
            "model_name": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 196, 384)"},
        },
    ),
    (
        Reshape1327,
        [((196, 1536), torch.float32)],
        {
            "model_name": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 14, 14, 1536)"},
        },
    ),
    (
        Reshape1328,
        [((196, 1536), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(4, 49, 1536)"}},
    ),
    (
        Reshape1329,
        [((1, 14, 14, 1536), torch.float32)],
        {
            "model_name": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(196, 1536)"},
        },
    ),
    (
        Reshape1320,
        [((1, 4, 12, 49, 49), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(4, 12, 49, 49)"},
        },
    ),
    (
        Reshape1330,
        [((1, 7, 7, 1536), torch.float32)],
        {
            "model_name": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(49, 1536)"},
        },
    ),
    (
        Reshape1331,
        [((1, 7, 7, 1536), torch.float32)],
        {
            "model_name": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 49, 1536)"},
        },
    ),
    (
        Reshape1332,
        [((49, 768), torch.float32)],
        {
            "model_name": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 7, 7, 768)"},
        },
    ),
    (
        Reshape1333,
        [((49, 768), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 49, 768)"},
        },
    ),
    (
        Reshape1334,
        [((1, 7, 7, 768), torch.float32)],
        {
            "model_name": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1, 7, 1, 7, 768)"},
        },
    ),
    (
        Reshape1335,
        [((1, 7, 7, 768), torch.float32)],
        {
            "model_name": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(49, 768)"},
        },
    ),
    (
        Reshape1335,
        [((1, 1, 1, 7, 7, 768), torch.float32)],
        {
            "model_name": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(49, 768)"},
        },
    ),
    (
        Reshape1336,
        [((49, 2304), torch.float32)],
        {
            "model_name": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 49, 2304)"},
        },
    ),
    (
        Reshape1337,
        [((1, 49, 2304), torch.float32)],
        {
            "model_name": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 49, 3, 24, 32)"},
        },
    ),
    (
        Reshape1338,
        [((1, 1, 24, 49, 32), torch.float32)],
        {
            "model_name": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 24, 49, 32)"},
        },
    ),
    (
        Reshape1339,
        [((1, 1, 24, 49, 32), torch.float32)],
        {
            "model_name": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(24, 49, 32)"},
        },
    ),
    (
        Reshape1339,
        [((1, 24, 49, 32), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(24, 49, 32)"},
        },
    ),
    (
        Reshape1340,
        [((24, 49, 49), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 24, 49, 49)"},
        },
    ),
    (
        Reshape1341,
        [((2401, 24), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(49, 49, 24)"},
        },
    ),
    (
        Reshape1342,
        [((1, 24, 49, 49), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(24, 49, 49)"},
        },
    ),
    (
        Reshape1343,
        [((1, 24, 32, 49), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(24, 32, 49)"},
        },
    ),
    (
        Reshape1338,
        [((24, 49, 32), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 24, 49, 32)"},
        },
    ),
    (
        Reshape1335,
        [((1, 49, 24, 32), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(49, 768)"},
        },
    ),
    (
        Reshape1344,
        [((1, 49, 768), torch.float32)],
        {
            "model_name": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1, 1, 7, 7, 768)"},
        },
    ),
    (
        Reshape1335,
        [((1, 49, 768), torch.float32)],
        {
            "model_name": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(49, 768)"},
        },
    ),
    (
        Reshape1345,
        [((1, 49, 768), torch.float32)],
        {
            "model_name": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 49, 24, 32)"},
        },
    ),
    (
        Reshape1333,
        [((1, 49, 768), torch.float32)],
        {
            "model_name": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 49, 768)"},
        },
    ),
    (
        Reshape1332,
        [((1, 1, 7, 1, 7, 768), torch.float32)],
        {
            "model_name": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 7, 7, 768)"},
        },
    ),
    (
        Reshape1346,
        [((49, 3072), torch.float32)],
        {
            "model_name": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 7, 7, 3072)"},
        },
    ),
    (
        Reshape1347,
        [((49, 3072), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(1, 49, 3072)"}},
    ),
    (
        Reshape1348,
        [((1, 7, 7, 3072), torch.float32)],
        {
            "model_name": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(49, 3072)"},
        },
    ),
    (
        Reshape1349,
        [((1, 56, 56, 128), torch.float32)],
        {
            "model_name": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 8, 7, 8, 7, 128)"},
        },
    ),
    (
        Reshape1350,
        [((1, 56, 56, 128), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(3136, 128)"}},
    ),
    (
        Reshape1350,
        [((1, 8, 8, 7, 7, 128), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(3136, 128)"}},
    ),
    (
        Reshape1351,
        [((64, 49, 384), torch.float32)],
        {
            "model_name": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(64, 49, 3, 4, 32)"},
        },
    ),
    (
        Reshape1352,
        [((1, 64, 4, 49, 32), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(64, 4, 49, 32)"}},
    ),
    (
        Reshape1353,
        [((1, 64, 4, 49, 32), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(256, 49, 32)"}},
    ),
    (
        Reshape1353,
        [((64, 4, 49, 32), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(256, 49, 32)"}},
    ),
    (
        Reshape1354,
        [((256, 49, 49), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(64, 4, 49, 49)"}},
    ),
    (
        Reshape1355,
        [((2401, 4), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(49, 49, 4)"}},
    ),
    (
        Reshape1356,
        [((64, 4, 49, 49), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(256, 49, 49)"}},
    ),
    (
        Reshape1357,
        [((64, 4, 49, 49), torch.float32)],
        {
            "model_name": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 64, 4, 49, 49)"},
        },
    ),
    (
        Reshape1358,
        [((64, 4, 32, 49), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(256, 32, 49)"}},
    ),
    (
        Reshape1352,
        [((256, 49, 32), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(64, 4, 49, 32)"}},
    ),
    (
        Reshape1350,
        [((64, 49, 4, 32), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(3136, 128)"}},
    ),
    (
        Reshape1359,
        [((3136, 128), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(64, 49, 128)"}},
    ),
    (
        Reshape1360,
        [((3136, 128), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(1, 56, 56, 128)"}},
    ),
    (
        Reshape1361,
        [((64, 49, 128), torch.float32)],
        {
            "model_name": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 8, 8, 7, 7, 128)"},
        },
    ),
    (
        Reshape1360,
        [((1, 8, 7, 8, 7, 128), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(1, 56, 56, 128)"}},
    ),
    (
        Reshape1362,
        [((3136, 512), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(1, 56, 56, 512)"}},
    ),
    (
        Reshape1363,
        [((1, 56, 56, 512), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(3136, 512)"}},
    ),
    (
        Reshape1354,
        [((1, 64, 4, 49, 49), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(64, 4, 49, 49)"}},
    ),
    (
        Reshape1364,
        [((1, 28, 28, 512), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(784, 512)"}},
    ),
    (
        Reshape1365,
        [((784, 256), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(1, 28, 28, 256)"}},
    ),
    (
        Reshape1366,
        [((784, 256), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(16, 49, 256)"}},
    ),
    (
        Reshape1367,
        [((1, 28, 28, 256), torch.float32)],
        {
            "model_name": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 4, 7, 4, 7, 256)"},
        },
    ),
    (
        Reshape1368,
        [((1, 28, 28, 256), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(784, 256)"}},
    ),
    (
        Reshape1368,
        [((1, 4, 4, 7, 7, 256), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(784, 256)"}},
    ),
    (
        Reshape1369,
        [((16, 49, 768), torch.float32)],
        {
            "model_name": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(16, 49, 3, 8, 32)"},
        },
    ),
    (
        Reshape1370,
        [((1, 16, 8, 49, 32), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(16, 8, 49, 32)"}},
    ),
    (
        Reshape1371,
        [((1, 16, 8, 49, 32), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(128, 49, 32)"}},
    ),
    (
        Reshape1371,
        [((16, 8, 49, 32), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(128, 49, 32)"}},
    ),
    (
        Reshape1372,
        [((128, 49, 49), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(16, 8, 49, 49)"}},
    ),
    (
        Reshape1373,
        [((2401, 8), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(49, 49, 8)"}},
    ),
    (
        Reshape1374,
        [((16, 8, 49, 49), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(128, 49, 49)"}},
    ),
    (
        Reshape1375,
        [((16, 8, 49, 49), torch.float32)],
        {
            "model_name": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 16, 8, 49, 49)"},
        },
    ),
    (
        Reshape1376,
        [((16, 8, 32, 49), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(128, 32, 49)"}},
    ),
    (
        Reshape1370,
        [((128, 49, 32), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(16, 8, 49, 32)"}},
    ),
    (
        Reshape1368,
        [((16, 49, 8, 32), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(784, 256)"}},
    ),
    (
        Reshape1377,
        [((16, 49, 256), torch.float32)],
        {
            "model_name": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 4, 4, 7, 7, 256)"},
        },
    ),
    (
        Reshape1365,
        [((1, 4, 7, 4, 7, 256), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(1, 28, 28, 256)"}},
    ),
    (
        Reshape1378,
        [((784, 1024), torch.float32)],
        {
            "model_name": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 28, 28, 1024)"},
        },
    ),
    (
        Reshape1379,
        [((1, 28, 28, 1024), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(784, 1024)"}},
    ),
    (
        Reshape1372,
        [((1, 16, 8, 49, 49), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(16, 8, 49, 49)"}},
    ),
    (
        Reshape1380,
        [((1, 14, 14, 1024), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(196, 1024)"}},
    ),
    (
        Reshape1381,
        [((196, 512), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(1, 14, 14, 512)"}},
    ),
    (
        Reshape1382,
        [((196, 512), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(4, 49, 512)"}},
    ),
    (
        Reshape1383,
        [((1, 14, 14, 512), torch.float32)],
        {
            "model_name": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 2, 7, 2, 7, 512)"},
        },
    ),
    (
        Reshape1384,
        [((1, 14, 14, 512), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(196, 512)"}},
    ),
    (
        Reshape1384,
        [((1, 2, 2, 7, 7, 512), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(196, 512)"}},
    ),
    (
        Reshape1385,
        [((4, 49, 1536), torch.float32)],
        {
            "model_name": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(4, 49, 3, 16, 32)"},
        },
    ),
    (
        Reshape1386,
        [((1, 4, 16, 49, 32), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(4, 16, 49, 32)"}},
    ),
    (
        Reshape1387,
        [((1, 4, 16, 49, 32), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(64, 49, 32)"}},
    ),
    (
        Reshape1387,
        [((4, 16, 49, 32), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(64, 49, 32)"}},
    ),
    (
        Reshape1388,
        [((64, 49, 49), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(4, 16, 49, 49)"}},
    ),
    (
        Reshape1389,
        [((2401, 16), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(49, 49, 16)"}},
    ),
    (
        Reshape1390,
        [((4, 16, 49, 49), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(64, 49, 49)"}},
    ),
    (
        Reshape1391,
        [((4, 16, 49, 49), torch.float32)],
        {
            "model_name": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 4, 16, 49, 49)"},
        },
    ),
    (
        Reshape1392,
        [((4, 16, 32, 49), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(64, 32, 49)"}},
    ),
    (
        Reshape1386,
        [((64, 49, 32), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(4, 16, 49, 32)"}},
    ),
    (
        Reshape1384,
        [((4, 49, 16, 32), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(196, 512)"}},
    ),
    (
        Reshape1393,
        [((4, 49, 512), torch.float32)],
        {
            "model_name": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 2, 2, 7, 7, 512)"},
        },
    ),
    (
        Reshape1381,
        [((1, 2, 7, 2, 7, 512), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(1, 14, 14, 512)"}},
    ),
    (
        Reshape1394,
        [((196, 2048), torch.float32)],
        {
            "model_name": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 14, 14, 2048)"},
        },
    ),
    (
        Reshape1395,
        [((1, 14, 14, 2048), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(196, 2048)"}},
    ),
    (
        Reshape1388,
        [((1, 4, 16, 49, 49), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(4, 16, 49, 49)"}},
    ),
    (
        Reshape1396,
        [((1, 7, 7, 2048), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(49, 2048)"}},
    ),
    (
        Reshape1397,
        [((49, 1024), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(1, 7, 7, 1024)"}},
    ),
    (
        Reshape1398,
        [((49, 1024), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(1, 49, 1024)"}},
    ),
    (
        Reshape1399,
        [((1, 7, 7, 1024), torch.float32)],
        {
            "model_name": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1, 7, 1, 7, 1024)"},
        },
    ),
    (
        Reshape1400,
        [((1, 7, 7, 1024), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(49, 1024)"}},
    ),
    (
        Reshape1400,
        [((1, 1, 1, 7, 7, 1024), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(49, 1024)"}},
    ),
    (
        Reshape1401,
        [((1, 49, 3072), torch.float32)],
        {
            "model_name": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 49, 3, 32, 32)"},
        },
    ),
    (
        Reshape1402,
        [((1, 1, 32, 49, 32), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(1, 32, 49, 32)"}},
    ),
    (
        Reshape1403,
        [((1, 1, 32, 49, 32), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(32, 49, 32)"}},
    ),
    (
        Reshape1403,
        [((1, 32, 49, 32), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(32, 49, 32)"}},
    ),
    (
        Reshape1404,
        [((32, 49, 49), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(1, 32, 49, 49)"}},
    ),
    (
        Reshape1405,
        [((2401, 32), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(49, 49, 32)"}},
    ),
    (
        Reshape1406,
        [((1, 32, 49, 49), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(32, 49, 49)"}},
    ),
    (
        Reshape1407,
        [((1, 32, 32, 49), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(32, 32, 49)"}},
    ),
    (
        Reshape1402,
        [((32, 49, 32), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(1, 32, 49, 32)"}},
    ),
    (
        Reshape1400,
        [((1, 49, 32, 32), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(49, 1024)"}},
    ),
    (
        Reshape1408,
        [((1, 49, 1024), torch.float32)],
        {
            "model_name": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1, 1, 7, 7, 1024)"},
        },
    ),
    (
        Reshape1397,
        [((1, 1, 7, 1, 7, 1024), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(1, 7, 7, 1024)"}},
    ),
    (
        Reshape1409,
        [((49, 4096), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(1, 7, 7, 4096)"}},
    ),
    (
        Reshape1410,
        [((1, 7, 7, 4096), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "op_params": {"shape": "(49, 4096)"}},
    ),
    (
        Reshape1411,
        [((1, 96, 56, 56), torch.float32)],
        {
            "model_name": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 96, 3136, 1)"},
        },
    ),
    (
        Reshape1269,
        [((1, 3136, 96), torch.float32)],
        {
            "model_name": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 8, 7, 8, 7, 96)"},
        },
    ),
    (
        Reshape1282,
        [((1, 3136, 96), torch.float32)],
        {
            "model_name": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 56, 56, 96)"},
        },
    ),
    pytest.param(
        (
            Reshape1412,
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
        Reshape1292,
        [((1, 784, 192), torch.float32)],
        {
            "model_name": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 4, 7, 4, 7, 192)"},
        },
    ),
    (
        Reshape1290,
        [((1, 784, 192), torch.float32)],
        {
            "model_name": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 28, 28, 192)"},
        },
    ),
    (
        Reshape1313,
        [((1, 196, 384), torch.float32)],
        {
            "model_name": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 2, 7, 2, 7, 384)"},
        },
    ),
    (
        Reshape1311,
        [((1, 196, 384), torch.float32)],
        {
            "model_name": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 14, 14, 384)"},
        },
    ),
    (
        Reshape1413,
        [((1, 768, 1), torch.float32)],
        {
            "model_name": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 768, 1)"},
        },
    ),
    (
        Reshape1414,
        [((1, 512, 7, 7), torch.float32)],
        {
            "model_name": [
                "pt_vgg_vgg13_img_cls_torchvision",
                "pt_vgg_vgg19_img_cls_torchvision",
                "pt_vgg_vgg19_bn_obj_det_torchhub",
                "pt_vgg_vgg13_bn_img_cls_torchvision",
                "pt_vgg_vgg16_img_cls_torchvision",
                "pt_vgg_vgg11_img_cls_torchvision",
                "pt_vgg_19_obj_det_hf",
                "pt_vgg_vgg16_bn_img_cls_torchvision",
                "pt_vgg_vgg11_bn_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 25088, 1, 1)"},
        },
    ),
    (
        Reshape1415,
        [((1, 512, 7, 7), torch.float32)],
        {
            "model_name": [
                "pt_vgg_vgg19_obj_det_osmr",
                "pt_vgg_vgg11_obj_det_osmr",
                "pt_vgg_bn_vgg19_obj_det_osmr",
                "pt_vgg_vgg13_obj_det_osmr",
                "pt_vgg_bn_vgg19b_obj_det_osmr",
                "pt_vgg_vgg16_obj_det_osmr",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 25088)"},
        },
    ),
    (
        Reshape1416,
        [((1, 4096, 1, 1), torch.float32)],
        {"model_name": ["pt_vgg_vgg19_bn_obj_det_timm"], "pcc": 0.99, "op_params": {"shape": "(1, 4096, 1, 1)"}},
    ),
    (
        Reshape1417,
        [((160, 1, 3, 3), torch.float32)],
        {
            "model_name": [
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(160, 1, 3, 3)"},
        },
    ),
    (
        Reshape1418,
        [((224, 1, 3, 3), torch.float32)],
        {
            "model_name": [
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(224, 1, 3, 3)"},
        },
    ),
    (
        Reshape1419,
        [((728, 1, 3, 3), torch.float32)],
        {
            "model_name": [
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
                "pt_xception_xception71_tf_in1k_img_cls_timm",
                "pt_xception_xception65_img_cls_timm",
                "pt_xception_xception_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(728, 1, 3, 3)"},
        },
    ),
    (
        Reshape1420,
        [((1536, 1, 3, 3), torch.float32)],
        {
            "model_name": [
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
                "pt_xception_xception71_tf_in1k_img_cls_timm",
                "pt_xception_xception65_img_cls_timm",
                "pt_xception_xception_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1536, 1, 3, 3)"},
        },
    ),
    (
        Reshape1421,
        [((1, 3, 85, 60, 60), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v5_yolov5s_imgcls_torchhub_480x480",
                "pt_yolo_v5_yolov5l_imgcls_torchhub_480x480",
                "pt_yolo_v5_yolov5x_imgcls_torchhub_480x480",
                "pt_yolo_v5_yolov5m_imgcls_torchhub_480x480",
                "pt_yolo_v5_yolov5n_imgcls_torchhub_480x480",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 255, 60, 60)"},
        },
    ),
    (
        Reshape1422,
        [((1, 255, 60, 60), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v5_yolov5s_imgcls_torchhub_480x480",
                "pt_yolo_v5_yolov5l_imgcls_torchhub_480x480",
                "pt_yolo_v5_yolov5x_imgcls_torchhub_480x480",
                "pt_yolo_v5_yolov5m_imgcls_torchhub_480x480",
                "pt_yolo_v5_yolov5n_imgcls_torchhub_480x480",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1, 255, 3600)"},
        },
    ),
    (
        Reshape1423,
        [((1, 1, 255, 3600), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v5_yolov5s_imgcls_torchhub_480x480",
                "pt_yolo_v5_yolov5l_imgcls_torchhub_480x480",
                "pt_yolo_v5_yolov5x_imgcls_torchhub_480x480",
                "pt_yolo_v5_yolov5m_imgcls_torchhub_480x480",
                "pt_yolo_v5_yolov5n_imgcls_torchhub_480x480",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 3, 85, 3600)"},
        },
    ),
    (
        Reshape1424,
        [((1, 3, 3600, 85), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v5_yolov5s_imgcls_torchhub_480x480",
                "pt_yolo_v5_yolov5l_imgcls_torchhub_480x480",
                "pt_yolo_v5_yolov5x_imgcls_torchhub_480x480",
                "pt_yolo_v5_yolov5m_imgcls_torchhub_480x480",
                "pt_yolo_v5_yolov5n_imgcls_torchhub_480x480",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 10800, 85)"},
        },
    ),
    (
        Reshape1425,
        [((1, 3, 85, 30, 30), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v5_yolov5s_imgcls_torchhub_480x480",
                "pt_yolo_v5_yolov5l_imgcls_torchhub_480x480",
                "pt_yolo_v5_yolov5x_imgcls_torchhub_480x480",
                "pt_yolo_v5_yolov5m_imgcls_torchhub_480x480",
                "pt_yolo_v5_yolov5n_imgcls_torchhub_480x480",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 255, 30, 30)"},
        },
    ),
    (
        Reshape1426,
        [((1, 255, 30, 30), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v5_yolov5s_imgcls_torchhub_480x480",
                "pt_yolo_v5_yolov5l_imgcls_torchhub_480x480",
                "pt_yolo_v5_yolov5x_imgcls_torchhub_480x480",
                "pt_yolo_v5_yolov5m_imgcls_torchhub_480x480",
                "pt_yolo_v5_yolov5n_imgcls_torchhub_480x480",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1, 255, 900)"},
        },
    ),
    (
        Reshape1427,
        [((1, 1, 255, 900), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v5_yolov5s_imgcls_torchhub_480x480",
                "pt_yolo_v5_yolov5l_imgcls_torchhub_480x480",
                "pt_yolo_v5_yolov5x_imgcls_torchhub_480x480",
                "pt_yolo_v5_yolov5m_imgcls_torchhub_480x480",
                "pt_yolo_v5_yolov5n_imgcls_torchhub_480x480",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 3, 85, 900)"},
        },
    ),
    (
        Reshape1428,
        [((1, 3, 900, 85), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v5_yolov5s_imgcls_torchhub_480x480",
                "pt_yolo_v5_yolov5l_imgcls_torchhub_480x480",
                "pt_yolo_v5_yolov5x_imgcls_torchhub_480x480",
                "pt_yolo_v5_yolov5m_imgcls_torchhub_480x480",
                "pt_yolo_v5_yolov5n_imgcls_torchhub_480x480",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 2700, 85)"},
        },
    ),
    (
        Reshape1429,
        [((1, 3, 85, 15, 15), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v5_yolov5s_imgcls_torchhub_480x480",
                "pt_yolo_v5_yolov5l_imgcls_torchhub_480x480",
                "pt_yolo_v5_yolov5x_imgcls_torchhub_480x480",
                "pt_yolo_v5_yolov5m_imgcls_torchhub_480x480",
                "pt_yolo_v5_yolov5n_imgcls_torchhub_480x480",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 255, 15, 15)"},
        },
    ),
    (
        Reshape1430,
        [((1, 255, 15, 15), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v5_yolov5s_imgcls_torchhub_480x480",
                "pt_yolo_v5_yolov5l_imgcls_torchhub_480x480",
                "pt_yolo_v5_yolov5x_imgcls_torchhub_480x480",
                "pt_yolo_v5_yolov5m_imgcls_torchhub_480x480",
                "pt_yolo_v5_yolov5n_imgcls_torchhub_480x480",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1, 255, 225)"},
        },
    ),
    (
        Reshape1431,
        [((1, 1, 255, 225), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v5_yolov5s_imgcls_torchhub_480x480",
                "pt_yolo_v5_yolov5l_imgcls_torchhub_480x480",
                "pt_yolo_v5_yolov5x_imgcls_torchhub_480x480",
                "pt_yolo_v5_yolov5m_imgcls_torchhub_480x480",
                "pt_yolo_v5_yolov5n_imgcls_torchhub_480x480",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 3, 85, 225)"},
        },
    ),
    (
        Reshape1432,
        [((1, 3, 225, 85), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v5_yolov5s_imgcls_torchhub_480x480",
                "pt_yolo_v5_yolov5l_imgcls_torchhub_480x480",
                "pt_yolo_v5_yolov5x_imgcls_torchhub_480x480",
                "pt_yolo_v5_yolov5m_imgcls_torchhub_480x480",
                "pt_yolo_v5_yolov5n_imgcls_torchhub_480x480",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 675, 85)"},
        },
    ),
    (
        Reshape1433,
        [((1, 3, 85, 40, 40), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v5_yolov5x_imgcls_torchhub_320x320",
                "pt_yolo_v5_yolov5s_imgcls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5l_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5x_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5n_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5m_imgcls_torchhub_320x320",
                "pt_yolo_v5_yolov5l_imgcls_torchhub_320x320",
                "pt_yolo_v5_yolov5s_imgcls_torchhub_1280x1280",
                "pt_yolo_v5_yolov5n_imgcls_torchhub_320x320",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 255, 40, 40)"},
        },
    ),
    (
        Reshape1434,
        [((1, 255, 40, 40), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v5_yolov5x_imgcls_torchhub_320x320",
                "pt_yolo_v5_yolov5s_imgcls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5l_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5x_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5n_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5m_imgcls_torchhub_320x320",
                "pt_yolo_v5_yolov5l_imgcls_torchhub_320x320",
                "pt_yolo_v5_yolov5s_imgcls_torchhub_1280x1280",
                "pt_yolo_v5_yolov5n_imgcls_torchhub_320x320",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1, 255, 1600)"},
        },
    ),
    (
        Reshape1435,
        [((1, 1, 255, 1600), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v5_yolov5x_imgcls_torchhub_320x320",
                "pt_yolo_v5_yolov5s_imgcls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5l_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5x_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5n_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5m_imgcls_torchhub_320x320",
                "pt_yolo_v5_yolov5l_imgcls_torchhub_320x320",
                "pt_yolo_v5_yolov5s_imgcls_torchhub_1280x1280",
                "pt_yolo_v5_yolov5n_imgcls_torchhub_320x320",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 3, 85, 1600)"},
        },
    ),
    (
        Reshape1436,
        [((1, 3, 1600, 85), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v5_yolov5x_imgcls_torchhub_320x320",
                "pt_yolo_v5_yolov5s_imgcls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5l_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5x_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5n_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5m_imgcls_torchhub_320x320",
                "pt_yolo_v5_yolov5l_imgcls_torchhub_320x320",
                "pt_yolo_v5_yolov5s_imgcls_torchhub_1280x1280",
                "pt_yolo_v5_yolov5n_imgcls_torchhub_320x320",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 4800, 85)"},
        },
    ),
    (
        Reshape1437,
        [((1, 3, 85, 20, 20), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v5_yolov5x_imgcls_torchhub_320x320",
                "pt_yolo_v5_yolov5s_imgcls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5l_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5x_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5n_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5m_imgcls_torchhub_320x320",
                "pt_yolo_v5_yolov5l_imgcls_torchhub_320x320",
                "pt_yolo_v5_yolov5n_imgcls_torchhub_320x320",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 255, 20, 20)"},
        },
    ),
    (
        Reshape1438,
        [((1, 255, 20, 20), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v5_yolov5x_imgcls_torchhub_320x320",
                "pt_yolo_v5_yolov5s_imgcls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5l_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5x_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5n_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5m_imgcls_torchhub_320x320",
                "pt_yolo_v5_yolov5l_imgcls_torchhub_320x320",
                "pt_yolo_v5_yolov5n_imgcls_torchhub_320x320",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1, 255, 400)"},
        },
    ),
    (
        Reshape1439,
        [((1, 1, 255, 400), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v5_yolov5x_imgcls_torchhub_320x320",
                "pt_yolo_v5_yolov5s_imgcls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5l_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5x_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5n_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5m_imgcls_torchhub_320x320",
                "pt_yolo_v5_yolov5l_imgcls_torchhub_320x320",
                "pt_yolo_v5_yolov5n_imgcls_torchhub_320x320",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 3, 85, 400)"},
        },
    ),
    (
        Reshape1440,
        [((1, 3, 400, 85), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v5_yolov5x_imgcls_torchhub_320x320",
                "pt_yolo_v5_yolov5s_imgcls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5l_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5x_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5n_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5m_imgcls_torchhub_320x320",
                "pt_yolo_v5_yolov5l_imgcls_torchhub_320x320",
                "pt_yolo_v5_yolov5n_imgcls_torchhub_320x320",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1200, 85)"},
        },
    ),
    (
        Reshape1441,
        [((1, 3, 85, 10, 10), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v5_yolov5x_imgcls_torchhub_320x320",
                "pt_yolo_v5_yolov5s_imgcls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_imgcls_torchhub_320x320",
                "pt_yolo_v5_yolov5l_imgcls_torchhub_320x320",
                "pt_yolo_v5_yolov5n_imgcls_torchhub_320x320",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 255, 10, 10)"},
        },
    ),
    (
        Reshape1442,
        [((1, 255, 10, 10), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v5_yolov5x_imgcls_torchhub_320x320",
                "pt_yolo_v5_yolov5s_imgcls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_imgcls_torchhub_320x320",
                "pt_yolo_v5_yolov5l_imgcls_torchhub_320x320",
                "pt_yolo_v5_yolov5n_imgcls_torchhub_320x320",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1, 255, 100)"},
        },
    ),
    (
        Reshape1443,
        [((1, 1, 255, 100), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v5_yolov5x_imgcls_torchhub_320x320",
                "pt_yolo_v5_yolov5s_imgcls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_imgcls_torchhub_320x320",
                "pt_yolo_v5_yolov5l_imgcls_torchhub_320x320",
                "pt_yolo_v5_yolov5n_imgcls_torchhub_320x320",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 3, 85, 100)"},
        },
    ),
    (
        Reshape1444,
        [((1, 3, 100, 85), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v5_yolov5x_imgcls_torchhub_320x320",
                "pt_yolo_v5_yolov5s_imgcls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_imgcls_torchhub_320x320",
                "pt_yolo_v5_yolov5l_imgcls_torchhub_320x320",
                "pt_yolo_v5_yolov5n_imgcls_torchhub_320x320",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 300, 85)"},
        },
    ),
    (
        Reshape1445,
        [((1, 3, 85, 80, 80), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v5_yolov5m_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5l_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5x_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5n_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_imgcls_torchhub_1280x1280",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 255, 80, 80)"},
        },
    ),
    (
        Reshape1446,
        [((1, 255, 80, 80), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v5_yolov5m_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5l_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5x_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5n_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_imgcls_torchhub_1280x1280",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1, 255, 6400)"},
        },
    ),
    (
        Reshape1447,
        [((1, 1, 255, 6400), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v5_yolov5m_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5l_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5x_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5n_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_imgcls_torchhub_1280x1280",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 3, 85, 6400)"},
        },
    ),
    (
        Reshape1448,
        [((1, 3, 6400, 85), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v5_yolov5m_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5l_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5x_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5n_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_imgcls_torchhub_1280x1280",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 19200, 85)"},
        },
    ),
    (
        Reshape1449,
        [((1, 3, 85, 160, 160), torch.float32)],
        {
            "model_name": ["pt_yolo_v5_yolov5s_imgcls_torchhub_1280x1280"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 255, 160, 160)"},
        },
    ),
    (
        Reshape1450,
        [((1, 255, 160, 160), torch.float32)],
        {
            "model_name": ["pt_yolo_v5_yolov5s_imgcls_torchhub_1280x1280"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 1, 255, 25600)"},
        },
    ),
    (
        Reshape1451,
        [((1, 1, 255, 25600), torch.float32)],
        {
            "model_name": ["pt_yolo_v5_yolov5s_imgcls_torchhub_1280x1280"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 3, 85, 25600)"},
        },
    ),
    (
        Reshape1452,
        [((1, 3, 25600, 85), torch.float32)],
        {
            "model_name": ["pt_yolo_v5_yolov5s_imgcls_torchhub_1280x1280"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 76800, 85)"},
        },
    ),
    (
        Reshape1453,
        [((1, 4, 56, 80), torch.float32)],
        {
            "model_name": ["pt_yolo_v6_yolov6n_obj_det_torchhub", "pt_yolo_v6_yolov6s_obj_det_torchhub"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 4, 4480)"},
        },
    ),
    (
        Reshape1454,
        [((1, 4, 28, 40), torch.float32)],
        {
            "model_name": ["pt_yolo_v6_yolov6n_obj_det_torchhub", "pt_yolo_v6_yolov6s_obj_det_torchhub"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 4, 1120)"},
        },
    ),
    (
        Reshape1455,
        [((1, 4, 14, 20), torch.float32)],
        {
            "model_name": ["pt_yolo_v6_yolov6n_obj_det_torchhub", "pt_yolo_v6_yolov6s_obj_det_torchhub"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 4, 280)"},
        },
    ),
    (
        Reshape1456,
        [((1, 80, 56, 80), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v6_yolov6n_obj_det_torchhub",
                "pt_yolo_v6_yolov6m_obj_det_torchhub",
                "pt_yolo_v6_yolov6l_obj_det_torchhub",
                "pt_yolo_v6_yolov6s_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 80, 4480)"},
        },
    ),
    (
        Reshape1457,
        [((1, 80, 28, 40), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v6_yolov6n_obj_det_torchhub",
                "pt_yolo_v6_yolov6m_obj_det_torchhub",
                "pt_yolo_v6_yolov6l_obj_det_torchhub",
                "pt_yolo_v6_yolov6s_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 80, 1120)"},
        },
    ),
    (
        Reshape1458,
        [((1, 80, 14, 20), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v6_yolov6n_obj_det_torchhub",
                "pt_yolo_v6_yolov6m_obj_det_torchhub",
                "pt_yolo_v6_yolov6l_obj_det_torchhub",
                "pt_yolo_v6_yolov6s_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 80, 280)"},
        },
    ),
    (
        Reshape1459,
        [((1, 68, 56, 80), torch.float32)],
        {
            "model_name": ["pt_yolo_v6_yolov6m_obj_det_torchhub", "pt_yolo_v6_yolov6l_obj_det_torchhub"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 4, 17, 4480)"},
        },
    ),
    (
        Reshape1453,
        [((1, 1, 4, 4480), torch.float32)],
        {
            "model_name": ["pt_yolo_v6_yolov6m_obj_det_torchhub", "pt_yolo_v6_yolov6l_obj_det_torchhub"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 4, 4480)"},
        },
    ),
    (
        Reshape1460,
        [((1, 68, 28, 40), torch.float32)],
        {
            "model_name": ["pt_yolo_v6_yolov6m_obj_det_torchhub", "pt_yolo_v6_yolov6l_obj_det_torchhub"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 4, 17, 1120)"},
        },
    ),
    (
        Reshape1454,
        [((1, 1, 4, 1120), torch.float32)],
        {
            "model_name": ["pt_yolo_v6_yolov6m_obj_det_torchhub", "pt_yolo_v6_yolov6l_obj_det_torchhub"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 4, 1120)"},
        },
    ),
    (
        Reshape1461,
        [((1, 68, 14, 20), torch.float32)],
        {
            "model_name": ["pt_yolo_v6_yolov6m_obj_det_torchhub", "pt_yolo_v6_yolov6l_obj_det_torchhub"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 4, 17, 280)"},
        },
    ),
    (
        Reshape1455,
        [((1, 1, 4, 280), torch.float32)],
        {
            "model_name": ["pt_yolo_v6_yolov6m_obj_det_torchhub", "pt_yolo_v6_yolov6l_obj_det_torchhub"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 4, 280)"},
        },
    ),
    (
        Reshape1462,
        [((1, 85, 80, 80), torch.float32)],
        {
            "model_name": [
                "pt_yolox_yolox_darknet_obj_det_torchhub",
                "pt_yolox_yolox_x_obj_det_torchhub",
                "pt_yolox_yolox_m_obj_det_torchhub",
                "pt_yolox_yolox_s_obj_det_torchhub",
                "pt_yolox_yolox_l_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 85, 6400, 1)"},
        },
    ),
    (
        Reshape1463,
        [((1, 85, 40, 40), torch.float32)],
        {
            "model_name": [
                "pt_yolox_yolox_darknet_obj_det_torchhub",
                "pt_yolox_yolox_x_obj_det_torchhub",
                "pt_yolox_yolox_m_obj_det_torchhub",
                "pt_yolox_yolox_s_obj_det_torchhub",
                "pt_yolox_yolox_l_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 85, 1600, 1)"},
        },
    ),
    (
        Reshape1464,
        [((1, 85, 20, 20), torch.float32)],
        {
            "model_name": [
                "pt_yolox_yolox_darknet_obj_det_torchhub",
                "pt_yolox_yolox_x_obj_det_torchhub",
                "pt_yolox_yolox_m_obj_det_torchhub",
                "pt_yolox_yolox_s_obj_det_torchhub",
                "pt_yolox_yolox_l_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 85, 400, 1)"},
        },
    ),
    (
        Reshape1465,
        [((1, 85, 52, 52), torch.float32)],
        {
            "model_name": ["pt_yolox_yolox_tiny_obj_det_torchhub", "pt_yolox_yolox_nano_obj_det_torchhub"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 85, 2704, 1)"},
        },
    ),
    (
        Reshape1466,
        [((1, 85, 26, 26), torch.float32)],
        {
            "model_name": ["pt_yolox_yolox_tiny_obj_det_torchhub", "pt_yolox_yolox_nano_obj_det_torchhub"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 85, 676, 1)"},
        },
    ),
    (
        Reshape1467,
        [((1, 85, 13, 13), torch.float32)],
        {
            "model_name": ["pt_yolox_yolox_tiny_obj_det_torchhub", "pt_yolox_yolox_nano_obj_det_torchhub"],
            "pcc": 0.99,
            "op_params": {"shape": "(1, 85, 169, 1)"},
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
