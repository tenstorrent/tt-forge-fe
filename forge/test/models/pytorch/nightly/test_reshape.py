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
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 16, 64))
        return reshape_output_1


class Reshape31(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 1024))
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
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2, 1024))
        return reshape_output_1


class Reshape36(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2, 16, 64))
        return reshape_output_1


class Reshape37(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 1, 64))
        return reshape_output_1


class Reshape38(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 16, 1, 64))
        return reshape_output_1


class Reshape39(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 13, 1024))
        return reshape_output_1


class Reshape40(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 13, 16, 64))
        return reshape_output_1


class Reshape41(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(26, 1024))
        return reshape_output_1


class Reshape42(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 13, 64))
        return reshape_output_1


class Reshape43(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 16, 1, 13))
        return reshape_output_1


class Reshape44(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 1, 13))
        return reshape_output_1


class Reshape45(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 1, 4096))
        return reshape_output_1


class Reshape46(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 4096))
        return reshape_output_1


class Reshape47(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1536))
        return reshape_output_1


class Reshape48(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 1536))
        return reshape_output_1


class Reshape49(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 1, 24, 64))
        return reshape_output_1


class Reshape50(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 1, 1536))
        return reshape_output_1


class Reshape51(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(48, 1, 64))
        return reshape_output_1


class Reshape52(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 24, 1, 64))
        return reshape_output_1


class Reshape53(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 13, 1536))
        return reshape_output_1


class Reshape54(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 13, 24, 64))
        return reshape_output_1


class Reshape55(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(26, 1536))
        return reshape_output_1


class Reshape56(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(48, 13, 64))
        return reshape_output_1


class Reshape57(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 24, 1, 13))
        return reshape_output_1


class Reshape58(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(48, 1, 13))
        return reshape_output_1


class Reshape59(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 1, 6144))
        return reshape_output_1


class Reshape60(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 6144))
        return reshape_output_1


class Reshape61(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2))
        return reshape_output_1


class Reshape62(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 768))
        return reshape_output_1


class Reshape63(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2, 12, 64))
        return reshape_output_1


class Reshape64(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2, 768))
        return reshape_output_1


class Reshape65(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 2, 64))
        return reshape_output_1


class Reshape66(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 2, 2))
        return reshape_output_1


class Reshape67(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 2, 2))
        return reshape_output_1


class Reshape68(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 64, 2))
        return reshape_output_1


class Reshape69(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 2, 64))
        return reshape_output_1


class Reshape70(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1500, 768))
        return reshape_output_1


class Reshape71(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1500, 12, 64))
        return reshape_output_1


class Reshape72(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1500, 768))
        return reshape_output_1


class Reshape73(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 1500, 64))
        return reshape_output_1


class Reshape74(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 2, 1500))
        return reshape_output_1


class Reshape75(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 2, 1500))
        return reshape_output_1


class Reshape76(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 64, 1500))
        return reshape_output_1


class Reshape77(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 1280))
        return reshape_output_1


class Reshape78(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2, 20, 64))
        return reshape_output_1


class Reshape79(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2, 1280))
        return reshape_output_1


class Reshape80(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(20, 2, 64))
        return reshape_output_1


class Reshape81(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 20, 2, 2))
        return reshape_output_1


class Reshape82(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(20, 2, 2))
        return reshape_output_1


class Reshape83(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(20, 64, 2))
        return reshape_output_1


class Reshape84(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 20, 2, 64))
        return reshape_output_1


class Reshape85(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1500, 1280))
        return reshape_output_1


class Reshape86(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1500, 20, 64))
        return reshape_output_1


class Reshape87(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1500, 1280))
        return reshape_output_1


class Reshape88(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(20, 1500, 64))
        return reshape_output_1


class Reshape89(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 20, 2, 1500))
        return reshape_output_1


class Reshape90(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(20, 2, 1500))
        return reshape_output_1


class Reshape91(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(20, 64, 1500))
        return reshape_output_1


class Reshape92(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 2, 64))
        return reshape_output_1


class Reshape93(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 2, 2))
        return reshape_output_1


class Reshape94(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 2, 2))
        return reshape_output_1


class Reshape95(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 64, 2))
        return reshape_output_1


class Reshape96(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 2, 64))
        return reshape_output_1


class Reshape97(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1500, 1024))
        return reshape_output_1


class Reshape98(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1500, 16, 64))
        return reshape_output_1


class Reshape99(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1500, 1024))
        return reshape_output_1


class Reshape100(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 1500, 64))
        return reshape_output_1


class Reshape101(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 2, 1500))
        return reshape_output_1


class Reshape102(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 2, 1500))
        return reshape_output_1


class Reshape103(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 64, 1500))
        return reshape_output_1


class Reshape104(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 384))
        return reshape_output_1


class Reshape105(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2, 6, 64))
        return reshape_output_1


class Reshape106(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2, 384))
        return reshape_output_1


class Reshape107(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(6, 2, 64))
        return reshape_output_1


class Reshape108(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 6, 2, 2))
        return reshape_output_1


class Reshape109(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(6, 2, 2))
        return reshape_output_1


class Reshape110(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(6, 64, 2))
        return reshape_output_1


class Reshape111(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 6, 2, 64))
        return reshape_output_1


class Reshape112(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1500, 384))
        return reshape_output_1


class Reshape113(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1500, 6, 64))
        return reshape_output_1


class Reshape114(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1500, 384))
        return reshape_output_1


class Reshape115(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(6, 1500, 64))
        return reshape_output_1


class Reshape116(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 6, 2, 1500))
        return reshape_output_1


class Reshape117(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(6, 2, 1500))
        return reshape_output_1


class Reshape118(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(6, 64, 1500))
        return reshape_output_1


class Reshape119(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 512))
        return reshape_output_1


class Reshape120(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2, 8, 64))
        return reshape_output_1


class Reshape121(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2, 512))
        return reshape_output_1


class Reshape122(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 2, 64))
        return reshape_output_1


class Reshape123(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 2, 2))
        return reshape_output_1


class Reshape124(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 2, 2))
        return reshape_output_1


class Reshape125(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 64, 2))
        return reshape_output_1


class Reshape126(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 2, 64))
        return reshape_output_1


class Reshape127(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1500, 512))
        return reshape_output_1


class Reshape128(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1500, 8, 64))
        return reshape_output_1


class Reshape129(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1500, 512))
        return reshape_output_1


class Reshape130(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 1500, 64))
        return reshape_output_1


class Reshape131(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 2, 1500))
        return reshape_output_1


class Reshape132(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 2, 1500))
        return reshape_output_1


class Reshape133(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 64, 1500))
        return reshape_output_1


class Reshape134(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 7))
        return reshape_output_1


class Reshape135(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(14, 512))
        return reshape_output_1


class Reshape136(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 7, 8, 64))
        return reshape_output_1


class Reshape137(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 7, 512))
        return reshape_output_1


class Reshape138(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 7, 64))
        return reshape_output_1


class Reshape139(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 8, 7, 7))
        return reshape_output_1


class Reshape140(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 7, 7))
        return reshape_output_1


class Reshape141(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 8, 7, 64))
        return reshape_output_1


class Reshape142(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 7, 2048))
        return reshape_output_1


class Reshape143(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(14, 2048))
        return reshape_output_1


class Reshape144(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(204, 768))
        return reshape_output_1


class Reshape145(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 204, 12, 64))
        return reshape_output_1


class Reshape146(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 204, 768))
        return reshape_output_1


class Reshape147(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 204, 64))
        return reshape_output_1


class Reshape148(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 204, 204))
        return reshape_output_1


class Reshape149(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 204, 204))
        return reshape_output_1


class Reshape150(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 64, 204))
        return reshape_output_1


class Reshape151(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 204, 64))
        return reshape_output_1


class Reshape152(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(201, 768))
        return reshape_output_1


class Reshape153(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 201, 12, 64))
        return reshape_output_1


class Reshape154(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 201, 768))
        return reshape_output_1


class Reshape155(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 201, 64))
        return reshape_output_1


class Reshape156(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 201, 201))
        return reshape_output_1


class Reshape157(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 201, 201))
        return reshape_output_1


class Reshape158(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 64, 201))
        return reshape_output_1


class Reshape159(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 201, 64))
        return reshape_output_1


class Reshape160(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 768))
        return reshape_output_1


class Reshape161(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(128, 768))
        return reshape_output_1


class Reshape162(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 12, 64))
        return reshape_output_1


class Reshape163(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 768))
        return reshape_output_1


class Reshape164(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 128, 64))
        return reshape_output_1


class Reshape165(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 128, 128))
        return reshape_output_1


class Reshape166(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 128, 128))
        return reshape_output_1


class Reshape167(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 64, 128))
        return reshape_output_1


class Reshape168(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 768, 128, 1))
        return reshape_output_1


class Reshape169(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 128, 64))
        return reshape_output_1


class Reshape170(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 768, 1))
        return reshape_output_1


class Reshape171(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(128, 2048))
        return reshape_output_1


class Reshape172(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 16, 128))
        return reshape_output_1


class Reshape173(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 2048))
        return reshape_output_1


class Reshape174(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 128, 128))
        return reshape_output_1


class Reshape175(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 128, 128))
        return reshape_output_1


class Reshape176(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 2048, 1))
        return reshape_output_1


class Reshape177(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(128, 1024))
        return reshape_output_1


class Reshape178(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 16, 64))
        return reshape_output_1


class Reshape179(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 1024))
        return reshape_output_1


class Reshape180(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 8, 128))
        return reshape_output_1


class Reshape181(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 128, 64))
        return reshape_output_1


class Reshape182(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 64, 128))
        return reshape_output_1


class Reshape183(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 128, 64))
        return reshape_output_1


class Reshape184(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 1024, 1))
        return reshape_output_1


class Reshape185(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(128, 4096))
        return reshape_output_1


class Reshape186(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 64, 64))
        return reshape_output_1


class Reshape187(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 4096))
        return reshape_output_1


class Reshape188(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 32, 128))
        return reshape_output_1


class Reshape189(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 128, 64))
        return reshape_output_1


class Reshape190(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 64, 128, 128))
        return reshape_output_1


class Reshape191(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 128, 128))
        return reshape_output_1


class Reshape192(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 64, 16384, 1))
        return reshape_output_1


class Reshape193(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 64, 128))
        return reshape_output_1


class Reshape194(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 64, 128, 64))
        return reshape_output_1


class Reshape195(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 4096, 1))
        return reshape_output_1


class Reshape196(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(256, 1024))
        return reshape_output_1


class Reshape197(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 16, 64))
        return reshape_output_1


class Reshape198(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 32, 32))
        return reshape_output_1


class Reshape199(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 1024))
        return reshape_output_1


class Reshape200(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 4, 256))
        return reshape_output_1


class Reshape201(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 8, 128))
        return reshape_output_1


class Reshape202(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 256, 64))
        return reshape_output_1


class Reshape203(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 256, 256))
        return reshape_output_1


class Reshape204(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 256, 256))
        return reshape_output_1


class Reshape205(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 256, 64))
        return reshape_output_1


class Reshape206(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256))
        return reshape_output_1


class Reshape207(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(384, 1024))
        return reshape_output_1


class Reshape208(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 384, 16, 64))
        return reshape_output_1


class Reshape209(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 384, 1024))
        return reshape_output_1


class Reshape210(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 384, 64))
        return reshape_output_1


class Reshape211(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 384, 384))
        return reshape_output_1


class Reshape212(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 384, 384))
        return reshape_output_1


class Reshape213(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 64, 384))
        return reshape_output_1


class Reshape214(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 384, 64))
        return reshape_output_1


class Reshape215(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 384, 1))
        return reshape_output_1


class Reshape216(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 16, 32, 1))
        return reshape_output_1


class Reshape217(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 64, 256))
        return reshape_output_1


class Reshape218(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 4096))
        return reshape_output_1


class Reshape219(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 32, 128))
        return reshape_output_1


class Reshape220(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 1, 128))
        return reshape_output_1


class Reshape221(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(384, 768))
        return reshape_output_1


class Reshape222(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 384, 12, 64))
        return reshape_output_1


class Reshape223(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 384, 768))
        return reshape_output_1


class Reshape224(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 384, 64))
        return reshape_output_1


class Reshape225(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 384, 384))
        return reshape_output_1


class Reshape226(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 1, 384))
        return reshape_output_1


class Reshape227(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 384, 384))
        return reshape_output_1


class Reshape228(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 64, 384))
        return reshape_output_1


class Reshape229(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 384, 64))
        return reshape_output_1


class Reshape230(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 1))
        return reshape_output_1


class Reshape231(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128))
        return reshape_output_1


class Reshape232(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1,))
        return reshape_output_1


class Reshape233(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(6, 4544))
        return reshape_output_1


class Reshape234(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 6, 18176))
        return reshape_output_1


class Reshape235(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 6, 73, 64))
        return reshape_output_1


class Reshape236(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 71, 6, 64))
        return reshape_output_1


class Reshape237(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(71, 6, 64))
        return reshape_output_1


class Reshape238(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 6, 1, 64))
        return reshape_output_1


class Reshape239(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(6, 1, 64))
        return reshape_output_1


class Reshape240(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 6, 64))
        return reshape_output_1


class Reshape241(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 6, 64))
        return reshape_output_1


class Reshape242(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 384))
        return reshape_output_1


class Reshape243(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 71, 6, 6))
        return reshape_output_1


class Reshape244(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(71, 6, 6))
        return reshape_output_1


class Reshape245(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 6, 4544))
        return reshape_output_1


class Reshape246(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 334, 64, 3, 64))
        return reshape_output_1


class Reshape247(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 334, 64, 64))
        return reshape_output_1


class Reshape248(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 334, 64))
        return reshape_output_1


class Reshape249(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 64, 334, 334))
        return reshape_output_1


class Reshape250(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 334, 334))
        return reshape_output_1


class Reshape251(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 64, 334))
        return reshape_output_1


class Reshape252(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 64, 334, 64))
        return reshape_output_1


class Reshape253(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(334, 4096))
        return reshape_output_1


class Reshape254(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 334, 4096))
        return reshape_output_1


class Reshape255(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(7, 2048))
        return reshape_output_1


class Reshape256(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 7, 8, 256))
        return reshape_output_1


class Reshape257(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 7, 2048))
        return reshape_output_1


class Reshape258(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 7, 256))
        return reshape_output_1


class Reshape259(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 7, 1, 256))
        return reshape_output_1


class Reshape260(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 7, 256))
        return reshape_output_1


class Reshape261(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 7, 7))
        return reshape_output_1


class Reshape262(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 7, 7))
        return reshape_output_1


class Reshape263(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 256, 7))
        return reshape_output_1


class Reshape264(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 7, 16384))
        return reshape_output_1


class Reshape265(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(256, 768))
        return reshape_output_1


class Reshape266(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 12, 64))
        return reshape_output_1


class Reshape267(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 768))
        return reshape_output_1


class Reshape268(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 256, 64))
        return reshape_output_1


class Reshape269(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 256, 256))
        return reshape_output_1


class Reshape270(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 256, 256))
        return reshape_output_1


class Reshape271(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 64, 256))
        return reshape_output_1


class Reshape272(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 256, 64))
        return reshape_output_1


class Reshape273(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 3072))
        return reshape_output_1


class Reshape274(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(256, 3072))
        return reshape_output_1


class Reshape275(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(256, 2560))
        return reshape_output_1


class Reshape276(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 32, 80))
        return reshape_output_1


class Reshape277(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 20, 128))
        return reshape_output_1


class Reshape278(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 2560))
        return reshape_output_1


class Reshape279(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(20, 256, 128))
        return reshape_output_1


class Reshape280(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 20, 256, 256))
        return reshape_output_1


class Reshape281(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(20, 256, 256))
        return reshape_output_1


class Reshape282(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(20, 128, 256))
        return reshape_output_1


class Reshape283(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 20, 256, 128))
        return reshape_output_1


class Reshape284(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(256, 2048))
        return reshape_output_1


class Reshape285(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 32, 64))
        return reshape_output_1


class Reshape286(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 16, 128))
        return reshape_output_1


class Reshape287(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 2048))
        return reshape_output_1


class Reshape288(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 256, 128))
        return reshape_output_1


class Reshape289(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 128, 256))
        return reshape_output_1


class Reshape290(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 256, 128))
        return reshape_output_1


class Reshape291(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(256, 4096))
        return reshape_output_1


class Reshape292(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 64, 64))
        return reshape_output_1


class Reshape293(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 256, 128))
        return reshape_output_1


class Reshape294(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 256, 128))
        return reshape_output_1


class Reshape295(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 256, 256))
        return reshape_output_1


class Reshape296(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 256, 256))
        return reshape_output_1


class Reshape297(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 128, 256))
        return reshape_output_1


class Reshape298(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 14336))
        return reshape_output_1


class Reshape299(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 256, 64))
        return reshape_output_1


class Reshape300(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 8, 64))
        return reshape_output_1


class Reshape301(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(256, 512))
        return reshape_output_1


class Reshape302(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 512))
        return reshape_output_1


class Reshape303(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 256, 64))
        return reshape_output_1


class Reshape304(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 64, 256))
        return reshape_output_1


class Reshape305(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 8192))
        return reshape_output_1


class Reshape306(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(4, 4096))
        return reshape_output_1


class Reshape307(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 32, 128))
        return reshape_output_1


class Reshape308(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 4096))
        return reshape_output_1


class Reshape309(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 4, 128))
        return reshape_output_1


class Reshape310(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 8, 128))
        return reshape_output_1


class Reshape311(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 4, 128))
        return reshape_output_1


class Reshape312(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 4, 4))
        return reshape_output_1


class Reshape313(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 4, 4))
        return reshape_output_1


class Reshape314(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 128, 4))
        return reshape_output_1


class Reshape315(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 14336))
        return reshape_output_1


class Reshape316(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(4, 2048))
        return reshape_output_1


class Reshape317(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 32, 64))
        return reshape_output_1


class Reshape318(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 2048))
        return reshape_output_1


class Reshape319(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 4, 64))
        return reshape_output_1


class Reshape320(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 8, 64))
        return reshape_output_1


class Reshape321(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 4, 64))
        return reshape_output_1


class Reshape322(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 64, 4))
        return reshape_output_1


class Reshape323(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 8192))
        return reshape_output_1


class Reshape324(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 128, 128))
        return reshape_output_1


class Reshape325(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 16384, 1))
        return reshape_output_1


class Reshape326(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 128, 128))
        return reshape_output_1


class Reshape327(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 14336))
        return reshape_output_1


class Reshape328(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32))
        return reshape_output_1


class Reshape329(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 768))
        return reshape_output_1


class Reshape330(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 12, 64))
        return reshape_output_1


class Reshape331(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 768))
        return reshape_output_1


class Reshape332(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 32, 64))
        return reshape_output_1


class Reshape333(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 32, 32))
        return reshape_output_1


class Reshape334(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 32, 32))
        return reshape_output_1


class Reshape335(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 32, 64))
        return reshape_output_1


class Reshape336(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 2))
        return reshape_output_1


class Reshape337(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 2048))
        return reshape_output_1


class Reshape338(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 32, 64))
        return reshape_output_1


class Reshape339(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 2048))
        return reshape_output_1


class Reshape340(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 32, 64))
        return reshape_output_1


class Reshape341(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 32, 32))
        return reshape_output_1


class Reshape342(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 32, 32))
        return reshape_output_1


class Reshape343(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 1))
        return reshape_output_1


class Reshape344(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 1024))
        return reshape_output_1


class Reshape345(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 16, 64))
        return reshape_output_1


class Reshape346(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 1024))
        return reshape_output_1


class Reshape347(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 32, 64))
        return reshape_output_1


class Reshape348(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 32, 32))
        return reshape_output_1


class Reshape349(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 32, 32))
        return reshape_output_1


class Reshape350(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 32, 64))
        return reshape_output_1


class Reshape351(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 512))
        return reshape_output_1


class Reshape352(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 50272))
        return reshape_output_1


class Reshape353(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 2560))
        return reshape_output_1


class Reshape354(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 32, 80))
        return reshape_output_1


class Reshape355(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 2560))
        return reshape_output_1


class Reshape356(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 12, 80))
        return reshape_output_1


class Reshape357(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 12, 12))
        return reshape_output_1


class Reshape358(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 12, 12))
        return reshape_output_1


class Reshape359(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 80, 12))
        return reshape_output_1


class Reshape360(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 12, 80))
        return reshape_output_1


class Reshape361(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 10240))
        return reshape_output_1


class Reshape362(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 256, 80))
        return reshape_output_1


class Reshape363(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 80, 256))
        return reshape_output_1


class Reshape364(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 256, 80))
        return reshape_output_1


class Reshape365(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 10240))
        return reshape_output_1


class Reshape366(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(11, 2560))
        return reshape_output_1


class Reshape367(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 11, 32, 80))
        return reshape_output_1


class Reshape368(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 11, 2560))
        return reshape_output_1


class Reshape369(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 11, 80))
        return reshape_output_1


class Reshape370(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 11, 11))
        return reshape_output_1


class Reshape371(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 11, 11))
        return reshape_output_1


class Reshape372(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 80, 11))
        return reshape_output_1


class Reshape373(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 11, 80))
        return reshape_output_1


class Reshape374(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 11, 10240))
        return reshape_output_1


class Reshape375(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(29, 1024))
        return reshape_output_1


class Reshape376(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 29, 16, 64))
        return reshape_output_1


class Reshape377(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 29, 1024))
        return reshape_output_1


class Reshape378(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 29, 64))
        return reshape_output_1


class Reshape379(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 29, 29))
        return reshape_output_1


class Reshape380(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 29, 29))
        return reshape_output_1


class Reshape381(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 64, 29))
        return reshape_output_1


class Reshape382(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 29, 64))
        return reshape_output_1


class Reshape383(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 29, 2816))
        return reshape_output_1


class Reshape384(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(6, 1024))
        return reshape_output_1


class Reshape385(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 6, 16, 64))
        return reshape_output_1


class Reshape386(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 6, 1024))
        return reshape_output_1


class Reshape387(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 6, 64))
        return reshape_output_1


class Reshape388(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 6, 6))
        return reshape_output_1


class Reshape389(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 6, 6))
        return reshape_output_1


class Reshape390(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 64, 6))
        return reshape_output_1


class Reshape391(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 6, 64))
        return reshape_output_1


class Reshape392(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 6, 2816))
        return reshape_output_1


class Reshape393(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(35, 1536))
        return reshape_output_1


class Reshape394(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 35, 12, 128))
        return reshape_output_1


class Reshape395(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 35, 1536))
        return reshape_output_1


class Reshape396(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 35, 128))
        return reshape_output_1


class Reshape397(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 35, 256))
        return reshape_output_1


class Reshape398(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 35, 2, 128))
        return reshape_output_1


class Reshape399(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 35, 128))
        return reshape_output_1


class Reshape400(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 35, 35))
        return reshape_output_1


class Reshape401(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 35, 35))
        return reshape_output_1


class Reshape402(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 128, 35))
        return reshape_output_1


class Reshape403(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 35, 8960))
        return reshape_output_1


class Reshape404(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(35, 2048))
        return reshape_output_1


class Reshape405(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 35, 16, 128))
        return reshape_output_1


class Reshape406(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 35, 2048))
        return reshape_output_1


class Reshape407(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 35, 128))
        return reshape_output_1


class Reshape408(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 35, 128))
        return reshape_output_1


class Reshape409(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 35, 35))
        return reshape_output_1


class Reshape410(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 35, 35))
        return reshape_output_1


class Reshape411(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 128, 35))
        return reshape_output_1


class Reshape412(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 35, 11008))
        return reshape_output_1


class Reshape413(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(35, 3584))
        return reshape_output_1


class Reshape414(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 35, 28, 128))
        return reshape_output_1


class Reshape415(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 35, 3584))
        return reshape_output_1


class Reshape416(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(28, 35, 128))
        return reshape_output_1


class Reshape417(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 35, 512))
        return reshape_output_1


class Reshape418(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 35, 4, 128))
        return reshape_output_1


class Reshape419(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 28, 35, 128))
        return reshape_output_1


class Reshape420(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 28, 35, 35))
        return reshape_output_1


class Reshape421(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(28, 35, 35))
        return reshape_output_1


class Reshape422(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(28, 128, 35))
        return reshape_output_1


class Reshape423(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 35, 18944))
        return reshape_output_1


class Reshape424(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(35, 896))
        return reshape_output_1


class Reshape425(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 35, 14, 64))
        return reshape_output_1


class Reshape426(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 35, 896))
        return reshape_output_1


class Reshape427(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(14, 35, 64))
        return reshape_output_1


class Reshape428(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 35, 128))
        return reshape_output_1


class Reshape429(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 35, 2, 64))
        return reshape_output_1


class Reshape430(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 14, 35, 64))
        return reshape_output_1


class Reshape431(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 14, 35, 35))
        return reshape_output_1


class Reshape432(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(14, 35, 35))
        return reshape_output_1


class Reshape433(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(14, 64, 35))
        return reshape_output_1


class Reshape434(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 35, 4864))
        return reshape_output_1


class Reshape435(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(39, 1536))
        return reshape_output_1


class Reshape436(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 39, 12, 128))
        return reshape_output_1


class Reshape437(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 39, 1536))
        return reshape_output_1


class Reshape438(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 39, 128))
        return reshape_output_1


class Reshape439(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 39, 256))
        return reshape_output_1


class Reshape440(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 39, 2, 128))
        return reshape_output_1


class Reshape441(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 39, 128))
        return reshape_output_1


class Reshape442(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 39, 39))
        return reshape_output_1


class Reshape443(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 39, 39))
        return reshape_output_1


class Reshape444(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 128, 39))
        return reshape_output_1


class Reshape445(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 39, 8960))
        return reshape_output_1


class Reshape446(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(29, 1536))
        return reshape_output_1


class Reshape447(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 29, 12, 128))
        return reshape_output_1


class Reshape448(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 29, 1536))
        return reshape_output_1


class Reshape449(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 29, 128))
        return reshape_output_1


class Reshape450(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 29, 256))
        return reshape_output_1


class Reshape451(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 29, 2, 128))
        return reshape_output_1


class Reshape452(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 29, 128))
        return reshape_output_1


class Reshape453(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 29, 29))
        return reshape_output_1


class Reshape454(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 29, 29))
        return reshape_output_1


class Reshape455(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 128, 29))
        return reshape_output_1


class Reshape456(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 29, 8960))
        return reshape_output_1


class Reshape457(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(29, 3584))
        return reshape_output_1


class Reshape458(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 29, 28, 128))
        return reshape_output_1


class Reshape459(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 29, 3584))
        return reshape_output_1


class Reshape460(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(28, 29, 128))
        return reshape_output_1


class Reshape461(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 29, 512))
        return reshape_output_1


class Reshape462(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 29, 4, 128))
        return reshape_output_1


class Reshape463(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 28, 29, 128))
        return reshape_output_1


class Reshape464(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 28, 29, 29))
        return reshape_output_1


class Reshape465(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(28, 29, 29))
        return reshape_output_1


class Reshape466(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(28, 128, 29))
        return reshape_output_1


class Reshape467(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 29, 18944))
        return reshape_output_1


class Reshape468(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(39, 896))
        return reshape_output_1


class Reshape469(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 39, 14, 64))
        return reshape_output_1


class Reshape470(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 39, 896))
        return reshape_output_1


class Reshape471(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(14, 39, 64))
        return reshape_output_1


class Reshape472(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 39, 128))
        return reshape_output_1


class Reshape473(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 39, 2, 64))
        return reshape_output_1


class Reshape474(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 14, 39, 64))
        return reshape_output_1


class Reshape475(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 14, 39, 39))
        return reshape_output_1


class Reshape476(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(14, 39, 39))
        return reshape_output_1


class Reshape477(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(14, 64, 39))
        return reshape_output_1


class Reshape478(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 39, 4864))
        return reshape_output_1


class Reshape479(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(29, 896))
        return reshape_output_1


class Reshape480(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 29, 14, 64))
        return reshape_output_1


class Reshape481(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 29, 896))
        return reshape_output_1


class Reshape482(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(14, 29, 64))
        return reshape_output_1


class Reshape483(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 29, 128))
        return reshape_output_1


class Reshape484(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 29, 2, 64))
        return reshape_output_1


class Reshape485(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 14, 29, 64))
        return reshape_output_1


class Reshape486(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 14, 29, 29))
        return reshape_output_1


class Reshape487(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(14, 29, 29))
        return reshape_output_1


class Reshape488(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(14, 64, 29))
        return reshape_output_1


class Reshape489(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 29, 4864))
        return reshape_output_1


class Reshape490(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(39, 2048))
        return reshape_output_1


class Reshape491(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 39, 16, 128))
        return reshape_output_1


class Reshape492(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 39, 2048))
        return reshape_output_1


class Reshape493(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 39, 128))
        return reshape_output_1


class Reshape494(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 39, 128))
        return reshape_output_1


class Reshape495(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 39, 39))
        return reshape_output_1


class Reshape496(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 39, 39))
        return reshape_output_1


class Reshape497(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 128, 39))
        return reshape_output_1


class Reshape498(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 39, 11008))
        return reshape_output_1


class Reshape499(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(39, 3584))
        return reshape_output_1


class Reshape500(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 39, 28, 128))
        return reshape_output_1


class Reshape501(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 39, 3584))
        return reshape_output_1


class Reshape502(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(28, 39, 128))
        return reshape_output_1


class Reshape503(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 39, 512))
        return reshape_output_1


class Reshape504(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 39, 4, 128))
        return reshape_output_1


class Reshape505(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 28, 39, 128))
        return reshape_output_1


class Reshape506(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 28, 39, 39))
        return reshape_output_1


class Reshape507(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(28, 39, 39))
        return reshape_output_1


class Reshape508(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(28, 128, 39))
        return reshape_output_1


class Reshape509(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 39, 18944))
        return reshape_output_1


class Reshape510(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(29, 2048))
        return reshape_output_1


class Reshape511(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 29, 16, 128))
        return reshape_output_1


class Reshape512(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 29, 2048))
        return reshape_output_1


class Reshape513(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 29, 128))
        return reshape_output_1


class Reshape514(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 29, 128))
        return reshape_output_1


class Reshape515(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 128, 29))
        return reshape_output_1


class Reshape516(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 29, 11008))
        return reshape_output_1


class Reshape517(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 64, 128))
        return reshape_output_1


class Reshape518(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(768, 768, 1, 1))
        return reshape_output_1


class Reshape519(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 768, 128))
        return reshape_output_1


class Reshape520(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1))
        return reshape_output_1


class Reshape521(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 1, 1024))
        return reshape_output_1


class Reshape522(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 1, 64))
        return reshape_output_1


class Reshape523(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 1, 1))
        return reshape_output_1


class Reshape524(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 1, 1))
        return reshape_output_1


class Reshape525(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 4))
        return reshape_output_1


class Reshape526(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 64, 1))
        return reshape_output_1


class Reshape527(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 1, 64))
        return reshape_output_1


class Reshape528(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 1, 256))
        return reshape_output_1


class Reshape529(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 1, 256))
        return reshape_output_1


class Reshape530(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 12, 64))
        return reshape_output_1


class Reshape531(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 768))
        return reshape_output_1


class Reshape532(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 1, 64))
        return reshape_output_1


class Reshape533(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 1, 1))
        return reshape_output_1


class Reshape534(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 1, 1))
        return reshape_output_1


class Reshape535(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 64, 1))
        return reshape_output_1


class Reshape536(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 1, 64))
        return reshape_output_1


class Reshape537(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 1, 256))
        return reshape_output_1


class Reshape538(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 1, 256))
        return reshape_output_1


class Reshape539(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 512))
        return reshape_output_1


class Reshape540(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 1, 512))
        return reshape_output_1


class Reshape541(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 8, 64))
        return reshape_output_1


class Reshape542(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 512))
        return reshape_output_1


class Reshape543(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 1, 64))
        return reshape_output_1


class Reshape544(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 1, 1))
        return reshape_output_1


class Reshape545(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 1, 1))
        return reshape_output_1


class Reshape546(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 64, 1))
        return reshape_output_1


class Reshape547(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 1, 64))
        return reshape_output_1


class Reshape548(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 6, 1, 1))
        return reshape_output_1


class Reshape549(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(6, 1, 1))
        return reshape_output_1


class Reshape550(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(6, 64, 1))
        return reshape_output_1


class Reshape551(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 9216, 1, 1))
        return reshape_output_1


class Reshape552(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 384, 196, 1))
        return reshape_output_1


class Reshape553(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(197, 384))
        return reshape_output_1


class Reshape554(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 197, 6, 64))
        return reshape_output_1


class Reshape555(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 197, 384))
        return reshape_output_1


class Reshape556(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(6, 197, 64))
        return reshape_output_1


class Reshape557(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 6, 197, 197))
        return reshape_output_1


class Reshape558(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(6, 197, 197))
        return reshape_output_1


class Reshape559(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(6, 64, 197))
        return reshape_output_1


class Reshape560(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 6, 197, 64))
        return reshape_output_1


class Reshape561(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 768, 196, 1))
        return reshape_output_1


class Reshape562(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(197, 768))
        return reshape_output_1


class Reshape563(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 197, 12, 64))
        return reshape_output_1


class Reshape564(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 197, 768))
        return reshape_output_1


class Reshape565(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 197, 64))
        return reshape_output_1


class Reshape566(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 197, 197))
        return reshape_output_1


class Reshape567(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 197, 197))
        return reshape_output_1


class Reshape568(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 64, 197))
        return reshape_output_1


class Reshape569(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 197, 64))
        return reshape_output_1


class Reshape570(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 192, 196, 1))
        return reshape_output_1


class Reshape571(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(197, 192))
        return reshape_output_1


class Reshape572(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 197, 3, 64))
        return reshape_output_1


class Reshape573(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 197, 192))
        return reshape_output_1


class Reshape574(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(3, 197, 64))
        return reshape_output_1


class Reshape575(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 3, 197, 197))
        return reshape_output_1


class Reshape576(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(3, 197, 197))
        return reshape_output_1


class Reshape577(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(3, 64, 197))
        return reshape_output_1


class Reshape578(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 3, 197, 64))
        return reshape_output_1


class Reshape579(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 192))
        return reshape_output_1


class Reshape580(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1024, 1, 1))
        return reshape_output_1


class Reshape581(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2208, 1, 1))
        return reshape_output_1


class Reshape582(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1664, 1, 1))
        return reshape_output_1


class Reshape583(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1920, 1, 1))
        return reshape_output_1


class Reshape584(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1000))
        return reshape_output_1


class Reshape585(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(48, 1, 3, 3))
        return reshape_output_1


class Reshape586(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(24, 1, 3, 3))
        return reshape_output_1


class Reshape587(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(144, 1, 3, 3))
        return reshape_output_1


class Reshape588(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(192, 1, 3, 3))
        return reshape_output_1


class Reshape589(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(192, 1, 5, 5))
        return reshape_output_1


class Reshape590(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(336, 1, 5, 5))
        return reshape_output_1


class Reshape591(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(336, 1, 3, 3))
        return reshape_output_1


class Reshape592(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(672, 1, 3, 3))
        return reshape_output_1


class Reshape593(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(672, 1, 5, 5))
        return reshape_output_1


class Reshape594(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(960, 1, 5, 5))
        return reshape_output_1


class Reshape595(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1632, 1, 5, 5))
        return reshape_output_1


class Reshape596(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1632, 1, 3, 3))
        return reshape_output_1


class Reshape597(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2688, 1, 3, 3))
        return reshape_output_1


class Reshape598(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1792, 1, 1))
        return reshape_output_1


class Reshape599(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 1, 3, 3))
        return reshape_output_1


class Reshape600(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(96, 1, 3, 3))
        return reshape_output_1


class Reshape601(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(144, 1, 5, 5))
        return reshape_output_1


class Reshape602(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(240, 1, 5, 5))
        return reshape_output_1


class Reshape603(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(240, 1, 3, 3))
        return reshape_output_1


class Reshape604(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(480, 1, 3, 3))
        return reshape_output_1


class Reshape605(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(480, 1, 5, 5))
        return reshape_output_1


class Reshape606(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1152, 1, 5, 5))
        return reshape_output_1


class Reshape607(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1152, 1, 3, 3))
        return reshape_output_1


class Reshape608(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1280, 1, 1))
        return reshape_output_1


class Reshape609(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 1, 3, 3))
        return reshape_output_1


class Reshape610(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 1, 3, 3))
        return reshape_output_1


class Reshape611(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 1, 3, 3))
        return reshape_output_1


class Reshape612(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(36, 1, 3, 3))
        return reshape_output_1


class Reshape613(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(72, 1, 5, 5))
        return reshape_output_1


class Reshape614(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(20, 1, 3, 3))
        return reshape_output_1


class Reshape615(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(24, 1, 5, 5))
        return reshape_output_1


class Reshape616(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(60, 1, 3, 3))
        return reshape_output_1


class Reshape617(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(120, 1, 3, 3))
        return reshape_output_1


class Reshape618(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(40, 1, 3, 3))
        return reshape_output_1


class Reshape619(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(100, 1, 3, 3))
        return reshape_output_1


class Reshape620(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(92, 1, 3, 3))
        return reshape_output_1


class Reshape621(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(56, 1, 3, 3))
        return reshape_output_1


class Reshape622(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(80, 1, 3, 3))
        return reshape_output_1


class Reshape623(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(112, 1, 5, 5))
        return reshape_output_1


class Reshape624(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 224, 224))
        return reshape_output_1


class Reshape625(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2048, 1, 1))
        return reshape_output_1


class Reshape626(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1536, 1, 1))
        return reshape_output_1


class Reshape627(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1024, 49, 1))
        return reshape_output_1


class Reshape628(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1024, 196, 1))
        return reshape_output_1


class Reshape629(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 512, 49, 1))
        return reshape_output_1


class Reshape630(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 25088))
        return reshape_output_1


class Reshape631(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 25088, 1, 1))
        return reshape_output_1


class Reshape632(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 512, 196, 1))
        return reshape_output_1


class Reshape633(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 768, 49, 1))
        return reshape_output_1


class Reshape634(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 1, 3, 3))
        return reshape_output_1


class Reshape635(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(128, 1, 3, 3))
        return reshape_output_1


class Reshape636(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(256, 1, 3, 3))
        return reshape_output_1


class Reshape637(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(512, 1, 3, 3))
        return reshape_output_1


class Reshape638(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1024, 1, 3, 3))
        return reshape_output_1


class Reshape639(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(384, 1, 3, 3))
        return reshape_output_1


class Reshape640(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(768, 1, 3, 3))
        return reshape_output_1


class Reshape641(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 768, 1, 1))
        return reshape_output_1


class Reshape642(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(576, 1, 3, 3))
        return reshape_output_1


class Reshape643(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(960, 1, 3, 3))
        return reshape_output_1


class Reshape644(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(288, 1, 3, 3))
        return reshape_output_1


class Reshape645(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(432, 1, 3, 3))
        return reshape_output_1


class Reshape646(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(720, 1, 3, 3))
        return reshape_output_1


class Reshape647(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(72, 1, 3, 3))
        return reshape_output_1


class Reshape648(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(88, 1, 3, 3))
        return reshape_output_1


class Reshape649(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(96, 1, 5, 5))
        return reshape_output_1


class Reshape650(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(120, 1, 5, 5))
        return reshape_output_1


class Reshape651(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(288, 1, 5, 5))
        return reshape_output_1


class Reshape652(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(576, 1, 5, 5))
        return reshape_output_1


class Reshape653(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 576, 1, 1))
        return reshape_output_1


class Reshape654(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(200, 1, 3, 3))
        return reshape_output_1


class Reshape655(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(184, 1, 3, 3))
        return reshape_output_1


class Reshape656(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 960, 1, 1))
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
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 512, 1, 322))
        return reshape_output_1


class Reshape675(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 3025, 64))
        return reshape_output_1


class Reshape676(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(3025, 322))
        return reshape_output_1


class Reshape677(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 3025, 1, 322))
        return reshape_output_1


class Reshape678(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 3025, 322))
        return reshape_output_1


class Reshape679(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 512, 3025))
        return reshape_output_1


class Reshape680(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 512, 3025))
        return reshape_output_1


class Reshape681(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 322, 3025))
        return reshape_output_1


class Reshape682(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 512, 1, 261))
        return reshape_output_1


class Reshape683(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 50176, 3))
        return reshape_output_1


class Reshape684(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(50176, 261))
        return reshape_output_1


class Reshape685(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 50176, 1, 261))
        return reshape_output_1


class Reshape686(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 50176, 261))
        return reshape_output_1


class Reshape687(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 261, 50176))
        return reshape_output_1


class Reshape688(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1088, 1, 1))
        return reshape_output_1


class Reshape689(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16384, 1, 64))
        return reshape_output_1


class Reshape690(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 128, 64))
        return reshape_output_1


class Reshape691(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 64, 256))
        return reshape_output_1


class Reshape692(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(256, 64))
        return reshape_output_1


class Reshape693(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 1, 64))
        return reshape_output_1


class Reshape694(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 2, 32))
        return reshape_output_1


class Reshape695(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 64))
        return reshape_output_1


class Reshape696(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 16384, 256))
        return reshape_output_1


class Reshape697(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16384, 256))
        return reshape_output_1


class Reshape698(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 128, 128))
        return reshape_output_1


class Reshape699(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 16384, 1))
        return reshape_output_1


class Reshape700(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4096, 2, 64))
        return reshape_output_1


class Reshape701(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 64, 64, 128))
        return reshape_output_1


class Reshape702(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 4096, 64))
        return reshape_output_1


class Reshape703(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 256))
        return reshape_output_1


class Reshape704(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(256, 128))
        return reshape_output_1


class Reshape705(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 2, 64))
        return reshape_output_1


class Reshape706(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 128))
        return reshape_output_1


class Reshape707(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 256, 64))
        return reshape_output_1


class Reshape708(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2, 4096, 256))
        return reshape_output_1


class Reshape709(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 4096, 256))
        return reshape_output_1


class Reshape710(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 64, 256))
        return reshape_output_1


class Reshape711(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2, 4096, 64))
        return reshape_output_1


class Reshape712(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(4096, 128))
        return reshape_output_1


class Reshape713(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4096, 128))
        return reshape_output_1


class Reshape714(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 512, 64, 64))
        return reshape_output_1


class Reshape715(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 512, 4096, 1))
        return reshape_output_1


class Reshape716(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 320, 1024, 1))
        return reshape_output_1


class Reshape717(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1024, 5, 64))
        return reshape_output_1


class Reshape718(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 32, 320))
        return reshape_output_1


class Reshape719(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(5, 1024, 64))
        return reshape_output_1


class Reshape720(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 320, 32, 32))
        return reshape_output_1


class Reshape721(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 320, 256))
        return reshape_output_1


class Reshape722(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(256, 320))
        return reshape_output_1


class Reshape723(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 5, 64))
        return reshape_output_1


class Reshape724(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 320))
        return reshape_output_1


class Reshape725(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(5, 256, 64))
        return reshape_output_1


class Reshape726(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 5, 1024, 256))
        return reshape_output_1


class Reshape727(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(5, 1024, 256))
        return reshape_output_1


class Reshape728(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(5, 64, 256))
        return reshape_output_1


class Reshape729(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 5, 1024, 64))
        return reshape_output_1


class Reshape730(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1024, 320))
        return reshape_output_1


class Reshape731(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1024, 320))
        return reshape_output_1


class Reshape732(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1280, 32, 32))
        return reshape_output_1


class Reshape733(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1280, 1, 3, 3))
        return reshape_output_1


class Reshape734(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1280, 1024, 1))
        return reshape_output_1


class Reshape735(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 512, 256, 1))
        return reshape_output_1


class Reshape736(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 16, 512))
        return reshape_output_1


class Reshape737(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 256, 64))
        return reshape_output_1


class Reshape738(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 256, 256))
        return reshape_output_1


class Reshape739(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 256, 256))
        return reshape_output_1


class Reshape740(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 64, 256))
        return reshape_output_1


class Reshape741(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 256, 64))
        return reshape_output_1


class Reshape742(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2048, 16, 16))
        return reshape_output_1


class Reshape743(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2048, 1, 3, 3))
        return reshape_output_1


class Reshape744(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2048, 256, 1))
        return reshape_output_1


class Reshape745(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16384, 1, 32))
        return reshape_output_1


class Reshape746(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 128, 32))
        return reshape_output_1


class Reshape747(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 256))
        return reshape_output_1


class Reshape748(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(256, 32))
        return reshape_output_1


class Reshape749(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 1, 32))
        return reshape_output_1


class Reshape750(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 32))
        return reshape_output_1


class Reshape751(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 128, 128))
        return reshape_output_1


class Reshape752(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 16384, 1))
        return reshape_output_1


class Reshape753(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 64, 4096, 1))
        return reshape_output_1


class Reshape754(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4096, 2, 32))
        return reshape_output_1


class Reshape755(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 64, 64, 64))
        return reshape_output_1


class Reshape756(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 4096, 32))
        return reshape_output_1


class Reshape757(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 256, 32))
        return reshape_output_1


class Reshape758(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 32, 256))
        return reshape_output_1


class Reshape759(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2, 4096, 32))
        return reshape_output_1


class Reshape760(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(4096, 64))
        return reshape_output_1


class Reshape761(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4096, 64))
        return reshape_output_1


class Reshape762(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 4096, 1))
        return reshape_output_1


class Reshape763(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 160, 1024, 1))
        return reshape_output_1


class Reshape764(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1024, 5, 32))
        return reshape_output_1


class Reshape765(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 32, 160))
        return reshape_output_1


class Reshape766(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(5, 1024, 32))
        return reshape_output_1


class Reshape767(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 160, 32, 32))
        return reshape_output_1


class Reshape768(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 160, 256))
        return reshape_output_1


class Reshape769(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(256, 160))
        return reshape_output_1


class Reshape770(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 5, 32))
        return reshape_output_1


class Reshape771(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 160))
        return reshape_output_1


class Reshape772(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(5, 256, 32))
        return reshape_output_1


class Reshape773(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(5, 32, 256))
        return reshape_output_1


class Reshape774(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 5, 1024, 32))
        return reshape_output_1


class Reshape775(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1024, 160))
        return reshape_output_1


class Reshape776(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1024, 160))
        return reshape_output_1


class Reshape777(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 640, 32, 32))
        return reshape_output_1


class Reshape778(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(640, 1, 3, 3))
        return reshape_output_1


class Reshape779(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 640, 1024, 1))
        return reshape_output_1


class Reshape780(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 256, 1))
        return reshape_output_1


class Reshape781(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(256, 256))
        return reshape_output_1


class Reshape782(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 8, 32))
        return reshape_output_1


class Reshape783(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 16, 256))
        return reshape_output_1


class Reshape784(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 16, 16))
        return reshape_output_1


class Reshape785(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 256))
        return reshape_output_1


class Reshape786(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 256, 32))
        return reshape_output_1


class Reshape787(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 32, 256))
        return reshape_output_1


class Reshape788(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 256, 32))
        return reshape_output_1


class Reshape789(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1024, 16, 16))
        return reshape_output_1


class Reshape790(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1024, 256, 1))
        return reshape_output_1


class Reshape791(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 768, 16, 16))
        return reshape_output_1


class Reshape792(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 768, 32, 32))
        return reshape_output_1


class Reshape793(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 768, 64, 64))
        return reshape_output_1


class Reshape794(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 768, 128, 128))
        return reshape_output_1


class Reshape795(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 5776))
        return reshape_output_1


class Reshape796(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 2166))
        return reshape_output_1


class Reshape797(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 600))
        return reshape_output_1


class Reshape798(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 150))
        return reshape_output_1


class Reshape799(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 36))
        return reshape_output_1


class Reshape800(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 81, 5776))
        return reshape_output_1


class Reshape801(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 81, 2166))
        return reshape_output_1


class Reshape802(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 81, 600))
        return reshape_output_1


class Reshape803(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 81, 150))
        return reshape_output_1


class Reshape804(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 81, 36))
        return reshape_output_1


class Reshape805(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 81, 4))
        return reshape_output_1


class Reshape806(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 96, 4096, 1))
        return reshape_output_1


class Reshape807(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 8, 8, 8, 96))
        return reshape_output_1


class Reshape808(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 64, 64, 96))
        return reshape_output_1


class Reshape809(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(4096, 96))
        return reshape_output_1


class Reshape810(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4096, 96))
        return reshape_output_1


class Reshape811(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 64, 96))
        return reshape_output_1


class Reshape812(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 64, 3, 32))
        return reshape_output_1


class Reshape813(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(192, 64, 32))
        return reshape_output_1


class Reshape814(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 3, 64, 64))
        return reshape_output_1


class Reshape815(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(225, 2))
        return reshape_output_1


class Reshape816(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 15, 15, 512))
        return reshape_output_1


class Reshape817(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(225, 512))
        return reshape_output_1


class Reshape818(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(225, 3))
        return reshape_output_1


class Reshape819(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(4096,))
        return reshape_output_1


class Reshape820(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 64, 3))
        return reshape_output_1


class Reshape821(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(192, 64, 64))
        return reshape_output_1


class Reshape822(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 64, 3, 64, 64))
        return reshape_output_1


class Reshape823(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(192, 32, 64))
        return reshape_output_1


class Reshape824(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 3, 64, 32))
        return reshape_output_1


class Reshape825(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1024, 384))
        return reshape_output_1


class Reshape826(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1024, 192))
        return reshape_output_1


class Reshape827(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 64, 192))
        return reshape_output_1


class Reshape828(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 64, 6, 32))
        return reshape_output_1


class Reshape829(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 8, 4, 8, 192))
        return reshape_output_1


class Reshape830(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 32, 192))
        return reshape_output_1


class Reshape831(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1024, 192))
        return reshape_output_1


class Reshape832(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 4, 8, 8, 192))
        return reshape_output_1


class Reshape833(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(96, 64, 32))
        return reshape_output_1


class Reshape834(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 6, 64, 64))
        return reshape_output_1


class Reshape835(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(225, 6))
        return reshape_output_1


class Reshape836(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 64, 6))
        return reshape_output_1


class Reshape837(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(96, 64, 64))
        return reshape_output_1


class Reshape838(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 6, 64, 64))
        return reshape_output_1


class Reshape839(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(96, 32, 64))
        return reshape_output_1


class Reshape840(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 6, 64, 32))
        return reshape_output_1


class Reshape841(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 384))
        return reshape_output_1


class Reshape842(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(4, 64, 384))
        return reshape_output_1


class Reshape843(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(4, 64, 12, 32))
        return reshape_output_1


class Reshape844(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2, 8, 2, 8, 384))
        return reshape_output_1


class Reshape845(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 16, 384))
        return reshape_output_1


class Reshape846(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(256, 384))
        return reshape_output_1


class Reshape847(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2, 2, 8, 8, 384))
        return reshape_output_1


class Reshape848(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(48, 64, 32))
        return reshape_output_1


class Reshape849(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(4, 12, 64, 64))
        return reshape_output_1


class Reshape850(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(225, 12))
        return reshape_output_1


class Reshape851(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 64, 12))
        return reshape_output_1


class Reshape852(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(48, 64, 64))
        return reshape_output_1


class Reshape853(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 12, 64, 64))
        return reshape_output_1


class Reshape854(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(48, 32, 64))
        return reshape_output_1


class Reshape855(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(4, 12, 64, 32))
        return reshape_output_1


class Reshape856(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 1536))
        return reshape_output_1


class Reshape857(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 64, 768))
        return reshape_output_1


class Reshape858(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 64, 24, 32))
        return reshape_output_1


class Reshape859(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 768))
        return reshape_output_1


class Reshape860(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(24, 64, 32))
        return reshape_output_1


class Reshape861(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 24, 64, 64))
        return reshape_output_1


class Reshape862(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(225, 24))
        return reshape_output_1


class Reshape863(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 64, 24))
        return reshape_output_1


class Reshape864(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(24, 64, 64))
        return reshape_output_1


class Reshape865(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(24, 32, 64))
        return reshape_output_1


class Reshape866(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 24, 64, 32))
        return reshape_output_1


class Reshape867(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 768, 8, 8))
        return reshape_output_1


class Reshape868(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4096, 1, 1))
        return reshape_output_1


class Reshape869(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(197, 1024))
        return reshape_output_1


class Reshape870(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 197, 16, 64))
        return reshape_output_1


class Reshape871(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 197, 1024))
        return reshape_output_1


class Reshape872(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 197, 64))
        return reshape_output_1


class Reshape873(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 197, 197))
        return reshape_output_1


class Reshape874(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 197, 197))
        return reshape_output_1


class Reshape875(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 64, 197))
        return reshape_output_1


class Reshape876(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 197, 64))
        return reshape_output_1


class Reshape877(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(160, 1, 3, 3))
        return reshape_output_1


class Reshape878(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(224, 1, 3, 3))
        return reshape_output_1


class Reshape879(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(728, 1, 3, 3))
        return reshape_output_1


class Reshape880(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1536, 1, 3, 3))
        return reshape_output_1


class Reshape881(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 255, 80, 80))
        return reshape_output_1


class Reshape882(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 255, 6400))
        return reshape_output_1


class Reshape883(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 3, 85, 6400))
        return reshape_output_1


class Reshape884(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 19200, 85))
        return reshape_output_1


class Reshape885(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 255, 40, 40))
        return reshape_output_1


class Reshape886(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 255, 1600))
        return reshape_output_1


class Reshape887(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 3, 85, 1600))
        return reshape_output_1


class Reshape888(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4800, 85))
        return reshape_output_1


class Reshape889(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 255, 20, 20))
        return reshape_output_1


class Reshape890(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 255, 400))
        return reshape_output_1


class Reshape891(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 3, 85, 400))
        return reshape_output_1


class Reshape892(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1200, 85))
        return reshape_output_1


class Reshape893(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 255, 10, 10))
        return reshape_output_1


class Reshape894(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 255, 100))
        return reshape_output_1


class Reshape895(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 3, 85, 100))
        return reshape_output_1


class Reshape896(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 300, 85))
        return reshape_output_1


class Reshape897(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 255, 160, 160))
        return reshape_output_1


class Reshape898(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 255, 25600))
        return reshape_output_1


class Reshape899(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 3, 85, 25600))
        return reshape_output_1


class Reshape900(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 76800, 85))
        return reshape_output_1


class Reshape901(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 255, 60, 60))
        return reshape_output_1


class Reshape902(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 255, 3600))
        return reshape_output_1


class Reshape903(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 3, 85, 3600))
        return reshape_output_1


class Reshape904(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 10800, 85))
        return reshape_output_1


class Reshape905(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 255, 30, 30))
        return reshape_output_1


class Reshape906(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 255, 900))
        return reshape_output_1


class Reshape907(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 3, 85, 900))
        return reshape_output_1


class Reshape908(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2700, 85))
        return reshape_output_1


class Reshape909(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 255, 15, 15))
        return reshape_output_1


class Reshape910(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 255, 225))
        return reshape_output_1


class Reshape911(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 3, 85, 225))
        return reshape_output_1


class Reshape912(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 675, 85))
        return reshape_output_1


class Reshape913(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 17, 4480))
        return reshape_output_1


class Reshape914(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 4480))
        return reshape_output_1


class Reshape915(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 17, 1120))
        return reshape_output_1


class Reshape916(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 1120))
        return reshape_output_1


class Reshape917(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 17, 280))
        return reshape_output_1


class Reshape918(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 280))
        return reshape_output_1


class Reshape919(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 80, 4480))
        return reshape_output_1


class Reshape920(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 80, 1120))
        return reshape_output_1


class Reshape921(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 80, 280))
        return reshape_output_1


class Reshape922(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 85, 6400, 1))
        return reshape_output_1


class Reshape923(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 85, 1600, 1))
        return reshape_output_1


class Reshape924(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 85, 400, 1))
        return reshape_output_1


class Reshape925(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 85, 2704, 1))
        return reshape_output_1


class Reshape926(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 85, 676, 1))
        return reshape_output_1


class Reshape927(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 85, 169, 1))
        return reshape_output_1


def ids_func(param):
    forge_module, shapes_dtypes, _ = param
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (
        Reshape0,
        [((8, 1), torch.float32)],
        {"model_name": ["pt_musicgen_large", "pt_musicgen_small", "pt_musicgen_medium"]},
    ),
    (
        Reshape1,
        [((2, 1, 1), torch.float32)],
        {"model_name": ["pt_musicgen_large", "pt_musicgen_small", "pt_musicgen_medium"]},
    ),
    (Reshape2, [((1, 2048), torch.float32)], {"model_name": ["pt_musicgen_large"]}),
    (Reshape3, [((1, 2048), torch.float32)], {"model_name": ["pt_google_flan_t5_base"]}),
    (Reshape4, [((2, 1, 2048), torch.float32)], {"model_name": ["pt_musicgen_large"]}),
    (Reshape5, [((2, 1, 2048), torch.float32)], {"model_name": ["pt_musicgen_large"]}),
    (
        Reshape6,
        [((2, 2048), torch.float32)],
        {"model_name": ["pt_musicgen_large", "pt_musicgen_small", "pt_musicgen_medium"]},
    ),
    (Reshape5, [((2, 2048), torch.float32)], {"model_name": ["pt_musicgen_large"]}),
    (Reshape7, [((2, 32, 1, 64), torch.float32)], {"model_name": ["pt_musicgen_large"]}),
    (Reshape8, [((64, 1, 64), torch.float32)], {"model_name": ["pt_musicgen_large"]}),
    (Reshape4, [((2, 1, 32, 64), torch.float32)], {"model_name": ["pt_musicgen_large"]}),
    (
        Reshape9,
        [((2, 13), torch.float32)],
        {"model_name": ["pt_musicgen_large", "pt_musicgen_small", "pt_musicgen_medium"]},
    ),
    (
        Reshape10,
        [((2, 13, 768), torch.float32)],
        {"model_name": ["pt_musicgen_large", "pt_musicgen_small", "pt_musicgen_medium"]},
    ),
    (
        Reshape11,
        [((26, 768), torch.float32)],
        {"model_name": ["pt_musicgen_large", "pt_musicgen_small", "pt_musicgen_medium"]},
    ),
    (
        Reshape12,
        [((26, 768), torch.float32)],
        {"model_name": ["pt_musicgen_large", "pt_musicgen_small", "pt_musicgen_medium"]},
    ),
    (
        Reshape13,
        [((2, 12, 13, 64), torch.float32)],
        {"model_name": ["pt_musicgen_large", "pt_musicgen_small", "pt_musicgen_medium"]},
    ),
    (
        Reshape14,
        [((24, 13, 13), torch.float32)],
        {"model_name": ["pt_musicgen_large", "pt_musicgen_small", "pt_musicgen_medium"]},
    ),
    (
        Reshape15,
        [((2, 12, 13, 13), torch.float32)],
        {"model_name": ["pt_musicgen_large", "pt_musicgen_small", "pt_musicgen_medium"]},
    ),
    (
        Reshape16,
        [((2, 12, 64, 13), torch.float32)],
        {"model_name": ["pt_musicgen_large", "pt_musicgen_small", "pt_musicgen_medium"]},
    ),
    (
        Reshape17,
        [((24, 13, 64), torch.float32)],
        {"model_name": ["pt_musicgen_large", "pt_musicgen_small", "pt_musicgen_medium"]},
    ),
    (
        Reshape10,
        [((2, 13, 12, 64), torch.float32)],
        {"model_name": ["pt_musicgen_large", "pt_musicgen_small", "pt_musicgen_medium"]},
    ),
    (
        Reshape18,
        [((26, 3072), torch.float32)],
        {"model_name": ["pt_musicgen_large", "pt_musicgen_small", "pt_musicgen_medium"]},
    ),
    (
        Reshape19,
        [((2, 13, 3072), torch.float32)],
        {"model_name": ["pt_musicgen_large", "pt_musicgen_small", "pt_musicgen_medium"]},
    ),
    (Reshape20, [((26, 2048), torch.float32)], {"model_name": ["pt_musicgen_large"]}),
    (Reshape21, [((26, 2048), torch.float32)], {"model_name": ["pt_musicgen_large"]}),
    (Reshape22, [((2, 13, 2048), torch.float32)], {"model_name": ["pt_musicgen_large"]}),
    (Reshape23, [((2, 32, 13, 64), torch.float32)], {"model_name": ["pt_musicgen_large"]}),
    (Reshape24, [((64, 1, 13), torch.float32)], {"model_name": ["pt_musicgen_large"]}),
    (Reshape25, [((2, 32, 1, 13), torch.float32)], {"model_name": ["pt_musicgen_large"]}),
    (Reshape26, [((2, 8192), torch.float32)], {"model_name": ["pt_musicgen_large"]}),
    (Reshape27, [((2, 1, 8192), torch.float32)], {"model_name": ["pt_musicgen_large"]}),
    (
        Reshape28,
        [((2, 4, 1, 2048), torch.float32)],
        {"model_name": ["pt_musicgen_large", "pt_musicgen_small", "pt_musicgen_medium"]},
    ),
    (Reshape29, [((1, 1024), torch.float32)], {"model_name": ["pt_musicgen_small"]}),
    (Reshape30, [((1, 1024), torch.float32)], {"model_name": ["pt_t5_large"]}),
    (Reshape31, [((1, 1024), torch.float32)], {"model_name": ["pt_t5_large", "pt_google_flan_t5_small"]}),
    (Reshape32, [((2, 1, 1024), torch.float32)], {"model_name": ["pt_musicgen_small"]}),
    (Reshape33, [((2, 1, 1024), torch.float32)], {"model_name": ["pt_musicgen_small"]}),
    (Reshape34, [((2, 1024), torch.float32)], {"model_name": ["pt_musicgen_small"]}),
    (Reshape33, [((2, 1024), torch.float32)], {"model_name": ["pt_musicgen_small"]}),
    (Reshape35, [((2, 1024), torch.float32)], {"model_name": ["pt_whisper_medium"]}),
    (Reshape36, [((2, 1024), torch.float32)], {"model_name": ["pt_whisper_medium"]}),
    (Reshape37, [((2, 16, 1, 64), torch.float32)], {"model_name": ["pt_musicgen_small"]}),
    (Reshape38, [((32, 1, 64), torch.float32)], {"model_name": ["pt_musicgen_small"]}),
    (Reshape32, [((2, 1, 16, 64), torch.float32)], {"model_name": ["pt_musicgen_small"]}),
    (Reshape39, [((26, 1024), torch.float32)], {"model_name": ["pt_musicgen_small"]}),
    (Reshape40, [((26, 1024), torch.float32)], {"model_name": ["pt_musicgen_small"]}),
    (Reshape41, [((2, 13, 1024), torch.float32)], {"model_name": ["pt_musicgen_small"]}),
    (Reshape42, [((2, 16, 13, 64), torch.float32)], {"model_name": ["pt_musicgen_small"]}),
    (Reshape43, [((32, 1, 13), torch.float32)], {"model_name": ["pt_musicgen_small"]}),
    (Reshape44, [((2, 16, 1, 13), torch.float32)], {"model_name": ["pt_musicgen_small"]}),
    (Reshape45, [((2, 4096), torch.float32)], {"model_name": ["pt_musicgen_small"]}),
    (Reshape46, [((2, 1, 4096), torch.float32)], {"model_name": ["pt_musicgen_small"]}),
    (Reshape47, [((1, 1536), torch.float32)], {"model_name": ["pt_musicgen_medium"]}),
    (Reshape48, [((2, 1, 1536), torch.float32)], {"model_name": ["pt_musicgen_medium"]}),
    (Reshape49, [((2, 1, 1536), torch.float32)], {"model_name": ["pt_musicgen_medium"]}),
    (Reshape50, [((2, 1536), torch.float32)], {"model_name": ["pt_musicgen_medium"]}),
    (Reshape49, [((2, 1536), torch.float32)], {"model_name": ["pt_musicgen_medium"]}),
    (Reshape51, [((2, 24, 1, 64), torch.float32)], {"model_name": ["pt_musicgen_medium"]}),
    (Reshape52, [((48, 1, 64), torch.float32)], {"model_name": ["pt_musicgen_medium"]}),
    (Reshape48, [((2, 1, 24, 64), torch.float32)], {"model_name": ["pt_musicgen_medium"]}),
    (Reshape53, [((26, 1536), torch.float32)], {"model_name": ["pt_musicgen_medium"]}),
    (Reshape54, [((26, 1536), torch.float32)], {"model_name": ["pt_musicgen_medium"]}),
    (Reshape55, [((2, 13, 1536), torch.float32)], {"model_name": ["pt_musicgen_medium"]}),
    (Reshape56, [((2, 24, 13, 64), torch.float32)], {"model_name": ["pt_musicgen_medium"]}),
    (Reshape57, [((48, 1, 13), torch.float32)], {"model_name": ["pt_musicgen_medium"]}),
    (Reshape58, [((2, 24, 1, 13), torch.float32)], {"model_name": ["pt_musicgen_medium"]}),
    (Reshape59, [((2, 6144), torch.float32)], {"model_name": ["pt_musicgen_medium"]}),
    (Reshape60, [((2, 1, 6144), torch.float32)], {"model_name": ["pt_musicgen_medium"]}),
    (
        Reshape61,
        [((1, 2), torch.int32)],
        {
            "model_name": [
                "pt_whisper_small",
                "pt_whisper_large",
                "pt_whisper_medium",
                "pt_whisper_tiny",
                "pt_whisper_base",
                "pt_whisper_large_v3_turbo",
            ]
        },
    ),
    (Reshape62, [((1, 2, 768), torch.float32)], {"model_name": ["pt_whisper_small"]}),
    (Reshape63, [((1, 2, 768), torch.float32)], {"model_name": ["pt_whisper_small"]}),
    (Reshape64, [((2, 768), torch.float32)], {"model_name": ["pt_whisper_small"]}),
    (Reshape63, [((2, 768), torch.float32)], {"model_name": ["pt_whisper_small"]}),
    (Reshape65, [((1, 12, 2, 64), torch.float32)], {"model_name": ["pt_whisper_small"]}),
    (Reshape66, [((12, 2, 2), torch.float32)], {"model_name": ["pt_whisper_small"]}),
    (Reshape67, [((1, 12, 2, 2), torch.float32)], {"model_name": ["pt_whisper_small"]}),
    (Reshape68, [((1, 12, 64, 2), torch.float32)], {"model_name": ["pt_whisper_small"]}),
    (Reshape69, [((12, 2, 64), torch.float32)], {"model_name": ["pt_whisper_small"]}),
    (Reshape62, [((1, 2, 12, 64), torch.float32)], {"model_name": ["pt_whisper_small"]}),
    (Reshape70, [((1, 1500, 768), torch.float32)], {"model_name": ["pt_whisper_small"]}),
    (Reshape71, [((1, 1500, 768), torch.float32)], {"model_name": ["pt_whisper_small"]}),
    (Reshape71, [((1500, 768), torch.float32)], {"model_name": ["pt_whisper_small"]}),
    (Reshape72, [((1500, 768), torch.float32)], {"model_name": ["pt_whisper_small"]}),
    (Reshape73, [((1, 12, 1500, 64), torch.float32)], {"model_name": ["pt_whisper_small"]}),
    (Reshape74, [((12, 2, 1500), torch.float32)], {"model_name": ["pt_whisper_small"]}),
    (Reshape75, [((1, 12, 2, 1500), torch.float32)], {"model_name": ["pt_whisper_small"]}),
    (Reshape76, [((1, 12, 64, 1500), torch.float32)], {"model_name": ["pt_whisper_small"]}),
    (Reshape77, [((1, 2, 1280), torch.float32)], {"model_name": ["pt_whisper_large", "pt_whisper_large_v3_turbo"]}),
    (Reshape78, [((1, 2, 1280), torch.float32)], {"model_name": ["pt_whisper_large", "pt_whisper_large_v3_turbo"]}),
    (Reshape79, [((2, 1280), torch.float32)], {"model_name": ["pt_whisper_large", "pt_whisper_large_v3_turbo"]}),
    (Reshape78, [((2, 1280), torch.float32)], {"model_name": ["pt_whisper_large", "pt_whisper_large_v3_turbo"]}),
    (Reshape80, [((1, 20, 2, 64), torch.float32)], {"model_name": ["pt_whisper_large", "pt_whisper_large_v3_turbo"]}),
    (Reshape81, [((20, 2, 2), torch.float32)], {"model_name": ["pt_whisper_large", "pt_whisper_large_v3_turbo"]}),
    (Reshape82, [((1, 20, 2, 2), torch.float32)], {"model_name": ["pt_whisper_large", "pt_whisper_large_v3_turbo"]}),
    (Reshape83, [((1, 20, 64, 2), torch.float32)], {"model_name": ["pt_whisper_large", "pt_whisper_large_v3_turbo"]}),
    (Reshape84, [((20, 2, 64), torch.float32)], {"model_name": ["pt_whisper_large", "pt_whisper_large_v3_turbo"]}),
    (Reshape77, [((1, 2, 20, 64), torch.float32)], {"model_name": ["pt_whisper_large", "pt_whisper_large_v3_turbo"]}),
    (Reshape85, [((1, 1500, 1280), torch.float32)], {"model_name": ["pt_whisper_large", "pt_whisper_large_v3_turbo"]}),
    (Reshape86, [((1, 1500, 1280), torch.float32)], {"model_name": ["pt_whisper_large", "pt_whisper_large_v3_turbo"]}),
    (Reshape86, [((1500, 1280), torch.float32)], {"model_name": ["pt_whisper_large", "pt_whisper_large_v3_turbo"]}),
    (Reshape87, [((1500, 1280), torch.float32)], {"model_name": ["pt_whisper_large", "pt_whisper_large_v3_turbo"]}),
    (
        Reshape88,
        [((1, 20, 1500, 64), torch.float32)],
        {"model_name": ["pt_whisper_large", "pt_whisper_large_v3_turbo"]},
    ),
    (Reshape89, [((20, 2, 1500), torch.float32)], {"model_name": ["pt_whisper_large", "pt_whisper_large_v3_turbo"]}),
    (Reshape90, [((1, 20, 2, 1500), torch.float32)], {"model_name": ["pt_whisper_large", "pt_whisper_large_v3_turbo"]}),
    (
        Reshape91,
        [((1, 20, 64, 1500), torch.float32)],
        {"model_name": ["pt_whisper_large", "pt_whisper_large_v3_turbo"]},
    ),
    (Reshape32, [((1, 2, 1024), torch.float32)], {"model_name": ["pt_whisper_medium"]}),
    (Reshape36, [((1, 2, 1024), torch.float32)], {"model_name": ["pt_whisper_medium"]}),
    (Reshape92, [((1, 16, 2, 64), torch.float32)], {"model_name": ["pt_whisper_medium"]}),
    (Reshape93, [((16, 2, 2), torch.float32)], {"model_name": ["pt_whisper_medium"]}),
    (Reshape94, [((1, 16, 2, 2), torch.float32)], {"model_name": ["pt_whisper_medium"]}),
    (Reshape95, [((1, 16, 64, 2), torch.float32)], {"model_name": ["pt_whisper_medium"]}),
    (Reshape96, [((16, 2, 64), torch.float32)], {"model_name": ["pt_whisper_medium"]}),
    (Reshape32, [((1, 2, 16, 64), torch.float32)], {"model_name": ["pt_whisper_medium"]}),
    (Reshape97, [((1, 1500, 1024), torch.float32)], {"model_name": ["pt_whisper_medium"]}),
    (Reshape98, [((1, 1500, 1024), torch.float32)], {"model_name": ["pt_whisper_medium"]}),
    (Reshape98, [((1500, 1024), torch.float32)], {"model_name": ["pt_whisper_medium"]}),
    (Reshape99, [((1500, 1024), torch.float32)], {"model_name": ["pt_whisper_medium"]}),
    (Reshape100, [((1, 16, 1500, 64), torch.float32)], {"model_name": ["pt_whisper_medium"]}),
    (Reshape101, [((16, 2, 1500), torch.float32)], {"model_name": ["pt_whisper_medium"]}),
    (Reshape102, [((1, 16, 2, 1500), torch.float32)], {"model_name": ["pt_whisper_medium"]}),
    (Reshape103, [((1, 16, 64, 1500), torch.float32)], {"model_name": ["pt_whisper_medium"]}),
    (Reshape104, [((1, 2, 384), torch.float32)], {"model_name": ["pt_whisper_tiny"]}),
    (Reshape105, [((1, 2, 384), torch.float32)], {"model_name": ["pt_whisper_tiny"]}),
    (Reshape106, [((2, 384), torch.float32)], {"model_name": ["pt_whisper_tiny"]}),
    (Reshape105, [((2, 384), torch.float32)], {"model_name": ["pt_whisper_tiny"]}),
    (Reshape107, [((1, 6, 2, 64), torch.float32)], {"model_name": ["pt_whisper_tiny"]}),
    (Reshape108, [((6, 2, 2), torch.float32)], {"model_name": ["pt_whisper_tiny"]}),
    (Reshape109, [((1, 6, 2, 2), torch.float32)], {"model_name": ["pt_whisper_tiny"]}),
    (Reshape110, [((1, 6, 64, 2), torch.float32)], {"model_name": ["pt_whisper_tiny"]}),
    (Reshape111, [((6, 2, 64), torch.float32)], {"model_name": ["pt_whisper_tiny"]}),
    (Reshape104, [((1, 2, 6, 64), torch.float32)], {"model_name": ["pt_whisper_tiny"]}),
    (Reshape112, [((1, 1500, 384), torch.float32)], {"model_name": ["pt_whisper_tiny"]}),
    (Reshape113, [((1, 1500, 384), torch.float32)], {"model_name": ["pt_whisper_tiny"]}),
    (Reshape113, [((1500, 384), torch.float32)], {"model_name": ["pt_whisper_tiny"]}),
    (Reshape114, [((1500, 384), torch.float32)], {"model_name": ["pt_whisper_tiny"]}),
    (Reshape115, [((1, 6, 1500, 64), torch.float32)], {"model_name": ["pt_whisper_tiny"]}),
    (Reshape116, [((6, 2, 1500), torch.float32)], {"model_name": ["pt_whisper_tiny"]}),
    (Reshape117, [((1, 6, 2, 1500), torch.float32)], {"model_name": ["pt_whisper_tiny"]}),
    (Reshape118, [((1, 6, 64, 1500), torch.float32)], {"model_name": ["pt_whisper_tiny"]}),
    (Reshape119, [((1, 2, 512), torch.float32)], {"model_name": ["pt_whisper_base"]}),
    (Reshape120, [((1, 2, 512), torch.float32)], {"model_name": ["pt_whisper_base"]}),
    (Reshape121, [((2, 512), torch.float32)], {"model_name": ["pt_whisper_base"]}),
    (Reshape120, [((2, 512), torch.float32)], {"model_name": ["pt_whisper_base"]}),
    (Reshape122, [((1, 8, 2, 64), torch.float32)], {"model_name": ["pt_whisper_base"]}),
    (Reshape123, [((8, 2, 2), torch.float32)], {"model_name": ["pt_whisper_base"]}),
    (Reshape124, [((1, 8, 2, 2), torch.float32)], {"model_name": ["pt_whisper_base"]}),
    (Reshape125, [((1, 8, 64, 2), torch.float32)], {"model_name": ["pt_whisper_base"]}),
    (Reshape126, [((8, 2, 64), torch.float32)], {"model_name": ["pt_whisper_base"]}),
    (Reshape119, [((1, 2, 8, 64), torch.float32)], {"model_name": ["pt_whisper_base"]}),
    (Reshape127, [((1, 1500, 512), torch.float32)], {"model_name": ["pt_whisper_base"]}),
    (Reshape128, [((1, 1500, 512), torch.float32)], {"model_name": ["pt_whisper_base"]}),
    (Reshape128, [((1500, 512), torch.float32)], {"model_name": ["pt_whisper_base"]}),
    (Reshape129, [((1500, 512), torch.float32)], {"model_name": ["pt_whisper_base"]}),
    (Reshape130, [((1, 8, 1500, 64), torch.float32)], {"model_name": ["pt_whisper_base"]}),
    (Reshape131, [((8, 2, 1500), torch.float32)], {"model_name": ["pt_whisper_base"]}),
    (Reshape132, [((1, 8, 2, 1500), torch.float32)], {"model_name": ["pt_whisper_base"]}),
    (Reshape133, [((1, 8, 64, 1500), torch.float32)], {"model_name": ["pt_whisper_base"]}),
    (Reshape134, [((2, 7), torch.float32)], {"model_name": ["pt_clip_vit_base_patch32_text"]}),
    (Reshape135, [((2, 7, 512), torch.float32)], {"model_name": ["pt_clip_vit_base_patch32_text"]}),
    (Reshape136, [((2, 7, 512), torch.float32)], {"model_name": ["pt_clip_vit_base_patch32_text"]}),
    (Reshape137, [((14, 512), torch.float32)], {"model_name": ["pt_clip_vit_base_patch32_text"]}),
    (Reshape138, [((2, 8, 7, 64), torch.float32)], {"model_name": ["pt_clip_vit_base_patch32_text"]}),
    (Reshape139, [((16, 7, 7), torch.float32)], {"model_name": ["pt_clip_vit_base_patch32_text"]}),
    (Reshape139, [((2, 8, 7, 7), torch.float32)], {"model_name": ["pt_clip_vit_base_patch32_text"]}),
    (Reshape140, [((2, 8, 7, 7), torch.float32)], {"model_name": ["pt_clip_vit_base_patch32_text"]}),
    (Reshape141, [((16, 7, 64), torch.float32)], {"model_name": ["pt_clip_vit_base_patch32_text"]}),
    (Reshape135, [((2, 7, 8, 64), torch.float32)], {"model_name": ["pt_clip_vit_base_patch32_text"]}),
    (Reshape142, [((14, 2048), torch.float32)], {"model_name": ["pt_clip_vit_base_patch32_text"]}),
    (Reshape143, [((2, 7, 2048), torch.float32)], {"model_name": ["pt_clip_vit_base_patch32_text"]}),
    (Reshape144, [((1, 204, 768), torch.float32)], {"model_name": ["pt_ViLt_maskedlm"]}),
    (Reshape145, [((1, 204, 768), torch.float32)], {"model_name": ["pt_ViLt_maskedlm"]}),
    (Reshape146, [((204, 768), torch.float32)], {"model_name": ["pt_ViLt_maskedlm"]}),
    (Reshape147, [((1, 12, 204, 64), torch.float32)], {"model_name": ["pt_ViLt_maskedlm"]}),
    (Reshape148, [((12, 204, 204), torch.float32)], {"model_name": ["pt_ViLt_maskedlm"]}),
    (Reshape149, [((1, 12, 204, 204), torch.float32)], {"model_name": ["pt_ViLt_maskedlm"]}),
    (Reshape150, [((1, 12, 64, 204), torch.float32)], {"model_name": ["pt_ViLt_maskedlm"]}),
    (Reshape151, [((12, 204, 64), torch.float32)], {"model_name": ["pt_ViLt_maskedlm"]}),
    (Reshape144, [((1, 204, 12, 64), torch.float32)], {"model_name": ["pt_ViLt_maskedlm"]}),
    (Reshape152, [((1, 201, 768), torch.float32)], {"model_name": ["pt_ViLt_question_answering"]}),
    (Reshape153, [((1, 201, 768), torch.float32)], {"model_name": ["pt_ViLt_question_answering"]}),
    (Reshape154, [((201, 768), torch.float32)], {"model_name": ["pt_ViLt_question_answering"]}),
    (Reshape155, [((1, 12, 201, 64), torch.float32)], {"model_name": ["pt_ViLt_question_answering"]}),
    (Reshape156, [((12, 201, 201), torch.float32)], {"model_name": ["pt_ViLt_question_answering"]}),
    (Reshape157, [((1, 12, 201, 201), torch.float32)], {"model_name": ["pt_ViLt_question_answering"]}),
    (Reshape158, [((1, 12, 64, 201), torch.float32)], {"model_name": ["pt_ViLt_question_answering"]}),
    (Reshape159, [((12, 201, 64), torch.float32)], {"model_name": ["pt_ViLt_question_answering"]}),
    (Reshape152, [((1, 201, 12, 64), torch.float32)], {"model_name": ["pt_ViLt_question_answering"]}),
    (
        Reshape160,
        [((1, 1, 768), torch.float32)],
        {
            "model_name": [
                "pt_ViLt_question_answering",
                "pt_distilbert_sequence_classification",
                "pt_dpr_ctx_encoder_single_nq_base",
                "pt_dpr_reader_single_nq_base",
                "pt_dpr_reader_multiset_base",
                "pt_dpr_question_encoder_single_nq_base",
                "pt_dpr_ctx_encoder_multiset_base",
                "pt_dpr_question_encoder_multiset_base",
                "pt_roberta_sentiment",
                "pt_squeezebert",
                "pt_t5_base",
                "pt_google_flan_t5_base",
                "pt_deit_base_patch16_224",
                "pt_deit_base_distilled_patch16_224",
                "pt_vit_base_patch16_224",
            ]
        },
    ),
    (
        Reshape161,
        [((1, 128, 768), torch.float32)],
        {
            "model_name": [
                "pt_albert_base_v2_masked_lm",
                "pt_albert_base_v1_masked_lm",
                "pt_albert_base_v1_token_cls",
                "pt_albert_base_v2_token_cls",
                "pt_bert_masked_lm",
                "pt_distilbert_masked_lm",
                "pt_distilbert_sequence_classification",
                "pt_distilbert_token_classification",
                "pt_dpr_ctx_encoder_single_nq_base",
                "pt_dpr_reader_single_nq_base",
                "pt_dpr_reader_multiset_base",
                "pt_dpr_question_encoder_single_nq_base",
                "pt_dpr_ctx_encoder_multiset_base",
                "pt_dpr_question_encoder_multiset_base",
                "pt_roberta_masked_lm",
                "pt_roberta_sentiment",
            ]
        },
    ),
    (
        Reshape162,
        [((1, 128, 768), torch.float32)],
        {
            "model_name": [
                "pt_albert_base_v2_masked_lm",
                "pt_albert_base_v1_masked_lm",
                "pt_albert_base_v1_token_cls",
                "pt_albert_base_v2_token_cls",
                "pt_bert_masked_lm",
                "pt_distilbert_masked_lm",
                "pt_distilbert_sequence_classification",
                "pt_distilbert_token_classification",
                "pt_dpr_ctx_encoder_single_nq_base",
                "pt_dpr_reader_single_nq_base",
                "pt_dpr_reader_multiset_base",
                "pt_dpr_question_encoder_single_nq_base",
                "pt_dpr_ctx_encoder_multiset_base",
                "pt_dpr_question_encoder_multiset_base",
                "pt_roberta_masked_lm",
                "pt_roberta_sentiment",
            ]
        },
    ),
    (
        Reshape163,
        [((128, 768), torch.float32)],
        {
            "model_name": [
                "pt_albert_base_v2_masked_lm",
                "pt_albert_base_v1_masked_lm",
                "pt_albert_base_v1_token_cls",
                "pt_albert_base_v2_token_cls",
                "pt_bert_masked_lm",
                "pt_distilbert_masked_lm",
                "pt_distilbert_sequence_classification",
                "pt_distilbert_token_classification",
                "pt_dpr_ctx_encoder_single_nq_base",
                "pt_dpr_reader_single_nq_base",
                "pt_dpr_reader_multiset_base",
                "pt_dpr_question_encoder_single_nq_base",
                "pt_dpr_ctx_encoder_multiset_base",
                "pt_dpr_question_encoder_multiset_base",
                "pt_roberta_masked_lm",
                "pt_roberta_sentiment",
            ]
        },
    ),
    (
        Reshape164,
        [((1, 12, 128, 64), torch.float32)],
        {
            "model_name": [
                "pt_albert_base_v2_masked_lm",
                "pt_albert_base_v1_masked_lm",
                "pt_albert_base_v1_token_cls",
                "pt_albert_base_v2_token_cls",
                "pt_bert_masked_lm",
                "pt_distilbert_masked_lm",
                "pt_distilbert_sequence_classification",
                "pt_distilbert_token_classification",
                "pt_dpr_ctx_encoder_single_nq_base",
                "pt_dpr_reader_single_nq_base",
                "pt_dpr_reader_multiset_base",
                "pt_dpr_question_encoder_single_nq_base",
                "pt_dpr_ctx_encoder_multiset_base",
                "pt_dpr_question_encoder_multiset_base",
                "pt_roberta_masked_lm",
                "pt_roberta_sentiment",
                "pt_squeezebert",
            ]
        },
    ),
    (
        Reshape165,
        [((12, 128, 128), torch.float32)],
        {
            "model_name": [
                "pt_albert_base_v2_masked_lm",
                "pt_albert_base_v1_masked_lm",
                "pt_albert_base_v1_token_cls",
                "pt_albert_base_v2_token_cls",
                "pt_bert_masked_lm",
                "pt_distilbert_masked_lm",
                "pt_distilbert_sequence_classification",
                "pt_distilbert_token_classification",
                "pt_dpr_ctx_encoder_single_nq_base",
                "pt_dpr_reader_single_nq_base",
                "pt_dpr_reader_multiset_base",
                "pt_dpr_question_encoder_single_nq_base",
                "pt_dpr_ctx_encoder_multiset_base",
                "pt_dpr_question_encoder_multiset_base",
                "pt_roberta_masked_lm",
                "pt_roberta_sentiment",
                "pt_squeezebert",
            ]
        },
    ),
    (
        Reshape166,
        [((1, 12, 128, 128), torch.float32)],
        {
            "model_name": [
                "pt_albert_base_v2_masked_lm",
                "pt_albert_base_v1_masked_lm",
                "pt_albert_base_v1_token_cls",
                "pt_albert_base_v2_token_cls",
                "pt_bert_masked_lm",
                "pt_distilbert_masked_lm",
                "pt_distilbert_sequence_classification",
                "pt_distilbert_token_classification",
                "pt_dpr_ctx_encoder_single_nq_base",
                "pt_dpr_reader_single_nq_base",
                "pt_dpr_reader_multiset_base",
                "pt_dpr_question_encoder_single_nq_base",
                "pt_dpr_ctx_encoder_multiset_base",
                "pt_dpr_question_encoder_multiset_base",
                "pt_roberta_masked_lm",
                "pt_roberta_sentiment",
                "pt_squeezebert",
            ]
        },
    ),
    (
        Reshape167,
        [((1, 12, 64, 128), torch.float32)],
        {
            "model_name": [
                "pt_albert_base_v2_masked_lm",
                "pt_albert_base_v1_masked_lm",
                "pt_albert_base_v1_token_cls",
                "pt_albert_base_v2_token_cls",
                "pt_bert_masked_lm",
                "pt_distilbert_masked_lm",
                "pt_distilbert_sequence_classification",
                "pt_distilbert_token_classification",
                "pt_dpr_ctx_encoder_single_nq_base",
                "pt_dpr_reader_single_nq_base",
                "pt_dpr_reader_multiset_base",
                "pt_dpr_question_encoder_single_nq_base",
                "pt_dpr_ctx_encoder_multiset_base",
                "pt_dpr_question_encoder_multiset_base",
                "pt_roberta_masked_lm",
                "pt_roberta_sentiment",
            ]
        },
    ),
    (Reshape168, [((1, 12, 64, 128), torch.float32)], {"model_name": ["pt_squeezebert"]}),
    (
        Reshape169,
        [((12, 128, 64), torch.float32)],
        {
            "model_name": [
                "pt_albert_base_v2_masked_lm",
                "pt_albert_base_v1_masked_lm",
                "pt_albert_base_v1_token_cls",
                "pt_albert_base_v2_token_cls",
                "pt_bert_masked_lm",
                "pt_distilbert_masked_lm",
                "pt_distilbert_sequence_classification",
                "pt_distilbert_token_classification",
                "pt_dpr_ctx_encoder_single_nq_base",
                "pt_dpr_reader_single_nq_base",
                "pt_dpr_reader_multiset_base",
                "pt_dpr_question_encoder_single_nq_base",
                "pt_dpr_ctx_encoder_multiset_base",
                "pt_dpr_question_encoder_multiset_base",
                "pt_roberta_masked_lm",
                "pt_roberta_sentiment",
                "pt_squeezebert",
            ]
        },
    ),
    (
        Reshape170,
        [((1, 128, 12, 64), torch.float32)],
        {
            "model_name": [
                "pt_albert_base_v2_masked_lm",
                "pt_albert_base_v1_masked_lm",
                "pt_albert_base_v1_token_cls",
                "pt_albert_base_v2_token_cls",
            ]
        },
    ),
    (
        Reshape161,
        [((1, 128, 12, 64), torch.float32)],
        {
            "model_name": [
                "pt_bert_masked_lm",
                "pt_distilbert_masked_lm",
                "pt_distilbert_sequence_classification",
                "pt_distilbert_token_classification",
                "pt_dpr_ctx_encoder_single_nq_base",
                "pt_dpr_reader_single_nq_base",
                "pt_dpr_reader_multiset_base",
                "pt_dpr_question_encoder_single_nq_base",
                "pt_dpr_ctx_encoder_multiset_base",
                "pt_dpr_question_encoder_multiset_base",
                "pt_roberta_masked_lm",
                "pt_roberta_sentiment",
            ]
        },
    ),
    (
        Reshape171,
        [((1, 128, 2048), torch.float32)],
        {
            "model_name": [
                "pt_albert_xlarge_v1_token_cls",
                "pt_albert_xlarge_v2_token_cls",
                "pt_albert_xlarge_v1_masked_lm",
                "pt_albert_xlarge_v2_masked_lm",
            ]
        },
    ),
    (
        Reshape172,
        [((1, 128, 2048), torch.float32)],
        {
            "model_name": [
                "pt_albert_xlarge_v1_token_cls",
                "pt_albert_xlarge_v2_token_cls",
                "pt_albert_xlarge_v1_masked_lm",
                "pt_albert_xlarge_v2_masked_lm",
            ]
        },
    ),
    (
        Reshape173,
        [((128, 2048), torch.float32)],
        {
            "model_name": [
                "pt_albert_xlarge_v1_token_cls",
                "pt_albert_xlarge_v2_token_cls",
                "pt_albert_xlarge_v1_masked_lm",
                "pt_albert_xlarge_v2_masked_lm",
            ]
        },
    ),
    (
        Reshape174,
        [((1, 16, 128, 128), torch.float32)],
        {
            "model_name": [
                "pt_albert_xlarge_v1_token_cls",
                "pt_albert_large_v1_masked_lm",
                "pt_albert_large_v2_token_cls",
                "pt_albert_large_v2_masked_lm",
                "pt_albert_xlarge_v2_token_cls",
                "pt_albert_xlarge_v1_masked_lm",
                "pt_albert_large_v1_token_cls",
                "pt_albert_xlarge_v2_masked_lm",
                "pt_bert_sequence_classification",
            ]
        },
    ),
    (
        Reshape175,
        [((16, 128, 128), torch.float32)],
        {
            "model_name": [
                "pt_albert_xlarge_v1_token_cls",
                "pt_albert_large_v1_masked_lm",
                "pt_albert_large_v2_token_cls",
                "pt_albert_large_v2_masked_lm",
                "pt_albert_xlarge_v2_token_cls",
                "pt_albert_xlarge_v1_masked_lm",
                "pt_albert_large_v1_token_cls",
                "pt_albert_xlarge_v2_masked_lm",
                "pt_bert_sequence_classification",
            ]
        },
    ),
    (
        Reshape176,
        [((1, 128, 16, 128), torch.float32)],
        {
            "model_name": [
                "pt_albert_xlarge_v1_token_cls",
                "pt_albert_xlarge_v2_token_cls",
                "pt_albert_xlarge_v1_masked_lm",
                "pt_albert_xlarge_v2_masked_lm",
            ]
        },
    ),
    (
        Reshape177,
        [((1, 128, 1024), torch.float32)],
        {
            "model_name": [
                "pt_albert_large_v1_masked_lm",
                "pt_albert_large_v2_token_cls",
                "pt_albert_large_v2_masked_lm",
                "pt_albert_large_v1_token_cls",
                "pt_bert_sequence_classification",
            ]
        },
    ),
    (
        Reshape178,
        [((1, 128, 1024), torch.float32)],
        {
            "model_name": [
                "pt_albert_large_v1_masked_lm",
                "pt_albert_large_v2_token_cls",
                "pt_albert_large_v2_masked_lm",
                "pt_albert_large_v1_token_cls",
                "pt_bert_sequence_classification",
            ]
        },
    ),
    (
        Reshape179,
        [((128, 1024), torch.float32)],
        {
            "model_name": [
                "pt_albert_large_v1_masked_lm",
                "pt_albert_large_v2_token_cls",
                "pt_albert_large_v2_masked_lm",
                "pt_albert_large_v1_token_cls",
                "pt_bert_sequence_classification",
            ]
        },
    ),
    (Reshape180, [((128, 1024), torch.float32)], {"model_name": ["pt_Mistral_7B_v0_1"]}),
    (
        Reshape181,
        [((1, 16, 128, 64), torch.float32)],
        {
            "model_name": [
                "pt_albert_large_v1_masked_lm",
                "pt_albert_large_v2_token_cls",
                "pt_albert_large_v2_masked_lm",
                "pt_albert_large_v1_token_cls",
                "pt_bert_sequence_classification",
            ]
        },
    ),
    (
        Reshape182,
        [((1, 16, 64, 128), torch.float32)],
        {
            "model_name": [
                "pt_albert_large_v1_masked_lm",
                "pt_albert_large_v2_token_cls",
                "pt_albert_large_v2_masked_lm",
                "pt_albert_large_v1_token_cls",
                "pt_bert_sequence_classification",
            ]
        },
    ),
    (
        Reshape183,
        [((16, 128, 64), torch.float32)],
        {
            "model_name": [
                "pt_albert_large_v1_masked_lm",
                "pt_albert_large_v2_token_cls",
                "pt_albert_large_v2_masked_lm",
                "pt_albert_large_v1_token_cls",
                "pt_bert_sequence_classification",
            ]
        },
    ),
    (
        Reshape184,
        [((1, 128, 16, 64), torch.float32)],
        {
            "model_name": [
                "pt_albert_large_v1_masked_lm",
                "pt_albert_large_v2_token_cls",
                "pt_albert_large_v2_masked_lm",
                "pt_albert_large_v1_token_cls",
            ]
        },
    ),
    (Reshape177, [((1, 128, 16, 64), torch.float32)], {"model_name": ["pt_bert_sequence_classification"]}),
    (
        Reshape185,
        [((1, 128, 4096), torch.float32)],
        {
            "model_name": [
                "pt_albert_xxlarge_v1_token_cls",
                "pt_albert_xxlarge_v2_masked_lm",
                "pt_albert_xxlarge_v1_masked_lm",
                "pt_albert_xxlarge_v2_token_cls",
                "pt_Mistral_7B_v0_1",
            ]
        },
    ),
    (
        Reshape186,
        [((1, 128, 4096), torch.float32)],
        {
            "model_name": [
                "pt_albert_xxlarge_v1_token_cls",
                "pt_albert_xxlarge_v2_masked_lm",
                "pt_albert_xxlarge_v1_masked_lm",
                "pt_albert_xxlarge_v2_token_cls",
                "pt_mit_b4",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Reshape187,
        [((128, 4096), torch.float32)],
        {
            "model_name": [
                "pt_albert_xxlarge_v1_token_cls",
                "pt_albert_xxlarge_v2_masked_lm",
                "pt_albert_xxlarge_v1_masked_lm",
                "pt_albert_xxlarge_v2_token_cls",
                "pt_Mistral_7B_v0_1",
            ]
        },
    ),
    (Reshape188, [((128, 4096), torch.float32)], {"model_name": ["pt_Mistral_7B_v0_1"]}),
    (
        Reshape189,
        [((1, 64, 128, 64), torch.float32)],
        {
            "model_name": [
                "pt_albert_xxlarge_v1_token_cls",
                "pt_albert_xxlarge_v2_masked_lm",
                "pt_albert_xxlarge_v1_masked_lm",
                "pt_albert_xxlarge_v2_token_cls",
            ]
        },
    ),
    (
        Reshape190,
        [((64, 128, 128), torch.float32)],
        {
            "model_name": [
                "pt_albert_xxlarge_v1_token_cls",
                "pt_albert_xxlarge_v2_masked_lm",
                "pt_albert_xxlarge_v1_masked_lm",
                "pt_albert_xxlarge_v2_token_cls",
            ]
        },
    ),
    (
        Reshape191,
        [((1, 64, 128, 128), torch.float32)],
        {
            "model_name": [
                "pt_albert_xxlarge_v1_token_cls",
                "pt_albert_xxlarge_v2_masked_lm",
                "pt_albert_xxlarge_v1_masked_lm",
                "pt_albert_xxlarge_v2_token_cls",
            ]
        },
    ),
    (
        Reshape192,
        [((1, 64, 128, 128), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Reshape193,
        [((1, 64, 64, 128), torch.float32)],
        {
            "model_name": [
                "pt_albert_xxlarge_v1_token_cls",
                "pt_albert_xxlarge_v2_masked_lm",
                "pt_albert_xxlarge_v1_masked_lm",
                "pt_albert_xxlarge_v2_token_cls",
            ]
        },
    ),
    (
        Reshape194,
        [((64, 128, 64), torch.float32)],
        {
            "model_name": [
                "pt_albert_xxlarge_v1_token_cls",
                "pt_albert_xxlarge_v2_masked_lm",
                "pt_albert_xxlarge_v1_masked_lm",
                "pt_albert_xxlarge_v2_token_cls",
            ]
        },
    ),
    (
        Reshape195,
        [((1, 128, 64, 64), torch.float32)],
        {
            "model_name": [
                "pt_albert_xxlarge_v1_token_cls",
                "pt_albert_xxlarge_v2_masked_lm",
                "pt_albert_xxlarge_v1_masked_lm",
                "pt_albert_xxlarge_v2_token_cls",
                "pt_mit_b4",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Reshape196,
        [((1, 256, 1024), torch.float32)],
        {"model_name": ["pt_bart", "pt_codegen_350M_mono", "pt_opt_350m_causal_lm", "pt_t5_large", "pt_xglm_564M"]},
    ),
    (
        Reshape197,
        [((1, 256, 1024), torch.float32)],
        {"model_name": ["pt_bart", "pt_opt_350m_causal_lm", "pt_xglm_564M"]},
    ),
    (
        Reshape198,
        [((1, 256, 1024), torch.float32)],
        {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_segformer_b1_finetuned_ade_512_512"]},
    ),
    (
        Reshape199,
        [((256, 1024), torch.float32)],
        {"model_name": ["pt_bart", "pt_codegen_350M_mono", "pt_opt_350m_causal_lm", "pt_xglm_564M"]},
    ),
    (Reshape200, [((256, 1024), torch.float32)], {"model_name": ["pt_codegen_350M_mono"]}),
    (
        Reshape201,
        [((256, 1024), torch.float32)],
        {
            "model_name": [
                "pt_Llama_3_1_8B_Instruct_causal_lm",
                "pt_Meta_Llama_3_8B_causal_lm",
                "pt_Meta_Llama_3_8B_Instruct_causal_lm",
                "pt_Llama_3_1_8B_causal_lm",
            ]
        },
    ),
    (Reshape197, [((256, 1024), torch.float32)], {"model_name": ["pt_t5_large"]}),
    (
        Reshape202,
        [((1, 16, 256, 64), torch.float32)],
        {"model_name": ["pt_bart", "pt_codegen_350M_mono", "pt_opt_350m_causal_lm", "pt_t5_large", "pt_xglm_564M"]},
    ),
    (
        Reshape203,
        [((16, 256, 256), torch.float32)],
        {
            "model_name": [
                "pt_bart",
                "pt_codegen_350M_mono",
                "pt_gpt_neo_1_3B_causal_lm",
                "pt_opt_350m_causal_lm",
                "pt_xglm_1_7B",
                "pt_xglm_564M",
            ]
        },
    ),
    (
        Reshape204,
        [((1, 16, 256, 256), torch.float32)],
        {
            "model_name": [
                "pt_bart",
                "pt_codegen_350M_mono",
                "pt_gpt_neo_1_3B_causal_lm",
                "pt_opt_350m_causal_lm",
                "pt_xglm_1_7B",
                "pt_xglm_564M",
            ]
        },
    ),
    (
        Reshape205,
        [((16, 256, 64), torch.float32)],
        {"model_name": ["pt_bart", "pt_codegen_350M_mono", "pt_opt_350m_causal_lm", "pt_xglm_564M"]},
    ),
    (
        Reshape196,
        [((1, 256, 16, 64), torch.float32)],
        {"model_name": ["pt_bart", "pt_codegen_350M_mono", "pt_opt_350m_causal_lm", "pt_xglm_564M"]},
    ),
    (
        Reshape206,
        [((1, 256), torch.float32)],
        {
            "model_name": [
                "pt_bart",
                "pt_gpt2_generation",
                "pt_opt_350m_causal_lm",
                "pt_opt_125m_causal_lm",
                "pt_opt_1_3b_causal_lm",
                "pt_xglm_1_7B",
                "pt_xglm_564M",
            ]
        },
    ),
    (Reshape207, [((1, 384, 1024), torch.float32)], {"model_name": ["pt_bert_qa"]}),
    (Reshape208, [((1, 384, 1024), torch.float32)], {"model_name": ["pt_bert_qa"]}),
    (Reshape209, [((384, 1024), torch.float32)], {"model_name": ["pt_bert_qa"]}),
    (Reshape210, [((1, 16, 384, 64), torch.float32)], {"model_name": ["pt_bert_qa"]}),
    (Reshape211, [((16, 384, 384), torch.float32)], {"model_name": ["pt_bert_qa"]}),
    (Reshape212, [((1, 16, 384, 384), torch.float32)], {"model_name": ["pt_bert_qa"]}),
    (Reshape213, [((1, 16, 64, 384), torch.float32)], {"model_name": ["pt_bert_qa"]}),
    (Reshape214, [((16, 384, 64), torch.float32)], {"model_name": ["pt_bert_qa"]}),
    (Reshape207, [((1, 384, 16, 64), torch.float32)], {"model_name": ["pt_bert_qa"]}),
    (Reshape215, [((384, 1), torch.float32)], {"model_name": ["pt_bert_qa", "pt_distilbert_question_answering"]}),
    (Reshape197, [((1, 256, 4, 256), torch.float32)], {"model_name": ["pt_codegen_350M_mono"]}),
    (Reshape216, [((1, 256, 16, 16, 2), torch.float32)], {"model_name": ["pt_codegen_350M_mono"]}),
    (Reshape217, [((1, 16, 64, 256), torch.float32)], {"model_name": ["pt_codegen_350M_mono", "pt_t5_large"]}),
    (
        Reshape218,
        [((256, 4096), torch.float32)],
        {
            "model_name": [
                "pt_codegen_350M_mono",
                "pt_Llama_3_1_8B_Instruct_causal_lm",
                "pt_Meta_Llama_3_8B_causal_lm",
                "pt_Meta_Llama_3_8B_Instruct_causal_lm",
                "pt_Llama_3_1_8B_causal_lm",
            ]
        },
    ),
    (
        Reshape219,
        [((256, 4096), torch.float32)],
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
        Reshape220,
        [((1, 128), torch.float32)],
        {
            "model_name": [
                "pt_distilbert_masked_lm",
                "pt_distilbert_sequence_classification",
                "pt_distilbert_token_classification",
            ]
        },
    ),
    (Reshape221, [((1, 384, 768), torch.float32)], {"model_name": ["pt_distilbert_question_answering"]}),
    (Reshape222, [((1, 384, 768), torch.float32)], {"model_name": ["pt_distilbert_question_answering"]}),
    (Reshape223, [((384, 768), torch.float32)], {"model_name": ["pt_distilbert_question_answering"]}),
    (Reshape224, [((1, 12, 384, 64), torch.float32)], {"model_name": ["pt_distilbert_question_answering"]}),
    (Reshape225, [((12, 384, 384), torch.float32)], {"model_name": ["pt_distilbert_question_answering"]}),
    (Reshape226, [((1, 384), torch.float32)], {"model_name": ["pt_distilbert_question_answering"]}),
    (Reshape227, [((1, 12, 384, 384), torch.float32)], {"model_name": ["pt_distilbert_question_answering"]}),
    (Reshape228, [((1, 12, 64, 384), torch.float32)], {"model_name": ["pt_distilbert_question_answering"]}),
    (Reshape229, [((12, 384, 64), torch.float32)], {"model_name": ["pt_distilbert_question_answering"]}),
    (Reshape221, [((1, 384, 12, 64), torch.float32)], {"model_name": ["pt_distilbert_question_answering"]}),
    (
        Reshape230,
        [((128, 1), torch.float32)],
        {"model_name": ["pt_dpr_reader_single_nq_base", "pt_dpr_reader_multiset_base"]},
    ),
    (
        Reshape231,
        [((1, 128), torch.float32)],
        {"model_name": ["pt_dpr_reader_single_nq_base", "pt_dpr_reader_multiset_base"]},
    ),
    (
        Reshape232,
        [((1, 1), torch.float32)],
        {"model_name": ["pt_dpr_reader_single_nq_base", "pt_dpr_reader_multiset_base"]},
    ),
    (Reshape233, [((1, 6, 4544), torch.float32)], {"model_name": ["pt_falcon"]}),
    (Reshape234, [((6, 18176), torch.float32)], {"model_name": ["pt_falcon"]}),
    (Reshape235, [((6, 4672), torch.float32)], {"model_name": ["pt_falcon"]}),
    (Reshape236, [((1, 71, 6, 64), torch.float32)], {"model_name": ["pt_falcon"]}),
    (Reshape237, [((1, 71, 6, 64), torch.float32)], {"model_name": ["pt_falcon"]}),
    (Reshape238, [((1, 6, 1, 64), torch.float32)], {"model_name": ["pt_falcon"]}),
    (Reshape239, [((1, 6, 1, 64), torch.float32)], {"model_name": ["pt_google_flan_t5_small"]}),
    (Reshape240, [((1, 1, 6, 64), torch.float32)], {"model_name": ["pt_falcon"]}),
    (Reshape241, [((1, 1, 6, 64), torch.float32)], {"model_name": ["pt_falcon"]}),
    (Reshape242, [((1, 1, 6, 64), torch.float32)], {"model_name": ["pt_google_flan_t5_small"]}),
    (Reshape243, [((71, 6, 6), torch.float32)], {"model_name": ["pt_falcon"]}),
    (Reshape244, [((1, 71, 6, 6), torch.float32)], {"model_name": ["pt_falcon"]}),
    (Reshape236, [((71, 6, 64), torch.float32)], {"model_name": ["pt_falcon"]}),
    (Reshape233, [((1, 6, 71, 64), torch.float32)], {"model_name": ["pt_falcon"]}),
    (Reshape245, [((6, 4544), torch.float32)], {"model_name": ["pt_falcon"]}),
    (Reshape246, [((1, 334, 12288), torch.float32)], {"model_name": ["pt_fuyu_8b"]}),
    (Reshape247, [((1, 334, 64, 1, 64), torch.float32)], {"model_name": ["pt_fuyu_8b"]}),
    (Reshape248, [((1, 64, 334, 64), torch.float32)], {"model_name": ["pt_fuyu_8b"]}),
    (Reshape249, [((64, 334, 334), torch.float32)], {"model_name": ["pt_fuyu_8b"]}),
    (Reshape250, [((1, 64, 334, 334), torch.float32)], {"model_name": ["pt_fuyu_8b"]}),
    (Reshape251, [((1, 64, 64, 334), torch.float32)], {"model_name": ["pt_fuyu_8b"]}),
    (Reshape252, [((64, 334, 64), torch.float32)], {"model_name": ["pt_fuyu_8b"]}),
    (Reshape253, [((1, 334, 64, 64), torch.float32)], {"model_name": ["pt_fuyu_8b"]}),
    (Reshape254, [((334, 4096), torch.float32)], {"model_name": ["pt_fuyu_8b"]}),
    (Reshape255, [((1, 7, 2048), torch.float32)], {"model_name": ["pt_gemma_2b"]}),
    (Reshape256, [((7, 2048), torch.float32)], {"model_name": ["pt_gemma_2b"]}),
    (Reshape257, [((7, 2048), torch.float32)], {"model_name": ["pt_gemma_2b"]}),
    (Reshape258, [((1, 8, 7, 256), torch.float32)], {"model_name": ["pt_gemma_2b"]}),
    (Reshape259, [((7, 256), torch.float32)], {"model_name": ["pt_gemma_2b"]}),
    (Reshape258, [((1, 1, 8, 7, 256), torch.float32)], {"model_name": ["pt_gemma_2b"]}),
    (Reshape260, [((1, 1, 8, 7, 256), torch.float32)], {"model_name": ["pt_gemma_2b"]}),
    (Reshape261, [((8, 7, 7), torch.float32)], {"model_name": ["pt_gemma_2b"]}),
    (Reshape262, [((1, 8, 7, 7), torch.float32)], {"model_name": ["pt_gemma_2b"]}),
    (Reshape263, [((1, 8, 256, 7), torch.float32)], {"model_name": ["pt_gemma_2b"]}),
    (Reshape260, [((8, 7, 256), torch.float32)], {"model_name": ["pt_gemma_2b"]}),
    (Reshape255, [((1, 7, 8, 256), torch.float32)], {"model_name": ["pt_gemma_2b"]}),
    (Reshape264, [((7, 16384), torch.float32)], {"model_name": ["pt_gemma_2b"]}),
    (
        Reshape265,
        [((1, 256, 768), torch.float32)],
        {
            "model_name": [
                "pt_gpt2_generation",
                "pt_gpt_neo_125M_causal_lm",
                "pt_opt_125m_causal_lm",
                "pt_t5_base",
                "pt_google_flan_t5_base",
            ]
        },
    ),
    (Reshape266, [((1, 256, 768), torch.float32)], {"model_name": ["pt_gpt2_generation", "pt_opt_125m_causal_lm"]}),
    (
        Reshape267,
        [((256, 768), torch.float32)],
        {"model_name": ["pt_gpt2_generation", "pt_gpt_neo_125M_causal_lm", "pt_opt_125m_causal_lm"]},
    ),
    (
        Reshape266,
        [((256, 768), torch.float32)],
        {"model_name": ["pt_gpt_neo_125M_causal_lm", "pt_t5_base", "pt_google_flan_t5_base"]},
    ),
    (
        Reshape268,
        [((1, 12, 256, 64), torch.float32)],
        {
            "model_name": [
                "pt_gpt2_generation",
                "pt_gpt_neo_125M_causal_lm",
                "pt_opt_125m_causal_lm",
                "pt_t5_base",
                "pt_google_flan_t5_base",
            ]
        },
    ),
    (
        Reshape269,
        [((12, 256, 256), torch.float32)],
        {"model_name": ["pt_gpt2_generation", "pt_gpt_neo_125M_causal_lm", "pt_opt_125m_causal_lm"]},
    ),
    (Reshape206, [((1, 256), torch.float32)], {"model_name": ["pt_gpt2_generation"]}),
    (
        Reshape270,
        [((1, 12, 256, 256), torch.float32)],
        {"model_name": ["pt_gpt2_generation", "pt_gpt_neo_125M_causal_lm", "pt_opt_125m_causal_lm"]},
    ),
    (
        Reshape271,
        [((1, 12, 64, 256), torch.float32)],
        {"model_name": ["pt_gpt2_generation", "pt_gpt_neo_125M_causal_lm", "pt_t5_base", "pt_google_flan_t5_base"]},
    ),
    (
        Reshape272,
        [((12, 256, 64), torch.float32)],
        {"model_name": ["pt_gpt2_generation", "pt_gpt_neo_125M_causal_lm", "pt_opt_125m_causal_lm"]},
    ),
    (
        Reshape265,
        [((1, 256, 12, 64), torch.float32)],
        {"model_name": ["pt_gpt2_generation", "pt_gpt_neo_125M_causal_lm", "pt_opt_125m_causal_lm"]},
    ),
    (Reshape273, [((256, 3072), torch.float32)], {"model_name": ["pt_gpt2_generation"]}),
    (Reshape274, [((1, 256, 3072), torch.float32)], {"model_name": ["pt_gpt2_generation"]}),
    (
        Reshape275,
        [((1, 256, 2560), torch.float32)],
        {"model_name": ["pt_gpt_neo_2_7B_causal_lm", "pt_phi_2_causal_lm", "pt_phi_2_pytdml_causal_lm"]},
    ),
    (
        Reshape276,
        [((1, 256, 2560), torch.float32)],
        {"model_name": ["pt_phi_2_causal_lm", "pt_phi_2_pytdml_causal_lm"]},
    ),
    (Reshape277, [((256, 2560), torch.float32)], {"model_name": ["pt_gpt_neo_2_7B_causal_lm"]}),
    (
        Reshape278,
        [((256, 2560), torch.float32)],
        {"model_name": ["pt_gpt_neo_2_7B_causal_lm", "pt_phi_2_causal_lm", "pt_phi_2_pytdml_causal_lm"]},
    ),
    (Reshape279, [((1, 20, 256, 128), torch.float32)], {"model_name": ["pt_gpt_neo_2_7B_causal_lm"]}),
    (Reshape280, [((20, 256, 256), torch.float32)], {"model_name": ["pt_gpt_neo_2_7B_causal_lm"]}),
    (Reshape281, [((1, 20, 256, 256), torch.float32)], {"model_name": ["pt_gpt_neo_2_7B_causal_lm"]}),
    (Reshape282, [((1, 20, 128, 256), torch.float32)], {"model_name": ["pt_gpt_neo_2_7B_causal_lm"]}),
    (Reshape283, [((20, 256, 128), torch.float32)], {"model_name": ["pt_gpt_neo_2_7B_causal_lm"]}),
    (Reshape275, [((1, 256, 20, 128), torch.float32)], {"model_name": ["pt_gpt_neo_2_7B_causal_lm"]}),
    (
        Reshape284,
        [((1, 256, 2048), torch.float32)],
        {
            "model_name": [
                "pt_gpt_neo_1_3B_causal_lm",
                "pt_Llama_3_2_1B_Instruct_causal_lm",
                "pt_Llama_3_2_1B_causal_lm",
                "pt_opt_1_3b_causal_lm",
                "pt_xglm_1_7B",
            ]
        },
    ),
    (Reshape285, [((1, 256, 2048), torch.float32)], {"model_name": ["pt_opt_1_3b_causal_lm"]}),
    (Reshape286, [((1, 256, 2048), torch.float32)], {"model_name": ["pt_xglm_1_7B"]}),
    (Reshape286, [((256, 2048), torch.float32)], {"model_name": ["pt_gpt_neo_1_3B_causal_lm"]}),
    (
        Reshape287,
        [((256, 2048), torch.float32)],
        {
            "model_name": [
                "pt_gpt_neo_1_3B_causal_lm",
                "pt_Llama_3_2_1B_Instruct_causal_lm",
                "pt_Llama_3_2_1B_causal_lm",
                "pt_opt_1_3b_causal_lm",
                "pt_xglm_1_7B",
            ]
        },
    ),
    (
        Reshape285,
        [((256, 2048), torch.float32)],
        {"model_name": ["pt_Llama_3_2_1B_Instruct_causal_lm", "pt_Llama_3_2_1B_causal_lm"]},
    ),
    (Reshape288, [((1, 16, 256, 128), torch.float32)], {"model_name": ["pt_gpt_neo_1_3B_causal_lm", "pt_xglm_1_7B"]}),
    (Reshape289, [((1, 16, 128, 256), torch.float32)], {"model_name": ["pt_gpt_neo_1_3B_causal_lm"]}),
    (Reshape290, [((16, 256, 128), torch.float32)], {"model_name": ["pt_gpt_neo_1_3B_causal_lm", "pt_xglm_1_7B"]}),
    (Reshape284, [((1, 256, 16, 128), torch.float32)], {"model_name": ["pt_gpt_neo_1_3B_causal_lm", "pt_xglm_1_7B"]}),
    (
        Reshape291,
        [((1, 256, 4096), torch.float32)],
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
        Reshape292,
        [((1, 256, 4096), torch.float32)],
        {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_mit_b0", "pt_segformer_b1_finetuned_ade_512_512"]},
    ),
    (
        Reshape293,
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
        Reshape293,
        [((1, 8, 4, 256, 128), torch.float32)],
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
        Reshape294,
        [((1, 8, 4, 256, 128), torch.float32)],
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
        Reshape295,
        [((32, 256, 256), torch.float32)],
        {
            "model_name": [
                "pt_Llama_3_1_8B_Instruct_causal_lm",
                "pt_Meta_Llama_3_8B_causal_lm",
                "pt_Llama_3_2_1B_Instruct_causal_lm",
                "pt_Meta_Llama_3_8B_Instruct_causal_lm",
                "pt_Llama_3_2_1B_causal_lm",
                "pt_Llama_3_1_8B_causal_lm",
                "pt_opt_1_3b_causal_lm",
                "pt_phi_2_causal_lm",
                "pt_phi_2_pytdml_causal_lm",
            ]
        },
    ),
    (
        Reshape296,
        [((1, 32, 256, 256), torch.float32)],
        {
            "model_name": [
                "pt_Llama_3_1_8B_Instruct_causal_lm",
                "pt_Meta_Llama_3_8B_causal_lm",
                "pt_Llama_3_2_1B_Instruct_causal_lm",
                "pt_Meta_Llama_3_8B_Instruct_causal_lm",
                "pt_Llama_3_2_1B_causal_lm",
                "pt_Llama_3_1_8B_causal_lm",
                "pt_opt_1_3b_causal_lm",
                "pt_phi_2_causal_lm",
                "pt_phi_2_pytdml_causal_lm",
            ]
        },
    ),
    (
        Reshape297,
        [((1, 32, 128, 256), torch.float32)],
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
        Reshape294,
        [((32, 256, 128), torch.float32)],
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
        Reshape291,
        [((1, 256, 32, 128), torch.float32)],
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
        Reshape298,
        [((256, 14336), torch.float32)],
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
        Reshape299,
        [((1, 32, 256, 64), torch.float32)],
        {"model_name": ["pt_Llama_3_2_1B_Instruct_causal_lm", "pt_Llama_3_2_1B_causal_lm", "pt_opt_1_3b_causal_lm"]},
    ),
    (
        Reshape300,
        [((256, 512), torch.float32)],
        {"model_name": ["pt_Llama_3_2_1B_Instruct_causal_lm", "pt_Llama_3_2_1B_causal_lm"]},
    ),
    (Reshape301, [((256, 512), torch.float32)], {"model_name": ["pt_opt_350m_causal_lm"]}),
    (
        Reshape302,
        [((256, 512), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Reshape299,
        [((1, 8, 4, 256, 64), torch.float32)],
        {"model_name": ["pt_Llama_3_2_1B_Instruct_causal_lm", "pt_Llama_3_2_1B_causal_lm"]},
    ),
    (
        Reshape303,
        [((1, 8, 4, 256, 64), torch.float32)],
        {"model_name": ["pt_Llama_3_2_1B_Instruct_causal_lm", "pt_Llama_3_2_1B_causal_lm"]},
    ),
    (
        Reshape304,
        [((1, 32, 64, 256), torch.float32)],
        {"model_name": ["pt_Llama_3_2_1B_Instruct_causal_lm", "pt_Llama_3_2_1B_causal_lm"]},
    ),
    (
        Reshape303,
        [((32, 256, 64), torch.float32)],
        {"model_name": ["pt_Llama_3_2_1B_Instruct_causal_lm", "pt_Llama_3_2_1B_causal_lm", "pt_opt_1_3b_causal_lm"]},
    ),
    (
        Reshape284,
        [((1, 256, 32, 64), torch.float32)],
        {"model_name": ["pt_Llama_3_2_1B_Instruct_causal_lm", "pt_Llama_3_2_1B_causal_lm", "pt_opt_1_3b_causal_lm"]},
    ),
    (
        Reshape305,
        [((256, 8192), torch.float32)],
        {"model_name": ["pt_Llama_3_2_1B_Instruct_causal_lm", "pt_Llama_3_2_1B_causal_lm"]},
    ),
    (
        Reshape306,
        [((1, 4, 4096), torch.float32)],
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
        Reshape307,
        [((4, 4096), torch.float32)],
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
        Reshape308,
        [((4, 4096), torch.float32)],
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
        Reshape309,
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
        Reshape310,
        [((4, 1024), torch.float32)],
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
        Reshape309,
        [((1, 8, 4, 4, 128), torch.float32)],
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
        Reshape311,
        [((1, 8, 4, 4, 128), torch.float32)],
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
        Reshape312,
        [((32, 4, 4), torch.float32)],
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
        Reshape313,
        [((1, 32, 4, 4), torch.float32)],
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
        Reshape314,
        [((1, 32, 128, 4), torch.float32)],
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
        Reshape311,
        [((32, 4, 128), torch.float32)],
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
        Reshape306,
        [((1, 4, 32, 128), torch.float32)],
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
        Reshape315,
        [((4, 14336), torch.float32)],
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
        Reshape61,
        [((1, 1, 2), torch.float32)],
        {
            "model_name": [
                "pt_Llama_3_1_8B_Instruct_seq_cls",
                "pt_Meta_Llama_3_8B_seq_cls",
                "pt_Llama_3_2_1B_Instruct_seq_cls",
                "pt_Llama_3_2_1B_seq_cls",
                "pt_Llama_3_1_8B_seq_cls",
                "pt_Meta_Llama_3_8B_Instruct_seq_cls",
                "pt_phi_2_seq_cls",
                "pt_phi_2_pytdml_seq_cls",
            ]
        },
    ),
    (
        Reshape316,
        [((1, 4, 2048), torch.float32)],
        {"model_name": ["pt_Llama_3_2_1B_Instruct_seq_cls", "pt_Llama_3_2_1B_seq_cls"]},
    ),
    (
        Reshape317,
        [((4, 2048), torch.float32)],
        {"model_name": ["pt_Llama_3_2_1B_Instruct_seq_cls", "pt_Llama_3_2_1B_seq_cls"]},
    ),
    (
        Reshape318,
        [((4, 2048), torch.float32)],
        {"model_name": ["pt_Llama_3_2_1B_Instruct_seq_cls", "pt_Llama_3_2_1B_seq_cls"]},
    ),
    (
        Reshape319,
        [((1, 32, 4, 64), torch.float32)],
        {"model_name": ["pt_Llama_3_2_1B_Instruct_seq_cls", "pt_Llama_3_2_1B_seq_cls"]},
    ),
    (
        Reshape320,
        [((4, 512), torch.float32)],
        {"model_name": ["pt_Llama_3_2_1B_Instruct_seq_cls", "pt_Llama_3_2_1B_seq_cls"]},
    ),
    (
        Reshape319,
        [((1, 8, 4, 4, 64), torch.float32)],
        {"model_name": ["pt_Llama_3_2_1B_Instruct_seq_cls", "pt_Llama_3_2_1B_seq_cls"]},
    ),
    (
        Reshape321,
        [((1, 8, 4, 4, 64), torch.float32)],
        {"model_name": ["pt_Llama_3_2_1B_Instruct_seq_cls", "pt_Llama_3_2_1B_seq_cls"]},
    ),
    (
        Reshape322,
        [((1, 32, 64, 4), torch.float32)],
        {"model_name": ["pt_Llama_3_2_1B_Instruct_seq_cls", "pt_Llama_3_2_1B_seq_cls"]},
    ),
    (
        Reshape321,
        [((32, 4, 64), torch.float32)],
        {"model_name": ["pt_Llama_3_2_1B_Instruct_seq_cls", "pt_Llama_3_2_1B_seq_cls"]},
    ),
    (
        Reshape316,
        [((1, 4, 32, 64), torch.float32)],
        {"model_name": ["pt_Llama_3_2_1B_Instruct_seq_cls", "pt_Llama_3_2_1B_seq_cls"]},
    ),
    (
        Reshape323,
        [((4, 8192), torch.float32)],
        {"model_name": ["pt_Llama_3_2_1B_Instruct_seq_cls", "pt_Llama_3_2_1B_seq_cls"]},
    ),
    (Reshape324, [((1, 32, 128, 128), torch.float32)], {"model_name": ["pt_Mistral_7B_v0_1"]}),
    (
        Reshape325,
        [((1, 32, 128, 128), torch.float32)],
        {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_mit_b0"]},
    ),
    (Reshape324, [((1, 8, 4, 128, 128), torch.float32)], {"model_name": ["pt_Mistral_7B_v0_1"]}),
    (Reshape326, [((1, 8, 4, 128, 128), torch.float32)], {"model_name": ["pt_Mistral_7B_v0_1"]}),
    (Reshape326, [((32, 128, 128), torch.float32)], {"model_name": ["pt_Mistral_7B_v0_1"]}),
    (Reshape185, [((1, 128, 32, 128), torch.float32)], {"model_name": ["pt_Mistral_7B_v0_1"]}),
    (Reshape327, [((128, 14336), torch.float32)], {"model_name": ["pt_Mistral_7B_v0_1"]}),
    (
        Reshape328,
        [((1, 32), torch.float32)],
        {
            "model_name": [
                "pt_opt_125m_seq_cls",
                "pt_opt_1_3b_seq_cls",
                "pt_opt_1_3b_qa",
                "pt_opt_350m_qa",
                "pt_opt_125m_qa",
                "pt_opt_350m_seq_cls",
            ]
        },
    ),
    (Reshape329, [((1, 32, 768), torch.float32)], {"model_name": ["pt_opt_125m_seq_cls", "pt_opt_125m_qa"]}),
    (Reshape330, [((1, 32, 768), torch.float32)], {"model_name": ["pt_opt_125m_seq_cls", "pt_opt_125m_qa"]}),
    (Reshape331, [((32, 768), torch.float32)], {"model_name": ["pt_opt_125m_seq_cls", "pt_opt_125m_qa"]}),
    (Reshape332, [((1, 12, 32, 64), torch.float32)], {"model_name": ["pt_opt_125m_seq_cls", "pt_opt_125m_qa"]}),
    (Reshape333, [((12, 32, 32), torch.float32)], {"model_name": ["pt_opt_125m_seq_cls", "pt_opt_125m_qa"]}),
    (Reshape334, [((1, 12, 32, 32), torch.float32)], {"model_name": ["pt_opt_125m_seq_cls", "pt_opt_125m_qa"]}),
    (Reshape335, [((12, 32, 64), torch.float32)], {"model_name": ["pt_opt_125m_seq_cls", "pt_opt_125m_qa"]}),
    (Reshape329, [((1, 32, 12, 64), torch.float32)], {"model_name": ["pt_opt_125m_seq_cls", "pt_opt_125m_qa"]}),
    (
        Reshape336,
        [((32, 2), torch.float32)],
        {"model_name": ["pt_opt_125m_seq_cls", "pt_opt_1_3b_seq_cls", "pt_opt_350m_seq_cls"]},
    ),
    (
        Reshape61,
        [((1, 2), torch.float32)],
        {"model_name": ["pt_opt_125m_seq_cls", "pt_opt_1_3b_seq_cls", "pt_opt_350m_seq_cls"]},
    ),
    (Reshape337, [((1, 32, 2048), torch.float32)], {"model_name": ["pt_opt_1_3b_seq_cls", "pt_opt_1_3b_qa"]}),
    (Reshape338, [((1, 32, 2048), torch.float32)], {"model_name": ["pt_opt_1_3b_seq_cls", "pt_opt_1_3b_qa"]}),
    (Reshape339, [((32, 2048), torch.float32)], {"model_name": ["pt_opt_1_3b_seq_cls", "pt_opt_1_3b_qa"]}),
    (Reshape340, [((1, 32, 32, 64), torch.float32)], {"model_name": ["pt_opt_1_3b_seq_cls", "pt_opt_1_3b_qa"]}),
    (Reshape337, [((1, 32, 32, 64), torch.float32)], {"model_name": ["pt_opt_1_3b_seq_cls", "pt_opt_1_3b_qa"]}),
    (Reshape341, [((32, 32, 32), torch.float32)], {"model_name": ["pt_opt_1_3b_seq_cls", "pt_opt_1_3b_qa"]}),
    (Reshape342, [((1, 32, 32, 32), torch.float32)], {"model_name": ["pt_opt_1_3b_seq_cls", "pt_opt_1_3b_qa"]}),
    (Reshape338, [((32, 32, 64), torch.float32)], {"model_name": ["pt_opt_1_3b_seq_cls", "pt_opt_1_3b_qa"]}),
    (Reshape343, [((32, 1), torch.float32)], {"model_name": ["pt_opt_1_3b_qa", "pt_opt_350m_qa", "pt_opt_125m_qa"]}),
    (Reshape344, [((1, 32, 1024), torch.float32)], {"model_name": ["pt_opt_350m_qa", "pt_opt_350m_seq_cls"]}),
    (Reshape345, [((1, 32, 1024), torch.float32)], {"model_name": ["pt_opt_350m_qa", "pt_opt_350m_seq_cls"]}),
    (Reshape346, [((32, 1024), torch.float32)], {"model_name": ["pt_opt_350m_qa", "pt_opt_350m_seq_cls"]}),
    (Reshape347, [((1, 16, 32, 64), torch.float32)], {"model_name": ["pt_opt_350m_qa", "pt_opt_350m_seq_cls"]}),
    (Reshape348, [((16, 32, 32), torch.float32)], {"model_name": ["pt_opt_350m_qa", "pt_opt_350m_seq_cls"]}),
    (Reshape349, [((1, 16, 32, 32), torch.float32)], {"model_name": ["pt_opt_350m_qa", "pt_opt_350m_seq_cls"]}),
    (Reshape350, [((16, 32, 64), torch.float32)], {"model_name": ["pt_opt_350m_qa", "pt_opt_350m_seq_cls"]}),
    (Reshape344, [((1, 32, 16, 64), torch.float32)], {"model_name": ["pt_opt_350m_qa", "pt_opt_350m_seq_cls"]}),
    (Reshape351, [((32, 512), torch.float32)], {"model_name": ["pt_opt_350m_qa", "pt_opt_350m_seq_cls"]}),
    (Reshape352, [((256, 50272), torch.float32)], {"model_name": ["pt_opt_350m_causal_lm"]}),
    (Reshape353, [((1, 12, 2560), torch.float32)], {"model_name": ["pt_phi_2_pytdml_token_cls", "pt_phi_2_token_cls"]}),
    (Reshape354, [((1, 12, 2560), torch.float32)], {"model_name": ["pt_phi_2_pytdml_token_cls", "pt_phi_2_token_cls"]}),
    (Reshape355, [((12, 2560), torch.float32)], {"model_name": ["pt_phi_2_pytdml_token_cls", "pt_phi_2_token_cls"]}),
    (
        Reshape356,
        [((1, 32, 12, 80), torch.float32)],
        {"model_name": ["pt_phi_2_pytdml_token_cls", "pt_phi_2_token_cls"]},
    ),
    (Reshape357, [((32, 12, 12), torch.float32)], {"model_name": ["pt_phi_2_pytdml_token_cls", "pt_phi_2_token_cls"]}),
    (
        Reshape358,
        [((1, 32, 12, 12), torch.float32)],
        {"model_name": ["pt_phi_2_pytdml_token_cls", "pt_phi_2_token_cls"]},
    ),
    (
        Reshape359,
        [((1, 32, 80, 12), torch.float32)],
        {"model_name": ["pt_phi_2_pytdml_token_cls", "pt_phi_2_token_cls"]},
    ),
    (Reshape360, [((32, 12, 80), torch.float32)], {"model_name": ["pt_phi_2_pytdml_token_cls", "pt_phi_2_token_cls"]}),
    (
        Reshape353,
        [((1, 12, 32, 80), torch.float32)],
        {"model_name": ["pt_phi_2_pytdml_token_cls", "pt_phi_2_token_cls"]},
    ),
    (Reshape361, [((12, 10240), torch.float32)], {"model_name": ["pt_phi_2_pytdml_token_cls", "pt_phi_2_token_cls"]}),
    (
        Reshape362,
        [((1, 32, 256, 80), torch.float32)],
        {"model_name": ["pt_phi_2_causal_lm", "pt_phi_2_pytdml_causal_lm"]},
    ),
    (
        Reshape363,
        [((1, 32, 80, 256), torch.float32)],
        {"model_name": ["pt_phi_2_causal_lm", "pt_phi_2_pytdml_causal_lm"]},
    ),
    (Reshape364, [((32, 256, 80), torch.float32)], {"model_name": ["pt_phi_2_causal_lm", "pt_phi_2_pytdml_causal_lm"]}),
    (
        Reshape275,
        [((1, 256, 32, 80), torch.float32)],
        {"model_name": ["pt_phi_2_causal_lm", "pt_phi_2_pytdml_causal_lm"]},
    ),
    (Reshape365, [((256, 10240), torch.float32)], {"model_name": ["pt_phi_2_causal_lm", "pt_phi_2_pytdml_causal_lm"]}),
    (Reshape366, [((1, 11, 2560), torch.float32)], {"model_name": ["pt_phi_2_seq_cls", "pt_phi_2_pytdml_seq_cls"]}),
    (Reshape367, [((1, 11, 2560), torch.float32)], {"model_name": ["pt_phi_2_seq_cls", "pt_phi_2_pytdml_seq_cls"]}),
    (Reshape368, [((11, 2560), torch.float32)], {"model_name": ["pt_phi_2_seq_cls", "pt_phi_2_pytdml_seq_cls"]}),
    (Reshape369, [((1, 32, 11, 80), torch.float32)], {"model_name": ["pt_phi_2_seq_cls", "pt_phi_2_pytdml_seq_cls"]}),
    (Reshape370, [((32, 11, 11), torch.float32)], {"model_name": ["pt_phi_2_seq_cls", "pt_phi_2_pytdml_seq_cls"]}),
    (Reshape371, [((1, 32, 11, 11), torch.float32)], {"model_name": ["pt_phi_2_seq_cls", "pt_phi_2_pytdml_seq_cls"]}),
    (Reshape372, [((1, 32, 80, 11), torch.float32)], {"model_name": ["pt_phi_2_seq_cls", "pt_phi_2_pytdml_seq_cls"]}),
    (Reshape373, [((32, 11, 80), torch.float32)], {"model_name": ["pt_phi_2_seq_cls", "pt_phi_2_pytdml_seq_cls"]}),
    (Reshape366, [((1, 11, 32, 80), torch.float32)], {"model_name": ["pt_phi_2_seq_cls", "pt_phi_2_pytdml_seq_cls"]}),
    (Reshape374, [((11, 10240), torch.float32)], {"model_name": ["pt_phi_2_seq_cls", "pt_phi_2_pytdml_seq_cls"]}),
    (Reshape375, [((1, 29, 1024), torch.float32)], {"model_name": ["pt_qwen_chat"]}),
    (Reshape376, [((1, 29, 1024), torch.float32)], {"model_name": ["pt_qwen_chat"]}),
    (Reshape377, [((29, 1024), torch.float32)], {"model_name": ["pt_qwen_chat"]}),
    (Reshape378, [((1, 16, 29, 64), torch.float32)], {"model_name": ["pt_qwen_chat"]}),
    (Reshape379, [((16, 29, 29), torch.float32)], {"model_name": ["pt_qwen_chat", "pt_Qwen_Qwen2_5_3B"]}),
    (Reshape380, [((1, 16, 29, 29), torch.float32)], {"model_name": ["pt_qwen_chat", "pt_Qwen_Qwen2_5_3B"]}),
    (Reshape381, [((1, 16, 64, 29), torch.float32)], {"model_name": ["pt_qwen_chat"]}),
    (Reshape382, [((16, 29, 64), torch.float32)], {"model_name": ["pt_qwen_chat"]}),
    (Reshape375, [((1, 29, 16, 64), torch.float32)], {"model_name": ["pt_qwen_chat"]}),
    (Reshape383, [((29, 2816), torch.float32)], {"model_name": ["pt_qwen_chat"]}),
    (Reshape384, [((1, 6, 1024), torch.float32)], {"model_name": ["pt_qwen_causal_lm"]}),
    (Reshape385, [((1, 6, 1024), torch.float32)], {"model_name": ["pt_qwen_causal_lm"]}),
    (Reshape386, [((6, 1024), torch.float32)], {"model_name": ["pt_qwen_causal_lm"]}),
    (Reshape387, [((1, 16, 6, 64), torch.float32)], {"model_name": ["pt_qwen_causal_lm"]}),
    (Reshape388, [((16, 6, 6), torch.float32)], {"model_name": ["pt_qwen_causal_lm"]}),
    (Reshape389, [((1, 16, 6, 6), torch.float32)], {"model_name": ["pt_qwen_causal_lm"]}),
    (Reshape390, [((1, 16, 64, 6), torch.float32)], {"model_name": ["pt_qwen_causal_lm"]}),
    (Reshape391, [((16, 6, 64), torch.float32)], {"model_name": ["pt_qwen_causal_lm"]}),
    (Reshape384, [((1, 6, 16, 64), torch.float32)], {"model_name": ["pt_qwen_causal_lm"]}),
    (Reshape392, [((6, 2816), torch.float32)], {"model_name": ["pt_qwen_causal_lm"]}),
    (
        Reshape393,
        [((1, 35, 1536), torch.float32)],
        {"model_name": ["pt_Qwen_Qwen2_5_Coder_1_5B_Instruct", "pt_Qwen_Qwen2_5_Coder_1_5B"]},
    ),
    (
        Reshape394,
        [((1, 35, 1536), torch.float32)],
        {"model_name": ["pt_Qwen_Qwen2_5_Coder_1_5B_Instruct", "pt_Qwen_Qwen2_5_Coder_1_5B"]},
    ),
    (
        Reshape395,
        [((35, 1536), torch.float32)],
        {"model_name": ["pt_Qwen_Qwen2_5_Coder_1_5B_Instruct", "pt_Qwen_Qwen2_5_Coder_1_5B"]},
    ),
    (
        Reshape396,
        [((1, 12, 35, 128), torch.float32)],
        {"model_name": ["pt_Qwen_Qwen2_5_Coder_1_5B_Instruct", "pt_Qwen_Qwen2_5_Coder_1_5B"]},
    ),
    (
        Reshape397,
        [((35, 256), torch.float32)],
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
        Reshape398,
        [((1, 35, 256), torch.float32)],
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
        Reshape396,
        [((1, 2, 6, 35, 128), torch.float32)],
        {"model_name": ["pt_Qwen_Qwen2_5_Coder_1_5B_Instruct", "pt_Qwen_Qwen2_5_Coder_1_5B"]},
    ),
    (
        Reshape399,
        [((1, 2, 6, 35, 128), torch.float32)],
        {"model_name": ["pt_Qwen_Qwen2_5_Coder_1_5B_Instruct", "pt_Qwen_Qwen2_5_Coder_1_5B"]},
    ),
    (
        Reshape400,
        [((12, 35, 35), torch.float32)],
        {"model_name": ["pt_Qwen_Qwen2_5_Coder_1_5B_Instruct", "pt_Qwen_Qwen2_5_Coder_1_5B"]},
    ),
    (
        Reshape401,
        [((1, 12, 35, 35), torch.float32)],
        {"model_name": ["pt_Qwen_Qwen2_5_Coder_1_5B_Instruct", "pt_Qwen_Qwen2_5_Coder_1_5B"]},
    ),
    (
        Reshape402,
        [((1, 12, 128, 35), torch.float32)],
        {"model_name": ["pt_Qwen_Qwen2_5_Coder_1_5B_Instruct", "pt_Qwen_Qwen2_5_Coder_1_5B"]},
    ),
    (
        Reshape399,
        [((12, 35, 128), torch.float32)],
        {"model_name": ["pt_Qwen_Qwen2_5_Coder_1_5B_Instruct", "pt_Qwen_Qwen2_5_Coder_1_5B"]},
    ),
    (
        Reshape393,
        [((1, 35, 12, 128), torch.float32)],
        {"model_name": ["pt_Qwen_Qwen2_5_Coder_1_5B_Instruct", "pt_Qwen_Qwen2_5_Coder_1_5B"]},
    ),
    (
        Reshape403,
        [((35, 8960), torch.float32)],
        {"model_name": ["pt_Qwen_Qwen2_5_Coder_1_5B_Instruct", "pt_Qwen_Qwen2_5_Coder_1_5B"]},
    ),
    (
        Reshape404,
        [((1, 35, 2048), torch.float32)],
        {"model_name": ["pt_Qwen_Qwen2_5_Coder_3B", "pt_Qwen_Qwen2_5_Coder_3B_Instruct"]},
    ),
    (
        Reshape405,
        [((1, 35, 2048), torch.float32)],
        {"model_name": ["pt_Qwen_Qwen2_5_Coder_3B", "pt_Qwen_Qwen2_5_Coder_3B_Instruct"]},
    ),
    (
        Reshape406,
        [((35, 2048), torch.float32)],
        {"model_name": ["pt_Qwen_Qwen2_5_Coder_3B", "pt_Qwen_Qwen2_5_Coder_3B_Instruct"]},
    ),
    (
        Reshape407,
        [((1, 16, 35, 128), torch.float32)],
        {"model_name": ["pt_Qwen_Qwen2_5_Coder_3B", "pt_Qwen_Qwen2_5_Coder_3B_Instruct"]},
    ),
    (
        Reshape407,
        [((1, 2, 8, 35, 128), torch.float32)],
        {"model_name": ["pt_Qwen_Qwen2_5_Coder_3B", "pt_Qwen_Qwen2_5_Coder_3B_Instruct"]},
    ),
    (
        Reshape408,
        [((1, 2, 8, 35, 128), torch.float32)],
        {"model_name": ["pt_Qwen_Qwen2_5_Coder_3B", "pt_Qwen_Qwen2_5_Coder_3B_Instruct"]},
    ),
    (
        Reshape409,
        [((16, 35, 35), torch.float32)],
        {"model_name": ["pt_Qwen_Qwen2_5_Coder_3B", "pt_Qwen_Qwen2_5_Coder_3B_Instruct"]},
    ),
    (
        Reshape410,
        [((1, 16, 35, 35), torch.float32)],
        {"model_name": ["pt_Qwen_Qwen2_5_Coder_3B", "pt_Qwen_Qwen2_5_Coder_3B_Instruct"]},
    ),
    (
        Reshape411,
        [((1, 16, 128, 35), torch.float32)],
        {"model_name": ["pt_Qwen_Qwen2_5_Coder_3B", "pt_Qwen_Qwen2_5_Coder_3B_Instruct"]},
    ),
    (
        Reshape408,
        [((16, 35, 128), torch.float32)],
        {"model_name": ["pt_Qwen_Qwen2_5_Coder_3B", "pt_Qwen_Qwen2_5_Coder_3B_Instruct"]},
    ),
    (
        Reshape404,
        [((1, 35, 16, 128), torch.float32)],
        {"model_name": ["pt_Qwen_Qwen2_5_Coder_3B", "pt_Qwen_Qwen2_5_Coder_3B_Instruct"]},
    ),
    (
        Reshape412,
        [((35, 11008), torch.float32)],
        {"model_name": ["pt_Qwen_Qwen2_5_Coder_3B", "pt_Qwen_Qwen2_5_Coder_3B_Instruct"]},
    ),
    (
        Reshape413,
        [((1, 35, 3584), torch.float32)],
        {"model_name": ["pt_Qwen_Qwen2_5_Coder_7B", "pt_Qwen_Qwen2_5_Coder_7B_Instruct"]},
    ),
    (
        Reshape414,
        [((1, 35, 3584), torch.float32)],
        {"model_name": ["pt_Qwen_Qwen2_5_Coder_7B", "pt_Qwen_Qwen2_5_Coder_7B_Instruct"]},
    ),
    (
        Reshape415,
        [((35, 3584), torch.float32)],
        {"model_name": ["pt_Qwen_Qwen2_5_Coder_7B", "pt_Qwen_Qwen2_5_Coder_7B_Instruct"]},
    ),
    (
        Reshape416,
        [((1, 28, 35, 128), torch.float32)],
        {"model_name": ["pt_Qwen_Qwen2_5_Coder_7B", "pt_Qwen_Qwen2_5_Coder_7B_Instruct"]},
    ),
    (
        Reshape417,
        [((35, 512), torch.float32)],
        {"model_name": ["pt_Qwen_Qwen2_5_Coder_7B", "pt_Qwen_Qwen2_5_Coder_7B_Instruct"]},
    ),
    (
        Reshape418,
        [((1, 35, 512), torch.float32)],
        {"model_name": ["pt_Qwen_Qwen2_5_Coder_7B", "pt_Qwen_Qwen2_5_Coder_7B_Instruct"]},
    ),
    (
        Reshape416,
        [((1, 4, 7, 35, 128), torch.float32)],
        {"model_name": ["pt_Qwen_Qwen2_5_Coder_7B", "pt_Qwen_Qwen2_5_Coder_7B_Instruct"]},
    ),
    (
        Reshape419,
        [((1, 4, 7, 35, 128), torch.float32)],
        {"model_name": ["pt_Qwen_Qwen2_5_Coder_7B", "pt_Qwen_Qwen2_5_Coder_7B_Instruct"]},
    ),
    (
        Reshape420,
        [((28, 35, 35), torch.float32)],
        {"model_name": ["pt_Qwen_Qwen2_5_Coder_7B", "pt_Qwen_Qwen2_5_Coder_7B_Instruct"]},
    ),
    (
        Reshape421,
        [((1, 28, 35, 35), torch.float32)],
        {"model_name": ["pt_Qwen_Qwen2_5_Coder_7B", "pt_Qwen_Qwen2_5_Coder_7B_Instruct"]},
    ),
    (
        Reshape422,
        [((1, 28, 128, 35), torch.float32)],
        {"model_name": ["pt_Qwen_Qwen2_5_Coder_7B", "pt_Qwen_Qwen2_5_Coder_7B_Instruct"]},
    ),
    (
        Reshape419,
        [((28, 35, 128), torch.float32)],
        {"model_name": ["pt_Qwen_Qwen2_5_Coder_7B", "pt_Qwen_Qwen2_5_Coder_7B_Instruct"]},
    ),
    (
        Reshape413,
        [((1, 35, 28, 128), torch.float32)],
        {"model_name": ["pt_Qwen_Qwen2_5_Coder_7B", "pt_Qwen_Qwen2_5_Coder_7B_Instruct"]},
    ),
    (
        Reshape423,
        [((35, 18944), torch.float32)],
        {"model_name": ["pt_Qwen_Qwen2_5_Coder_7B", "pt_Qwen_Qwen2_5_Coder_7B_Instruct"]},
    ),
    (Reshape424, [((1, 35, 896), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_Coder_0_5B"]}),
    (Reshape425, [((1, 35, 896), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_Coder_0_5B"]}),
    (Reshape426, [((35, 896), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_Coder_0_5B"]}),
    (Reshape427, [((1, 14, 35, 64), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_Coder_0_5B"]}),
    (Reshape428, [((35, 128), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_Coder_0_5B"]}),
    (Reshape429, [((1, 35, 128), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_Coder_0_5B"]}),
    (Reshape427, [((1, 2, 7, 35, 64), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_Coder_0_5B"]}),
    (Reshape430, [((1, 2, 7, 35, 64), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_Coder_0_5B"]}),
    (Reshape431, [((14, 35, 35), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_Coder_0_5B"]}),
    (Reshape432, [((1, 14, 35, 35), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_Coder_0_5B"]}),
    (Reshape433, [((1, 14, 64, 35), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_Coder_0_5B"]}),
    (Reshape430, [((14, 35, 64), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_Coder_0_5B"]}),
    (Reshape424, [((1, 35, 14, 64), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_Coder_0_5B"]}),
    (Reshape434, [((35, 4864), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_Coder_0_5B"]}),
    (Reshape435, [((1, 39, 1536), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_1_5B_Instruct"]}),
    (Reshape436, [((1, 39, 1536), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_1_5B_Instruct"]}),
    (Reshape437, [((39, 1536), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_1_5B_Instruct"]}),
    (Reshape438, [((1, 12, 39, 128), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_1_5B_Instruct"]}),
    (
        Reshape439,
        [((39, 256), torch.float32)],
        {"model_name": ["pt_Qwen_Qwen2_5_1_5B_Instruct", "pt_Qwen_Qwen2_5_3B_Instruct"]},
    ),
    (
        Reshape440,
        [((1, 39, 256), torch.float32)],
        {"model_name": ["pt_Qwen_Qwen2_5_1_5B_Instruct", "pt_Qwen_Qwen2_5_3B_Instruct"]},
    ),
    (Reshape438, [((1, 2, 6, 39, 128), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_1_5B_Instruct"]}),
    (Reshape441, [((1, 2, 6, 39, 128), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_1_5B_Instruct"]}),
    (Reshape442, [((12, 39, 39), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_1_5B_Instruct"]}),
    (Reshape443, [((1, 12, 39, 39), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_1_5B_Instruct"]}),
    (Reshape444, [((1, 12, 128, 39), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_1_5B_Instruct"]}),
    (Reshape441, [((12, 39, 128), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_1_5B_Instruct"]}),
    (Reshape435, [((1, 39, 12, 128), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_1_5B_Instruct"]}),
    (Reshape445, [((39, 8960), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_1_5B_Instruct"]}),
    (Reshape446, [((1, 29, 1536), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_1_5B"]}),
    (Reshape447, [((1, 29, 1536), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_1_5B"]}),
    (Reshape448, [((29, 1536), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_1_5B"]}),
    (Reshape449, [((1, 12, 29, 128), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_1_5B"]}),
    (Reshape450, [((29, 256), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_1_5B", "pt_Qwen_Qwen2_5_3B"]}),
    (Reshape451, [((1, 29, 256), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_1_5B", "pt_Qwen_Qwen2_5_3B"]}),
    (Reshape449, [((1, 2, 6, 29, 128), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_1_5B"]}),
    (Reshape452, [((1, 2, 6, 29, 128), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_1_5B"]}),
    (Reshape453, [((12, 29, 29), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_1_5B"]}),
    (Reshape454, [((1, 12, 29, 29), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_1_5B"]}),
    (Reshape455, [((1, 12, 128, 29), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_1_5B"]}),
    (Reshape452, [((12, 29, 128), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_1_5B"]}),
    (Reshape446, [((1, 29, 12, 128), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_1_5B"]}),
    (Reshape456, [((29, 8960), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_1_5B"]}),
    (Reshape457, [((1, 29, 3584), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_7B"]}),
    (Reshape458, [((1, 29, 3584), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_7B"]}),
    (Reshape459, [((29, 3584), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_7B"]}),
    (Reshape460, [((1, 28, 29, 128), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_7B"]}),
    (Reshape461, [((29, 512), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_7B"]}),
    (Reshape462, [((1, 29, 512), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_7B"]}),
    (Reshape460, [((1, 4, 7, 29, 128), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_7B"]}),
    (Reshape463, [((1, 4, 7, 29, 128), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_7B"]}),
    (Reshape464, [((28, 29, 29), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_7B"]}),
    (Reshape465, [((1, 28, 29, 29), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_7B"]}),
    (Reshape466, [((1, 28, 128, 29), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_7B"]}),
    (Reshape463, [((28, 29, 128), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_7B"]}),
    (Reshape457, [((1, 29, 28, 128), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_7B"]}),
    (Reshape467, [((29, 18944), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_7B"]}),
    (Reshape468, [((1, 39, 896), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_0_5B_Instruct"]}),
    (Reshape469, [((1, 39, 896), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_0_5B_Instruct"]}),
    (Reshape470, [((39, 896), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_0_5B_Instruct"]}),
    (Reshape471, [((1, 14, 39, 64), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_0_5B_Instruct"]}),
    (Reshape472, [((39, 128), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_0_5B_Instruct"]}),
    (Reshape473, [((1, 39, 128), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_0_5B_Instruct"]}),
    (Reshape471, [((1, 2, 7, 39, 64), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_0_5B_Instruct"]}),
    (Reshape474, [((1, 2, 7, 39, 64), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_0_5B_Instruct"]}),
    (Reshape475, [((14, 39, 39), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_0_5B_Instruct"]}),
    (Reshape476, [((1, 14, 39, 39), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_0_5B_Instruct"]}),
    (Reshape477, [((1, 14, 64, 39), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_0_5B_Instruct"]}),
    (Reshape474, [((14, 39, 64), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_0_5B_Instruct"]}),
    (Reshape468, [((1, 39, 14, 64), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_0_5B_Instruct"]}),
    (Reshape478, [((39, 4864), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_0_5B_Instruct"]}),
    (Reshape479, [((1, 29, 896), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_0_5B"]}),
    (Reshape480, [((1, 29, 896), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_0_5B"]}),
    (Reshape481, [((29, 896), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_0_5B"]}),
    (Reshape482, [((1, 14, 29, 64), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_0_5B"]}),
    (Reshape483, [((29, 128), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_0_5B"]}),
    (Reshape484, [((1, 29, 128), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_0_5B"]}),
    (Reshape482, [((1, 2, 7, 29, 64), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_0_5B"]}),
    (Reshape485, [((1, 2, 7, 29, 64), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_0_5B"]}),
    (Reshape486, [((14, 29, 29), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_0_5B"]}),
    (Reshape487, [((1, 14, 29, 29), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_0_5B"]}),
    (Reshape488, [((1, 14, 64, 29), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_0_5B"]}),
    (Reshape485, [((14, 29, 64), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_0_5B"]}),
    (Reshape479, [((1, 29, 14, 64), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_0_5B"]}),
    (Reshape489, [((29, 4864), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_0_5B"]}),
    (Reshape490, [((1, 39, 2048), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_3B_Instruct"]}),
    (Reshape491, [((1, 39, 2048), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_3B_Instruct"]}),
    (Reshape492, [((39, 2048), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_3B_Instruct"]}),
    (Reshape493, [((1, 16, 39, 128), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_3B_Instruct"]}),
    (Reshape493, [((1, 2, 8, 39, 128), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_3B_Instruct"]}),
    (Reshape494, [((1, 2, 8, 39, 128), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_3B_Instruct"]}),
    (Reshape495, [((16, 39, 39), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_3B_Instruct"]}),
    (Reshape496, [((1, 16, 39, 39), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_3B_Instruct"]}),
    (Reshape497, [((1, 16, 128, 39), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_3B_Instruct"]}),
    (Reshape494, [((16, 39, 128), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_3B_Instruct"]}),
    (Reshape490, [((1, 39, 16, 128), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_3B_Instruct"]}),
    (Reshape498, [((39, 11008), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_3B_Instruct"]}),
    (Reshape499, [((1, 39, 3584), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_7B_Instruct"]}),
    (Reshape500, [((1, 39, 3584), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_7B_Instruct"]}),
    (Reshape501, [((39, 3584), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_7B_Instruct"]}),
    (Reshape502, [((1, 28, 39, 128), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_7B_Instruct"]}),
    (Reshape503, [((39, 512), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_7B_Instruct"]}),
    (Reshape504, [((1, 39, 512), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_7B_Instruct"]}),
    (Reshape502, [((1, 4, 7, 39, 128), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_7B_Instruct"]}),
    (Reshape505, [((1, 4, 7, 39, 128), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_7B_Instruct"]}),
    (Reshape506, [((28, 39, 39), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_7B_Instruct"]}),
    (Reshape507, [((1, 28, 39, 39), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_7B_Instruct"]}),
    (Reshape508, [((1, 28, 128, 39), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_7B_Instruct"]}),
    (Reshape505, [((28, 39, 128), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_7B_Instruct"]}),
    (Reshape499, [((1, 39, 28, 128), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_7B_Instruct"]}),
    (Reshape509, [((39, 18944), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_7B_Instruct"]}),
    (Reshape510, [((1, 29, 2048), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_3B"]}),
    (Reshape511, [((1, 29, 2048), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_3B"]}),
    (Reshape512, [((29, 2048), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_3B"]}),
    (Reshape513, [((1, 16, 29, 128), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_3B"]}),
    (Reshape513, [((1, 2, 8, 29, 128), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_3B"]}),
    (Reshape514, [((1, 2, 8, 29, 128), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_3B"]}),
    (Reshape515, [((1, 16, 128, 29), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_3B"]}),
    (Reshape514, [((16, 29, 128), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_3B"]}),
    (Reshape510, [((1, 29, 16, 128), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_3B"]}),
    (Reshape516, [((29, 11008), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_3B"]}),
    (Reshape517, [((1, 768, 128), torch.float32)], {"model_name": ["pt_squeezebert"]}),
    (Reshape167, [((1, 768, 128), torch.float32)], {"model_name": ["pt_squeezebert"]}),
    (Reshape518, [((768, 768, 1), torch.float32)], {"model_name": ["pt_squeezebert"]}),
    (Reshape519, [((1, 768, 128, 1), torch.float32)], {"model_name": ["pt_squeezebert"]}),
    (
        Reshape520,
        [((1, 1), torch.int32)],
        {
            "model_name": [
                "pt_t5_large",
                "pt_t5_base",
                "pt_google_flan_t5_base",
                "pt_t5_small",
                "pt_google_flan_t5_small",
            ]
        },
    ),
    (Reshape29, [((1, 1, 1024), torch.float32)], {"model_name": ["pt_t5_large", "pt_vit_large_patch16_224"]}),
    (
        Reshape521,
        [((1, 1, 1024), torch.float32)],
        {"model_name": ["pt_vision_perceiver_learned", "pt_vision_perceiver_conv", "pt_vision_perceiver_fourier"]},
    ),
    (Reshape522, [((1, 16, 1, 64), torch.float32)], {"model_name": ["pt_t5_large"]}),
    (Reshape523, [((16, 1, 1), torch.float32)], {"model_name": ["pt_t5_large"]}),
    (Reshape524, [((1, 16, 1, 1), torch.float32)], {"model_name": ["pt_t5_large"]}),
    (Reshape525, [((1, 16, 1, 1), torch.float32)], {"model_name": ["pt_ssd300_resnet50"]}),
    (Reshape526, [((1, 16, 64, 1), torch.float32)], {"model_name": ["pt_t5_large"]}),
    (Reshape527, [((16, 1, 64), torch.float32)], {"model_name": ["pt_t5_large"]}),
    (Reshape29, [((1, 1, 16, 64), torch.float32)], {"model_name": ["pt_t5_large"]}),
    (Reshape528, [((16, 1, 256), torch.float32)], {"model_name": ["pt_t5_large"]}),
    (Reshape529, [((1, 16, 1, 256), torch.float32)], {"model_name": ["pt_t5_large"]}),
    (Reshape530, [((1, 768), torch.float32)], {"model_name": ["pt_t5_base", "pt_google_flan_t5_base"]}),
    (Reshape531, [((1, 768), torch.float32)], {"model_name": ["pt_t5_base", "pt_google_flan_t5_base"]}),
    (Reshape532, [((1, 12, 1, 64), torch.float32)], {"model_name": ["pt_t5_base", "pt_google_flan_t5_base"]}),
    (Reshape533, [((12, 1, 1), torch.float32)], {"model_name": ["pt_t5_base", "pt_google_flan_t5_base"]}),
    (Reshape534, [((1, 12, 1, 1), torch.float32)], {"model_name": ["pt_t5_base", "pt_google_flan_t5_base"]}),
    (Reshape535, [((1, 12, 64, 1), torch.float32)], {"model_name": ["pt_t5_base", "pt_google_flan_t5_base"]}),
    (Reshape536, [((12, 1, 64), torch.float32)], {"model_name": ["pt_t5_base", "pt_google_flan_t5_base"]}),
    (Reshape160, [((1, 1, 12, 64), torch.float32)], {"model_name": ["pt_t5_base", "pt_google_flan_t5_base"]}),
    (Reshape537, [((12, 1, 256), torch.float32)], {"model_name": ["pt_t5_base", "pt_google_flan_t5_base"]}),
    (Reshape538, [((1, 12, 1, 256), torch.float32)], {"model_name": ["pt_t5_base", "pt_google_flan_t5_base"]}),
    (Reshape539, [((1, 1, 512), torch.float32)], {"model_name": ["pt_t5_small", "pt_google_flan_t5_small"]}),
    (
        Reshape540,
        [((1, 1, 512), torch.float32)],
        {"model_name": ["pt_vision_perceiver_learned", "pt_vision_perceiver_conv", "pt_vision_perceiver_fourier"]},
    ),
    (Reshape541, [((1, 512), torch.float32)], {"model_name": ["pt_t5_small"]}),
    (Reshape542, [((1, 512), torch.float32)], {"model_name": ["pt_t5_small", "pt_google_flan_t5_small"]}),
    (Reshape543, [((1, 8, 1, 64), torch.float32)], {"model_name": ["pt_t5_small"]}),
    (Reshape544, [((8, 1, 1), torch.float32)], {"model_name": ["pt_t5_small"]}),
    (Reshape545, [((1, 8, 1, 1), torch.float32)], {"model_name": ["pt_t5_small"]}),
    (Reshape546, [((1, 8, 64, 1), torch.float32)], {"model_name": ["pt_t5_small"]}),
    (Reshape547, [((8, 1, 64), torch.float32)], {"model_name": ["pt_t5_small"]}),
    (Reshape539, [((1, 1, 8, 64), torch.float32)], {"model_name": ["pt_t5_small"]}),
    (Reshape240, [((1, 384), torch.float32)], {"model_name": ["pt_google_flan_t5_small"]}),
    (Reshape548, [((6, 1, 1), torch.float32)], {"model_name": ["pt_google_flan_t5_small"]}),
    (Reshape549, [((1, 6, 1, 1), torch.float32)], {"model_name": ["pt_google_flan_t5_small"]}),
    (Reshape550, [((1, 6, 64, 1), torch.float32)], {"model_name": ["pt_google_flan_t5_small"]}),
    (Reshape238, [((6, 1, 64), torch.float32)], {"model_name": ["pt_google_flan_t5_small"]}),
    (Reshape551, [((1, 256, 6, 6), torch.float32)], {"model_name": ["pt_alexnet_torchhub", "pt_rcnn"]}),
    (Reshape552, [((1, 384, 14, 14), torch.float32)], {"model_name": ["pt_deit_small_patch16_224"]}),
    (Reshape553, [((1, 197, 384), torch.float32)], {"model_name": ["pt_deit_small_patch16_224"]}),
    (Reshape554, [((1, 197, 384), torch.float32)], {"model_name": ["pt_deit_small_patch16_224"]}),
    (Reshape555, [((197, 384), torch.float32)], {"model_name": ["pt_deit_small_patch16_224"]}),
    (Reshape556, [((1, 6, 197, 64), torch.float32)], {"model_name": ["pt_deit_small_patch16_224"]}),
    (Reshape557, [((6, 197, 197), torch.float32)], {"model_name": ["pt_deit_small_patch16_224"]}),
    (Reshape558, [((1, 6, 197, 197), torch.float32)], {"model_name": ["pt_deit_small_patch16_224"]}),
    (Reshape559, [((1, 6, 64, 197), torch.float32)], {"model_name": ["pt_deit_small_patch16_224"]}),
    (Reshape560, [((6, 197, 64), torch.float32)], {"model_name": ["pt_deit_small_patch16_224"]}),
    (Reshape553, [((1, 197, 6, 64), torch.float32)], {"model_name": ["pt_deit_small_patch16_224"]}),
    (Reshape242, [((1, 1, 384), torch.float32)], {"model_name": ["pt_deit_small_patch16_224"]}),
    (
        Reshape561,
        [((1, 768, 14, 14), torch.float32)],
        {
            "model_name": [
                "pt_deit_base_patch16_224",
                "pt_deit_base_distilled_patch16_224",
                "pt_mixer_b16_224_miil_in21k",
                "pt_mixer_b16_224",
                "pt_mixer_b16_224_in21k",
                "pt_mixer_b16_224_miil",
                "pt_vit_base_patch16_224",
            ]
        },
    ),
    (
        Reshape562,
        [((1, 197, 768), torch.float32)],
        {"model_name": ["pt_deit_base_patch16_224", "pt_deit_base_distilled_patch16_224", "pt_vit_base_patch16_224"]},
    ),
    (
        Reshape563,
        [((1, 197, 768), torch.float32)],
        {"model_name": ["pt_deit_base_patch16_224", "pt_deit_base_distilled_patch16_224", "pt_vit_base_patch16_224"]},
    ),
    (
        Reshape564,
        [((197, 768), torch.float32)],
        {"model_name": ["pt_deit_base_patch16_224", "pt_deit_base_distilled_patch16_224", "pt_vit_base_patch16_224"]},
    ),
    (
        Reshape565,
        [((1, 12, 197, 64), torch.float32)],
        {"model_name": ["pt_deit_base_patch16_224", "pt_deit_base_distilled_patch16_224", "pt_vit_base_patch16_224"]},
    ),
    (
        Reshape566,
        [((12, 197, 197), torch.float32)],
        {"model_name": ["pt_deit_base_patch16_224", "pt_deit_base_distilled_patch16_224", "pt_vit_base_patch16_224"]},
    ),
    (
        Reshape567,
        [((1, 12, 197, 197), torch.float32)],
        {"model_name": ["pt_deit_base_patch16_224", "pt_deit_base_distilled_patch16_224", "pt_vit_base_patch16_224"]},
    ),
    (
        Reshape568,
        [((1, 12, 64, 197), torch.float32)],
        {"model_name": ["pt_deit_base_patch16_224", "pt_deit_base_distilled_patch16_224", "pt_vit_base_patch16_224"]},
    ),
    (
        Reshape569,
        [((12, 197, 64), torch.float32)],
        {"model_name": ["pt_deit_base_patch16_224", "pt_deit_base_distilled_patch16_224", "pt_vit_base_patch16_224"]},
    ),
    (
        Reshape562,
        [((1, 197, 12, 64), torch.float32)],
        {"model_name": ["pt_deit_base_patch16_224", "pt_deit_base_distilled_patch16_224", "pt_vit_base_patch16_224"]},
    ),
    (Reshape570, [((1, 192, 14, 14), torch.float32)], {"model_name": ["pt_deit_tiny_patch16_224"]}),
    (Reshape571, [((1, 197, 192), torch.float32)], {"model_name": ["pt_deit_tiny_patch16_224"]}),
    (Reshape572, [((1, 197, 192), torch.float32)], {"model_name": ["pt_deit_tiny_patch16_224"]}),
    (Reshape573, [((197, 192), torch.float32)], {"model_name": ["pt_deit_tiny_patch16_224"]}),
    (Reshape574, [((1, 3, 197, 64), torch.float32)], {"model_name": ["pt_deit_tiny_patch16_224"]}),
    (Reshape575, [((3, 197, 197), torch.float32)], {"model_name": ["pt_deit_tiny_patch16_224"]}),
    (Reshape576, [((1, 3, 197, 197), torch.float32)], {"model_name": ["pt_deit_tiny_patch16_224"]}),
    (Reshape577, [((1, 3, 64, 197), torch.float32)], {"model_name": ["pt_deit_tiny_patch16_224"]}),
    (Reshape578, [((3, 197, 64), torch.float32)], {"model_name": ["pt_deit_tiny_patch16_224"]}),
    (Reshape571, [((1, 197, 3, 64), torch.float32)], {"model_name": ["pt_deit_tiny_patch16_224"]}),
    (Reshape579, [((1, 1, 192), torch.float32)], {"model_name": ["pt_deit_tiny_patch16_224"]}),
    (
        Reshape580,
        [((1, 1024, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_densenet121",
                "pt_googlenet",
                "pt_mobilenet_v1_224",
                "pt_mobilenetv3_small_100",
                "pt_ese_vovnet19b_dw",
                "pt_ese_vovnet39b",
                "pt_ese_vovnet99b",
            ]
        },
    ),
    (
        Reshape29,
        [((1, 1024, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_mobilenet_v1_basic",
                "pt_vovnet39",
                "vovnet_57_stigma_pt",
                "pt_vovnet_39_stigma",
                "pt_vovnet57",
            ]
        },
    ),
    (Reshape581, [((1, 2208, 1, 1), torch.float32)], {"model_name": ["pt_densenet_161"]}),
    (Reshape582, [((1, 1664, 1, 1), torch.float32)], {"model_name": ["pt_densenet_169"]}),
    (Reshape583, [((1, 1920, 1, 1), torch.float32)], {"model_name": ["pt_densenet_201"]}),
    (
        Reshape584,
        [((1, 1000, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_dla102x",
                "pt_dla169",
                "pt_dla60",
                "pt_dla46_c",
                "pt_dla102x2",
                "pt_dla46x_c",
                "pt_dla60x",
                "pt_dla102",
                "pt_dla34",
                "pt_dla60x_c",
            ]
        },
    ),
    (
        Reshape585,
        [((48, 1, 3, 3), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_b4_timm",
                "pt_efficientnet_b4_torchvision",
                "pt_ghostnet_100",
                "pt_mobilenet_v1_192",
                "mobilenetv2_96",
            ]
        },
    ),
    (
        Reshape586,
        [((24, 1, 3, 3), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_b4_timm",
                "pt_efficientnet_b4_torchvision",
                "pt_ghostnet_100",
                "pt_mobilenet_v1_192",
                "mobilenetv2_160",
            ]
        },
    ),
    (
        Reshape587,
        [((144, 1, 3, 3), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_b4_timm",
                "pt_efficientnet_b4_torchvision",
                "pt_efficientnet_b0_torchvision",
                "pt_efficientnet_b0_timm",
                "mobilenetv2_basic",
                "mobilenetv2_deeplabv3",
                "mobilenetv2_160",
                "mobilenetv2_96",
                "mobilenetv2_timm",
                "mobilenetv2_224",
            ]
        },
    ),
    (
        Reshape588,
        [((192, 1, 3, 3), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_b4_timm",
                "pt_efficientnet_b4_torchvision",
                "pt_mobilenet_v1_192",
                "mobilenetv2_basic",
                "mobilenetv2_deeplabv3",
                "mobilenetv2_96",
                "mobilenetv2_timm",
                "mobilenetv2_224",
                "pt_ese_vovnet19b_dw",
            ]
        },
    ),
    (
        Reshape589,
        [((192, 1, 5, 5), torch.float32)],
        {"model_name": ["pt_efficientnet_b4_timm", "pt_efficientnet_b4_torchvision"]},
    ),
    (
        Reshape590,
        [((336, 1, 5, 5), torch.float32)],
        {"model_name": ["pt_efficientnet_b4_timm", "pt_efficientnet_b4_torchvision"]},
    ),
    (
        Reshape591,
        [((336, 1, 3, 3), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_b4_timm",
                "pt_efficientnet_b4_torchvision",
                "pt_ghostnet_100",
                "mobilenetv2_96",
            ]
        },
    ),
    (
        Reshape592,
        [((672, 1, 3, 3), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_b4_timm",
                "pt_efficientnet_b4_torchvision",
                "pt_mobilenet_v3_large",
                "pt_mobilenetv3_large_100",
            ]
        },
    ),
    (
        Reshape593,
        [((672, 1, 5, 5), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_b4_timm",
                "pt_efficientnet_b4_torchvision",
                "pt_efficientnet_b0_torchvision",
                "pt_efficientnet_b0_timm",
                "pt_ghostnet_100",
                "pt_mobilenet_v3_large",
                "pt_mobilenetv3_large_100",
            ]
        },
    ),
    (
        Reshape594,
        [((960, 1, 5, 5), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_b4_timm",
                "pt_efficientnet_b4_torchvision",
                "pt_mobilenet_v3_large",
                "pt_mobilenetv3_large_100",
            ]
        },
    ),
    (
        Reshape595,
        [((1632, 1, 5, 5), torch.float32)],
        {"model_name": ["pt_efficientnet_b4_timm", "pt_efficientnet_b4_torchvision"]},
    ),
    (
        Reshape596,
        [((1632, 1, 3, 3), torch.float32)],
        {"model_name": ["pt_efficientnet_b4_timm", "pt_efficientnet_b4_torchvision"]},
    ),
    (
        Reshape597,
        [((2688, 1, 3, 3), torch.float32)],
        {"model_name": ["pt_efficientnet_b4_timm", "pt_efficientnet_b4_torchvision"]},
    ),
    (
        Reshape598,
        [((1, 1792, 1, 1), torch.float32)],
        {"model_name": ["pt_efficientnet_b4_timm", "pt_efficientnet_b4_torchvision"]},
    ),
    (
        Reshape599,
        [((32, 1, 3, 3), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_b0_torchvision",
                "pt_efficientnet_b0_timm",
                "pt_mobilenet_v1_224",
                "pt_mobilenet_v1_basic",
                "mobilenetv2_basic",
                "mobilenetv2_deeplabv3",
                "mobilenetv2_timm",
                "mobilenetv2_224",
                "pt_yolox_nano",
            ]
        },
    ),
    (
        Reshape600,
        [((96, 1, 3, 3), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_b0_torchvision",
                "pt_efficientnet_b0_timm",
                "pt_mobilenet_v1_192",
                "mobilenetv2_basic",
                "mobilenetv2_deeplabv3",
                "mobilenetv2_160",
                "mobilenetv2_96",
                "mobilenetv2_timm",
                "mobilenetv2_224",
            ]
        },
    ),
    (
        Reshape601,
        [((144, 1, 5, 5), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_b0_torchvision",
                "pt_efficientnet_b0_timm",
                "pt_mobilenet_v3_small",
                "pt_mobilenetv3_small_100",
            ]
        },
    ),
    (
        Reshape602,
        [((240, 1, 5, 5), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_b0_torchvision",
                "pt_efficientnet_b0_timm",
                "pt_mobilenet_v3_small",
                "pt_mobilenetv3_small_100",
            ]
        },
    ),
    (
        Reshape603,
        [((240, 1, 3, 3), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_b0_torchvision",
                "pt_efficientnet_b0_timm",
                "pt_ghostnet_100",
                "pt_mobilenet_v3_large",
                "pt_mobilenetv3_large_100",
            ]
        },
    ),
    (
        Reshape604,
        [((480, 1, 3, 3), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_b0_torchvision",
                "pt_efficientnet_b0_timm",
                "pt_ghostnet_100",
                "pt_mobilenet_v3_large",
                "pt_mobilenetv3_large_100",
            ]
        },
    ),
    (
        Reshape605,
        [((480, 1, 5, 5), torch.float32)],
        {"model_name": ["pt_efficientnet_b0_torchvision", "pt_efficientnet_b0_timm"]},
    ),
    (
        Reshape606,
        [((1152, 1, 5, 5), torch.float32)],
        {"model_name": ["pt_efficientnet_b0_torchvision", "pt_efficientnet_b0_timm"]},
    ),
    (
        Reshape607,
        [((1152, 1, 3, 3), torch.float32)],
        {"model_name": ["pt_efficientnet_b0_torchvision", "pt_efficientnet_b0_timm"]},
    ),
    (
        Reshape608,
        [((1, 1280, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_b0_torchvision",
                "pt_efficientnet_b0_timm",
                "pt_ghostnet_100",
                "mobilenetv2_basic",
                "mobilenetv2_160",
                "mobilenetv2_96",
                "mobilenetv2_timm",
                "mobilenetv2_224",
                "pt_mobilenetv3_large_100",
            ]
        },
    ),
    (Reshape609, [((8, 1, 3, 3), torch.float32)], {"model_name": ["pt_ghostnet_100"]}),
    (Reshape610, [((12, 1, 3, 3), torch.float32)], {"model_name": ["pt_ghostnet_100"]}),
    (
        Reshape611,
        [((16, 1, 3, 3), torch.float32)],
        {
            "model_name": [
                "pt_ghostnet_100",
                "mobilenetv2_96",
                "pt_mobilenet_v3_small",
                "pt_mobilenet_v3_large",
                "pt_mobilenetv3_small_100",
                "pt_mobilenetv3_large_100",
                "pt_yolox_nano",
            ]
        },
    ),
    (Reshape612, [((36, 1, 3, 3), torch.float32)], {"model_name": ["pt_ghostnet_100"]}),
    (
        Reshape613,
        [((72, 1, 5, 5), torch.float32)],
        {"model_name": ["pt_ghostnet_100", "pt_mobilenet_v3_large", "pt_mobilenetv3_large_100"]},
    ),
    (Reshape614, [((20, 1, 3, 3), torch.float32)], {"model_name": ["pt_ghostnet_100"]}),
    (Reshape615, [((24, 1, 5, 5), torch.float32)], {"model_name": ["pt_ghostnet_100"]}),
    (Reshape616, [((60, 1, 3, 3), torch.float32)], {"model_name": ["pt_ghostnet_100"]}),
    (Reshape617, [((120, 1, 3, 3), torch.float32)], {"model_name": ["pt_ghostnet_100"]}),
    (Reshape618, [((40, 1, 3, 3), torch.float32)], {"model_name": ["pt_ghostnet_100"]}),
    (Reshape619, [((100, 1, 3, 3), torch.float32)], {"model_name": ["pt_ghostnet_100"]}),
    (Reshape620, [((92, 1, 3, 3), torch.float32)], {"model_name": ["pt_ghostnet_100"]}),
    (Reshape621, [((56, 1, 3, 3), torch.float32)], {"model_name": ["pt_ghostnet_100"]}),
    (Reshape622, [((80, 1, 3, 3), torch.float32)], {"model_name": ["pt_ghostnet_100"]}),
    (Reshape623, [((112, 1, 5, 5), torch.float32)], {"model_name": ["pt_ghostnet_100"]}),
    (Reshape624, [((1, 1, 224, 224), torch.float32)], {"model_name": ["pt_googlenet"]}),
    (
        Reshape625,
        [((1, 2048, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_hrnet_timm_hrnet_w18",
                "pt_hrnet_timm_hrnet_w30",
                "pt_hrnet_timm_hrnet_w32",
                "pt_hrnet_timm_hrnet_w48",
                "pt_hrnet_timm_hrnet_w40",
                "pt_hrnet_timm_hrnet_w44",
                "pt_hrnet_timm_hrnet_w18_small",
                "pt_hrnet_timm_hrnet_w64",
                "pt_hrnet_timm_hrnet_w18_small_v2",
                "pt_resnet50_timm",
                "pt_resnet50",
                "pt_resnext50_torchhub",
                "pt_resnext101_torchhub",
                "pt_resnext101_fb_wsl",
                "pt_wide_resnet101_2_timm",
                "pt_wide_resnet50_2_hub",
                "pt_wide_resnet50_2_timm",
                "pt_wide_resnet101_2_hub",
                "pt_xception65_timm",
                "pt_xception71_timm",
                "pt_xception41_timm",
                "pt_xception_timm",
            ]
        },
    ),
    (
        Reshape2,
        [((1, 2048, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_hrnet_osmr_hrnet_w18_small_v2",
                "pt_hrnet_osmr_hrnetv2_w64",
                "pt_hrnet_osmr_hrnetv2_w40",
                "pt_hrnet_osmr_hrnetv2_w18",
                "pt_hrnet_osmr_hrnetv2_w32",
                "pt_hrnet_osmr_hrnetv2_w30",
                "pt_hrnet_osmr_hrnetv2_w44",
                "pt_hrnet_osmr_hrnetv2_w48",
                "pt_hrnet_osmr_hrnet_w18_small_v1",
                "pt_resnext14_osmr",
                "pt_resnext26_osmr",
                "pt_resnext101_osmr",
                "pt_resnext50_osmr",
            ]
        },
    ),
    (Reshape626, [((1, 1536, 1, 1), torch.float32)], {"model_name": ["pt_timm_inception_v4"]}),
    (Reshape47, [((1, 1536, 1, 1), torch.float32)], {"model_name": ["pt_osmr_inception_v4"]}),
    (Reshape627, [((1, 1024, 7, 7), torch.float32)], {"model_name": ["pt_mixer_l32_224"]}),
    (
        Reshape628,
        [((1, 1024, 14, 14), torch.float32)],
        {"model_name": ["pt_mixer_l16_224", "pt_mixer_l16_224_in21k", "pt_vit_large_patch16_224"]},
    ),
    (Reshape629, [((1, 512, 7, 7), torch.float32)], {"model_name": ["pt_mixer_s32_224"]}),
    (
        Reshape630,
        [((1, 512, 7, 7), torch.float32)],
        {
            "model_name": [
                "pt_vgg13_osmr",
                "pt_bn_vgg19_osmr",
                "pt_bn_vgg19b_osmr",
                "pt_vgg16_osmr",
                "pt_vgg19_osmr",
                "pt_vgg11_osmr",
            ]
        },
    ),
    (Reshape631, [((1, 512, 7, 7), torch.float32)], {"model_name": ["pt_vgg_19_hf", "pt_vgg_bn19_torchhub"]}),
    (Reshape632, [((1, 512, 14, 14), torch.float32)], {"model_name": ["pt_mixer_s16_224"]}),
    (Reshape633, [((1, 768, 7, 7), torch.float32)], {"model_name": ["pt_mixer_b32_224"]}),
    (
        Reshape634,
        [((64, 1, 3, 3), torch.float32)],
        {
            "model_name": [
                "pt_mobilenet_v1_224",
                "pt_mobilenet_v1_basic",
                "pt_mobilenet_v3_large",
                "pt_mobilenetv3_large_100",
                "pt_ese_vovnet19b_dw",
                "pt_xception65_timm",
                "pt_xception71_timm",
                "pt_xception41_timm",
                "pt_xception_timm",
                "pt_yolox_nano",
            ]
        },
    ),
    (
        Reshape635,
        [((128, 1, 3, 3), torch.float32)],
        {
            "model_name": [
                "pt_mobilenet_v1_224",
                "pt_mobilenet_v1_basic",
                "pt_segformer_b0_finetuned_ade_512_512",
                "pt_mit_b0",
                "pt_ese_vovnet19b_dw",
                "pt_xception65_timm",
                "pt_xception71_timm",
                "pt_xception41_timm",
                "pt_xception_timm",
                "pt_yolox_nano",
            ]
        },
    ),
    (
        Reshape636,
        [((256, 1, 3, 3), torch.float32)],
        {
            "model_name": [
                "pt_mobilenet_v1_224",
                "pt_mobilenet_v1_basic",
                "pt_mit_b4",
                "pt_segformer_b0_finetuned_ade_512_512",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_mit_b0",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
                "pt_xception65_timm",
                "pt_xception71_timm",
                "pt_xception41_timm",
                "pt_xception_timm",
            ]
        },
    ),
    (
        Reshape637,
        [((512, 1, 3, 3), torch.float32)],
        {
            "model_name": [
                "pt_mobilenet_v1_224",
                "pt_mobilenet_v1_basic",
                "pt_mit_b4",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Reshape638,
        [((1024, 1, 3, 3), torch.float32)],
        {
            "model_name": [
                "pt_mobilenet_v1_224",
                "pt_mobilenet_v1_basic",
                "pt_segformer_b0_finetuned_ade_512_512",
                "pt_mit_b0",
                "pt_xception65_timm",
                "pt_xception71_timm",
                "pt_xception41_timm",
                "pt_xception_timm",
            ]
        },
    ),
    (
        Reshape639,
        [((384, 1, 3, 3), torch.float32)],
        {
            "model_name": [
                "pt_mobilenet_v1_192",
                "mobilenetv2_basic",
                "mobilenetv2_deeplabv3",
                "mobilenetv2_timm",
                "mobilenetv2_224",
            ]
        },
    ),
    (Reshape640, [((768, 1, 3, 3), torch.float32)], {"model_name": ["pt_mobilenet_v1_192"]}),
    (Reshape641, [((1, 768, 1, 1), torch.float32)], {"model_name": ["pt_mobilenet_v1_192"]}),
    (
        Reshape642,
        [((576, 1, 3, 3), torch.float32)],
        {"model_name": ["mobilenetv2_basic", "mobilenetv2_deeplabv3", "mobilenetv2_timm", "mobilenetv2_224"]},
    ),
    (
        Reshape643,
        [((960, 1, 3, 3), torch.float32)],
        {"model_name": ["mobilenetv2_basic", "mobilenetv2_deeplabv3", "mobilenetv2_timm", "mobilenetv2_224"]},
    ),
    (Reshape644, [((288, 1, 3, 3), torch.float32)], {"model_name": ["mobilenetv2_160"]}),
    (Reshape645, [((432, 1, 3, 3), torch.float32)], {"model_name": ["mobilenetv2_160"]}),
    (Reshape646, [((720, 1, 3, 3), torch.float32)], {"model_name": ["mobilenetv2_160"]}),
    (
        Reshape647,
        [((72, 1, 3, 3), torch.float32)],
        {
            "model_name": [
                "pt_mobilenet_v3_small",
                "pt_mobilenet_v3_large",
                "pt_mobilenetv3_small_100",
                "pt_mobilenetv3_large_100",
            ]
        },
    ),
    (
        Reshape648,
        [((88, 1, 3, 3), torch.float32)],
        {"model_name": ["pt_mobilenet_v3_small", "pt_mobilenetv3_small_100"]},
    ),
    (
        Reshape649,
        [((96, 1, 5, 5), torch.float32)],
        {"model_name": ["pt_mobilenet_v3_small", "pt_mobilenetv3_small_100"]},
    ),
    (
        Reshape650,
        [((120, 1, 5, 5), torch.float32)],
        {
            "model_name": [
                "pt_mobilenet_v3_small",
                "pt_mobilenet_v3_large",
                "pt_mobilenetv3_small_100",
                "pt_mobilenetv3_large_100",
            ]
        },
    ),
    (
        Reshape651,
        [((288, 1, 5, 5), torch.float32)],
        {"model_name": ["pt_mobilenet_v3_small", "pt_mobilenetv3_small_100"]},
    ),
    (
        Reshape652,
        [((576, 1, 5, 5), torch.float32)],
        {"model_name": ["pt_mobilenet_v3_small", "pt_mobilenetv3_small_100"]},
    ),
    (Reshape653, [((1, 576, 1, 1), torch.float32)], {"model_name": ["pt_mobilenet_v3_small"]}),
    (
        Reshape654,
        [((200, 1, 3, 3), torch.float32)],
        {"model_name": ["pt_mobilenet_v3_large", "pt_mobilenetv3_large_100"]},
    ),
    (
        Reshape655,
        [((184, 1, 3, 3), torch.float32)],
        {"model_name": ["pt_mobilenet_v3_large", "pt_mobilenetv3_large_100"]},
    ),
    (Reshape656, [((1, 960, 1, 1), torch.float32)], {"model_name": ["pt_mobilenet_v3_large"]}),
    (Reshape657, [((1, 512, 512), torch.float32)], {"model_name": ["pt_vision_perceiver_learned"]}),
    (Reshape658, [((1, 224, 224, 256), torch.float32)], {"model_name": ["pt_vision_perceiver_learned"]}),
    (Reshape659, [((1, 50176, 512), torch.float32)], {"model_name": ["pt_vision_perceiver_learned"]}),
    (Reshape660, [((1, 50176, 512), torch.float32)], {"model_name": ["pt_vision_perceiver_learned"]}),
    (Reshape661, [((50176, 512), torch.float32)], {"model_name": ["pt_vision_perceiver_learned"]}),
    (
        Reshape662,
        [((1, 512, 50176), torch.float32)],
        {"model_name": ["pt_vision_perceiver_learned", "pt_vision_perceiver_fourier"]},
    ),
    (
        Reshape663,
        [((1, 1, 512, 50176), torch.float32)],
        {"model_name": ["pt_vision_perceiver_learned", "pt_vision_perceiver_fourier"]},
    ),
    (
        Reshape664,
        [((1, 512, 1024), torch.float32)],
        {"model_name": ["pt_vision_perceiver_learned", "pt_vision_perceiver_conv", "pt_vision_perceiver_fourier"]},
    ),
    (
        Reshape665,
        [((1, 512, 1024), torch.float32)],
        {"model_name": ["pt_vision_perceiver_learned", "pt_vision_perceiver_conv", "pt_vision_perceiver_fourier"]},
    ),
    (
        Reshape666,
        [((1, 512, 1024), torch.float32)],
        {"model_name": ["pt_vision_perceiver_learned", "pt_vision_perceiver_conv", "pt_vision_perceiver_fourier"]},
    ),
    (
        Reshape667,
        [((512, 1024), torch.float32)],
        {"model_name": ["pt_vision_perceiver_learned", "pt_vision_perceiver_conv", "pt_vision_perceiver_fourier"]},
    ),
    (
        Reshape668,
        [((1, 8, 512, 128), torch.float32)],
        {"model_name": ["pt_vision_perceiver_learned", "pt_vision_perceiver_conv", "pt_vision_perceiver_fourier"]},
    ),
    (
        Reshape669,
        [((8, 512, 512), torch.float32)],
        {"model_name": ["pt_vision_perceiver_learned", "pt_vision_perceiver_conv", "pt_vision_perceiver_fourier"]},
    ),
    (
        Reshape670,
        [((1, 8, 512, 512), torch.float32)],
        {"model_name": ["pt_vision_perceiver_learned", "pt_vision_perceiver_conv", "pt_vision_perceiver_fourier"]},
    ),
    (
        Reshape671,
        [((1, 8, 128, 512), torch.float32)],
        {"model_name": ["pt_vision_perceiver_learned", "pt_vision_perceiver_conv", "pt_vision_perceiver_fourier"]},
    ),
    (
        Reshape672,
        [((8, 512, 128), torch.float32)],
        {"model_name": ["pt_vision_perceiver_learned", "pt_vision_perceiver_conv", "pt_vision_perceiver_fourier"]},
    ),
    (
        Reshape664,
        [((1, 512, 8, 128), torch.float32)],
        {"model_name": ["pt_vision_perceiver_learned", "pt_vision_perceiver_conv", "pt_vision_perceiver_fourier"]},
    ),
    (
        Reshape542,
        [((1, 1, 1, 512), torch.float32)],
        {"model_name": ["pt_vision_perceiver_learned", "pt_vision_perceiver_conv", "pt_vision_perceiver_fourier"]},
    ),
    (
        Reshape673,
        [((1, 1, 1024, 512), torch.float32)],
        {"model_name": ["pt_vision_perceiver_learned", "pt_vision_perceiver_conv", "pt_vision_perceiver_fourier"]},
    ),
    (
        Reshape584,
        [((1, 1, 1000), torch.float32)],
        {"model_name": ["pt_vision_perceiver_learned", "pt_vision_perceiver_conv", "pt_vision_perceiver_fourier"]},
    ),
    (Reshape674, [((1, 512, 322), torch.float32)], {"model_name": ["pt_vision_perceiver_conv"]}),
    (Reshape675, [((1, 55, 55, 64), torch.float32)], {"model_name": ["pt_vision_perceiver_conv"]}),
    (Reshape676, [((1, 3025, 322), torch.float32)], {"model_name": ["pt_vision_perceiver_conv"]}),
    (Reshape677, [((1, 3025, 322), torch.float32)], {"model_name": ["pt_vision_perceiver_conv"]}),
    (Reshape678, [((3025, 322), torch.float32)], {"model_name": ["pt_vision_perceiver_conv"]}),
    (Reshape679, [((1, 512, 3025), torch.float32)], {"model_name": ["pt_vision_perceiver_conv"]}),
    (Reshape680, [((1, 1, 512, 3025), torch.float32)], {"model_name": ["pt_vision_perceiver_conv"]}),
    (Reshape681, [((1, 1, 322, 3025), torch.float32)], {"model_name": ["pt_vision_perceiver_conv"]}),
    (Reshape682, [((1, 512, 261), torch.float32)], {"model_name": ["pt_vision_perceiver_fourier"]}),
    (Reshape683, [((1, 224, 224, 3), torch.float32)], {"model_name": ["pt_vision_perceiver_fourier"]}),
    (Reshape684, [((1, 50176, 261), torch.float32)], {"model_name": ["pt_vision_perceiver_fourier"]}),
    (Reshape685, [((1, 50176, 261), torch.float32)], {"model_name": ["pt_vision_perceiver_fourier"]}),
    (Reshape686, [((50176, 261), torch.float32)], {"model_name": ["pt_vision_perceiver_fourier"]}),
    (Reshape687, [((1, 1, 261, 50176), torch.float32)], {"model_name": ["pt_vision_perceiver_fourier"]}),
    (Reshape688, [((1, 1088, 1, 1), torch.float32)], {"model_name": ["pt_regnet_y_040"]}),
    (
        Reshape689,
        [((1, 16384, 64), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Reshape690,
        [((1, 16384, 64), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Reshape190,
        [((1, 64, 16384), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Reshape691,
        [((1, 64, 16, 16), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_segformer_b0_finetuned_ade_512_512",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_mit_b0",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Reshape692,
        [((1, 256, 64), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_segformer_b0_finetuned_ade_512_512",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_mit_b0",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Reshape693,
        [((1, 256, 64), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Reshape694,
        [((1, 256, 64), torch.float32)],
        {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_mit_b0"]},
    ),
    (
        Reshape695,
        [((256, 64), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_segformer_b0_finetuned_ade_512_512",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_mit_b0",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Reshape696,
        [((1, 16384, 256), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_segformer_b0_finetuned_ade_512_512",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_mit_b0",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Reshape697,
        [((1, 1, 16384, 256), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_segformer_b0_finetuned_ade_512_512",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_mit_b0",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Reshape691,
        [((1, 1, 64, 256), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Reshape698,
        [((1, 256, 16384), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_segformer_b0_finetuned_ade_512_512",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Reshape699,
        [((1, 256, 128, 128), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Reshape700,
        [((1, 4096, 128), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Reshape701,
        [((1, 4096, 128), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Reshape702,
        [((1, 2, 4096, 64), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Reshape703,
        [((1, 128, 16, 16), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Reshape704,
        [((1, 256, 128), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Reshape705,
        [((1, 256, 128), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Reshape706,
        [((256, 128), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Reshape707,
        [((1, 2, 256, 64), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Reshape708,
        [((2, 4096, 256), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_segformer_b0_finetuned_ade_512_512",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_mit_b0",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Reshape709,
        [((1, 2, 4096, 256), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_segformer_b0_finetuned_ade_512_512",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_mit_b0",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Reshape710,
        [((1, 2, 64, 256), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Reshape711,
        [((2, 4096, 64), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Reshape712,
        [((1, 4096, 2, 64), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Reshape713,
        [((4096, 128), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Reshape714,
        [((1, 512, 4096), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Reshape715,
        [((1, 512, 64, 64), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Reshape716,
        [((1, 320, 32, 32), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Reshape717,
        [((1, 1024, 320), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Reshape718,
        [((1, 1024, 320), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Reshape719,
        [((1, 5, 1024, 64), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Reshape720,
        [((1, 320, 1024), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Reshape721,
        [((1, 320, 16, 16), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Reshape722,
        [((1, 256, 320), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Reshape723,
        [((1, 256, 320), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Reshape724,
        [((256, 320), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Reshape725,
        [((1, 5, 256, 64), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Reshape726,
        [((5, 1024, 256), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_segformer_b0_finetuned_ade_512_512",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_mit_b0",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Reshape727,
        [((1, 5, 1024, 256), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_segformer_b0_finetuned_ade_512_512",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_mit_b0",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Reshape728,
        [((1, 5, 64, 256), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Reshape729,
        [((5, 1024, 64), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Reshape730,
        [((1, 1024, 5, 64), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Reshape731,
        [((1024, 320), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Reshape732,
        [((1, 1280, 1024), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Reshape733,
        [((1280, 1, 3, 3), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Reshape734,
        [((1, 1280, 32, 32), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Reshape735,
        [((1, 512, 16, 16), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Reshape301,
        [((1, 256, 512), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Reshape300,
        [((1, 256, 512), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Reshape302,
        [((1, 256, 512), torch.float32)],
        {"model_name": ["pt_mit_b4", "pt_mit_b1", "pt_mit_b2", "pt_mit_b3", "pt_mit_b5"]},
    ),
    (
        Reshape736,
        [((1, 256, 512), torch.float32)],
        {
            "model_name": [
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
            ]
        },
    ),
    (
        Reshape737,
        [((1, 8, 256, 64), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Reshape738,
        [((8, 256, 256), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_segformer_b0_finetuned_ade_512_512",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_mit_b0",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Reshape739,
        [((1, 8, 256, 256), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_segformer_b0_finetuned_ade_512_512",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_mit_b0",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Reshape740,
        [((1, 8, 64, 256), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Reshape741,
        [((8, 256, 64), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Reshape301,
        [((1, 256, 8, 64), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Reshape742,
        [((1, 2048, 256), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Reshape743,
        [((2048, 1, 3, 3), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Reshape744,
        [((1, 2048, 16, 16), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Reshape745,
        [((1, 16384, 32), torch.float32)],
        {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_mit_b0"]},
    ),
    (
        Reshape746,
        [((1, 16384, 32), torch.float32)],
        {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_mit_b0"]},
    ),
    (
        Reshape326,
        [((1, 32, 16384), torch.float32)],
        {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_mit_b0"]},
    ),
    (
        Reshape747,
        [((1, 32, 16, 16), torch.float32)],
        {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_mit_b0"]},
    ),
    (
        Reshape748,
        [((1, 256, 32), torch.float32)],
        {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_mit_b0"]},
    ),
    (
        Reshape749,
        [((1, 256, 32), torch.float32)],
        {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_mit_b0"]},
    ),
    (Reshape750, [((256, 32), torch.float32)], {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_mit_b0"]}),
    (
        Reshape747,
        [((1, 1, 32, 256), torch.float32)],
        {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_mit_b0"]},
    ),
    (
        Reshape751,
        [((1, 128, 16384), torch.float32)],
        {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_mit_b0"]},
    ),
    (
        Reshape752,
        [((1, 128, 128, 128), torch.float32)],
        {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_mit_b0"]},
    ),
    (
        Reshape753,
        [((1, 64, 64, 64), torch.float32)],
        {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_mit_b0"]},
    ),
    (
        Reshape754,
        [((1, 4096, 64), torch.float32)],
        {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_mit_b0"]},
    ),
    (
        Reshape755,
        [((1, 4096, 64), torch.float32)],
        {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_mit_b0"]},
    ),
    (
        Reshape756,
        [((1, 2, 4096, 32), torch.float32)],
        {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_mit_b0"]},
    ),
    (
        Reshape755,
        [((1, 64, 4096), torch.float32)],
        {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_mit_b0"]},
    ),
    (
        Reshape757,
        [((1, 2, 256, 32), torch.float32)],
        {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_mit_b0"]},
    ),
    (
        Reshape758,
        [((1, 2, 32, 256), torch.float32)],
        {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_mit_b0"]},
    ),
    (
        Reshape759,
        [((2, 4096, 32), torch.float32)],
        {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_mit_b0"]},
    ),
    (
        Reshape760,
        [((1, 4096, 2, 32), torch.float32)],
        {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_mit_b0"]},
    ),
    (Reshape761, [((4096, 64), torch.float32)], {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_mit_b0"]}),
    (
        Reshape762,
        [((1, 256, 64, 64), torch.float32)],
        {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_mit_b0"]},
    ),
    (
        Reshape763,
        [((1, 160, 32, 32), torch.float32)],
        {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_mit_b0"]},
    ),
    (
        Reshape764,
        [((1, 1024, 160), torch.float32)],
        {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_mit_b0"]},
    ),
    (
        Reshape765,
        [((1, 1024, 160), torch.float32)],
        {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_mit_b0"]},
    ),
    (
        Reshape766,
        [((1, 5, 1024, 32), torch.float32)],
        {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_mit_b0"]},
    ),
    (
        Reshape767,
        [((1, 160, 1024), torch.float32)],
        {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_mit_b0"]},
    ),
    (
        Reshape768,
        [((1, 160, 16, 16), torch.float32)],
        {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_mit_b0"]},
    ),
    (
        Reshape769,
        [((1, 256, 160), torch.float32)],
        {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_mit_b0"]},
    ),
    (
        Reshape770,
        [((1, 256, 160), torch.float32)],
        {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_mit_b0"]},
    ),
    (Reshape771, [((256, 160), torch.float32)], {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_mit_b0"]}),
    (
        Reshape772,
        [((1, 5, 256, 32), torch.float32)],
        {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_mit_b0"]},
    ),
    (
        Reshape773,
        [((1, 5, 32, 256), torch.float32)],
        {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_mit_b0"]},
    ),
    (
        Reshape774,
        [((5, 1024, 32), torch.float32)],
        {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_mit_b0"]},
    ),
    (
        Reshape775,
        [((1, 1024, 5, 32), torch.float32)],
        {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_mit_b0"]},
    ),
    (
        Reshape776,
        [((1024, 160), torch.float32)],
        {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_mit_b0"]},
    ),
    (
        Reshape777,
        [((1, 640, 1024), torch.float32)],
        {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_mit_b0"]},
    ),
    (
        Reshape778,
        [((640, 1, 3, 3), torch.float32)],
        {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_mit_b0"]},
    ),
    (
        Reshape779,
        [((1, 640, 32, 32), torch.float32)],
        {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_mit_b0"]},
    ),
    (
        Reshape780,
        [((1, 256, 16, 16), torch.float32)],
        {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_mit_b0"]},
    ),
    (
        Reshape781,
        [((1, 256, 256), torch.float32)],
        {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_mit_b0"]},
    ),
    (
        Reshape782,
        [((1, 256, 256), torch.float32)],
        {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_mit_b0"]},
    ),
    (Reshape783, [((1, 256, 256), torch.float32)], {"model_name": ["pt_segformer_b0_finetuned_ade_512_512"]}),
    (
        Reshape784,
        [((1, 256, 256), torch.float32)],
        {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_segformer_b1_finetuned_ade_512_512"]},
    ),
    (Reshape785, [((1, 256, 256), torch.float32)], {"model_name": ["pt_mit_b0"]}),
    (Reshape785, [((256, 256), torch.float32)], {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_mit_b0"]}),
    (
        Reshape786,
        [((1, 8, 256, 32), torch.float32)],
        {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_mit_b0"]},
    ),
    (
        Reshape787,
        [((1, 8, 32, 256), torch.float32)],
        {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_mit_b0"]},
    ),
    (
        Reshape788,
        [((8, 256, 32), torch.float32)],
        {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_mit_b0"]},
    ),
    (
        Reshape781,
        [((1, 256, 8, 32), torch.float32)],
        {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_mit_b0"]},
    ),
    (
        Reshape789,
        [((1, 1024, 256), torch.float32)],
        {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_mit_b0"]},
    ),
    (
        Reshape790,
        [((1, 1024, 16, 16), torch.float32)],
        {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_mit_b0"]},
    ),
    (
        Reshape791,
        [((1, 768, 256), torch.float32)],
        {
            "model_name": [
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
            ]
        },
    ),
    (
        Reshape792,
        [((1, 768, 1024), torch.float32)],
        {
            "model_name": [
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
            ]
        },
    ),
    (
        Reshape793,
        [((1, 768, 4096), torch.float32)],
        {
            "model_name": [
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
            ]
        },
    ),
    (
        Reshape794,
        [((1, 768, 16384), torch.float32)],
        {
            "model_name": [
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
            ]
        },
    ),
    (Reshape795, [((1, 16, 38, 38), torch.float32)], {"model_name": ["pt_ssd300_resnet50"]}),
    (Reshape796, [((1, 24, 19, 19), torch.float32)], {"model_name": ["pt_ssd300_resnet50"]}),
    (Reshape797, [((1, 24, 10, 10), torch.float32)], {"model_name": ["pt_ssd300_resnet50"]}),
    (Reshape798, [((1, 24, 5, 5), torch.float32)], {"model_name": ["pt_ssd300_resnet50"]}),
    (Reshape799, [((1, 16, 3, 3), torch.float32)], {"model_name": ["pt_ssd300_resnet50"]}),
    (Reshape800, [((1, 324, 38, 38), torch.float32)], {"model_name": ["pt_ssd300_resnet50"]}),
    (Reshape801, [((1, 486, 19, 19), torch.float32)], {"model_name": ["pt_ssd300_resnet50"]}),
    (Reshape802, [((1, 486, 10, 10), torch.float32)], {"model_name": ["pt_ssd300_resnet50"]}),
    (Reshape803, [((1, 486, 5, 5), torch.float32)], {"model_name": ["pt_ssd300_resnet50"]}),
    (Reshape804, [((1, 324, 3, 3), torch.float32)], {"model_name": ["pt_ssd300_resnet50"]}),
    (Reshape805, [((1, 324, 1, 1), torch.float32)], {"model_name": ["pt_ssd300_resnet50"]}),
    (Reshape806, [((1, 96, 64, 64), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Reshape807, [((1, 4096, 96), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Reshape808, [((1, 4096, 96), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Reshape809, [((1, 8, 8, 8, 8, 96), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Reshape810, [((1, 8, 8, 8, 8, 96), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Reshape808, [((1, 8, 8, 8, 8, 96), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Reshape811, [((4096, 96), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Reshape812, [((4096, 96), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Reshape812, [((64, 64, 96), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Reshape807, [((64, 64, 96), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Reshape813, [((64, 3, 64, 32), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Reshape814, [((192, 64, 64), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Reshape815, [((1, 15, 15, 2), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Reshape816, [((225, 512), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Reshape817, [((1, 15, 15, 512), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Reshape818, [((225, 3), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Reshape819, [((64, 64), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Reshape820, [((4096, 3), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Reshape821, [((64, 3, 64, 64), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Reshape822, [((64, 3, 64, 64), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Reshape823, [((64, 3, 32, 64), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Reshape824, [((192, 64, 32), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Reshape809, [((64, 64, 3, 32), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Reshape807, [((1, 64, 64, 96), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Reshape810, [((1, 64, 64, 96), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Reshape814, [((1, 64, 3, 64, 64), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Reshape825, [((1, 32, 32, 384), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Reshape826, [((1024, 192), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Reshape827, [((1024, 192), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Reshape828, [((1024, 192), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Reshape829, [((1, 1024, 192), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Reshape830, [((1, 1024, 192), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Reshape831, [((1, 4, 4, 8, 8, 192), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Reshape828, [((16, 64, 192), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Reshape832, [((16, 64, 192), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Reshape833, [((16, 6, 64, 32), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Reshape834, [((96, 64, 64), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Reshape835, [((225, 6), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Reshape836, [((4096, 6), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Reshape837, [((16, 6, 64, 64), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Reshape838, [((16, 6, 64, 64), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Reshape839, [((16, 6, 32, 64), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Reshape840, [((96, 64, 32), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Reshape831, [((16, 64, 6, 32), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Reshape826, [((1, 4, 8, 4, 8, 192), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Reshape830, [((1, 4, 8, 4, 8, 192), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Reshape829, [((1, 32, 32, 192), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Reshape826, [((1, 32, 32, 192), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Reshape834, [((1, 16, 6, 64, 64), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Reshape265, [((1, 16, 16, 768), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Reshape841, [((256, 384), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Reshape842, [((256, 384), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Reshape843, [((256, 384), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Reshape844, [((1, 256, 384), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Reshape845, [((1, 256, 384), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Reshape846, [((1, 2, 2, 8, 8, 384), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Reshape843, [((4, 64, 384), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Reshape847, [((4, 64, 384), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Reshape848, [((4, 12, 64, 32), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Reshape849, [((48, 64, 64), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Reshape850, [((225, 12), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Reshape851, [((4096, 12), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Reshape852, [((4, 12, 64, 64), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Reshape853, [((4, 12, 64, 64), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Reshape854, [((4, 12, 32, 64), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Reshape855, [((48, 64, 32), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Reshape846, [((4, 64, 12, 32), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Reshape841, [((1, 2, 8, 2, 8, 384), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Reshape845, [((1, 2, 8, 2, 8, 384), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Reshape844, [((1, 16, 16, 384), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Reshape841, [((1, 16, 16, 384), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Reshape849, [((1, 4, 12, 64, 64), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Reshape856, [((1, 8, 8, 1536), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Reshape857, [((64, 768), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Reshape858, [((64, 768), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Reshape859, [((1, 64, 768), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Reshape858, [((1, 64, 768), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Reshape857, [((1, 64, 768), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Reshape860, [((1, 24, 64, 32), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Reshape861, [((24, 64, 64), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Reshape862, [((225, 24), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Reshape863, [((4096, 24), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Reshape864, [((1, 24, 64, 64), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Reshape865, [((1, 24, 32, 64), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Reshape866, [((24, 64, 32), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Reshape859, [((1, 64, 24, 32), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Reshape867, [((1, 768, 64), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Reshape868, [((1, 4096, 1, 1), torch.float32)], {"model_name": ["pt_vgg19_bn_timm"]}),
    (Reshape869, [((1, 197, 1024), torch.float32)], {"model_name": ["pt_vit_large_patch16_224"]}),
    (Reshape870, [((1, 197, 1024), torch.float32)], {"model_name": ["pt_vit_large_patch16_224"]}),
    (Reshape871, [((197, 1024), torch.float32)], {"model_name": ["pt_vit_large_patch16_224"]}),
    (Reshape872, [((1, 16, 197, 64), torch.float32)], {"model_name": ["pt_vit_large_patch16_224"]}),
    (Reshape873, [((16, 197, 197), torch.float32)], {"model_name": ["pt_vit_large_patch16_224"]}),
    (Reshape874, [((1, 16, 197, 197), torch.float32)], {"model_name": ["pt_vit_large_patch16_224"]}),
    (Reshape875, [((1, 16, 64, 197), torch.float32)], {"model_name": ["pt_vit_large_patch16_224"]}),
    (Reshape876, [((16, 197, 64), torch.float32)], {"model_name": ["pt_vit_large_patch16_224"]}),
    (Reshape869, [((1, 197, 16, 64), torch.float32)], {"model_name": ["pt_vit_large_patch16_224"]}),
    (Reshape877, [((160, 1, 3, 3), torch.float32)], {"model_name": ["pt_ese_vovnet19b_dw"]}),
    (Reshape878, [((224, 1, 3, 3), torch.float32)], {"model_name": ["pt_ese_vovnet19b_dw"]}),
    (Reshape539, [((1, 512, 1, 1), torch.float32)], {"model_name": ["pt_vovnet27s"]}),
    (
        Reshape879,
        [((728, 1, 3, 3), torch.float32)],
        {"model_name": ["pt_xception65_timm", "pt_xception71_timm", "pt_xception41_timm", "pt_xception_timm"]},
    ),
    (
        Reshape880,
        [((1536, 1, 3, 3), torch.float32)],
        {"model_name": ["pt_xception65_timm", "pt_xception71_timm", "pt_xception41_timm", "pt_xception_timm"]},
    ),
    (
        Reshape881,
        [((1, 3, 85, 80, 80), torch.float32)],
        {
            "model_name": [
                "pt_yolov5x_640x640",
                "pt_yolov5s_1280x1280",
                "pt_yolov5s_640x640",
                "pt_yolov5m_640x640",
                "pt_yolov5l_640x640",
                "pt_yolov5n_640x640",
            ]
        },
    ),
    (
        Reshape882,
        [((1, 255, 80, 80), torch.float32)],
        {
            "model_name": [
                "pt_yolov5x_640x640",
                "pt_yolov5s_1280x1280",
                "pt_yolov5s_640x640",
                "pt_yolov5m_640x640",
                "pt_yolov5l_640x640",
                "pt_yolov5n_640x640",
            ]
        },
    ),
    (
        Reshape883,
        [((1, 1, 255, 6400), torch.float32)],
        {
            "model_name": [
                "pt_yolov5x_640x640",
                "pt_yolov5s_1280x1280",
                "pt_yolov5s_640x640",
                "pt_yolov5m_640x640",
                "pt_yolov5l_640x640",
                "pt_yolov5n_640x640",
            ]
        },
    ),
    (
        Reshape884,
        [((1, 3, 6400, 85), torch.float32)],
        {
            "model_name": [
                "pt_yolov5x_640x640",
                "pt_yolov5s_1280x1280",
                "pt_yolov5s_640x640",
                "pt_yolov5m_640x640",
                "pt_yolov5l_640x640",
                "pt_yolov5n_640x640",
            ]
        },
    ),
    (
        Reshape885,
        [((1, 3, 85, 40, 40), torch.float32)],
        {
            "model_name": [
                "pt_yolov5x_640x640",
                "pt_yolov5l_320x320",
                "pt_yolov5s_1280x1280",
                "pt_yolov5m_320x320",
                "pt_yolov5s_640x640",
                "pt_yolov5x_320x320",
                "pt_yolov5s_320x320",
                "pt_yolov5m_640x640",
                "pt_yolov5l_640x640",
                "pt_yolov5n_640x640",
                "pt_yolov5n_320x320",
            ]
        },
    ),
    (
        Reshape886,
        [((1, 255, 40, 40), torch.float32)],
        {
            "model_name": [
                "pt_yolov5x_640x640",
                "pt_yolov5l_320x320",
                "pt_yolov5s_1280x1280",
                "pt_yolov5m_320x320",
                "pt_yolov5s_640x640",
                "pt_yolov5x_320x320",
                "pt_yolov5s_320x320",
                "pt_yolov5m_640x640",
                "pt_yolov5l_640x640",
                "pt_yolov5n_640x640",
                "pt_yolov5n_320x320",
            ]
        },
    ),
    (
        Reshape887,
        [((1, 1, 255, 1600), torch.float32)],
        {
            "model_name": [
                "pt_yolov5x_640x640",
                "pt_yolov5l_320x320",
                "pt_yolov5s_1280x1280",
                "pt_yolov5m_320x320",
                "pt_yolov5s_640x640",
                "pt_yolov5x_320x320",
                "pt_yolov5s_320x320",
                "pt_yolov5m_640x640",
                "pt_yolov5l_640x640",
                "pt_yolov5n_640x640",
                "pt_yolov5n_320x320",
            ]
        },
    ),
    (
        Reshape888,
        [((1, 3, 1600, 85), torch.float32)],
        {
            "model_name": [
                "pt_yolov5x_640x640",
                "pt_yolov5l_320x320",
                "pt_yolov5s_1280x1280",
                "pt_yolov5m_320x320",
                "pt_yolov5s_640x640",
                "pt_yolov5x_320x320",
                "pt_yolov5s_320x320",
                "pt_yolov5m_640x640",
                "pt_yolov5l_640x640",
                "pt_yolov5n_640x640",
                "pt_yolov5n_320x320",
            ]
        },
    ),
    (
        Reshape889,
        [((1, 3, 85, 20, 20), torch.float32)],
        {
            "model_name": [
                "pt_yolov5x_640x640",
                "pt_yolov5l_320x320",
                "pt_yolov5m_320x320",
                "pt_yolov5s_640x640",
                "pt_yolov5x_320x320",
                "pt_yolov5s_320x320",
                "pt_yolov5m_640x640",
                "pt_yolov5l_640x640",
                "pt_yolov5n_640x640",
                "pt_yolov5n_320x320",
            ]
        },
    ),
    (
        Reshape890,
        [((1, 255, 20, 20), torch.float32)],
        {
            "model_name": [
                "pt_yolov5x_640x640",
                "pt_yolov5l_320x320",
                "pt_yolov5m_320x320",
                "pt_yolov5s_640x640",
                "pt_yolov5x_320x320",
                "pt_yolov5s_320x320",
                "pt_yolov5m_640x640",
                "pt_yolov5l_640x640",
                "pt_yolov5n_640x640",
                "pt_yolov5n_320x320",
            ]
        },
    ),
    (
        Reshape891,
        [((1, 1, 255, 400), torch.float32)],
        {
            "model_name": [
                "pt_yolov5x_640x640",
                "pt_yolov5l_320x320",
                "pt_yolov5m_320x320",
                "pt_yolov5s_640x640",
                "pt_yolov5x_320x320",
                "pt_yolov5s_320x320",
                "pt_yolov5m_640x640",
                "pt_yolov5l_640x640",
                "pt_yolov5n_640x640",
                "pt_yolov5n_320x320",
            ]
        },
    ),
    (
        Reshape892,
        [((1, 3, 400, 85), torch.float32)],
        {
            "model_name": [
                "pt_yolov5x_640x640",
                "pt_yolov5l_320x320",
                "pt_yolov5m_320x320",
                "pt_yolov5s_640x640",
                "pt_yolov5x_320x320",
                "pt_yolov5s_320x320",
                "pt_yolov5m_640x640",
                "pt_yolov5l_640x640",
                "pt_yolov5n_640x640",
                "pt_yolov5n_320x320",
            ]
        },
    ),
    (
        Reshape893,
        [((1, 3, 85, 10, 10), torch.float32)],
        {
            "model_name": [
                "pt_yolov5l_320x320",
                "pt_yolov5m_320x320",
                "pt_yolov5x_320x320",
                "pt_yolov5s_320x320",
                "pt_yolov5n_320x320",
            ]
        },
    ),
    (
        Reshape894,
        [((1, 255, 10, 10), torch.float32)],
        {
            "model_name": [
                "pt_yolov5l_320x320",
                "pt_yolov5m_320x320",
                "pt_yolov5x_320x320",
                "pt_yolov5s_320x320",
                "pt_yolov5n_320x320",
            ]
        },
    ),
    (
        Reshape895,
        [((1, 1, 255, 100), torch.float32)],
        {
            "model_name": [
                "pt_yolov5l_320x320",
                "pt_yolov5m_320x320",
                "pt_yolov5x_320x320",
                "pt_yolov5s_320x320",
                "pt_yolov5n_320x320",
            ]
        },
    ),
    (
        Reshape896,
        [((1, 3, 100, 85), torch.float32)],
        {
            "model_name": [
                "pt_yolov5l_320x320",
                "pt_yolov5m_320x320",
                "pt_yolov5x_320x320",
                "pt_yolov5s_320x320",
                "pt_yolov5n_320x320",
            ]
        },
    ),
    (Reshape897, [((1, 3, 85, 160, 160), torch.float32)], {"model_name": ["pt_yolov5s_1280x1280"]}),
    (Reshape898, [((1, 255, 160, 160), torch.float32)], {"model_name": ["pt_yolov5s_1280x1280"]}),
    (Reshape899, [((1, 1, 255, 25600), torch.float32)], {"model_name": ["pt_yolov5s_1280x1280"]}),
    (Reshape900, [((1, 3, 25600, 85), torch.float32)], {"model_name": ["pt_yolov5s_1280x1280"]}),
    (
        Reshape901,
        [((1, 3, 85, 60, 60), torch.float32)],
        {
            "model_name": [
                "pt_yolov5m_480x480",
                "pt_yolov5n_480x480",
                "pt_yolov5l_480x480",
                "pt_yolov5x_480x480",
                "pt_yolov5s_480x480",
            ]
        },
    ),
    (
        Reshape902,
        [((1, 255, 60, 60), torch.float32)],
        {
            "model_name": [
                "pt_yolov5m_480x480",
                "pt_yolov5n_480x480",
                "pt_yolov5l_480x480",
                "pt_yolov5x_480x480",
                "pt_yolov5s_480x480",
            ]
        },
    ),
    (
        Reshape903,
        [((1, 1, 255, 3600), torch.float32)],
        {
            "model_name": [
                "pt_yolov5m_480x480",
                "pt_yolov5n_480x480",
                "pt_yolov5l_480x480",
                "pt_yolov5x_480x480",
                "pt_yolov5s_480x480",
            ]
        },
    ),
    (
        Reshape904,
        [((1, 3, 3600, 85), torch.float32)],
        {
            "model_name": [
                "pt_yolov5m_480x480",
                "pt_yolov5n_480x480",
                "pt_yolov5l_480x480",
                "pt_yolov5x_480x480",
                "pt_yolov5s_480x480",
            ]
        },
    ),
    (
        Reshape905,
        [((1, 3, 85, 30, 30), torch.float32)],
        {
            "model_name": [
                "pt_yolov5m_480x480",
                "pt_yolov5n_480x480",
                "pt_yolov5l_480x480",
                "pt_yolov5x_480x480",
                "pt_yolov5s_480x480",
            ]
        },
    ),
    (
        Reshape906,
        [((1, 255, 30, 30), torch.float32)],
        {
            "model_name": [
                "pt_yolov5m_480x480",
                "pt_yolov5n_480x480",
                "pt_yolov5l_480x480",
                "pt_yolov5x_480x480",
                "pt_yolov5s_480x480",
            ]
        },
    ),
    (
        Reshape907,
        [((1, 1, 255, 900), torch.float32)],
        {
            "model_name": [
                "pt_yolov5m_480x480",
                "pt_yolov5n_480x480",
                "pt_yolov5l_480x480",
                "pt_yolov5x_480x480",
                "pt_yolov5s_480x480",
            ]
        },
    ),
    (
        Reshape908,
        [((1, 3, 900, 85), torch.float32)],
        {
            "model_name": [
                "pt_yolov5m_480x480",
                "pt_yolov5n_480x480",
                "pt_yolov5l_480x480",
                "pt_yolov5x_480x480",
                "pt_yolov5s_480x480",
            ]
        },
    ),
    (
        Reshape909,
        [((1, 3, 85, 15, 15), torch.float32)],
        {
            "model_name": [
                "pt_yolov5m_480x480",
                "pt_yolov5n_480x480",
                "pt_yolov5l_480x480",
                "pt_yolov5x_480x480",
                "pt_yolov5s_480x480",
            ]
        },
    ),
    (
        Reshape910,
        [((1, 255, 15, 15), torch.float32)],
        {
            "model_name": [
                "pt_yolov5m_480x480",
                "pt_yolov5n_480x480",
                "pt_yolov5l_480x480",
                "pt_yolov5x_480x480",
                "pt_yolov5s_480x480",
            ]
        },
    ),
    (
        Reshape911,
        [((1, 1, 255, 225), torch.float32)],
        {
            "model_name": [
                "pt_yolov5m_480x480",
                "pt_yolov5n_480x480",
                "pt_yolov5l_480x480",
                "pt_yolov5x_480x480",
                "pt_yolov5s_480x480",
            ]
        },
    ),
    (
        Reshape912,
        [((1, 3, 225, 85), torch.float32)],
        {
            "model_name": [
                "pt_yolov5m_480x480",
                "pt_yolov5n_480x480",
                "pt_yolov5l_480x480",
                "pt_yolov5x_480x480",
                "pt_yolov5s_480x480",
            ]
        },
    ),
    (Reshape913, [((1, 68, 56, 80), torch.float32)], {"model_name": ["pt_yolov6m", "pt_yolov6l"]}),
    (Reshape914, [((1, 1, 4, 4480), torch.float32)], {"model_name": ["pt_yolov6m", "pt_yolov6l"]}),
    (Reshape915, [((1, 68, 28, 40), torch.float32)], {"model_name": ["pt_yolov6m", "pt_yolov6l"]}),
    (Reshape916, [((1, 1, 4, 1120), torch.float32)], {"model_name": ["pt_yolov6m", "pt_yolov6l"]}),
    (Reshape917, [((1, 68, 14, 20), torch.float32)], {"model_name": ["pt_yolov6m", "pt_yolov6l"]}),
    (Reshape918, [((1, 1, 4, 280), torch.float32)], {"model_name": ["pt_yolov6m", "pt_yolov6l"]}),
    (
        Reshape919,
        [((1, 80, 56, 80), torch.float32)],
        {"model_name": ["pt_yolov6m", "pt_yolov6n", "pt_yolov6l", "pt_yolov6s"]},
    ),
    (
        Reshape920,
        [((1, 80, 28, 40), torch.float32)],
        {"model_name": ["pt_yolov6m", "pt_yolov6n", "pt_yolov6l", "pt_yolov6s"]},
    ),
    (
        Reshape921,
        [((1, 80, 14, 20), torch.float32)],
        {"model_name": ["pt_yolov6m", "pt_yolov6n", "pt_yolov6l", "pt_yolov6s"]},
    ),
    (Reshape914, [((1, 4, 56, 80), torch.float32)], {"model_name": ["pt_yolov6n", "pt_yolov6s"]}),
    (Reshape916, [((1, 4, 28, 40), torch.float32)], {"model_name": ["pt_yolov6n", "pt_yolov6s"]}),
    (Reshape918, [((1, 4, 14, 20), torch.float32)], {"model_name": ["pt_yolov6n", "pt_yolov6s"]}),
    (
        Reshape922,
        [((1, 85, 80, 80), torch.float32)],
        {"model_name": ["pt_yolox_m", "pt_yolox_s", "pt_yolox_darknet", "pt_yolox_x", "pt_yolox_l"]},
    ),
    (
        Reshape923,
        [((1, 85, 40, 40), torch.float32)],
        {"model_name": ["pt_yolox_m", "pt_yolox_s", "pt_yolox_darknet", "pt_yolox_x", "pt_yolox_l"]},
    ),
    (
        Reshape924,
        [((1, 85, 20, 20), torch.float32)],
        {"model_name": ["pt_yolox_m", "pt_yolox_s", "pt_yolox_darknet", "pt_yolox_x", "pt_yolox_l"]},
    ),
    (Reshape925, [((1, 85, 52, 52), torch.float32)], {"model_name": ["pt_yolox_nano", "pt_yolox_tiny"]}),
    (Reshape926, [((1, 85, 26, 26), torch.float32)], {"model_name": ["pt_yolox_nano", "pt_yolox_tiny"]}),
    (Reshape927, [((1, 85, 13, 13), torch.float32)], {"model_name": ["pt_yolox_nano", "pt_yolox_tiny"]}),
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
