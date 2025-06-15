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
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(128, 768))
        return reshape_output_1


class Reshape9(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 12, 64))
        return reshape_output_1


class Reshape10(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 768))
        return reshape_output_1


class Reshape11(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 128, 64))
        return reshape_output_1


class Reshape12(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 128, 128))
        return reshape_output_1


class Reshape13(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 128, 128))
        return reshape_output_1


class Reshape14(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 128, 64))
        return reshape_output_1


class Reshape15(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 768))
        return reshape_output_1


class Reshape16(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 12, 64))
        return reshape_output_1


class Reshape17(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32))
        return reshape_output_1


class Reshape18(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 768))
        return reshape_output_1


class Reshape19(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 12, 64))
        return reshape_output_1


class Reshape20(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 768))
        return reshape_output_1


class Reshape21(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 32, 64))
        return reshape_output_1


class Reshape22(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 2048))
        return reshape_output_1


class Reshape23(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 32, 32))
        return reshape_output_1


class Reshape24(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 32, 32))
        return reshape_output_1


class Reshape25(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 32, 64))
        return reshape_output_1


class Reshape26(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 12, 64))
        return reshape_output_1


class Reshape27(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 1))
        return reshape_output_1


class Reshape28(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(7, 2048))
        return reshape_output_1


class Reshape29(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 7, 32, 64))
        return reshape_output_1


class Reshape30(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 7, 2048))
        return reshape_output_1


class Reshape31(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 7, 64))
        return reshape_output_1


class Reshape32(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 7, 7))
        return reshape_output_1


class Reshape33(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 7, 7))
        return reshape_output_1


class Reshape34(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 7, 64))
        return reshape_output_1


class Reshape35(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 7, 8192))
        return reshape_output_1


class Reshape36(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(29, 1024))
        return reshape_output_1


class Reshape37(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 29, 16, 64))
        return reshape_output_1


class Reshape38(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 29, 1024))
        return reshape_output_1


class Reshape39(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 29, 64))
        return reshape_output_1


class Reshape40(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 29, 29))
        return reshape_output_1


class Reshape41(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 29, 29))
        return reshape_output_1


class Reshape42(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 29, 64))
        return reshape_output_1


class Reshape43(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 29, 2816))
        return reshape_output_1


class Reshape44(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 672, 1, 1))
        return reshape_output_1


class Reshape45(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2048, 1, 1))
        return reshape_output_1


class Reshape46(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2048))
        return reshape_output_1


class Reshape47(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1))
        return reshape_output_1


class Reshape48(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 768))
        return reshape_output_1


class Reshape49(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 1, 64))
        return reshape_output_1


class Reshape50(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 1, 1))
        return reshape_output_1


class Reshape51(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 1, 1))
        return reshape_output_1


class Reshape52(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 64, 1))
        return reshape_output_1


class Reshape53(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 1, 64))
        return reshape_output_1


class Reshape54(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 61))
        return reshape_output_1


class Reshape55(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(61, 768))
        return reshape_output_1


class Reshape56(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 61, 12, 64))
        return reshape_output_1


class Reshape57(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 61, 768))
        return reshape_output_1


class Reshape58(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 61, 64))
        return reshape_output_1


class Reshape59(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 61, 61))
        return reshape_output_1


class Reshape60(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 61, 61))
        return reshape_output_1


class Reshape61(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 64, 61))
        return reshape_output_1


class Reshape62(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 61, 64))
        return reshape_output_1


class Reshape63(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 1, 61))
        return reshape_output_1


class Reshape64(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 1, 61))
        return reshape_output_1


class Reshape65(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 25088, 1, 1))
        return reshape_output_1


class Reshape66(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 25088))
        return reshape_output_1


class Reshape67(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 512))
        return reshape_output_1


class Reshape68(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 8, 64))
        return reshape_output_1


class Reshape69(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 512))
        return reshape_output_1


class Reshape70(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 1, 64))
        return reshape_output_1


class Reshape71(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 1, 1))
        return reshape_output_1


class Reshape72(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 1, 1))
        return reshape_output_1


class Reshape73(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 1, 64))
        return reshape_output_1


class Reshape74(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 80, 3000, 1))
        return reshape_output_1


class Reshape75(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(512, 80, 3, 1))
        return reshape_output_1


class Reshape76(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 512, 3000))
        return reshape_output_1


class Reshape77(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 512, 3000, 1))
        return reshape_output_1


class Reshape78(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(512, 512, 3, 1))
        return reshape_output_1


class Reshape79(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 512, 1500))
        return reshape_output_1


class Reshape80(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1500, 512))
        return reshape_output_1


class Reshape81(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1500, 8, 64))
        return reshape_output_1


class Reshape82(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1500, 512))
        return reshape_output_1


class Reshape83(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 1500, 64))
        return reshape_output_1


class Reshape84(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 1500, 1500))
        return reshape_output_1


class Reshape85(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 1500, 1500))
        return reshape_output_1


class Reshape86(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 1500, 64))
        return reshape_output_1


class Reshape87(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 1, 1500))
        return reshape_output_1


class Reshape88(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 1, 1500))
        return reshape_output_1


class Reshape89(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1000))
        return reshape_output_1


class Reshape90(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 64, 16384))
        return reshape_output_1


class Reshape91(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 128, 128))
        return reshape_output_1


class Reshape92(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16384, 1, 64))
        return reshape_output_1


class Reshape93(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 128, 64))
        return reshape_output_1


class Reshape94(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 64, 128, 128))
        return reshape_output_1


class Reshape95(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 64, 256))
        return reshape_output_1


class Reshape96(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(256, 64))
        return reshape_output_1


class Reshape97(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 1, 64))
        return reshape_output_1


class Reshape98(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 2, 32))
        return reshape_output_1


class Reshape99(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 64))
        return reshape_output_1


class Reshape100(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 16384, 256))
        return reshape_output_1


class Reshape101(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16384, 256))
        return reshape_output_1


class Reshape102(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 128, 128))
        return reshape_output_1


class Reshape103(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 16384))
        return reshape_output_1


class Reshape104(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 4096))
        return reshape_output_1


class Reshape105(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(128, 4096))
        return reshape_output_1


class Reshape106(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4096, 2, 64))
        return reshape_output_1


class Reshape107(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 64, 64, 128))
        return reshape_output_1


class Reshape108(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 4096, 64))
        return reshape_output_1


class Reshape109(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 64, 64))
        return reshape_output_1


class Reshape110(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 256))
        return reshape_output_1


class Reshape111(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(256, 128))
        return reshape_output_1


class Reshape112(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 2, 64))
        return reshape_output_1


class Reshape113(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 128))
        return reshape_output_1


class Reshape114(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 64, 256))
        return reshape_output_1


class Reshape115(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2, 4096, 256))
        return reshape_output_1


class Reshape116(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 4096, 256))
        return reshape_output_1


class Reshape117(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 256, 64))
        return reshape_output_1


class Reshape118(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2, 4096, 64))
        return reshape_output_1


class Reshape119(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(4096, 128))
        return reshape_output_1


class Reshape120(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4096, 128))
        return reshape_output_1


class Reshape121(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 64, 128))
        return reshape_output_1


class Reshape122(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 4096, 128))
        return reshape_output_1


class Reshape123(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 512, 64, 64))
        return reshape_output_1


class Reshape124(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 512, 4096))
        return reshape_output_1


class Reshape125(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 320, 1024))
        return reshape_output_1


class Reshape126(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1024, 5, 64))
        return reshape_output_1


class Reshape127(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 32, 320))
        return reshape_output_1


class Reshape128(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(5, 1024, 64))
        return reshape_output_1


class Reshape129(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 320, 32, 32))
        return reshape_output_1


class Reshape130(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 320, 256))
        return reshape_output_1


class Reshape131(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(256, 320))
        return reshape_output_1


class Reshape132(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 5, 64))
        return reshape_output_1


class Reshape133(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 320))
        return reshape_output_1


class Reshape134(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(5, 64, 256))
        return reshape_output_1


class Reshape135(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 5, 1024, 256))
        return reshape_output_1


class Reshape136(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(5, 1024, 256))
        return reshape_output_1


class Reshape137(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(5, 256, 64))
        return reshape_output_1


class Reshape138(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 5, 1024, 64))
        return reshape_output_1


class Reshape139(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1024, 320))
        return reshape_output_1


class Reshape140(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1024, 320))
        return reshape_output_1


class Reshape141(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1280, 32, 32))
        return reshape_output_1


class Reshape142(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1280, 1024))
        return reshape_output_1


class Reshape143(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 512, 256))
        return reshape_output_1


class Reshape144(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(256, 512))
        return reshape_output_1


class Reshape145(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 8, 64))
        return reshape_output_1


class Reshape146(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 16, 512))
        return reshape_output_1


class Reshape147(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 512))
        return reshape_output_1


class Reshape148(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(4, 64, 512))
        return reshape_output_1


class Reshape149(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 256, 64))
        return reshape_output_1


class Reshape150(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 64, 256))
        return reshape_output_1


class Reshape151(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 256, 256))
        return reshape_output_1


class Reshape152(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 256, 256))
        return reshape_output_1


class Reshape153(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 256, 64))
        return reshape_output_1


class Reshape154(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2048, 16, 16))
        return reshape_output_1


class Reshape155(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2048, 256))
        return reshape_output_1


class Reshape156(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 768, 16, 16))
        return reshape_output_1


class Reshape157(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 768, 32, 32))
        return reshape_output_1


class Reshape158(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 768, 64, 64))
        return reshape_output_1


class Reshape159(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 768, 128, 128))
        return reshape_output_1


class Reshape160(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(11, 768))
        return reshape_output_1


class Reshape161(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 11, 12, 64))
        return reshape_output_1


class Reshape162(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 11, 768))
        return reshape_output_1


class Reshape163(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 11, 64))
        return reshape_output_1


class Reshape164(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 64, 11))
        return reshape_output_1


class Reshape165(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 11, 64))
        return reshape_output_1


class Reshape166(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(9, 768))
        return reshape_output_1


class Reshape167(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 9, 12, 64))
        return reshape_output_1


class Reshape168(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 9, 768))
        return reshape_output_1


class Reshape169(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 9, 64))
        return reshape_output_1


class Reshape170(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 64, 9))
        return reshape_output_1


class Reshape171(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 9, 9))
        return reshape_output_1


class Reshape172(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 9, 9))
        return reshape_output_1


class Reshape173(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 9, 64))
        return reshape_output_1


class Reshape174(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 512, 1, 1))
        return reshape_output_1


class Reshape175(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(128, 2048))
        return reshape_output_1


class Reshape176(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 16, 128))
        return reshape_output_1


class Reshape177(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 2048))
        return reshape_output_1


class Reshape178(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 128, 128))
        return reshape_output_1


class Reshape179(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 128, 128))
        return reshape_output_1


class Reshape180(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 128, 64))
        return reshape_output_1


class Reshape181(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 64, 128, 64))
        return reshape_output_1


class Reshape182(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(588, 2048))
        return reshape_output_1


class Reshape183(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 588, 16, 128))
        return reshape_output_1


class Reshape184(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 588, 2048))
        return reshape_output_1


class Reshape185(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 588, 128))
        return reshape_output_1


class Reshape186(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 588, 588))
        return reshape_output_1


class Reshape187(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 588, 588))
        return reshape_output_1


class Reshape188(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 588, 128))
        return reshape_output_1


class Reshape189(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 588, 5504))
        return reshape_output_1


class Reshape190(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1000, 1, 1))
        return reshape_output_1


class Reshape191(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(40, 1, 3, 3))
        return reshape_output_1


class Reshape192(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(24, 1, 3, 3))
        return reshape_output_1


class Reshape193(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(144, 1, 3, 3))
        return reshape_output_1


class Reshape194(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(192, 1, 3, 3))
        return reshape_output_1


class Reshape195(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(192, 1, 5, 5))
        return reshape_output_1


class Reshape196(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(288, 1, 5, 5))
        return reshape_output_1


class Reshape197(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(288, 1, 3, 3))
        return reshape_output_1


class Reshape198(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(576, 1, 3, 3))
        return reshape_output_1


class Reshape199(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(576, 1, 5, 5))
        return reshape_output_1


class Reshape200(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(816, 1, 5, 5))
        return reshape_output_1


class Reshape201(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1392, 1, 5, 5))
        return reshape_output_1


class Reshape202(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1392, 1, 3, 3))
        return reshape_output_1


class Reshape203(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2304, 1, 3, 3))
        return reshape_output_1


class Reshape204(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1536, 1, 1))
        return reshape_output_1


class Reshape205(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(256, 1, 3, 3))
        return reshape_output_1


class Reshape206(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(512, 1, 3, 3))
        return reshape_output_1


class Reshape207(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(768, 1, 3, 3))
        return reshape_output_1


class Reshape208(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(960, 1, 3, 3))
        return reshape_output_1


class Reshape209(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1536, 1, 3, 3))
        return reshape_output_1


class Reshape210(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1280, 1, 1))
        return reshape_output_1


class Reshape211(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 1, 3, 3))
        return reshape_output_1


class Reshape212(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(48, 1, 3, 3))
        return reshape_output_1


class Reshape213(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 1, 3, 3))
        return reshape_output_1


class Reshape214(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 1, 3, 3))
        return reshape_output_1


class Reshape215(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(36, 1, 3, 3))
        return reshape_output_1


class Reshape216(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(72, 1, 5, 5))
        return reshape_output_1


class Reshape217(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(20, 1, 3, 3))
        return reshape_output_1


class Reshape218(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(24, 1, 5, 5))
        return reshape_output_1


class Reshape219(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(60, 1, 3, 3))
        return reshape_output_1


class Reshape220(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(120, 1, 3, 3))
        return reshape_output_1


class Reshape221(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(240, 1, 3, 3))
        return reshape_output_1


class Reshape222(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(100, 1, 3, 3))
        return reshape_output_1


class Reshape223(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(92, 1, 3, 3))
        return reshape_output_1


class Reshape224(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(56, 1, 3, 3))
        return reshape_output_1


class Reshape225(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(80, 1, 3, 3))
        return reshape_output_1


class Reshape226(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(336, 1, 3, 3))
        return reshape_output_1


class Reshape227(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(672, 1, 5, 5))
        return reshape_output_1


class Reshape228(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(112, 1, 5, 5))
        return reshape_output_1


class Reshape229(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(480, 1, 3, 3))
        return reshape_output_1


class Reshape230(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256))
        return reshape_output_1


class Reshape231(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(256, 768))
        return reshape_output_1


class Reshape232(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 12, 64))
        return reshape_output_1


class Reshape233(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 768))
        return reshape_output_1


class Reshape234(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 256, 64))
        return reshape_output_1


class Reshape235(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 256, 256))
        return reshape_output_1


class Reshape236(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 256))
        return reshape_output_1


class Reshape237(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 256, 256))
        return reshape_output_1


class Reshape238(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 256, 64))
        return reshape_output_1


class Reshape239(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 3072))
        return reshape_output_1


class Reshape240(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(256, 3072))
        return reshape_output_1


class Reshape241(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 7))
        return reshape_output_1


class Reshape242(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(7, 768))
        return reshape_output_1


class Reshape243(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 7, 12, 64))
        return reshape_output_1


class Reshape244(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 7, 768))
        return reshape_output_1


class Reshape245(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 7, 64))
        return reshape_output_1


class Reshape246(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 7, 7))
        return reshape_output_1


class Reshape247(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 7, 7))
        return reshape_output_1


class Reshape248(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 7, 64))
        return reshape_output_1


class Reshape249(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 7, 3072))
        return reshape_output_1


class Reshape250(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(7, 3072))
        return reshape_output_1


class Reshape251(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(7, 2))
        return reshape_output_1


class Reshape252(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2))
        return reshape_output_1


class Reshape253(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(3072, 1, 4))
        return reshape_output_1


class Reshape254(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(6, 3072))
        return reshape_output_1


class Reshape255(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 6, 96))
        return reshape_output_1


class Reshape256(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 3072, 16))
        return reshape_output_1


class Reshape257(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 6, 16))
        return reshape_output_1


class Reshape258(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16))
        return reshape_output_1


class Reshape259(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 3072))
        return reshape_output_1


class Reshape260(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 1, 3, 3))
        return reshape_output_1


class Reshape261(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(96, 1, 3, 3))
        return reshape_output_1


class Reshape262(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(384, 1, 3, 3))
        return reshape_output_1


class Reshape263(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(72, 1, 3, 3))
        return reshape_output_1


class Reshape264(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(88, 1, 3, 3))
        return reshape_output_1


class Reshape265(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(96, 1, 5, 5))
        return reshape_output_1


class Reshape266(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(240, 1, 5, 5))
        return reshape_output_1


class Reshape267(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(120, 1, 5, 5))
        return reshape_output_1


class Reshape268(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(144, 1, 5, 5))
        return reshape_output_1


class Reshape269(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 576, 1, 1))
        return reshape_output_1


class Reshape270(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 1024))
        return reshape_output_1


class Reshape271(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 16, 64))
        return reshape_output_1


class Reshape272(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 1024))
        return reshape_output_1


class Reshape273(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 32, 64))
        return reshape_output_1


class Reshape274(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 32, 32))
        return reshape_output_1


class Reshape275(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 32, 32))
        return reshape_output_1


class Reshape276(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 32, 64))
        return reshape_output_1


class Reshape277(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 512))
        return reshape_output_1


class Reshape278(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(256, 2560))
        return reshape_output_1


class Reshape279(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 32, 80))
        return reshape_output_1


class Reshape280(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 2560))
        return reshape_output_1


class Reshape281(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 256, 80))
        return reshape_output_1


class Reshape282(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 256, 256))
        return reshape_output_1


class Reshape283(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 256, 256))
        return reshape_output_1


class Reshape284(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 65536))
        return reshape_output_1


class Reshape285(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 256, 80))
        return reshape_output_1


class Reshape286(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 10240))
        return reshape_output_1


class Reshape287(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(39, 896))
        return reshape_output_1


class Reshape288(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 39, 14, 64))
        return reshape_output_1


class Reshape289(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 39, 896))
        return reshape_output_1


class Reshape290(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(14, 39, 64))
        return reshape_output_1


class Reshape291(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 39, 128))
        return reshape_output_1


class Reshape292(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 39, 2, 64))
        return reshape_output_1


class Reshape293(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 14, 39, 39))
        return reshape_output_1


class Reshape294(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(14, 39, 39))
        return reshape_output_1


class Reshape295(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 14, 39, 64))
        return reshape_output_1


class Reshape296(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 39, 4864))
        return reshape_output_1


class Reshape297(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(29, 1536))
        return reshape_output_1


class Reshape298(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 29, 12, 128))
        return reshape_output_1


class Reshape299(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 29, 1536))
        return reshape_output_1


class Reshape300(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 29, 128))
        return reshape_output_1


class Reshape301(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 29, 256))
        return reshape_output_1


class Reshape302(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 29, 2, 128))
        return reshape_output_1


class Reshape303(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 29, 29))
        return reshape_output_1


class Reshape304(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 29, 29))
        return reshape_output_1


class Reshape305(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 29, 128))
        return reshape_output_1


class Reshape306(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 29, 8960))
        return reshape_output_1


class Reshape307(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(39, 1536))
        return reshape_output_1


class Reshape308(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 39, 12, 128))
        return reshape_output_1


class Reshape309(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 39, 1536))
        return reshape_output_1


class Reshape310(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 39, 128))
        return reshape_output_1


class Reshape311(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 39, 256))
        return reshape_output_1


class Reshape312(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 39, 2, 128))
        return reshape_output_1


class Reshape313(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 39, 39))
        return reshape_output_1


class Reshape314(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 39, 39))
        return reshape_output_1


class Reshape315(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 39, 128))
        return reshape_output_1


class Reshape316(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 39, 8960))
        return reshape_output_1


class Reshape317(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 912, 1, 1))
        return reshape_output_1


class Reshape318(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 400, 1, 1))
        return reshape_output_1


class Reshape319(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1920, 1, 1))
        return reshape_output_1


class Reshape320(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 64, 16384, 1))
        return reshape_output_1


class Reshape321(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 16384, 1))
        return reshape_output_1


class Reshape322(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 4096, 1))
        return reshape_output_1


class Reshape323(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 512, 4096, 1))
        return reshape_output_1


class Reshape324(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 320, 1024, 1))
        return reshape_output_1


class Reshape325(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1280, 1, 3, 3))
        return reshape_output_1


class Reshape326(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1280, 1024, 1))
        return reshape_output_1


class Reshape327(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 512, 256, 1))
        return reshape_output_1


class Reshape328(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 512, 1))
        return reshape_output_1


class Reshape329(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2048, 1, 3, 3))
        return reshape_output_1


class Reshape330(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2048, 256, 1))
        return reshape_output_1


class Reshape331(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 16, 16))
        return reshape_output_1


class Reshape332(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(256, 256))
        return reshape_output_1


class Reshape333(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 8, 32))
        return reshape_output_1


class Reshape334(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 256))
        return reshape_output_1


class Reshape335(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 16, 256))
        return reshape_output_1


class Reshape336(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 32, 32))
        return reshape_output_1


class Reshape337(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 64, 64))
        return reshape_output_1


class Reshape338(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1024, 49))
        return reshape_output_1


class Reshape339(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(50, 1024))
        return reshape_output_1


class Reshape340(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(50, 1, 3072))
        return reshape_output_1


class Reshape341(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(50, 1, 3, 1024))
        return reshape_output_1


class Reshape342(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(50, 16, 64))
        return reshape_output_1


class Reshape343(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 50, 64))
        return reshape_output_1


class Reshape344(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 50, 64))
        return reshape_output_1


class Reshape345(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 50, 50))
        return reshape_output_1


class Reshape346(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 50, 50))
        return reshape_output_1


class Reshape347(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(50, 1, 1024))
        return reshape_output_1


class Reshape348(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1024))
        return reshape_output_1


class Reshape349(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 1, 1024))
        return reshape_output_1


class Reshape350(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 4480))
        return reshape_output_1


class Reshape351(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 1120))
        return reshape_output_1


class Reshape352(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 280))
        return reshape_output_1


class Reshape353(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 80, 4480))
        return reshape_output_1


class Reshape354(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 80, 1120))
        return reshape_output_1


class Reshape355(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 80, 280))
        return reshape_output_1


class Reshape356(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(640, 1, 3, 3))
        return reshape_output_1


class Reshape357(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(320, 1, 3, 3))
        return reshape_output_1


class Reshape358(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 5, 128, 400))
        return reshape_output_1


class Reshape359(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(5, 64, 400))
        return reshape_output_1


class Reshape360(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 320, 20, 20))
        return reshape_output_1


class Reshape361(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(5, 400, 32))
        return reshape_output_1


class Reshape362(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 5, 400, 400))
        return reshape_output_1


class Reshape363(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(5, 400, 400))
        return reshape_output_1


class Reshape364(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 144, 6400))
        return reshape_output_1


class Reshape365(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 144, 1600))
        return reshape_output_1


class Reshape366(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 144, 400))
        return reshape_output_1


class Reshape367(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 16, 8400))
        return reshape_output_1


class Reshape368(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 8400))
        return reshape_output_1


class Reshape369(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 85, 6400, 1))
        return reshape_output_1


class Reshape370(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 85, 1600, 1))
        return reshape_output_1


class Reshape371(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 85, 400, 1))
        return reshape_output_1


class Reshape372(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1280))
        return reshape_output_1


class Reshape373(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 5, 256))
        return reshape_output_1


class Reshape374(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(5, 256))
        return reshape_output_1


class Reshape375(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 5, 8, 32))
        return reshape_output_1


class Reshape376(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 5, 32))
        return reshape_output_1


class Reshape377(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 32, 5))
        return reshape_output_1


class Reshape378(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 5, 5))
        return reshape_output_1


class Reshape379(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 5, 5))
        return reshape_output_1


class Reshape380(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 5, 32))
        return reshape_output_1


class Reshape381(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 5, 8, 16))
        return reshape_output_1


class Reshape382(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 5, 16))
        return reshape_output_1


class Reshape383(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 5, 14, 5, 14, 768))
        return reshape_output_1


class Reshape384(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(4900, 768))
        return reshape_output_1


class Reshape385(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(25, 14, 14, 2304))
        return reshape_output_1


class Reshape386(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(25, 196, 3, 12, 64))
        return reshape_output_1


class Reshape387(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(3, 300, 196, 64))
        return reshape_output_1


class Reshape388(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(300, 14, 14, 14, 14))
        return reshape_output_1


class Reshape389(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(300, 14, 14, 64))
        return reshape_output_1


class Reshape390(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(25, 12, 14, 14, 64))
        return reshape_output_1


class Reshape391(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 27))
        return reshape_output_1


class Reshape392(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(300, 196, 196))
        return reshape_output_1


class Reshape393(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(25, 14, 14, 768))
        return reshape_output_1


class Reshape394(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 5, 5, 14, 14, 768))
        return reshape_output_1


class Reshape395(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 70, 70, 768))
        return reshape_output_1


class Reshape396(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(4096, 768))
        return reshape_output_1


class Reshape397(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 64, 64, 3072))
        return reshape_output_1


class Reshape398(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(4096, 3072))
        return reshape_output_1


class Reshape399(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 64, 64, 768))
        return reshape_output_1


class Reshape400(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 64, 64, 2304))
        return reshape_output_1


class Reshape401(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4096, 3, 12, 64))
        return reshape_output_1


class Reshape402(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(3, 12, 4096, 64))
        return reshape_output_1


class Reshape403(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 64, 64, 64, 64))
        return reshape_output_1


class Reshape404(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 64, 64, 64))
        return reshape_output_1


class Reshape405(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 64, 64, 64))
        return reshape_output_1


class Reshape406(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 127))
        return reshape_output_1


class Reshape407(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 4096, 4096))
        return reshape_output_1


class Reshape408(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 4096))
        return reshape_output_1


class Reshape409(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(4096, 256))
        return reshape_output_1


class Reshape410(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4096, 8, 16))
        return reshape_output_1


class Reshape411(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 16, 4096))
        return reshape_output_1


class Reshape412(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 5, 4096))
        return reshape_output_1


class Reshape413(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 5, 4096))
        return reshape_output_1


class Reshape414(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 4096, 16))
        return reshape_output_1


class Reshape415(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 5, 16))
        return reshape_output_1


class Reshape416(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(5, 128))
        return reshape_output_1


class Reshape417(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 5, 128))
        return reshape_output_1


class Reshape418(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 16, 5))
        return reshape_output_1


class Reshape419(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 4096, 5))
        return reshape_output_1


class Reshape420(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 4096, 5))
        return reshape_output_1


class Reshape421(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 4096, 16))
        return reshape_output_1


class Reshape422(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 4096, 256))
        return reshape_output_1


class Reshape423(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 32))
        return reshape_output_1


class Reshape424(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 4, 256, 256))
        return reshape_output_1


class Reshape425(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 768))
        return reshape_output_1


class Reshape426(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 12, 64))
        return reshape_output_1


class Reshape427(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 768))
        return reshape_output_1


class Reshape428(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 8, 64))
        return reshape_output_1


class Reshape429(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 64, 8))
        return reshape_output_1


class Reshape430(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 8, 8))
        return reshape_output_1


class Reshape431(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 8, 8))
        return reshape_output_1


class Reshape432(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 8, 64))
        return reshape_output_1


class Reshape433(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1152, 1, 1))
        return reshape_output_1


class Reshape434(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(25, 1, 96))
        return reshape_output_1


class Reshape435(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 768, 196, 1))
        return reshape_output_1


class Reshape436(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 768, 196))
        return reshape_output_1


class Reshape437(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(197, 768))
        return reshape_output_1


class Reshape438(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 197, 12, 64))
        return reshape_output_1


class Reshape439(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 197, 768))
        return reshape_output_1


class Reshape440(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(197, 1, 768))
        return reshape_output_1


class Reshape441(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 197, 64))
        return reshape_output_1


class Reshape442(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 197, 197))
        return reshape_output_1


class Reshape443(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 27, 27, 12))
        return reshape_output_1


class Reshape444(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(729, 12))
        return reshape_output_1


class Reshape445(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(197, 197, 12))
        return reshape_output_1


class Reshape446(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 197, 197))
        return reshape_output_1


class Reshape447(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 64, 197))
        return reshape_output_1


class Reshape448(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 197, 64))
        return reshape_output_1


class Reshape449(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1024, 196, 1))
        return reshape_output_1


class Reshape450(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(197, 1024))
        return reshape_output_1


class Reshape451(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 197, 16, 64))
        return reshape_output_1


class Reshape452(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 197, 1024))
        return reshape_output_1


class Reshape453(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 197, 64))
        return reshape_output_1


class Reshape454(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 197, 197))
        return reshape_output_1


class Reshape455(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 27, 27, 16))
        return reshape_output_1


class Reshape456(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(729, 16))
        return reshape_output_1


class Reshape457(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(197, 197, 16))
        return reshape_output_1


class Reshape458(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 197, 197))
        return reshape_output_1


class Reshape459(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 64, 197))
        return reshape_output_1


class Reshape460(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 197, 64))
        return reshape_output_1


class Reshape461(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 16, 3, 96))
        return reshape_output_1


class Reshape462(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 16, 96))
        return reshape_output_1


class Reshape463(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 32, 96))
        return reshape_output_1


class Reshape464(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 1, 32))
        return reshape_output_1


class Reshape465(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 32, 96))
        return reshape_output_1


class Reshape466(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 1536))
        return reshape_output_1


class Reshape467(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 1536))
        return reshape_output_1


class Reshape468(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 384, 196, 1))
        return reshape_output_1


class Reshape469(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(197, 384))
        return reshape_output_1


class Reshape470(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 197, 6, 64))
        return reshape_output_1


class Reshape471(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 197, 384))
        return reshape_output_1


class Reshape472(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(6, 197, 64))
        return reshape_output_1


class Reshape473(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 6, 197, 197))
        return reshape_output_1


class Reshape474(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(6, 197, 197))
        return reshape_output_1


class Reshape475(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 6, 197, 64))
        return reshape_output_1


class Reshape476(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 384))
        return reshape_output_1


class Reshape477(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(100, 256))
        return reshape_output_1


class Reshape478(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 100, 8, 32))
        return reshape_output_1


class Reshape479(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 100, 256))
        return reshape_output_1


class Reshape480(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 100, 32))
        return reshape_output_1


class Reshape481(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 100, 32))
        return reshape_output_1


class Reshape482(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 64, 1, 1))
        return reshape_output_1


class Reshape483(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 1, 1))
        return reshape_output_1


class Reshape484(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 1, 1))
        return reshape_output_1


class Reshape485(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1024, 1, 1))
        return reshape_output_1


class Reshape486(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 850, 1))
        return reshape_output_1


class Reshape487(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 25, 34))
        return reshape_output_1


class Reshape488(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 25, 34, 128, 1))
        return reshape_output_1


class Reshape489(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(850, 256))
        return reshape_output_1


class Reshape490(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 850, 8, 32))
        return reshape_output_1


class Reshape491(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 850, 256))
        return reshape_output_1


class Reshape492(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 850, 32))
        return reshape_output_1


class Reshape493(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 850, 850))
        return reshape_output_1


class Reshape494(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 850, 1))
        return reshape_output_1


class Reshape495(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 850, 850))
        return reshape_output_1


class Reshape496(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 850, 32))
        return reshape_output_1


class Reshape497(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 100, 850))
        return reshape_output_1


class Reshape498(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 100, 850))
        return reshape_output_1


class Reshape499(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 100, 92))
        return reshape_output_1


class Reshape500(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(480, 1, 5, 5))
        return reshape_output_1


class Reshape501(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1152, 1, 5, 5))
        return reshape_output_1


class Reshape502(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1152, 1, 3, 3))
        return reshape_output_1


class Reshape503(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(384, 1, 5, 5))
        return reshape_output_1


class Reshape504(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(768, 1, 5, 5))
        return reshape_output_1


class Reshape505(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1056, 1, 5, 5))
        return reshape_output_1


class Reshape506(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1824, 1, 5, 5))
        return reshape_output_1


class Reshape507(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1824, 1, 3, 3))
        return reshape_output_1


class Reshape508(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(3072, 1, 3, 3))
        return reshape_output_1


class Reshape509(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 224, 224))
        return reshape_output_1


class Reshape510(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 64, 256))
        return reshape_output_1


class Reshape511(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 768, 256, 1))
        return reshape_output_1


class Reshape512(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 257, 3, 12, 64))
        return reshape_output_1


class Reshape513(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 257, 64))
        return reshape_output_1


class Reshape514(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 257, 64))
        return reshape_output_1


class Reshape515(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 257, 257))
        return reshape_output_1


class Reshape516(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 257, 257))
        return reshape_output_1


class Reshape517(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 64, 257))
        return reshape_output_1


class Reshape518(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(257, 768))
        return reshape_output_1


class Reshape519(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 257, 768))
        return reshape_output_1


class Reshape520(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 27, 257, 1))
        return reshape_output_1


class Reshape521(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 768, 257, 1))
        return reshape_output_1


class Reshape522(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 2))
        return reshape_output_1


class Reshape523(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 512, 1, 322))
        return reshape_output_1


class Reshape524(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 3025, 64))
        return reshape_output_1


class Reshape525(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(3025, 322))
        return reshape_output_1


class Reshape526(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 3025, 1, 322))
        return reshape_output_1


class Reshape527(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 3025, 322))
        return reshape_output_1


class Reshape528(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 512, 3025))
        return reshape_output_1


class Reshape529(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 512, 3025))
        return reshape_output_1


class Reshape530(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 322, 3025))
        return reshape_output_1


class Reshape531(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(512, 1024))
        return reshape_output_1


class Reshape532(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 512, 8, 128))
        return reshape_output_1


class Reshape533(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 512, 1, 1024))
        return reshape_output_1


class Reshape534(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 512, 1024))
        return reshape_output_1


class Reshape535(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 512, 128))
        return reshape_output_1


class Reshape536(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 512, 512))
        return reshape_output_1


class Reshape537(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 512, 512))
        return reshape_output_1


class Reshape538(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 128, 512))
        return reshape_output_1


class Reshape539(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 512, 128))
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
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1024, 512))
        return reshape_output_1


class Reshape542(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 2048))
        return reshape_output_1


class Reshape543(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 12, 12))
        return reshape_output_1


class Reshape544(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 12, 12))
        return reshape_output_1


class Reshape545(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 8192))
        return reshape_output_1


class Reshape546(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(6, 1024))
        return reshape_output_1


class Reshape547(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 6, 16, 64))
        return reshape_output_1


class Reshape548(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 6, 1024))
        return reshape_output_1


class Reshape549(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 6, 64))
        return reshape_output_1


class Reshape550(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 6, 6))
        return reshape_output_1


class Reshape551(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 6, 6))
        return reshape_output_1


class Reshape552(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 6, 64))
        return reshape_output_1


class Reshape553(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 6, 2816))
        return reshape_output_1


class Reshape554(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 888, 1, 1))
        return reshape_output_1


class Reshape555(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1512, 1, 1))
        return reshape_output_1


class Reshape556(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 784, 1, 1))
        return reshape_output_1


class Reshape557(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 1, 3, 3))
        return reshape_output_1


class Reshape558(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(128, 1, 3, 3))
        return reshape_output_1


class Reshape559(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(728, 1, 3, 3))
        return reshape_output_1


class Reshape560(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1024, 1, 3, 3))
        return reshape_output_1


class Reshape561(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 255, 40, 40))
        return reshape_output_1


class Reshape562(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 255, 1600))
        return reshape_output_1


class Reshape563(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 3, 85, 1600))
        return reshape_output_1


class Reshape564(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4800, 85))
        return reshape_output_1


class Reshape565(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 255, 20, 20))
        return reshape_output_1


class Reshape566(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 255, 400))
        return reshape_output_1


class Reshape567(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 3, 85, 400))
        return reshape_output_1


class Reshape568(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1200, 85))
        return reshape_output_1


class Reshape569(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 255, 10, 10))
        return reshape_output_1


class Reshape570(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 255, 100))
        return reshape_output_1


class Reshape571(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 3, 85, 100))
        return reshape_output_1


class Reshape572(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 300, 85))
        return reshape_output_1


class Reshape573(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64,))
        return reshape_output_1


class Reshape574(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 1, 64))
        return reshape_output_1


class Reshape575(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(256,))
        return reshape_output_1


class Reshape576(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 1, 256))
        return reshape_output_1


class Reshape577(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(512,))
        return reshape_output_1


class Reshape578(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(128,))
        return reshape_output_1


class Reshape579(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 1, 128))
        return reshape_output_1


class Reshape580(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1024,))
        return reshape_output_1


class Reshape581(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2048,))
        return reshape_output_1


class Reshape582(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 1, 2048))
        return reshape_output_1


class Reshape583(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 100, 100))
        return reshape_output_1


class Reshape584(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 280))
        return reshape_output_1


class Reshape585(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(280, 256))
        return reshape_output_1


class Reshape586(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 280, 8, 32))
        return reshape_output_1


class Reshape587(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 280, 256))
        return reshape_output_1


class Reshape588(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 280, 32))
        return reshape_output_1


class Reshape589(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 280, 280))
        return reshape_output_1


class Reshape590(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 280, 280))
        return reshape_output_1


class Reshape591(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 280, 32))
        return reshape_output_1


class Reshape592(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 100, 280))
        return reshape_output_1


class Reshape593(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 100, 280))
        return reshape_output_1


class Reshape594(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(256, 1024))
        return reshape_output_1


class Reshape595(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 16, 64))
        return reshape_output_1


class Reshape596(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(10, 768))
        return reshape_output_1


class Reshape597(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 10, 12, 64))
        return reshape_output_1


class Reshape598(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 10, 768))
        return reshape_output_1


class Reshape599(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 10, 64))
        return reshape_output_1


class Reshape600(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 64, 10))
        return reshape_output_1


class Reshape601(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 10, 10))
        return reshape_output_1


class Reshape602(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 10, 10))
        return reshape_output_1


class Reshape603(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 10, 64))
        return reshape_output_1


class Reshape604(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 9216))
        return reshape_output_1


class Reshape605(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 4, 256))
        return reshape_output_1


class Reshape606(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 1024))
        return reshape_output_1


class Reshape607(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 16, 32, 1))
        return reshape_output_1


class Reshape608(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 256, 64))
        return reshape_output_1


class Reshape609(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 256, 256))
        return reshape_output_1


class Reshape610(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 256, 256))
        return reshape_output_1


class Reshape611(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 64, 256))
        return reshape_output_1


class Reshape612(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 256, 64))
        return reshape_output_1


class Reshape613(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(528, 1, 3, 3))
        return reshape_output_1


class Reshape614(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(528, 1, 5, 5))
        return reshape_output_1


class Reshape615(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(720, 1, 5, 5))
        return reshape_output_1


class Reshape616(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1248, 1, 5, 5))
        return reshape_output_1


class Reshape617(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1248, 1, 3, 3))
        return reshape_output_1


class Reshape618(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2112, 1, 3, 3))
        return reshape_output_1


class Reshape619(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1408, 1, 1))
        return reshape_output_1


class Reshape620(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(432, 1, 5, 5))
        return reshape_output_1


class Reshape621(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(432, 1, 3, 3))
        return reshape_output_1


class Reshape622(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(864, 1, 3, 3))
        return reshape_output_1


class Reshape623(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(864, 1, 5, 5))
        return reshape_output_1


class Reshape624(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1200, 1, 5, 5))
        return reshape_output_1


class Reshape625(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2064, 1, 5, 5))
        return reshape_output_1


class Reshape626(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2064, 1, 3, 3))
        return reshape_output_1


class Reshape627(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(3456, 1, 3, 3))
        return reshape_output_1


class Reshape628(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2304, 1, 1))
        return reshape_output_1


class Reshape629(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(336, 1, 5, 5))
        return reshape_output_1


class Reshape630(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(672, 1, 3, 3))
        return reshape_output_1


class Reshape631(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(960, 1, 5, 5))
        return reshape_output_1


class Reshape632(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1632, 1, 5, 5))
        return reshape_output_1


class Reshape633(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1632, 1, 3, 3))
        return reshape_output_1


class Reshape634(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2688, 1, 3, 3))
        return reshape_output_1


class Reshape635(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1792, 1, 1))
        return reshape_output_1


class Reshape636(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 64, 19200, 1))
        return reshape_output_1


class Reshape637(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 19200, 1, 64))
        return reshape_output_1


class Reshape638(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 120, 160, 64))
        return reshape_output_1


class Reshape639(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 64, 120, 160))
        return reshape_output_1


class Reshape640(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 64, 300))
        return reshape_output_1


class Reshape641(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(300, 64))
        return reshape_output_1


class Reshape642(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 300, 1, 64))
        return reshape_output_1


class Reshape643(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 300, 64))
        return reshape_output_1


class Reshape644(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 19200, 300))
        return reshape_output_1


class Reshape645(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 19200, 300))
        return reshape_output_1


class Reshape646(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 120, 160))
        return reshape_output_1


class Reshape647(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 19200, 1))
        return reshape_output_1


class Reshape648(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 4800, 1))
        return reshape_output_1


class Reshape649(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4800, 2, 64))
        return reshape_output_1


class Reshape650(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 60, 80, 128))
        return reshape_output_1


class Reshape651(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 4800, 64))
        return reshape_output_1


class Reshape652(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 60, 80))
        return reshape_output_1


class Reshape653(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 300))
        return reshape_output_1


class Reshape654(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(300, 128))
        return reshape_output_1


class Reshape655(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 300, 2, 64))
        return reshape_output_1


class Reshape656(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 300, 128))
        return reshape_output_1


class Reshape657(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 300, 64))
        return reshape_output_1


class Reshape658(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2, 4800, 300))
        return reshape_output_1


class Reshape659(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 4800, 300))
        return reshape_output_1


class Reshape660(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 64, 300))
        return reshape_output_1


class Reshape661(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2, 4800, 64))
        return reshape_output_1


class Reshape662(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(4800, 128))
        return reshape_output_1


class Reshape663(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4800, 128))
        return reshape_output_1


class Reshape664(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 512, 60, 80))
        return reshape_output_1


class Reshape665(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 512, 4800, 1))
        return reshape_output_1


class Reshape666(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 320, 1200, 1))
        return reshape_output_1


class Reshape667(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1200, 5, 64))
        return reshape_output_1


class Reshape668(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 30, 40, 320))
        return reshape_output_1


class Reshape669(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(5, 1200, 64))
        return reshape_output_1


class Reshape670(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 320, 30, 40))
        return reshape_output_1


class Reshape671(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 320, 300))
        return reshape_output_1


class Reshape672(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(300, 320))
        return reshape_output_1


class Reshape673(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 300, 5, 64))
        return reshape_output_1


class Reshape674(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 300, 320))
        return reshape_output_1


class Reshape675(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(5, 300, 64))
        return reshape_output_1


class Reshape676(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 5, 1200, 300))
        return reshape_output_1


class Reshape677(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(5, 1200, 300))
        return reshape_output_1


class Reshape678(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(5, 64, 300))
        return reshape_output_1


class Reshape679(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 5, 1200, 64))
        return reshape_output_1


class Reshape680(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1200, 320))
        return reshape_output_1


class Reshape681(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1200, 320))
        return reshape_output_1


class Reshape682(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1280, 30, 40))
        return reshape_output_1


class Reshape683(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1280, 1200, 1))
        return reshape_output_1


class Reshape684(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 512, 300, 1))
        return reshape_output_1


class Reshape685(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(300, 512))
        return reshape_output_1


class Reshape686(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 300, 8, 64))
        return reshape_output_1


class Reshape687(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 15, 20, 512))
        return reshape_output_1


class Reshape688(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 300, 512))
        return reshape_output_1


class Reshape689(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 300, 64))
        return reshape_output_1


class Reshape690(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 300, 300))
        return reshape_output_1


class Reshape691(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 300, 300))
        return reshape_output_1


class Reshape692(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 64, 300))
        return reshape_output_1


class Reshape693(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 300, 64))
        return reshape_output_1


class Reshape694(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2048, 15, 20))
        return reshape_output_1


class Reshape695(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2048, 300, 1))
        return reshape_output_1


class Reshape696(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 30, 40))
        return reshape_output_1


class Reshape697(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 60, 80))
        return reshape_output_1


class Reshape698(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 120, 160))
        return reshape_output_1


class Reshape699(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 3, 16, 16, 16, 16))
        return reshape_output_1


class Reshape700(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1024, 256, 1, 1))
        return reshape_output_1


class Reshape701(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1024, 512, 1))
        return reshape_output_1


class Reshape702(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(256, 1024, 1, 1))
        return reshape_output_1


class Reshape703(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 512, 1, 512))
        return reshape_output_1


class Reshape704(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 50176, 256))
        return reshape_output_1


class Reshape705(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(50176, 512))
        return reshape_output_1


class Reshape706(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 50176, 1, 512))
        return reshape_output_1


class Reshape707(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 50176, 512))
        return reshape_output_1


class Reshape708(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 512, 50176))
        return reshape_output_1


class Reshape709(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 512, 50176))
        return reshape_output_1


class Reshape710(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(35, 1536))
        return reshape_output_1


class Reshape711(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 35, 12, 128))
        return reshape_output_1


class Reshape712(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 35, 1536))
        return reshape_output_1


class Reshape713(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 35, 128))
        return reshape_output_1


class Reshape714(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 35, 256))
        return reshape_output_1


class Reshape715(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 35, 2, 128))
        return reshape_output_1


class Reshape716(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 35, 35))
        return reshape_output_1


class Reshape717(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 35, 35))
        return reshape_output_1


class Reshape718(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 35, 128))
        return reshape_output_1


class Reshape719(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 35, 8960))
        return reshape_output_1


class Reshape720(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 440, 1, 1))
        return reshape_output_1


class Reshape721(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 5776))
        return reshape_output_1


class Reshape722(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 2166))
        return reshape_output_1


class Reshape723(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 600))
        return reshape_output_1


class Reshape724(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 150))
        return reshape_output_1


class Reshape725(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 36))
        return reshape_output_1


class Reshape726(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 4))
        return reshape_output_1


class Reshape727(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 81, 5776))
        return reshape_output_1


class Reshape728(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 81, 2166))
        return reshape_output_1


class Reshape729(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 81, 600))
        return reshape_output_1


class Reshape730(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 81, 150))
        return reshape_output_1


class Reshape731(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 81, 36))
        return reshape_output_1


class Reshape732(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 81, 4))
        return reshape_output_1


class Reshape733(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 7, 8, 7, 96))
        return reshape_output_1


class Reshape734(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(3136, 96))
        return reshape_output_1


class Reshape735(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 49, 288))
        return reshape_output_1


class Reshape736(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 49, 3, 3, 32))
        return reshape_output_1


class Reshape737(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 3, 49, 32))
        return reshape_output_1


class Reshape738(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(192, 49, 32))
        return reshape_output_1


class Reshape739(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 3, 49, 49))
        return reshape_output_1


class Reshape740(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(49, 49, 3))
        return reshape_output_1


class Reshape741(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(192, 49, 49))
        return reshape_output_1


class Reshape742(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 64, 3, 49, 49))
        return reshape_output_1


class Reshape743(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(192, 32, 49))
        return reshape_output_1


class Reshape744(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 49, 96))
        return reshape_output_1


class Reshape745(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 56, 56, 96))
        return reshape_output_1


class Reshape746(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 8, 7, 7, 96))
        return reshape_output_1


class Reshape747(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 56, 56, 384))
        return reshape_output_1


class Reshape748(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 49, 384))
        return reshape_output_1


class Reshape749(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(3136, 384))
        return reshape_output_1


class Reshape750(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(784, 384))
        return reshape_output_1


class Reshape751(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 28, 28, 192))
        return reshape_output_1


class Reshape752(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 49, 192))
        return reshape_output_1


class Reshape753(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 7, 4, 7, 192))
        return reshape_output_1


class Reshape754(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(784, 192))
        return reshape_output_1


class Reshape755(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 49, 576))
        return reshape_output_1


class Reshape756(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 49, 3, 6, 32))
        return reshape_output_1


class Reshape757(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 6, 49, 32))
        return reshape_output_1


class Reshape758(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(96, 49, 32))
        return reshape_output_1


class Reshape759(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 6, 49, 49))
        return reshape_output_1


class Reshape760(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(49, 49, 6))
        return reshape_output_1


class Reshape761(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(96, 49, 49))
        return reshape_output_1


class Reshape762(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 6, 49, 49))
        return reshape_output_1


class Reshape763(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(96, 32, 49))
        return reshape_output_1


class Reshape764(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 4, 7, 7, 192))
        return reshape_output_1


class Reshape765(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 28, 28, 768))
        return reshape_output_1


class Reshape766(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 49, 768))
        return reshape_output_1


class Reshape767(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(784, 768))
        return reshape_output_1


class Reshape768(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(196, 768))
        return reshape_output_1


class Reshape769(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 14, 14, 384))
        return reshape_output_1


class Reshape770(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(4, 49, 384))
        return reshape_output_1


class Reshape771(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2, 7, 2, 7, 384))
        return reshape_output_1


class Reshape772(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(196, 384))
        return reshape_output_1


class Reshape773(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(4, 49, 1152))
        return reshape_output_1


class Reshape774(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(4, 49, 3, 12, 32))
        return reshape_output_1


class Reshape775(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(4, 12, 49, 32))
        return reshape_output_1


class Reshape776(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(48, 49, 32))
        return reshape_output_1


class Reshape777(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(4, 12, 49, 49))
        return reshape_output_1


class Reshape778(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(49, 49, 12))
        return reshape_output_1


class Reshape779(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(48, 49, 49))
        return reshape_output_1


class Reshape780(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 12, 49, 49))
        return reshape_output_1


class Reshape781(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(48, 32, 49))
        return reshape_output_1


class Reshape782(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2, 2, 7, 7, 384))
        return reshape_output_1


class Reshape783(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 14, 14, 1536))
        return reshape_output_1


class Reshape784(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(4, 49, 1536))
        return reshape_output_1


class Reshape785(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(196, 1536))
        return reshape_output_1


class Reshape786(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(49, 1536))
        return reshape_output_1


class Reshape787(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 7, 7, 768))
        return reshape_output_1


class Reshape788(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 49, 768))
        return reshape_output_1


class Reshape789(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 7, 1, 7, 768))
        return reshape_output_1


class Reshape790(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(49, 768))
        return reshape_output_1


class Reshape791(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 49, 2304))
        return reshape_output_1


class Reshape792(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 49, 3, 24, 32))
        return reshape_output_1


class Reshape793(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 24, 49, 32))
        return reshape_output_1


class Reshape794(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(24, 49, 32))
        return reshape_output_1


class Reshape795(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 24, 49, 49))
        return reshape_output_1


class Reshape796(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(49, 49, 24))
        return reshape_output_1


class Reshape797(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(24, 49, 49))
        return reshape_output_1


class Reshape798(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(24, 32, 49))
        return reshape_output_1


class Reshape799(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 1, 7, 7, 768))
        return reshape_output_1


class Reshape800(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 7, 7, 3072))
        return reshape_output_1


class Reshape801(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 49, 3072))
        return reshape_output_1


class Reshape802(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(49, 3072))
        return reshape_output_1


class Reshape803(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 768, 1, 1))
        return reshape_output_1


class Reshape804(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 6, 64))
        return reshape_output_1


class Reshape805(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 384))
        return reshape_output_1


class Reshape806(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(6, 1, 64))
        return reshape_output_1


class Reshape807(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 6, 1, 1))
        return reshape_output_1


class Reshape808(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(6, 1, 1))
        return reshape_output_1


class Reshape809(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(6, 64, 1))
        return reshape_output_1


class Reshape810(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 6, 1, 64))
        return reshape_output_1


class Reshape811(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(61, 512))
        return reshape_output_1


class Reshape812(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 61, 6, 64))
        return reshape_output_1


class Reshape813(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(6, 61, 64))
        return reshape_output_1


class Reshape814(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 6, 61, 61))
        return reshape_output_1


class Reshape815(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(6, 61, 61))
        return reshape_output_1


class Reshape816(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(6, 64, 61))
        return reshape_output_1


class Reshape817(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 6, 61, 64))
        return reshape_output_1


class Reshape818(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(61, 384))
        return reshape_output_1


class Reshape819(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 61, 512))
        return reshape_output_1


class Reshape820(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 61, 8, 64))
        return reshape_output_1


class Reshape821(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 61, 1024))
        return reshape_output_1


class Reshape822(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 6, 1, 61))
        return reshape_output_1


class Reshape823(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(6, 1, 61))
        return reshape_output_1


class Reshape824(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 1024))
        return reshape_output_1


class Reshape825(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 255, 80, 80))
        return reshape_output_1


class Reshape826(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 255, 6400))
        return reshape_output_1


class Reshape827(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 3, 85, 6400))
        return reshape_output_1


class Reshape828(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 19200, 85))
        return reshape_output_1


class Reshape829(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 255, 60, 60))
        return reshape_output_1


class Reshape830(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 255, 3600))
        return reshape_output_1


class Reshape831(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 3, 85, 3600))
        return reshape_output_1


class Reshape832(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 10800, 85))
        return reshape_output_1


class Reshape833(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 255, 30, 30))
        return reshape_output_1


class Reshape834(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 255, 900))
        return reshape_output_1


class Reshape835(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 3, 85, 900))
        return reshape_output_1


class Reshape836(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2700, 85))
        return reshape_output_1


class Reshape837(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 255, 15, 15))
        return reshape_output_1


class Reshape838(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 255, 225))
        return reshape_output_1


class Reshape839(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 3, 85, 225))
        return reshape_output_1


class Reshape840(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 675, 85))
        return reshape_output_1


class Reshape841(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1536))
        return reshape_output_1


class Reshape842(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(13, 384))
        return reshape_output_1


class Reshape843(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 13, 12, 32))
        return reshape_output_1


class Reshape844(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 13, 384))
        return reshape_output_1


class Reshape845(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 13, 32))
        return reshape_output_1


class Reshape846(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 32, 13))
        return reshape_output_1


class Reshape847(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 13, 13))
        return reshape_output_1


class Reshape848(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 13, 13))
        return reshape_output_1


class Reshape849(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 13, 32))
        return reshape_output_1


class Reshape850(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(14, 768))
        return reshape_output_1


class Reshape851(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 14, 12, 64))
        return reshape_output_1


class Reshape852(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 14, 768))
        return reshape_output_1


class Reshape853(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 14, 64))
        return reshape_output_1


class Reshape854(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 14, 14))
        return reshape_output_1


class Reshape855(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 14, 14))
        return reshape_output_1


class Reshape856(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 14, 64))
        return reshape_output_1


class Reshape857(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 14, 1))
        return reshape_output_1


class Reshape858(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 768, 49, 1))
        return reshape_output_1


class Reshape859(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 768, 49))
        return reshape_output_1


class Reshape860(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 9216, 1, 1))
        return reshape_output_1


class Reshape861(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(256, 2048))
        return reshape_output_1


class Reshape862(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 32, 64))
        return reshape_output_1


class Reshape863(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 16, 128))
        return reshape_output_1


class Reshape864(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 2048))
        return reshape_output_1


class Reshape865(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 16, 2048))
        return reshape_output_1


class Reshape866(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 256, 64))
        return reshape_output_1


class Reshape867(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 256, 64))
        return reshape_output_1


class Reshape868(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 2048))
        return reshape_output_1


class Reshape869(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 32, 64))
        return reshape_output_1


class Reshape870(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 2048))
        return reshape_output_1


class Reshape871(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 32, 64))
        return reshape_output_1


class Reshape872(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 32, 32))
        return reshape_output_1


class Reshape873(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 32, 32))
        return reshape_output_1


class Reshape874(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2240, 1, 1))
        return reshape_output_1


class Reshape875(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 3024, 1, 1))
        return reshape_output_1


class Reshape876(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 7, 8, 7, 128))
        return reshape_output_1


class Reshape877(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(3136, 128))
        return reshape_output_1


class Reshape878(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 49, 3, 4, 32))
        return reshape_output_1


class Reshape879(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 4, 49, 32))
        return reshape_output_1


class Reshape880(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(256, 49, 32))
        return reshape_output_1


class Reshape881(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 4, 49, 49))
        return reshape_output_1


class Reshape882(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(49, 49, 4))
        return reshape_output_1


class Reshape883(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(256, 49, 49))
        return reshape_output_1


class Reshape884(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 64, 4, 49, 49))
        return reshape_output_1


class Reshape885(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(256, 32, 49))
        return reshape_output_1


class Reshape886(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 49, 128))
        return reshape_output_1


class Reshape887(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 56, 56, 128))
        return reshape_output_1


class Reshape888(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 8, 7, 7, 128))
        return reshape_output_1


class Reshape889(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 56, 56, 512))
        return reshape_output_1


class Reshape890(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(3136, 512))
        return reshape_output_1


class Reshape891(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(784, 512))
        return reshape_output_1


class Reshape892(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 28, 28, 256))
        return reshape_output_1


class Reshape893(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 49, 256))
        return reshape_output_1


class Reshape894(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 7, 4, 7, 256))
        return reshape_output_1


class Reshape895(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(784, 256))
        return reshape_output_1


class Reshape896(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 49, 3, 8, 32))
        return reshape_output_1


class Reshape897(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 8, 49, 32))
        return reshape_output_1


class Reshape898(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(128, 49, 32))
        return reshape_output_1


class Reshape899(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 8, 49, 49))
        return reshape_output_1


class Reshape900(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(49, 49, 8))
        return reshape_output_1


class Reshape901(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(128, 49, 49))
        return reshape_output_1


class Reshape902(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 8, 49, 49))
        return reshape_output_1


class Reshape903(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(128, 32, 49))
        return reshape_output_1


class Reshape904(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 4, 7, 7, 256))
        return reshape_output_1


class Reshape905(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 28, 28, 1024))
        return reshape_output_1


class Reshape906(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(784, 1024))
        return reshape_output_1


class Reshape907(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(196, 1024))
        return reshape_output_1


class Reshape908(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 14, 14, 512))
        return reshape_output_1


class Reshape909(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(4, 49, 512))
        return reshape_output_1


class Reshape910(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2, 7, 2, 7, 512))
        return reshape_output_1


class Reshape911(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(196, 512))
        return reshape_output_1


class Reshape912(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(4, 49, 3, 16, 32))
        return reshape_output_1


class Reshape913(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(4, 16, 49, 32))
        return reshape_output_1


class Reshape914(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 49, 32))
        return reshape_output_1


class Reshape915(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(4, 16, 49, 49))
        return reshape_output_1


class Reshape916(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(49, 49, 16))
        return reshape_output_1


class Reshape917(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 49, 49))
        return reshape_output_1


class Reshape918(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 16, 49, 49))
        return reshape_output_1


class Reshape919(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 32, 49))
        return reshape_output_1


class Reshape920(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2, 2, 7, 7, 512))
        return reshape_output_1


class Reshape921(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 14, 14, 2048))
        return reshape_output_1


class Reshape922(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(196, 2048))
        return reshape_output_1


class Reshape923(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(49, 2048))
        return reshape_output_1


class Reshape924(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 7, 7, 1024))
        return reshape_output_1


class Reshape925(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 49, 1024))
        return reshape_output_1


class Reshape926(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 7, 1, 7, 1024))
        return reshape_output_1


class Reshape927(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(49, 1024))
        return reshape_output_1


class Reshape928(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 49, 3, 32, 32))
        return reshape_output_1


class Reshape929(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 49, 32))
        return reshape_output_1


class Reshape930(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 49, 32))
        return reshape_output_1


class Reshape931(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 49, 49))
        return reshape_output_1


class Reshape932(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(49, 49, 32))
        return reshape_output_1


class Reshape933(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 49, 49))
        return reshape_output_1


class Reshape934(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 32, 49))
        return reshape_output_1


class Reshape935(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 1, 7, 7, 1024))
        return reshape_output_1


class Reshape936(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 7, 7, 4096))
        return reshape_output_1


class Reshape937(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(49, 4096))
        return reshape_output_1


class Reshape938(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(160, 1, 3, 3))
        return reshape_output_1


class Reshape939(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(224, 1, 3, 3))
        return reshape_output_1


class Reshape940(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(384, 80, 3, 1))
        return reshape_output_1


class Reshape941(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 384, 3000))
        return reshape_output_1


class Reshape942(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 384, 3000, 1))
        return reshape_output_1


class Reshape943(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(384, 384, 3, 1))
        return reshape_output_1


class Reshape944(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 384, 1500))
        return reshape_output_1


class Reshape945(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1500, 384))
        return reshape_output_1


class Reshape946(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1500, 6, 64))
        return reshape_output_1


class Reshape947(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1500, 384))
        return reshape_output_1


class Reshape948(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(6, 1500, 64))
        return reshape_output_1


class Reshape949(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 6, 1500, 1500))
        return reshape_output_1


class Reshape950(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(6, 1500, 1500))
        return reshape_output_1


class Reshape951(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 6, 1500, 64))
        return reshape_output_1


class Reshape952(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 6, 1, 1500))
        return reshape_output_1


class Reshape953(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(6, 1, 1500))
        return reshape_output_1


class Reshape954(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 256, 128))
        return reshape_output_1


class Reshape955(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 256, 128))
        return reshape_output_1


class Reshape956(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 85, 2704, 1))
        return reshape_output_1


class Reshape957(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 85, 676, 1))
        return reshape_output_1


class Reshape958(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 85, 169, 1))
        return reshape_output_1


class Reshape959(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(6, 768))
        return reshape_output_1


class Reshape960(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 6, 12, 64))
        return reshape_output_1


class Reshape961(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 6, 768))
        return reshape_output_1


class Reshape962(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 6, 64))
        return reshape_output_1


class Reshape963(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 64, 6))
        return reshape_output_1


class Reshape964(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 6, 6))
        return reshape_output_1


class Reshape965(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 6, 6))
        return reshape_output_1


class Reshape966(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 6, 64))
        return reshape_output_1


class Reshape967(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(384, 1024))
        return reshape_output_1


class Reshape968(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 384, 16, 64))
        return reshape_output_1


class Reshape969(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 384, 1024))
        return reshape_output_1


class Reshape970(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 384, 64))
        return reshape_output_1


class Reshape971(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 64, 384))
        return reshape_output_1


class Reshape972(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 384, 384))
        return reshape_output_1


class Reshape973(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 384, 384))
        return reshape_output_1


class Reshape974(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 384, 64))
        return reshape_output_1


class Reshape975(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 384, 1))
        return reshape_output_1


class Reshape976(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 16384))
        return reshape_output_1


class Reshape977(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16384, 1, 32))
        return reshape_output_1


class Reshape978(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 128, 32))
        return reshape_output_1


class Reshape979(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 128, 128))
        return reshape_output_1


class Reshape980(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 256))
        return reshape_output_1


class Reshape981(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(256, 32))
        return reshape_output_1


class Reshape982(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 1, 32))
        return reshape_output_1


class Reshape983(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 32))
        return reshape_output_1


class Reshape984(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 128, 128))
        return reshape_output_1


class Reshape985(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 16384))
        return reshape_output_1


class Reshape986(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 64, 4096))
        return reshape_output_1


class Reshape987(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4096, 2, 32))
        return reshape_output_1


class Reshape988(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 64, 64, 64))
        return reshape_output_1


class Reshape989(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 4096, 32))
        return reshape_output_1


class Reshape990(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 32, 256))
        return reshape_output_1


class Reshape991(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 256, 32))
        return reshape_output_1


class Reshape992(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2, 4096, 32))
        return reshape_output_1


class Reshape993(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(4096, 64))
        return reshape_output_1


class Reshape994(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4096, 64))
        return reshape_output_1


class Reshape995(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 160, 1024))
        return reshape_output_1


class Reshape996(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1024, 5, 32))
        return reshape_output_1


class Reshape997(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 32, 160))
        return reshape_output_1


class Reshape998(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(5, 1024, 32))
        return reshape_output_1


class Reshape999(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 160, 32, 32))
        return reshape_output_1


class Reshape1000(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 160, 256))
        return reshape_output_1


class Reshape1001(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(256, 160))
        return reshape_output_1


class Reshape1002(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 5, 32))
        return reshape_output_1


class Reshape1003(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 160))
        return reshape_output_1


class Reshape1004(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(5, 32, 256))
        return reshape_output_1


class Reshape1005(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(5, 256, 32))
        return reshape_output_1


class Reshape1006(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 5, 1024, 32))
        return reshape_output_1


class Reshape1007(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1024, 160))
        return reshape_output_1


class Reshape1008(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1024, 160))
        return reshape_output_1


class Reshape1009(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 640, 32, 32))
        return reshape_output_1


class Reshape1010(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 640, 1024))
        return reshape_output_1


class Reshape1011(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 256, 32))
        return reshape_output_1


class Reshape1012(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 32, 256))
        return reshape_output_1


class Reshape1013(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 256, 32))
        return reshape_output_1


class Reshape1014(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1024, 16, 16))
        return reshape_output_1


class Reshape1015(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1024, 256))
        return reshape_output_1


class Reshape1016(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 96, 4096))
        return reshape_output_1


class Reshape1017(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 64, 64, 96))
        return reshape_output_1


class Reshape1018(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 8, 8, 8, 96))
        return reshape_output_1


class Reshape1019(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4096, 96))
        return reshape_output_1


class Reshape1020(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(4096, 96))
        return reshape_output_1


class Reshape1021(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 64, 96))
        return reshape_output_1


class Reshape1022(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 64, 3, 32))
        return reshape_output_1


class Reshape1023(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(192, 64, 32))
        return reshape_output_1


class Reshape1024(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(192, 32, 64))
        return reshape_output_1


class Reshape1025(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 3, 64, 64))
        return reshape_output_1


class Reshape1026(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 15, 15, 512))
        return reshape_output_1


class Reshape1027(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(225, 512))
        return reshape_output_1


class Reshape1028(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(225, 3))
        return reshape_output_1


class Reshape1029(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 64, 3))
        return reshape_output_1


class Reshape1030(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(192, 64, 64))
        return reshape_output_1


class Reshape1031(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 64, 3, 64, 64))
        return reshape_output_1


class Reshape1032(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 3, 64, 32))
        return reshape_output_1


class Reshape1033(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1024, 384))
        return reshape_output_1


class Reshape1034(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1024, 192))
        return reshape_output_1


class Reshape1035(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 64, 192))
        return reshape_output_1


class Reshape1036(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 64, 6, 32))
        return reshape_output_1


class Reshape1037(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 32, 192))
        return reshape_output_1


class Reshape1038(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 8, 4, 8, 192))
        return reshape_output_1


class Reshape1039(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1024, 192))
        return reshape_output_1


class Reshape1040(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 4, 8, 8, 192))
        return reshape_output_1


class Reshape1041(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(96, 64, 32))
        return reshape_output_1


class Reshape1042(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(96, 32, 64))
        return reshape_output_1


class Reshape1043(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 6, 64, 64))
        return reshape_output_1


class Reshape1044(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(225, 6))
        return reshape_output_1


class Reshape1045(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 64, 6))
        return reshape_output_1


class Reshape1046(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(96, 64, 64))
        return reshape_output_1


class Reshape1047(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 6, 64, 64))
        return reshape_output_1


class Reshape1048(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 6, 64, 32))
        return reshape_output_1


class Reshape1049(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 384))
        return reshape_output_1


class Reshape1050(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(4, 64, 384))
        return reshape_output_1


class Reshape1051(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(4, 64, 12, 32))
        return reshape_output_1


class Reshape1052(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 16, 384))
        return reshape_output_1


class Reshape1053(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2, 8, 2, 8, 384))
        return reshape_output_1


class Reshape1054(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(256, 384))
        return reshape_output_1


class Reshape1055(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2, 2, 8, 8, 384))
        return reshape_output_1


class Reshape1056(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(48, 64, 32))
        return reshape_output_1


class Reshape1057(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(48, 32, 64))
        return reshape_output_1


class Reshape1058(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(4, 12, 64, 64))
        return reshape_output_1


class Reshape1059(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(225, 12))
        return reshape_output_1


class Reshape1060(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 64, 12))
        return reshape_output_1


class Reshape1061(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(48, 64, 64))
        return reshape_output_1


class Reshape1062(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 12, 64, 64))
        return reshape_output_1


class Reshape1063(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(4, 12, 64, 32))
        return reshape_output_1


class Reshape1064(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 1536))
        return reshape_output_1


class Reshape1065(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 64, 768))
        return reshape_output_1


class Reshape1066(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 64, 24, 32))
        return reshape_output_1


class Reshape1067(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 8, 768))
        return reshape_output_1


class Reshape1068(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 1, 8, 8, 768))
        return reshape_output_1


class Reshape1069(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 8, 1, 8, 768))
        return reshape_output_1


class Reshape1070(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 768))
        return reshape_output_1


class Reshape1071(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(24, 64, 32))
        return reshape_output_1


class Reshape1072(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(24, 32, 64))
        return reshape_output_1


class Reshape1073(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 24, 64, 64))
        return reshape_output_1


class Reshape1074(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(225, 24))
        return reshape_output_1


class Reshape1075(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 64, 24))
        return reshape_output_1


class Reshape1076(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(24, 64, 64))
        return reshape_output_1


class Reshape1077(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 24, 64, 32))
        return reshape_output_1


class Reshape1078(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 768, 8, 8))
        return reshape_output_1


class Reshape1079(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 3, 32, 32, 8, 8))
        return reshape_output_1


class Reshape1080(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 3, 256, 256))
        return reshape_output_1


class Reshape1081(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(15, 768))
        return reshape_output_1


class Reshape1082(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 15, 12, 64))
        return reshape_output_1


class Reshape1083(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 15, 768))
        return reshape_output_1


class Reshape1084(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 15, 64))
        return reshape_output_1


class Reshape1085(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 64, 15))
        return reshape_output_1


class Reshape1086(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 15, 15))
        return reshape_output_1


class Reshape1087(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 15, 15))
        return reshape_output_1


class Reshape1088(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 15, 64))
        return reshape_output_1


class Reshape1089(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(128, 1024))
        return reshape_output_1


class Reshape1090(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 16, 64))
        return reshape_output_1


class Reshape1091(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 1024))
        return reshape_output_1


class Reshape1092(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 128, 64))
        return reshape_output_1


class Reshape1093(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 128, 64))
        return reshape_output_1


class Reshape1094(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1664, 1, 1))
        return reshape_output_1


class Reshape1095(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 1))
        return reshape_output_1


class Reshape1096(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128))
        return reshape_output_1


class Reshape1097(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1,))
        return reshape_output_1


class Reshape1098(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 128, 256))
        return reshape_output_1


class Reshape1099(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 8192))
        return reshape_output_1


class Reshape1100(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2048, 1, 4))
        return reshape_output_1


class Reshape1101(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(6, 2048))
        return reshape_output_1


class Reshape1102(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 6, 64))
        return reshape_output_1


class Reshape1103(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2048, 16))
        return reshape_output_1


class Reshape1104(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(200, 1, 3, 3))
        return reshape_output_1


class Reshape1105(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(184, 1, 3, 3))
        return reshape_output_1


class Reshape1106(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 960, 1, 1))
        return reshape_output_1


class Reshape1107(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(35, 896))
        return reshape_output_1


class Reshape1108(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 35, 14, 64))
        return reshape_output_1


class Reshape1109(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 35, 896))
        return reshape_output_1


class Reshape1110(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(14, 35, 64))
        return reshape_output_1


class Reshape1111(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 35, 128))
        return reshape_output_1


class Reshape1112(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 35, 2, 64))
        return reshape_output_1


class Reshape1113(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 14, 35, 35))
        return reshape_output_1


class Reshape1114(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(14, 35, 35))
        return reshape_output_1


class Reshape1115(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 14, 35, 64))
        return reshape_output_1


class Reshape1116(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 35, 4864))
        return reshape_output_1


class Reshape1117(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 64, 1))
        return reshape_output_1


class Reshape1118(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 61, 64))
        return reshape_output_1


class Reshape1119(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 61, 61))
        return reshape_output_1


class Reshape1120(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 61, 61))
        return reshape_output_1


class Reshape1121(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 64, 61))
        return reshape_output_1


class Reshape1122(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 61, 64))
        return reshape_output_1


class Reshape1123(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 1, 61))
        return reshape_output_1


class Reshape1124(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 1, 61))
        return reshape_output_1


class Reshape1125(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(201, 768))
        return reshape_output_1


class Reshape1126(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 201, 12, 64))
        return reshape_output_1


class Reshape1127(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 201, 768))
        return reshape_output_1


class Reshape1128(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 201, 64))
        return reshape_output_1


class Reshape1129(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 201, 201))
        return reshape_output_1


class Reshape1130(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 201, 201))
        return reshape_output_1


class Reshape1131(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 64, 201))
        return reshape_output_1


class Reshape1132(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 201, 64))
        return reshape_output_1


class Reshape1133(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(50, 768))
        return reshape_output_1


class Reshape1134(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(50, 1, 2304))
        return reshape_output_1


class Reshape1135(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(50, 1, 3, 768))
        return reshape_output_1


class Reshape1136(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(50, 12, 64))
        return reshape_output_1


class Reshape1137(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 50, 64))
        return reshape_output_1


class Reshape1138(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 50, 64))
        return reshape_output_1


class Reshape1139(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 50, 50))
        return reshape_output_1


class Reshape1140(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 50, 50))
        return reshape_output_1


class Reshape1141(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(50, 1, 768))
        return reshape_output_1


class Reshape1142(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 32, 40, 40))
        return reshape_output_1


class Reshape1143(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 80, 128))
        return reshape_output_1


class Reshape1144(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 80, 4, 32))
        return reshape_output_1


class Reshape1145(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 40, 40))
        return reshape_output_1


class Reshape1146(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2, 32, 80, 80))
        return reshape_output_1


class Reshape1147(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 80, 64))
        return reshape_output_1


class Reshape1148(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 80, 2, 32))
        return reshape_output_1


class Reshape1149(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 64, 80, 80))
        return reshape_output_1


class Reshape1150(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 80, 8, 32))
        return reshape_output_1


class Reshape1151(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 80, 32))
        return reshape_output_1


class Reshape1152(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 9))
        return reshape_output_1


class Reshape1153(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 27, 8, 32))
        return reshape_output_1


class Reshape1154(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 32, 27))
        return reshape_output_1


class Reshape1155(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 80, 27))
        return reshape_output_1


class Reshape1156(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(80, 256))
        return reshape_output_1


class Reshape1157(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 80, 512))
        return reshape_output_1


class Reshape1158(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(80, 512))
        return reshape_output_1


class Reshape1159(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 32, 20, 20))
        return reshape_output_1


class Reshape1160(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 80, 256))
        return reshape_output_1


class Reshape1161(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 20, 20))
        return reshape_output_1


class Reshape1162(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(84, 8400))
        return reshape_output_1


class Reshape1163(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 768))
        return reshape_output_1


class Reshape1164(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 12, 64))
        return reshape_output_1


class Reshape1165(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 768))
        return reshape_output_1


class Reshape1166(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 16, 64))
        return reshape_output_1


class Reshape1167(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 16, 16))
        return reshape_output_1


class Reshape1168(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 16, 16))
        return reshape_output_1


class Reshape1169(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 16, 64))
        return reshape_output_1


class Reshape1170(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1920, 1, 3, 3))
        return reshape_output_1


class Reshape1171(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1344, 1, 5, 5))
        return reshape_output_1


class Reshape1172(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2304, 1, 5, 5))
        return reshape_output_1


class Reshape1173(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(3840, 1, 3, 3))
        return reshape_output_1


class Reshape1174(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2560, 1, 1))
        return reshape_output_1


class Reshape1175(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(72, 1, 1, 5))
        return reshape_output_1


class Reshape1176(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(72, 1, 5, 1))
        return reshape_output_1


class Reshape1177(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(120, 1, 1, 5))
        return reshape_output_1


class Reshape1178(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(120, 1, 5, 1))
        return reshape_output_1


class Reshape1179(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(240, 1, 1, 5))
        return reshape_output_1


class Reshape1180(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(240, 1, 5, 1))
        return reshape_output_1


class Reshape1181(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(200, 1, 1, 5))
        return reshape_output_1


class Reshape1182(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(200, 1, 5, 1))
        return reshape_output_1


class Reshape1183(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(184, 1, 1, 5))
        return reshape_output_1


class Reshape1184(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(184, 1, 5, 1))
        return reshape_output_1


class Reshape1185(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(480, 1, 1, 5))
        return reshape_output_1


class Reshape1186(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(480, 1, 5, 1))
        return reshape_output_1


class Reshape1187(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(672, 1, 1, 5))
        return reshape_output_1


class Reshape1188(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(672, 1, 5, 1))
        return reshape_output_1


class Reshape1189(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(960, 1, 1, 5))
        return reshape_output_1


class Reshape1190(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(960, 1, 5, 1))
        return reshape_output_1


class Reshape1191(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1296, 1, 1))
        return reshape_output_1


class Reshape1192(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 3712, 1, 1))
        return reshape_output_1


class Reshape1193(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 16384, 1))
        return reshape_output_1


class Reshape1194(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 16384, 1))
        return reshape_output_1


class Reshape1195(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 64, 4096, 1))
        return reshape_output_1


class Reshape1196(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 4096, 1))
        return reshape_output_1


class Reshape1197(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 160, 1024, 1))
        return reshape_output_1


class Reshape1198(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 640, 1024, 1))
        return reshape_output_1


class Reshape1199(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 256, 1))
        return reshape_output_1


class Reshape1200(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1024, 256, 1))
        return reshape_output_1


class Reshape1201(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 64, 128))
        return reshape_output_1


class Reshape1202(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 64, 128))
        return reshape_output_1


class Reshape1203(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 768, 128, 1))
        return reshape_output_1


class Reshape1204(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(768, 768, 1, 1))
        return reshape_output_1


class Reshape1205(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 768, 128))
        return reshape_output_1


class Reshape1206(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 8, 8, 8, 128))
        return reshape_output_1


class Reshape1207(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 64, 384))
        return reshape_output_1


class Reshape1208(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 64, 64, 384))
        return reshape_output_1


class Reshape1209(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 64, 3, 4, 32))
        return reshape_output_1


class Reshape1210(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 4, 64, 32))
        return reshape_output_1


class Reshape1211(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(256, 64, 32))
        return reshape_output_1


class Reshape1212(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 4, 64, 64))
        return reshape_output_1


class Reshape1213(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(225, 2))
        return reshape_output_1


class Reshape1214(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(225, 4))
        return reshape_output_1


class Reshape1215(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 64, 4))
        return reshape_output_1


class Reshape1216(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(256, 64, 64))
        return reshape_output_1


class Reshape1217(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 64, 4, 64, 64))
        return reshape_output_1


class Reshape1218(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(256, 32, 64))
        return reshape_output_1


class Reshape1219(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 64, 64, 512))
        return reshape_output_1


class Reshape1220(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(4096, 512))
        return reshape_output_1


class Reshape1221(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1024, 512))
        return reshape_output_1


class Reshape1222(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 32, 256))
        return reshape_output_1


class Reshape1223(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 8, 4, 8, 256))
        return reshape_output_1


class Reshape1224(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1024, 256))
        return reshape_output_1


class Reshape1225(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 64, 768))
        return reshape_output_1


class Reshape1226(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 32, 768))
        return reshape_output_1


class Reshape1227(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 64, 3, 8, 32))
        return reshape_output_1


class Reshape1228(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 8, 64, 32))
        return reshape_output_1


class Reshape1229(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(128, 64, 32))
        return reshape_output_1


class Reshape1230(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 8, 64, 64))
        return reshape_output_1


class Reshape1231(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(225, 8))
        return reshape_output_1


class Reshape1232(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 64, 8))
        return reshape_output_1


class Reshape1233(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(128, 64, 64))
        return reshape_output_1


class Reshape1234(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 8, 64, 64))
        return reshape_output_1


class Reshape1235(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(128, 32, 64))
        return reshape_output_1


class Reshape1236(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 4, 8, 8, 256))
        return reshape_output_1


class Reshape1237(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 32, 1024))
        return reshape_output_1


class Reshape1238(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1024, 1024))
        return reshape_output_1


class Reshape1239(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2, 8, 2, 8, 512))
        return reshape_output_1


class Reshape1240(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(4, 64, 1536))
        return reshape_output_1


class Reshape1241(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 16, 1536))
        return reshape_output_1


class Reshape1242(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(4, 64, 3, 16, 32))
        return reshape_output_1


class Reshape1243(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(4, 16, 64, 32))
        return reshape_output_1


class Reshape1244(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 64, 32))
        return reshape_output_1


class Reshape1245(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(4, 16, 64, 64))
        return reshape_output_1


class Reshape1246(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(225, 16))
        return reshape_output_1


class Reshape1247(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 64, 16))
        return reshape_output_1


class Reshape1248(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 64, 64))
        return reshape_output_1


class Reshape1249(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 16, 64, 64))
        return reshape_output_1


class Reshape1250(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 32, 64))
        return reshape_output_1


class Reshape1251(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2, 2, 8, 8, 512))
        return reshape_output_1


class Reshape1252(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 2048))
        return reshape_output_1


class Reshape1253(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 8, 1024))
        return reshape_output_1


class Reshape1254(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 64, 1024))
        return reshape_output_1


class Reshape1255(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 8, 1, 8, 1024))
        return reshape_output_1


class Reshape1256(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 1024))
        return reshape_output_1


class Reshape1257(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 64, 3072))
        return reshape_output_1


class Reshape1258(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 8, 3072))
        return reshape_output_1


class Reshape1259(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 64, 3, 32, 32))
        return reshape_output_1


class Reshape1260(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 64, 32))
        return reshape_output_1


class Reshape1261(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 64, 32))
        return reshape_output_1


class Reshape1262(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 64, 64))
        return reshape_output_1


class Reshape1263(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(225, 32))
        return reshape_output_1


class Reshape1264(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 64, 64))
        return reshape_output_1


class Reshape1265(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 1, 8, 8, 1024))
        return reshape_output_1


class Reshape1266(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 8, 4096))
        return reshape_output_1


class Reshape1267(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 4096))
        return reshape_output_1


class Reshape1268(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 64, 288))
        return reshape_output_1


class Reshape1269(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 64, 3, 3, 32))
        return reshape_output_1


class Reshape1270(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(4096, 384))
        return reshape_output_1


class Reshape1271(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 64, 576))
        return reshape_output_1


class Reshape1272(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 64, 3, 6, 32))
        return reshape_output_1


class Reshape1273(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1024, 768))
        return reshape_output_1


class Reshape1274(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(4, 64, 1152))
        return reshape_output_1


class Reshape1275(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(4, 64, 3, 12, 32))
        return reshape_output_1


class Reshape1276(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(256, 1536))
        return reshape_output_1


class Reshape1277(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 64, 2304))
        return reshape_output_1


class Reshape1278(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 64, 3, 24, 32))
        return reshape_output_1


class Reshape1279(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 3072))
        return reshape_output_1


class Reshape1280(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(204, 768))
        return reshape_output_1


class Reshape1281(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 204, 12, 64))
        return reshape_output_1


class Reshape1282(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 204, 768))
        return reshape_output_1


class Reshape1283(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 204, 64))
        return reshape_output_1


class Reshape1284(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 204, 204))
        return reshape_output_1


class Reshape1285(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 204, 204))
        return reshape_output_1


class Reshape1286(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 64, 204))
        return reshape_output_1


class Reshape1287(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 204, 64))
        return reshape_output_1


class Reshape1288(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1280, 1369))
        return reshape_output_1


class Reshape1289(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1370, 1280))
        return reshape_output_1


class Reshape1290(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1370, 1, 3840))
        return reshape_output_1


class Reshape1291(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1370, 1, 3, 1280))
        return reshape_output_1


class Reshape1292(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1370, 16, 80))
        return reshape_output_1


class Reshape1293(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 1370, 80))
        return reshape_output_1


class Reshape1294(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 1370, 80))
        return reshape_output_1


class Reshape1295(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 1370, 1370))
        return reshape_output_1


class Reshape1296(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 1370, 1370))
        return reshape_output_1


class Reshape1297(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1370, 1, 1280))
        return reshape_output_1


class Reshape1298(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 1280))
        return reshape_output_1


class Reshape1299(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2, 20, 64))
        return reshape_output_1


class Reshape1300(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2, 1280))
        return reshape_output_1


class Reshape1301(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(20, 2, 64))
        return reshape_output_1


class Reshape1302(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 20, 2, 2))
        return reshape_output_1


class Reshape1303(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(20, 2, 2))
        return reshape_output_1


class Reshape1304(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 20, 2, 64))
        return reshape_output_1


class Reshape1305(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 3000, 1))
        return reshape_output_1


class Reshape1306(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1280, 128, 3, 1))
        return reshape_output_1


class Reshape1307(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1280, 3000))
        return reshape_output_1


class Reshape1308(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1280, 3000, 1))
        return reshape_output_1


class Reshape1309(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1280, 1280, 3, 1))
        return reshape_output_1


class Reshape1310(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1280, 1500))
        return reshape_output_1


class Reshape1311(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1500, 1280))
        return reshape_output_1


class Reshape1312(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1500, 20, 64))
        return reshape_output_1


class Reshape1313(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1500, 1280))
        return reshape_output_1


class Reshape1314(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(20, 1500, 64))
        return reshape_output_1


class Reshape1315(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 20, 1500, 1500))
        return reshape_output_1


class Reshape1316(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(20, 1500, 1500))
        return reshape_output_1


class Reshape1317(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 20, 1500, 64))
        return reshape_output_1


class Reshape1318(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 20, 2, 1500))
        return reshape_output_1


class Reshape1319(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(20, 2, 1500))
        return reshape_output_1


class Reshape1320(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(768, 80, 3, 1))
        return reshape_output_1


class Reshape1321(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 768, 3000))
        return reshape_output_1


class Reshape1322(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 768, 3000, 1))
        return reshape_output_1


class Reshape1323(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(768, 768, 3, 1))
        return reshape_output_1


class Reshape1324(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 768, 1500))
        return reshape_output_1


class Reshape1325(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1500, 768))
        return reshape_output_1


class Reshape1326(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1500, 12, 64))
        return reshape_output_1


class Reshape1327(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1500, 768))
        return reshape_output_1


class Reshape1328(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 1500, 64))
        return reshape_output_1


class Reshape1329(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 1500, 1500))
        return reshape_output_1


class Reshape1330(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 1500, 1500))
        return reshape_output_1


class Reshape1331(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 1500, 64))
        return reshape_output_1


class Reshape1332(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 1, 1500))
        return reshape_output_1


class Reshape1333(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 1, 1500))
        return reshape_output_1


class Reshape1334(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 255, 160, 160))
        return reshape_output_1


class Reshape1335(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 255, 25600))
        return reshape_output_1


class Reshape1336(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 3, 85, 25600))
        return reshape_output_1


class Reshape1337(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 76800, 85))
        return reshape_output_1


class Reshape1338(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 17, 4480))
        return reshape_output_1


class Reshape1339(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 17, 1120))
        return reshape_output_1


class Reshape1340(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 17, 280))
        return reshape_output_1


class Reshape1341(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 192, 1344, 1))
        return reshape_output_1


class Reshape1342(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 192))
        return reshape_output_1


class Reshape1343(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 192, 50, 83))
        return reshape_output_1


class Reshape1344(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1445, 192))
        return reshape_output_1


class Reshape1345(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1445, 3, 64))
        return reshape_output_1


class Reshape1346(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1445, 192))
        return reshape_output_1


class Reshape1347(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(3, 1445, 64))
        return reshape_output_1


class Reshape1348(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 3, 1445, 1445))
        return reshape_output_1


class Reshape1349(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(3, 1445, 1445))
        return reshape_output_1


class Reshape1350(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 3, 1445, 64))
        return reshape_output_1


class Reshape1351(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(100, 192))
        return reshape_output_1


class Reshape1352(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 100, 192))
        return reshape_output_1


class Reshape1353(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1408))
        return reshape_output_1


class Reshape1354(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1792))
        return reshape_output_1


class Reshape1355(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 120, 12, 1))
        return reshape_output_1


class Reshape1356(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 3, 8, 15))
        return reshape_output_1


class Reshape1357(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 12, 15))
        return reshape_output_1


class Reshape1358(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 15, 12))
        return reshape_output_1


class Reshape1359(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 12, 12))
        return reshape_output_1


class Reshape1360(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 12, 12))
        return reshape_output_1


class Reshape1361(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 12, 15))
        return reshape_output_1


class Reshape1362(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 120))
        return reshape_output_1


class Reshape1363(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 120))
        return reshape_output_1


class Reshape1364(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 12, 120))
        return reshape_output_1


class Reshape1365(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(522, 2048))
        return reshape_output_1


class Reshape1366(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 522, 8, 256))
        return reshape_output_1


class Reshape1367(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 522, 2048))
        return reshape_output_1


class Reshape1368(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 522, 256))
        return reshape_output_1


class Reshape1369(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 522, 4, 256))
        return reshape_output_1


class Reshape1370(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 522, 522))
        return reshape_output_1


class Reshape1371(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 522, 522))
        return reshape_output_1


class Reshape1372(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 522, 256))
        return reshape_output_1


class Reshape1373(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 522, 8192))
        return reshape_output_1


class Reshape1374(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(207, 2304))
        return reshape_output_1


class Reshape1375(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 207, 8, 256))
        return reshape_output_1


class Reshape1376(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 207, 256))
        return reshape_output_1


class Reshape1377(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 207, 4, 256))
        return reshape_output_1


class Reshape1378(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 207, 256))
        return reshape_output_1


class Reshape1379(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 207, 207))
        return reshape_output_1


class Reshape1380(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 207, 207))
        return reshape_output_1


class Reshape1381(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 256, 207))
        return reshape_output_1


class Reshape1382(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(207, 2048))
        return reshape_output_1


class Reshape1383(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 207, 2304))
        return reshape_output_1


class Reshape1384(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 207, 9216))
        return reshape_output_1


class Reshape1385(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 512, 196, 1))
        return reshape_output_1


class Reshape1386(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1008, 1, 1))
        return reshape_output_1


class Reshape1387(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 61, 2048))
        return reshape_output_1


class Reshape1388(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 2048))
        return reshape_output_1


class Reshape1389(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(197, 1, 2304))
        return reshape_output_1


class Reshape1390(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(197, 1, 3, 768))
        return reshape_output_1


class Reshape1391(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(197, 12, 64))
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
        {
            "model_names": [
                "pd_albert_chinese_tiny_mlm_padlenlp",
                "pd_bert_chinese_roberta_base_seq_cls_padlenlp",
                "pd_roberta_rbt4_ch_clm_padlenlp",
                "pd_bert_chinese_roberta_base_qa_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 12, 11, 11)"},
        },
    ),
    (
        Reshape6,
        [((1, 12, 11, 11), torch.float32)],
        {
            "model_names": [
                "pd_albert_chinese_tiny_mlm_padlenlp",
                "pd_bert_chinese_roberta_base_seq_cls_padlenlp",
                "pd_roberta_rbt4_ch_clm_padlenlp",
                "pd_bert_chinese_roberta_base_qa_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"shape": "(12, 11, 11)"},
        },
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
        [((1, 128, 768), torch.float32)],
        {
            "model_names": [
                "pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
                "pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder",
                "pt_albert_base_v2_mlm_hf",
                "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf",
                "pt_bert_bert_base_uncased_mlm_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder",
                "pt_albert_base_v1_token_cls_hf",
                "pt_albert_base_v2_token_cls_hf",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_albert_base_v1_mlm_hf",
                "pt_distilbert_distilbert_base_uncased_mlm_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf",
                "pt_roberta_xlm_roberta_base_mlm_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(128, 768)"},
        },
    ),
    (
        Reshape9,
        [((1, 128, 768), torch.float32)],
        {
            "model_names": [
                "pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
                "pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder",
                "pt_albert_base_v2_mlm_hf",
                "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf",
                "pt_bert_bert_base_uncased_mlm_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder",
                "pt_albert_base_v1_token_cls_hf",
                "pt_albert_base_v2_token_cls_hf",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_albert_base_v1_mlm_hf",
                "pt_distilbert_distilbert_base_uncased_mlm_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf",
                "pt_roberta_xlm_roberta_base_mlm_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 128, 12, 64)"},
        },
    ),
    (
        Reshape10,
        [((128, 768), torch.float32)],
        {
            "model_names": [
                "pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
                "pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder",
                "pt_albert_base_v2_mlm_hf",
                "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf",
                "pt_bert_bert_base_uncased_mlm_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder",
                "pt_albert_base_v1_token_cls_hf",
                "pt_albert_base_v2_token_cls_hf",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_albert_base_v1_mlm_hf",
                "pt_distilbert_distilbert_base_uncased_mlm_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf",
                "pt_roberta_xlm_roberta_base_mlm_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 128, 768)"},
        },
    ),
    (
        Reshape11,
        [((1, 12, 128, 64), torch.float32)],
        {
            "model_names": [
                "pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
                "pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder",
                "pt_albert_base_v2_mlm_hf",
                "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf",
                "pt_bert_bert_base_uncased_mlm_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder",
                "pt_albert_base_v1_token_cls_hf",
                "pt_albert_base_v2_token_cls_hf",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_albert_base_v1_mlm_hf",
                "pt_distilbert_distilbert_base_uncased_mlm_hf",
                "pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf",
                "pt_roberta_xlm_roberta_base_mlm_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(12, 128, 64)"},
        },
    ),
    (
        Reshape12,
        [((12, 128, 128), torch.float32)],
        {
            "model_names": [
                "pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
                "pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder",
                "pt_albert_base_v2_mlm_hf",
                "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf",
                "pt_bert_bert_base_uncased_mlm_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder",
                "pt_albert_base_v1_token_cls_hf",
                "pt_albert_base_v2_token_cls_hf",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_albert_base_v1_mlm_hf",
                "pt_distilbert_distilbert_base_uncased_mlm_hf",
                "pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf",
                "pt_roberta_xlm_roberta_base_mlm_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 12, 128, 128)"},
        },
    ),
    (
        Reshape13,
        [((1, 12, 128, 128), torch.float32)],
        {
            "model_names": [
                "pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
                "pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder",
                "pt_albert_base_v2_mlm_hf",
                "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf",
                "pt_bert_bert_base_uncased_mlm_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder",
                "pt_albert_base_v1_token_cls_hf",
                "pt_albert_base_v2_token_cls_hf",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_albert_base_v1_mlm_hf",
                "pt_distilbert_distilbert_base_uncased_mlm_hf",
                "pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf",
                "pt_roberta_xlm_roberta_base_mlm_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(12, 128, 128)"},
        },
    ),
    (
        Reshape14,
        [((12, 128, 64), torch.float32)],
        {
            "model_names": [
                "pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
                "pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder",
                "pt_albert_base_v2_mlm_hf",
                "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf",
                "pt_bert_bert_base_uncased_mlm_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder",
                "pt_albert_base_v1_token_cls_hf",
                "pt_albert_base_v2_token_cls_hf",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_albert_base_v1_mlm_hf",
                "pt_distilbert_distilbert_base_uncased_mlm_hf",
                "pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf",
                "pt_roberta_xlm_roberta_base_mlm_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 12, 128, 64)"},
        },
    ),
    (
        Reshape8,
        [((1, 128, 12, 64), torch.float32)],
        {
            "model_names": [
                "pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
                "pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder",
                "pt_albert_base_v2_mlm_hf",
                "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf",
                "pt_bert_bert_base_uncased_mlm_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder",
                "pt_albert_base_v1_token_cls_hf",
                "pt_albert_base_v2_token_cls_hf",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_albert_base_v1_mlm_hf",
                "pt_distilbert_distilbert_base_uncased_mlm_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf",
                "pt_roberta_xlm_roberta_base_mlm_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(128, 768)"},
        },
    ),
    (
        Reshape15,
        [((1, 1, 768), torch.float32)],
        {
            "model_names": [
                "pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf",
                "pt_t5_t5_base_text_gen_hf",
                "pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder",
                "pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder",
                "onnx_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf",
                "pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "onnx_vit_base_google_vit_base_patch16_224_img_cls_hf",
                "pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf",
                "pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf",
                "pt_whisper_openai_whisper_small_speech_recognition_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf",
                "pt_t5_google_flan_t5_base_text_gen_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 768)"},
        },
    ),
    (
        Reshape16,
        [((1, 1, 768), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 1, 12, 64)"},
        },
    ),
    (
        Reshape17,
        [((1, 32), torch.int64)],
        {
            "model_names": [
                "pt_opt_facebook_opt_125m_qa_hf",
                "pt_opt_facebook_opt_350m_qa_hf",
                "pt_opt_facebook_opt_125m_seq_cls_hf",
                "pt_opt_facebook_opt_350m_seq_cls_hf",
                "pt_opt_facebook_opt_1_3b_qa_hf",
                "pt_opt_facebook_opt_1_3b_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 32)"},
        },
    ),
    (
        Reshape18,
        [((1, 32, 768), torch.float32)],
        {
            "model_names": ["pt_opt_facebook_opt_125m_qa_hf", "pt_opt_facebook_opt_125m_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(32, 768)"},
        },
    ),
    (
        Reshape19,
        [((1, 32, 768), torch.float32)],
        {
            "model_names": ["pt_opt_facebook_opt_125m_qa_hf", "pt_opt_facebook_opt_125m_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 32, 12, 64)"},
        },
    ),
    (
        Reshape20,
        [((32, 768), torch.float32)],
        {
            "model_names": ["pt_opt_facebook_opt_125m_qa_hf", "pt_opt_facebook_opt_125m_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 32, 768)"},
        },
    ),
    (
        Reshape21,
        [((1, 12, 32, 64), torch.float32)],
        {
            "model_names": ["pt_opt_facebook_opt_125m_qa_hf", "pt_opt_facebook_opt_125m_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(12, 32, 64)"},
        },
    ),
    (
        Reshape22,
        [((1, 12, 32, 64), torch.float32)],
        {
            "model_names": ["pt_phi1_microsoft_phi_1_token_cls_hf", "pt_phi_1_5_microsoft_phi_1_5_token_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(12, 2048)"},
        },
    ),
    (
        Reshape23,
        [((12, 32, 32), torch.float32)],
        {
            "model_names": ["pt_opt_facebook_opt_125m_qa_hf", "pt_opt_facebook_opt_125m_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 12, 32, 32)"},
        },
    ),
    (
        Reshape24,
        [((1, 12, 32, 32), torch.float32)],
        {
            "model_names": ["pt_opt_facebook_opt_125m_qa_hf", "pt_opt_facebook_opt_125m_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(12, 32, 32)"},
        },
    ),
    (
        Reshape25,
        [((12, 32, 64), torch.float32)],
        {
            "model_names": ["pt_opt_facebook_opt_125m_qa_hf", "pt_opt_facebook_opt_125m_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 12, 32, 64)"},
        },
    ),
    (
        Reshape18,
        [((1, 32, 12, 64), torch.float32)],
        {
            "model_names": ["pt_opt_facebook_opt_125m_qa_hf", "pt_opt_facebook_opt_125m_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(32, 768)"},
        },
    ),
    (
        Reshape26,
        [((1, 32, 12, 64), torch.float32)],
        {
            "model_names": ["pt_phi1_microsoft_phi_1_token_cls_hf", "pt_phi_1_5_microsoft_phi_1_5_token_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(32, 12, 64)"},
        },
    ),
    (
        Reshape27,
        [((32, 1), torch.float32)],
        {
            "model_names": [
                "pt_opt_facebook_opt_125m_qa_hf",
                "pt_opt_facebook_opt_350m_qa_hf",
                "pt_opt_facebook_opt_1_3b_qa_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 32, 1)"},
        },
    ),
    (
        Reshape28,
        [((1, 7, 2048), torch.float32)],
        {
            "model_names": ["pt_phi1_microsoft_phi_1_clm_hf", "pt_phi_1_5_microsoft_phi_1_5_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(7, 2048)"},
        },
    ),
    (
        Reshape29,
        [((1, 7, 2048), torch.float32)],
        {
            "model_names": ["pt_phi1_microsoft_phi_1_clm_hf", "pt_phi_1_5_microsoft_phi_1_5_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 7, 32, 64)"},
        },
    ),
    (
        Reshape30,
        [((7, 2048), torch.float32)],
        {
            "model_names": ["pt_phi1_microsoft_phi_1_clm_hf", "pt_phi_1_5_microsoft_phi_1_5_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 7, 2048)"},
        },
    ),
    (
        Reshape31,
        [((1, 32, 7, 64), torch.float32)],
        {
            "model_names": ["pt_phi1_microsoft_phi_1_clm_hf", "pt_phi_1_5_microsoft_phi_1_5_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(32, 7, 64)"},
        },
    ),
    (
        Reshape32,
        [((32, 7, 7), torch.float32)],
        {
            "model_names": ["pt_phi1_microsoft_phi_1_clm_hf", "pt_phi_1_5_microsoft_phi_1_5_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 32, 7, 7)"},
        },
    ),
    (
        Reshape33,
        [((1, 32, 7, 7), torch.float32)],
        {
            "model_names": ["pt_phi1_microsoft_phi_1_clm_hf", "pt_phi_1_5_microsoft_phi_1_5_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(32, 7, 7)"},
        },
    ),
    (
        Reshape34,
        [((32, 7, 64), torch.float32)],
        {
            "model_names": ["pt_phi1_microsoft_phi_1_clm_hf", "pt_phi_1_5_microsoft_phi_1_5_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 32, 7, 64)"},
        },
    ),
    (
        Reshape28,
        [((1, 7, 32, 64), torch.float32)],
        {
            "model_names": ["pt_phi1_microsoft_phi_1_clm_hf", "pt_phi_1_5_microsoft_phi_1_5_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(7, 2048)"},
        },
    ),
    (
        Reshape35,
        [((7, 8192), torch.float32)],
        {
            "model_names": ["pt_phi1_microsoft_phi_1_clm_hf", "pt_phi_1_5_microsoft_phi_1_5_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 7, 8192)"},
        },
    ),
    (
        Reshape36,
        [((1, 29, 1024), torch.float32)],
        {"model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_chat_clm_hf"], "pcc": 0.99, "args": {"shape": "(29, 1024)"}},
    ),
    (
        Reshape37,
        [((1, 29, 1024), torch.float32)],
        {
            "model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_chat_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 29, 16, 64)"},
        },
    ),
    (
        Reshape38,
        [((29, 1024), torch.float32)],
        {"model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_chat_clm_hf"], "pcc": 0.99, "args": {"shape": "(1, 29, 1024)"}},
    ),
    (
        Reshape39,
        [((1, 16, 29, 64), torch.float32)],
        {"model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_chat_clm_hf"], "pcc": 0.99, "args": {"shape": "(16, 29, 64)"}},
    ),
    (
        Reshape40,
        [((16, 29, 29), torch.float32)],
        {
            "model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_chat_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 16, 29, 29)"},
        },
    ),
    (
        Reshape41,
        [((1, 16, 29, 29), torch.float32)],
        {"model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_chat_clm_hf"], "pcc": 0.99, "args": {"shape": "(16, 29, 29)"}},
    ),
    (
        Reshape42,
        [((16, 29, 64), torch.float32)],
        {
            "model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_chat_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 16, 29, 64)"},
        },
    ),
    (
        Reshape36,
        [((1, 29, 16, 64), torch.float32)],
        {"model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_chat_clm_hf"], "pcc": 0.99, "args": {"shape": "(29, 1024)"}},
    ),
    (
        Reshape43,
        [((29, 2816), torch.float32)],
        {"model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_chat_clm_hf"], "pcc": 0.99, "args": {"shape": "(1, 29, 2816)"}},
    ),
    (
        Reshape44,
        [((1, 672, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_x_800mf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 672, 1, 1)"},
        },
    ),
    (
        Reshape45,
        [((1, 2048, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_resnext_resnext50_32x4d_img_cls_torchhub",
                "pt_wideresnet_wide_resnet50_2_img_cls_torchvision",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_xception_xception71_tf_in1k_img_cls_timm",
                "pt_xception_xception_img_cls_timm",
                "pt_mobilenetv3_ssd_resnet101_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet152_img_cls_torchvision",
                "pt_resnet_resnet50_img_cls_torchvision",
                "pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub",
                "pt_hrnet_hrnet_w18_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_resnet_50_img_cls_hf",
                "pt_mobilenetv3_ssd_resnet50_img_cls_torchvision",
                "pt_regnet_regnet_x_16gf_img_cls_torchvision",
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception65_img_cls_timm",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_resnet_resnet101_img_cls_torchvision",
                "pt_resnet_resnet152_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 2048, 1, 1)"},
        },
    ),
    (
        Reshape46,
        [((1, 2048, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_resnext_resnext14_32x4d_img_cls_osmr",
                "pt_hrnet_hrnetv2_w40_pose_estimation_osmr",
                "pt_resnext_resnext50_32x4d_img_cls_osmr",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w48_pose_estimation_osmr",
                "pt_resnext_resnext101_64x4d_img_cls_osmr",
                "pt_resnext_resnext26_32x4d_img_cls_osmr",
                "pt_hrnet_hrnetv2_w30_pose_estimation_osmr",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 2048)"},
        },
    ),
    (
        Reshape47,
        [((1, 1), torch.int64)],
        {
            "model_names": [
                "pt_t5_t5_base_text_gen_hf",
                "pt_whisper_openai_whisper_base_speech_recognition_hf",
                "pt_t5_google_flan_t5_small_text_gen_hf",
                "pt_whisper_openai_whisper_tiny_speech_recognition_hf",
                "pt_t5_t5_small_text_gen_hf",
                "pt_whisper_openai_whisper_small_speech_recognition_hf",
                "pt_t5_google_flan_t5_base_text_gen_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 1)"},
        },
    ),
    (
        Reshape16,
        [((1, 768), torch.float32)],
        {
            "model_names": [
                "pt_t5_t5_base_text_gen_hf",
                "pt_whisper_openai_whisper_small_speech_recognition_hf",
                "pt_t5_google_flan_t5_base_text_gen_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 1, 12, 64)"},
        },
    ),
    (
        Reshape48,
        [((1, 768), torch.float32)],
        {
            "model_names": [
                "pt_t5_t5_base_text_gen_hf",
                "pt_whisper_openai_whisper_small_speech_recognition_hf",
                "pt_t5_google_flan_t5_base_text_gen_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 1, 768)"},
        },
    ),
    (
        Reshape49,
        [((1, 12, 1, 64), torch.float32)],
        {
            "model_names": [
                "pt_t5_t5_base_text_gen_hf",
                "pt_whisper_openai_whisper_small_speech_recognition_hf",
                "pt_t5_google_flan_t5_base_text_gen_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(12, 1, 64)"},
        },
    ),
    (
        Reshape50,
        [((12, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_t5_t5_base_text_gen_hf",
                "pt_whisper_openai_whisper_small_speech_recognition_hf",
                "pt_t5_google_flan_t5_base_text_gen_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 12, 1, 1)"},
        },
    ),
    (
        Reshape51,
        [((1, 12, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_t5_t5_base_text_gen_hf",
                "pt_whisper_openai_whisper_small_speech_recognition_hf",
                "pt_t5_google_flan_t5_base_text_gen_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(12, 1, 1)"},
        },
    ),
    (
        Reshape52,
        [((1, 12, 64, 1), torch.float32)],
        {
            "model_names": ["pt_t5_t5_base_text_gen_hf", "pt_t5_google_flan_t5_base_text_gen_hf"],
            "pcc": 0.99,
            "args": {"shape": "(12, 64, 1)"},
        },
    ),
    (
        Reshape53,
        [((12, 1, 64), torch.float32)],
        {
            "model_names": [
                "pt_t5_t5_base_text_gen_hf",
                "pt_whisper_openai_whisper_small_speech_recognition_hf",
                "pt_t5_google_flan_t5_base_text_gen_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 12, 1, 64)"},
        },
    ),
    (
        Reshape15,
        [((1, 1, 12, 64), torch.float32)],
        {
            "model_names": [
                "pt_t5_t5_base_text_gen_hf",
                "pt_whisper_openai_whisper_small_speech_recognition_hf",
                "pt_t5_google_flan_t5_base_text_gen_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 768)"},
        },
    ),
    (
        Reshape54,
        [((1, 61), torch.int64)],
        {
            "model_names": [
                "pt_t5_t5_base_text_gen_hf",
                "pt_t5_google_flan_t5_small_text_gen_hf",
                "pt_t5_t5_small_text_gen_hf",
                "pt_t5_google_flan_t5_base_text_gen_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 61)"},
        },
    ),
    (
        Reshape55,
        [((1, 61, 768), torch.float32)],
        {
            "model_names": ["pt_t5_t5_base_text_gen_hf", "pt_t5_google_flan_t5_base_text_gen_hf"],
            "pcc": 0.99,
            "args": {"shape": "(61, 768)"},
        },
    ),
    (
        Reshape56,
        [((61, 768), torch.float32)],
        {
            "model_names": ["pt_t5_t5_base_text_gen_hf", "pt_t5_google_flan_t5_base_text_gen_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 61, 12, 64)"},
        },
    ),
    (
        Reshape57,
        [((61, 768), torch.float32)],
        {
            "model_names": ["pt_t5_t5_base_text_gen_hf", "pt_t5_google_flan_t5_base_text_gen_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 61, 768)"},
        },
    ),
    (
        Reshape58,
        [((1, 12, 61, 64), torch.float32)],
        {
            "model_names": ["pt_t5_t5_base_text_gen_hf", "pt_t5_google_flan_t5_base_text_gen_hf"],
            "pcc": 0.99,
            "args": {"shape": "(12, 61, 64)"},
        },
    ),
    (
        Reshape59,
        [((12, 61, 61), torch.float32)],
        {
            "model_names": ["pt_t5_t5_base_text_gen_hf", "pt_t5_google_flan_t5_base_text_gen_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 12, 61, 61)"},
        },
    ),
    (
        Reshape60,
        [((1, 12, 61, 61), torch.float32)],
        {
            "model_names": ["pt_t5_t5_base_text_gen_hf", "pt_t5_google_flan_t5_base_text_gen_hf"],
            "pcc": 0.99,
            "args": {"shape": "(12, 61, 61)"},
        },
    ),
    (
        Reshape61,
        [((1, 12, 64, 61), torch.float32)],
        {
            "model_names": ["pt_t5_t5_base_text_gen_hf", "pt_t5_google_flan_t5_base_text_gen_hf"],
            "pcc": 0.99,
            "args": {"shape": "(12, 64, 61)"},
        },
    ),
    (
        Reshape62,
        [((12, 61, 64), torch.float32)],
        {
            "model_names": ["pt_t5_t5_base_text_gen_hf", "pt_t5_google_flan_t5_base_text_gen_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 12, 61, 64)"},
        },
    ),
    (
        Reshape55,
        [((1, 61, 12, 64), torch.float32)],
        {
            "model_names": ["pt_t5_t5_base_text_gen_hf", "pt_t5_google_flan_t5_base_text_gen_hf"],
            "pcc": 0.99,
            "args": {"shape": "(61, 768)"},
        },
    ),
    (
        Reshape63,
        [((12, 1, 61), torch.float32)],
        {
            "model_names": ["pt_t5_t5_base_text_gen_hf", "pt_t5_google_flan_t5_base_text_gen_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 12, 1, 61)"},
        },
    ),
    (
        Reshape64,
        [((1, 12, 1, 61), torch.float32)],
        {
            "model_names": ["pt_t5_t5_base_text_gen_hf", "pt_t5_google_flan_t5_base_text_gen_hf"],
            "pcc": 0.99,
            "args": {"shape": "(12, 1, 61)"},
        },
    ),
    (
        Reshape65,
        [((1, 512, 7, 7), torch.bfloat16)],
        {
            "model_names": [
                "pt_vgg_vgg11_bn_img_cls_torchvision",
                "pt_vgg_vgg11_img_cls_torchvision",
                "pt_vgg_vgg19_img_cls_torchvision",
                "pt_vgg_vgg16_bn_img_cls_torchvision",
                "pt_vgg_vgg19_bn_obj_det_torchhub",
                "pt_vgg_vgg16_img_cls_torchvision",
                "pt_vgg_vgg13_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 25088, 1, 1)"},
        },
    ),
    (
        Reshape66,
        [((1, 512, 7, 7), torch.bfloat16)],
        {
            "model_names": [
                "pt_vgg_vgg16_obj_det_osmr",
                "pt_vgg_vgg19_obj_det_osmr",
                "pt_vgg_bn_vgg19_obj_det_osmr",
                "pt_vgg_vgg13_obj_det_osmr",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 25088)"},
        },
    ),
    (
        Reshape67,
        [((1, 1, 512), torch.float32)],
        {
            "model_names": [
                "pt_whisper_openai_whisper_base_speech_recognition_hf",
                "pt_t5_google_flan_t5_small_text_gen_hf",
                "pt_t5_t5_small_text_gen_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 512)"},
        },
    ),
    (
        Reshape68,
        [((1, 1, 512), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_base_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 1, 8, 64)"},
        },
    ),
    (
        Reshape69,
        [((1, 512), torch.float32)],
        {
            "model_names": [
                "pt_whisper_openai_whisper_base_speech_recognition_hf",
                "pt_t5_google_flan_t5_small_text_gen_hf",
                "pt_t5_t5_small_text_gen_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 1, 512)"},
        },
    ),
    (
        Reshape68,
        [((1, 512), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_base_speech_recognition_hf", "pt_t5_t5_small_text_gen_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 1, 8, 64)"},
        },
    ),
    (
        Reshape70,
        [((1, 8, 1, 64), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_base_speech_recognition_hf", "pt_t5_t5_small_text_gen_hf"],
            "pcc": 0.99,
            "args": {"shape": "(8, 1, 64)"},
        },
    ),
    (
        Reshape71,
        [((8, 1, 1), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_base_speech_recognition_hf", "pt_t5_t5_small_text_gen_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 8, 1, 1)"},
        },
    ),
    (
        Reshape72,
        [((1, 8, 1, 1), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_base_speech_recognition_hf", "pt_t5_t5_small_text_gen_hf"],
            "pcc": 0.99,
            "args": {"shape": "(8, 1, 1)"},
        },
    ),
    (
        Reshape73,
        [((8, 1, 64), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_base_speech_recognition_hf", "pt_t5_t5_small_text_gen_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 8, 1, 64)"},
        },
    ),
    (
        Reshape67,
        [((1, 1, 8, 64), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_base_speech_recognition_hf", "pt_t5_t5_small_text_gen_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 512)"},
        },
    ),
    (
        Reshape74,
        [((1, 80, 3000), torch.float32)],
        {
            "model_names": [
                "pt_whisper_openai_whisper_base_speech_recognition_hf",
                "pt_whisper_openai_whisper_tiny_speech_recognition_hf",
                "pt_whisper_openai_whisper_small_speech_recognition_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 80, 3000, 1)"},
        },
    ),
    (
        Reshape75,
        [((512, 80, 3), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_base_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(512, 80, 3, 1)"},
        },
    ),
    (
        Reshape76,
        [((1, 512, 3000, 1), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_base_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 512, 3000)"},
        },
    ),
    (
        Reshape77,
        [((1, 512, 3000), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_base_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 512, 3000, 1)"},
        },
    ),
    (
        Reshape78,
        [((512, 512, 3), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_base_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(512, 512, 3, 1)"},
        },
    ),
    (
        Reshape79,
        [((1, 512, 1500, 1), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_base_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 512, 1500)"},
        },
    ),
    (
        Reshape80,
        [((1, 1500, 512), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_base_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1500, 512)"},
        },
    ),
    (
        Reshape81,
        [((1, 1500, 512), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_base_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 1500, 8, 64)"},
        },
    ),
    (
        Reshape82,
        [((1500, 512), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_base_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 1500, 512)"},
        },
    ),
    (
        Reshape81,
        [((1500, 512), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_base_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 1500, 8, 64)"},
        },
    ),
    (
        Reshape83,
        [((1, 8, 1500, 64), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_base_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(8, 1500, 64)"},
        },
    ),
    (
        Reshape84,
        [((8, 1500, 1500), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_base_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 8, 1500, 1500)"},
        },
    ),
    (
        Reshape85,
        [((1, 8, 1500, 1500), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_base_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(8, 1500, 1500)"},
        },
    ),
    (
        Reshape86,
        [((8, 1500, 64), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_base_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 8, 1500, 64)"},
        },
    ),
    (
        Reshape80,
        [((1, 1500, 8, 64), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_base_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1500, 512)"},
        },
    ),
    (
        Reshape87,
        [((8, 1, 1500), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_base_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 8, 1, 1500)"},
        },
    ),
    (
        Reshape88,
        [((1, 8, 1, 1500), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_base_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(8, 1, 1500)"},
        },
    ),
    (
        Reshape89,
        [((1, 1000, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_dla_dla60_visual_bb_torchvision",
                "onnx_dla_dla102x_visual_bb_torchvision",
                "onnx_dla_dla60x_c_visual_bb_torchvision",
                "onnx_dla_dla34_visual_bb_torchvision",
                "onnx_dla_dla46x_c_visual_bb_torchvision",
                "onnx_dla_dla60x_visual_bb_torchvision",
                "onnx_dla_dla102_visual_bb_torchvision",
                "onnx_dla_dla169_visual_bb_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 1000)"},
        },
    ),
    (
        Reshape90,
        [((1, 64, 128, 128), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 64, 16384)"},
        },
    ),
    (
        Reshape91,
        [((1, 64, 128, 128), torch.float32)],
        {
            "model_names": [
                "pt_albert_xxlarge_v2_token_cls_hf",
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
                "pt_albert_xxlarge_v1_mlm_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(64, 128, 128)"},
        },
    ),
    (
        Reshape92,
        [((1, 16384, 64), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 16384, 1, 64)"},
        },
    ),
    (
        Reshape93,
        [((1, 16384, 64), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 128, 128, 64)"},
        },
    ),
    (
        Reshape94,
        [((1, 64, 16384), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 64, 128, 128)"},
        },
    ),
    (
        Reshape95,
        [((1, 64, 16, 16), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 64, 256)"},
        },
    ),
    (
        Reshape96,
        [((1, 256, 64), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(256, 64)"},
        },
    ),
    (
        Reshape97,
        [((1, 256, 64), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 256, 1, 64)"},
        },
    ),
    (
        Reshape98,
        [((1, 256, 64), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 256, 2, 32)"},
        },
    ),
    (
        Reshape99,
        [((256, 64), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 256, 64)"},
        },
    ),
    (
        Reshape95,
        [((1, 1, 64, 256), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 64, 256)"},
        },
    ),
    (
        Reshape100,
        [((1, 16384, 256), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 1, 16384, 256)"},
        },
    ),
    (
        Reshape101,
        [((1, 1, 16384, 256), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 16384, 256)"},
        },
    ),
    (
        Reshape102,
        [((1, 256, 16384), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 256, 128, 128)"},
        },
    ),
    (
        Reshape103,
        [((1, 256, 128, 128), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 256, 16384)"},
        },
    ),
    (
        Reshape104,
        [((1, 128, 64, 64), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 128, 4096)"},
        },
    ),
    (
        Reshape105,
        [((1, 128, 64, 64), torch.float32)],
        {
            "model_names": [
                "pt_albert_xxlarge_v2_token_cls_hf",
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
                "pt_albert_xxlarge_v1_mlm_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(128, 4096)"},
        },
    ),
    (
        Reshape106,
        [((1, 4096, 128), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 4096, 2, 64)"},
        },
    ),
    (
        Reshape107,
        [((1, 4096, 128), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 64, 64, 128)"},
        },
    ),
    (
        Reshape108,
        [((1, 2, 4096, 64), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(2, 4096, 64)"},
        },
    ),
    (
        Reshape109,
        [((1, 128, 4096), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_albert_xxlarge_v2_token_cls_hf",
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_albert_xxlarge_v1_mlm_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 128, 64, 64)"},
        },
    ),
    (
        Reshape105,
        [((1, 128, 4096), torch.float32)],
        {
            "model_names": [
                "pt_albert_xxlarge_v2_token_cls_hf",
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
                "pt_albert_xxlarge_v1_mlm_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(128, 4096)"},
        },
    ),
    (
        Reshape110,
        [((1, 128, 16, 16), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 128, 256)"},
        },
    ),
    (
        Reshape111,
        [((1, 256, 128), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(256, 128)"},
        },
    ),
    (
        Reshape112,
        [((1, 256, 128), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 256, 2, 64)"},
        },
    ),
    (
        Reshape113,
        [((256, 128), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 256, 128)"},
        },
    ),
    (
        Reshape114,
        [((1, 2, 64, 256), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(2, 64, 256)"},
        },
    ),
    (
        Reshape115,
        [((2, 4096, 256), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 2, 4096, 256)"},
        },
    ),
    (
        Reshape116,
        [((1, 2, 4096, 256), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(2, 4096, 256)"},
        },
    ),
    (
        Reshape117,
        [((1, 2, 256, 64), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(2, 256, 64)"},
        },
    ),
    (
        Reshape118,
        [((2, 4096, 64), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 2, 4096, 64)"},
        },
    ),
    (
        Reshape119,
        [((1, 4096, 2, 64), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(4096, 128)"},
        },
    ),
    (
        Reshape120,
        [((4096, 128), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 4096, 128)"},
        },
    ),
    (
        Reshape121,
        [((4096, 128), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github", "pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(64, 64, 128)"},
        },
    ),
    (
        Reshape122,
        [((4096, 128), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"shape": "(1, 1, 4096, 128)"},
        },
    ),
    (
        Reshape107,
        [((4096, 128), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(1, 64, 64, 128)"}},
    ),
    (
        Reshape123,
        [((1, 512, 4096), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 512, 64, 64)"},
        },
    ),
    (
        Reshape124,
        [((1, 512, 64, 64), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 512, 4096)"},
        },
    ),
    (
        Reshape125,
        [((1, 320, 32, 32), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 320, 1024)"},
        },
    ),
    (
        Reshape126,
        [((1, 1024, 320), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 1024, 5, 64)"},
        },
    ),
    (
        Reshape127,
        [((1, 1024, 320), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 32, 32, 320)"},
        },
    ),
    (
        Reshape128,
        [((1, 5, 1024, 64), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(5, 1024, 64)"},
        },
    ),
    (
        Reshape129,
        [((1, 320, 1024), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 320, 32, 32)"},
        },
    ),
    (
        Reshape130,
        [((1, 320, 16, 16), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 320, 256)"},
        },
    ),
    (
        Reshape131,
        [((1, 256, 320), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(256, 320)"},
        },
    ),
    (
        Reshape132,
        [((1, 256, 320), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 256, 5, 64)"},
        },
    ),
    (
        Reshape133,
        [((256, 320), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 256, 320)"},
        },
    ),
    (
        Reshape134,
        [((1, 5, 64, 256), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(5, 64, 256)"},
        },
    ),
    (
        Reshape135,
        [((5, 1024, 256), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 5, 1024, 256)"},
        },
    ),
    (
        Reshape136,
        [((1, 5, 1024, 256), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(5, 1024, 256)"},
        },
    ),
    (
        Reshape137,
        [((1, 5, 256, 64), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(5, 256, 64)"},
        },
    ),
    (
        Reshape138,
        [((5, 1024, 64), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 5, 1024, 64)"},
        },
    ),
    (
        Reshape139,
        [((1, 1024, 5, 64), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1024, 320)"},
        },
    ),
    (
        Reshape140,
        [((1024, 320), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 1024, 320)"},
        },
    ),
    (
        Reshape141,
        [((1, 1280, 1024), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 1280, 32, 32)"},
        },
    ),
    (
        Reshape142,
        [((1, 1280, 32, 32), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 1280, 1024)"},
        },
    ),
    (
        Reshape143,
        [((1, 512, 16, 16), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 512, 256)"},
        },
    ),
    (
        Reshape144,
        [((1, 256, 512), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(256, 512)"},
        },
    ),
    (
        Reshape145,
        [((1, 256, 512), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 256, 8, 64)"},
        },
    ),
    (
        Reshape146,
        [((1, 256, 512), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 16, 16, 512)"},
        },
    ),
    (
        Reshape147,
        [((256, 512), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 256, 512)"},
        },
    ),
    (
        Reshape145,
        [((256, 512), torch.float32)],
        {
            "model_names": ["pt_llama3_meta_llama_llama_3_2_1b_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 256, 8, 64)"},
        },
    ),
    (
        Reshape146,
        [((256, 512), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(1, 16, 16, 512)"}},
    ),
    (
        Reshape148,
        [((256, 512), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(4, 64, 512)"}},
    ),
    (
        Reshape149,
        [((1, 8, 256, 64), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(8, 256, 64)"},
        },
    ),
    (
        Reshape150,
        [((1, 8, 64, 256), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(8, 64, 256)"},
        },
    ),
    (
        Reshape151,
        [((8, 256, 256), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 8, 256, 256)"},
        },
    ),
    (
        Reshape152,
        [((1, 8, 256, 256), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(8, 256, 256)"},
        },
    ),
    (
        Reshape153,
        [((8, 256, 64), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 8, 256, 64)"},
        },
    ),
    (
        Reshape144,
        [((1, 256, 8, 64), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(256, 512)"},
        },
    ),
    (
        Reshape154,
        [((1, 2048, 256), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 2048, 16, 16)"},
        },
    ),
    (
        Reshape155,
        [((1, 2048, 16, 16), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 2048, 256)"},
        },
    ),
    (
        Reshape156,
        [((1, 768, 256), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 768, 16, 16)"},
        },
    ),
    (
        Reshape157,
        [((1, 768, 1024), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 768, 32, 32)"},
        },
    ),
    (
        Reshape158,
        [((1, 768, 4096), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 768, 64, 64)"},
        },
    ),
    (
        Reshape159,
        [((1, 768, 16384), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 768, 128, 128)"},
        },
    ),
    (
        Reshape160,
        [((1, 11, 768), torch.float32)],
        {
            "model_names": [
                "pd_bert_chinese_roberta_base_seq_cls_padlenlp",
                "pd_roberta_rbt4_ch_clm_padlenlp",
                "pd_bert_chinese_roberta_base_qa_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"shape": "(11, 768)"},
        },
    ),
    (
        Reshape161,
        [((1, 11, 768), torch.float32)],
        {
            "model_names": [
                "pd_bert_chinese_roberta_base_seq_cls_padlenlp",
                "pd_roberta_rbt4_ch_clm_padlenlp",
                "pd_bert_chinese_roberta_base_qa_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 11, 12, 64)"},
        },
    ),
    (
        Reshape162,
        [((11, 768), torch.float32)],
        {
            "model_names": [
                "pd_bert_chinese_roberta_base_seq_cls_padlenlp",
                "pd_roberta_rbt4_ch_clm_padlenlp",
                "pd_bert_chinese_roberta_base_qa_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 11, 768)"},
        },
    ),
    (
        Reshape163,
        [((1, 12, 11, 64), torch.float32)],
        {
            "model_names": [
                "pd_bert_chinese_roberta_base_seq_cls_padlenlp",
                "pd_roberta_rbt4_ch_clm_padlenlp",
                "pd_bert_chinese_roberta_base_qa_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"shape": "(12, 11, 64)"},
        },
    ),
    (
        Reshape164,
        [((1, 12, 64, 11), torch.float32)],
        {
            "model_names": [
                "pd_bert_chinese_roberta_base_seq_cls_padlenlp",
                "pd_roberta_rbt4_ch_clm_padlenlp",
                "pd_bert_chinese_roberta_base_qa_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"shape": "(12, 64, 11)"},
        },
    ),
    (
        Reshape165,
        [((12, 11, 64), torch.float32)],
        {
            "model_names": [
                "pd_bert_chinese_roberta_base_seq_cls_padlenlp",
                "pd_roberta_rbt4_ch_clm_padlenlp",
                "pd_bert_chinese_roberta_base_qa_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 12, 11, 64)"},
        },
    ),
    (
        Reshape160,
        [((1, 11, 12, 64), torch.float32)],
        {
            "model_names": [
                "pd_bert_chinese_roberta_base_seq_cls_padlenlp",
                "pd_roberta_rbt4_ch_clm_padlenlp",
                "pd_bert_chinese_roberta_base_qa_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"shape": "(11, 768)"},
        },
    ),
    (
        Reshape166,
        [((1, 9, 768), torch.float32)],
        {
            "model_names": [
                "pd_ernie_1_0_qa_padlenlp",
                "pd_ernie_1_0_seq_cls_padlenlp",
                "pd_bert_bert_base_uncased_mlm_padlenlp",
                "pd_bert_bert_base_uncased_qa_padlenlp",
                "pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf",
                "pd_bert_chinese_roberta_base_mlm_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"shape": "(9, 768)"},
        },
    ),
    (
        Reshape167,
        [((1, 9, 768), torch.float32)],
        {
            "model_names": [
                "pd_ernie_1_0_qa_padlenlp",
                "pd_ernie_1_0_seq_cls_padlenlp",
                "pd_bert_bert_base_uncased_mlm_padlenlp",
                "pd_bert_bert_base_uncased_qa_padlenlp",
                "pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf",
                "pd_bert_chinese_roberta_base_mlm_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 9, 12, 64)"},
        },
    ),
    (
        Reshape168,
        [((9, 768), torch.float32)],
        {
            "model_names": [
                "pd_ernie_1_0_qa_padlenlp",
                "pd_ernie_1_0_seq_cls_padlenlp",
                "pd_bert_bert_base_uncased_mlm_padlenlp",
                "pd_bert_bert_base_uncased_qa_padlenlp",
                "pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf",
                "pd_bert_chinese_roberta_base_mlm_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 9, 768)"},
        },
    ),
    (
        Reshape169,
        [((1, 12, 9, 64), torch.float32)],
        {
            "model_names": [
                "pd_ernie_1_0_qa_padlenlp",
                "pd_ernie_1_0_seq_cls_padlenlp",
                "pd_bert_bert_base_uncased_mlm_padlenlp",
                "pd_bert_bert_base_uncased_qa_padlenlp",
                "pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf",
                "pd_bert_chinese_roberta_base_mlm_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"shape": "(12, 9, 64)"},
        },
    ),
    (
        Reshape170,
        [((1, 12, 64, 9), torch.float32)],
        {
            "model_names": [
                "pd_ernie_1_0_qa_padlenlp",
                "pd_ernie_1_0_seq_cls_padlenlp",
                "pd_bert_bert_base_uncased_mlm_padlenlp",
                "pd_bert_bert_base_uncased_qa_padlenlp",
                "pd_bert_chinese_roberta_base_mlm_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"shape": "(12, 64, 9)"},
        },
    ),
    (
        Reshape171,
        [((12, 9, 9), torch.float32)],
        {
            "model_names": [
                "pd_ernie_1_0_qa_padlenlp",
                "pd_ernie_1_0_seq_cls_padlenlp",
                "pd_bert_bert_base_uncased_mlm_padlenlp",
                "pd_bert_bert_base_uncased_qa_padlenlp",
                "pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf",
                "pd_bert_chinese_roberta_base_mlm_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 12, 9, 9)"},
        },
    ),
    (
        Reshape172,
        [((1, 12, 9, 9), torch.float32)],
        {
            "model_names": [
                "pd_ernie_1_0_qa_padlenlp",
                "pd_ernie_1_0_seq_cls_padlenlp",
                "pd_bert_bert_base_uncased_mlm_padlenlp",
                "pd_bert_bert_base_uncased_qa_padlenlp",
                "pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf",
                "pd_bert_chinese_roberta_base_mlm_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"shape": "(12, 9, 9)"},
        },
    ),
    (
        Reshape173,
        [((12, 9, 64), torch.float32)],
        {
            "model_names": [
                "pd_ernie_1_0_qa_padlenlp",
                "pd_ernie_1_0_seq_cls_padlenlp",
                "pd_bert_bert_base_uncased_mlm_padlenlp",
                "pd_bert_bert_base_uncased_qa_padlenlp",
                "pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf",
                "pd_bert_chinese_roberta_base_mlm_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 12, 9, 64)"},
        },
    ),
    (
        Reshape166,
        [((1, 9, 12, 64), torch.float32)],
        {
            "model_names": [
                "pd_ernie_1_0_qa_padlenlp",
                "pd_ernie_1_0_seq_cls_padlenlp",
                "pd_bert_bert_base_uncased_mlm_padlenlp",
                "pd_bert_bert_base_uncased_qa_padlenlp",
                "pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf",
                "pd_bert_chinese_roberta_base_mlm_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"shape": "(9, 768)"},
        },
    ),
    (
        Reshape174,
        [((1, 512, 1, 1), torch.float32)],
        {
            "model_names": ["pd_resnet_18_img_cls_paddlemodels", "pd_resnet_34_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {"shape": "(1, 512, 1, 1)"},
        },
    ),
    (
        Reshape175,
        [((1, 128, 2048), torch.float32)],
        {
            "model_names": [
                "pt_albert_xlarge_v2_token_cls_hf",
                "pt_albert_xlarge_v1_mlm_hf",
                "pt_albert_xlarge_v2_mlm_hf",
                "pt_albert_xlarge_v1_token_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(128, 2048)"},
        },
    ),
    (
        Reshape176,
        [((1, 128, 2048), torch.float32)],
        {
            "model_names": [
                "pt_albert_xlarge_v2_token_cls_hf",
                "pt_albert_xlarge_v1_mlm_hf",
                "pt_albert_xlarge_v2_mlm_hf",
                "pt_albert_xlarge_v1_token_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 128, 16, 128)"},
        },
    ),
    (
        Reshape177,
        [((128, 2048), torch.float32)],
        {
            "model_names": [
                "pt_albert_xlarge_v2_token_cls_hf",
                "pt_albert_xlarge_v1_mlm_hf",
                "pt_albert_xlarge_v2_mlm_hf",
                "pt_albert_xlarge_v1_token_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 128, 2048)"},
        },
    ),
    (
        Reshape178,
        [((1, 16, 128, 128), torch.float32)],
        {
            "model_names": [
                "pt_albert_xlarge_v2_token_cls_hf",
                "pt_albert_xlarge_v1_mlm_hf",
                "pt_albert_xlarge_v2_mlm_hf",
                "pt_albert_large_v1_token_cls_hf",
                "pt_albert_large_v2_mlm_hf",
                "pt_albert_large_v2_token_cls_hf",
                "pt_albert_xlarge_v1_token_cls_hf",
                "pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(16, 128, 128)"},
        },
    ),
    (
        Reshape179,
        [((16, 128, 128), torch.float32)],
        {
            "model_names": [
                "pt_albert_xlarge_v2_token_cls_hf",
                "pt_albert_xlarge_v1_mlm_hf",
                "pt_albert_xlarge_v2_mlm_hf",
                "pt_albert_large_v1_token_cls_hf",
                "pt_albert_large_v2_mlm_hf",
                "pt_albert_large_v2_token_cls_hf",
                "pt_albert_xlarge_v1_token_cls_hf",
                "pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 16, 128, 128)"},
        },
    ),
    (
        Reshape175,
        [((1, 128, 16, 128), torch.float32)],
        {
            "model_names": [
                "pt_albert_xlarge_v2_token_cls_hf",
                "pt_albert_xlarge_v1_mlm_hf",
                "pt_albert_xlarge_v2_mlm_hf",
                "pt_albert_xlarge_v1_token_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(128, 2048)"},
        },
    ),
    (
        Reshape104,
        [((128, 4096), torch.float32)],
        {
            "model_names": [
                "pt_albert_xxlarge_v2_token_cls_hf",
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
                "pt_albert_xxlarge_v1_mlm_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 128, 4096)"},
        },
    ),
    (
        Reshape180,
        [((1, 64, 128, 64), torch.float32)],
        {
            "model_names": [
                "pt_albert_xxlarge_v2_token_cls_hf",
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
                "pt_albert_xxlarge_v1_mlm_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(64, 128, 64)"},
        },
    ),
    (
        Reshape94,
        [((64, 128, 128), torch.float32)],
        {
            "model_names": [
                "pt_albert_xxlarge_v2_token_cls_hf",
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
                "pt_albert_xxlarge_v1_mlm_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 64, 128, 128)"},
        },
    ),
    (
        Reshape181,
        [((64, 128, 64), torch.float32)],
        {
            "model_names": [
                "pt_albert_xxlarge_v2_token_cls_hf",
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
                "pt_albert_xxlarge_v1_mlm_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 64, 128, 64)"},
        },
    ),
    (
        Reshape182,
        [((1, 588, 2048), torch.float32)],
        {
            "model_names": ["pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf"],
            "pcc": 0.99,
            "args": {"shape": "(588, 2048)"},
        },
    ),
    (
        Reshape183,
        [((588, 2048), torch.float32)],
        {
            "model_names": ["pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 588, 16, 128)"},
        },
    ),
    (
        Reshape184,
        [((588, 2048), torch.float32)],
        {
            "model_names": ["pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 588, 2048)"},
        },
    ),
    (
        Reshape185,
        [((1, 16, 588, 128), torch.float32)],
        {
            "model_names": ["pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf"],
            "pcc": 0.99,
            "args": {"shape": "(16, 588, 128)"},
        },
    ),
    (
        Reshape186,
        [((16, 588, 588), torch.float32)],
        {
            "model_names": ["pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 16, 588, 588)"},
        },
    ),
    (
        Reshape187,
        [((1, 16, 588, 588), torch.float32)],
        {
            "model_names": ["pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf"],
            "pcc": 0.99,
            "args": {"shape": "(16, 588, 588)"},
        },
    ),
    (
        Reshape188,
        [((16, 588, 128), torch.float32)],
        {
            "model_names": ["pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 16, 588, 128)"},
        },
    ),
    (
        Reshape182,
        [((1, 588, 16, 128), torch.float32)],
        {
            "model_names": ["pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf"],
            "pcc": 0.99,
            "args": {"shape": "(588, 2048)"},
        },
    ),
    (
        Reshape189,
        [((588, 5504), torch.float32)],
        {
            "model_names": ["pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 588, 5504)"},
        },
    ),
    (
        Reshape89,
        [((1, 1000, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_dla_dla46_c_visual_bb_torchvision",
                "pt_dla_dla60x_visual_bb_torchvision",
                "pt_dla_dla46x_c_visual_bb_torchvision",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_dla_dla60x_c_visual_bb_torchvision",
                "pt_dla_dla169_visual_bb_torchvision",
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_dla_dla34_visual_bb_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 1000)"},
        },
    ),
    (
        Reshape190,
        [((1, 1000, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_dla_dla34_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 1000, 1, 1)"},
        },
    ),
    (
        Reshape191,
        [((40, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(40, 1, 3, 3)"},
        },
    ),
    (
        Reshape192,
        [((24, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b4_ra2_in1k_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(24, 1, 3, 3)"},
        },
    ),
    (
        Reshape193,
        [((144, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b4_ra2_in1k_img_cls_timm",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                "pt_efficientnet_efficientnet_b1_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b0_ra_in1k_img_cls_timm",
                "pt_efficientnet_hf_hub_timm_tf_efficientnet_b0_aa_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(144, 1, 3, 3)"},
        },
    ),
    (
        Reshape194,
        [((192, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b4_ra2_in1k_img_cls_timm",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(192, 1, 3, 3)"},
        },
    ),
    (
        Reshape195,
        [((192, 1, 5, 5), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b4_ra2_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(192, 1, 5, 5)"},
        },
    ),
    (
        Reshape196,
        [((288, 1, 5, 5), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(288, 1, 5, 5)"},
        },
    ),
    (
        Reshape197,
        [((288, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(288, 1, 3, 3)"},
        },
    ),
    (
        Reshape198,
        [((576, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(576, 1, 3, 3)"},
        },
    ),
    (
        Reshape199,
        [((576, 1, 5, 5), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(576, 1, 5, 5)"},
        },
    ),
    (
        Reshape200,
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
        Reshape201,
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
        Reshape202,
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
        Reshape203,
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
        Reshape204,
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
        Reshape205,
        [((256, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_hf_hub_timm_tf_efficientnetv2_s_in21k_img_cls_timm",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_xception_xception71_tf_in1k_img_cls_timm",
                "pt_xception_xception_img_cls_timm",
                "pt_efficientnet_hf_hub_timm_efficientnetv2_rw_s_ra2_in1k_img_cls_timm",
                "pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf",
                "pt_mobilenetv1_basic_img_cls_torchvision",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception65_img_cls_timm",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(256, 1, 3, 3)"},
        },
    ),
    (
        Reshape206,
        [((512, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_hf_hub_timm_tf_efficientnetv2_s_in21k_img_cls_timm",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_efficientnet_hf_hub_timm_efficientnetv2_rw_s_ra2_in1k_img_cls_timm",
                "pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf",
                "pt_mobilenetv1_basic_img_cls_torchvision",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(512, 1, 3, 3)"},
        },
    ),
    (
        Reshape207,
        [((768, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_hf_hub_timm_tf_efficientnetv2_s_in21k_img_cls_timm",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnetv2_rw_s_ra2_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(768, 1, 3, 3)"},
        },
    ),
    (
        Reshape208,
        [((960, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_hf_hub_timm_tf_efficientnetv2_s_in21k_img_cls_timm",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_efficientnet_hf_hub_timm_efficientnetv2_rw_s_ra2_in1k_img_cls_timm",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(960, 1, 3, 3)"},
        },
    ),
    (
        Reshape209,
        [((1536, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_hf_hub_timm_tf_efficientnetv2_s_in21k_img_cls_timm",
                "pt_xception_xception71_tf_in1k_img_cls_timm",
                "pt_xception_xception_img_cls_timm",
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception65_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1536, 1, 3, 3)"},
        },
    ),
    (
        Reshape210,
        [((1, 1280, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_hf_hub_timm_tf_efficientnetv2_s_in21k_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                "pt_efficientnet_efficientnet_b1_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b0_ra_in1k_img_cls_timm",
                "pt_efficientnet_hf_hub_timm_tf_efficientnet_b0_aa_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 1280, 1, 1)"},
        },
    ),
    (
        Reshape211,
        [((8, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(8, 1, 3, 3)"},
        },
    ),
    (
        Reshape212,
        [((48, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b4_ra2_in1k_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(48, 1, 3, 3)"},
        },
    ),
    (
        Reshape213,
        [((12, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(12, 1, 3, 3)"},
        },
    ),
    (
        Reshape214,
        [((16, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_efficientnet_efficientnet_b1_img_cls_torchvision",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_yolox_yolox_nano_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(16, 1, 3, 3)"},
        },
    ),
    (
        Reshape215,
        [((36, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(36, 1, 3, 3)"},
        },
    ),
    (
        Reshape216,
        [((72, 1, 5, 5), torch.bfloat16)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(72, 1, 5, 5)"},
        },
    ),
    (
        Reshape217,
        [((20, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(20, 1, 3, 3)"},
        },
    ),
    (
        Reshape218,
        [((24, 1, 5, 5), torch.bfloat16)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(24, 1, 5, 5)"},
        },
    ),
    (
        Reshape219,
        [((60, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(60, 1, 3, 3)"},
        },
    ),
    (
        Reshape220,
        [((120, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(120, 1, 3, 3)"},
        },
    ),
    (
        Reshape221,
        [((240, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_efficientnet_efficientnet_b1_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b0_ra_in1k_img_cls_timm",
                "pt_efficientnet_hf_hub_timm_tf_efficientnet_b0_aa_in1k_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(240, 1, 3, 3)"},
        },
    ),
    (
        Reshape222,
        [((100, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(100, 1, 3, 3)"},
        },
    ),
    (
        Reshape223,
        [((92, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(92, 1, 3, 3)"},
        },
    ),
    (
        Reshape224,
        [((56, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(56, 1, 3, 3)"},
        },
    ),
    (
        Reshape225,
        [((80, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(80, 1, 3, 3)"},
        },
    ),
    (
        Reshape226,
        [((336, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_efficientnet_hf_hub_timm_efficientnet_b4_ra2_in1k_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf",
                "pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(336, 1, 3, 3)"},
        },
    ),
    (
        Reshape227,
        [((672, 1, 5, 5), torch.bfloat16)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_hf_hub_timm_efficientnet_b4_ra2_in1k_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_efficientnet_efficientnet_b1_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b0_ra_in1k_img_cls_timm",
                "pt_efficientnet_hf_hub_timm_tf_efficientnet_b0_aa_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(672, 1, 5, 5)"},
        },
    ),
    (
        Reshape228,
        [((112, 1, 5, 5), torch.bfloat16)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(112, 1, 5, 5)"},
        },
    ),
    (
        Reshape229,
        [((480, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_efficientnet_efficientnet_b1_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b0_ra_in1k_img_cls_timm",
                "pt_efficientnet_hf_hub_timm_tf_efficientnet_b0_aa_in1k_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(480, 1, 3, 3)"},
        },
    ),
    (
        Reshape230,
        [((1, 256), torch.int64)],
        {
            "model_names": [
                "pt_gpt_gpt2_text_gen_hf",
                "pt_xglm_facebook_xglm_564m_clm_hf",
                "pt_bart_facebook_bart_large_mnli_seq_cls_hf",
                "pt_opt_facebook_opt_1_3b_clm_hf",
                "pt_xglm_facebook_xglm_1_7b_clm_hf",
                "pt_opt_facebook_opt_125m_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 256)"},
        },
    ),
    (
        Reshape231,
        [((1, 256, 768), torch.float32)],
        {
            "model_names": [
                "pt_gpt_gpt2_text_gen_hf",
                "pt_gptneo_eleutherai_gpt_neo_125m_clm_hf",
                "pt_opt_facebook_opt_125m_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(256, 768)"},
        },
    ),
    (
        Reshape232,
        [((1, 256, 768), torch.float32)],
        {
            "model_names": ["pt_gpt_gpt2_text_gen_hf", "pt_opt_facebook_opt_125m_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 256, 12, 64)"},
        },
    ),
    (
        Reshape233,
        [((256, 768), torch.float32)],
        {
            "model_names": [
                "pt_gpt_gpt2_text_gen_hf",
                "pt_gptneo_eleutherai_gpt_neo_125m_clm_hf",
                "pt_opt_facebook_opt_125m_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 256, 768)"},
        },
    ),
    (
        Reshape232,
        [((256, 768), torch.float32)],
        {
            "model_names": ["pt_gptneo_eleutherai_gpt_neo_125m_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 256, 12, 64)"},
        },
    ),
    (
        Reshape234,
        [((1, 12, 256, 64), torch.float32)],
        {
            "model_names": [
                "pt_gpt_gpt2_text_gen_hf",
                "pt_gptneo_eleutherai_gpt_neo_125m_clm_hf",
                "pt_opt_facebook_opt_125m_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(12, 256, 64)"},
        },
    ),
    (
        Reshape235,
        [((12, 256, 256), torch.float32)],
        {
            "model_names": [
                "pt_gpt_gpt2_text_gen_hf",
                "pt_gptneo_eleutherai_gpt_neo_125m_clm_hf",
                "pt_opt_facebook_opt_125m_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 12, 256, 256)"},
        },
    ),
    (
        Reshape230,
        [((1, 256), torch.float32)],
        {"model_names": ["pt_gpt_gpt2_text_gen_hf"], "pcc": 0.99, "args": {"shape": "(1, 256)"}},
    ),
    (
        Reshape236,
        [((1, 256), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"shape": "(1, 1, 256)"},
        },
    ),
    (
        Reshape237,
        [((1, 12, 256, 256), torch.float32)],
        {
            "model_names": [
                "pt_gpt_gpt2_text_gen_hf",
                "pt_gptneo_eleutherai_gpt_neo_125m_clm_hf",
                "pt_opt_facebook_opt_125m_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(12, 256, 256)"},
        },
    ),
    (
        Reshape238,
        [((12, 256, 64), torch.float32)],
        {
            "model_names": [
                "pt_gpt_gpt2_text_gen_hf",
                "pt_gptneo_eleutherai_gpt_neo_125m_clm_hf",
                "pt_opt_facebook_opt_125m_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 12, 256, 64)"},
        },
    ),
    (
        Reshape231,
        [((1, 256, 12, 64), torch.float32)],
        {
            "model_names": [
                "pt_gpt_gpt2_text_gen_hf",
                "pt_gptneo_eleutherai_gpt_neo_125m_clm_hf",
                "pt_opt_facebook_opt_125m_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(256, 768)"},
        },
    ),
    (
        Reshape239,
        [((256, 3072), torch.float32)],
        {"model_names": ["pt_gpt_gpt2_text_gen_hf"], "pcc": 0.99, "args": {"shape": "(1, 256, 3072)"}},
    ),
    (
        Reshape240,
        [((1, 256, 3072), torch.float32)],
        {"model_names": ["pt_gpt_gpt2_text_gen_hf"], "pcc": 0.99, "args": {"shape": "(256, 3072)"}},
    ),
    (
        Reshape241,
        [((1, 7), torch.int64)],
        {
            "model_names": [
                "pt_gpt_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 7)"},
        },
    ),
    (
        Reshape242,
        [((1, 7, 768), torch.float32)],
        {
            "model_names": [
                "pt_gpt_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(7, 768)"},
        },
    ),
    (
        Reshape243,
        [((1, 7, 768), torch.float32)],
        {
            "model_names": [
                "pt_gpt_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 7, 12, 64)"},
        },
    ),
    (
        Reshape244,
        [((1, 7, 768), torch.float32)],
        {
            "model_names": ["pt_nanogpt_financialsupport_nanogpt_text_gen_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 7, 768)"},
        },
    ),
    (
        Reshape244,
        [((7, 768), torch.float32)],
        {
            "model_names": [
                "pt_gpt_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 7, 768)"},
        },
    ),
    (
        Reshape245,
        [((1, 12, 7, 64), torch.float32)],
        {
            "model_names": [
                "pt_gpt_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(12, 7, 64)"},
        },
    ),
    (
        Reshape246,
        [((12, 7, 7), torch.float32)],
        {
            "model_names": [
                "pt_gpt_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 12, 7, 7)"},
        },
    ),
    (
        Reshape247,
        [((1, 12, 7, 7), torch.float32)],
        {
            "model_names": [
                "pt_gpt_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(12, 7, 7)"},
        },
    ),
    (
        Reshape248,
        [((12, 7, 64), torch.float32)],
        {
            "model_names": [
                "pt_gpt_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 12, 7, 64)"},
        },
    ),
    (
        Reshape242,
        [((1, 7, 12, 64), torch.float32)],
        {
            "model_names": [
                "pt_gpt_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(7, 768)"},
        },
    ),
    (
        Reshape249,
        [((7, 3072), torch.float32)],
        {
            "model_names": [
                "pt_gpt_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 7, 3072)"},
        },
    ),
    (
        Reshape250,
        [((1, 7, 3072), torch.float32)],
        {
            "model_names": [
                "pt_gpt_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(7, 3072)"},
        },
    ),
    (
        Reshape251,
        [((7, 2), torch.float32)],
        {
            "model_names": ["pt_gpt_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(7, 2)"},
        },
    ),
    (
        Reshape252,
        [((1, 2), torch.float32)],
        {
            "model_names": [
                "pt_gpt_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
                "pt_opt_facebook_opt_125m_seq_cls_hf",
                "pt_opt_facebook_opt_350m_seq_cls_hf",
                "pt_opt_facebook_opt_1_3b_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 2)"},
        },
    ),
    (
        Reshape253,
        [((3072, 1, 4), torch.float32)],
        {"model_names": ["pt_mamba_state_spaces_mamba_790m_hf_clm_hf"], "pcc": 0.99, "args": {"shape": "(3072, 1, 4)"}},
    ),
    (
        Reshape254,
        [((1, 6, 3072), torch.float32)],
        {"model_names": ["pt_mamba_state_spaces_mamba_790m_hf_clm_hf"], "pcc": 0.99, "args": {"shape": "(6, 3072)"}},
    ),
    (
        Reshape255,
        [((6, 96), torch.float32)],
        {"model_names": ["pt_mamba_state_spaces_mamba_790m_hf_clm_hf"], "pcc": 0.99, "args": {"shape": "(1, 6, 96)"}},
    ),
    (
        Reshape256,
        [((1, 3072, 1, 16), torch.float32)],
        {
            "model_names": ["pt_mamba_state_spaces_mamba_790m_hf_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 3072, 16)"},
        },
    ),
    (
        Reshape257,
        [((6, 16), torch.float32)],
        {
            "model_names": ["pt_mamba_state_spaces_mamba_790m_hf_clm_hf", "pt_mamba_state_spaces_mamba_370m_hf_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 6, 16)"},
        },
    ),
    (
        Reshape258,
        [((1, 1, 16), torch.float32)],
        {
            "model_names": ["pt_mamba_state_spaces_mamba_790m_hf_clm_hf", "pt_mamba_state_spaces_mamba_370m_hf_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 16)"},
        },
    ),
    (
        Reshape259,
        [((1, 3072, 1), torch.float32)],
        {"model_names": ["pt_mamba_state_spaces_mamba_790m_hf_clm_hf"], "pcc": 0.99, "args": {"shape": "(1, 3072)"}},
    ),
    (
        Reshape260,
        [((32, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                "pt_mobilenetv1_basic_img_cls_torchvision",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                "pt_efficientnet_efficientnet_b1_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b0_ra_in1k_img_cls_timm",
                "pt_efficientnet_hf_hub_timm_tf_efficientnet_b0_aa_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_yolox_yolox_nano_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(32, 1, 3, 3)"},
        },
    ),
    (
        Reshape261,
        [((96, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b1_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b0_ra_in1k_img_cls_timm",
                "pt_efficientnet_hf_hub_timm_tf_efficientnet_b0_aa_in1k_img_cls_timm",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(96, 1, 3, 3)"},
        },
    ),
    (
        Reshape262,
        [((384, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(384, 1, 3, 3)"},
        },
    ),
    (
        Reshape263,
        [((72, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(72, 1, 3, 3)"},
        },
    ),
    (
        Reshape264,
        [((88, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(88, 1, 3, 3)"},
        },
    ),
    (
        Reshape265,
        [((96, 1, 5, 5), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(96, 1, 5, 5)"},
        },
    ),
    (
        Reshape266,
        [((240, 1, 5, 5), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_efficientnet_efficientnet_b1_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b0_ra_in1k_img_cls_timm",
                "pt_efficientnet_hf_hub_timm_tf_efficientnet_b0_aa_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(240, 1, 5, 5)"},
        },
    ),
    (
        Reshape267,
        [((120, 1, 5, 5), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(120, 1, 5, 5)"},
        },
    ),
    (
        Reshape268,
        [((144, 1, 5, 5), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_efficientnet_efficientnet_b1_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b0_ra_in1k_img_cls_timm",
                "pt_efficientnet_hf_hub_timm_tf_efficientnet_b0_aa_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(144, 1, 5, 5)"},
        },
    ),
    (
        Reshape269,
        [((1, 576, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 576, 1, 1)"},
        },
    ),
    (
        Reshape270,
        [((1, 32, 1024), torch.float32)],
        {
            "model_names": ["pt_opt_facebook_opt_350m_qa_hf", "pt_opt_facebook_opt_350m_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(32, 1024)"},
        },
    ),
    (
        Reshape271,
        [((1, 32, 1024), torch.float32)],
        {
            "model_names": ["pt_opt_facebook_opt_350m_qa_hf", "pt_opt_facebook_opt_350m_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 32, 16, 64)"},
        },
    ),
    (
        Reshape272,
        [((32, 1024), torch.float32)],
        {
            "model_names": ["pt_opt_facebook_opt_350m_qa_hf", "pt_opt_facebook_opt_350m_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 32, 1024)"},
        },
    ),
    (
        Reshape273,
        [((1, 16, 32, 64), torch.float32)],
        {
            "model_names": ["pt_opt_facebook_opt_350m_qa_hf", "pt_opt_facebook_opt_350m_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(16, 32, 64)"},
        },
    ),
    (
        Reshape274,
        [((16, 32, 32), torch.float32)],
        {
            "model_names": [
                "pt_opt_facebook_opt_350m_qa_hf",
                "pt_bloom_bigscience_bloom_1b1_clm_hf",
                "pt_opt_facebook_opt_350m_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 16, 32, 32)"},
        },
    ),
    (
        Reshape275,
        [((1, 16, 32, 32), torch.float32)],
        {
            "model_names": [
                "pt_opt_facebook_opt_350m_qa_hf",
                "pt_bloom_bigscience_bloom_1b1_clm_hf",
                "pt_opt_facebook_opt_350m_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(16, 32, 32)"},
        },
    ),
    (
        Reshape276,
        [((16, 32, 64), torch.float32)],
        {
            "model_names": ["pt_opt_facebook_opt_350m_qa_hf", "pt_opt_facebook_opt_350m_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 16, 32, 64)"},
        },
    ),
    (
        Reshape270,
        [((1, 32, 16, 64), torch.float32)],
        {
            "model_names": ["pt_opt_facebook_opt_350m_qa_hf", "pt_opt_facebook_opt_350m_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(32, 1024)"},
        },
    ),
    (
        Reshape277,
        [((32, 512), torch.float32)],
        {
            "model_names": ["pt_opt_facebook_opt_350m_qa_hf", "pt_opt_facebook_opt_350m_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(32, 512)"},
        },
    ),
    (
        Reshape278,
        [((1, 256, 2560), torch.float32)],
        {"model_names": ["pt_phi2_microsoft_phi_2_clm_hf"], "pcc": 0.99, "args": {"shape": "(256, 2560)"}},
    ),
    (
        Reshape279,
        [((1, 256, 2560), torch.float32)],
        {"model_names": ["pt_phi2_microsoft_phi_2_clm_hf"], "pcc": 0.99, "args": {"shape": "(1, 256, 32, 80)"}},
    ),
    (
        Reshape280,
        [((256, 2560), torch.float32)],
        {"model_names": ["pt_phi2_microsoft_phi_2_clm_hf"], "pcc": 0.99, "args": {"shape": "(1, 256, 2560)"}},
    ),
    (
        Reshape281,
        [((1, 32, 256, 80), torch.float32)],
        {"model_names": ["pt_phi2_microsoft_phi_2_clm_hf"], "pcc": 0.99, "args": {"shape": "(32, 256, 80)"}},
    ),
    (
        Reshape282,
        [((32, 256, 256), torch.float32)],
        {
            "model_names": [
                "pt_phi2_microsoft_phi_2_clm_hf",
                "pt_opt_facebook_opt_1_3b_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_clm_hf",
                "pt_phi1_microsoft_phi_1_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 32, 256, 256)"},
        },
    ),
    (
        Reshape283,
        [((1, 32, 256, 256), torch.float32)],
        {
            "model_names": [
                "pt_phi2_microsoft_phi_2_clm_hf",
                "pt_opt_facebook_opt_1_3b_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_clm_hf",
                "pt_phi1_microsoft_phi_1_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(32, 256, 256)"},
        },
    ),
    (
        Reshape284,
        [((1, 32, 256, 256), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"shape": "(1, 32, 65536)"},
        },
    ),
    (
        Reshape285,
        [((32, 256, 80), torch.float32)],
        {"model_names": ["pt_phi2_microsoft_phi_2_clm_hf"], "pcc": 0.99, "args": {"shape": "(1, 32, 256, 80)"}},
    ),
    (
        Reshape278,
        [((1, 256, 32, 80), torch.float32)],
        {"model_names": ["pt_phi2_microsoft_phi_2_clm_hf"], "pcc": 0.99, "args": {"shape": "(256, 2560)"}},
    ),
    (
        Reshape286,
        [((256, 10240), torch.float32)],
        {"model_names": ["pt_phi2_microsoft_phi_2_clm_hf"], "pcc": 0.99, "args": {"shape": "(1, 256, 10240)"}},
    ),
    (
        Reshape287,
        [((1, 39, 896), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf"], "pcc": 0.99, "args": {"shape": "(39, 896)"}},
    ),
    (
        Reshape288,
        [((1, 39, 896), torch.float32)],
        {
            "model_names": ["pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 39, 14, 64)"},
        },
    ),
    (
        Reshape289,
        [((39, 896), torch.float32)],
        {
            "model_names": ["pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 39, 896)"},
        },
    ),
    (
        Reshape290,
        [((1, 14, 39, 64), torch.float32)],
        {
            "model_names": ["pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(14, 39, 64)"},
        },
    ),
    (
        Reshape291,
        [((39, 128), torch.float32)],
        {
            "model_names": ["pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 39, 128)"},
        },
    ),
    (
        Reshape292,
        [((1, 39, 128), torch.float32)],
        {
            "model_names": ["pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 39, 2, 64)"},
        },
    ),
    (
        Reshape290,
        [((1, 2, 7, 39, 64), torch.float32)],
        {
            "model_names": ["pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(14, 39, 64)"},
        },
    ),
    (
        Reshape293,
        [((14, 39, 39), torch.float32)],
        {
            "model_names": ["pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 14, 39, 39)"},
        },
    ),
    (
        Reshape294,
        [((1, 14, 39, 39), torch.float32)],
        {
            "model_names": ["pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(14, 39, 39)"},
        },
    ),
    (
        Reshape295,
        [((14, 39, 64), torch.float32)],
        {
            "model_names": ["pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 14, 39, 64)"},
        },
    ),
    (
        Reshape287,
        [((1, 39, 14, 64), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf"], "pcc": 0.99, "args": {"shape": "(39, 896)"}},
    ),
    (
        Reshape296,
        [((39, 4864), torch.float32)],
        {
            "model_names": ["pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 39, 4864)"},
        },
    ),
    (
        Reshape297,
        [((1, 29, 1536), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99, "args": {"shape": "(29, 1536)"}},
    ),
    (
        Reshape298,
        [((1, 29, 1536), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99, "args": {"shape": "(1, 29, 12, 128)"}},
    ),
    (
        Reshape299,
        [((29, 1536), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99, "args": {"shape": "(1, 29, 1536)"}},
    ),
    (
        Reshape300,
        [((1, 12, 29, 128), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99, "args": {"shape": "(12, 29, 128)"}},
    ),
    (
        Reshape301,
        [((29, 256), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99, "args": {"shape": "(1, 29, 256)"}},
    ),
    (
        Reshape302,
        [((1, 29, 256), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99, "args": {"shape": "(1, 29, 2, 128)"}},
    ),
    (
        Reshape300,
        [((1, 2, 6, 29, 128), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99, "args": {"shape": "(12, 29, 128)"}},
    ),
    (
        Reshape303,
        [((12, 29, 29), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99, "args": {"shape": "(1, 12, 29, 29)"}},
    ),
    (
        Reshape304,
        [((1, 12, 29, 29), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99, "args": {"shape": "(12, 29, 29)"}},
    ),
    (
        Reshape305,
        [((12, 29, 128), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99, "args": {"shape": "(1, 12, 29, 128)"}},
    ),
    (
        Reshape297,
        [((1, 29, 12, 128), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99, "args": {"shape": "(29, 1536)"}},
    ),
    (
        Reshape306,
        [((29, 8960), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99, "args": {"shape": "(1, 29, 8960)"}},
    ),
    (
        Reshape307,
        [((1, 39, 1536), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf"], "pcc": 0.99, "args": {"shape": "(39, 1536)"}},
    ),
    (
        Reshape308,
        [((1, 39, 1536), torch.float32)],
        {
            "model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 39, 12, 128)"},
        },
    ),
    (
        Reshape309,
        [((39, 1536), torch.float32)],
        {
            "model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 39, 1536)"},
        },
    ),
    (
        Reshape310,
        [((1, 12, 39, 128), torch.float32)],
        {
            "model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(12, 39, 128)"},
        },
    ),
    (
        Reshape311,
        [((39, 256), torch.float32)],
        {
            "model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 39, 256)"},
        },
    ),
    (
        Reshape312,
        [((1, 39, 256), torch.float32)],
        {
            "model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 39, 2, 128)"},
        },
    ),
    (
        Reshape310,
        [((1, 2, 6, 39, 128), torch.float32)],
        {
            "model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(12, 39, 128)"},
        },
    ),
    (
        Reshape313,
        [((12, 39, 39), torch.float32)],
        {
            "model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 12, 39, 39)"},
        },
    ),
    (
        Reshape314,
        [((1, 12, 39, 39), torch.float32)],
        {
            "model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(12, 39, 39)"},
        },
    ),
    (
        Reshape315,
        [((12, 39, 128), torch.float32)],
        {
            "model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 12, 39, 128)"},
        },
    ),
    (
        Reshape307,
        [((1, 39, 12, 128), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf"], "pcc": 0.99, "args": {"shape": "(39, 1536)"}},
    ),
    (
        Reshape316,
        [((39, 8960), torch.float32)],
        {
            "model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 39, 8960)"},
        },
    ),
    (
        Reshape317,
        [((1, 912, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_x_1_6gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 912, 1, 1)"},
        },
    ),
    (
        Reshape318,
        [((1, 400, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_x_400mf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 400, 1, 1)"},
        },
    ),
    (
        Reshape319,
        [((1, 1920, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_regnet_regnet_x_8gf_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 1920, 1, 1)"},
        },
    ),
    (
        Reshape320,
        [((1, 64, 128, 128), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 64, 16384, 1)"},
        },
    ),
    (
        Reshape92,
        [((1, 16384, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 16384, 1, 64)"},
        },
    ),
    (
        Reshape93,
        [((1, 16384, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 128, 128, 64)"},
        },
    ),
    (
        Reshape94,
        [((1, 64, 16384), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 64, 128, 128)"},
        },
    ),
    (
        Reshape95,
        [((1, 64, 16, 16), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 64, 256)"},
        },
    ),
    (
        Reshape96,
        [((1, 256, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(256, 64)"},
        },
    ),
    (
        Reshape97,
        [((1, 256, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 256, 1, 64)"},
        },
    ),
    (
        Reshape98,
        [((1, 256, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 256, 2, 32)"},
        },
    ),
    (
        Reshape99,
        [((256, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 256, 64)"},
        },
    ),
    (
        Reshape100,
        [((1, 16384, 256), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 1, 16384, 256)"},
        },
    ),
    (
        Reshape101,
        [((1, 1, 16384, 256), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 16384, 256)"},
        },
    ),
    (
        Reshape95,
        [((1, 1, 64, 256), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 64, 256)"},
        },
    ),
    (
        Reshape102,
        [((1, 256, 16384), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 256, 128, 128)"},
        },
    ),
    (
        Reshape321,
        [((1, 256, 128, 128), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 256, 16384, 1)"},
        },
    ),
    (
        Reshape322,
        [((1, 128, 64, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 128, 4096, 1)"},
        },
    ),
    (
        Reshape106,
        [((1, 4096, 128), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 4096, 2, 64)"},
        },
    ),
    (
        Reshape107,
        [((1, 4096, 128), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 64, 64, 128)"},
        },
    ),
    (
        Reshape108,
        [((1, 2, 4096, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(2, 4096, 64)"},
        },
    ),
    (
        Reshape109,
        [((1, 128, 4096), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 128, 64, 64)"},
        },
    ),
    (
        Reshape110,
        [((1, 128, 16, 16), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 128, 256)"},
        },
    ),
    (
        Reshape111,
        [((1, 256, 128), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(256, 128)"},
        },
    ),
    (
        Reshape112,
        [((1, 256, 128), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 256, 2, 64)"},
        },
    ),
    (
        Reshape113,
        [((256, 128), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 256, 128)"},
        },
    ),
    (
        Reshape117,
        [((1, 2, 256, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(2, 256, 64)"},
        },
    ),
    (
        Reshape115,
        [((2, 4096, 256), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 2, 4096, 256)"},
        },
    ),
    (
        Reshape116,
        [((1, 2, 4096, 256), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(2, 4096, 256)"},
        },
    ),
    (
        Reshape114,
        [((1, 2, 64, 256), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(2, 64, 256)"},
        },
    ),
    (
        Reshape118,
        [((2, 4096, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 2, 4096, 64)"},
        },
    ),
    (
        Reshape119,
        [((1, 4096, 2, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(4096, 128)"},
        },
    ),
    (
        Reshape120,
        [((4096, 128), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 4096, 128)"},
        },
    ),
    (
        Reshape123,
        [((1, 512, 4096), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 512, 64, 64)"},
        },
    ),
    (
        Reshape323,
        [((1, 512, 64, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 512, 4096, 1)"},
        },
    ),
    (
        Reshape324,
        [((1, 320, 32, 32), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 320, 1024, 1)"},
        },
    ),
    (
        Reshape126,
        [((1, 1024, 320), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 1024, 5, 64)"},
        },
    ),
    (
        Reshape127,
        [((1, 1024, 320), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 32, 32, 320)"},
        },
    ),
    (
        Reshape128,
        [((1, 5, 1024, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(5, 1024, 64)"},
        },
    ),
    (
        Reshape129,
        [((1, 320, 1024), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 320, 32, 32)"},
        },
    ),
    (
        Reshape130,
        [((1, 320, 16, 16), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 320, 256)"},
        },
    ),
    (
        Reshape131,
        [((1, 256, 320), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(256, 320)"},
        },
    ),
    (
        Reshape132,
        [((1, 256, 320), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 256, 5, 64)"},
        },
    ),
    (
        Reshape133,
        [((256, 320), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 256, 320)"},
        },
    ),
    (
        Reshape137,
        [((1, 5, 256, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(5, 256, 64)"},
        },
    ),
    (
        Reshape135,
        [((5, 1024, 256), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 5, 1024, 256)"},
        },
    ),
    (
        Reshape136,
        [((1, 5, 1024, 256), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(5, 1024, 256)"},
        },
    ),
    (
        Reshape134,
        [((1, 5, 64, 256), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(5, 64, 256)"},
        },
    ),
    (
        Reshape138,
        [((5, 1024, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 5, 1024, 64)"},
        },
    ),
    (
        Reshape139,
        [((1, 1024, 5, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1024, 320)"},
        },
    ),
    (
        Reshape140,
        [((1024, 320), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 1024, 320)"},
        },
    ),
    (
        Reshape141,
        [((1, 1280, 1024), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 1280, 32, 32)"},
        },
    ),
    (
        Reshape325,
        [((1280, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1280, 1, 3, 3)"},
        },
    ),
    (
        Reshape326,
        [((1, 1280, 32, 32), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 1280, 1024, 1)"},
        },
    ),
    (
        Reshape327,
        [((1, 512, 16, 16), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 512, 256, 1)"},
        },
    ),
    (
        Reshape144,
        [((1, 256, 512), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(256, 512)"},
        },
    ),
    (
        Reshape145,
        [((1, 256, 512), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 256, 8, 64)"},
        },
    ),
    (
        Reshape147,
        [((1, 256, 512), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 256, 512)"},
        },
    ),
    (
        Reshape146,
        [((1, 256, 512), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 16, 16, 512)"},
        },
    ),
    (
        Reshape328,
        [((1, 256, 512), torch.bfloat16)],
        {
            "model_names": ["pt_mlp_mixer_base_img_cls_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 256, 512, 1)"},
        },
    ),
    (
        Reshape147,
        [((256, 512), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_mlp_mixer_base_img_cls_github",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 256, 512)"},
        },
    ),
    (
        Reshape149,
        [((1, 8, 256, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(8, 256, 64)"},
        },
    ),
    (
        Reshape151,
        [((8, 256, 256), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 8, 256, 256)"},
        },
    ),
    (
        Reshape152,
        [((1, 8, 256, 256), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(8, 256, 256)"},
        },
    ),
    (
        Reshape150,
        [((1, 8, 64, 256), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(8, 64, 256)"},
        },
    ),
    (
        Reshape153,
        [((8, 256, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 8, 256, 64)"},
        },
    ),
    (
        Reshape144,
        [((1, 256, 8, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(256, 512)"},
        },
    ),
    (
        Reshape154,
        [((1, 2048, 256), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 2048, 16, 16)"},
        },
    ),
    (
        Reshape329,
        [((2048, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(2048, 1, 3, 3)"},
        },
    ),
    (
        Reshape330,
        [((1, 2048, 16, 16), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 2048, 256, 1)"},
        },
    ),
    (
        Reshape331,
        [((1, 256, 256), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 256, 16, 16)"},
        },
    ),
    (
        Reshape332,
        [((1, 256, 256), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(256, 256)"},
        },
    ),
    (
        Reshape333,
        [((1, 256, 256), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 256, 8, 32)"},
        },
    ),
    (
        Reshape334,
        [((1, 256, 256), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 256, 256)"},
        },
    ),
    (
        Reshape335,
        [((1, 256, 256), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 16, 16, 256)"},
        },
    ),
    (
        Reshape336,
        [((1, 256, 1024), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 256, 32, 32)"},
        },
    ),
    (
        Reshape337,
        [((1, 256, 4096), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 256, 64, 64)"},
        },
    ),
    (
        Reshape338,
        [((1, 1024, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_l_32_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 1024, 49)"},
        },
    ),
    (
        Reshape339,
        [((50, 1, 1024), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_l_32_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(50, 1024)"},
        },
    ),
    (
        Reshape340,
        [((50, 3072), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_l_32_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(50, 1, 3072)"},
        },
    ),
    (
        Reshape341,
        [((50, 1, 3072), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_l_32_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(50, 1, 3, 1024)"},
        },
    ),
    (
        Reshape342,
        [((1, 50, 1, 1024), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_l_32_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(50, 16, 64)"},
        },
    ),
    (
        Reshape343,
        [((16, 50, 64), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_l_32_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(16, 50, 64)"},
        },
    ),
    (
        Reshape344,
        [((16, 50, 64), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_l_32_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 16, 50, 64)"},
        },
    ),
    (
        Reshape345,
        [((16, 50, 50), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_l_32_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 16, 50, 50)"},
        },
    ),
    (
        Reshape346,
        [((1, 16, 50, 50), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_l_32_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(16, 50, 50)"},
        },
    ),
    (
        Reshape339,
        [((50, 1, 16, 64), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_l_32_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(50, 1024)"},
        },
    ),
    (
        Reshape347,
        [((50, 1024), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_l_32_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(50, 1, 1024)"},
        },
    ),
    (
        Reshape348,
        [((1, 1, 1024), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_l_32_img_cls_torchvision", "pt_vit_google_vit_large_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 1024)"},
        },
    ),
    (
        Reshape349,
        [((1, 1, 1024), torch.bfloat16)],
        {
            "model_names": [
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 1, 1, 1024)"},
        },
    ),
    (
        Reshape350,
        [((1, 4, 56, 80), torch.bfloat16)],
        {
            "model_names": ["pt_yolo_v6_yolov6n_obj_det_torchhub"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 4, 4480)"},
        },
    ),
    (
        Reshape351,
        [((1, 4, 28, 40), torch.bfloat16)],
        {
            "model_names": ["pt_yolo_v6_yolov6n_obj_det_torchhub"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 4, 1120)"},
        },
    ),
    (
        Reshape352,
        [((1, 4, 14, 20), torch.bfloat16)],
        {
            "model_names": ["pt_yolo_v6_yolov6n_obj_det_torchhub"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 4, 280)"},
        },
    ),
    (
        Reshape353,
        [((1, 80, 56, 80), torch.bfloat16)],
        {
            "model_names": ["pt_yolo_v6_yolov6n_obj_det_torchhub", "pt_yolo_v6_yolov6l_obj_det_torchhub"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 80, 4480)"},
        },
    ),
    (
        Reshape354,
        [((1, 80, 28, 40), torch.bfloat16)],
        {
            "model_names": ["pt_yolo_v6_yolov6n_obj_det_torchhub", "pt_yolo_v6_yolov6l_obj_det_torchhub"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 80, 1120)"},
        },
    ),
    (
        Reshape355,
        [((1, 80, 14, 20), torch.bfloat16)],
        {
            "model_names": ["pt_yolo_v6_yolov6n_obj_det_torchhub", "pt_yolo_v6_yolov6l_obj_det_torchhub"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 80, 280)"},
        },
    ),
    (
        Reshape356,
        [((640, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolov10_yolov10x_obj_det_github",
                "pt_yolov10_yolov10n_obj_det_github",
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(640, 1, 3, 3)"},
        },
    ),
    (
        Reshape357,
        [((320, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": ["pt_yolov10_yolov10x_obj_det_github", "pt_yolov10_yolov10n_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(320, 1, 3, 3)"},
        },
    ),
    (
        Reshape358,
        [((1, 640, 20, 20), torch.bfloat16)],
        {
            "model_names": ["pt_yolov10_yolov10x_obj_det_github", "pt_yolov10_yolov10n_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 5, 128, 400)"},
        },
    ),
    (
        Reshape359,
        [((1, 5, 64, 400), torch.bfloat16)],
        {
            "model_names": ["pt_yolov10_yolov10x_obj_det_github", "pt_yolov10_yolov10n_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(5, 64, 400)"},
        },
    ),
    (
        Reshape360,
        [((1, 5, 64, 400), torch.bfloat16)],
        {
            "model_names": ["pt_yolov10_yolov10x_obj_det_github", "pt_yolov10_yolov10n_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 320, 20, 20)"},
        },
    ),
    (
        Reshape361,
        [((1, 5, 400, 32), torch.bfloat16)],
        {
            "model_names": ["pt_yolov10_yolov10x_obj_det_github", "pt_yolov10_yolov10n_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(5, 400, 32)"},
        },
    ),
    (
        Reshape362,
        [((5, 400, 400), torch.bfloat16)],
        {
            "model_names": ["pt_yolov10_yolov10x_obj_det_github", "pt_yolov10_yolov10n_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 5, 400, 400)"},
        },
    ),
    (
        Reshape363,
        [((1, 5, 400, 400), torch.bfloat16)],
        {
            "model_names": ["pt_yolov10_yolov10x_obj_det_github", "pt_yolov10_yolov10n_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(5, 400, 400)"},
        },
    ),
    (
        Reshape360,
        [((5, 64, 400), torch.bfloat16)],
        {
            "model_names": ["pt_yolov10_yolov10x_obj_det_github", "pt_yolov10_yolov10n_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 320, 20, 20)"},
        },
    ),
    (
        Reshape364,
        [((1, 144, 80, 80), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolov10_yolov10x_obj_det_github",
                "pt_yolo_world_default_obj_det_github",
                "pt_yolov10_yolov10n_obj_det_github",
                "pt_yolov8_yolov8x_obj_det_github",
                "pt_yolov8_yolov8n_obj_det_github",
                "pt_yolov9_default_obj_det_github",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 144, 6400)"},
        },
    ),
    (
        Reshape365,
        [((1, 144, 40, 40), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolov10_yolov10x_obj_det_github",
                "pt_yolo_world_default_obj_det_github",
                "pt_yolov10_yolov10n_obj_det_github",
                "pt_yolov8_yolov8x_obj_det_github",
                "pt_yolov8_yolov8n_obj_det_github",
                "pt_yolov9_default_obj_det_github",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 144, 1600)"},
        },
    ),
    (
        Reshape366,
        [((1, 144, 20, 20), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolov10_yolov10x_obj_det_github",
                "pt_yolo_world_default_obj_det_github",
                "pt_yolov10_yolov10n_obj_det_github",
                "pt_yolov8_yolov8x_obj_det_github",
                "pt_yolov8_yolov8n_obj_det_github",
                "pt_yolov9_default_obj_det_github",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 144, 400)"},
        },
    ),
    (
        Reshape367,
        [((1, 64, 8400), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolov10_yolov10x_obj_det_github",
                "pt_yolo_world_default_obj_det_github",
                "pt_yolov10_yolov10n_obj_det_github",
                "pt_yolov8_yolov8x_obj_det_github",
                "pt_yolov8_yolov8n_obj_det_github",
                "pt_yolov9_default_obj_det_github",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 4, 16, 8400)"},
        },
    ),
    (
        Reshape368,
        [((1, 1, 4, 8400), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolov10_yolov10x_obj_det_github",
                "pt_yolo_world_default_obj_det_github",
                "pt_yolov10_yolov10n_obj_det_github",
                "pt_yolov8_yolov8x_obj_det_github",
                "pt_yolov8_yolov8n_obj_det_github",
                "pt_yolov9_default_obj_det_github",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 4, 8400)"},
        },
    ),
    (
        Reshape369,
        [((1, 85, 80, 80), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolox_yolox_x_obj_det_torchhub",
                "pt_yolox_yolox_darknet_obj_det_torchhub",
                "pt_yolox_yolox_l_obj_det_torchhub",
                "pt_yolox_yolox_m_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 85, 6400, 1)"},
        },
    ),
    (
        Reshape370,
        [((1, 85, 40, 40), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolox_yolox_x_obj_det_torchhub",
                "pt_yolox_yolox_darknet_obj_det_torchhub",
                "pt_yolox_yolox_l_obj_det_torchhub",
                "pt_yolox_yolox_m_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 85, 1600, 1)"},
        },
    ),
    (
        Reshape371,
        [((1, 85, 20, 20), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolox_yolox_x_obj_det_torchhub",
                "pt_yolox_yolox_darknet_obj_det_torchhub",
                "pt_yolox_yolox_l_obj_det_torchhub",
                "pt_yolox_yolox_m_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 85, 400, 1)"},
        },
    ),
    (
        Reshape372,
        [((1, 1280, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_lite0_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_050_img_cls_timm",
                "onnx_efficientnet_efficientnet_b1_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 1280)"},
        },
    ),
    (
        Reshape210,
        [((1, 1280, 1, 1), torch.float32)],
        {
            "model_names": ["pd_mobilenetv2_basic_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {"shape": "(1, 1280, 1, 1)"},
        },
    ),
    (
        Reshape46,
        [((1, 2048, 1, 1), torch.float32)],
        {
            "model_names": ["onnx_resnet_50_img_cls_hf", "jax_resnet_50_img_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 2048)"},
        },
    ),
    (
        Reshape45,
        [((1, 2048, 1, 1), torch.float32)],
        {
            "model_names": ["pd_resnet_152_img_cls_paddlemodels", "pd_resnet_101_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {"shape": "(1, 2048, 1, 1)"},
        },
    ),
    (
        Reshape373,
        [((5, 256), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"shape": "(1, 1, 5, 256)"},
        },
    ),
    (
        Reshape374,
        [((1, 1, 5, 256), torch.float32)],
        {"model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"], "pcc": 0.99, "args": {"shape": "(5, 256)"}},
    ),
    (
        Reshape375,
        [((1, 1, 5, 256), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"shape": "(1, 5, 8, 32)"},
        },
    ),
    (
        Reshape376,
        [((1, 8, 5, 32), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"shape": "(8, 5, 32)"},
        },
    ),
    (
        Reshape377,
        [((1, 8, 32, 5), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"shape": "(8, 32, 5)"},
        },
    ),
    (
        Reshape378,
        [((8, 5, 5), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"shape": "(1, 8, 5, 5)"},
        },
    ),
    (
        Reshape379,
        [((1, 8, 5, 5), torch.float32)],
        {"model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"], "pcc": 0.99, "args": {"shape": "(8, 5, 5)"}},
    ),
    (
        Reshape380,
        [((8, 5, 32), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"shape": "(1, 8, 5, 32)"},
        },
    ),
    (
        Reshape374,
        [((1, 5, 8, 32), torch.float32)],
        {"model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"], "pcc": 0.99, "args": {"shape": "(5, 256)"}},
    ),
    (
        Reshape381,
        [((1, 1, 5, 128), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"shape": "(1, 5, 8, 16)"},
        },
    ),
    (
        Reshape382,
        [((1, 8, 5, 16), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"shape": "(8, 5, 16)"},
        },
    ),
    (
        Reshape383,
        [((1, 70, 70, 768), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"shape": "(1, 5, 14, 5, 14, 768)"},
        },
    ),
    (
        Reshape384,
        [((1, 5, 5, 14, 14, 768), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"shape": "(4900, 768)"},
        },
    ),
    (
        Reshape385,
        [((4900, 2304), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"shape": "(25, 14, 14, 2304)"},
        },
    ),
    (
        Reshape386,
        [((25, 14, 14, 2304), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"shape": "(25, 196, 3, 12, 64)"},
        },
    ),
    (
        Reshape387,
        [((3, 25, 12, 196, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"shape": "(3, 300, 196, 64)"},
        },
    ),
    (
        Reshape388,
        [((300, 196, 196), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"shape": "(300, 14, 14, 14, 14)"},
        },
    ),
    (
        Reshape389,
        [((300, 196, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"shape": "(300, 14, 14, 64)"},
        },
    ),
    (
        Reshape390,
        [((300, 196, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"shape": "(25, 12, 14, 14, 64)"},
        },
    ),
    (
        Reshape391,
        [((1, 64, 27), torch.float32)],
        {"model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"], "pcc": 0.99, "args": {"shape": "(64, 27)"}},
    ),
    (
        Reshape392,
        [((300, 14, 14, 14, 14), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"shape": "(300, 196, 196)"},
        },
    ),
    (
        Reshape384,
        [((25, 14, 14, 12, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"shape": "(4900, 768)"},
        },
    ),
    (
        Reshape393,
        [((4900, 768), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"shape": "(25, 14, 14, 768)"},
        },
    ),
    (
        Reshape394,
        [((25, 14, 14, 768), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"shape": "(1, 5, 5, 14, 14, 768)"},
        },
    ),
    (
        Reshape395,
        [((1, 5, 14, 5, 14, 768), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"shape": "(1, 70, 70, 768)"},
        },
    ),
    (
        Reshape396,
        [((1, 64, 64, 768), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"shape": "(4096, 768)"},
        },
    ),
    (
        Reshape397,
        [((4096, 3072), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"shape": "(1, 64, 64, 3072)"},
        },
    ),
    (
        Reshape398,
        [((1, 64, 64, 3072), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"shape": "(4096, 3072)"},
        },
    ),
    (
        Reshape399,
        [((4096, 768), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"shape": "(1, 64, 64, 768)"},
        },
    ),
    (
        Reshape400,
        [((4096, 2304), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"shape": "(1, 64, 64, 2304)"},
        },
    ),
    (
        Reshape401,
        [((1, 64, 64, 2304), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"shape": "(1, 4096, 3, 12, 64)"},
        },
    ),
    (
        Reshape402,
        [((3, 1, 12, 4096, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"shape": "(3, 12, 4096, 64)"},
        },
    ),
    (
        Reshape403,
        [((12, 4096, 4096), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"shape": "(12, 64, 64, 64, 64)"},
        },
    ),
    (
        Reshape404,
        [((12, 4096, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"shape": "(12, 64, 64, 64)"},
        },
    ),
    (
        Reshape405,
        [((12, 4096, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"shape": "(1, 12, 64, 64, 64)"},
        },
    ),
    (
        Reshape406,
        [((1, 64, 127), torch.float32)],
        {"model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"], "pcc": 0.99, "args": {"shape": "(64, 127)"}},
    ),
    (
        Reshape407,
        [((12, 64, 64, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"shape": "(12, 4096, 4096)"},
        },
    ),
    (
        Reshape396,
        [((1, 64, 64, 12, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"shape": "(4096, 768)"},
        },
    ),
    (
        Reshape408,
        [((1, 1, 256, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"shape": "(1, 256, 4096)"},
        },
    ),
    (
        Reshape409,
        [((1, 1, 4096, 256), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"shape": "(4096, 256)"},
        },
    ),
    (
        Reshape410,
        [((1, 1, 4096, 128), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"shape": "(1, 4096, 8, 16)"},
        },
    ),
    (
        Reshape411,
        [((1, 8, 16, 4096), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"shape": "(8, 16, 4096)"},
        },
    ),
    (
        Reshape412,
        [((8, 5, 4096), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"shape": "(1, 8, 5, 4096)"},
        },
    ),
    (
        Reshape413,
        [((1, 8, 5, 4096), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"shape": "(8, 5, 4096)"},
        },
    ),
    (
        Reshape414,
        [((1, 8, 4096, 16), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"shape": "(8, 4096, 16)"},
        },
    ),
    (
        Reshape415,
        [((8, 5, 16), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"shape": "(1, 8, 5, 16)"},
        },
    ),
    (
        Reshape416,
        [((1, 5, 8, 16), torch.float32)],
        {"model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"], "pcc": 0.99, "args": {"shape": "(5, 128)"}},
    ),
    (
        Reshape417,
        [((5, 128), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"shape": "(1, 1, 5, 128)"},
        },
    ),
    (
        Reshape418,
        [((1, 8, 16, 5), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"shape": "(8, 16, 5)"},
        },
    ),
    (
        Reshape419,
        [((8, 4096, 5), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"shape": "(1, 8, 4096, 5)"},
        },
    ),
    (
        Reshape420,
        [((1, 8, 4096, 5), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"shape": "(8, 4096, 5)"},
        },
    ),
    (
        Reshape421,
        [((8, 4096, 16), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"shape": "(1, 8, 4096, 16)"},
        },
    ),
    (
        Reshape119,
        [((1, 4096, 8, 16), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"shape": "(4096, 128)"},
        },
    ),
    (
        Reshape422,
        [((4096, 256), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"shape": "(1, 1, 4096, 256)"},
        },
    ),
    (
        Reshape230,
        [((1, 1, 1, 256), torch.float32)],
        {"model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"], "pcc": 0.99, "args": {"shape": "(1, 256)"}},
    ),
    (
        Reshape423,
        [((1, 1, 4, 32), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"shape": "(1, 4, 32)"},
        },
    ),
    (
        Reshape337,
        [((1, 1, 256, 4096), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"shape": "(1, 256, 64, 64)"},
        },
    ),
    (
        Reshape424,
        [((1, 4, 65536), torch.float32)],
        {
            "model_names": ["onnx_sam_facebook_sam_vit_base_img_seg_github"],
            "pcc": 0.99,
            "args": {"shape": "(1, 1, 4, 256, 256)"},
        },
    ),
    (
        Reshape364,
        [((1, 144, 80, 80), torch.float32)],
        {"model_names": ["onnx_yolov8_default_obj_det_github"], "pcc": 0.99, "args": {"shape": "(1, 144, 6400)"}},
    ),
    (
        Reshape365,
        [((1, 144, 40, 40), torch.float32)],
        {"model_names": ["onnx_yolov8_default_obj_det_github"], "pcc": 0.99, "args": {"shape": "(1, 144, 1600)"}},
    ),
    (
        Reshape366,
        [((1, 144, 20, 20), torch.float32)],
        {"model_names": ["onnx_yolov8_default_obj_det_github"], "pcc": 0.99, "args": {"shape": "(1, 144, 400)"}},
    ),
    (
        Reshape367,
        [((1, 64, 8400), torch.float32)],
        {"model_names": ["onnx_yolov8_default_obj_det_github"], "pcc": 0.99, "args": {"shape": "(1, 4, 16, 8400)"}},
    ),
    (
        Reshape368,
        [((1, 1, 4, 8400), torch.float32)],
        {"model_names": ["onnx_yolov8_default_obj_det_github"], "pcc": 0.99, "args": {"shape": "(1, 4, 8400)"}},
    ),
    (
        Reshape425,
        [((1, 8, 768), torch.float32)],
        {
            "model_names": [
                "pd_blip_text_salesforce_blip_image_captioning_base_text_enc_padlenlp",
                "pd_chineseclip_text_ofa_sys_chinese_clip_vit_base_patch16_text_enc_padlenlp",
                "pd_bert_bert_base_uncased_seq_cls_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"shape": "(8, 768)"},
        },
    ),
    (
        Reshape426,
        [((1, 8, 768), torch.float32)],
        {
            "model_names": [
                "pd_blip_text_salesforce_blip_image_captioning_base_text_enc_padlenlp",
                "pd_chineseclip_text_ofa_sys_chinese_clip_vit_base_patch16_text_enc_padlenlp",
                "pd_bert_bert_base_uncased_seq_cls_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 8, 12, 64)"},
        },
    ),
    (
        Reshape427,
        [((8, 768), torch.float32)],
        {
            "model_names": [
                "pd_blip_text_salesforce_blip_image_captioning_base_text_enc_padlenlp",
                "pd_chineseclip_text_ofa_sys_chinese_clip_vit_base_patch16_text_enc_padlenlp",
                "pd_bert_bert_base_uncased_seq_cls_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 8, 768)"},
        },
    ),
    (
        Reshape428,
        [((1, 12, 8, 64), torch.float32)],
        {
            "model_names": [
                "pd_blip_text_salesforce_blip_image_captioning_base_text_enc_padlenlp",
                "pd_chineseclip_text_ofa_sys_chinese_clip_vit_base_patch16_text_enc_padlenlp",
                "pd_bert_bert_base_uncased_seq_cls_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"shape": "(12, 8, 64)"},
        },
    ),
    (
        Reshape429,
        [((1, 12, 64, 8), torch.float32)],
        {
            "model_names": [
                "pd_blip_text_salesforce_blip_image_captioning_base_text_enc_padlenlp",
                "pd_chineseclip_text_ofa_sys_chinese_clip_vit_base_patch16_text_enc_padlenlp",
                "pd_bert_bert_base_uncased_seq_cls_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"shape": "(12, 64, 8)"},
        },
    ),
    (
        Reshape430,
        [((12, 8, 8), torch.float32)],
        {
            "model_names": [
                "pd_blip_text_salesforce_blip_image_captioning_base_text_enc_padlenlp",
                "pd_chineseclip_text_ofa_sys_chinese_clip_vit_base_patch16_text_enc_padlenlp",
                "pd_bert_bert_base_uncased_seq_cls_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 12, 8, 8)"},
        },
    ),
    (
        Reshape431,
        [((1, 12, 8, 8), torch.float32)],
        {
            "model_names": [
                "pd_blip_text_salesforce_blip_image_captioning_base_text_enc_padlenlp",
                "pd_chineseclip_text_ofa_sys_chinese_clip_vit_base_patch16_text_enc_padlenlp",
                "pd_bert_bert_base_uncased_seq_cls_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"shape": "(12, 8, 8)"},
        },
    ),
    (
        Reshape432,
        [((12, 8, 64), torch.float32)],
        {
            "model_names": [
                "pd_blip_text_salesforce_blip_image_captioning_base_text_enc_padlenlp",
                "pd_chineseclip_text_ofa_sys_chinese_clip_vit_base_patch16_text_enc_padlenlp",
                "pd_bert_bert_base_uncased_seq_cls_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 12, 8, 64)"},
        },
    ),
    (
        Reshape425,
        [((1, 8, 12, 64), torch.float32)],
        {
            "model_names": [
                "pd_blip_text_salesforce_blip_image_captioning_base_text_enc_padlenlp",
                "pd_chineseclip_text_ofa_sys_chinese_clip_vit_base_patch16_text_enc_padlenlp",
                "pd_bert_bert_base_uncased_seq_cls_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"shape": "(8, 768)"},
        },
    ),
    (
        Reshape433,
        [((1, 128, 3, 3), torch.float32)],
        {"model_names": ["pd_googlenet_base_img_cls_paddlemodels"], "pcc": 0.99, "args": {"shape": "(1, 1152, 1, 1)"}},
    ),
    (
        Reshape434,
        [((25, 1, 2, 48), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"shape": "(25, 1, 96)"},
        },
    ),
    (
        Reshape435,
        [((1, 768, 14, 14), torch.bfloat16)],
        {
            "model_names": [
                "pt_beit_microsoft_beit_base_patch16_224_img_cls_hf",
                "pt_mlp_mixer_mixer_b16_224_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_goog_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_miil_in21k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 768, 196, 1)"},
        },
    ),
    (
        Reshape436,
        [((1, 768, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_b_16_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 768, 196)"},
        },
    ),
    (
        Reshape437,
        [((1, 197, 768), torch.bfloat16)],
        {
            "model_names": ["pt_beit_microsoft_beit_base_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(197, 768)"},
        },
    ),
    (
        Reshape438,
        [((1, 197, 768), torch.bfloat16)],
        {
            "model_names": ["pt_beit_microsoft_beit_base_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 197, 12, 64)"},
        },
    ),
    (
        Reshape439,
        [((197, 768), torch.bfloat16)],
        {
            "model_names": ["pt_beit_microsoft_beit_base_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 197, 768)"},
        },
    ),
    (
        Reshape438,
        [((197, 768), torch.bfloat16)],
        {
            "model_names": ["pt_beit_microsoft_beit_base_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 197, 12, 64)"},
        },
    ),
    (
        Reshape440,
        [((197, 768), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_b_16_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(197, 1, 768)"},
        },
    ),
    (
        Reshape441,
        [((1, 12, 197, 64), torch.bfloat16)],
        {
            "model_names": ["pt_beit_microsoft_beit_base_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(12, 197, 64)"},
        },
    ),
    (
        Reshape442,
        [((12, 197, 197), torch.bfloat16)],
        {
            "model_names": [
                "pt_beit_microsoft_beit_base_patch16_224_img_cls_hf",
                "pt_vit_vit_b_16_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 12, 197, 197)"},
        },
    ),
    (
        Reshape443,
        [((729, 12), torch.bfloat16)],
        {
            "model_names": ["pt_beit_microsoft_beit_base_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 27, 27, 12)"},
        },
    ),
    (
        Reshape444,
        [((1, 27, 27, 12), torch.bfloat16)],
        {
            "model_names": ["pt_beit_microsoft_beit_base_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(729, 12)"},
        },
    ),
    (
        Reshape445,
        [((38809, 12), torch.bfloat16)],
        {
            "model_names": ["pt_beit_microsoft_beit_base_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(197, 197, 12)"},
        },
    ),
    (
        Reshape446,
        [((1, 12, 197, 197), torch.bfloat16)],
        {
            "model_names": [
                "pt_beit_microsoft_beit_base_patch16_224_img_cls_hf",
                "pt_vit_vit_b_16_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(12, 197, 197)"},
        },
    ),
    (
        Reshape447,
        [((1, 12, 64, 197), torch.bfloat16)],
        {
            "model_names": ["pt_beit_microsoft_beit_base_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(12, 64, 197)"},
        },
    ),
    (
        Reshape448,
        [((12, 197, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_beit_microsoft_beit_base_patch16_224_img_cls_hf",
                "pt_vit_vit_b_16_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 12, 197, 64)"},
        },
    ),
    (
        Reshape441,
        [((12, 197, 64), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_b_16_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(12, 197, 64)"},
        },
    ),
    (
        Reshape437,
        [((1, 197, 12, 64), torch.bfloat16)],
        {
            "model_names": ["pt_beit_microsoft_beit_base_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(197, 768)"},
        },
    ),
    (
        Reshape449,
        [((1, 1024, 14, 14), torch.bfloat16)],
        {
            "model_names": [
                "pt_beit_microsoft_beit_large_patch16_224_img_cls_hf",
                "pt_mlp_mixer_mixer_l16_224_img_cls_timm",
                "pt_mlp_mixer_mixer_l16_224_in21k_img_cls_timm",
                "pt_vit_google_vit_large_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 1024, 196, 1)"},
        },
    ),
    (
        Reshape450,
        [((1, 197, 1024), torch.bfloat16)],
        {
            "model_names": [
                "pt_beit_microsoft_beit_large_patch16_224_img_cls_hf",
                "pt_vit_google_vit_large_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(197, 1024)"},
        },
    ),
    (
        Reshape451,
        [((1, 197, 1024), torch.bfloat16)],
        {
            "model_names": [
                "pt_beit_microsoft_beit_large_patch16_224_img_cls_hf",
                "pt_vit_google_vit_large_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 197, 16, 64)"},
        },
    ),
    (
        Reshape452,
        [((197, 1024), torch.bfloat16)],
        {
            "model_names": [
                "pt_beit_microsoft_beit_large_patch16_224_img_cls_hf",
                "pt_vit_google_vit_large_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 197, 1024)"},
        },
    ),
    (
        Reshape451,
        [((197, 1024), torch.bfloat16)],
        {
            "model_names": ["pt_beit_microsoft_beit_large_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 197, 16, 64)"},
        },
    ),
    (
        Reshape453,
        [((1, 16, 197, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_beit_microsoft_beit_large_patch16_224_img_cls_hf",
                "pt_vit_google_vit_large_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(16, 197, 64)"},
        },
    ),
    (
        Reshape454,
        [((16, 197, 197), torch.bfloat16)],
        {
            "model_names": [
                "pt_beit_microsoft_beit_large_patch16_224_img_cls_hf",
                "pt_vit_google_vit_large_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 16, 197, 197)"},
        },
    ),
    (
        Reshape455,
        [((729, 16), torch.bfloat16)],
        {
            "model_names": ["pt_beit_microsoft_beit_large_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 27, 27, 16)"},
        },
    ),
    (
        Reshape456,
        [((1, 27, 27, 16), torch.bfloat16)],
        {
            "model_names": ["pt_beit_microsoft_beit_large_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(729, 16)"},
        },
    ),
    (
        Reshape457,
        [((38809, 16), torch.bfloat16)],
        {
            "model_names": ["pt_beit_microsoft_beit_large_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(197, 197, 16)"},
        },
    ),
    (
        Reshape458,
        [((1, 16, 197, 197), torch.bfloat16)],
        {
            "model_names": [
                "pt_beit_microsoft_beit_large_patch16_224_img_cls_hf",
                "pt_vit_google_vit_large_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(16, 197, 197)"},
        },
    ),
    (
        Reshape459,
        [((1, 16, 64, 197), torch.bfloat16)],
        {
            "model_names": ["pt_beit_microsoft_beit_large_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(16, 64, 197)"},
        },
    ),
    (
        Reshape460,
        [((16, 197, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_beit_microsoft_beit_large_patch16_224_img_cls_hf",
                "pt_vit_google_vit_large_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 16, 197, 64)"},
        },
    ),
    (
        Reshape450,
        [((1, 197, 16, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_beit_microsoft_beit_large_patch16_224_img_cls_hf",
                "pt_vit_google_vit_large_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(197, 1024)"},
        },
    ),
    (
        Reshape461,
        [((1, 32, 4608), torch.float32)],
        {"model_names": ["pt_bloom_bigscience_bloom_1b1_clm_hf"], "pcc": 0.99, "args": {"shape": "(1, 32, 16, 3, 96)"}},
    ),
    (
        Reshape462,
        [((1, 32, 16, 1, 96), torch.float32)],
        {"model_names": ["pt_bloom_bigscience_bloom_1b1_clm_hf"], "pcc": 0.99, "args": {"shape": "(1, 32, 16, 96)"}},
    ),
    (
        Reshape463,
        [((1, 16, 32, 96), torch.float32)],
        {"model_names": ["pt_bloom_bigscience_bloom_1b1_clm_hf"], "pcc": 0.99, "args": {"shape": "(16, 32, 96)"}},
    ),
    (
        Reshape464,
        [((1, 16, 32), torch.float32)],
        {"model_names": ["pt_bloom_bigscience_bloom_1b1_clm_hf"], "pcc": 0.99, "args": {"shape": "(16, 1, 32)"}},
    ),
    (
        Reshape465,
        [((16, 32, 96), torch.float32)],
        {"model_names": ["pt_bloom_bigscience_bloom_1b1_clm_hf"], "pcc": 0.99, "args": {"shape": "(1, 16, 32, 96)"}},
    ),
    (
        Reshape466,
        [((1, 32, 16, 96), torch.float32)],
        {"model_names": ["pt_bloom_bigscience_bloom_1b1_clm_hf"], "pcc": 0.99, "args": {"shape": "(32, 1536)"}},
    ),
    (
        Reshape467,
        [((32, 1536), torch.float32)],
        {"model_names": ["pt_bloom_bigscience_bloom_1b1_clm_hf"], "pcc": 0.99, "args": {"shape": "(1, 32, 1536)"}},
    ),
    (
        Reshape468,
        [((1, 384, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_deit_facebook_deit_small_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 384, 196, 1)"},
        },
    ),
    (
        Reshape469,
        [((1, 197, 384), torch.bfloat16)],
        {
            "model_names": ["pt_deit_facebook_deit_small_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(197, 384)"},
        },
    ),
    (
        Reshape470,
        [((1, 197, 384), torch.bfloat16)],
        {
            "model_names": ["pt_deit_facebook_deit_small_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 197, 6, 64)"},
        },
    ),
    (
        Reshape471,
        [((197, 384), torch.bfloat16)],
        {
            "model_names": ["pt_deit_facebook_deit_small_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 197, 384)"},
        },
    ),
    (
        Reshape472,
        [((1, 6, 197, 64), torch.bfloat16)],
        {
            "model_names": ["pt_deit_facebook_deit_small_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(6, 197, 64)"},
        },
    ),
    (
        Reshape473,
        [((6, 197, 197), torch.bfloat16)],
        {
            "model_names": ["pt_deit_facebook_deit_small_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 6, 197, 197)"},
        },
    ),
    (
        Reshape474,
        [((1, 6, 197, 197), torch.bfloat16)],
        {
            "model_names": ["pt_deit_facebook_deit_small_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(6, 197, 197)"},
        },
    ),
    (
        Reshape475,
        [((6, 197, 64), torch.bfloat16)],
        {
            "model_names": ["pt_deit_facebook_deit_small_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 6, 197, 64)"},
        },
    ),
    (
        Reshape469,
        [((1, 197, 6, 64), torch.bfloat16)],
        {
            "model_names": ["pt_deit_facebook_deit_small_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(197, 384)"},
        },
    ),
    (
        Reshape476,
        [((1, 1, 384), torch.bfloat16)],
        {
            "model_names": ["pt_deit_facebook_deit_small_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 384)"},
        },
    ),
    (
        Reshape477,
        [((1, 100, 256), torch.bfloat16)],
        {
            "model_names": ["pt_detr_facebook_detr_resnet_50_obj_det_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(100, 256)"},
        },
    ),
    (
        Reshape478,
        [((1, 100, 256), torch.bfloat16)],
        {
            "model_names": ["pt_detr_facebook_detr_resnet_50_obj_det_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 100, 8, 32)"},
        },
    ),
    (
        Reshape479,
        [((100, 256), torch.bfloat16)],
        {
            "model_names": ["pt_detr_facebook_detr_resnet_50_obj_det_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 100, 256)"},
        },
    ),
    (
        Reshape480,
        [((1, 8, 100, 32), torch.bfloat16)],
        {
            "model_names": ["pt_detr_facebook_detr_resnet_50_obj_det_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(8, 100, 32)"},
        },
    ),
    (
        Reshape481,
        [((8, 100, 32), torch.bfloat16)],
        {
            "model_names": ["pt_detr_facebook_detr_resnet_50_obj_det_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 8, 100, 32)"},
        },
    ),
    (
        Reshape477,
        [((1, 100, 8, 32), torch.bfloat16)],
        {
            "model_names": ["pt_detr_facebook_detr_resnet_50_obj_det_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(100, 256)"},
        },
    ),
    (
        Reshape482,
        [((64,), torch.bfloat16)],
        {
            "model_names": ["pt_detr_facebook_detr_resnet_50_obj_det_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 64, 1, 1)"},
        },
    ),
    (
        Reshape483,
        [((256,), torch.bfloat16)],
        {
            "model_names": ["pt_detr_facebook_detr_resnet_50_obj_det_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 256, 1, 1)"},
        },
    ),
    (
        Reshape484,
        [((128,), torch.bfloat16)],
        {
            "model_names": ["pt_detr_facebook_detr_resnet_50_obj_det_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 128, 1, 1)"},
        },
    ),
    (
        Reshape174,
        [((512,), torch.bfloat16)],
        {
            "model_names": ["pt_detr_facebook_detr_resnet_50_obj_det_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 512, 1, 1)"},
        },
    ),
    (
        Reshape485,
        [((1024,), torch.bfloat16)],
        {
            "model_names": ["pt_detr_facebook_detr_resnet_50_obj_det_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 1024, 1, 1)"},
        },
    ),
    (
        Reshape45,
        [((2048,), torch.bfloat16)],
        {
            "model_names": ["pt_detr_facebook_detr_resnet_50_obj_det_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 2048, 1, 1)"},
        },
    ),
    (
        Reshape486,
        [((1, 256, 25, 34), torch.bfloat16)],
        {
            "model_names": ["pt_detr_facebook_detr_resnet_50_obj_det_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 256, 850, 1)"},
        },
    ),
    (
        Reshape487,
        [((1, 1, 25, 34), torch.bfloat16)],
        {
            "model_names": ["pt_detr_facebook_detr_resnet_50_obj_det_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 25, 34)"},
        },
    ),
    (
        Reshape488,
        [((1, 25, 34, 64, 2), torch.bfloat16)],
        {
            "model_names": ["pt_detr_facebook_detr_resnet_50_obj_det_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 25, 34, 128, 1)"},
        },
    ),
    (
        Reshape489,
        [((1, 850, 256), torch.bfloat16)],
        {
            "model_names": ["pt_detr_facebook_detr_resnet_50_obj_det_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(850, 256)"},
        },
    ),
    (
        Reshape490,
        [((1, 850, 256), torch.bfloat16)],
        {
            "model_names": ["pt_detr_facebook_detr_resnet_50_obj_det_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 850, 8, 32)"},
        },
    ),
    (
        Reshape491,
        [((850, 256), torch.bfloat16)],
        {
            "model_names": ["pt_detr_facebook_detr_resnet_50_obj_det_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 850, 256)"},
        },
    ),
    (
        Reshape492,
        [((1, 8, 850, 32), torch.bfloat16)],
        {
            "model_names": ["pt_detr_facebook_detr_resnet_50_obj_det_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(8, 850, 32)"},
        },
    ),
    (
        Reshape493,
        [((8, 850, 850), torch.bfloat16)],
        {
            "model_names": ["pt_detr_facebook_detr_resnet_50_obj_det_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 8, 850, 850)"},
        },
    ),
    (
        Reshape494,
        [((1, 25, 34), torch.bfloat16)],
        {
            "model_names": ["pt_detr_facebook_detr_resnet_50_obj_det_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 850, 1)"},
        },
    ),
    (
        Reshape495,
        [((1, 8, 850, 850), torch.bfloat16)],
        {
            "model_names": ["pt_detr_facebook_detr_resnet_50_obj_det_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(8, 850, 850)"},
        },
    ),
    (
        Reshape496,
        [((8, 850, 32), torch.bfloat16)],
        {
            "model_names": ["pt_detr_facebook_detr_resnet_50_obj_det_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 8, 850, 32)"},
        },
    ),
    (
        Reshape489,
        [((1, 850, 8, 32), torch.bfloat16)],
        {
            "model_names": ["pt_detr_facebook_detr_resnet_50_obj_det_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(850, 256)"},
        },
    ),
    (
        Reshape497,
        [((8, 100, 850), torch.bfloat16)],
        {
            "model_names": ["pt_detr_facebook_detr_resnet_50_obj_det_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 8, 100, 850)"},
        },
    ),
    (
        Reshape498,
        [((1, 8, 100, 850), torch.bfloat16)],
        {
            "model_names": ["pt_detr_facebook_detr_resnet_50_obj_det_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(8, 100, 850)"},
        },
    ),
    (
        Reshape499,
        [((100, 92), torch.bfloat16)],
        {
            "model_names": ["pt_detr_facebook_detr_resnet_50_obj_det_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 100, 92)"},
        },
    ),
    (
        Reshape500,
        [((480, 1, 5, 5), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_efficientnet_b1_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b0_ra_in1k_img_cls_timm",
                "pt_efficientnet_hf_hub_timm_tf_efficientnet_b0_aa_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(480, 1, 5, 5)"},
        },
    ),
    (
        Reshape501,
        [((1152, 1, 5, 5), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_efficientnet_b1_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b0_ra_in1k_img_cls_timm",
                "pt_efficientnet_hf_hub_timm_tf_efficientnet_b0_aa_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1152, 1, 5, 5)"},
        },
    ),
    (
        Reshape502,
        [((1152, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_efficientnet_b1_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b0_ra_in1k_img_cls_timm",
                "pt_efficientnet_hf_hub_timm_tf_efficientnet_b0_aa_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1152, 1, 3, 3)"},
        },
    ),
    (
        Reshape503,
        [((384, 1, 5, 5), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b5_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(384, 1, 5, 5)"},
        },
    ),
    (
        Reshape504,
        [((768, 1, 5, 5), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b5_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(768, 1, 5, 5)"},
        },
    ),
    (
        Reshape505,
        [((1056, 1, 5, 5), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b5_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1056, 1, 5, 5)"},
        },
    ),
    (
        Reshape506,
        [((1824, 1, 5, 5), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b5_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1824, 1, 5, 5)"},
        },
    ),
    (
        Reshape507,
        [((1824, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b5_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1824, 1, 3, 3)"},
        },
    ),
    (
        Reshape508,
        [((3072, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b5_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(3072, 1, 3, 3)"},
        },
    ),
    (
        Reshape509,
        [((1, 1, 224, 224), torch.bfloat16)],
        {
            "model_names": ["pt_googlenet_base_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 224, 224)"},
        },
    ),
    (
        Reshape485,
        [((1, 1024, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_googlenet_base_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_swin_swin_b_img_cls_torchvision",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_vovnet_ese_vovnet39b_obj_det_torchhub",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 1024, 1, 1)"},
        },
    ),
    (
        Reshape348,
        [((1, 1024, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv1_basic_img_cls_torchvision",
                "pt_vovnet_vovnet39_img_cls_osmr",
                "pt_vovnet_vovnet57_img_cls_osmr",
                "pt_vovnet_vovnet_v1_57_obj_det_torchhub",
                "pt_vovnet_v1_vovnet39_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 1024)"},
        },
    ),
    (
        Reshape510,
        [((1, 12, 64, 256), torch.float32)],
        {"model_names": ["pt_gptneo_eleutherai_gpt_neo_125m_clm_hf"], "pcc": 0.99, "args": {"shape": "(12, 64, 256)"}},
    ),
    (
        Reshape511,
        [((1, 768, 8, 32), torch.bfloat16)],
        {
            "model_names": ["pt_mgp_alibaba_damo_mgp_str_base_scene_text_recognition_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 768, 256, 1)"},
        },
    ),
    (
        Reshape512,
        [((1, 257, 2304), torch.bfloat16)],
        {
            "model_names": ["pt_mgp_alibaba_damo_mgp_str_base_scene_text_recognition_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 257, 3, 12, 64)"},
        },
    ),
    (
        Reshape513,
        [((1, 1, 12, 257, 64), torch.bfloat16)],
        {
            "model_names": ["pt_mgp_alibaba_damo_mgp_str_base_scene_text_recognition_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(12, 257, 64)"},
        },
    ),
    (
        Reshape514,
        [((1, 1, 12, 257, 64), torch.bfloat16)],
        {
            "model_names": ["pt_mgp_alibaba_damo_mgp_str_base_scene_text_recognition_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 12, 257, 64)"},
        },
    ),
    (
        Reshape515,
        [((12, 257, 257), torch.bfloat16)],
        {
            "model_names": ["pt_mgp_alibaba_damo_mgp_str_base_scene_text_recognition_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 12, 257, 257)"},
        },
    ),
    (
        Reshape516,
        [((1, 12, 257, 257), torch.bfloat16)],
        {
            "model_names": ["pt_mgp_alibaba_damo_mgp_str_base_scene_text_recognition_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(12, 257, 257)"},
        },
    ),
    (
        Reshape517,
        [((1, 12, 64, 257), torch.bfloat16)],
        {
            "model_names": ["pt_mgp_alibaba_damo_mgp_str_base_scene_text_recognition_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(12, 64, 257)"},
        },
    ),
    (
        Reshape514,
        [((12, 257, 64), torch.bfloat16)],
        {
            "model_names": ["pt_mgp_alibaba_damo_mgp_str_base_scene_text_recognition_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 12, 257, 64)"},
        },
    ),
    (
        Reshape518,
        [((1, 257, 12, 64), torch.bfloat16)],
        {
            "model_names": ["pt_mgp_alibaba_damo_mgp_str_base_scene_text_recognition_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(257, 768)"},
        },
    ),
    (
        Reshape519,
        [((257, 768), torch.bfloat16)],
        {
            "model_names": ["pt_mgp_alibaba_damo_mgp_str_base_scene_text_recognition_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 257, 768)"},
        },
    ),
    (
        Reshape520,
        [((1, 27, 257, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mgp_alibaba_damo_mgp_str_base_scene_text_recognition_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 27, 257, 1)"},
        },
    ),
    (
        Reshape521,
        [((1, 768, 257, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mgp_alibaba_damo_mgp_str_base_scene_text_recognition_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 768, 257, 1)"},
        },
    ),
    (
        Reshape174,
        [((1, 512, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv3_ssd_resnet18_img_cls_torchvision",
                "pt_resnet_resnet18_img_cls_torchvision",
                "pt_resnet_resnet34_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet34_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 512, 1, 1)"},
        },
    ),
    (
        Reshape522,
        [((32, 2), torch.float32)],
        {
            "model_names": [
                "pt_opt_facebook_opt_125m_seq_cls_hf",
                "pt_opt_facebook_opt_350m_seq_cls_hf",
                "pt_opt_facebook_opt_1_3b_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(32, 2)"},
        },
    ),
    (
        Reshape523,
        [((1, 512, 322), torch.bfloat16)],
        {
            "model_names": ["pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 512, 1, 322)"},
        },
    ),
    (
        Reshape524,
        [((1, 55, 55, 64), torch.bfloat16)],
        {
            "model_names": ["pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 3025, 64)"},
        },
    ),
    (
        Reshape525,
        [((1, 3025, 322), torch.bfloat16)],
        {
            "model_names": ["pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(3025, 322)"},
        },
    ),
    (
        Reshape526,
        [((1, 3025, 322), torch.bfloat16)],
        {
            "model_names": ["pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 3025, 1, 322)"},
        },
    ),
    (
        Reshape527,
        [((3025, 322), torch.bfloat16)],
        {
            "model_names": ["pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 3025, 322)"},
        },
    ),
    (
        Reshape528,
        [((1, 512, 3025), torch.bfloat16)],
        {
            "model_names": ["pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 1, 512, 3025)"},
        },
    ),
    (
        Reshape529,
        [((1, 1, 512, 3025), torch.bfloat16)],
        {
            "model_names": ["pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 512, 3025)"},
        },
    ),
    (
        Reshape530,
        [((1, 1, 322, 3025), torch.bfloat16)],
        {
            "model_names": ["pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 322, 3025)"},
        },
    ),
    (
        Reshape531,
        [((1, 512, 1024), torch.bfloat16)],
        {
            "model_names": [
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(512, 1024)"},
        },
    ),
    (
        Reshape532,
        [((1, 512, 1024), torch.bfloat16)],
        {
            "model_names": [
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 512, 8, 128)"},
        },
    ),
    (
        Reshape533,
        [((1, 512, 1024), torch.bfloat16)],
        {
            "model_names": [
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 512, 1, 1024)"},
        },
    ),
    (
        Reshape534,
        [((512, 1024), torch.bfloat16)],
        {
            "model_names": [
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 512, 1024)"},
        },
    ),
    (
        Reshape535,
        [((1, 8, 512, 128), torch.bfloat16)],
        {
            "model_names": [
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(8, 512, 128)"},
        },
    ),
    (
        Reshape536,
        [((8, 512, 512), torch.bfloat16)],
        {
            "model_names": [
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 8, 512, 512)"},
        },
    ),
    (
        Reshape537,
        [((1, 8, 512, 512), torch.bfloat16)],
        {
            "model_names": [
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(8, 512, 512)"},
        },
    ),
    (
        Reshape538,
        [((1, 8, 128, 512), torch.bfloat16)],
        {
            "model_names": [
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(8, 128, 512)"},
        },
    ),
    (
        Reshape539,
        [((8, 512, 128), torch.bfloat16)],
        {
            "model_names": [
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 8, 512, 128)"},
        },
    ),
    (
        Reshape531,
        [((1, 512, 8, 128), torch.bfloat16)],
        {
            "model_names": [
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(512, 1024)"},
        },
    ),
    (
        Reshape540,
        [((1, 1, 512), torch.bfloat16)],
        {
            "model_names": [
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 1, 1, 512)"},
        },
    ),
    (
        Reshape69,
        [((1, 1, 1, 512), torch.bfloat16)],
        {
            "model_names": [
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 1, 512)"},
        },
    ),
    (
        Reshape541,
        [((1, 1, 1024, 512), torch.bfloat16)],
        {
            "model_names": [
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 1024, 512)"},
        },
    ),
    (
        Reshape89,
        [((1, 1, 1000), torch.bfloat16)],
        {
            "model_names": [
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 1000)"},
        },
    ),
    (
        Reshape22,
        [((1, 12, 2048), torch.float32)],
        {
            "model_names": ["pt_phi1_microsoft_phi_1_token_cls_hf", "pt_phi_1_5_microsoft_phi_1_5_token_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(12, 2048)"},
        },
    ),
    (
        Reshape25,
        [((1, 12, 2048), torch.float32)],
        {
            "model_names": ["pt_phi1_microsoft_phi_1_token_cls_hf", "pt_phi_1_5_microsoft_phi_1_5_token_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 12, 32, 64)"},
        },
    ),
    (
        Reshape542,
        [((12, 2048), torch.float32)],
        {
            "model_names": ["pt_phi1_microsoft_phi_1_token_cls_hf", "pt_phi_1_5_microsoft_phi_1_5_token_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 12, 2048)"},
        },
    ),
    (
        Reshape543,
        [((32, 12, 12), torch.float32)],
        {
            "model_names": ["pt_phi1_microsoft_phi_1_token_cls_hf", "pt_phi_1_5_microsoft_phi_1_5_token_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 32, 12, 12)"},
        },
    ),
    (
        Reshape544,
        [((1, 32, 12, 12), torch.float32)],
        {
            "model_names": ["pt_phi1_microsoft_phi_1_token_cls_hf", "pt_phi_1_5_microsoft_phi_1_5_token_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(32, 12, 12)"},
        },
    ),
    (
        Reshape19,
        [((32, 12, 64), torch.float32)],
        {
            "model_names": ["pt_phi1_microsoft_phi_1_token_cls_hf", "pt_phi_1_5_microsoft_phi_1_5_token_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 32, 12, 64)"},
        },
    ),
    (
        Reshape545,
        [((12, 8192), torch.float32)],
        {
            "model_names": ["pt_phi1_microsoft_phi_1_token_cls_hf", "pt_phi_1_5_microsoft_phi_1_5_token_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 12, 8192)"},
        },
    ),
    (
        Reshape546,
        [((1, 6, 1024), torch.float32)],
        {"model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"], "pcc": 0.99, "args": {"shape": "(6, 1024)"}},
    ),
    (
        Reshape547,
        [((1, 6, 1024), torch.float32)],
        {"model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"], "pcc": 0.99, "args": {"shape": "(1, 6, 16, 64)"}},
    ),
    (
        Reshape548,
        [((6, 1024), torch.float32)],
        {"model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"], "pcc": 0.99, "args": {"shape": "(1, 6, 1024)"}},
    ),
    (
        Reshape549,
        [((1, 16, 6, 64), torch.float32)],
        {"model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"], "pcc": 0.99, "args": {"shape": "(16, 6, 64)"}},
    ),
    (
        Reshape550,
        [((16, 6, 6), torch.float32)],
        {"model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"], "pcc": 0.99, "args": {"shape": "(1, 16, 6, 6)"}},
    ),
    (
        Reshape551,
        [((1, 16, 6, 6), torch.float32)],
        {"model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"], "pcc": 0.99, "args": {"shape": "(16, 6, 6)"}},
    ),
    (
        Reshape552,
        [((16, 6, 64), torch.float32)],
        {"model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"], "pcc": 0.99, "args": {"shape": "(1, 16, 6, 64)"}},
    ),
    (
        Reshape546,
        [((1, 6, 16, 64), torch.float32)],
        {"model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"], "pcc": 0.99, "args": {"shape": "(6, 1024)"}},
    ),
    (
        Reshape553,
        [((6, 2816), torch.float32)],
        {"model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"], "pcc": 0.99, "args": {"shape": "(1, 6, 2816)"}},
    ),
    (
        Reshape554,
        [((1, 888, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_1_6gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 888, 1, 1)"},
        },
    ),
    (
        Reshape555,
        [((1, 1512, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_3_2gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 1512, 1, 1)"},
        },
    ),
    (
        Reshape556,
        [((1, 784, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_800mf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 784, 1, 1)"},
        },
    ),
    (
        Reshape156,
        [((1, 768, 256), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 768, 16, 16)"},
        },
    ),
    (
        Reshape157,
        [((1, 768, 1024), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 768, 32, 32)"},
        },
    ),
    (
        Reshape158,
        [((1, 768, 4096), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 768, 64, 64)"},
        },
    ),
    (
        Reshape159,
        [((1, 768, 16384), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 768, 128, 128)"},
        },
    ),
    (
        Reshape557,
        [((64, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": [
                "pt_xception_xception71_tf_in1k_img_cls_timm",
                "pt_xception_xception_img_cls_timm",
                "pt_mobilenetv1_basic_img_cls_torchvision",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception65_img_cls_timm",
                "pt_yolox_yolox_nano_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(64, 1, 3, 3)"},
        },
    ),
    (
        Reshape558,
        [((128, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": [
                "pt_xception_xception71_tf_in1k_img_cls_timm",
                "pt_xception_xception_img_cls_timm",
                "pt_mobilenetv1_basic_img_cls_torchvision",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception65_img_cls_timm",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pt_yolox_yolox_nano_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(128, 1, 3, 3)"},
        },
    ),
    (
        Reshape559,
        [((728, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": [
                "pt_xception_xception71_tf_in1k_img_cls_timm",
                "pt_xception_xception_img_cls_timm",
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception65_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(728, 1, 3, 3)"},
        },
    ),
    (
        Reshape560,
        [((1024, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": [
                "pt_xception_xception71_tf_in1k_img_cls_timm",
                "pt_xception_xception_img_cls_timm",
                "pt_mobilenetv1_basic_img_cls_torchvision",
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception65_img_cls_timm",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1024, 1, 3, 3)"},
        },
    ),
    (
        Reshape561,
        [((1, 3, 85, 40, 40), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5l_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_1280x1280",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_640x640",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 255, 40, 40)"},
        },
    ),
    (
        Reshape562,
        [((1, 255, 40, 40), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5l_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_1280x1280",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_640x640",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 1, 255, 1600)"},
        },
    ),
    (
        Reshape563,
        [((1, 1, 255, 1600), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5l_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_1280x1280",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_640x640",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 3, 85, 1600)"},
        },
    ),
    (
        Reshape564,
        [((1, 3, 1600, 85), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5l_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_1280x1280",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_640x640",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 4800, 85)"},
        },
    ),
    (
        Reshape565,
        [((1, 3, 85, 20, 20), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5l_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_640x640",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 255, 20, 20)"},
        },
    ),
    (
        Reshape566,
        [((1, 255, 20, 20), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5l_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_640x640",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 1, 255, 400)"},
        },
    ),
    (
        Reshape567,
        [((1, 1, 255, 400), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5l_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_640x640",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 3, 85, 400)"},
        },
    ),
    (
        Reshape568,
        [((1, 3, 400, 85), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5l_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_640x640",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 1200, 85)"},
        },
    ),
    (
        Reshape569,
        [((1, 3, 85, 10, 10), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5l_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_320x320",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 255, 10, 10)"},
        },
    ),
    (
        Reshape570,
        [((1, 255, 10, 10), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5l_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_320x320",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 1, 255, 100)"},
        },
    ),
    (
        Reshape571,
        [((1, 1, 255, 100), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5l_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_320x320",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 3, 85, 100)"},
        },
    ),
    (
        Reshape572,
        [((1, 3, 100, 85), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5l_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_320x320",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 300, 85)"},
        },
    ),
    (
        Reshape573,
        [((64,), torch.float32)],
        {"model_names": ["tf_resnet_resnet50_img_cls_keras"], "pcc": 0.99, "args": {"shape": "(64,)"}},
    ),
    (
        Reshape574,
        [((64,), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99, "args": {"shape": "(1, 1, 1, 64)"}},
    ),
    (
        Reshape575,
        [((256,), torch.float32)],
        {"model_names": ["tf_resnet_resnet50_img_cls_keras"], "pcc": 0.99, "args": {"shape": "(256,)"}},
    ),
    (
        Reshape576,
        [((256,), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99, "args": {"shape": "(1, 1, 1, 256)"}},
    ),
    (
        Reshape577,
        [((512,), torch.float32)],
        {"model_names": ["tf_resnet_resnet50_img_cls_keras"], "pcc": 0.99, "args": {"shape": "(512,)"}},
    ),
    (
        Reshape540,
        [((512,), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99, "args": {"shape": "(1, 1, 1, 512)"}},
    ),
    (
        Reshape578,
        [((128,), torch.float32)],
        {"model_names": ["tf_resnet_resnet50_img_cls_keras"], "pcc": 0.99, "args": {"shape": "(128,)"}},
    ),
    (
        Reshape579,
        [((128,), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99, "args": {"shape": "(1, 1, 1, 128)"}},
    ),
    (
        Reshape580,
        [((1024,), torch.float32)],
        {"model_names": ["tf_resnet_resnet50_img_cls_keras"], "pcc": 0.99, "args": {"shape": "(1024,)"}},
    ),
    (
        Reshape349,
        [((1024,), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99, "args": {"shape": "(1, 1, 1, 1024)"}},
    ),
    (
        Reshape581,
        [((2048,), torch.float32)],
        {"model_names": ["tf_resnet_resnet50_img_cls_keras"], "pcc": 0.99, "args": {"shape": "(2048,)"}},
    ),
    (
        Reshape582,
        [((2048,), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99, "args": {"shape": "(1, 1, 1, 2048)"}},
    ),
    (
        Reshape477,
        [((1, 100, 256), torch.float32)],
        {"model_names": ["onnx_detr_facebook_detr_resnet_50_obj_det_hf"], "pcc": 0.99, "args": {"shape": "(100, 256)"}},
    ),
    (
        Reshape478,
        [((1, 100, 256), torch.float32)],
        {
            "model_names": ["onnx_detr_facebook_detr_resnet_50_obj_det_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 100, 8, 32)"},
        },
    ),
    (
        Reshape479,
        [((100, 256), torch.float32)],
        {
            "model_names": ["onnx_detr_facebook_detr_resnet_50_obj_det_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 100, 256)"},
        },
    ),
    (
        Reshape480,
        [((1, 8, 100, 32), torch.float32)],
        {
            "model_names": ["onnx_detr_facebook_detr_resnet_50_obj_det_hf"],
            "pcc": 0.99,
            "args": {"shape": "(8, 100, 32)"},
        },
    ),
    (
        Reshape583,
        [((8, 100, 100), torch.float32)],
        {
            "model_names": ["onnx_detr_facebook_detr_resnet_50_obj_det_hf"],
            "pcc": 0.99,
            "args": {"shape": "(8, 100, 100)"},
        },
    ),
    (
        Reshape481,
        [((8, 100, 32), torch.float32)],
        {
            "model_names": ["onnx_detr_facebook_detr_resnet_50_obj_det_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 8, 100, 32)"},
        },
    ),
    (
        Reshape477,
        [((1, 100, 8, 32), torch.float32)],
        {"model_names": ["onnx_detr_facebook_detr_resnet_50_obj_det_hf"], "pcc": 0.99, "args": {"shape": "(100, 256)"}},
    ),
    (
        Reshape584,
        [((1, 256, 14, 20), torch.float32)],
        {
            "model_names": ["onnx_detr_facebook_detr_resnet_50_obj_det_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 256, 280)"},
        },
    ),
    (
        Reshape585,
        [((1, 280, 256), torch.float32)],
        {"model_names": ["onnx_detr_facebook_detr_resnet_50_obj_det_hf"], "pcc": 0.99, "args": {"shape": "(280, 256)"}},
    ),
    (
        Reshape586,
        [((1, 280, 256), torch.float32)],
        {
            "model_names": ["onnx_detr_facebook_detr_resnet_50_obj_det_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 280, 8, 32)"},
        },
    ),
    (
        Reshape587,
        [((280, 256), torch.float32)],
        {
            "model_names": ["onnx_detr_facebook_detr_resnet_50_obj_det_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 280, 256)"},
        },
    ),
    (
        Reshape588,
        [((1, 8, 280, 32), torch.float32)],
        {
            "model_names": ["onnx_detr_facebook_detr_resnet_50_obj_det_hf"],
            "pcc": 0.99,
            "args": {"shape": "(8, 280, 32)"},
        },
    ),
    (
        Reshape589,
        [((8, 280, 280), torch.float32)],
        {
            "model_names": ["onnx_detr_facebook_detr_resnet_50_obj_det_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 8, 280, 280)"},
        },
    ),
    (
        Reshape590,
        [((1, 8, 280, 280), torch.float32)],
        {
            "model_names": ["onnx_detr_facebook_detr_resnet_50_obj_det_hf"],
            "pcc": 0.99,
            "args": {"shape": "(8, 280, 280)"},
        },
    ),
    (
        Reshape591,
        [((8, 280, 32), torch.float32)],
        {
            "model_names": ["onnx_detr_facebook_detr_resnet_50_obj_det_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 8, 280, 32)"},
        },
    ),
    (
        Reshape585,
        [((1, 280, 8, 32), torch.float32)],
        {"model_names": ["onnx_detr_facebook_detr_resnet_50_obj_det_hf"], "pcc": 0.99, "args": {"shape": "(280, 256)"}},
    ),
    (
        Reshape592,
        [((8, 100, 280), torch.float32)],
        {
            "model_names": ["onnx_detr_facebook_detr_resnet_50_obj_det_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 8, 100, 280)"},
        },
    ),
    (
        Reshape593,
        [((1, 8, 100, 280), torch.float32)],
        {
            "model_names": ["onnx_detr_facebook_detr_resnet_50_obj_det_hf"],
            "pcc": 0.99,
            "args": {"shape": "(8, 100, 280)"},
        },
    ),
    (
        Reshape499,
        [((100, 92), torch.float32)],
        {
            "model_names": ["onnx_detr_facebook_detr_resnet_50_obj_det_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 100, 92)"},
        },
    ),
    (
        Reshape331,
        [((1, 256, 256), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 256, 16, 16)"},
        },
    ),
    (
        Reshape332,
        [((1, 256, 256), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(256, 256)"},
        },
    ),
    (
        Reshape333,
        [((1, 256, 256), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 256, 8, 32)"},
        },
    ),
    (
        Reshape334,
        [((1, 256, 256), torch.float32)],
        {"model_names": ["onnx_segformer_nvidia_mit_b0_img_cls_hf"], "pcc": 0.99, "args": {"shape": "(1, 256, 256)"}},
    ),
    (
        Reshape335,
        [((1, 256, 256), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 16, 16, 256)"},
        },
    ),
    (
        Reshape336,
        [((1, 256, 1024), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 256, 32, 32)"},
        },
    ),
    (
        Reshape594,
        [((1, 256, 1024), torch.float32)],
        {
            "model_names": [
                "pt_codegen_salesforce_codegen_350m_mono_clm_hf",
                "pt_xglm_facebook_xglm_564m_clm_hf",
                "pt_bart_facebook_bart_large_mnli_seq_cls_hf",
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(256, 1024)"},
        },
    ),
    (
        Reshape595,
        [((1, 256, 1024), torch.float32)],
        {
            "model_names": ["pt_xglm_facebook_xglm_564m_clm_hf", "pt_bart_facebook_bart_large_mnli_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 256, 16, 64)"},
        },
    ),
    (
        Reshape337,
        [((1, 256, 4096), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 256, 64, 64)"},
        },
    ),
    (
        Reshape596,
        [((1, 10, 768), torch.float32)],
        {"model_names": ["pd_bert_bert_base_japanese_mlm_padlenlp"], "pcc": 0.99, "args": {"shape": "(10, 768)"}},
    ),
    (
        Reshape597,
        [((1, 10, 768), torch.float32)],
        {"model_names": ["pd_bert_bert_base_japanese_mlm_padlenlp"], "pcc": 0.99, "args": {"shape": "(1, 10, 12, 64)"}},
    ),
    (
        Reshape598,
        [((10, 768), torch.float32)],
        {"model_names": ["pd_bert_bert_base_japanese_mlm_padlenlp"], "pcc": 0.99, "args": {"shape": "(1, 10, 768)"}},
    ),
    (
        Reshape599,
        [((1, 12, 10, 64), torch.float32)],
        {"model_names": ["pd_bert_bert_base_japanese_mlm_padlenlp"], "pcc": 0.99, "args": {"shape": "(12, 10, 64)"}},
    ),
    (
        Reshape600,
        [((1, 12, 64, 10), torch.float32)],
        {"model_names": ["pd_bert_bert_base_japanese_mlm_padlenlp"], "pcc": 0.99, "args": {"shape": "(12, 64, 10)"}},
    ),
    (
        Reshape601,
        [((12, 10, 10), torch.float32)],
        {"model_names": ["pd_bert_bert_base_japanese_mlm_padlenlp"], "pcc": 0.99, "args": {"shape": "(1, 12, 10, 10)"}},
    ),
    (
        Reshape602,
        [((1, 12, 10, 10), torch.float32)],
        {"model_names": ["pd_bert_bert_base_japanese_mlm_padlenlp"], "pcc": 0.99, "args": {"shape": "(12, 10, 10)"}},
    ),
    (
        Reshape603,
        [((12, 10, 64), torch.float32)],
        {"model_names": ["pd_bert_bert_base_japanese_mlm_padlenlp"], "pcc": 0.99, "args": {"shape": "(1, 12, 10, 64)"}},
    ),
    (
        Reshape596,
        [((1, 10, 12, 64), torch.float32)],
        {"model_names": ["pd_bert_bert_base_japanese_mlm_padlenlp"], "pcc": 0.99, "args": {"shape": "(10, 768)"}},
    ),
    (
        Reshape485,
        [((1, 1024, 1, 1), torch.float32)],
        {
            "model_names": ["pd_densenet_121_img_cls_paddlemodels", "pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(1, 1024, 1, 1)"},
        },
    ),
    (
        Reshape348,
        [((1, 1024, 1, 1), torch.float32)],
        {
            "model_names": ["onnx_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub"],
            "pcc": 0.99,
            "args": {"shape": "(1, 1024)"},
        },
    ),
    (
        Reshape604,
        [((1, 256, 6, 6), torch.bfloat16)],
        {
            "model_names": ["pt_alexnet_base_img_cls_osmr"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 9216)"},
        },
    ),
    (
        Reshape605,
        [((256, 1024), torch.float32)],
        {
            "model_names": [
                "pt_codegen_salesforce_codegen_350m_mono_clm_hf",
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 256, 4, 256)"},
        },
    ),
    (
        Reshape606,
        [((256, 1024), torch.float32)],
        {
            "model_names": [
                "pt_codegen_salesforce_codegen_350m_mono_clm_hf",
                "pt_xglm_facebook_xglm_564m_clm_hf",
                "pt_bart_facebook_bart_large_mnli_seq_cls_hf",
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 256, 1024)"},
        },
    ),
    (
        Reshape595,
        [((1, 256, 4, 256), torch.float32)],
        {
            "model_names": [
                "pt_codegen_salesforce_codegen_350m_mono_clm_hf",
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 256, 16, 64)"},
        },
    ),
    (
        Reshape607,
        [((1, 256, 16, 16, 2), torch.float32)],
        {
            "model_names": [
                "pt_codegen_salesforce_codegen_350m_mono_clm_hf",
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 256, 16, 32, 1)"},
        },
    ),
    (
        Reshape608,
        [((1, 16, 256, 64), torch.float32)],
        {
            "model_names": [
                "pt_codegen_salesforce_codegen_350m_mono_clm_hf",
                "pt_xglm_facebook_xglm_564m_clm_hf",
                "pt_bart_facebook_bart_large_mnli_seq_cls_hf",
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(16, 256, 64)"},
        },
    ),
    (
        Reshape609,
        [((16, 256, 256), torch.float32)],
        {
            "model_names": [
                "pt_codegen_salesforce_codegen_350m_mono_clm_hf",
                "pt_xglm_facebook_xglm_564m_clm_hf",
                "pt_bart_facebook_bart_large_mnli_seq_cls_hf",
                "pt_xglm_facebook_xglm_1_7b_clm_hf",
                "pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf",
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 16, 256, 256)"},
        },
    ),
    (
        Reshape610,
        [((1, 16, 256, 256), torch.float32)],
        {
            "model_names": [
                "pt_codegen_salesforce_codegen_350m_mono_clm_hf",
                "pt_xglm_facebook_xglm_564m_clm_hf",
                "pt_bart_facebook_bart_large_mnli_seq_cls_hf",
                "pt_xglm_facebook_xglm_1_7b_clm_hf",
                "pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf",
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(16, 256, 256)"},
        },
    ),
    (
        Reshape611,
        [((1, 16, 64, 256), torch.float32)],
        {
            "model_names": [
                "pt_codegen_salesforce_codegen_350m_mono_clm_hf",
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(16, 64, 256)"},
        },
    ),
    (
        Reshape612,
        [((16, 256, 64), torch.float32)],
        {
            "model_names": [
                "pt_codegen_salesforce_codegen_350m_mono_clm_hf",
                "pt_xglm_facebook_xglm_564m_clm_hf",
                "pt_bart_facebook_bart_large_mnli_seq_cls_hf",
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 16, 256, 64)"},
        },
    ),
    (
        Reshape594,
        [((1, 256, 16, 64), torch.float32)],
        {
            "model_names": [
                "pt_codegen_salesforce_codegen_350m_mono_clm_hf",
                "pt_xglm_facebook_xglm_564m_clm_hf",
                "pt_bart_facebook_bart_large_mnli_seq_cls_hf",
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(256, 1024)"},
        },
    ),
    (
        Reshape408,
        [((256, 4096), torch.float32)],
        {
            "model_names": [
                "pt_codegen_salesforce_codegen_350m_mono_clm_hf",
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 256, 4096)"},
        },
    ),
    (
        Reshape613,
        [((528, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b2_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(528, 1, 3, 3)"},
        },
    ),
    (
        Reshape614,
        [((528, 1, 5, 5), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b2_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(528, 1, 5, 5)"},
        },
    ),
    (
        Reshape615,
        [((720, 1, 5, 5), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b2_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(720, 1, 5, 5)"},
        },
    ),
    (
        Reshape616,
        [((1248, 1, 5, 5), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b2_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1248, 1, 5, 5)"},
        },
    ),
    (
        Reshape617,
        [((1248, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b2_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1248, 1, 3, 3)"},
        },
    ),
    (
        Reshape618,
        [((2112, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b2_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(2112, 1, 3, 3)"},
        },
    ),
    (
        Reshape619,
        [((1, 1408, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b2_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 1408, 1, 1)"},
        },
    ),
    (
        Reshape620,
        [((432, 1, 5, 5), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b6_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(432, 1, 5, 5)"},
        },
    ),
    (
        Reshape621,
        [((432, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b6_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(432, 1, 3, 3)"},
        },
    ),
    (
        Reshape622,
        [((864, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b6_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(864, 1, 3, 3)"},
        },
    ),
    (
        Reshape623,
        [((864, 1, 5, 5), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b6_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(864, 1, 5, 5)"},
        },
    ),
    (
        Reshape624,
        [((1200, 1, 5, 5), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b6_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1200, 1, 5, 5)"},
        },
    ),
    (
        Reshape625,
        [((2064, 1, 5, 5), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b6_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(2064, 1, 5, 5)"},
        },
    ),
    (
        Reshape626,
        [((2064, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b6_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(2064, 1, 3, 3)"},
        },
    ),
    (
        Reshape627,
        [((3456, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b6_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(3456, 1, 3, 3)"},
        },
    ),
    (
        Reshape628,
        [((1, 2304, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b6_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 2304, 1, 1)"},
        },
    ),
    (
        Reshape629,
        [((336, 1, 5, 5), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_hf_hub_timm_efficientnet_b4_ra2_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(336, 1, 5, 5)"},
        },
    ),
    (
        Reshape630,
        [((672, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_hf_hub_timm_efficientnet_b4_ra2_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(672, 1, 3, 3)"},
        },
    ),
    (
        Reshape631,
        [((960, 1, 5, 5), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_hf_hub_timm_efficientnet_b4_ra2_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(960, 1, 5, 5)"},
        },
    ),
    (
        Reshape632,
        [((1632, 1, 5, 5), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_hf_hub_timm_efficientnet_b4_ra2_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1632, 1, 5, 5)"},
        },
    ),
    (
        Reshape633,
        [((1632, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_hf_hub_timm_efficientnet_b4_ra2_in1k_img_cls_timm",
                "pt_efficientnet_hf_hub_timm_efficientnetv2_rw_s_ra2_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1632, 1, 3, 3)"},
        },
    ),
    (
        Reshape634,
        [((2688, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_hf_hub_timm_efficientnet_b4_ra2_in1k_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(2688, 1, 3, 3)"},
        },
    ),
    (
        Reshape635,
        [((1, 1792, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_hf_hub_timm_efficientnet_b4_ra2_in1k_img_cls_timm",
                "pt_efficientnet_hf_hub_timm_efficientnetv2_rw_s_ra2_in1k_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 1792, 1, 1)"},
        },
    ),
    (
        Reshape636,
        [((1, 64, 120, 160), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 64, 19200, 1)"},
        },
    ),
    (
        Reshape637,
        [((1, 19200, 64), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 19200, 1, 64)"},
        },
    ),
    (
        Reshape638,
        [((1, 19200, 64), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 120, 160, 64)"},
        },
    ),
    (
        Reshape639,
        [((1, 64, 19200), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 64, 120, 160)"},
        },
    ),
    (
        Reshape640,
        [((1, 64, 15, 20), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 64, 300)"},
        },
    ),
    (
        Reshape641,
        [((1, 300, 64), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(300, 64)"},
        },
    ),
    (
        Reshape642,
        [((1, 300, 64), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 300, 1, 64)"},
        },
    ),
    (
        Reshape643,
        [((300, 64), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 300, 64)"},
        },
    ),
    (
        Reshape644,
        [((1, 19200, 300), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 1, 19200, 300)"},
        },
    ),
    (
        Reshape645,
        [((1, 1, 19200, 300), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 19200, 300)"},
        },
    ),
    (
        Reshape640,
        [((1, 1, 64, 300), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 64, 300)"},
        },
    ),
    (
        Reshape646,
        [((1, 256, 19200), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 256, 120, 160)"},
        },
    ),
    (
        Reshape647,
        [((1, 256, 120, 160), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 256, 19200, 1)"},
        },
    ),
    (
        Reshape648,
        [((1, 128, 60, 80), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 128, 4800, 1)"},
        },
    ),
    (
        Reshape649,
        [((1, 4800, 128), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 4800, 2, 64)"},
        },
    ),
    (
        Reshape650,
        [((1, 4800, 128), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 60, 80, 128)"},
        },
    ),
    (
        Reshape651,
        [((1, 2, 4800, 64), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(2, 4800, 64)"},
        },
    ),
    (
        Reshape652,
        [((1, 128, 4800), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 128, 60, 80)"},
        },
    ),
    (
        Reshape653,
        [((1, 128, 15, 20), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 128, 300)"},
        },
    ),
    (
        Reshape654,
        [((1, 300, 128), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(300, 128)"},
        },
    ),
    (
        Reshape655,
        [((1, 300, 128), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 300, 2, 64)"},
        },
    ),
    (
        Reshape656,
        [((300, 128), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 300, 128)"},
        },
    ),
    (
        Reshape657,
        [((1, 2, 300, 64), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(2, 300, 64)"},
        },
    ),
    (
        Reshape658,
        [((2, 4800, 300), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 2, 4800, 300)"},
        },
    ),
    (
        Reshape659,
        [((1, 2, 4800, 300), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(2, 4800, 300)"},
        },
    ),
    (
        Reshape660,
        [((1, 2, 64, 300), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(2, 64, 300)"},
        },
    ),
    (
        Reshape661,
        [((2, 4800, 64), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 2, 4800, 64)"},
        },
    ),
    (
        Reshape662,
        [((1, 4800, 2, 64), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(4800, 128)"},
        },
    ),
    (
        Reshape663,
        [((4800, 128), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 4800, 128)"},
        },
    ),
    (
        Reshape664,
        [((1, 512, 4800), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 512, 60, 80)"},
        },
    ),
    (
        Reshape665,
        [((1, 512, 60, 80), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 512, 4800, 1)"},
        },
    ),
    (
        Reshape666,
        [((1, 320, 30, 40), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 320, 1200, 1)"},
        },
    ),
    (
        Reshape667,
        [((1, 1200, 320), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 1200, 5, 64)"},
        },
    ),
    (
        Reshape668,
        [((1, 1200, 320), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 30, 40, 320)"},
        },
    ),
    (
        Reshape669,
        [((1, 5, 1200, 64), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(5, 1200, 64)"},
        },
    ),
    (
        Reshape670,
        [((1, 320, 1200), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 320, 30, 40)"},
        },
    ),
    (
        Reshape671,
        [((1, 320, 15, 20), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 320, 300)"},
        },
    ),
    (
        Reshape672,
        [((1, 300, 320), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(300, 320)"},
        },
    ),
    (
        Reshape673,
        [((1, 300, 320), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 300, 5, 64)"},
        },
    ),
    (
        Reshape674,
        [((300, 320), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 300, 320)"},
        },
    ),
    (
        Reshape675,
        [((1, 5, 300, 64), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(5, 300, 64)"},
        },
    ),
    (
        Reshape676,
        [((5, 1200, 300), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 5, 1200, 300)"},
        },
    ),
    (
        Reshape677,
        [((1, 5, 1200, 300), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(5, 1200, 300)"},
        },
    ),
    (
        Reshape678,
        [((1, 5, 64, 300), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(5, 64, 300)"},
        },
    ),
    (
        Reshape679,
        [((5, 1200, 64), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 5, 1200, 64)"},
        },
    ),
    (
        Reshape680,
        [((1, 1200, 5, 64), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1200, 320)"},
        },
    ),
    (
        Reshape681,
        [((1200, 320), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 1200, 320)"},
        },
    ),
    (
        Reshape682,
        [((1, 1280, 1200), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 1280, 30, 40)"},
        },
    ),
    (
        Reshape683,
        [((1, 1280, 30, 40), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 1280, 1200, 1)"},
        },
    ),
    (
        Reshape684,
        [((1, 512, 15, 20), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 512, 300, 1)"},
        },
    ),
    (
        Reshape685,
        [((1, 300, 512), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(300, 512)"},
        },
    ),
    (
        Reshape686,
        [((1, 300, 512), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 300, 8, 64)"},
        },
    ),
    (
        Reshape687,
        [((1, 300, 512), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 15, 20, 512)"},
        },
    ),
    (
        Reshape688,
        [((300, 512), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 300, 512)"},
        },
    ),
    (
        Reshape689,
        [((1, 8, 300, 64), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(8, 300, 64)"},
        },
    ),
    (
        Reshape690,
        [((8, 300, 300), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 8, 300, 300)"},
        },
    ),
    (
        Reshape691,
        [((1, 8, 300, 300), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(8, 300, 300)"},
        },
    ),
    (
        Reshape692,
        [((1, 8, 64, 300), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(8, 64, 300)"},
        },
    ),
    (
        Reshape693,
        [((8, 300, 64), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 8, 300, 64)"},
        },
    ),
    (
        Reshape685,
        [((1, 300, 8, 64), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(300, 512)"},
        },
    ),
    (
        Reshape694,
        [((1, 2048, 300), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 2048, 15, 20)"},
        },
    ),
    (
        Reshape695,
        [((1, 2048, 15, 20), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 2048, 300, 1)"},
        },
    ),
    (
        Reshape696,
        [((1, 1, 30, 40), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 30, 40)"},
        },
    ),
    (
        Reshape697,
        [((1, 1, 60, 80), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 60, 80)"},
        },
    ),
    (
        Reshape698,
        [((1, 1, 120, 160), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 120, 160)"},
        },
    ),
    (
        Reshape699,
        [((1, 3, 256, 256), torch.bfloat16)],
        {
            "model_names": ["pt_mlp_mixer_base_img_cls_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 3, 16, 16, 16, 16)"},
        },
    ),
    (
        Reshape231,
        [((1, 16, 16, 16, 16, 3), torch.bfloat16)],
        {
            "model_names": ["pt_mlp_mixer_base_img_cls_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(256, 768)"},
        },
    ),
    (
        Reshape700,
        [((1024, 256, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mlp_mixer_base_img_cls_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1024, 256, 1, 1)"},
        },
    ),
    (
        Reshape541,
        [((1, 1024, 512, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mlp_mixer_base_img_cls_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 1024, 512)"},
        },
    ),
    (
        Reshape701,
        [((1, 1024, 512), torch.bfloat16)],
        {
            "model_names": ["pt_mlp_mixer_base_img_cls_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 1024, 512, 1)"},
        },
    ),
    (
        Reshape702,
        [((256, 1024, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mlp_mixer_base_img_cls_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(256, 1024, 1, 1)"},
        },
    ),
    (
        Reshape147,
        [((1, 256, 512, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mlp_mixer_base_img_cls_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 256, 512)"},
        },
    ),
    (
        Reshape703,
        [((1, 512, 512), torch.bfloat16)],
        {
            "model_names": ["pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 512, 1, 512)"},
        },
    ),
    (
        Reshape704,
        [((1, 224, 224, 256), torch.bfloat16)],
        {
            "model_names": ["pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 50176, 256)"},
        },
    ),
    (
        Reshape705,
        [((1, 50176, 512), torch.bfloat16)],
        {
            "model_names": ["pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(50176, 512)"},
        },
    ),
    (
        Reshape706,
        [((1, 50176, 512), torch.bfloat16)],
        {
            "model_names": ["pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 50176, 1, 512)"},
        },
    ),
    (
        Reshape707,
        [((50176, 512), torch.bfloat16)],
        {
            "model_names": ["pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 50176, 512)"},
        },
    ),
    (
        Reshape708,
        [((1, 512, 50176), torch.bfloat16)],
        {
            "model_names": ["pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 1, 512, 50176)"},
        },
    ),
    (
        Reshape709,
        [((1, 1, 512, 50176), torch.bfloat16)],
        {
            "model_names": ["pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 512, 50176)"},
        },
    ),
    (
        Reshape710,
        [((1, 35, 1536), torch.float32)],
        {"model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf"], "pcc": 0.99, "args": {"shape": "(35, 1536)"}},
    ),
    (
        Reshape711,
        [((1, 35, 1536), torch.float32)],
        {
            "model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 35, 12, 128)"},
        },
    ),
    (
        Reshape712,
        [((35, 1536), torch.float32)],
        {
            "model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 35, 1536)"},
        },
    ),
    (
        Reshape713,
        [((1, 12, 35, 128), torch.float32)],
        {
            "model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(12, 35, 128)"},
        },
    ),
    (
        Reshape714,
        [((35, 256), torch.float32)],
        {
            "model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 35, 256)"},
        },
    ),
    (
        Reshape715,
        [((1, 35, 256), torch.float32)],
        {
            "model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 35, 2, 128)"},
        },
    ),
    (
        Reshape713,
        [((1, 2, 6, 35, 128), torch.float32)],
        {
            "model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(12, 35, 128)"},
        },
    ),
    (
        Reshape716,
        [((12, 35, 35), torch.float32)],
        {
            "model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 12, 35, 35)"},
        },
    ),
    (
        Reshape717,
        [((1, 12, 35, 35), torch.float32)],
        {
            "model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(12, 35, 35)"},
        },
    ),
    (
        Reshape718,
        [((12, 35, 128), torch.float32)],
        {
            "model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 12, 35, 128)"},
        },
    ),
    (
        Reshape710,
        [((1, 35, 12, 128), torch.float32)],
        {"model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf"], "pcc": 0.99, "args": {"shape": "(35, 1536)"}},
    ),
    (
        Reshape719,
        [((35, 8960), torch.float32)],
        {
            "model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 35, 8960)"},
        },
    ),
    (
        Reshape720,
        [((1, 440, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_400mf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 440, 1, 1)"},
        },
    ),
    (
        Reshape721,
        [((1, 16, 38, 38), torch.bfloat16)],
        {
            "model_names": ["pt_ssd300_resnet50_base_img_cls_torchhub"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 4, 5776)"},
        },
    ),
    (
        Reshape722,
        [((1, 24, 19, 19), torch.bfloat16)],
        {
            "model_names": ["pt_ssd300_resnet50_base_img_cls_torchhub"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 4, 2166)"},
        },
    ),
    (
        Reshape723,
        [((1, 24, 10, 10), torch.bfloat16)],
        {
            "model_names": ["pt_ssd300_resnet50_base_img_cls_torchhub"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 4, 600)"},
        },
    ),
    (
        Reshape724,
        [((1, 24, 5, 5), torch.bfloat16)],
        {
            "model_names": ["pt_ssd300_resnet50_base_img_cls_torchhub"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 4, 150)"},
        },
    ),
    (
        Reshape725,
        [((1, 16, 3, 3), torch.bfloat16)],
        {
            "model_names": ["pt_ssd300_resnet50_base_img_cls_torchhub"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 4, 36)"},
        },
    ),
    (
        Reshape726,
        [((1, 16, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_ssd300_resnet50_base_img_cls_torchhub"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 4, 4)"},
        },
    ),
    (
        Reshape727,
        [((1, 324, 38, 38), torch.bfloat16)],
        {
            "model_names": ["pt_ssd300_resnet50_base_img_cls_torchhub"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 81, 5776)"},
        },
    ),
    (
        Reshape728,
        [((1, 486, 19, 19), torch.bfloat16)],
        {
            "model_names": ["pt_ssd300_resnet50_base_img_cls_torchhub"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 81, 2166)"},
        },
    ),
    (
        Reshape729,
        [((1, 486, 10, 10), torch.bfloat16)],
        {
            "model_names": ["pt_ssd300_resnet50_base_img_cls_torchhub"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 81, 600)"},
        },
    ),
    (
        Reshape730,
        [((1, 486, 5, 5), torch.bfloat16)],
        {
            "model_names": ["pt_ssd300_resnet50_base_img_cls_torchhub"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 81, 150)"},
        },
    ),
    (
        Reshape731,
        [((1, 324, 3, 3), torch.bfloat16)],
        {
            "model_names": ["pt_ssd300_resnet50_base_img_cls_torchhub"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 81, 36)"},
        },
    ),
    (
        Reshape732,
        [((1, 324, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_ssd300_resnet50_base_img_cls_torchhub"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 81, 4)"},
        },
    ),
    (
        Reshape733,
        [((1, 56, 56, 96), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 8, 7, 8, 7, 96)"},
        },
    ),
    (
        Reshape734,
        [((1, 56, 56, 96), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(3136, 96)"},
        },
    ),
    (
        Reshape734,
        [((1, 8, 8, 7, 7, 96), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(3136, 96)"},
        },
    ),
    (
        Reshape735,
        [((3136, 288), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(64, 49, 288)"},
        },
    ),
    (
        Reshape736,
        [((64, 49, 288), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(64, 49, 3, 3, 32)"},
        },
    ),
    (
        Reshape737,
        [((1, 64, 3, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(64, 3, 49, 32)"},
        },
    ),
    (
        Reshape738,
        [((1, 64, 3, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(192, 49, 32)"},
        },
    ),
    (
        Reshape738,
        [((64, 3, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(192, 49, 32)"},
        },
    ),
    (
        Reshape739,
        [((192, 49, 49), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(64, 3, 49, 49)"},
        },
    ),
    (
        Reshape740,
        [((2401, 3), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(49, 49, 3)"},
        },
    ),
    (
        Reshape741,
        [((64, 3, 49, 49), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(192, 49, 49)"},
        },
    ),
    (
        Reshape742,
        [((64, 3, 49, 49), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 64, 3, 49, 49)"},
        },
    ),
    (
        Reshape743,
        [((64, 3, 32, 49), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(192, 32, 49)"},
        },
    ),
    (
        Reshape737,
        [((192, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(64, 3, 49, 32)"},
        },
    ),
    (
        Reshape734,
        [((64, 49, 3, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(3136, 96)"},
        },
    ),
    (
        Reshape744,
        [((3136, 96), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(64, 49, 96)"},
        },
    ),
    (
        Reshape745,
        [((3136, 96), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 56, 56, 96)"},
        },
    ),
    (
        Reshape746,
        [((64, 49, 96), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 8, 8, 7, 7, 96)"},
        },
    ),
    (
        Reshape745,
        [((1, 8, 7, 8, 7, 96), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 56, 56, 96)"},
        },
    ),
    (
        Reshape747,
        [((3136, 384), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 56, 56, 384)"},
        },
    ),
    (
        Reshape748,
        [((3136, 384), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(64, 49, 384)"},
        },
    ),
    (
        Reshape749,
        [((1, 56, 56, 384), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(3136, 384)"},
        },
    ),
    (
        Reshape739,
        [((1, 64, 3, 49, 49), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(64, 3, 49, 49)"},
        },
    ),
    (
        Reshape750,
        [((1, 28, 28, 384), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(784, 384)"},
        },
    ),
    (
        Reshape751,
        [((784, 192), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 28, 28, 192)"},
        },
    ),
    (
        Reshape752,
        [((784, 192), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(16, 49, 192)"},
        },
    ),
    (
        Reshape753,
        [((1, 28, 28, 192), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 4, 7, 4, 7, 192)"},
        },
    ),
    (
        Reshape754,
        [((1, 28, 28, 192), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(784, 192)"},
        },
    ),
    (
        Reshape754,
        [((1, 4, 4, 7, 7, 192), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(784, 192)"},
        },
    ),
    (
        Reshape755,
        [((784, 576), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(16, 49, 576)"},
        },
    ),
    (
        Reshape756,
        [((16, 49, 576), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(16, 49, 3, 6, 32)"},
        },
    ),
    (
        Reshape757,
        [((1, 16, 6, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(16, 6, 49, 32)"},
        },
    ),
    (
        Reshape758,
        [((1, 16, 6, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(96, 49, 32)"},
        },
    ),
    (
        Reshape758,
        [((16, 6, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(96, 49, 32)"},
        },
    ),
    (
        Reshape759,
        [((96, 49, 49), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(16, 6, 49, 49)"},
        },
    ),
    (
        Reshape760,
        [((2401, 6), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(49, 49, 6)"},
        },
    ),
    (
        Reshape761,
        [((16, 6, 49, 49), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(96, 49, 49)"},
        },
    ),
    (
        Reshape762,
        [((16, 6, 49, 49), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 16, 6, 49, 49)"},
        },
    ),
    (
        Reshape763,
        [((16, 6, 32, 49), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(96, 32, 49)"},
        },
    ),
    (
        Reshape757,
        [((96, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(16, 6, 49, 32)"},
        },
    ),
    (
        Reshape754,
        [((16, 49, 6, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(784, 192)"},
        },
    ),
    (
        Reshape764,
        [((16, 49, 192), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 4, 4, 7, 7, 192)"},
        },
    ),
    (
        Reshape751,
        [((1, 4, 7, 4, 7, 192), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 28, 28, 192)"},
        },
    ),
    (
        Reshape765,
        [((784, 768), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 28, 28, 768)"},
        },
    ),
    (
        Reshape766,
        [((784, 768), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(16, 49, 768)"},
        },
    ),
    (
        Reshape767,
        [((1, 28, 28, 768), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(784, 768)"},
        },
    ),
    (
        Reshape759,
        [((1, 16, 6, 49, 49), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(16, 6, 49, 49)"},
        },
    ),
    (
        Reshape768,
        [((1, 14, 14, 768), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(196, 768)"},
        },
    ),
    (
        Reshape769,
        [((196, 384), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 14, 14, 384)"},
        },
    ),
    (
        Reshape770,
        [((196, 384), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(4, 49, 384)"},
        },
    ),
    (
        Reshape771,
        [((1, 14, 14, 384), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 2, 7, 2, 7, 384)"},
        },
    ),
    (
        Reshape772,
        [((1, 14, 14, 384), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(196, 384)"},
        },
    ),
    (
        Reshape772,
        [((1, 2, 2, 7, 7, 384), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(196, 384)"},
        },
    ),
    (
        Reshape773,
        [((196, 1152), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(4, 49, 1152)"},
        },
    ),
    (
        Reshape774,
        [((4, 49, 1152), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(4, 49, 3, 12, 32)"},
        },
    ),
    (
        Reshape775,
        [((1, 4, 12, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(4, 12, 49, 32)"},
        },
    ),
    (
        Reshape776,
        [((1, 4, 12, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(48, 49, 32)"},
        },
    ),
    (
        Reshape776,
        [((4, 12, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(48, 49, 32)"},
        },
    ),
    (
        Reshape777,
        [((48, 49, 49), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(4, 12, 49, 49)"},
        },
    ),
    (
        Reshape778,
        [((2401, 12), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(49, 49, 12)"},
        },
    ),
    (
        Reshape779,
        [((4, 12, 49, 49), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(48, 49, 49)"},
        },
    ),
    (
        Reshape780,
        [((4, 12, 49, 49), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 4, 12, 49, 49)"},
        },
    ),
    (
        Reshape781,
        [((4, 12, 32, 49), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(48, 32, 49)"},
        },
    ),
    (
        Reshape775,
        [((48, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(4, 12, 49, 32)"},
        },
    ),
    (
        Reshape772,
        [((4, 49, 12, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(196, 384)"},
        },
    ),
    (
        Reshape782,
        [((4, 49, 384), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 2, 2, 7, 7, 384)"},
        },
    ),
    (
        Reshape769,
        [((1, 2, 7, 2, 7, 384), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 14, 14, 384)"},
        },
    ),
    (
        Reshape783,
        [((196, 1536), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 14, 14, 1536)"},
        },
    ),
    (
        Reshape784,
        [((196, 1536), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(4, 49, 1536)"},
        },
    ),
    (
        Reshape785,
        [((1, 14, 14, 1536), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(196, 1536)"},
        },
    ),
    (
        Reshape777,
        [((1, 4, 12, 49, 49), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(4, 12, 49, 49)"},
        },
    ),
    (
        Reshape786,
        [((1, 7, 7, 1536), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(49, 1536)"},
        },
    ),
    (
        Reshape787,
        [((49, 768), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 7, 7, 768)"},
        },
    ),
    (
        Reshape788,
        [((49, 768), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 49, 768)"},
        },
    ),
    (
        Reshape789,
        [((1, 7, 7, 768), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 1, 7, 1, 7, 768)"},
        },
    ),
    (
        Reshape790,
        [((1, 7, 7, 768), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(49, 768)"},
        },
    ),
    (
        Reshape790,
        [((1, 1, 1, 7, 7, 768), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(49, 768)"},
        },
    ),
    (
        Reshape791,
        [((49, 2304), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 49, 2304)"},
        },
    ),
    (
        Reshape792,
        [((1, 49, 2304), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 49, 3, 24, 32)"},
        },
    ),
    (
        Reshape793,
        [((1, 1, 24, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 24, 49, 32)"},
        },
    ),
    (
        Reshape794,
        [((1, 1, 24, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(24, 49, 32)"},
        },
    ),
    (
        Reshape794,
        [((1, 24, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(24, 49, 32)"},
        },
    ),
    (
        Reshape795,
        [((24, 49, 49), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 24, 49, 49)"},
        },
    ),
    (
        Reshape796,
        [((2401, 24), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(49, 49, 24)"},
        },
    ),
    (
        Reshape797,
        [((1, 24, 49, 49), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(24, 49, 49)"},
        },
    ),
    (
        Reshape798,
        [((1, 24, 32, 49), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(24, 32, 49)"},
        },
    ),
    (
        Reshape793,
        [((24, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 24, 49, 32)"},
        },
    ),
    (
        Reshape790,
        [((1, 49, 24, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(49, 768)"},
        },
    ),
    (
        Reshape799,
        [((1, 49, 768), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 1, 1, 7, 7, 768)"},
        },
    ),
    (
        Reshape787,
        [((1, 1, 7, 1, 7, 768), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 7, 7, 768)"},
        },
    ),
    (
        Reshape800,
        [((49, 3072), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 7, 7, 3072)"},
        },
    ),
    (
        Reshape801,
        [((49, 3072), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 49, 3072)"},
        },
    ),
    (
        Reshape802,
        [((1, 7, 7, 3072), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(49, 3072)"},
        },
    ),
    (
        Reshape803,
        [((1, 768, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 768, 1, 1)"},
        },
    ),
    (
        Reshape804,
        [((1, 384), torch.float32)],
        {
            "model_names": [
                "pt_t5_google_flan_t5_small_text_gen_hf",
                "pt_whisper_openai_whisper_tiny_speech_recognition_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 1, 6, 64)"},
        },
    ),
    (
        Reshape805,
        [((1, 384), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 1, 384)"},
        },
    ),
    (
        Reshape806,
        [((1, 6, 1, 64), torch.float32)],
        {
            "model_names": [
                "pt_t5_google_flan_t5_small_text_gen_hf",
                "pt_whisper_openai_whisper_tiny_speech_recognition_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(6, 1, 64)"},
        },
    ),
    (
        Reshape807,
        [((6, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_t5_google_flan_t5_small_text_gen_hf",
                "pt_whisper_openai_whisper_tiny_speech_recognition_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 6, 1, 1)"},
        },
    ),
    (
        Reshape808,
        [((1, 6, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_t5_google_flan_t5_small_text_gen_hf",
                "pt_whisper_openai_whisper_tiny_speech_recognition_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(6, 1, 1)"},
        },
    ),
    (
        Reshape809,
        [((1, 6, 64, 1), torch.float32)],
        {"model_names": ["pt_t5_google_flan_t5_small_text_gen_hf"], "pcc": 0.99, "args": {"shape": "(6, 64, 1)"}},
    ),
    (
        Reshape810,
        [((6, 1, 64), torch.float32)],
        {
            "model_names": [
                "pt_t5_google_flan_t5_small_text_gen_hf",
                "pt_whisper_openai_whisper_tiny_speech_recognition_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 6, 1, 64)"},
        },
    ),
    (
        Reshape476,
        [((1, 1, 6, 64), torch.float32)],
        {
            "model_names": [
                "pt_t5_google_flan_t5_small_text_gen_hf",
                "pt_whisper_openai_whisper_tiny_speech_recognition_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 384)"},
        },
    ),
    (
        Reshape811,
        [((1, 61, 512), torch.float32)],
        {
            "model_names": ["pt_t5_google_flan_t5_small_text_gen_hf", "pt_t5_t5_small_text_gen_hf"],
            "pcc": 0.99,
            "args": {"shape": "(61, 512)"},
        },
    ),
    (
        Reshape812,
        [((61, 384), torch.float32)],
        {"model_names": ["pt_t5_google_flan_t5_small_text_gen_hf"], "pcc": 0.99, "args": {"shape": "(1, 61, 6, 64)"}},
    ),
    (
        Reshape813,
        [((1, 6, 61, 64), torch.float32)],
        {"model_names": ["pt_t5_google_flan_t5_small_text_gen_hf"], "pcc": 0.99, "args": {"shape": "(6, 61, 64)"}},
    ),
    (
        Reshape814,
        [((6, 61, 61), torch.float32)],
        {"model_names": ["pt_t5_google_flan_t5_small_text_gen_hf"], "pcc": 0.99, "args": {"shape": "(1, 6, 61, 61)"}},
    ),
    (
        Reshape815,
        [((1, 6, 61, 61), torch.float32)],
        {"model_names": ["pt_t5_google_flan_t5_small_text_gen_hf"], "pcc": 0.99, "args": {"shape": "(6, 61, 61)"}},
    ),
    (
        Reshape816,
        [((1, 6, 64, 61), torch.float32)],
        {"model_names": ["pt_t5_google_flan_t5_small_text_gen_hf"], "pcc": 0.99, "args": {"shape": "(6, 64, 61)"}},
    ),
    (
        Reshape817,
        [((6, 61, 64), torch.float32)],
        {"model_names": ["pt_t5_google_flan_t5_small_text_gen_hf"], "pcc": 0.99, "args": {"shape": "(1, 6, 61, 64)"}},
    ),
    (
        Reshape818,
        [((1, 61, 6, 64), torch.float32)],
        {"model_names": ["pt_t5_google_flan_t5_small_text_gen_hf"], "pcc": 0.99, "args": {"shape": "(61, 384)"}},
    ),
    (
        Reshape819,
        [((61, 512), torch.float32)],
        {
            "model_names": ["pt_t5_google_flan_t5_small_text_gen_hf", "pt_t5_t5_small_text_gen_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 61, 512)"},
        },
    ),
    (
        Reshape820,
        [((61, 512), torch.float32)],
        {"model_names": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99, "args": {"shape": "(1, 61, 8, 64)"}},
    ),
    (
        Reshape821,
        [((61, 1024), torch.float32)],
        {"model_names": ["pt_t5_google_flan_t5_small_text_gen_hf"], "pcc": 0.99, "args": {"shape": "(1, 61, 1024)"}},
    ),
    (
        Reshape822,
        [((6, 1, 61), torch.float32)],
        {"model_names": ["pt_t5_google_flan_t5_small_text_gen_hf"], "pcc": 0.99, "args": {"shape": "(1, 6, 1, 61)"}},
    ),
    (
        Reshape823,
        [((1, 6, 1, 61), torch.float32)],
        {"model_names": ["pt_t5_google_flan_t5_small_text_gen_hf"], "pcc": 0.99, "args": {"shape": "(6, 1, 61)"}},
    ),
    (
        Reshape824,
        [((1, 1024), torch.float32)],
        {"model_names": ["pt_t5_google_flan_t5_small_text_gen_hf"], "pcc": 0.99, "args": {"shape": "(1, 1, 1024)"}},
    ),
    (
        Reshape825,
        [((1, 3, 85, 80, 80), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5n_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_1280x1280",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_640x640",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 255, 80, 80)"},
        },
    ),
    (
        Reshape826,
        [((1, 255, 80, 80), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5n_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_1280x1280",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_640x640",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 1, 255, 6400)"},
        },
    ),
    (
        Reshape827,
        [((1, 1, 255, 6400), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5n_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_1280x1280",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_640x640",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 3, 85, 6400)"},
        },
    ),
    (
        Reshape828,
        [((1, 3, 6400, 85), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5n_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_1280x1280",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_640x640",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 19200, 85)"},
        },
    ),
    (
        Reshape829,
        [((1, 3, 85, 60, 60), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5s_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_480x480",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 255, 60, 60)"},
        },
    ),
    (
        Reshape830,
        [((1, 255, 60, 60), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5s_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_480x480",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 1, 255, 3600)"},
        },
    ),
    (
        Reshape831,
        [((1, 1, 255, 3600), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5s_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_480x480",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 3, 85, 3600)"},
        },
    ),
    (
        Reshape832,
        [((1, 3, 3600, 85), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5s_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_480x480",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 10800, 85)"},
        },
    ),
    (
        Reshape833,
        [((1, 3, 85, 30, 30), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5s_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_480x480",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 255, 30, 30)"},
        },
    ),
    (
        Reshape834,
        [((1, 255, 30, 30), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5s_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_480x480",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 1, 255, 900)"},
        },
    ),
    (
        Reshape835,
        [((1, 1, 255, 900), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5s_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_480x480",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 3, 85, 900)"},
        },
    ),
    (
        Reshape836,
        [((1, 3, 900, 85), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5s_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_480x480",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 2700, 85)"},
        },
    ),
    (
        Reshape837,
        [((1, 3, 85, 15, 15), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5s_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_480x480",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 255, 15, 15)"},
        },
    ),
    (
        Reshape838,
        [((1, 255, 15, 15), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5s_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_480x480",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 1, 255, 225)"},
        },
    ),
    (
        Reshape839,
        [((1, 1, 255, 225), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5s_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_480x480",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 3, 85, 225)"},
        },
    ),
    (
        Reshape840,
        [((1, 3, 225, 85), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5s_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_480x480",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 675, 85)"},
        },
    ),
    (
        Reshape89,
        [((1000,), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99, "args": {"shape": "(1, 1000)"}},
    ),
    (
        Reshape841,
        [((1, 1536, 1, 1), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b3_img_cls_timm"],
            "pcc": 0.99,
            "args": {"shape": "(1, 1536)"},
        },
    ),
    (
        Reshape842,
        [((1, 13, 384), torch.float32)],
        {
            "model_names": ["onnx_minilm_sentence_transformers_all_minilm_l6_v2_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(13, 384)"},
        },
    ),
    (
        Reshape843,
        [((1, 13, 384), torch.float32)],
        {
            "model_names": ["onnx_minilm_sentence_transformers_all_minilm_l6_v2_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 13, 12, 32)"},
        },
    ),
    (
        Reshape844,
        [((13, 384), torch.float32)],
        {
            "model_names": ["onnx_minilm_sentence_transformers_all_minilm_l6_v2_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 13, 384)"},
        },
    ),
    (
        Reshape845,
        [((1, 12, 13, 32), torch.float32)],
        {
            "model_names": ["onnx_minilm_sentence_transformers_all_minilm_l6_v2_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(12, 13, 32)"},
        },
    ),
    (
        Reshape846,
        [((1, 12, 32, 13), torch.float32)],
        {
            "model_names": ["onnx_minilm_sentence_transformers_all_minilm_l6_v2_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(12, 32, 13)"},
        },
    ),
    (
        Reshape847,
        [((12, 13, 13), torch.float32)],
        {
            "model_names": ["onnx_minilm_sentence_transformers_all_minilm_l6_v2_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 12, 13, 13)"},
        },
    ),
    (
        Reshape848,
        [((1, 12, 13, 13), torch.float32)],
        {
            "model_names": ["onnx_minilm_sentence_transformers_all_minilm_l6_v2_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(12, 13, 13)"},
        },
    ),
    (
        Reshape849,
        [((12, 13, 32), torch.float32)],
        {
            "model_names": ["onnx_minilm_sentence_transformers_all_minilm_l6_v2_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 12, 13, 32)"},
        },
    ),
    (
        Reshape842,
        [((1, 13, 12, 32), torch.float32)],
        {
            "model_names": ["onnx_minilm_sentence_transformers_all_minilm_l6_v2_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(13, 384)"},
        },
    ),
    (
        Reshape476,
        [((1, 1, 384), torch.float32)],
        {
            "model_names": [
                "onnx_minilm_sentence_transformers_all_minilm_l6_v2_seq_cls_hf",
                "pt_whisper_openai_whisper_tiny_speech_recognition_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 384)"},
        },
    ),
    (
        Reshape804,
        [((1, 1, 384), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 1, 6, 64)"},
        },
    ),
    (
        Reshape850,
        [((1, 14, 768), torch.float32)],
        {"model_names": ["pt_albert_twmkn9_albert_base_v2_squad2_qa_hf"], "pcc": 0.99, "args": {"shape": "(14, 768)"}},
    ),
    (
        Reshape851,
        [((1, 14, 768), torch.float32)],
        {
            "model_names": ["pt_albert_twmkn9_albert_base_v2_squad2_qa_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 14, 12, 64)"},
        },
    ),
    (
        Reshape852,
        [((14, 768), torch.float32)],
        {
            "model_names": ["pt_albert_twmkn9_albert_base_v2_squad2_qa_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 14, 768)"},
        },
    ),
    (
        Reshape853,
        [((1, 12, 14, 64), torch.float32)],
        {
            "model_names": ["pt_albert_twmkn9_albert_base_v2_squad2_qa_hf"],
            "pcc": 0.99,
            "args": {"shape": "(12, 14, 64)"},
        },
    ),
    (
        Reshape854,
        [((12, 14, 14), torch.float32)],
        {
            "model_names": ["pt_albert_twmkn9_albert_base_v2_squad2_qa_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 12, 14, 14)"},
        },
    ),
    (
        Reshape855,
        [((1, 12, 14, 14), torch.float32)],
        {
            "model_names": ["pt_albert_twmkn9_albert_base_v2_squad2_qa_hf"],
            "pcc": 0.99,
            "args": {"shape": "(12, 14, 14)"},
        },
    ),
    (
        Reshape856,
        [((12, 14, 64), torch.float32)],
        {
            "model_names": ["pt_albert_twmkn9_albert_base_v2_squad2_qa_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 12, 14, 64)"},
        },
    ),
    (
        Reshape850,
        [((1, 14, 12, 64), torch.float32)],
        {"model_names": ["pt_albert_twmkn9_albert_base_v2_squad2_qa_hf"], "pcc": 0.99, "args": {"shape": "(14, 768)"}},
    ),
    (
        Reshape857,
        [((14, 1), torch.float32)],
        {"model_names": ["pt_albert_twmkn9_albert_base_v2_squad2_qa_hf"], "pcc": 0.99, "args": {"shape": "(1, 14, 1)"}},
    ),
    (
        Reshape858,
        [((1, 768, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_mlp_mixer_mixer_b32_224_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 768, 49, 1)"},
        },
    ),
    (
        Reshape859,
        [((1, 768, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_b_32_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 768, 49)"},
        },
    ),
    (
        Reshape860,
        [((1, 64, 12, 12), torch.bfloat16)],
        {
            "model_names": ["pt_mnist_base_img_cls_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 9216, 1, 1)"},
        },
    ),
    (
        Reshape861,
        [((1, 256, 2048), torch.float32)],
        {
            "model_names": [
                "pt_opt_facebook_opt_1_3b_clm_hf",
                "pt_xglm_facebook_xglm_1_7b_clm_hf",
                "pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_clm_hf",
                "pt_phi1_microsoft_phi_1_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(256, 2048)"},
        },
    ),
    (
        Reshape862,
        [((1, 256, 2048), torch.float32)],
        {
            "model_names": ["pt_opt_facebook_opt_1_3b_clm_hf", "pt_phi1_microsoft_phi_1_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 256, 32, 64)"},
        },
    ),
    (
        Reshape863,
        [((1, 256, 2048), torch.float32)],
        {"model_names": ["pt_xglm_facebook_xglm_1_7b_clm_hf"], "pcc": 0.99, "args": {"shape": "(1, 256, 16, 128)"}},
    ),
    (
        Reshape864,
        [((256, 2048), torch.float32)],
        {
            "model_names": [
                "pt_opt_facebook_opt_1_3b_clm_hf",
                "pt_xglm_facebook_xglm_1_7b_clm_hf",
                "pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_clm_hf",
                "pt_phi1_microsoft_phi_1_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 256, 2048)"},
        },
    ),
    (
        Reshape863,
        [((256, 2048), torch.float32)],
        {
            "model_names": ["pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 256, 16, 128)"},
        },
    ),
    (
        Reshape862,
        [((256, 2048), torch.float32)],
        {
            "model_names": ["pt_llama3_meta_llama_llama_3_2_1b_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 256, 32, 64)"},
        },
    ),
    (
        Reshape865,
        [((256, 2048), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(1, 16, 16, 2048)"}},
    ),
    (
        Reshape866,
        [((1, 32, 256, 64), torch.float32)],
        {
            "model_names": [
                "pt_opt_facebook_opt_1_3b_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_clm_hf",
                "pt_phi1_microsoft_phi_1_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(32, 256, 64)"},
        },
    ),
    (
        Reshape867,
        [((32, 256, 64), torch.float32)],
        {
            "model_names": [
                "pt_opt_facebook_opt_1_3b_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_clm_hf",
                "pt_phi1_microsoft_phi_1_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 32, 256, 64)"},
        },
    ),
    (
        Reshape861,
        [((1, 256, 32, 64), torch.float32)],
        {
            "model_names": [
                "pt_opt_facebook_opt_1_3b_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_clm_hf",
                "pt_phi1_microsoft_phi_1_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(256, 2048)"},
        },
    ),
    (
        Reshape868,
        [((1, 32, 2048), torch.float32)],
        {
            "model_names": ["pt_opt_facebook_opt_1_3b_qa_hf", "pt_opt_facebook_opt_1_3b_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(32, 2048)"},
        },
    ),
    (
        Reshape869,
        [((1, 32, 2048), torch.float32)],
        {
            "model_names": ["pt_opt_facebook_opt_1_3b_qa_hf", "pt_opt_facebook_opt_1_3b_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 32, 32, 64)"},
        },
    ),
    (
        Reshape870,
        [((32, 2048), torch.float32)],
        {
            "model_names": ["pt_opt_facebook_opt_1_3b_qa_hf", "pt_opt_facebook_opt_1_3b_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 32, 2048)"},
        },
    ),
    (
        Reshape871,
        [((1, 32, 32, 64), torch.float32)],
        {
            "model_names": [
                "pt_opt_facebook_opt_1_3b_qa_hf",
                "pt_opt_facebook_opt_1_3b_seq_cls_hf",
                "pt_swin_swin_v2_b_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(32, 32, 64)"},
        },
    ),
    (
        Reshape868,
        [((1, 32, 32, 64), torch.float32)],
        {
            "model_names": ["pt_opt_facebook_opt_1_3b_qa_hf", "pt_opt_facebook_opt_1_3b_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(32, 2048)"},
        },
    ),
    (
        Reshape872,
        [((32, 32, 32), torch.float32)],
        {
            "model_names": ["pt_opt_facebook_opt_1_3b_qa_hf", "pt_opt_facebook_opt_1_3b_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 32, 32, 32)"},
        },
    ),
    (
        Reshape873,
        [((1, 32, 32, 32), torch.float32)],
        {
            "model_names": ["pt_opt_facebook_opt_1_3b_qa_hf", "pt_opt_facebook_opt_1_3b_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(32, 32, 32)"},
        },
    ),
    (
        Reshape869,
        [((32, 32, 64), torch.float32)],
        {
            "model_names": ["pt_opt_facebook_opt_1_3b_qa_hf", "pt_opt_facebook_opt_1_3b_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 32, 32, 64)"},
        },
    ),
    (
        Reshape874,
        [((1, 2240, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_facebook_regnet_y_120_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 2240, 1, 1)"},
        },
    ),
    (
        Reshape875,
        [((1, 3024, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_16gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 3024, 1, 1)"},
        },
    ),
    (
        Reshape876,
        [((1, 56, 56, 128), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 8, 7, 8, 7, 128)"},
        },
    ),
    (
        Reshape877,
        [((1, 56, 56, 128), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(3136, 128)"},
        },
    ),
    (
        Reshape877,
        [((1, 8, 8, 7, 7, 128), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(3136, 128)"},
        },
    ),
    (
        Reshape878,
        [((64, 49, 384), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(64, 49, 3, 4, 32)"},
        },
    ),
    (
        Reshape879,
        [((1, 64, 4, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(64, 4, 49, 32)"},
        },
    ),
    (
        Reshape880,
        [((1, 64, 4, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(256, 49, 32)"},
        },
    ),
    (
        Reshape880,
        [((64, 4, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(256, 49, 32)"},
        },
    ),
    (
        Reshape881,
        [((256, 49, 49), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(64, 4, 49, 49)"},
        },
    ),
    (
        Reshape882,
        [((2401, 4), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(49, 49, 4)"},
        },
    ),
    (
        Reshape883,
        [((64, 4, 49, 49), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(256, 49, 49)"},
        },
    ),
    (
        Reshape884,
        [((64, 4, 49, 49), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 64, 4, 49, 49)"},
        },
    ),
    (
        Reshape885,
        [((64, 4, 32, 49), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(256, 32, 49)"},
        },
    ),
    (
        Reshape879,
        [((256, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(64, 4, 49, 32)"},
        },
    ),
    (
        Reshape877,
        [((64, 49, 4, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(3136, 128)"},
        },
    ),
    (
        Reshape886,
        [((3136, 128), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(64, 49, 128)"},
        },
    ),
    (
        Reshape887,
        [((3136, 128), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 56, 56, 128)"},
        },
    ),
    (
        Reshape888,
        [((64, 49, 128), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 8, 8, 7, 7, 128)"},
        },
    ),
    (
        Reshape887,
        [((1, 8, 7, 8, 7, 128), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 56, 56, 128)"},
        },
    ),
    (
        Reshape889,
        [((3136, 512), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 56, 56, 512)"},
        },
    ),
    (
        Reshape890,
        [((1, 56, 56, 512), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(3136, 512)"},
        },
    ),
    (
        Reshape881,
        [((1, 64, 4, 49, 49), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(64, 4, 49, 49)"},
        },
    ),
    (
        Reshape891,
        [((1, 28, 28, 512), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(784, 512)"},
        },
    ),
    (
        Reshape892,
        [((784, 256), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 28, 28, 256)"},
        },
    ),
    (
        Reshape893,
        [((784, 256), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(16, 49, 256)"},
        },
    ),
    (
        Reshape894,
        [((1, 28, 28, 256), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 4, 7, 4, 7, 256)"},
        },
    ),
    (
        Reshape895,
        [((1, 28, 28, 256), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(784, 256)"},
        },
    ),
    (
        Reshape895,
        [((1, 4, 4, 7, 7, 256), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(784, 256)"},
        },
    ),
    (
        Reshape896,
        [((16, 49, 768), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(16, 49, 3, 8, 32)"},
        },
    ),
    (
        Reshape897,
        [((1, 16, 8, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(16, 8, 49, 32)"},
        },
    ),
    (
        Reshape898,
        [((1, 16, 8, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(128, 49, 32)"},
        },
    ),
    (
        Reshape898,
        [((16, 8, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(128, 49, 32)"},
        },
    ),
    (
        Reshape899,
        [((128, 49, 49), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(16, 8, 49, 49)"},
        },
    ),
    (
        Reshape900,
        [((2401, 8), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(49, 49, 8)"},
        },
    ),
    (
        Reshape901,
        [((16, 8, 49, 49), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(128, 49, 49)"},
        },
    ),
    (
        Reshape902,
        [((16, 8, 49, 49), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 16, 8, 49, 49)"},
        },
    ),
    (
        Reshape903,
        [((16, 8, 32, 49), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(128, 32, 49)"},
        },
    ),
    (
        Reshape897,
        [((128, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(16, 8, 49, 32)"},
        },
    ),
    (
        Reshape895,
        [((16, 49, 8, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(784, 256)"},
        },
    ),
    (
        Reshape904,
        [((16, 49, 256), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 4, 4, 7, 7, 256)"},
        },
    ),
    (
        Reshape892,
        [((1, 4, 7, 4, 7, 256), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 28, 28, 256)"},
        },
    ),
    (
        Reshape905,
        [((784, 1024), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 28, 28, 1024)"},
        },
    ),
    (
        Reshape906,
        [((1, 28, 28, 1024), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(784, 1024)"},
        },
    ),
    (
        Reshape899,
        [((1, 16, 8, 49, 49), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(16, 8, 49, 49)"},
        },
    ),
    (
        Reshape907,
        [((1, 14, 14, 1024), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(196, 1024)"},
        },
    ),
    (
        Reshape908,
        [((196, 512), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 14, 14, 512)"},
        },
    ),
    (
        Reshape909,
        [((196, 512), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(4, 49, 512)"},
        },
    ),
    (
        Reshape910,
        [((1, 14, 14, 512), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 2, 7, 2, 7, 512)"},
        },
    ),
    (
        Reshape911,
        [((1, 14, 14, 512), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(196, 512)"},
        },
    ),
    (
        Reshape911,
        [((1, 2, 2, 7, 7, 512), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(196, 512)"},
        },
    ),
    (
        Reshape912,
        [((4, 49, 1536), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(4, 49, 3, 16, 32)"},
        },
    ),
    (
        Reshape913,
        [((1, 4, 16, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(4, 16, 49, 32)"},
        },
    ),
    (
        Reshape914,
        [((1, 4, 16, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(64, 49, 32)"},
        },
    ),
    (
        Reshape914,
        [((4, 16, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(64, 49, 32)"},
        },
    ),
    (
        Reshape915,
        [((64, 49, 49), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(4, 16, 49, 49)"},
        },
    ),
    (
        Reshape916,
        [((2401, 16), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(49, 49, 16)"},
        },
    ),
    (
        Reshape917,
        [((4, 16, 49, 49), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(64, 49, 49)"},
        },
    ),
    (
        Reshape918,
        [((4, 16, 49, 49), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 4, 16, 49, 49)"},
        },
    ),
    (
        Reshape919,
        [((4, 16, 32, 49), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(64, 32, 49)"},
        },
    ),
    (
        Reshape913,
        [((64, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(4, 16, 49, 32)"},
        },
    ),
    (
        Reshape911,
        [((4, 49, 16, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(196, 512)"},
        },
    ),
    (
        Reshape920,
        [((4, 49, 512), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 2, 2, 7, 7, 512)"},
        },
    ),
    (
        Reshape908,
        [((1, 2, 7, 2, 7, 512), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 14, 14, 512)"},
        },
    ),
    (
        Reshape921,
        [((196, 2048), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 14, 14, 2048)"},
        },
    ),
    (
        Reshape922,
        [((1, 14, 14, 2048), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(196, 2048)"},
        },
    ),
    (
        Reshape915,
        [((1, 4, 16, 49, 49), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(4, 16, 49, 49)"},
        },
    ),
    (
        Reshape923,
        [((1, 7, 7, 2048), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(49, 2048)"},
        },
    ),
    (
        Reshape924,
        [((49, 1024), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 7, 7, 1024)"},
        },
    ),
    (
        Reshape925,
        [((49, 1024), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 49, 1024)"},
        },
    ),
    (
        Reshape926,
        [((1, 7, 7, 1024), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 1, 7, 1, 7, 1024)"},
        },
    ),
    (
        Reshape927,
        [((1, 7, 7, 1024), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(49, 1024)"},
        },
    ),
    (
        Reshape927,
        [((1, 1, 1, 7, 7, 1024), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(49, 1024)"},
        },
    ),
    (
        Reshape928,
        [((1, 49, 3072), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 49, 3, 32, 32)"},
        },
    ),
    (
        Reshape929,
        [((1, 1, 32, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 32, 49, 32)"},
        },
    ),
    (
        Reshape930,
        [((1, 1, 32, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(32, 49, 32)"},
        },
    ),
    (
        Reshape930,
        [((1, 32, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(32, 49, 32)"},
        },
    ),
    (
        Reshape931,
        [((32, 49, 49), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 32, 49, 49)"},
        },
    ),
    (
        Reshape932,
        [((2401, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(49, 49, 32)"},
        },
    ),
    (
        Reshape933,
        [((1, 32, 49, 49), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(32, 49, 49)"},
        },
    ),
    (
        Reshape934,
        [((1, 32, 32, 49), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(32, 32, 49)"},
        },
    ),
    (
        Reshape929,
        [((32, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 32, 49, 32)"},
        },
    ),
    (
        Reshape927,
        [((1, 49, 32, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(49, 1024)"},
        },
    ),
    (
        Reshape935,
        [((1, 49, 1024), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 1, 1, 7, 7, 1024)"},
        },
    ),
    (
        Reshape924,
        [((1, 1, 7, 1, 7, 1024), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 7, 7, 1024)"},
        },
    ),
    (
        Reshape936,
        [((49, 4096), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 7, 7, 4096)"},
        },
    ),
    (
        Reshape937,
        [((1, 7, 7, 4096), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(49, 4096)"},
        },
    ),
    (
        Reshape938,
        [((160, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": ["pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(160, 1, 3, 3)"},
        },
    ),
    (
        Reshape939,
        [((224, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": ["pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(224, 1, 3, 3)"},
        },
    ),
    (
        Reshape940,
        [((384, 80, 3), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(384, 80, 3, 1)"},
        },
    ),
    (
        Reshape941,
        [((1, 384, 3000, 1), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 384, 3000)"},
        },
    ),
    (
        Reshape942,
        [((1, 384, 3000), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 384, 3000, 1)"},
        },
    ),
    (
        Reshape943,
        [((384, 384, 3), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(384, 384, 3, 1)"},
        },
    ),
    (
        Reshape944,
        [((1, 384, 1500, 1), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 384, 1500)"},
        },
    ),
    (
        Reshape945,
        [((1, 1500, 384), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1500, 384)"},
        },
    ),
    (
        Reshape946,
        [((1, 1500, 384), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 1500, 6, 64)"},
        },
    ),
    (
        Reshape947,
        [((1500, 384), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 1500, 384)"},
        },
    ),
    (
        Reshape946,
        [((1500, 384), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 1500, 6, 64)"},
        },
    ),
    (
        Reshape948,
        [((1, 6, 1500, 64), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(6, 1500, 64)"},
        },
    ),
    (
        Reshape949,
        [((6, 1500, 1500), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 6, 1500, 1500)"},
        },
    ),
    (
        Reshape950,
        [((1, 6, 1500, 1500), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(6, 1500, 1500)"},
        },
    ),
    (
        Reshape951,
        [((6, 1500, 64), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 6, 1500, 64)"},
        },
    ),
    (
        Reshape945,
        [((1, 1500, 6, 64), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1500, 384)"},
        },
    ),
    (
        Reshape952,
        [((6, 1, 1500), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 6, 1, 1500)"},
        },
    ),
    (
        Reshape953,
        [((1, 6, 1, 1500), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(6, 1, 1500)"},
        },
    ),
    (
        Reshape954,
        [((1, 16, 256, 128), torch.float32)],
        {
            "model_names": ["pt_xglm_facebook_xglm_1_7b_clm_hf", "pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(16, 256, 128)"},
        },
    ),
    (
        Reshape955,
        [((16, 256, 128), torch.float32)],
        {
            "model_names": ["pt_xglm_facebook_xglm_1_7b_clm_hf", "pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 16, 256, 128)"},
        },
    ),
    (
        Reshape861,
        [((1, 256, 16, 128), torch.float32)],
        {
            "model_names": ["pt_xglm_facebook_xglm_1_7b_clm_hf", "pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(256, 2048)"},
        },
    ),
    (
        Reshape956,
        [((1, 85, 52, 52), torch.bfloat16)],
        {
            "model_names": ["pt_yolox_yolox_tiny_obj_det_torchhub", "pt_yolox_yolox_nano_obj_det_torchhub"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 85, 2704, 1)"},
        },
    ),
    (
        Reshape957,
        [((1, 85, 26, 26), torch.bfloat16)],
        {
            "model_names": ["pt_yolox_yolox_tiny_obj_det_torchhub", "pt_yolox_yolox_nano_obj_det_torchhub"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 85, 676, 1)"},
        },
    ),
    (
        Reshape958,
        [((1, 85, 13, 13), torch.bfloat16)],
        {
            "model_names": ["pt_yolox_yolox_tiny_obj_det_torchhub", "pt_yolox_yolox_nano_obj_det_torchhub"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 85, 169, 1)"},
        },
    ),
    (
        Reshape959,
        [((1, 6, 768), torch.float32)],
        {
            "model_names": ["onnx_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf"],
            "pcc": 0.99,
            "args": {"shape": "(6, 768)"},
        },
    ),
    (
        Reshape960,
        [((1, 6, 768), torch.float32)],
        {
            "model_names": ["onnx_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 6, 12, 64)"},
        },
    ),
    (
        Reshape961,
        [((6, 768), torch.float32)],
        {
            "model_names": ["onnx_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 6, 768)"},
        },
    ),
    (
        Reshape962,
        [((1, 12, 6, 64), torch.float32)],
        {
            "model_names": ["onnx_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf"],
            "pcc": 0.99,
            "args": {"shape": "(12, 6, 64)"},
        },
    ),
    (
        Reshape963,
        [((1, 12, 64, 6), torch.float32)],
        {
            "model_names": ["onnx_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf"],
            "pcc": 0.99,
            "args": {"shape": "(12, 64, 6)"},
        },
    ),
    (
        Reshape964,
        [((12, 6, 6), torch.float32)],
        {
            "model_names": ["onnx_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 12, 6, 6)"},
        },
    ),
    (
        Reshape965,
        [((1, 12, 6, 6), torch.float32)],
        {
            "model_names": ["onnx_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf"],
            "pcc": 0.99,
            "args": {"shape": "(12, 6, 6)"},
        },
    ),
    (
        Reshape966,
        [((12, 6, 64), torch.float32)],
        {
            "model_names": ["onnx_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 12, 6, 64)"},
        },
    ),
    (
        Reshape959,
        [((1, 6, 12, 64), torch.float32)],
        {
            "model_names": ["onnx_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf"],
            "pcc": 0.99,
            "args": {"shape": "(6, 768)"},
        },
    ),
    (
        Reshape967,
        [((1, 384, 1024), torch.float32)],
        {
            "model_names": [
                "onnx_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                "pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(384, 1024)"},
        },
    ),
    (
        Reshape968,
        [((1, 384, 1024), torch.float32)],
        {
            "model_names": [
                "onnx_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                "pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 384, 16, 64)"},
        },
    ),
    (
        Reshape969,
        [((384, 1024), torch.float32)],
        {
            "model_names": [
                "onnx_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                "pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 384, 1024)"},
        },
    ),
    (
        Reshape970,
        [((1, 16, 384, 64), torch.float32)],
        {
            "model_names": [
                "onnx_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                "pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(16, 384, 64)"},
        },
    ),
    (
        Reshape971,
        [((1, 16, 64, 384), torch.float32)],
        {
            "model_names": ["onnx_bert_phiyodr_bert_large_finetuned_squad2_qa_hf"],
            "pcc": 0.99,
            "args": {"shape": "(16, 64, 384)"},
        },
    ),
    (
        Reshape972,
        [((16, 384, 384), torch.float32)],
        {
            "model_names": [
                "onnx_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                "pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 16, 384, 384)"},
        },
    ),
    (
        Reshape973,
        [((1, 16, 384, 384), torch.float32)],
        {
            "model_names": [
                "onnx_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                "pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(16, 384, 384)"},
        },
    ),
    (
        Reshape974,
        [((16, 384, 64), torch.float32)],
        {
            "model_names": [
                "onnx_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                "pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 16, 384, 64)"},
        },
    ),
    (
        Reshape967,
        [((1, 384, 16, 64), torch.float32)],
        {
            "model_names": [
                "onnx_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                "pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(384, 1024)"},
        },
    ),
    (
        Reshape975,
        [((384, 1), torch.float32)],
        {
            "model_names": [
                "onnx_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                "pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 384, 1)"},
        },
    ),
    (
        Reshape976,
        [((1, 32, 128, 128), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 32, 16384)"},
        },
    ),
    (
        Reshape977,
        [((1, 16384, 32), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 16384, 1, 32)"},
        },
    ),
    (
        Reshape978,
        [((1, 16384, 32), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 128, 128, 32)"},
        },
    ),
    (
        Reshape979,
        [((1, 32, 16384), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 32, 128, 128)"},
        },
    ),
    (
        Reshape980,
        [((1, 32, 16, 16), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 32, 256)"},
        },
    ),
    (
        Reshape981,
        [((1, 256, 32), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(256, 32)"},
        },
    ),
    (
        Reshape982,
        [((1, 256, 32), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 256, 1, 32)"},
        },
    ),
    (
        Reshape983,
        [((256, 32), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 256, 32)"},
        },
    ),
    (
        Reshape980,
        [((1, 1, 32, 256), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 32, 256)"},
        },
    ),
    (
        Reshape984,
        [((1, 128, 16384), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 128, 128, 128)"},
        },
    ),
    (
        Reshape985,
        [((1, 128, 128, 128), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 128, 16384)"},
        },
    ),
    (
        Reshape986,
        [((1, 64, 64, 64), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 64, 4096)"},
        },
    ),
    (
        Reshape987,
        [((1, 4096, 64), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 4096, 2, 32)"},
        },
    ),
    (
        Reshape988,
        [((1, 4096, 64), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 64, 64, 64)"},
        },
    ),
    (
        Reshape989,
        [((1, 2, 4096, 32), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(2, 4096, 32)"},
        },
    ),
    (
        Reshape988,
        [((1, 64, 4096), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 64, 64, 64)"},
        },
    ),
    (
        Reshape990,
        [((1, 2, 32, 256), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(2, 32, 256)"},
        },
    ),
    (
        Reshape991,
        [((1, 2, 256, 32), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(2, 256, 32)"},
        },
    ),
    (
        Reshape992,
        [((2, 4096, 32), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 2, 4096, 32)"},
        },
    ),
    (
        Reshape993,
        [((1, 4096, 2, 32), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(4096, 64)"},
        },
    ),
    (
        Reshape994,
        [((4096, 64), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 4096, 64)"},
        },
    ),
    (
        Reshape408,
        [((1, 256, 64, 64), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 256, 4096)"},
        },
    ),
    (
        Reshape995,
        [((1, 160, 32, 32), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 160, 1024)"},
        },
    ),
    (
        Reshape996,
        [((1, 1024, 160), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 1024, 5, 32)"},
        },
    ),
    (
        Reshape997,
        [((1, 1024, 160), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 32, 32, 160)"},
        },
    ),
    (
        Reshape998,
        [((1, 5, 1024, 32), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(5, 1024, 32)"},
        },
    ),
    (
        Reshape999,
        [((1, 160, 1024), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 160, 32, 32)"},
        },
    ),
    (
        Reshape1000,
        [((1, 160, 16, 16), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 160, 256)"},
        },
    ),
    (
        Reshape1001,
        [((1, 256, 160), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(256, 160)"},
        },
    ),
    (
        Reshape1002,
        [((1, 256, 160), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 256, 5, 32)"},
        },
    ),
    (
        Reshape1003,
        [((256, 160), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 256, 160)"},
        },
    ),
    (
        Reshape1004,
        [((1, 5, 32, 256), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(5, 32, 256)"},
        },
    ),
    (
        Reshape1005,
        [((1, 5, 256, 32), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(5, 256, 32)"},
        },
    ),
    (
        Reshape1006,
        [((5, 1024, 32), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 5, 1024, 32)"},
        },
    ),
    (
        Reshape1007,
        [((1, 1024, 5, 32), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1024, 160)"},
        },
    ),
    (
        Reshape1008,
        [((1024, 160), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 1024, 160)"},
        },
    ),
    (
        Reshape1009,
        [((1, 640, 1024), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 640, 32, 32)"},
        },
    ),
    (
        Reshape1010,
        [((1, 640, 32, 32), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 640, 1024)"},
        },
    ),
    (
        Reshape334,
        [((1, 256, 16, 16), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 256, 256)"},
        },
    ),
    (
        Reshape334,
        [((256, 256), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 256, 256)"},
        },
    ),
    (
        Reshape1011,
        [((1, 8, 256, 32), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(8, 256, 32)"},
        },
    ),
    (
        Reshape1012,
        [((1, 8, 32, 256), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(8, 32, 256)"},
        },
    ),
    (
        Reshape1013,
        [((8, 256, 32), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 8, 256, 32)"},
        },
    ),
    (
        Reshape332,
        [((1, 256, 8, 32), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(256, 256)"},
        },
    ),
    (
        Reshape1014,
        [((1, 1024, 256), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 1024, 16, 16)"},
        },
    ),
    (
        Reshape1015,
        [((1, 1024, 16, 16), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 1024, 256)"},
        },
    ),
    (
        Reshape1016,
        [((1, 96, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 96, 4096)"},
        },
    ),
    (
        Reshape1017,
        [((1, 4096, 96), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 64, 64, 96)"},
        },
    ),
    (
        Reshape1018,
        [((1, 64, 64, 96), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 8, 8, 8, 8, 96)"},
        },
    ),
    (
        Reshape1019,
        [((1, 64, 64, 96), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 4096, 96)"},
        },
    ),
    (
        Reshape1020,
        [((1, 64, 64, 96), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_s_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(4096, 96)"}},
    ),
    (
        Reshape1020,
        [((1, 8, 8, 8, 8, 96), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(4096, 96)"},
        },
    ),
    (
        Reshape1019,
        [((1, 8, 8, 8, 8, 96), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 4096, 96)"},
        },
    ),
    (
        Reshape1017,
        [((1, 8, 8, 8, 8, 96), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 64, 64, 96)"},
        },
    ),
    (
        Reshape1021,
        [((4096, 96), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(64, 64, 96)"},
        },
    ),
    (
        Reshape1022,
        [((4096, 96), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf"],
            "pcc": 0.99,
            "args": {"shape": "(64, 64, 3, 32)"},
        },
    ),
    (
        Reshape1017,
        [((4096, 96), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_s_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(1, 64, 64, 96)"}},
    ),
    (
        Reshape1022,
        [((64, 64, 96), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf"],
            "pcc": 0.99,
            "args": {"shape": "(64, 64, 3, 32)"},
        },
    ),
    (
        Reshape1018,
        [((64, 64, 96), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 8, 8, 8, 8, 96)"},
        },
    ),
    (
        Reshape1023,
        [((64, 3, 64, 32), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(192, 64, 32)"},
        },
    ),
    (
        Reshape1024,
        [((64, 3, 32, 64), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(192, 32, 64)"},
        },
    ),
    (
        Reshape1025,
        [((192, 64, 64), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(64, 3, 64, 64)"},
        },
    ),
    (
        Reshape1026,
        [((225, 512), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_b_img_cls_torchvision",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 15, 15, 512)"},
        },
    ),
    (
        Reshape1027,
        [((1, 15, 15, 512), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_b_img_cls_torchvision",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(225, 512)"},
        },
    ),
    (
        Reshape1028,
        [((225, 3), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(225, 3)"},
        },
    ),
    (
        Reshape1029,
        [((4096, 3), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(64, 64, 3)"},
        },
    ),
    (
        Reshape1030,
        [((64, 3, 64, 64), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(192, 64, 64)"},
        },
    ),
    (
        Reshape1031,
        [((64, 3, 64, 64), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 64, 3, 64, 64)"},
        },
    ),
    (
        Reshape1032,
        [((192, 64, 32), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(64, 3, 64, 32)"},
        },
    ),
    (
        Reshape1020,
        [((64, 64, 3, 32), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(4096, 96)"},
        },
    ),
    (
        Reshape1025,
        [((1, 64, 3, 64, 64), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(64, 3, 64, 64)"},
        },
    ),
    (
        Reshape1033,
        [((1, 32, 32, 384), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1024, 384)"},
        },
    ),
    (
        Reshape1034,
        [((1024, 192), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 1024, 192)"},
        },
    ),
    (
        Reshape1035,
        [((1024, 192), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(16, 64, 192)"},
        },
    ),
    (
        Reshape1036,
        [((1024, 192), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf"],
            "pcc": 0.99,
            "args": {"shape": "(16, 64, 6, 32)"},
        },
    ),
    (
        Reshape1037,
        [((1024, 192), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_s_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(1, 32, 32, 192)"}},
    ),
    (
        Reshape1037,
        [((1, 1024, 192), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 32, 32, 192)"},
        },
    ),
    (
        Reshape1038,
        [((1, 32, 32, 192), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 4, 8, 4, 8, 192)"},
        },
    ),
    (
        Reshape1034,
        [((1, 32, 32, 192), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 1024, 192)"},
        },
    ),
    (
        Reshape1039,
        [((1, 32, 32, 192), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_s_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(1024, 192)"}},
    ),
    (
        Reshape1039,
        [((1, 4, 4, 8, 8, 192), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1024, 192)"},
        },
    ),
    (
        Reshape1036,
        [((16, 64, 192), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf"],
            "pcc": 0.99,
            "args": {"shape": "(16, 64, 6, 32)"},
        },
    ),
    (
        Reshape1040,
        [((16, 64, 192), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 4, 4, 8, 8, 192)"},
        },
    ),
    (
        Reshape1041,
        [((16, 6, 64, 32), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(96, 64, 32)"},
        },
    ),
    (
        Reshape1042,
        [((16, 6, 32, 64), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(96, 32, 64)"},
        },
    ),
    (
        Reshape1043,
        [((96, 64, 64), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(16, 6, 64, 64)"},
        },
    ),
    (
        Reshape1044,
        [((225, 6), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(225, 6)"},
        },
    ),
    (
        Reshape1045,
        [((4096, 6), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(64, 64, 6)"},
        },
    ),
    (
        Reshape1046,
        [((16, 6, 64, 64), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(96, 64, 64)"},
        },
    ),
    (
        Reshape1047,
        [((16, 6, 64, 64), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 16, 6, 64, 64)"},
        },
    ),
    (
        Reshape1048,
        [((96, 64, 32), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(16, 6, 64, 32)"},
        },
    ),
    (
        Reshape1039,
        [((16, 64, 6, 32), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1024, 192)"},
        },
    ),
    (
        Reshape1034,
        [((1, 4, 8, 4, 8, 192), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 1024, 192)"},
        },
    ),
    (
        Reshape1037,
        [((1, 4, 8, 4, 8, 192), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 32, 32, 192)"},
        },
    ),
    (
        Reshape1043,
        [((1, 16, 6, 64, 64), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(16, 6, 64, 64)"},
        },
    ),
    (
        Reshape231,
        [((1, 16, 16, 768), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(256, 768)"},
        },
    ),
    (
        Reshape1049,
        [((256, 384), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 256, 384)"},
        },
    ),
    (
        Reshape1050,
        [((256, 384), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(4, 64, 384)"},
        },
    ),
    (
        Reshape1051,
        [((256, 384), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf"],
            "pcc": 0.99,
            "args": {"shape": "(4, 64, 12, 32)"},
        },
    ),
    (
        Reshape1052,
        [((256, 384), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_s_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(1, 16, 16, 384)"}},
    ),
    (
        Reshape1052,
        [((1, 256, 384), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 16, 16, 384)"},
        },
    ),
    (
        Reshape1053,
        [((1, 16, 16, 384), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 2, 8, 2, 8, 384)"},
        },
    ),
    (
        Reshape1049,
        [((1, 16, 16, 384), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 256, 384)"},
        },
    ),
    (
        Reshape1054,
        [((1, 16, 16, 384), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_s_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(256, 384)"}},
    ),
    (
        Reshape1054,
        [((1, 2, 2, 8, 8, 384), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(256, 384)"},
        },
    ),
    (
        Reshape1051,
        [((4, 64, 384), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf"],
            "pcc": 0.99,
            "args": {"shape": "(4, 64, 12, 32)"},
        },
    ),
    (
        Reshape1055,
        [((4, 64, 384), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 2, 2, 8, 8, 384)"},
        },
    ),
    (
        Reshape1056,
        [((4, 12, 64, 32), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(48, 64, 32)"},
        },
    ),
    (
        Reshape1057,
        [((4, 12, 32, 64), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(48, 32, 64)"},
        },
    ),
    (
        Reshape1058,
        [((48, 64, 64), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(4, 12, 64, 64)"},
        },
    ),
    (
        Reshape1059,
        [((225, 12), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(225, 12)"},
        },
    ),
    (
        Reshape1060,
        [((4096, 12), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(64, 64, 12)"},
        },
    ),
    (
        Reshape1061,
        [((4, 12, 64, 64), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(48, 64, 64)"},
        },
    ),
    (
        Reshape1062,
        [((4, 12, 64, 64), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 4, 12, 64, 64)"},
        },
    ),
    (
        Reshape1063,
        [((48, 64, 32), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(4, 12, 64, 32)"},
        },
    ),
    (
        Reshape1054,
        [((4, 64, 12, 32), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(256, 384)"},
        },
    ),
    (
        Reshape1049,
        [((1, 2, 8, 2, 8, 384), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 256, 384)"},
        },
    ),
    (
        Reshape1052,
        [((1, 2, 8, 2, 8, 384), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 16, 16, 384)"},
        },
    ),
    (
        Reshape1058,
        [((1, 4, 12, 64, 64), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(4, 12, 64, 64)"},
        },
    ),
    (
        Reshape1064,
        [((1, 8, 8, 1536), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(64, 1536)"},
        },
    ),
    (
        Reshape1065,
        [((64, 768), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 64, 768)"},
        },
    ),
    (
        Reshape1066,
        [((64, 768), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 64, 24, 32)"},
        },
    ),
    (
        Reshape1067,
        [((64, 768), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_s_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(1, 8, 8, 768)"}},
    ),
    (
        Reshape1067,
        [((1, 64, 768), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 8, 8, 768)"},
        },
    ),
    (
        Reshape1066,
        [((1, 64, 768), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 64, 24, 32)"},
        },
    ),
    (
        Reshape1065,
        [((1, 64, 768), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 64, 768)"},
        },
    ),
    (
        Reshape1068,
        [((1, 64, 768), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(1, 1, 1, 8, 8, 768)"},
        },
    ),
    (
        Reshape1069,
        [((1, 8, 8, 768), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 1, 8, 1, 8, 768)"},
        },
    ),
    (
        Reshape1070,
        [((1, 8, 8, 768), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_s_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(64, 768)"}},
    ),
    (
        Reshape1070,
        [((1, 1, 1, 8, 8, 768), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(64, 768)"},
        },
    ),
    (
        Reshape1071,
        [((1, 24, 64, 32), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(24, 64, 32)"},
        },
    ),
    (
        Reshape1072,
        [((1, 24, 32, 64), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(24, 32, 64)"},
        },
    ),
    (
        Reshape1073,
        [((24, 64, 64), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 24, 64, 64)"},
        },
    ),
    (
        Reshape1074,
        [((225, 24), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(225, 24)"},
        },
    ),
    (
        Reshape1075,
        [((4096, 24), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(64, 64, 24)"},
        },
    ),
    (
        Reshape1076,
        [((1, 24, 64, 64), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(24, 64, 64)"},
        },
    ),
    (
        Reshape1077,
        [((24, 64, 32), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 24, 64, 32)"},
        },
    ),
    (
        Reshape1070,
        [((1, 64, 24, 32), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(64, 768)"},
        },
    ),
    (
        Reshape1078,
        [((1, 768, 64), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 768, 8, 8)"},
        },
    ),
    (
        Reshape1079,
        [((1, 3072, 8, 8), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 3, 32, 32, 8, 8)"},
        },
    ),
    (
        Reshape1080,
        [((1, 3, 8, 32, 8, 32), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 3, 256, 256)"},
        },
    ),
    (
        Reshape1081,
        [((1, 15, 768), torch.float32)],
        {"model_names": ["pd_bert_bert_base_japanese_seq_cls_padlenlp"], "pcc": 0.99, "args": {"shape": "(15, 768)"}},
    ),
    (
        Reshape1082,
        [((1, 15, 768), torch.float32)],
        {
            "model_names": ["pd_bert_bert_base_japanese_seq_cls_padlenlp"],
            "pcc": 0.99,
            "args": {"shape": "(1, 15, 12, 64)"},
        },
    ),
    (
        Reshape1083,
        [((15, 768), torch.float32)],
        {
            "model_names": ["pd_bert_bert_base_japanese_seq_cls_padlenlp"],
            "pcc": 0.99,
            "args": {"shape": "(1, 15, 768)"},
        },
    ),
    (
        Reshape1084,
        [((1, 12, 15, 64), torch.float32)],
        {
            "model_names": ["pd_bert_bert_base_japanese_seq_cls_padlenlp"],
            "pcc": 0.99,
            "args": {"shape": "(12, 15, 64)"},
        },
    ),
    (
        Reshape1085,
        [((1, 12, 64, 15), torch.float32)],
        {
            "model_names": ["pd_bert_bert_base_japanese_seq_cls_padlenlp"],
            "pcc": 0.99,
            "args": {"shape": "(12, 64, 15)"},
        },
    ),
    (
        Reshape1086,
        [((12, 15, 15), torch.float32)],
        {
            "model_names": ["pd_bert_bert_base_japanese_seq_cls_padlenlp"],
            "pcc": 0.99,
            "args": {"shape": "(1, 12, 15, 15)"},
        },
    ),
    (
        Reshape1087,
        [((1, 12, 15, 15), torch.float32)],
        {
            "model_names": ["pd_bert_bert_base_japanese_seq_cls_padlenlp"],
            "pcc": 0.99,
            "args": {"shape": "(12, 15, 15)"},
        },
    ),
    (
        Reshape1088,
        [((12, 15, 64), torch.float32)],
        {
            "model_names": ["pd_bert_bert_base_japanese_seq_cls_padlenlp"],
            "pcc": 0.99,
            "args": {"shape": "(1, 12, 15, 64)"},
        },
    ),
    (
        Reshape1081,
        [((1, 15, 12, 64), torch.float32)],
        {"model_names": ["pd_bert_bert_base_japanese_seq_cls_padlenlp"], "pcc": 0.99, "args": {"shape": "(15, 768)"}},
    ),
    (
        Reshape1089,
        [((1, 128, 1024), torch.float32)],
        {
            "model_names": [
                "pt_albert_large_v1_token_cls_hf",
                "pt_albert_large_v2_mlm_hf",
                "pt_albert_large_v2_token_cls_hf",
                "pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(128, 1024)"},
        },
    ),
    (
        Reshape1090,
        [((1, 128, 1024), torch.float32)],
        {
            "model_names": [
                "pt_albert_large_v1_token_cls_hf",
                "pt_albert_large_v2_mlm_hf",
                "pt_albert_large_v2_token_cls_hf",
                "pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 128, 16, 64)"},
        },
    ),
    (
        Reshape1091,
        [((128, 1024), torch.float32)],
        {
            "model_names": [
                "pt_albert_large_v1_token_cls_hf",
                "pt_albert_large_v2_mlm_hf",
                "pt_albert_large_v2_token_cls_hf",
                "pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 128, 1024)"},
        },
    ),
    (
        Reshape1092,
        [((1, 16, 128, 64), torch.float32)],
        {
            "model_names": [
                "pt_albert_large_v1_token_cls_hf",
                "pt_albert_large_v2_mlm_hf",
                "pt_albert_large_v2_token_cls_hf",
                "pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(16, 128, 64)"},
        },
    ),
    (
        Reshape1093,
        [((16, 128, 64), torch.float32)],
        {
            "model_names": [
                "pt_albert_large_v1_token_cls_hf",
                "pt_albert_large_v2_mlm_hf",
                "pt_albert_large_v2_token_cls_hf",
                "pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 16, 128, 64)"},
        },
    ),
    (
        Reshape1089,
        [((1, 128, 16, 64), torch.float32)],
        {
            "model_names": [
                "pt_albert_large_v1_token_cls_hf",
                "pt_albert_large_v2_mlm_hf",
                "pt_albert_large_v2_token_cls_hf",
                "pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(128, 1024)"},
        },
    ),
    (
        Reshape1094,
        [((1, 1664, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet169_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 1664, 1, 1)"},
        },
    ),
    (
        Reshape1095,
        [((128, 1), torch.float32)],
        {
            "model_names": ["pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader"],
            "pcc": 0.99,
            "args": {"shape": "(1, 128, 1)"},
        },
    ),
    (
        Reshape1096,
        [((1, 128), torch.float32)],
        {
            "model_names": ["pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader"],
            "pcc": 0.99,
            "args": {"shape": "(1, 128)"},
        },
    ),
    (
        Reshape1097,
        [((1, 1), torch.float32)],
        {
            "model_names": ["pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader"],
            "pcc": 0.99,
            "args": {"shape": "(1,)"},
        },
    ),
    (
        Reshape1098,
        [((1, 16, 128, 256), torch.float32)],
        {"model_names": ["pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf"], "pcc": 0.99, "args": {"shape": "(16, 128, 256)"}},
    ),
    (
        Reshape866,
        [((1, 8, 4, 256, 64), torch.float32)],
        {"model_names": ["pt_llama3_meta_llama_llama_3_2_1b_clm_hf"], "pcc": 0.99, "args": {"shape": "(32, 256, 64)"}},
    ),
    (
        Reshape1099,
        [((256, 8192), torch.float32)],
        {
            "model_names": ["pt_llama3_meta_llama_llama_3_2_1b_clm_hf", "pt_phi1_microsoft_phi_1_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 256, 8192)"},
        },
    ),
    (
        Reshape1100,
        [((2048, 1, 4), torch.float32)],
        {"model_names": ["pt_mamba_state_spaces_mamba_370m_hf_clm_hf"], "pcc": 0.99, "args": {"shape": "(2048, 1, 4)"}},
    ),
    (
        Reshape1101,
        [((1, 6, 2048), torch.float32)],
        {"model_names": ["pt_mamba_state_spaces_mamba_370m_hf_clm_hf"], "pcc": 0.99, "args": {"shape": "(6, 2048)"}},
    ),
    (
        Reshape1102,
        [((6, 64), torch.float32)],
        {"model_names": ["pt_mamba_state_spaces_mamba_370m_hf_clm_hf"], "pcc": 0.99, "args": {"shape": "(1, 6, 64)"}},
    ),
    (
        Reshape1103,
        [((1, 2048, 1, 16), torch.float32)],
        {
            "model_names": ["pt_mamba_state_spaces_mamba_370m_hf_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 2048, 16)"},
        },
    ),
    (
        Reshape46,
        [((1, 2048, 1), torch.float32)],
        {"model_names": ["pt_mamba_state_spaces_mamba_370m_hf_clm_hf"], "pcc": 0.99, "args": {"shape": "(1, 2048)"}},
    ),
    (
        Reshape1104,
        [((200, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(200, 1, 3, 3)"},
        },
    ),
    (
        Reshape1105,
        [((184, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(184, 1, 3, 3)"},
        },
    ),
    (
        Reshape1106,
        [((1, 960, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 960, 1, 1)"},
        },
    ),
    (
        Reshape1107,
        [((1, 35, 896), torch.float32)],
        {"model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf"], "pcc": 0.99, "args": {"shape": "(35, 896)"}},
    ),
    (
        Reshape1108,
        [((1, 35, 896), torch.float32)],
        {
            "model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 35, 14, 64)"},
        },
    ),
    (
        Reshape1109,
        [((35, 896), torch.float32)],
        {
            "model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 35, 896)"},
        },
    ),
    (
        Reshape1110,
        [((1, 14, 35, 64), torch.float32)],
        {
            "model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(14, 35, 64)"},
        },
    ),
    (
        Reshape1111,
        [((35, 128), torch.float32)],
        {
            "model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 35, 128)"},
        },
    ),
    (
        Reshape1112,
        [((1, 35, 128), torch.float32)],
        {
            "model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 35, 2, 64)"},
        },
    ),
    (
        Reshape1110,
        [((1, 2, 7, 35, 64), torch.float32)],
        {
            "model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(14, 35, 64)"},
        },
    ),
    (
        Reshape1113,
        [((14, 35, 35), torch.float32)],
        {
            "model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 14, 35, 35)"},
        },
    ),
    (
        Reshape1114,
        [((1, 14, 35, 35), torch.float32)],
        {
            "model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(14, 35, 35)"},
        },
    ),
    (
        Reshape1115,
        [((14, 35, 64), torch.float32)],
        {
            "model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 14, 35, 64)"},
        },
    ),
    (
        Reshape1107,
        [((1, 35, 14, 64), torch.float32)],
        {"model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf"], "pcc": 0.99, "args": {"shape": "(35, 896)"}},
    ),
    (
        Reshape1116,
        [((35, 4864), torch.float32)],
        {
            "model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 35, 4864)"},
        },
    ),
    (
        Reshape1117,
        [((1, 8, 64, 1), torch.float32)],
        {"model_names": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99, "args": {"shape": "(8, 64, 1)"}},
    ),
    (
        Reshape1118,
        [((1, 8, 61, 64), torch.float32)],
        {"model_names": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99, "args": {"shape": "(8, 61, 64)"}},
    ),
    (
        Reshape1119,
        [((8, 61, 61), torch.float32)],
        {"model_names": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99, "args": {"shape": "(1, 8, 61, 61)"}},
    ),
    (
        Reshape1120,
        [((1, 8, 61, 61), torch.float32)],
        {"model_names": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99, "args": {"shape": "(8, 61, 61)"}},
    ),
    (
        Reshape1121,
        [((1, 8, 64, 61), torch.float32)],
        {"model_names": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99, "args": {"shape": "(8, 64, 61)"}},
    ),
    (
        Reshape1122,
        [((8, 61, 64), torch.float32)],
        {"model_names": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99, "args": {"shape": "(1, 8, 61, 64)"}},
    ),
    (
        Reshape811,
        [((1, 61, 8, 64), torch.float32)],
        {"model_names": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99, "args": {"shape": "(61, 512)"}},
    ),
    (
        Reshape1123,
        [((8, 1, 61), torch.float32)],
        {"model_names": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99, "args": {"shape": "(1, 8, 1, 61)"}},
    ),
    (
        Reshape1124,
        [((1, 8, 1, 61), torch.float32)],
        {"model_names": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99, "args": {"shape": "(8, 1, 61)"}},
    ),
    (
        Reshape1125,
        [((1, 201, 768), torch.bfloat16)],
        {
            "model_names": ["pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(201, 768)"},
        },
    ),
    (
        Reshape1126,
        [((1, 201, 768), torch.bfloat16)],
        {
            "model_names": ["pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 201, 12, 64)"},
        },
    ),
    (
        Reshape1127,
        [((201, 768), torch.bfloat16)],
        {
            "model_names": ["pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 201, 768)"},
        },
    ),
    (
        Reshape1128,
        [((1, 12, 201, 64), torch.bfloat16)],
        {
            "model_names": ["pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(12, 201, 64)"},
        },
    ),
    (
        Reshape1129,
        [((12, 201, 201), torch.bfloat16)],
        {
            "model_names": ["pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 12, 201, 201)"},
        },
    ),
    (
        Reshape1130,
        [((1, 12, 201, 201), torch.bfloat16)],
        {
            "model_names": ["pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(12, 201, 201)"},
        },
    ),
    (
        Reshape1131,
        [((1, 12, 64, 201), torch.bfloat16)],
        {
            "model_names": ["pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(12, 64, 201)"},
        },
    ),
    (
        Reshape1132,
        [((12, 201, 64), torch.bfloat16)],
        {
            "model_names": ["pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 12, 201, 64)"},
        },
    ),
    (
        Reshape1125,
        [((1, 201, 12, 64), torch.bfloat16)],
        {
            "model_names": ["pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(201, 768)"},
        },
    ),
    (
        Reshape15,
        [((1, 1, 768), torch.bfloat16)],
        {
            "model_names": [
                "pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf",
                "pt_vit_vit_b_32_img_cls_torchvision",
                "pt_vilt_dandelin_vilt_b32_mlm_mlm_hf",
                "pt_vit_vit_b_16_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 768)"},
        },
    ),
    (
        Reshape1133,
        [((50, 1, 768), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_b_32_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(50, 768)"},
        },
    ),
    (
        Reshape1134,
        [((50, 2304), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_b_32_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(50, 1, 2304)"},
        },
    ),
    (
        Reshape1135,
        [((50, 1, 2304), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_b_32_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(50, 1, 3, 768)"},
        },
    ),
    (
        Reshape1136,
        [((1, 50, 1, 768), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_b_32_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(50, 12, 64)"},
        },
    ),
    (
        Reshape1137,
        [((12, 50, 64), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_b_32_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(12, 50, 64)"},
        },
    ),
    (
        Reshape1138,
        [((12, 50, 64), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_b_32_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 12, 50, 64)"},
        },
    ),
    (
        Reshape1139,
        [((12, 50, 50), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_b_32_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 12, 50, 50)"},
        },
    ),
    (
        Reshape1140,
        [((1, 12, 50, 50), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_b_32_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(12, 50, 50)"},
        },
    ),
    (
        Reshape1133,
        [((50, 1, 12, 64), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_b_32_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(50, 768)"},
        },
    ),
    (
        Reshape1141,
        [((50, 768), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_b_32_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(50, 1, 768)"},
        },
    ),
    (
        Reshape1142,
        [((1, 128, 40, 40), torch.bfloat16)],
        {
            "model_names": ["pt_yolo_world_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 4, 32, 40, 40)"},
        },
    ),
    (
        Reshape1143,
        [((80, 128), torch.bfloat16)],
        {
            "model_names": ["pt_yolo_world_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 80, 128)"},
        },
    ),
    (
        Reshape1144,
        [((1, 80, 128), torch.bfloat16)],
        {
            "model_names": ["pt_yolo_world_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 80, 4, 32)"},
        },
    ),
    (
        Reshape1145,
        [((1, 4, 32, 40, 40), torch.bfloat16)],
        {
            "model_names": ["pt_yolo_world_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 128, 40, 40)"},
        },
    ),
    (
        Reshape1146,
        [((1, 64, 80, 80), torch.bfloat16)],
        {
            "model_names": ["pt_yolo_world_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 2, 32, 80, 80)"},
        },
    ),
    (
        Reshape1147,
        [((80, 64), torch.bfloat16)],
        {
            "model_names": ["pt_yolo_world_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 80, 64)"},
        },
    ),
    (
        Reshape1148,
        [((1, 80, 64), torch.bfloat16)],
        {
            "model_names": ["pt_yolo_world_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 80, 2, 32)"},
        },
    ),
    (
        Reshape1149,
        [((1, 2, 32, 80, 80), torch.bfloat16)],
        {
            "model_names": ["pt_yolo_world_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 64, 80, 80)"},
        },
    ),
    (
        Reshape1150,
        [((1, 80, 256), torch.bfloat16)],
        {
            "model_names": ["pt_yolo_world_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 80, 8, 32)"},
        },
    ),
    (
        Reshape1151,
        [((1, 8, 80, 32), torch.bfloat16)],
        {
            "model_names": ["pt_yolo_world_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(8, 80, 32)"},
        },
    ),
    (
        Reshape1152,
        [((1, 256, 3, 3), torch.bfloat16)],
        {
            "model_names": ["pt_yolo_world_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 256, 9)"},
        },
    ),
    (
        Reshape1153,
        [((1, 27, 256), torch.bfloat16)],
        {
            "model_names": ["pt_yolo_world_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 27, 8, 32)"},
        },
    ),
    (
        Reshape1154,
        [((1, 8, 32, 27), torch.bfloat16)],
        {
            "model_names": ["pt_yolo_world_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(8, 32, 27)"},
        },
    ),
    (
        Reshape1155,
        [((8, 80, 27), torch.bfloat16)],
        {
            "model_names": ["pt_yolo_world_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 8, 80, 27)"},
        },
    ),
    (
        Reshape1156,
        [((1, 80, 8, 32), torch.bfloat16)],
        {
            "model_names": ["pt_yolo_world_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(80, 256)"},
        },
    ),
    (
        Reshape1157,
        [((80, 512), torch.bfloat16)],
        {
            "model_names": ["pt_yolo_world_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 80, 512)"},
        },
    ),
    (
        Reshape1158,
        [((1, 80, 512), torch.bfloat16)],
        {
            "model_names": ["pt_yolo_world_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(80, 512)"},
        },
    ),
    (
        Reshape1159,
        [((1, 256, 20, 20), torch.bfloat16)],
        {
            "model_names": ["pt_yolo_world_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 8, 32, 20, 20)"},
        },
    ),
    (
        Reshape1160,
        [((80, 256), torch.bfloat16)],
        {
            "model_names": ["pt_yolo_world_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 80, 256)"},
        },
    ),
    (
        Reshape1161,
        [((1, 8, 32, 20, 20), torch.bfloat16)],
        {
            "model_names": ["pt_yolo_world_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 256, 20, 20)"},
        },
    ),
    (
        Reshape1162,
        [((1, 84, 8400), torch.bfloat16)],
        {
            "model_names": ["pt_yolo_world_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(84, 8400)"},
        },
    ),
    (
        Reshape436,
        [((1, 768, 14, 14), torch.float32)],
        {
            "model_names": ["onnx_vit_base_google_vit_base_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 768, 196)"},
        },
    ),
    (
        Reshape437,
        [((1, 197, 768), torch.float32)],
        {
            "model_names": ["onnx_vit_base_google_vit_base_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(197, 768)"},
        },
    ),
    (
        Reshape438,
        [((1, 197, 768), torch.float32)],
        {
            "model_names": ["onnx_vit_base_google_vit_base_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 197, 12, 64)"},
        },
    ),
    (
        Reshape439,
        [((197, 768), torch.float32)],
        {
            "model_names": ["onnx_vit_base_google_vit_base_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 197, 768)"},
        },
    ),
    (
        Reshape441,
        [((1, 12, 197, 64), torch.float32)],
        {
            "model_names": ["onnx_vit_base_google_vit_base_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(12, 197, 64)"},
        },
    ),
    (
        Reshape447,
        [((1, 12, 64, 197), torch.float32)],
        {
            "model_names": ["onnx_vit_base_google_vit_base_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(12, 64, 197)"},
        },
    ),
    (
        Reshape442,
        [((12, 197, 197), torch.float32)],
        {
            "model_names": ["onnx_vit_base_google_vit_base_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 12, 197, 197)"},
        },
    ),
    (
        Reshape446,
        [((1, 12, 197, 197), torch.float32)],
        {
            "model_names": ["onnx_vit_base_google_vit_base_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(12, 197, 197)"},
        },
    ),
    (
        Reshape448,
        [((12, 197, 64), torch.float32)],
        {
            "model_names": ["onnx_vit_base_google_vit_base_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 12, 197, 64)"},
        },
    ),
    (
        Reshape437,
        [((1, 197, 12, 64), torch.float32)],
        {
            "model_names": ["onnx_vit_base_google_vit_base_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(197, 768)"},
        },
    ),
    (
        Reshape1163,
        [((1, 16, 768), torch.float32)],
        {
            "model_names": ["pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf"],
            "pcc": 0.99,
            "args": {"shape": "(16, 768)"},
        },
    ),
    (
        Reshape1164,
        [((1, 16, 768), torch.float32)],
        {
            "model_names": ["pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 16, 12, 64)"},
        },
    ),
    (
        Reshape1165,
        [((16, 768), torch.float32)],
        {
            "model_names": ["pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 16, 768)"},
        },
    ),
    (
        Reshape1166,
        [((1, 12, 16, 64), torch.float32)],
        {
            "model_names": ["pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf"],
            "pcc": 0.99,
            "args": {"shape": "(12, 16, 64)"},
        },
    ),
    (
        Reshape1167,
        [((12, 16, 16), torch.float32)],
        {
            "model_names": ["pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 12, 16, 16)"},
        },
    ),
    (
        Reshape1168,
        [((1, 12, 16, 16), torch.float32)],
        {
            "model_names": ["pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf"],
            "pcc": 0.99,
            "args": {"shape": "(12, 16, 16)"},
        },
    ),
    (
        Reshape1169,
        [((12, 16, 64), torch.float32)],
        {
            "model_names": ["pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 12, 16, 64)"},
        },
    ),
    (
        Reshape1163,
        [((1, 16, 12, 64), torch.float32)],
        {
            "model_names": ["pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf"],
            "pcc": 0.99,
            "args": {"shape": "(16, 768)"},
        },
    ),
    (
        Reshape1170,
        [((1920, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b1_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1920, 1, 3, 3)"},
        },
    ),
    (
        Reshape1171,
        [((1344, 1, 5, 5), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b7_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1344, 1, 5, 5)"},
        },
    ),
    (
        Reshape1172,
        [((2304, 1, 5, 5), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b7_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(2304, 1, 5, 5)"},
        },
    ),
    (
        Reshape1173,
        [((3840, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b7_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(3840, 1, 3, 3)"},
        },
    ),
    (
        Reshape1174,
        [((1, 2560, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b7_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 2560, 1, 1)"},
        },
    ),
    (
        Reshape1175,
        [((72, 1, 1, 5), torch.bfloat16)],
        {
            "model_names": ["pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(72, 1, 1, 5)"},
        },
    ),
    (
        Reshape1176,
        [((72, 1, 5, 1), torch.bfloat16)],
        {
            "model_names": ["pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(72, 1, 5, 1)"},
        },
    ),
    (
        Reshape1177,
        [((120, 1, 1, 5), torch.bfloat16)],
        {
            "model_names": ["pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(120, 1, 1, 5)"},
        },
    ),
    (
        Reshape1178,
        [((120, 1, 5, 1), torch.bfloat16)],
        {
            "model_names": ["pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(120, 1, 5, 1)"},
        },
    ),
    (
        Reshape1179,
        [((240, 1, 1, 5), torch.bfloat16)],
        {
            "model_names": ["pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(240, 1, 1, 5)"},
        },
    ),
    (
        Reshape1180,
        [((240, 1, 5, 1), torch.bfloat16)],
        {
            "model_names": ["pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(240, 1, 5, 1)"},
        },
    ),
    (
        Reshape1181,
        [((200, 1, 1, 5), torch.bfloat16)],
        {
            "model_names": ["pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(200, 1, 1, 5)"},
        },
    ),
    (
        Reshape1182,
        [((200, 1, 5, 1), torch.bfloat16)],
        {
            "model_names": ["pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(200, 1, 5, 1)"},
        },
    ),
    (
        Reshape1183,
        [((184, 1, 1, 5), torch.bfloat16)],
        {
            "model_names": ["pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(184, 1, 1, 5)"},
        },
    ),
    (
        Reshape1184,
        [((184, 1, 5, 1), torch.bfloat16)],
        {
            "model_names": ["pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(184, 1, 5, 1)"},
        },
    ),
    (
        Reshape1185,
        [((480, 1, 1, 5), torch.bfloat16)],
        {
            "model_names": ["pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(480, 1, 1, 5)"},
        },
    ),
    (
        Reshape1186,
        [((480, 1, 5, 1), torch.bfloat16)],
        {
            "model_names": ["pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(480, 1, 5, 1)"},
        },
    ),
    (
        Reshape1187,
        [((672, 1, 1, 5), torch.bfloat16)],
        {
            "model_names": ["pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(672, 1, 1, 5)"},
        },
    ),
    (
        Reshape1188,
        [((672, 1, 5, 1), torch.bfloat16)],
        {
            "model_names": ["pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(672, 1, 5, 1)"},
        },
    ),
    (
        Reshape1189,
        [((960, 1, 1, 5), torch.bfloat16)],
        {
            "model_names": ["pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(960, 1, 1, 5)"},
        },
    ),
    (
        Reshape1190,
        [((960, 1, 5, 1), torch.bfloat16)],
        {
            "model_names": ["pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(960, 1, 5, 1)"},
        },
    ),
    (
        Reshape1191,
        [((1, 1296, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_facebook_regnet_y_064_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 1296, 1, 1)"},
        },
    ),
    (
        Reshape1192,
        [((1, 3712, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_regnet_facebook_regnet_y_320_img_cls_hf",
                "pt_regnet_regnet_y_32gf_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 3712, 1, 1)"},
        },
    ),
    (
        Reshape1193,
        [((1, 32, 128, 128), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 32, 16384, 1)"},
        },
    ),
    (
        Reshape977,
        [((1, 16384, 32), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 16384, 1, 32)"},
        },
    ),
    (
        Reshape978,
        [((1, 16384, 32), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 128, 128, 32)"},
        },
    ),
    (
        Reshape979,
        [((1, 32, 16384), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 32, 128, 128)"},
        },
    ),
    (
        Reshape980,
        [((1, 32, 16, 16), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 32, 256)"},
        },
    ),
    (
        Reshape981,
        [((1, 256, 32), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(256, 32)"},
        },
    ),
    (
        Reshape982,
        [((1, 256, 32), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 256, 1, 32)"},
        },
    ),
    (
        Reshape983,
        [((256, 32), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 256, 32)"},
        },
    ),
    (
        Reshape980,
        [((1, 1, 32, 256), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 32, 256)"},
        },
    ),
    (
        Reshape984,
        [((1, 128, 16384), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 128, 128, 128)"},
        },
    ),
    (
        Reshape1194,
        [((1, 128, 128, 128), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 128, 16384, 1)"},
        },
    ),
    (
        Reshape1195,
        [((1, 64, 64, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 64, 4096, 1)"},
        },
    ),
    (
        Reshape987,
        [((1, 4096, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 4096, 2, 32)"},
        },
    ),
    (
        Reshape988,
        [((1, 4096, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 64, 64, 64)"},
        },
    ),
    (
        Reshape989,
        [((1, 2, 4096, 32), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(2, 4096, 32)"},
        },
    ),
    (
        Reshape988,
        [((1, 64, 4096), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 64, 64, 64)"},
        },
    ),
    (
        Reshape991,
        [((1, 2, 256, 32), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(2, 256, 32)"},
        },
    ),
    (
        Reshape990,
        [((1, 2, 32, 256), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(2, 32, 256)"},
        },
    ),
    (
        Reshape992,
        [((2, 4096, 32), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 2, 4096, 32)"},
        },
    ),
    (
        Reshape993,
        [((1, 4096, 2, 32), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(4096, 64)"},
        },
    ),
    (
        Reshape994,
        [((4096, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 4096, 64)"},
        },
    ),
    (
        Reshape1196,
        [((1, 256, 64, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 256, 4096, 1)"},
        },
    ),
    (
        Reshape1197,
        [((1, 160, 32, 32), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 160, 1024, 1)"},
        },
    ),
    (
        Reshape996,
        [((1, 1024, 160), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 1024, 5, 32)"},
        },
    ),
    (
        Reshape997,
        [((1, 1024, 160), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 32, 32, 160)"},
        },
    ),
    (
        Reshape998,
        [((1, 5, 1024, 32), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(5, 1024, 32)"},
        },
    ),
    (
        Reshape999,
        [((1, 160, 1024), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 160, 32, 32)"},
        },
    ),
    (
        Reshape1000,
        [((1, 160, 16, 16), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 160, 256)"},
        },
    ),
    (
        Reshape1001,
        [((1, 256, 160), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(256, 160)"},
        },
    ),
    (
        Reshape1002,
        [((1, 256, 160), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 256, 5, 32)"},
        },
    ),
    (
        Reshape1003,
        [((256, 160), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 256, 160)"},
        },
    ),
    (
        Reshape1005,
        [((1, 5, 256, 32), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(5, 256, 32)"},
        },
    ),
    (
        Reshape1004,
        [((1, 5, 32, 256), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(5, 32, 256)"},
        },
    ),
    (
        Reshape1006,
        [((5, 1024, 32), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 5, 1024, 32)"},
        },
    ),
    (
        Reshape1007,
        [((1, 1024, 5, 32), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1024, 160)"},
        },
    ),
    (
        Reshape1008,
        [((1024, 160), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 1024, 160)"},
        },
    ),
    (
        Reshape1009,
        [((1, 640, 1024), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 640, 32, 32)"},
        },
    ),
    (
        Reshape1198,
        [((1, 640, 32, 32), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 640, 1024, 1)"},
        },
    ),
    (
        Reshape1199,
        [((1, 256, 16, 16), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 256, 256, 1)"},
        },
    ),
    (
        Reshape334,
        [((256, 256), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 256, 256)"},
        },
    ),
    (
        Reshape1011,
        [((1, 8, 256, 32), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(8, 256, 32)"},
        },
    ),
    (
        Reshape1012,
        [((1, 8, 32, 256), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(8, 32, 256)"},
        },
    ),
    (
        Reshape1013,
        [((8, 256, 32), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 8, 256, 32)"},
        },
    ),
    (
        Reshape332,
        [((1, 256, 8, 32), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(256, 256)"},
        },
    ),
    (
        Reshape1014,
        [((1, 1024, 256), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 1024, 16, 16)"},
        },
    ),
    (
        Reshape1200,
        [((1, 1024, 16, 16), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 1024, 256, 1)"},
        },
    ),
    (
        Reshape1201,
        [((1, 768, 128), torch.float32)],
        {
            "model_names": ["pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 12, 64, 128)"},
        },
    ),
    (
        Reshape1202,
        [((1, 768, 128), torch.float32)],
        {
            "model_names": ["pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(12, 64, 128)"},
        },
    ),
    (
        Reshape1203,
        [((1, 12, 64, 128), torch.float32)],
        {
            "model_names": ["pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 768, 128, 1)"},
        },
    ),
    (
        Reshape1204,
        [((768, 768, 1), torch.float32)],
        {
            "model_names": ["pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(768, 768, 1, 1)"},
        },
    ),
    (
        Reshape1205,
        [((1, 768, 128, 1), torch.float32)],
        {
            "model_names": ["pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 768, 128)"},
        },
    ),
    (
        Reshape1206,
        [((1, 64, 64, 128), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(1, 8, 8, 8, 8, 128)"},
        },
    ),
    (
        Reshape119,
        [((1, 64, 64, 128), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(4096, 128)"}},
    ),
    (
        Reshape119,
        [((1, 8, 8, 8, 8, 128), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(4096, 128)"}},
    ),
    (
        Reshape107,
        [((1, 8, 8, 8, 8, 128), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(1, 64, 64, 128)"}},
    ),
    (
        Reshape1207,
        [((4096, 384), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(64, 64, 384)"}},
    ),
    (
        Reshape1208,
        [((4096, 384), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_s_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(1, 64, 64, 384)"}},
    ),
    (
        Reshape1209,
        [((64, 64, 384), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(64, 64, 3, 4, 32)"},
        },
    ),
    (
        Reshape1210,
        [((1, 64, 4, 64, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(64, 4, 64, 32)"}},
    ),
    (
        Reshape1211,
        [((64, 4, 64, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(256, 64, 32)"}},
    ),
    (
        Reshape1212,
        [((256, 64, 64), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(64, 4, 64, 64)"}},
    ),
    (
        Reshape1213,
        [((1, 15, 15, 2), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision", "pt_swin_swin_v2_s_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(225, 2)"},
        },
    ),
    (
        Reshape1214,
        [((225, 4), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(225, 4)"}},
    ),
    (
        Reshape1215,
        [((4096, 4), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(64, 64, 4)"}},
    ),
    (
        Reshape1216,
        [((64, 4, 64, 64), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(256, 64, 64)"}},
    ),
    (
        Reshape1217,
        [((64, 4, 64, 64), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(1, 64, 4, 64, 64)"},
        },
    ),
    (
        Reshape1218,
        [((64, 4, 32, 64), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(256, 32, 64)"}},
    ),
    (
        Reshape1210,
        [((256, 64, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(64, 4, 64, 32)"}},
    ),
    (
        Reshape119,
        [((64, 64, 4, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(4096, 128)"}},
    ),
    (
        Reshape1206,
        [((64, 64, 128), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(1, 8, 8, 8, 8, 128)"},
        },
    ),
    (
        Reshape1219,
        [((4096, 512), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(1, 64, 64, 512)"}},
    ),
    (
        Reshape1220,
        [((1, 64, 64, 512), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(4096, 512)"}},
    ),
    (
        Reshape1212,
        [((1, 64, 4, 64, 64), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(64, 4, 64, 64)"}},
    ),
    (
        Reshape1221,
        [((1, 32, 32, 512), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(1024, 512)"}},
    ),
    (
        Reshape1222,
        [((1024, 256), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(1, 32, 32, 256)"}},
    ),
    (
        Reshape611,
        [((1024, 256), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(16, 64, 256)"}},
    ),
    (
        Reshape1223,
        [((1, 32, 32, 256), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(1, 4, 8, 4, 8, 256)"},
        },
    ),
    (
        Reshape1224,
        [((1, 32, 32, 256), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(1024, 256)"}},
    ),
    (
        Reshape1224,
        [((1, 4, 4, 8, 8, 256), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(1024, 256)"}},
    ),
    (
        Reshape1225,
        [((1024, 768), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(16, 64, 768)"}},
    ),
    (
        Reshape1226,
        [((1024, 768), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_s_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(1, 32, 32, 768)"}},
    ),
    (
        Reshape1227,
        [((16, 64, 768), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(16, 64, 3, 8, 32)"},
        },
    ),
    (
        Reshape1228,
        [((1, 16, 8, 64, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(16, 8, 64, 32)"}},
    ),
    (
        Reshape1229,
        [((16, 8, 64, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(128, 64, 32)"}},
    ),
    (
        Reshape1230,
        [((128, 64, 64), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(16, 8, 64, 64)"}},
    ),
    (
        Reshape1231,
        [((225, 8), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(225, 8)"}},
    ),
    (
        Reshape1232,
        [((4096, 8), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(64, 64, 8)"}},
    ),
    (
        Reshape1233,
        [((16, 8, 64, 64), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(128, 64, 64)"}},
    ),
    (
        Reshape1234,
        [((16, 8, 64, 64), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(1, 16, 8, 64, 64)"},
        },
    ),
    (
        Reshape1235,
        [((16, 8, 32, 64), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(128, 32, 64)"}},
    ),
    (
        Reshape1228,
        [((128, 64, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(16, 8, 64, 32)"}},
    ),
    (
        Reshape1224,
        [((16, 64, 8, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(1024, 256)"}},
    ),
    (
        Reshape1236,
        [((16, 64, 256), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(1, 4, 4, 8, 8, 256)"},
        },
    ),
    (
        Reshape1222,
        [((1, 4, 8, 4, 8, 256), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(1, 32, 32, 256)"}},
    ),
    (
        Reshape1237,
        [((1024, 1024), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(1, 32, 32, 1024)"}},
    ),
    (
        Reshape1238,
        [((1, 32, 32, 1024), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(1024, 1024)"}},
    ),
    (
        Reshape1230,
        [((1, 16, 8, 64, 64), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(16, 8, 64, 64)"}},
    ),
    (
        Reshape594,
        [((1, 16, 16, 1024), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(256, 1024)"}},
    ),
    (
        Reshape1239,
        [((1, 16, 16, 512), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(1, 2, 8, 2, 8, 512)"},
        },
    ),
    (
        Reshape144,
        [((1, 16, 16, 512), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(256, 512)"}},
    ),
    (
        Reshape144,
        [((1, 2, 2, 8, 8, 512), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(256, 512)"}},
    ),
    (
        Reshape1240,
        [((256, 1536), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(4, 64, 1536)"}},
    ),
    (
        Reshape1241,
        [((256, 1536), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_s_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(1, 16, 16, 1536)"}},
    ),
    (
        Reshape1242,
        [((4, 64, 1536), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(4, 64, 3, 16, 32)"},
        },
    ),
    (
        Reshape1243,
        [((1, 4, 16, 64, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(4, 16, 64, 32)"}},
    ),
    (
        Reshape1244,
        [((4, 16, 64, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(64, 64, 32)"}},
    ),
    (
        Reshape1245,
        [((64, 64, 64), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(4, 16, 64, 64)"}},
    ),
    (
        Reshape1246,
        [((225, 16), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(225, 16)"}},
    ),
    (
        Reshape1247,
        [((4096, 16), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(64, 64, 16)"}},
    ),
    (
        Reshape1248,
        [((4, 16, 64, 64), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(64, 64, 64)"}},
    ),
    (
        Reshape1249,
        [((4, 16, 64, 64), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(1, 4, 16, 64, 64)"},
        },
    ),
    (
        Reshape1250,
        [((4, 16, 32, 64), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(64, 32, 64)"}},
    ),
    (
        Reshape1243,
        [((64, 64, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(4, 16, 64, 32)"}},
    ),
    (
        Reshape144,
        [((4, 64, 16, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(256, 512)"}},
    ),
    (
        Reshape1251,
        [((4, 64, 512), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(1, 2, 2, 8, 8, 512)"},
        },
    ),
    (
        Reshape146,
        [((1, 2, 8, 2, 8, 512), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(1, 16, 16, 512)"}},
    ),
    (
        Reshape861,
        [((1, 16, 16, 2048), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(256, 2048)"}},
    ),
    (
        Reshape1245,
        [((1, 4, 16, 64, 64), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(4, 16, 64, 64)"}},
    ),
    (
        Reshape1252,
        [((1, 8, 8, 2048), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(64, 2048)"}},
    ),
    (
        Reshape1253,
        [((64, 1024), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(1, 8, 8, 1024)"}},
    ),
    (
        Reshape1254,
        [((64, 1024), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(1, 64, 1024)"}},
    ),
    (
        Reshape1255,
        [((1, 8, 8, 1024), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(1, 1, 8, 1, 8, 1024)"},
        },
    ),
    (
        Reshape1256,
        [((1, 8, 8, 1024), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(64, 1024)"}},
    ),
    (
        Reshape1256,
        [((1, 1, 1, 8, 8, 1024), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(64, 1024)"}},
    ),
    (
        Reshape1257,
        [((64, 3072), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(1, 64, 3072)"}},
    ),
    (
        Reshape1258,
        [((64, 3072), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_s_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(1, 8, 8, 3072)"}},
    ),
    (
        Reshape1259,
        [((1, 64, 3072), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(1, 64, 3, 32, 32)"},
        },
    ),
    (
        Reshape1260,
        [((1, 1, 32, 64, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(1, 32, 64, 32)"}},
    ),
    (
        Reshape1261,
        [((1, 32, 64, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(32, 64, 32)"}},
    ),
    (
        Reshape1262,
        [((32, 64, 64), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(1, 32, 64, 64)"}},
    ),
    (
        Reshape1263,
        [((225, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(225, 32)"}},
    ),
    (
        Reshape1244,
        [((4096, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(64, 64, 32)"}},
    ),
    (
        Reshape1264,
        [((1, 32, 64, 64), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(32, 64, 64)"}},
    ),
    (
        Reshape1260,
        [((32, 64, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(1, 32, 64, 32)"}},
    ),
    (
        Reshape1256,
        [((1, 64, 32, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(64, 1024)"}},
    ),
    (
        Reshape1265,
        [((1, 64, 1024), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(1, 1, 1, 8, 8, 1024)"},
        },
    ),
    (
        Reshape1253,
        [((1, 1, 8, 1, 8, 1024), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(1, 8, 8, 1024)"}},
    ),
    (
        Reshape1266,
        [((64, 4096), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(1, 8, 8, 4096)"}},
    ),
    (
        Reshape1267,
        [((1, 8, 8, 4096), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(64, 4096)"}},
    ),
    (
        Reshape1268,
        [((4096, 288), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_s_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(64, 64, 288)"}},
    ),
    (
        Reshape1269,
        [((64, 64, 288), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(64, 64, 3, 3, 32)"},
        },
    ),
    (
        Reshape1032,
        [((1, 64, 3, 64, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_s_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(64, 3, 64, 32)"}},
    ),
    (
        Reshape1270,
        [((1, 64, 64, 384), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_s_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(4096, 384)"}},
    ),
    (
        Reshape1271,
        [((1024, 576), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_s_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(16, 64, 576)"}},
    ),
    (
        Reshape1272,
        [((16, 64, 576), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(16, 64, 3, 6, 32)"},
        },
    ),
    (
        Reshape1048,
        [((1, 16, 6, 64, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_s_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(16, 6, 64, 32)"}},
    ),
    (
        Reshape1273,
        [((1, 32, 32, 768), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_s_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(1024, 768)"}},
    ),
    (
        Reshape1274,
        [((256, 1152), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_s_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(4, 64, 1152)"}},
    ),
    (
        Reshape1275,
        [((4, 64, 1152), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(4, 64, 3, 12, 32)"},
        },
    ),
    (
        Reshape1063,
        [((1, 4, 12, 64, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_s_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(4, 12, 64, 32)"}},
    ),
    (
        Reshape1276,
        [((1, 16, 16, 1536), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_s_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(256, 1536)"}},
    ),
    (
        Reshape1277,
        [((64, 2304), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_s_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(1, 64, 2304)"}},
    ),
    (
        Reshape1278,
        [((1, 64, 2304), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(1, 64, 3, 24, 32)"},
        },
    ),
    (
        Reshape1077,
        [((1, 1, 24, 64, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_s_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(1, 24, 64, 32)"}},
    ),
    (
        Reshape1067,
        [((1, 1, 8, 1, 8, 768), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_s_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(1, 8, 8, 768)"}},
    ),
    (
        Reshape1279,
        [((1, 8, 8, 3072), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_s_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(64, 3072)"}},
    ),
    (
        Reshape803,
        [((1, 768, 1, 1), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_s_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(1, 768, 1, 1)"}},
    ),
    (
        Reshape1280,
        [((1, 204, 768), torch.bfloat16)],
        {
            "model_names": ["pt_vilt_dandelin_vilt_b32_mlm_mlm_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(204, 768)"},
        },
    ),
    (
        Reshape1281,
        [((1, 204, 768), torch.bfloat16)],
        {
            "model_names": ["pt_vilt_dandelin_vilt_b32_mlm_mlm_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 204, 12, 64)"},
        },
    ),
    (
        Reshape1282,
        [((204, 768), torch.bfloat16)],
        {
            "model_names": ["pt_vilt_dandelin_vilt_b32_mlm_mlm_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 204, 768)"},
        },
    ),
    (
        Reshape1283,
        [((1, 12, 204, 64), torch.bfloat16)],
        {
            "model_names": ["pt_vilt_dandelin_vilt_b32_mlm_mlm_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(12, 204, 64)"},
        },
    ),
    (
        Reshape1284,
        [((12, 204, 204), torch.bfloat16)],
        {
            "model_names": ["pt_vilt_dandelin_vilt_b32_mlm_mlm_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 12, 204, 204)"},
        },
    ),
    (
        Reshape1285,
        [((1, 12, 204, 204), torch.bfloat16)],
        {
            "model_names": ["pt_vilt_dandelin_vilt_b32_mlm_mlm_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(12, 204, 204)"},
        },
    ),
    (
        Reshape1286,
        [((1, 12, 64, 204), torch.bfloat16)],
        {
            "model_names": ["pt_vilt_dandelin_vilt_b32_mlm_mlm_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(12, 64, 204)"},
        },
    ),
    (
        Reshape1287,
        [((12, 204, 64), torch.bfloat16)],
        {
            "model_names": ["pt_vilt_dandelin_vilt_b32_mlm_mlm_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 12, 204, 64)"},
        },
    ),
    (
        Reshape1280,
        [((1, 204, 12, 64), torch.bfloat16)],
        {
            "model_names": ["pt_vilt_dandelin_vilt_b32_mlm_mlm_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(204, 768)"},
        },
    ),
    (
        Reshape1288,
        [((1, 1280, 37, 37), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_h_14_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 1280, 1369)"},
        },
    ),
    (
        Reshape1289,
        [((1370, 1, 1280), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_h_14_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1370, 1280)"},
        },
    ),
    (
        Reshape1290,
        [((1370, 3840), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_h_14_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1370, 1, 3840)"},
        },
    ),
    (
        Reshape1291,
        [((1370, 1, 3840), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_h_14_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1370, 1, 3, 1280)"},
        },
    ),
    (
        Reshape1292,
        [((1, 1370, 1, 1280), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_h_14_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1370, 16, 80)"},
        },
    ),
    (
        Reshape1293,
        [((16, 1370, 80), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_h_14_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(16, 1370, 80)"},
        },
    ),
    (
        Reshape1294,
        [((16, 1370, 80), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_h_14_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 16, 1370, 80)"},
        },
    ),
    (
        Reshape1295,
        [((16, 1370, 1370), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_h_14_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 16, 1370, 1370)"},
        },
    ),
    (
        Reshape1296,
        [((1, 16, 1370, 1370), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_h_14_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(16, 1370, 1370)"},
        },
    ),
    (
        Reshape1289,
        [((1370, 1, 16, 80), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_h_14_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1370, 1280)"},
        },
    ),
    (
        Reshape1297,
        [((1370, 1280), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_h_14_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1370, 1, 1280)"},
        },
    ),
    (
        Reshape372,
        [((1, 1, 1280), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_h_14_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 1280)"},
        },
    ),
    (
        Reshape252,
        [((1, 2), torch.int64)],
        {"model_names": ["pt_whisper_openai_whisper_large_v3_clm_hf"], "pcc": 0.99, "args": {"shape": "(1, 2)"}},
    ),
    (
        Reshape1298,
        [((1, 2, 1280), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_large_v3_clm_hf"], "pcc": 0.99, "args": {"shape": "(2, 1280)"}},
    ),
    (
        Reshape1299,
        [((1, 2, 1280), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_large_v3_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 2, 20, 64)"},
        },
    ),
    (
        Reshape1300,
        [((2, 1280), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_large_v3_clm_hf"], "pcc": 0.99, "args": {"shape": "(1, 2, 1280)"}},
    ),
    (
        Reshape1299,
        [((2, 1280), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_large_v3_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 2, 20, 64)"},
        },
    ),
    (
        Reshape1301,
        [((1, 20, 2, 64), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_large_v3_clm_hf"], "pcc": 0.99, "args": {"shape": "(20, 2, 64)"}},
    ),
    (
        Reshape1302,
        [((20, 2, 2), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_large_v3_clm_hf"], "pcc": 0.99, "args": {"shape": "(1, 20, 2, 2)"}},
    ),
    (
        Reshape1303,
        [((1, 20, 2, 2), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_large_v3_clm_hf"], "pcc": 0.99, "args": {"shape": "(20, 2, 2)"}},
    ),
    (
        Reshape1304,
        [((20, 2, 64), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_large_v3_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 20, 2, 64)"},
        },
    ),
    (
        Reshape1298,
        [((1, 2, 20, 64), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_large_v3_clm_hf"], "pcc": 0.99, "args": {"shape": "(2, 1280)"}},
    ),
    (
        Reshape1305,
        [((1, 128, 3000), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_large_v3_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 128, 3000, 1)"},
        },
    ),
    (
        Reshape1306,
        [((1280, 128, 3), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_large_v3_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1280, 128, 3, 1)"},
        },
    ),
    (
        Reshape1307,
        [((1, 1280, 3000, 1), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_large_v3_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 1280, 3000)"},
        },
    ),
    (
        Reshape1308,
        [((1, 1280, 3000), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_large_v3_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 1280, 3000, 1)"},
        },
    ),
    (
        Reshape1309,
        [((1280, 1280, 3), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_large_v3_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1280, 1280, 3, 1)"},
        },
    ),
    (
        Reshape1310,
        [((1, 1280, 1500, 1), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_large_v3_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 1280, 1500)"},
        },
    ),
    (
        Reshape1311,
        [((1, 1500, 1280), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_large_v3_clm_hf"], "pcc": 0.99, "args": {"shape": "(1500, 1280)"}},
    ),
    (
        Reshape1312,
        [((1, 1500, 1280), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_large_v3_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 1500, 20, 64)"},
        },
    ),
    (
        Reshape1313,
        [((1500, 1280), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_large_v3_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 1500, 1280)"},
        },
    ),
    (
        Reshape1312,
        [((1500, 1280), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_large_v3_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 1500, 20, 64)"},
        },
    ),
    (
        Reshape1314,
        [((1, 20, 1500, 64), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_large_v3_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(20, 1500, 64)"},
        },
    ),
    (
        Reshape1315,
        [((20, 1500, 1500), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_large_v3_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 20, 1500, 1500)"},
        },
    ),
    (
        Reshape1316,
        [((1, 20, 1500, 1500), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_large_v3_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(20, 1500, 1500)"},
        },
    ),
    (
        Reshape1317,
        [((20, 1500, 64), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_large_v3_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 20, 1500, 64)"},
        },
    ),
    (
        Reshape1311,
        [((1, 1500, 20, 64), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_large_v3_clm_hf"], "pcc": 0.99, "args": {"shape": "(1500, 1280)"}},
    ),
    (
        Reshape1318,
        [((20, 2, 1500), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_large_v3_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 20, 2, 1500)"},
        },
    ),
    (
        Reshape1319,
        [((1, 20, 2, 1500), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_large_v3_clm_hf"], "pcc": 0.99, "args": {"shape": "(20, 2, 1500)"}},
    ),
    (
        Reshape1320,
        [((768, 80, 3), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(768, 80, 3, 1)"},
        },
    ),
    (
        Reshape1321,
        [((1, 768, 3000, 1), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 768, 3000)"},
        },
    ),
    (
        Reshape1322,
        [((1, 768, 3000), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 768, 3000, 1)"},
        },
    ),
    (
        Reshape1323,
        [((768, 768, 3), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(768, 768, 3, 1)"},
        },
    ),
    (
        Reshape1324,
        [((1, 768, 1500, 1), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 768, 1500)"},
        },
    ),
    (
        Reshape1325,
        [((1, 1500, 768), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1500, 768)"},
        },
    ),
    (
        Reshape1326,
        [((1, 1500, 768), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 1500, 12, 64)"},
        },
    ),
    (
        Reshape1327,
        [((1500, 768), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 1500, 768)"},
        },
    ),
    (
        Reshape1326,
        [((1500, 768), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 1500, 12, 64)"},
        },
    ),
    (
        Reshape1328,
        [((1, 12, 1500, 64), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(12, 1500, 64)"},
        },
    ),
    (
        Reshape1329,
        [((12, 1500, 1500), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 12, 1500, 1500)"},
        },
    ),
    (
        Reshape1330,
        [((1, 12, 1500, 1500), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(12, 1500, 1500)"},
        },
    ),
    (
        Reshape1331,
        [((12, 1500, 64), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 12, 1500, 64)"},
        },
    ),
    (
        Reshape1325,
        [((1, 1500, 12, 64), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1500, 768)"},
        },
    ),
    (
        Reshape1332,
        [((12, 1, 1500), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 12, 1, 1500)"},
        },
    ),
    (
        Reshape1333,
        [((1, 12, 1, 1500), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(12, 1, 1500)"},
        },
    ),
    (
        Reshape1334,
        [((1, 3, 85, 160, 160), torch.float32)],
        {
            "model_names": ["pt_yolo_v5_yolov5s_img_cls_torchhub_1280x1280"],
            "pcc": 0.99,
            "args": {"shape": "(1, 255, 160, 160)"},
        },
    ),
    (
        Reshape1335,
        [((1, 255, 160, 160), torch.float32)],
        {
            "model_names": ["pt_yolo_v5_yolov5s_img_cls_torchhub_1280x1280"],
            "pcc": 0.99,
            "args": {"shape": "(1, 1, 255, 25600)"},
        },
    ),
    (
        Reshape1336,
        [((1, 1, 255, 25600), torch.float32)],
        {
            "model_names": ["pt_yolo_v5_yolov5s_img_cls_torchhub_1280x1280"],
            "pcc": 0.99,
            "args": {"shape": "(1, 3, 85, 25600)"},
        },
    ),
    (
        Reshape1337,
        [((1, 3, 25600, 85), torch.float32)],
        {
            "model_names": ["pt_yolo_v5_yolov5s_img_cls_torchhub_1280x1280"],
            "pcc": 0.99,
            "args": {"shape": "(1, 76800, 85)"},
        },
    ),
    (
        Reshape1338,
        [((1, 68, 56, 80), torch.bfloat16)],
        {
            "model_names": ["pt_yolo_v6_yolov6l_obj_det_torchhub"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 4, 17, 4480)"},
        },
    ),
    (
        Reshape350,
        [((1, 1, 4, 4480), torch.bfloat16)],
        {
            "model_names": ["pt_yolo_v6_yolov6l_obj_det_torchhub"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 4, 4480)"},
        },
    ),
    (
        Reshape1339,
        [((1, 68, 28, 40), torch.bfloat16)],
        {
            "model_names": ["pt_yolo_v6_yolov6l_obj_det_torchhub"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 4, 17, 1120)"},
        },
    ),
    (
        Reshape351,
        [((1, 1, 4, 1120), torch.bfloat16)],
        {
            "model_names": ["pt_yolo_v6_yolov6l_obj_det_torchhub"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 4, 1120)"},
        },
    ),
    (
        Reshape1340,
        [((1, 68, 14, 20), torch.bfloat16)],
        {
            "model_names": ["pt_yolo_v6_yolov6l_obj_det_torchhub"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 4, 17, 280)"},
        },
    ),
    (
        Reshape352,
        [((1, 1, 4, 280), torch.bfloat16)],
        {
            "model_names": ["pt_yolo_v6_yolov6l_obj_det_torchhub"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 4, 280)"},
        },
    ),
    (
        Reshape1341,
        [((1, 192, 32, 42), torch.bfloat16)],
        {
            "model_names": ["pt_yolos_hustvl_yolos_tiny_obj_det_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 192, 1344, 1)"},
        },
    ),
    (
        Reshape1342,
        [((1, 1, 192), torch.bfloat16)],
        {
            "model_names": ["pt_yolos_hustvl_yolos_tiny_obj_det_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 192)"},
        },
    ),
    (
        Reshape1343,
        [((1, 192, 4150), torch.bfloat16)],
        {
            "model_names": ["pt_yolos_hustvl_yolos_tiny_obj_det_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 192, 50, 83)"},
        },
    ),
    (
        Reshape1344,
        [((1, 1445, 192), torch.bfloat16)],
        {
            "model_names": ["pt_yolos_hustvl_yolos_tiny_obj_det_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1445, 192)"},
        },
    ),
    (
        Reshape1345,
        [((1, 1445, 192), torch.bfloat16)],
        {
            "model_names": ["pt_yolos_hustvl_yolos_tiny_obj_det_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 1445, 3, 64)"},
        },
    ),
    (
        Reshape1346,
        [((1445, 192), torch.bfloat16)],
        {
            "model_names": ["pt_yolos_hustvl_yolos_tiny_obj_det_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 1445, 192)"},
        },
    ),
    (
        Reshape1347,
        [((1, 3, 1445, 64), torch.bfloat16)],
        {
            "model_names": ["pt_yolos_hustvl_yolos_tiny_obj_det_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(3, 1445, 64)"},
        },
    ),
    (
        Reshape1348,
        [((3, 1445, 1445), torch.bfloat16)],
        {
            "model_names": ["pt_yolos_hustvl_yolos_tiny_obj_det_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 3, 1445, 1445)"},
        },
    ),
    (
        Reshape1349,
        [((1, 3, 1445, 1445), torch.bfloat16)],
        {
            "model_names": ["pt_yolos_hustvl_yolos_tiny_obj_det_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(3, 1445, 1445)"},
        },
    ),
    (
        Reshape1350,
        [((3, 1445, 64), torch.bfloat16)],
        {
            "model_names": ["pt_yolos_hustvl_yolos_tiny_obj_det_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 3, 1445, 64)"},
        },
    ),
    (
        Reshape1344,
        [((1, 1445, 3, 64), torch.bfloat16)],
        {
            "model_names": ["pt_yolos_hustvl_yolos_tiny_obj_det_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1445, 192)"},
        },
    ),
    (
        Reshape1351,
        [((1, 100, 192), torch.bfloat16)],
        {
            "model_names": ["pt_yolos_hustvl_yolos_tiny_obj_det_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(100, 192)"},
        },
    ),
    (
        Reshape1352,
        [((100, 192), torch.bfloat16)],
        {
            "model_names": ["pt_yolos_hustvl_yolos_tiny_obj_det_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 100, 192)"},
        },
    ),
    (
        Reshape1353,
        [((1, 1408, 1, 1), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"],
            "pcc": 0.99,
            "args": {"shape": "(1, 1408)"},
        },
    ),
    (
        Reshape1354,
        [((1, 1792, 1, 1), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "args": {"shape": "(1, 1792)"},
        },
    ),
    (
        Reshape860,
        [((1, 256, 6, 6), torch.float32)],
        {"model_names": ["pd_alexnet_base_img_cls_paddlemodels"], "pcc": 0.99, "args": {"shape": "(1, 9216, 1, 1)"}},
    ),
    (
        Reshape1355,
        [((1, 120, 1, 12), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"shape": "(1, 120, 12, 1)"},
        },
    ),
    (
        Reshape1356,
        [((1, 12, 360), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"shape": "(1, 12, 3, 8, 15)"},
        },
    ),
    (
        Reshape1357,
        [((1, 8, 12, 15), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"shape": "(8, 12, 15)"},
        },
    ),
    (
        Reshape1358,
        [((1, 8, 15, 12), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"shape": "(8, 15, 12)"},
        },
    ),
    (
        Reshape1359,
        [((8, 12, 12), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"shape": "(1, 8, 12, 12)"},
        },
    ),
    (
        Reshape1360,
        [((1, 8, 12, 12), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"shape": "(8, 12, 12)"},
        },
    ),
    (
        Reshape1361,
        [((8, 12, 15), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"shape": "(1, 8, 12, 15)"},
        },
    ),
    (
        Reshape1362,
        [((1, 12, 8, 15), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"shape": "(12, 120)"},
        },
    ),
    (
        Reshape1363,
        [((12, 120), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"shape": "(1, 12, 120)"},
        },
    ),
    (
        Reshape1364,
        [((1, 12, 120), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"shape": "(1, 1, 12, 120)"},
        },
    ),
    (
        Reshape1365,
        [((1, 522, 2048), torch.float32)],
        {"model_names": ["pt_falcon3_tiiuae_falcon3_1b_base_clm_hf"], "pcc": 0.99, "args": {"shape": "(522, 2048)"}},
    ),
    (
        Reshape1366,
        [((522, 2048), torch.float32)],
        {
            "model_names": ["pt_falcon3_tiiuae_falcon3_1b_base_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 522, 8, 256)"},
        },
    ),
    (
        Reshape1367,
        [((522, 2048), torch.float32)],
        {"model_names": ["pt_falcon3_tiiuae_falcon3_1b_base_clm_hf"], "pcc": 0.99, "args": {"shape": "(1, 522, 2048)"}},
    ),
    (
        Reshape1368,
        [((1, 8, 522, 256), torch.float32)],
        {"model_names": ["pt_falcon3_tiiuae_falcon3_1b_base_clm_hf"], "pcc": 0.99, "args": {"shape": "(8, 522, 256)"}},
    ),
    (
        Reshape1369,
        [((522, 1024), torch.float32)],
        {
            "model_names": ["pt_falcon3_tiiuae_falcon3_1b_base_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 522, 4, 256)"},
        },
    ),
    (
        Reshape1368,
        [((1, 4, 2, 522, 256), torch.float32)],
        {"model_names": ["pt_falcon3_tiiuae_falcon3_1b_base_clm_hf"], "pcc": 0.99, "args": {"shape": "(8, 522, 256)"}},
    ),
    (
        Reshape1370,
        [((8, 522, 522), torch.float32)],
        {
            "model_names": ["pt_falcon3_tiiuae_falcon3_1b_base_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 8, 522, 522)"},
        },
    ),
    (
        Reshape1371,
        [((1, 8, 522, 522), torch.float32)],
        {"model_names": ["pt_falcon3_tiiuae_falcon3_1b_base_clm_hf"], "pcc": 0.99, "args": {"shape": "(8, 522, 522)"}},
    ),
    (
        Reshape1372,
        [((8, 522, 256), torch.float32)],
        {
            "model_names": ["pt_falcon3_tiiuae_falcon3_1b_base_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 8, 522, 256)"},
        },
    ),
    (
        Reshape1365,
        [((1, 522, 8, 256), torch.float32)],
        {"model_names": ["pt_falcon3_tiiuae_falcon3_1b_base_clm_hf"], "pcc": 0.99, "args": {"shape": "(522, 2048)"}},
    ),
    (
        Reshape1373,
        [((522, 8192), torch.float32)],
        {"model_names": ["pt_falcon3_tiiuae_falcon3_1b_base_clm_hf"], "pcc": 0.99, "args": {"shape": "(1, 522, 8192)"}},
    ),
    (
        Reshape1374,
        [((1, 207, 2304), torch.float32)],
        {"model_names": ["pt_gemma_google_gemma_2_2b_it_qa_hf"], "pcc": 0.99, "args": {"shape": "(207, 2304)"}},
    ),
    (
        Reshape1375,
        [((207, 2048), torch.float32)],
        {"model_names": ["pt_gemma_google_gemma_2_2b_it_qa_hf"], "pcc": 0.99, "args": {"shape": "(1, 207, 8, 256)"}},
    ),
    (
        Reshape1376,
        [((1, 8, 207, 256), torch.float32)],
        {"model_names": ["pt_gemma_google_gemma_2_2b_it_qa_hf"], "pcc": 0.99, "args": {"shape": "(8, 207, 256)"}},
    ),
    (
        Reshape1377,
        [((207, 1024), torch.float32)],
        {"model_names": ["pt_gemma_google_gemma_2_2b_it_qa_hf"], "pcc": 0.99, "args": {"shape": "(1, 207, 4, 256)"}},
    ),
    (
        Reshape1376,
        [((1, 4, 2, 207, 256), torch.float32)],
        {"model_names": ["pt_gemma_google_gemma_2_2b_it_qa_hf"], "pcc": 0.99, "args": {"shape": "(8, 207, 256)"}},
    ),
    (
        Reshape1378,
        [((1, 4, 2, 207, 256), torch.float32)],
        {"model_names": ["pt_gemma_google_gemma_2_2b_it_qa_hf"], "pcc": 0.99, "args": {"shape": "(1, 8, 207, 256)"}},
    ),
    (
        Reshape1379,
        [((8, 207, 207), torch.float32)],
        {"model_names": ["pt_gemma_google_gemma_2_2b_it_qa_hf"], "pcc": 0.99, "args": {"shape": "(1, 8, 207, 207)"}},
    ),
    (
        Reshape1380,
        [((1, 8, 207, 207), torch.float32)],
        {"model_names": ["pt_gemma_google_gemma_2_2b_it_qa_hf"], "pcc": 0.99, "args": {"shape": "(8, 207, 207)"}},
    ),
    (
        Reshape1381,
        [((1, 8, 256, 207), torch.float32)],
        {"model_names": ["pt_gemma_google_gemma_2_2b_it_qa_hf"], "pcc": 0.99, "args": {"shape": "(8, 256, 207)"}},
    ),
    (
        Reshape1378,
        [((8, 207, 256), torch.float32)],
        {"model_names": ["pt_gemma_google_gemma_2_2b_it_qa_hf"], "pcc": 0.99, "args": {"shape": "(1, 8, 207, 256)"}},
    ),
    (
        Reshape1382,
        [((1, 207, 8, 256), torch.float32)],
        {"model_names": ["pt_gemma_google_gemma_2_2b_it_qa_hf"], "pcc": 0.99, "args": {"shape": "(207, 2048)"}},
    ),
    (
        Reshape1383,
        [((207, 2304), torch.float32)],
        {"model_names": ["pt_gemma_google_gemma_2_2b_it_qa_hf"], "pcc": 0.99, "args": {"shape": "(1, 207, 2304)"}},
    ),
    (
        Reshape1384,
        [((207, 9216), torch.float32)],
        {"model_names": ["pt_gemma_google_gemma_2_2b_it_qa_hf"], "pcc": 0.99, "args": {"shape": "(1, 207, 9216)"}},
    ),
    (
        Reshape1385,
        [((1, 512, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_mlp_mixer_mixer_s16_224_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 512, 196, 1)"},
        },
    ),
    (
        Reshape252,
        [((1, 1, 2), torch.float32)],
        {"model_names": ["pt_phi1_microsoft_phi_1_seq_cls_hf"], "pcc": 0.99, "args": {"shape": "(1, 2)"}},
    ),
    (
        Reshape1386,
        [((1, 1008, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_x_3_2gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(1, 1008, 1, 1)"},
        },
    ),
    (
        Reshape1387,
        [((61, 2048), torch.float32)],
        {"model_names": ["pt_t5_google_flan_t5_base_text_gen_hf"], "pcc": 0.99, "args": {"shape": "(1, 61, 2048)"}},
    ),
    (
        Reshape1388,
        [((1, 2048), torch.float32)],
        {"model_names": ["pt_t5_google_flan_t5_base_text_gen_hf"], "pcc": 0.99, "args": {"shape": "(1, 1, 2048)"}},
    ),
    (
        Reshape437,
        [((197, 1, 768), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_b_16_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(197, 768)"},
        },
    ),
    (
        Reshape1389,
        [((197, 2304), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_b_16_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(197, 1, 2304)"},
        },
    ),
    (
        Reshape1390,
        [((197, 1, 2304), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_b_16_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(197, 1, 3, 768)"},
        },
    ),
    (
        Reshape1391,
        [((1, 197, 1, 768), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_b_16_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(197, 12, 64)"},
        },
    ),
    (
        Reshape437,
        [((197, 1, 12, 64), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_b_16_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"shape": "(197, 768)"},
        },
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
