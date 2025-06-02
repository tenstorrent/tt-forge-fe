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
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1000))
        return reshape_output_1


class Reshape1(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1000, 1, 1))
        return reshape_output_1


class Reshape2(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1280))
        return reshape_output_1


class Reshape3(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1280, 1, 1))
        return reshape_output_1


class Reshape4(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(13, 384))
        return reshape_output_1


class Reshape5(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 13, 12, 32))
        return reshape_output_1


class Reshape6(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 13, 384))
        return reshape_output_1


class Reshape7(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 13, 32))
        return reshape_output_1


class Reshape8(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 32, 13))
        return reshape_output_1


class Reshape9(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 13, 13))
        return reshape_output_1


class Reshape10(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 13, 13))
        return reshape_output_1


class Reshape11(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 13, 32))
        return reshape_output_1


class Reshape12(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 384))
        return reshape_output_1


class Reshape13(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 6, 64))
        return reshape_output_1


class Reshape14(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 64, 16384))
        return reshape_output_1


class Reshape15(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 64, 16384, 1))
        return reshape_output_1


class Reshape16(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 128, 128))
        return reshape_output_1


class Reshape17(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16384, 1, 64))
        return reshape_output_1


class Reshape18(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 128, 64))
        return reshape_output_1


class Reshape19(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 64, 128, 128))
        return reshape_output_1


class Reshape20(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 64, 256))
        return reshape_output_1


class Reshape21(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(256, 64))
        return reshape_output_1


class Reshape22(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 1, 64))
        return reshape_output_1


class Reshape23(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 2, 32))
        return reshape_output_1


class Reshape24(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 64))
        return reshape_output_1


class Reshape25(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 16384, 256))
        return reshape_output_1


class Reshape26(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16384, 256))
        return reshape_output_1


class Reshape27(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 128, 128))
        return reshape_output_1


class Reshape28(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 16384))
        return reshape_output_1


class Reshape29(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 16384, 1))
        return reshape_output_1


class Reshape30(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 4096))
        return reshape_output_1


class Reshape31(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 4096, 1))
        return reshape_output_1


class Reshape32(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4096, 2, 64))
        return reshape_output_1


class Reshape33(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 64, 64, 128))
        return reshape_output_1


class Reshape34(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 4096, 64))
        return reshape_output_1


class Reshape35(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 64, 64))
        return reshape_output_1


class Reshape36(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(128, 4096))
        return reshape_output_1


class Reshape37(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 256))
        return reshape_output_1


class Reshape38(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(256, 128))
        return reshape_output_1


class Reshape39(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 2, 64))
        return reshape_output_1


class Reshape40(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 128))
        return reshape_output_1


class Reshape41(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 64, 256))
        return reshape_output_1


class Reshape42(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2, 4096, 256))
        return reshape_output_1


class Reshape43(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 4096, 256))
        return reshape_output_1


class Reshape44(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 256, 64))
        return reshape_output_1


class Reshape45(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2, 4096, 64))
        return reshape_output_1


class Reshape46(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(4096, 128))
        return reshape_output_1


class Reshape47(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4096, 128))
        return reshape_output_1


class Reshape48(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 64, 128))
        return reshape_output_1


class Reshape49(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 512, 64, 64))
        return reshape_output_1


class Reshape50(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 512, 4096))
        return reshape_output_1


class Reshape51(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 512, 4096, 1))
        return reshape_output_1


class Reshape52(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 320, 1024))
        return reshape_output_1


class Reshape53(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 320, 1024, 1))
        return reshape_output_1


class Reshape54(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1024, 5, 64))
        return reshape_output_1


class Reshape55(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 32, 320))
        return reshape_output_1


class Reshape56(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(5, 1024, 64))
        return reshape_output_1


class Reshape57(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 320, 32, 32))
        return reshape_output_1


class Reshape58(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 320, 256))
        return reshape_output_1


class Reshape59(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(256, 320))
        return reshape_output_1


class Reshape60(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 5, 64))
        return reshape_output_1


class Reshape61(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 320))
        return reshape_output_1


class Reshape62(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(5, 64, 256))
        return reshape_output_1


class Reshape63(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 5, 1024, 256))
        return reshape_output_1


class Reshape64(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(5, 1024, 256))
        return reshape_output_1


class Reshape65(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(5, 256, 64))
        return reshape_output_1


class Reshape66(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 5, 1024, 64))
        return reshape_output_1


class Reshape67(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1024, 320))
        return reshape_output_1


class Reshape68(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1024, 320))
        return reshape_output_1


class Reshape69(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1280, 32, 32))
        return reshape_output_1


class Reshape70(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1280, 1024))
        return reshape_output_1


class Reshape71(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1280, 1024, 1))
        return reshape_output_1


class Reshape72(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 512, 256))
        return reshape_output_1


class Reshape73(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 512, 256, 1))
        return reshape_output_1


class Reshape74(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(256, 512))
        return reshape_output_1


class Reshape75(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 8, 64))
        return reshape_output_1


class Reshape76(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 512))
        return reshape_output_1


class Reshape77(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 16, 512))
        return reshape_output_1


class Reshape78(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 512, 1))
        return reshape_output_1


class Reshape79(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(4, 64, 512))
        return reshape_output_1


class Reshape80(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 256, 64))
        return reshape_output_1


class Reshape81(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 64, 256))
        return reshape_output_1


class Reshape82(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 256, 256))
        return reshape_output_1


class Reshape83(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 256, 256))
        return reshape_output_1


class Reshape84(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 256, 64))
        return reshape_output_1


class Reshape85(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2048, 16, 16))
        return reshape_output_1


class Reshape86(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2048, 8, 32))
        return reshape_output_1


class Reshape87(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2048, 256))
        return reshape_output_1


class Reshape88(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2048, 256, 1))
        return reshape_output_1


class Reshape89(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 16, 16))
        return reshape_output_1


class Reshape90(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 8, 32))
        return reshape_output_1


class Reshape91(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(256, 256))
        return reshape_output_1


class Reshape92(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 256))
        return reshape_output_1


class Reshape93(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 16, 256))
        return reshape_output_1


class Reshape94(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 32, 32))
        return reshape_output_1


class Reshape95(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(256, 1024))
        return reshape_output_1


class Reshape96(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 16, 64))
        return reshape_output_1


class Reshape97(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 64, 64))
        return reshape_output_1


class Reshape98(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1024))
        return reshape_output_1


class Reshape99(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1024, 1, 1))
        return reshape_output_1


class Reshape100(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(14, 768))
        return reshape_output_1


class Reshape101(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 14, 12, 64))
        return reshape_output_1


class Reshape102(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 14, 768))
        return reshape_output_1


class Reshape103(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 14, 64))
        return reshape_output_1


class Reshape104(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 64, 14))
        return reshape_output_1


class Reshape105(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 14, 14))
        return reshape_output_1


class Reshape106(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 14, 14))
        return reshape_output_1


class Reshape107(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 14, 64))
        return reshape_output_1


class Reshape108(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 14, 768, 1))
        return reshape_output_1


class Reshape109(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2048, 1, 1))
        return reshape_output_1


class Reshape110(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2048))
        return reshape_output_1


class Reshape111(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(128, 768))
        return reshape_output_1


class Reshape112(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 12, 64))
        return reshape_output_1


class Reshape113(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 768))
        return reshape_output_1


class Reshape114(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 128, 64))
        return reshape_output_1


class Reshape115(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 128, 128))
        return reshape_output_1


class Reshape116(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 128, 128))
        return reshape_output_1


class Reshape117(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 64, 128))
        return reshape_output_1


class Reshape118(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 768, 128, 1))
        return reshape_output_1


class Reshape119(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 128, 64))
        return reshape_output_1


class Reshape120(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 768, 1))
        return reshape_output_1


class Reshape121(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(9, 768))
        return reshape_output_1


class Reshape122(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 9, 12, 64))
        return reshape_output_1


class Reshape123(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 9, 768))
        return reshape_output_1


class Reshape124(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 9, 64))
        return reshape_output_1


class Reshape125(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 9, 9))
        return reshape_output_1


class Reshape126(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 9, 9))
        return reshape_output_1


class Reshape127(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 64, 9))
        return reshape_output_1


class Reshape128(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 9, 64))
        return reshape_output_1


class Reshape129(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 9, 768, 1))
        return reshape_output_1


class Reshape130(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 768))
        return reshape_output_1


class Reshape131(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 12, 64))
        return reshape_output_1


class Reshape132(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 9216))
        return reshape_output_1


class Reshape133(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 9216, 1, 1))
        return reshape_output_1


class Reshape134(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 768, 196, 1))
        return reshape_output_1


class Reshape135(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 768, 196))
        return reshape_output_1


class Reshape136(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(197, 768))
        return reshape_output_1


class Reshape137(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 197, 12, 64))
        return reshape_output_1


class Reshape138(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 197, 768))
        return reshape_output_1


class Reshape139(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(197, 1, 768))
        return reshape_output_1


class Reshape140(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 197, 64))
        return reshape_output_1


class Reshape141(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 197, 197))
        return reshape_output_1


class Reshape142(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 197, 197))
        return reshape_output_1


class Reshape143(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 64, 197))
        return reshape_output_1


class Reshape144(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 197, 64))
        return reshape_output_1


class Reshape145(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(100, 256))
        return reshape_output_1


class Reshape146(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 100, 8, 32))
        return reshape_output_1


class Reshape147(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 100, 256))
        return reshape_output_1


class Reshape148(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 100, 32))
        return reshape_output_1


class Reshape149(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 100, 32))
        return reshape_output_1


class Reshape150(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 64, 1, 1))
        return reshape_output_1


class Reshape151(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64,))
        return reshape_output_1


class Reshape152(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 1, 64))
        return reshape_output_1


class Reshape153(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 1, 1))
        return reshape_output_1


class Reshape154(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(256,))
        return reshape_output_1


class Reshape155(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 1, 256))
        return reshape_output_1


class Reshape156(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 1, 1))
        return reshape_output_1


class Reshape157(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(128,))
        return reshape_output_1


class Reshape158(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 1, 128))
        return reshape_output_1


class Reshape159(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 512, 1, 1))
        return reshape_output_1


class Reshape160(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(512,))
        return reshape_output_1


class Reshape161(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 1, 512))
        return reshape_output_1


class Reshape162(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1024,))
        return reshape_output_1


class Reshape163(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 1, 1024))
        return reshape_output_1


class Reshape164(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2048,))
        return reshape_output_1


class Reshape165(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 1, 2048))
        return reshape_output_1


class Reshape166(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 280, 1))
        return reshape_output_1


class Reshape167(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 280))
        return reshape_output_1


class Reshape168(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 32, 280))
        return reshape_output_1


class Reshape169(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(280, 256))
        return reshape_output_1


class Reshape170(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 280, 8, 32))
        return reshape_output_1


class Reshape171(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 280, 256))
        return reshape_output_1


class Reshape172(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 280, 32))
        return reshape_output_1


class Reshape173(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 280, 280))
        return reshape_output_1


class Reshape174(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 280, 280))
        return reshape_output_1


class Reshape175(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 280, 32))
        return reshape_output_1


class Reshape176(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 100, 280))
        return reshape_output_1


class Reshape177(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 100, 280))
        return reshape_output_1


class Reshape178(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 1, 3, 3))
        return reshape_output_1


class Reshape179(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(144, 1, 3, 3))
        return reshape_output_1


class Reshape180(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(192, 1, 3, 3))
        return reshape_output_1


class Reshape181(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(192, 1, 5, 5))
        return reshape_output_1


class Reshape182(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(288, 1, 5, 5))
        return reshape_output_1


class Reshape183(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(288, 1, 3, 3))
        return reshape_output_1


class Reshape184(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(576, 1, 3, 3))
        return reshape_output_1


class Reshape185(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(576, 1, 5, 5))
        return reshape_output_1


class Reshape186(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(816, 1, 5, 5))
        return reshape_output_1


class Reshape187(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1392, 1, 5, 5))
        return reshape_output_1


class Reshape188(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1392, 1, 3, 3))
        return reshape_output_1


class Reshape189(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(207, 2304))
        return reshape_output_1


class Reshape190(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 207, 8, 256))
        return reshape_output_1


class Reshape191(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 207, 256))
        return reshape_output_1


class Reshape192(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 207, 4, 256))
        return reshape_output_1


class Reshape193(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 207, 256))
        return reshape_output_1


class Reshape194(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 207, 207))
        return reshape_output_1


class Reshape195(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 207, 207))
        return reshape_output_1


class Reshape196(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 256, 207))
        return reshape_output_1


class Reshape197(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(207, 2048))
        return reshape_output_1


class Reshape198(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 207, 2304))
        return reshape_output_1


class Reshape199(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 207, 9216))
        return reshape_output_1


class Reshape200(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 224, 224))
        return reshape_output_1


class Reshape201(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1536))
        return reshape_output_1


class Reshape202(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1536, 1, 1))
        return reshape_output_1


class Reshape203(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1024, 49, 1))
        return reshape_output_1


class Reshape204(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(96, 1, 3, 3))
        return reshape_output_1


class Reshape205(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(384, 1, 3, 3))
        return reshape_output_1


class Reshape206(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(960, 1, 3, 3))
        return reshape_output_1


class Reshape207(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 7))
        return reshape_output_1


class Reshape208(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(7, 768))
        return reshape_output_1


class Reshape209(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 7, 12, 64))
        return reshape_output_1


class Reshape210(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 7, 768))
        return reshape_output_1


class Reshape211(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 7, 64))
        return reshape_output_1


class Reshape212(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 7, 7))
        return reshape_output_1


class Reshape213(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 7, 7))
        return reshape_output_1


class Reshape214(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 64, 7))
        return reshape_output_1


class Reshape215(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 7, 64))
        return reshape_output_1


class Reshape216(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 7, 3072))
        return reshape_output_1


class Reshape217(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(7, 3072))
        return reshape_output_1


class Reshape218(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32))
        return reshape_output_1


class Reshape219(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 1024))
        return reshape_output_1


class Reshape220(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 16, 64))
        return reshape_output_1


class Reshape221(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 1024))
        return reshape_output_1


class Reshape222(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 32, 64))
        return reshape_output_1


class Reshape223(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 32, 32))
        return reshape_output_1


class Reshape224(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 32, 32))
        return reshape_output_1


class Reshape225(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 32, 64))
        return reshape_output_1


class Reshape226(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 512))
        return reshape_output_1


class Reshape227(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 1))
        return reshape_output_1


class Reshape228(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 16, 64))
        return reshape_output_1


class Reshape229(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 512, 1, 322))
        return reshape_output_1


class Reshape230(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 3025, 64))
        return reshape_output_1


class Reshape231(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(3025, 322))
        return reshape_output_1


class Reshape232(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 3025, 1, 322))
        return reshape_output_1


class Reshape233(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 3025, 322))
        return reshape_output_1


class Reshape234(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 512, 3025))
        return reshape_output_1


class Reshape235(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 512, 3025))
        return reshape_output_1


class Reshape236(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 322, 3025))
        return reshape_output_1


class Reshape237(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(512, 1024))
        return reshape_output_1


class Reshape238(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 512, 8, 128))
        return reshape_output_1


class Reshape239(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 512, 1, 1024))
        return reshape_output_1


class Reshape240(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 512, 1024))
        return reshape_output_1


class Reshape241(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 512, 128))
        return reshape_output_1


class Reshape242(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 512, 512))
        return reshape_output_1


class Reshape243(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 512, 512))
        return reshape_output_1


class Reshape244(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 128, 512))
        return reshape_output_1


class Reshape245(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 512, 128))
        return reshape_output_1


class Reshape246(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 512))
        return reshape_output_1


class Reshape247(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 8, 64))
        return reshape_output_1


class Reshape248(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 512))
        return reshape_output_1


class Reshape249(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1024, 512))
        return reshape_output_1


class Reshape250(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 2048))
        return reshape_output_1


class Reshape251(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 32, 64))
        return reshape_output_1


class Reshape252(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 2048))
        return reshape_output_1


class Reshape253(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 12, 64))
        return reshape_output_1


class Reshape254(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 768))
        return reshape_output_1


class Reshape255(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 12, 12))
        return reshape_output_1


class Reshape256(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 12, 12))
        return reshape_output_1


class Reshape257(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 64, 12))
        return reshape_output_1


class Reshape258(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 12, 64))
        return reshape_output_1


class Reshape259(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 32, 64))
        return reshape_output_1


class Reshape260(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 8192))
        return reshape_output_1


class Reshape261(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(35, 1536))
        return reshape_output_1


class Reshape262(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 35, 12, 128))
        return reshape_output_1


class Reshape263(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 35, 1536))
        return reshape_output_1


class Reshape264(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 35, 128))
        return reshape_output_1


class Reshape265(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 35, 256))
        return reshape_output_1


class Reshape266(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 35, 2, 128))
        return reshape_output_1


class Reshape267(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 35, 128))
        return reshape_output_1


class Reshape268(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 35, 35))
        return reshape_output_1


class Reshape269(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 35, 35))
        return reshape_output_1


class Reshape270(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 128, 35))
        return reshape_output_1


class Reshape271(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 35, 8960))
        return reshape_output_1


class Reshape272(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2520, 1, 1))
        return reshape_output_1


class Reshape273(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 440, 1, 1))
        return reshape_output_1


class Reshape274(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 8, 8, 8, 96))
        return reshape_output_1


class Reshape275(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(4096, 96))
        return reshape_output_1


class Reshape276(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4096, 96))
        return reshape_output_1


class Reshape277(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 64, 64, 96))
        return reshape_output_1


class Reshape278(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 64, 288))
        return reshape_output_1


class Reshape279(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 64, 3, 3, 32))
        return reshape_output_1


class Reshape280(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 3, 64, 32))
        return reshape_output_1


class Reshape281(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(192, 64, 32))
        return reshape_output_1


class Reshape282(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 3, 64, 64))
        return reshape_output_1


class Reshape283(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(225, 2))
        return reshape_output_1


class Reshape284(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 15, 15, 512))
        return reshape_output_1


class Reshape285(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(225, 512))
        return reshape_output_1


class Reshape286(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(225, 3))
        return reshape_output_1


class Reshape287(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 64, 3))
        return reshape_output_1


class Reshape288(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(192, 64, 64))
        return reshape_output_1


class Reshape289(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 64, 3, 64, 64))
        return reshape_output_1


class Reshape290(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(192, 32, 64))
        return reshape_output_1


class Reshape291(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 64, 96))
        return reshape_output_1


class Reshape292(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 64, 3, 32))
        return reshape_output_1


class Reshape293(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 64, 64, 384))
        return reshape_output_1


class Reshape294(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 64, 384))
        return reshape_output_1


class Reshape295(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(4096, 384))
        return reshape_output_1


class Reshape296(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1024, 384))
        return reshape_output_1


class Reshape297(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 32, 192))
        return reshape_output_1


class Reshape298(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 64, 192))
        return reshape_output_1


class Reshape299(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1024, 192))
        return reshape_output_1


class Reshape300(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 64, 6, 32))
        return reshape_output_1


class Reshape301(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 8, 4, 8, 192))
        return reshape_output_1


class Reshape302(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1024, 192))
        return reshape_output_1


class Reshape303(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 64, 576))
        return reshape_output_1


class Reshape304(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 64, 3, 6, 32))
        return reshape_output_1


class Reshape305(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 6, 64, 32))
        return reshape_output_1


class Reshape306(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(96, 64, 32))
        return reshape_output_1


class Reshape307(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 6, 64, 64))
        return reshape_output_1


class Reshape308(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(225, 6))
        return reshape_output_1


class Reshape309(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 64, 6))
        return reshape_output_1


class Reshape310(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(96, 64, 64))
        return reshape_output_1


class Reshape311(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 6, 64, 64))
        return reshape_output_1


class Reshape312(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(96, 32, 64))
        return reshape_output_1


class Reshape313(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 4, 8, 8, 192))
        return reshape_output_1


class Reshape314(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 32, 768))
        return reshape_output_1


class Reshape315(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 64, 768))
        return reshape_output_1


class Reshape316(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1024, 768))
        return reshape_output_1


class Reshape317(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(256, 768))
        return reshape_output_1


class Reshape318(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 16, 384))
        return reshape_output_1


class Reshape319(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(4, 64, 384))
        return reshape_output_1


class Reshape320(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 384))
        return reshape_output_1


class Reshape321(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(4, 64, 12, 32))
        return reshape_output_1


class Reshape322(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2, 8, 2, 8, 384))
        return reshape_output_1


class Reshape323(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(256, 384))
        return reshape_output_1


class Reshape324(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(4, 64, 1152))
        return reshape_output_1


class Reshape325(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(4, 64, 3, 12, 32))
        return reshape_output_1


class Reshape326(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(4, 12, 64, 32))
        return reshape_output_1


class Reshape327(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(48, 64, 32))
        return reshape_output_1


class Reshape328(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(4, 12, 64, 64))
        return reshape_output_1


class Reshape329(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(225, 12))
        return reshape_output_1


class Reshape330(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 64, 12))
        return reshape_output_1


class Reshape331(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(48, 64, 64))
        return reshape_output_1


class Reshape332(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 12, 64, 64))
        return reshape_output_1


class Reshape333(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(48, 32, 64))
        return reshape_output_1


class Reshape334(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2, 2, 8, 8, 384))
        return reshape_output_1


class Reshape335(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 16, 1536))
        return reshape_output_1


class Reshape336(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(4, 64, 1536))
        return reshape_output_1


class Reshape337(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(256, 1536))
        return reshape_output_1


class Reshape338(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 1536))
        return reshape_output_1


class Reshape339(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 8, 768))
        return reshape_output_1


class Reshape340(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 64, 768))
        return reshape_output_1


class Reshape341(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 64, 24, 32))
        return reshape_output_1


class Reshape342(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 8, 1, 8, 768))
        return reshape_output_1


class Reshape343(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 768))
        return reshape_output_1


class Reshape344(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 64, 2304))
        return reshape_output_1


class Reshape345(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 64, 3, 24, 32))
        return reshape_output_1


class Reshape346(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 24, 64, 32))
        return reshape_output_1


class Reshape347(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(24, 64, 32))
        return reshape_output_1


class Reshape348(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 24, 64, 64))
        return reshape_output_1


class Reshape349(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(225, 24))
        return reshape_output_1


class Reshape350(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 64, 24))
        return reshape_output_1


class Reshape351(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(24, 64, 64))
        return reshape_output_1


class Reshape352(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(24, 32, 64))
        return reshape_output_1


class Reshape353(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 1, 8, 8, 768))
        return reshape_output_1


class Reshape354(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 8, 3072))
        return reshape_output_1


class Reshape355(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 64, 3072))
        return reshape_output_1


class Reshape356(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 3072))
        return reshape_output_1


class Reshape357(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 768, 1, 1))
        return reshape_output_1


class Reshape358(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1))
        return reshape_output_1


class Reshape359(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 384))
        return reshape_output_1


class Reshape360(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(6, 1, 64))
        return reshape_output_1


class Reshape361(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 6, 1, 1))
        return reshape_output_1


class Reshape362(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(6, 1, 1))
        return reshape_output_1


class Reshape363(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(6, 64, 1))
        return reshape_output_1


class Reshape364(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 6, 1, 64))
        return reshape_output_1


class Reshape365(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 61))
        return reshape_output_1


class Reshape366(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(61, 512))
        return reshape_output_1


class Reshape367(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 61, 6, 64))
        return reshape_output_1


class Reshape368(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(6, 61, 64))
        return reshape_output_1


class Reshape369(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 6, 61, 61))
        return reshape_output_1


class Reshape370(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(6, 61, 61))
        return reshape_output_1


class Reshape371(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(6, 64, 61))
        return reshape_output_1


class Reshape372(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 6, 61, 64))
        return reshape_output_1


class Reshape373(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(61, 384))
        return reshape_output_1


class Reshape374(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 61, 512))
        return reshape_output_1


class Reshape375(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 61, 8, 64))
        return reshape_output_1


class Reshape376(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 61, 1024))
        return reshape_output_1


class Reshape377(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 61, 16, 64))
        return reshape_output_1


class Reshape378(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 6, 1, 61))
        return reshape_output_1


class Reshape379(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(6, 1, 61))
        return reshape_output_1


class Reshape380(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 1024))
        return reshape_output_1


class Reshape381(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 25088, 1, 1))
        return reshape_output_1


class Reshape382(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 25088))
        return reshape_output_1


class Reshape383(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 512, 49, 1))
        return reshape_output_1


class Reshape384(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1024, 196, 1))
        return reshape_output_1


class Reshape385(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1024, 196))
        return reshape_output_1


class Reshape386(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(197, 1024))
        return reshape_output_1


class Reshape387(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 197, 16, 64))
        return reshape_output_1


class Reshape388(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 197, 1024))
        return reshape_output_1


class Reshape389(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(197, 1, 1024))
        return reshape_output_1


class Reshape390(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 197, 64))
        return reshape_output_1


class Reshape391(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 197, 197))
        return reshape_output_1


class Reshape392(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 197, 197))
        return reshape_output_1


class Reshape393(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 64, 197))
        return reshape_output_1


class Reshape394(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 197, 64))
        return reshape_output_1


class Reshape395(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 1, 64))
        return reshape_output_1


class Reshape396(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 1, 1))
        return reshape_output_1


class Reshape397(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 1, 1))
        return reshape_output_1


class Reshape398(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 4))
        return reshape_output_1


class Reshape399(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 64, 1))
        return reshape_output_1


class Reshape400(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 1, 64))
        return reshape_output_1


class Reshape401(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 80, 3000, 1))
        return reshape_output_1


class Reshape402(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1024, 80, 3, 1))
        return reshape_output_1


class Reshape403(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1024, 3000))
        return reshape_output_1


class Reshape404(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1024, 3000, 1))
        return reshape_output_1


class Reshape405(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1024, 1024, 3, 1))
        return reshape_output_1


class Reshape406(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1024, 1500))
        return reshape_output_1


class Reshape407(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1500, 1024))
        return reshape_output_1


class Reshape408(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1500, 16, 64))
        return reshape_output_1


class Reshape409(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1500, 1024))
        return reshape_output_1


class Reshape410(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 1500, 64))
        return reshape_output_1


class Reshape411(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 1500, 1500))
        return reshape_output_1


class Reshape412(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 1500, 1500))
        return reshape_output_1


class Reshape413(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 64, 1500))
        return reshape_output_1


class Reshape414(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 1500, 64))
        return reshape_output_1


class Reshape415(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 1, 1500))
        return reshape_output_1


class Reshape416(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 1, 1500))
        return reshape_output_1


class Reshape417(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 1, 3, 3))
        return reshape_output_1


class Reshape418(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(128, 1, 3, 3))
        return reshape_output_1


class Reshape419(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(256, 1, 3, 3))
        return reshape_output_1


class Reshape420(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(728, 1, 3, 3))
        return reshape_output_1


class Reshape421(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1024, 1, 3, 3))
        return reshape_output_1


class Reshape422(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1536, 1, 3, 3))
        return reshape_output_1


class Reshape423(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 255, 40, 40))
        return reshape_output_1


class Reshape424(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 255, 1600))
        return reshape_output_1


class Reshape425(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 3, 85, 1600))
        return reshape_output_1


class Reshape426(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4800, 85))
        return reshape_output_1


class Reshape427(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 255, 20, 20))
        return reshape_output_1


class Reshape428(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 255, 400))
        return reshape_output_1


class Reshape429(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 3, 85, 400))
        return reshape_output_1


class Reshape430(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1200, 85))
        return reshape_output_1


class Reshape431(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 255, 10, 10))
        return reshape_output_1


class Reshape432(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 255, 100))
        return reshape_output_1


class Reshape433(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 3, 85, 100))
        return reshape_output_1


class Reshape434(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 300, 85))
        return reshape_output_1


class Reshape435(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 255, 80, 80))
        return reshape_output_1


class Reshape436(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 255, 6400))
        return reshape_output_1


class Reshape437(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 3, 85, 6400))
        return reshape_output_1


class Reshape438(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 19200, 85))
        return reshape_output_1


class Reshape439(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 4480))
        return reshape_output_1


class Reshape440(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 1120))
        return reshape_output_1


class Reshape441(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 280))
        return reshape_output_1


class Reshape442(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 80, 4480))
        return reshape_output_1


class Reshape443(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 80, 1120))
        return reshape_output_1


class Reshape444(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 80, 280))
        return reshape_output_1


class Reshape445(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 85, 2704, 1))
        return reshape_output_1


class Reshape446(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 85, 676, 1))
        return reshape_output_1


class Reshape447(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 85, 169, 1))
        return reshape_output_1


class Reshape448(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 100, 100))
        return reshape_output_1


class Reshape449(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 100, 92))
        return reshape_output_1


class Reshape450(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 768, 16, 16))
        return reshape_output_1


class Reshape451(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 768, 32, 32))
        return reshape_output_1


class Reshape452(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 768, 64, 64))
        return reshape_output_1


class Reshape453(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 768, 128, 128))
        return reshape_output_1


class Reshape454(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(11, 768))
        return reshape_output_1


class Reshape455(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 11, 12, 64))
        return reshape_output_1


class Reshape456(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 11, 768))
        return reshape_output_1


class Reshape457(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 11, 64))
        return reshape_output_1


class Reshape458(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 64, 11))
        return reshape_output_1


class Reshape459(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 11, 11))
        return reshape_output_1


class Reshape460(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 11, 11))
        return reshape_output_1


class Reshape461(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 11, 64))
        return reshape_output_1


class Reshape462(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 1024))
        return reshape_output_1


class Reshape463(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 4, 256))
        return reshape_output_1


class Reshape464(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 256, 64))
        return reshape_output_1


class Reshape465(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 256, 256))
        return reshape_output_1


class Reshape466(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 256, 256))
        return reshape_output_1


class Reshape467(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 256, 64))
        return reshape_output_1


class Reshape468(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256))
        return reshape_output_1


class Reshape469(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(128, 1024))
        return reshape_output_1


class Reshape470(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 16, 64))
        return reshape_output_1


class Reshape471(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 1024))
        return reshape_output_1


class Reshape472(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 128, 64))
        return reshape_output_1


class Reshape473(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 128, 128))
        return reshape_output_1


class Reshape474(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 128, 128))
        return reshape_output_1


class Reshape475(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 64, 128))
        return reshape_output_1


class Reshape476(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 128, 64))
        return reshape_output_1


class Reshape477(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 1024, 1))
        return reshape_output_1


class Reshape478(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(144, 1, 5, 5))
        return reshape_output_1


class Reshape479(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(240, 1, 5, 5))
        return reshape_output_1


class Reshape480(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(240, 1, 3, 3))
        return reshape_output_1


class Reshape481(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(480, 1, 3, 3))
        return reshape_output_1


class Reshape482(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(480, 1, 5, 5))
        return reshape_output_1


class Reshape483(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(672, 1, 5, 5))
        return reshape_output_1


class Reshape484(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1152, 1, 5, 5))
        return reshape_output_1


class Reshape485(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1152, 1, 3, 3))
        return reshape_output_1


class Reshape486(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(336, 1, 5, 5))
        return reshape_output_1


class Reshape487(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(336, 1, 3, 3))
        return reshape_output_1


class Reshape488(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(672, 1, 3, 3))
        return reshape_output_1


class Reshape489(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(960, 1, 5, 5))
        return reshape_output_1


class Reshape490(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1632, 1, 5, 5))
        return reshape_output_1


class Reshape491(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1632, 1, 3, 3))
        return reshape_output_1


class Reshape492(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 1, 3, 3))
        return reshape_output_1


class Reshape493(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(24, 1, 3, 3))
        return reshape_output_1


class Reshape494(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(48, 1, 3, 3))
        return reshape_output_1


class Reshape495(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 1, 3, 3))
        return reshape_output_1


class Reshape496(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 1, 3, 3))
        return reshape_output_1


class Reshape497(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(36, 1, 3, 3))
        return reshape_output_1


class Reshape498(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(72, 1, 5, 5))
        return reshape_output_1


class Reshape499(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(20, 1, 3, 3))
        return reshape_output_1


class Reshape500(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(24, 1, 5, 5))
        return reshape_output_1


class Reshape501(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(60, 1, 3, 3))
        return reshape_output_1


class Reshape502(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(120, 1, 3, 3))
        return reshape_output_1


class Reshape503(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(40, 1, 3, 3))
        return reshape_output_1


class Reshape504(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(100, 1, 3, 3))
        return reshape_output_1


class Reshape505(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(92, 1, 3, 3))
        return reshape_output_1


class Reshape506(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(56, 1, 3, 3))
        return reshape_output_1


class Reshape507(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(80, 1, 3, 3))
        return reshape_output_1


class Reshape508(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(112, 1, 5, 5))
        return reshape_output_1


class Reshape509(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(7, 2))
        return reshape_output_1


class Reshape510(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2))
        return reshape_output_1


class Reshape511(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 3, 16, 16, 16, 16))
        return reshape_output_1


class Reshape512(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1024, 256, 1, 1))
        return reshape_output_1


class Reshape513(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1024, 512, 1))
        return reshape_output_1


class Reshape514(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(256, 1024, 1, 1))
        return reshape_output_1


class Reshape515(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 512, 196, 1))
        return reshape_output_1


class Reshape516(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(72, 1, 3, 3))
        return reshape_output_1


class Reshape517(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(120, 1, 5, 5))
        return reshape_output_1


class Reshape518(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(200, 1, 3, 3))
        return reshape_output_1


class Reshape519(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(184, 1, 3, 3))
        return reshape_output_1


class Reshape520(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 960, 1, 1))
        return reshape_output_1


class Reshape521(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 2048))
        return reshape_output_1


class Reshape522(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 32, 64))
        return reshape_output_1


class Reshape523(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 2048))
        return reshape_output_1


class Reshape524(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 32, 64))
        return reshape_output_1


class Reshape525(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 32, 32))
        return reshape_output_1


class Reshape526(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 32, 32))
        return reshape_output_1


class Reshape527(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 2))
        return reshape_output_1


class Reshape528(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 512, 1, 261))
        return reshape_output_1


class Reshape529(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 50176, 3))
        return reshape_output_1


class Reshape530(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(50176, 261))
        return reshape_output_1


class Reshape531(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 50176, 1, 261))
        return reshape_output_1


class Reshape532(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 50176, 261))
        return reshape_output_1


class Reshape533(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 512, 50176))
        return reshape_output_1


class Reshape534(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 512, 50176))
        return reshape_output_1


class Reshape535(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 261, 50176))
        return reshape_output_1


class Reshape536(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1008, 1, 1))
        return reshape_output_1


class Reshape537(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 784, 1, 1))
        return reshape_output_1


class Reshape538(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 64, 128))
        return reshape_output_1


class Reshape539(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(768, 768, 1, 1))
        return reshape_output_1


class Reshape540(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 768, 128))
        return reshape_output_1


class Reshape541(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 768))
        return reshape_output_1


class Reshape542(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 1, 64))
        return reshape_output_1


class Reshape543(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 1, 1))
        return reshape_output_1


class Reshape544(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 1, 1))
        return reshape_output_1


class Reshape545(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 64, 1))
        return reshape_output_1


class Reshape546(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 1, 64))
        return reshape_output_1


class Reshape547(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(61, 768))
        return reshape_output_1


class Reshape548(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 61, 12, 64))
        return reshape_output_1


class Reshape549(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 61, 768))
        return reshape_output_1


class Reshape550(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 61, 64))
        return reshape_output_1


class Reshape551(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 61, 61))
        return reshape_output_1


class Reshape552(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 61, 61))
        return reshape_output_1


class Reshape553(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 64, 61))
        return reshape_output_1


class Reshape554(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 61, 64))
        return reshape_output_1


class Reshape555(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 1, 61))
        return reshape_output_1


class Reshape556(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 1, 61))
        return reshape_output_1


class Reshape557(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4096, 1, 1))
        return reshape_output_1


class Reshape558(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(197, 1, 2304))
        return reshape_output_1


class Reshape559(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(197, 1, 3, 768))
        return reshape_output_1


class Reshape560(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(197, 12, 64))
        return reshape_output_1


class Reshape561(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(160, 1, 3, 3))
        return reshape_output_1


class Reshape562(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(224, 1, 3, 3))
        return reshape_output_1


class Reshape563(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(768, 80, 3, 1))
        return reshape_output_1


class Reshape564(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 768, 3000))
        return reshape_output_1


class Reshape565(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 768, 3000, 1))
        return reshape_output_1


class Reshape566(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(768, 768, 3, 1))
        return reshape_output_1


class Reshape567(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 768, 1500))
        return reshape_output_1


class Reshape568(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1500, 768))
        return reshape_output_1


class Reshape569(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1500, 12, 64))
        return reshape_output_1


class Reshape570(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1500, 768))
        return reshape_output_1


class Reshape571(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 1500, 64))
        return reshape_output_1


class Reshape572(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 1500, 1500))
        return reshape_output_1


class Reshape573(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 1500, 1500))
        return reshape_output_1


class Reshape574(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 64, 1500))
        return reshape_output_1


class Reshape575(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 1500, 64))
        return reshape_output_1


class Reshape576(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 1, 1500))
        return reshape_output_1


class Reshape577(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 1, 1500))
        return reshape_output_1


class Reshape578(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 85, 6400, 1))
        return reshape_output_1


class Reshape579(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 85, 1600, 1))
        return reshape_output_1


class Reshape580(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 85, 400, 1))
        return reshape_output_1


class Reshape581(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 100, 251))
        return reshape_output_1


class Reshape582(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(100, 32, 107, 160))
        return reshape_output_1


class Reshape583(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(100, 1, 32, 107, 160))
        return reshape_output_1


class Reshape584(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(100, 64, 54, 80))
        return reshape_output_1


class Reshape585(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(100, 1, 64, 54, 80))
        return reshape_output_1


class Reshape586(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(100, 128, 27, 40))
        return reshape_output_1


class Reshape587(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(100, 1, 128, 27, 40))
        return reshape_output_1


class Reshape588(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(100, 256, 14, 20))
        return reshape_output_1


class Reshape589(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(100, 1, 256, 14, 20))
        return reshape_output_1


class Reshape590(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 14, 20))
        return reshape_output_1


class Reshape591(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 100, 8, 14, 20))
        return reshape_output_1


class Reshape592(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 100, 2240))
        return reshape_output_1


class Reshape593(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 100, 2240, 1, 1))
        return reshape_output_1


class Reshape594(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(100, 1, 8, 14, 20))
        return reshape_output_1


class Reshape595(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(100, 8, 14, 20))
        return reshape_output_1


class Reshape596(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(100, 8, 9240))
        return reshape_output_1


class Reshape597(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(100, 8, 33, 280))
        return reshape_output_1


class Reshape598(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(100, 264, 14, 20))
        return reshape_output_1


class Reshape599(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(100, 8, 4480))
        return reshape_output_1


class Reshape600(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(100, 8, 16, 280))
        return reshape_output_1


class Reshape601(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(100, 128, 14, 20))
        return reshape_output_1


class Reshape602(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(100, 8, 8640))
        return reshape_output_1


class Reshape603(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(100, 8, 8, 1080))
        return reshape_output_1


class Reshape604(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(100, 64, 27, 40))
        return reshape_output_1


class Reshape605(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(100, 8, 17280))
        return reshape_output_1


class Reshape606(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(100, 8, 4, 4320))
        return reshape_output_1


class Reshape607(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(100, 32, 54, 80))
        return reshape_output_1


class Reshape608(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(100, 8, 34240))
        return reshape_output_1


class Reshape609(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(100, 8, 2, 17120))
        return reshape_output_1


class Reshape610(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(100, 16, 107, 160))
        return reshape_output_1


class Reshape611(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 100, 107, 160))
        return reshape_output_1


class Reshape612(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 27, 27, 16))
        return reshape_output_1


class Reshape613(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(729, 16))
        return reshape_output_1


class Reshape614(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(197, 197, 16))
        return reshape_output_1


class Reshape615(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(384, 1024))
        return reshape_output_1


class Reshape616(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 384, 16, 64))
        return reshape_output_1


class Reshape617(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 384, 1024))
        return reshape_output_1


class Reshape618(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 384, 64))
        return reshape_output_1


class Reshape619(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 384, 384))
        return reshape_output_1


class Reshape620(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 384, 384))
        return reshape_output_1


class Reshape621(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 64, 384))
        return reshape_output_1


class Reshape622(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 384, 64))
        return reshape_output_1


class Reshape623(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 384, 1))
        return reshape_output_1


class Reshape624(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 192, 196, 1))
        return reshape_output_1


class Reshape625(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(197, 192))
        return reshape_output_1


class Reshape626(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 197, 3, 64))
        return reshape_output_1


class Reshape627(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 197, 192))
        return reshape_output_1


class Reshape628(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(3, 197, 64))
        return reshape_output_1


class Reshape629(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 3, 197, 197))
        return reshape_output_1


class Reshape630(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(3, 197, 197))
        return reshape_output_1


class Reshape631(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(3, 64, 197))
        return reshape_output_1


class Reshape632(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 3, 197, 64))
        return reshape_output_1


class Reshape633(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 192))
        return reshape_output_1


class Reshape634(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(72, 1, 1, 5))
        return reshape_output_1


class Reshape635(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(72, 1, 5, 1))
        return reshape_output_1


class Reshape636(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(120, 1, 1, 5))
        return reshape_output_1


class Reshape637(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(120, 1, 5, 1))
        return reshape_output_1


class Reshape638(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(240, 1, 1, 5))
        return reshape_output_1


class Reshape639(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(240, 1, 5, 1))
        return reshape_output_1


class Reshape640(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(200, 1, 1, 5))
        return reshape_output_1


class Reshape641(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(200, 1, 5, 1))
        return reshape_output_1


class Reshape642(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(184, 1, 1, 5))
        return reshape_output_1


class Reshape643(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(184, 1, 5, 1))
        return reshape_output_1


class Reshape644(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(480, 1, 1, 5))
        return reshape_output_1


class Reshape645(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(480, 1, 5, 1))
        return reshape_output_1


class Reshape646(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(672, 1, 1, 5))
        return reshape_output_1


class Reshape647(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(672, 1, 5, 1))
        return reshape_output_1


class Reshape648(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(960, 1, 1, 5))
        return reshape_output_1


class Reshape649(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(960, 1, 5, 1))
        return reshape_output_1


class Reshape650(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(256, 2048))
        return reshape_output_1


class Reshape651(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 16, 128))
        return reshape_output_1


class Reshape652(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 32, 64))
        return reshape_output_1


class Reshape653(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 2048))
        return reshape_output_1


class Reshape654(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 16, 2048))
        return reshape_output_1


class Reshape655(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 256, 128))
        return reshape_output_1


class Reshape656(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 128, 256))
        return reshape_output_1


class Reshape657(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 256, 128))
        return reshape_output_1


class Reshape658(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 2048, 32))
        return reshape_output_1


class Reshape659(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 256, 32))
        return reshape_output_1


class Reshape660(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2048, 768))
        return reshape_output_1


class Reshape661(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 256, 2048))
        return reshape_output_1


class Reshape662(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 256, 2048))
        return reshape_output_1


class Reshape663(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2048, 1280))
        return reshape_output_1


class Reshape664(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2048, 8, 160))
        return reshape_output_1


class Reshape665(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 160, 2048))
        return reshape_output_1


class Reshape666(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 256, 160))
        return reshape_output_1


class Reshape667(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(256, 1280))
        return reshape_output_1


class Reshape668(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 1280))
        return reshape_output_1


class Reshape669(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 8, 160))
        return reshape_output_1


class Reshape670(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 160, 256))
        return reshape_output_1


class Reshape671(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 2048, 256))
        return reshape_output_1


class Reshape672(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 2048, 256))
        return reshape_output_1


class Reshape673(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 768))
        return reshape_output_1


class Reshape674(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 12, 64))
        return reshape_output_1


class Reshape675(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 8, 96))
        return reshape_output_1


class Reshape676(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 96, 256))
        return reshape_output_1


class Reshape677(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 2048, 96))
        return reshape_output_1


class Reshape678(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2048, 768))
        return reshape_output_1


class Reshape679(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2048, 262))
        return reshape_output_1


class Reshape680(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(6, 1024))
        return reshape_output_1


class Reshape681(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 6, 16, 64))
        return reshape_output_1


class Reshape682(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 6, 1024))
        return reshape_output_1


class Reshape683(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 6, 64))
        return reshape_output_1


class Reshape684(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 6, 6))
        return reshape_output_1


class Reshape685(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 6, 6))
        return reshape_output_1


class Reshape686(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 64, 6))
        return reshape_output_1


class Reshape687(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 6, 64))
        return reshape_output_1


class Reshape688(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 6, 2816))
        return reshape_output_1


class Reshape689(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1296, 1, 1))
        return reshape_output_1


class Reshape690(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 672, 1, 1))
        return reshape_output_1


class Reshape691(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 16384, 1))
        return reshape_output_1


class Reshape692(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 16384))
        return reshape_output_1


class Reshape693(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16384, 1, 32))
        return reshape_output_1


class Reshape694(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 128, 32))
        return reshape_output_1


class Reshape695(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 128, 128))
        return reshape_output_1


class Reshape696(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 256))
        return reshape_output_1


class Reshape697(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(256, 32))
        return reshape_output_1


class Reshape698(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 1, 32))
        return reshape_output_1


class Reshape699(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 32))
        return reshape_output_1


class Reshape700(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 128, 128))
        return reshape_output_1


class Reshape701(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 16384, 1))
        return reshape_output_1


class Reshape702(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 16384))
        return reshape_output_1


class Reshape703(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 64, 4096, 1))
        return reshape_output_1


class Reshape704(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 64, 4096))
        return reshape_output_1


class Reshape705(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4096, 2, 32))
        return reshape_output_1


class Reshape706(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 64, 64, 64))
        return reshape_output_1


class Reshape707(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 4096, 32))
        return reshape_output_1


class Reshape708(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 256, 32))
        return reshape_output_1


class Reshape709(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 32, 256))
        return reshape_output_1


class Reshape710(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2, 4096, 32))
        return reshape_output_1


class Reshape711(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(4096, 64))
        return reshape_output_1


class Reshape712(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4096, 64))
        return reshape_output_1


class Reshape713(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 4096, 1))
        return reshape_output_1


class Reshape714(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 4096))
        return reshape_output_1


class Reshape715(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 160, 1024, 1))
        return reshape_output_1


class Reshape716(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 160, 1024))
        return reshape_output_1


class Reshape717(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1024, 5, 32))
        return reshape_output_1


class Reshape718(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 32, 160))
        return reshape_output_1


class Reshape719(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(5, 1024, 32))
        return reshape_output_1


class Reshape720(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 160, 32, 32))
        return reshape_output_1


class Reshape721(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 160, 256))
        return reshape_output_1


class Reshape722(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(256, 160))
        return reshape_output_1


class Reshape723(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 5, 32))
        return reshape_output_1


class Reshape724(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 160))
        return reshape_output_1


class Reshape725(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(5, 256, 32))
        return reshape_output_1


class Reshape726(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(5, 32, 256))
        return reshape_output_1


class Reshape727(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 5, 1024, 32))
        return reshape_output_1


class Reshape728(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1024, 160))
        return reshape_output_1


class Reshape729(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1024, 160))
        return reshape_output_1


class Reshape730(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 640, 32, 32))
        return reshape_output_1


class Reshape731(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(640, 1, 3, 3))
        return reshape_output_1


class Reshape732(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 640, 1024, 1))
        return reshape_output_1


class Reshape733(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 640, 1024))
        return reshape_output_1


class Reshape734(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 256, 1))
        return reshape_output_1


class Reshape735(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 32, 256))
        return reshape_output_1


class Reshape736(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 256, 32))
        return reshape_output_1


class Reshape737(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1024, 16, 16))
        return reshape_output_1


class Reshape738(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1024, 256, 1))
        return reshape_output_1


class Reshape739(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1024, 256))
        return reshape_output_1


class Reshape740(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 96, 3136, 1))
        return reshape_output_1


class Reshape741(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 7, 8, 7, 96))
        return reshape_output_1


class Reshape742(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 56, 56, 96))
        return reshape_output_1


class Reshape743(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(3136, 96))
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
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 49, 3, 32))
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
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(192, 49, 32))
        return reshape_output_1


class Reshape748(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 3, 49, 49))
        return reshape_output_1


class Reshape749(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2401,))
        return reshape_output_1


class Reshape750(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(49, 49, 3))
        return reshape_output_1


class Reshape751(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(192, 49, 49))
        return reshape_output_1


class Reshape752(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 64, 3, 49, 49))
        return reshape_output_1


class Reshape753(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(192, 32, 49))
        return reshape_output_1


class Reshape754(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 3, 49, 32))
        return reshape_output_1


class Reshape755(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 3136, 96))
        return reshape_output_1


class Reshape756(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 784, 384))
        return reshape_output_1


class Reshape757(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(784, 384))
        return reshape_output_1


class Reshape758(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 7, 4, 7, 192))
        return reshape_output_1


class Reshape759(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 28, 28, 192))
        return reshape_output_1


class Reshape760(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(784, 192))
        return reshape_output_1


class Reshape761(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 49, 192))
        return reshape_output_1


class Reshape762(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 49, 6, 32))
        return reshape_output_1


class Reshape763(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 4, 7, 7, 192))
        return reshape_output_1


class Reshape764(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(96, 49, 32))
        return reshape_output_1


class Reshape765(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 6, 49, 49))
        return reshape_output_1


class Reshape766(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(49, 49, 6))
        return reshape_output_1


class Reshape767(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(96, 49, 49))
        return reshape_output_1


class Reshape768(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 6, 49, 49))
        return reshape_output_1


class Reshape769(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(96, 32, 49))
        return reshape_output_1


class Reshape770(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 6, 49, 32))
        return reshape_output_1


class Reshape771(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 784, 192))
        return reshape_output_1


class Reshape772(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 196, 768))
        return reshape_output_1


class Reshape773(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(196, 768))
        return reshape_output_1


class Reshape774(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2, 7, 2, 7, 384))
        return reshape_output_1


class Reshape775(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 14, 14, 384))
        return reshape_output_1


class Reshape776(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(196, 384))
        return reshape_output_1


class Reshape777(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(4, 49, 384))
        return reshape_output_1


class Reshape778(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(4, 49, 12, 32))
        return reshape_output_1


class Reshape779(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2, 2, 7, 7, 384))
        return reshape_output_1


class Reshape780(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(48, 49, 32))
        return reshape_output_1


class Reshape781(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(4, 12, 49, 49))
        return reshape_output_1


class Reshape782(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(49, 49, 12))
        return reshape_output_1


class Reshape783(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(48, 49, 49))
        return reshape_output_1


class Reshape784(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 12, 49, 49))
        return reshape_output_1


class Reshape785(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(48, 32, 49))
        return reshape_output_1


class Reshape786(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(4, 12, 49, 32))
        return reshape_output_1


class Reshape787(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 196, 384))
        return reshape_output_1


class Reshape788(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 49, 1536))
        return reshape_output_1


class Reshape789(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(49, 1536))
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
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 49, 24, 32))
        return reshape_output_1


class Reshape792(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 49, 768))
        return reshape_output_1


class Reshape793(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 1, 7, 7, 768))
        return reshape_output_1


class Reshape794(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 7, 7, 768))
        return reshape_output_1


class Reshape795(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(24, 49, 32))
        return reshape_output_1


class Reshape796(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 24, 49, 49))
        return reshape_output_1


class Reshape797(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(49, 49, 24))
        return reshape_output_1


class Reshape798(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(24, 49, 49))
        return reshape_output_1


class Reshape799(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(24, 32, 49))
        return reshape_output_1


class Reshape800(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 24, 49, 32))
        return reshape_output_1


class Reshape801(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 768, 1))
        return reshape_output_1


class Reshape802(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 1, 64))
        return reshape_output_1


class Reshape803(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 1, 1))
        return reshape_output_1


class Reshape804(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 1, 1))
        return reshape_output_1


class Reshape805(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 64, 1))
        return reshape_output_1


class Reshape806(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 1, 64))
        return reshape_output_1


class Reshape807(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 61, 64))
        return reshape_output_1


class Reshape808(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 61, 61))
        return reshape_output_1


class Reshape809(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 61, 61))
        return reshape_output_1


class Reshape810(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 64, 61))
        return reshape_output_1


class Reshape811(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 61, 64))
        return reshape_output_1


class Reshape812(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 1, 61))
        return reshape_output_1


class Reshape813(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 1, 61))
        return reshape_output_1


class Reshape814(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1280, 1369))
        return reshape_output_1


class Reshape815(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1370, 1280))
        return reshape_output_1


class Reshape816(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1370, 1, 3840))
        return reshape_output_1


class Reshape817(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1370, 1, 3, 1280))
        return reshape_output_1


class Reshape818(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1370, 16, 80))
        return reshape_output_1


class Reshape819(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 1370, 80))
        return reshape_output_1


class Reshape820(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 1370, 80))
        return reshape_output_1


class Reshape821(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 1370, 1370))
        return reshape_output_1


class Reshape822(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 1370, 1370))
        return reshape_output_1


class Reshape823(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1370, 1, 1280))
        return reshape_output_1


class Reshape824(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 1280))
        return reshape_output_1


class Reshape825(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2, 20, 64))
        return reshape_output_1


class Reshape826(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2, 1280))
        return reshape_output_1


class Reshape827(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(20, 2, 64))
        return reshape_output_1


class Reshape828(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 20, 2, 2))
        return reshape_output_1


class Reshape829(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(20, 2, 2))
        return reshape_output_1


class Reshape830(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(20, 64, 2))
        return reshape_output_1


class Reshape831(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 20, 2, 64))
        return reshape_output_1


class Reshape832(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 3000, 1))
        return reshape_output_1


class Reshape833(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1280, 128, 3, 1))
        return reshape_output_1


class Reshape834(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1280, 3000))
        return reshape_output_1


class Reshape835(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1280, 3000, 1))
        return reshape_output_1


class Reshape836(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1280, 1280, 3, 1))
        return reshape_output_1


class Reshape837(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1280, 1500))
        return reshape_output_1


class Reshape838(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1500, 1280))
        return reshape_output_1


class Reshape839(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1500, 20, 64))
        return reshape_output_1


class Reshape840(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1500, 1280))
        return reshape_output_1


class Reshape841(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(20, 1500, 64))
        return reshape_output_1


class Reshape842(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 20, 1500, 1500))
        return reshape_output_1


class Reshape843(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(20, 1500, 1500))
        return reshape_output_1


class Reshape844(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(20, 64, 1500))
        return reshape_output_1


class Reshape845(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 20, 1500, 64))
        return reshape_output_1


class Reshape846(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 20, 2, 1500))
        return reshape_output_1


class Reshape847(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(20, 2, 1500))
        return reshape_output_1


class Reshape848(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 255, 60, 60))
        return reshape_output_1


class Reshape849(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 255, 3600))
        return reshape_output_1


class Reshape850(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 3, 85, 3600))
        return reshape_output_1


class Reshape851(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 10800, 85))
        return reshape_output_1


class Reshape852(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 255, 30, 30))
        return reshape_output_1


class Reshape853(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 255, 900))
        return reshape_output_1


class Reshape854(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 3, 85, 900))
        return reshape_output_1


class Reshape855(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2700, 85))
        return reshape_output_1


class Reshape856(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 255, 15, 15))
        return reshape_output_1


class Reshape857(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 255, 225))
        return reshape_output_1


class Reshape858(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 3, 85, 225))
        return reshape_output_1


class Reshape859(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 675, 85))
        return reshape_output_1


class Reshape860(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 192, 1344, 1))
        return reshape_output_1


class Reshape861(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 192, 50, 83))
        return reshape_output_1


class Reshape862(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1445, 192))
        return reshape_output_1


class Reshape863(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1445, 3, 64))
        return reshape_output_1


class Reshape864(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1445, 192))
        return reshape_output_1


class Reshape865(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(3, 1445, 64))
        return reshape_output_1


class Reshape866(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 3, 1445, 1445))
        return reshape_output_1


class Reshape867(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(3, 1445, 1445))
        return reshape_output_1


class Reshape868(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(3, 64, 1445))
        return reshape_output_1


class Reshape869(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 3, 1445, 64))
        return reshape_output_1


class Reshape870(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(100, 192))
        return reshape_output_1


class Reshape871(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 100, 192))
        return reshape_output_1


class Reshape872(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 144, 6400))
        return reshape_output_1


class Reshape873(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 144, 1600))
        return reshape_output_1


class Reshape874(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 144, 400))
        return reshape_output_1


class Reshape875(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 16, 8400))
        return reshape_output_1


class Reshape876(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 8400))
        return reshape_output_1


class Reshape877(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1408))
        return reshape_output_1


class Reshape878(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 96, 4096))
        return reshape_output_1


class Reshape879(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2, 128, 400))
        return reshape_output_1


class Reshape880(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 64, 400))
        return reshape_output_1


class Reshape881(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 20, 20))
        return reshape_output_1


class Reshape882(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 400, 32))
        return reshape_output_1


class Reshape883(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 32, 400))
        return reshape_output_1


class Reshape884(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2, 400, 400))
        return reshape_output_1


class Reshape885(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 400, 400))
        return reshape_output_1


class Reshape886(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(128, 2048))
        return reshape_output_1


class Reshape887(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 16, 128))
        return reshape_output_1


class Reshape888(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 2048))
        return reshape_output_1


class Reshape889(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 2048, 1))
        return reshape_output_1


class Reshape890(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2688, 1, 3, 3))
        return reshape_output_1


class Reshape891(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1792, 1, 1))
        return reshape_output_1


class Reshape892(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1792))
        return reshape_output_1


class Reshape893(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 256, 64))
        return reshape_output_1


class Reshape894(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 256, 256))
        return reshape_output_1


class Reshape895(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 256, 256))
        return reshape_output_1


class Reshape896(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 64, 256))
        return reshape_output_1


class Reshape897(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 256, 64))
        return reshape_output_1


class Reshape898(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 256, 64))
        return reshape_output_1


class Reshape899(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 256, 64))
        return reshape_output_1


class Reshape900(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 256, 256))
        return reshape_output_1


class Reshape901(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 256, 256))
        return reshape_output_1


class Reshape902(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 64, 256))
        return reshape_output_1


class Reshape903(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 8192))
        return reshape_output_1


class Reshape904(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(512, 1, 3, 3))
        return reshape_output_1


class Reshape905(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(256, 2560))
        return reshape_output_1


class Reshape906(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 32, 80))
        return reshape_output_1


class Reshape907(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 2560))
        return reshape_output_1


class Reshape908(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 256, 80))
        return reshape_output_1


class Reshape909(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 80, 256))
        return reshape_output_1


class Reshape910(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 256, 80))
        return reshape_output_1


class Reshape911(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 10240))
        return reshape_output_1


class Reshape912(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(29, 1024))
        return reshape_output_1


class Reshape913(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 29, 16, 64))
        return reshape_output_1


class Reshape914(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 29, 1024))
        return reshape_output_1


class Reshape915(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 29, 64))
        return reshape_output_1


class Reshape916(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 29, 29))
        return reshape_output_1


class Reshape917(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 29, 29))
        return reshape_output_1


class Reshape918(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 64, 29))
        return reshape_output_1


class Reshape919(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 29, 64))
        return reshape_output_1


class Reshape920(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 29, 2816))
        return reshape_output_1


class Reshape921(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2016, 1, 1))
        return reshape_output_1


class Reshape922(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1920, 1, 1))
        return reshape_output_1


class Reshape923(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1280, 1, 3, 3))
        return reshape_output_1


class Reshape924(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2048, 1, 3, 3))
        return reshape_output_1


class Reshape925(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 4, 1))
        return reshape_output_1


class Reshape926(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 1))
        return reshape_output_1


class Reshape927(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 1536))
        return reshape_output_1


class Reshape928(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 1, 24, 64))
        return reshape_output_1


class Reshape929(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 1, 1536))
        return reshape_output_1


class Reshape930(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(48, 1, 64))
        return reshape_output_1


class Reshape931(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 24, 1, 64))
        return reshape_output_1


class Reshape932(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 13))
        return reshape_output_1


class Reshape933(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(26, 768))
        return reshape_output_1


class Reshape934(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 13, 12, 64))
        return reshape_output_1


class Reshape935(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 13, 768))
        return reshape_output_1


class Reshape936(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(24, 13, 64))
        return reshape_output_1


class Reshape937(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 12, 13, 13))
        return reshape_output_1


class Reshape938(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(24, 13, 13))
        return reshape_output_1


class Reshape939(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(24, 64, 13))
        return reshape_output_1


class Reshape940(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 12, 13, 64))
        return reshape_output_1


class Reshape941(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 13, 3072))
        return reshape_output_1


class Reshape942(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(26, 3072))
        return reshape_output_1


class Reshape943(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 13, 1536))
        return reshape_output_1


class Reshape944(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 13, 24, 64))
        return reshape_output_1


class Reshape945(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(26, 1536))
        return reshape_output_1


class Reshape946(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(48, 13, 64))
        return reshape_output_1


class Reshape947(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 24, 1, 13))
        return reshape_output_1


class Reshape948(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(48, 1, 13))
        return reshape_output_1


class Reshape949(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 1, 6144))
        return reshape_output_1


class Reshape950(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 6144))
        return reshape_output_1


class Reshape951(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 1, 2048))
        return reshape_output_1


class Reshape952(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 1, 2048))
        return reshape_output_1


class Reshape953(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(197, 1, 3072))
        return reshape_output_1


class Reshape954(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(197, 1, 3, 1024))
        return reshape_output_1


class Reshape955(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(197, 16, 64))
        return reshape_output_1


class Reshape956(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(6, 768))
        return reshape_output_1


class Reshape957(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 6, 12, 64))
        return reshape_output_1


class Reshape958(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 6, 768))
        return reshape_output_1


class Reshape959(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 6, 64))
        return reshape_output_1


class Reshape960(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 64, 6))
        return reshape_output_1


class Reshape961(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 6, 6))
        return reshape_output_1


class Reshape962(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 6, 6))
        return reshape_output_1


class Reshape963(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 6, 64))
        return reshape_output_1


class Reshape964(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 768))
        return reshape_output_1


class Reshape965(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 12, 64))
        return reshape_output_1


class Reshape966(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 768))
        return reshape_output_1


class Reshape967(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 8, 64))
        return reshape_output_1


class Reshape968(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 64, 8))
        return reshape_output_1


class Reshape969(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 8, 8))
        return reshape_output_1


class Reshape970(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 8, 8))
        return reshape_output_1


class Reshape971(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 8, 64))
        return reshape_output_1


class Reshape972(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(10, 768))
        return reshape_output_1


class Reshape973(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 10, 12, 64))
        return reshape_output_1


class Reshape974(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 10, 768))
        return reshape_output_1


class Reshape975(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 10, 64))
        return reshape_output_1


class Reshape976(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 64, 10))
        return reshape_output_1


class Reshape977(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 10, 10))
        return reshape_output_1


class Reshape978(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 10, 10))
        return reshape_output_1


class Reshape979(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 10, 64))
        return reshape_output_1


class Reshape980(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(25, 1, 96))
        return reshape_output_1


class Reshape981(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 128, 64))
        return reshape_output_1


class Reshape982(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 8, 8, 8, 128))
        return reshape_output_1


class Reshape983(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 64, 128, 64))
        return reshape_output_1


class Reshape984(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 16, 3, 96))
        return reshape_output_1


class Reshape985(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 16, 96))
        return reshape_output_1


class Reshape986(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 32, 96))
        return reshape_output_1


class Reshape987(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 1, 32))
        return reshape_output_1


class Reshape988(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 32, 96))
        return reshape_output_1


class Reshape989(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 1536))
        return reshape_output_1


class Reshape990(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 1536))
        return reshape_output_1


class Reshape991(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 16, 32, 1))
        return reshape_output_1


class Reshape992(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 64, 256))
        return reshape_output_1


class Reshape993(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1664, 1, 1))
        return reshape_output_1


class Reshape994(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(384, 768))
        return reshape_output_1


class Reshape995(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 384, 12, 64))
        return reshape_output_1


class Reshape996(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 384, 768))
        return reshape_output_1


class Reshape997(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 384, 64))
        return reshape_output_1


class Reshape998(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 384, 384))
        return reshape_output_1


class Reshape999(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 1, 384))
        return reshape_output_1


class Reshape1000(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 384, 384))
        return reshape_output_1


class Reshape1001(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 64, 384))
        return reshape_output_1


class Reshape1002(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 384, 64))
        return reshape_output_1


class Reshape1003(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2048, 1, 4))
        return reshape_output_1


class Reshape1004(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(6, 2048))
        return reshape_output_1


class Reshape1005(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 6, 64))
        return reshape_output_1


class Reshape1006(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2048, 16))
        return reshape_output_1


class Reshape1007(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 6, 16))
        return reshape_output_1


class Reshape1008(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16))
        return reshape_output_1


class Reshape1009(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 768, 49, 1))
        return reshape_output_1


class Reshape1010(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 768, 49))
        return reshape_output_1


class Reshape1011(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 50272))
        return reshape_output_1


class Reshape1012(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(39, 1536))
        return reshape_output_1


class Reshape1013(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 39, 12, 128))
        return reshape_output_1


class Reshape1014(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 39, 1536))
        return reshape_output_1


class Reshape1015(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 39, 128))
        return reshape_output_1


class Reshape1016(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 39, 256))
        return reshape_output_1


class Reshape1017(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 39, 2, 128))
        return reshape_output_1


class Reshape1018(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 39, 128))
        return reshape_output_1


class Reshape1019(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 39, 39))
        return reshape_output_1


class Reshape1020(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 39, 39))
        return reshape_output_1


class Reshape1021(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 128, 39))
        return reshape_output_1


class Reshape1022(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 39, 8960))
        return reshape_output_1


class Reshape1023(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 3712, 1, 1))
        return reshape_output_1


class Reshape1024(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 888, 1, 1))
        return reshape_output_1


class Reshape1025(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 49, 288))
        return reshape_output_1


class Reshape1026(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 49, 3, 3, 32))
        return reshape_output_1


class Reshape1027(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 56, 56, 384))
        return reshape_output_1


class Reshape1028(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(3136, 384))
        return reshape_output_1


class Reshape1029(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 49, 576))
        return reshape_output_1


class Reshape1030(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 49, 3, 6, 32))
        return reshape_output_1


class Reshape1031(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 28, 28, 768))
        return reshape_output_1


class Reshape1032(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(784, 768))
        return reshape_output_1


class Reshape1033(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(4, 49, 1152))
        return reshape_output_1


class Reshape1034(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(4, 49, 3, 12, 32))
        return reshape_output_1


class Reshape1035(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 14, 14, 1536))
        return reshape_output_1


class Reshape1036(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(196, 1536))
        return reshape_output_1


class Reshape1037(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 7, 1, 7, 768))
        return reshape_output_1


class Reshape1038(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 49, 2304))
        return reshape_output_1


class Reshape1039(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 49, 3, 24, 32))
        return reshape_output_1


class Reshape1040(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 7, 7, 3072))
        return reshape_output_1


class Reshape1041(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(49, 3072))
        return reshape_output_1


class Reshape1042(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(201, 768))
        return reshape_output_1


class Reshape1043(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 201, 12, 64))
        return reshape_output_1


class Reshape1044(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 201, 768))
        return reshape_output_1


class Reshape1045(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 201, 64))
        return reshape_output_1


class Reshape1046(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 201, 201))
        return reshape_output_1


class Reshape1047(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 201, 201))
        return reshape_output_1


class Reshape1048(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 64, 201))
        return reshape_output_1


class Reshape1049(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 201, 64))
        return reshape_output_1


class Reshape1050(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(11, 312))
        return reshape_output_1


class Reshape1051(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 11, 12, 26))
        return reshape_output_1


class Reshape1052(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 11, 312))
        return reshape_output_1


class Reshape1053(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 11, 26))
        return reshape_output_1


class Reshape1054(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 26, 11))
        return reshape_output_1


class Reshape1055(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 11, 26))
        return reshape_output_1


class Reshape1056(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(15, 768))
        return reshape_output_1


class Reshape1057(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 15, 12, 64))
        return reshape_output_1


class Reshape1058(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 15, 768))
        return reshape_output_1


class Reshape1059(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 15, 64))
        return reshape_output_1


class Reshape1060(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 64, 15))
        return reshape_output_1


class Reshape1061(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 15, 15))
        return reshape_output_1


class Reshape1062(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 15, 15))
        return reshape_output_1


class Reshape1063(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 15, 64))
        return reshape_output_1


class Reshape1064(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 120, 12, 1))
        return reshape_output_1


class Reshape1065(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 3, 8, 15))
        return reshape_output_1


class Reshape1066(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 12, 15))
        return reshape_output_1


class Reshape1067(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 15, 12))
        return reshape_output_1


class Reshape1068(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 12, 12))
        return reshape_output_1


class Reshape1069(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 12, 12))
        return reshape_output_1


class Reshape1070(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 12, 15))
        return reshape_output_1


class Reshape1071(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 120))
        return reshape_output_1


class Reshape1072(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 120))
        return reshape_output_1


class Reshape1073(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 12, 120))
        return reshape_output_1


class Reshape1074(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(522, 2048))
        return reshape_output_1


class Reshape1075(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 522, 8, 256))
        return reshape_output_1


class Reshape1076(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 522, 2048))
        return reshape_output_1


class Reshape1077(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 522, 256))
        return reshape_output_1


class Reshape1078(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 522, 4, 256))
        return reshape_output_1


class Reshape1079(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 522, 256))
        return reshape_output_1


class Reshape1080(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 522, 522))
        return reshape_output_1


class Reshape1081(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 522, 522))
        return reshape_output_1


class Reshape1082(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 256, 522))
        return reshape_output_1


class Reshape1083(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 522, 8192))
        return reshape_output_1


class Reshape1084(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 64, 19200, 1))
        return reshape_output_1


class Reshape1085(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 19200, 1, 64))
        return reshape_output_1


class Reshape1086(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 120, 160, 64))
        return reshape_output_1


class Reshape1087(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 64, 120, 160))
        return reshape_output_1


class Reshape1088(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 64, 300))
        return reshape_output_1


class Reshape1089(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(300, 64))
        return reshape_output_1


class Reshape1090(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 300, 1, 64))
        return reshape_output_1


class Reshape1091(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 300, 64))
        return reshape_output_1


class Reshape1092(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 19200, 300))
        return reshape_output_1


class Reshape1093(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 19200, 300))
        return reshape_output_1


class Reshape1094(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 120, 160))
        return reshape_output_1


class Reshape1095(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 19200, 1))
        return reshape_output_1


class Reshape1096(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 4800, 1))
        return reshape_output_1


class Reshape1097(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4800, 2, 64))
        return reshape_output_1


class Reshape1098(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 60, 80, 128))
        return reshape_output_1


class Reshape1099(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 4800, 64))
        return reshape_output_1


class Reshape1100(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 60, 80))
        return reshape_output_1


class Reshape1101(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 300))
        return reshape_output_1


class Reshape1102(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(300, 128))
        return reshape_output_1


class Reshape1103(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 300, 2, 64))
        return reshape_output_1


class Reshape1104(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 300, 128))
        return reshape_output_1


class Reshape1105(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 300, 64))
        return reshape_output_1


class Reshape1106(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2, 4800, 300))
        return reshape_output_1


class Reshape1107(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 4800, 300))
        return reshape_output_1


class Reshape1108(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 64, 300))
        return reshape_output_1


class Reshape1109(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2, 4800, 64))
        return reshape_output_1


class Reshape1110(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(4800, 128))
        return reshape_output_1


class Reshape1111(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4800, 128))
        return reshape_output_1


class Reshape1112(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 512, 60, 80))
        return reshape_output_1


class Reshape1113(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 512, 4800, 1))
        return reshape_output_1


class Reshape1114(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 320, 1200, 1))
        return reshape_output_1


class Reshape1115(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1200, 5, 64))
        return reshape_output_1


class Reshape1116(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 30, 40, 320))
        return reshape_output_1


class Reshape1117(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(5, 1200, 64))
        return reshape_output_1


class Reshape1118(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 320, 30, 40))
        return reshape_output_1


class Reshape1119(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 320, 300))
        return reshape_output_1


class Reshape1120(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(300, 320))
        return reshape_output_1


class Reshape1121(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 300, 5, 64))
        return reshape_output_1


class Reshape1122(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 300, 320))
        return reshape_output_1


class Reshape1123(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(5, 300, 64))
        return reshape_output_1


class Reshape1124(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 5, 1200, 300))
        return reshape_output_1


class Reshape1125(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(5, 1200, 300))
        return reshape_output_1


class Reshape1126(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(5, 64, 300))
        return reshape_output_1


class Reshape1127(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 5, 1200, 64))
        return reshape_output_1


class Reshape1128(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1200, 320))
        return reshape_output_1


class Reshape1129(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1200, 320))
        return reshape_output_1


class Reshape1130(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1280, 30, 40))
        return reshape_output_1


class Reshape1131(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1280, 1200, 1))
        return reshape_output_1


class Reshape1132(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 512, 300, 1))
        return reshape_output_1


class Reshape1133(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(300, 512))
        return reshape_output_1


class Reshape1134(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 300, 8, 64))
        return reshape_output_1


class Reshape1135(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 15, 20, 512))
        return reshape_output_1


class Reshape1136(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 300, 512))
        return reshape_output_1


class Reshape1137(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 300, 64))
        return reshape_output_1


class Reshape1138(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 300, 300))
        return reshape_output_1


class Reshape1139(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 300, 300))
        return reshape_output_1


class Reshape1140(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 64, 300))
        return reshape_output_1


class Reshape1141(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 300, 64))
        return reshape_output_1


class Reshape1142(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2048, 15, 20))
        return reshape_output_1


class Reshape1143(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2048, 300, 1))
        return reshape_output_1


class Reshape1144(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 30, 40))
        return reshape_output_1


class Reshape1145(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 60, 80))
        return reshape_output_1


class Reshape1146(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 120, 160))
        return reshape_output_1


class Reshape1147(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(3072, 1, 4))
        return reshape_output_1


class Reshape1148(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(6, 3072))
        return reshape_output_1


class Reshape1149(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 6, 96))
        return reshape_output_1


class Reshape1150(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 3072, 16))
        return reshape_output_1


class Reshape1151(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 3072))
        return reshape_output_1


class Reshape1152(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(7, 2048))
        return reshape_output_1


class Reshape1153(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 7, 32, 64))
        return reshape_output_1


class Reshape1154(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 7, 2048))
        return reshape_output_1


class Reshape1155(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 7, 64))
        return reshape_output_1


class Reshape1156(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 7, 7))
        return reshape_output_1


class Reshape1157(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 7, 7))
        return reshape_output_1


class Reshape1158(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 64, 7))
        return reshape_output_1


class Reshape1159(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 7, 64))
        return reshape_output_1


class Reshape1160(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 7, 8192))
        return reshape_output_1


class Reshape1161(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(29, 1536))
        return reshape_output_1


class Reshape1162(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 29, 12, 128))
        return reshape_output_1


class Reshape1163(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 29, 1536))
        return reshape_output_1


class Reshape1164(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 29, 128))
        return reshape_output_1


class Reshape1165(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 29, 256))
        return reshape_output_1


class Reshape1166(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 29, 2, 128))
        return reshape_output_1


class Reshape1167(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 29, 128))
        return reshape_output_1


class Reshape1168(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 29, 29))
        return reshape_output_1


class Reshape1169(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 29, 29))
        return reshape_output_1


class Reshape1170(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 128, 29))
        return reshape_output_1


class Reshape1171(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 29, 8960))
        return reshape_output_1


class Reshape1172(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 61, 2048))
        return reshape_output_1


class Reshape1173(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 2048))
        return reshape_output_1


class Reshape1174(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 17, 4480))
        return reshape_output_1


class Reshape1175(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 17, 1120))
        return reshape_output_1


class Reshape1176(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 17, 280))
        return reshape_output_1


class Reshape1177(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1152, 1, 1))
        return reshape_output_1


class Reshape1178(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 14, 1))
        return reshape_output_1


class Reshape1179(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(588, 2048))
        return reshape_output_1


class Reshape1180(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 588, 16, 128))
        return reshape_output_1


class Reshape1181(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 588, 2048))
        return reshape_output_1


class Reshape1182(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 588, 128))
        return reshape_output_1


class Reshape1183(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 588, 588))
        return reshape_output_1


class Reshape1184(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 588, 588))
        return reshape_output_1


class Reshape1185(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 128, 588))
        return reshape_output_1


class Reshape1186(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 588, 128))
        return reshape_output_1


class Reshape1187(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 588, 5504))
        return reshape_output_1


class Reshape1188(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(528, 1, 3, 3))
        return reshape_output_1


class Reshape1189(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(528, 1, 5, 5))
        return reshape_output_1


class Reshape1190(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(720, 1, 5, 5))
        return reshape_output_1


class Reshape1191(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1248, 1, 5, 5))
        return reshape_output_1


class Reshape1192(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1248, 1, 3, 3))
        return reshape_output_1


class Reshape1193(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 768, 256, 1))
        return reshape_output_1


class Reshape1194(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 257, 3, 12, 64))
        return reshape_output_1


class Reshape1195(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 257, 64))
        return reshape_output_1


class Reshape1196(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 257, 64))
        return reshape_output_1


class Reshape1197(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 257, 257))
        return reshape_output_1


class Reshape1198(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 257, 257))
        return reshape_output_1


class Reshape1199(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 64, 257))
        return reshape_output_1


class Reshape1200(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(257, 768))
        return reshape_output_1


class Reshape1201(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 257, 768))
        return reshape_output_1


class Reshape1202(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 27, 257, 1))
        return reshape_output_1


class Reshape1203(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 768, 257, 1))
        return reshape_output_1


class Reshape1204(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(432, 1, 3, 3))
        return reshape_output_1


class Reshape1205(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(720, 1, 3, 3))
        return reshape_output_1


class Reshape1206(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(88, 1, 3, 3))
        return reshape_output_1


class Reshape1207(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(96, 1, 5, 5))
        return reshape_output_1


class Reshape1208(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 768))
        return reshape_output_1


class Reshape1209(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 32, 32))
        return reshape_output_1


class Reshape1210(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 32, 32))
        return reshape_output_1


class Reshape1211(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(35, 896))
        return reshape_output_1


class Reshape1212(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 35, 14, 64))
        return reshape_output_1


class Reshape1213(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 35, 896))
        return reshape_output_1


class Reshape1214(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(14, 35, 64))
        return reshape_output_1


class Reshape1215(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 35, 128))
        return reshape_output_1


class Reshape1216(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 35, 2, 64))
        return reshape_output_1


class Reshape1217(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 14, 35, 64))
        return reshape_output_1


class Reshape1218(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 14, 35, 35))
        return reshape_output_1


class Reshape1219(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(14, 35, 35))
        return reshape_output_1


class Reshape1220(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(14, 64, 35))
        return reshape_output_1


class Reshape1221(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 35, 4864))
        return reshape_output_1


class Reshape1222(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 912, 1, 1))
        return reshape_output_1


class Reshape1223(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1512, 1, 1))
        return reshape_output_1


class Reshape1224(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 64, 3, 4, 32))
        return reshape_output_1


class Reshape1225(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 4, 64, 32))
        return reshape_output_1


class Reshape1226(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(256, 64, 32))
        return reshape_output_1


class Reshape1227(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 4, 64, 64))
        return reshape_output_1


class Reshape1228(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(225, 4))
        return reshape_output_1


class Reshape1229(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 64, 4))
        return reshape_output_1


class Reshape1230(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(256, 64, 64))
        return reshape_output_1


class Reshape1231(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 64, 4, 64, 64))
        return reshape_output_1


class Reshape1232(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(256, 32, 64))
        return reshape_output_1


class Reshape1233(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 64, 64, 512))
        return reshape_output_1


class Reshape1234(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(4096, 512))
        return reshape_output_1


class Reshape1235(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1024, 512))
        return reshape_output_1


class Reshape1236(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 32, 256))
        return reshape_output_1


class Reshape1237(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 8, 4, 8, 256))
        return reshape_output_1


class Reshape1238(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1024, 256))
        return reshape_output_1


class Reshape1239(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 64, 3, 8, 32))
        return reshape_output_1


class Reshape1240(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 8, 64, 32))
        return reshape_output_1


class Reshape1241(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(128, 64, 32))
        return reshape_output_1


class Reshape1242(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 8, 64, 64))
        return reshape_output_1


class Reshape1243(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(225, 8))
        return reshape_output_1


class Reshape1244(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 64, 8))
        return reshape_output_1


class Reshape1245(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(128, 64, 64))
        return reshape_output_1


class Reshape1246(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 8, 64, 64))
        return reshape_output_1


class Reshape1247(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(128, 32, 64))
        return reshape_output_1


class Reshape1248(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 4, 8, 8, 256))
        return reshape_output_1


class Reshape1249(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 32, 1024))
        return reshape_output_1


class Reshape1250(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1024, 1024))
        return reshape_output_1


class Reshape1251(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2, 8, 2, 8, 512))
        return reshape_output_1


class Reshape1252(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(4, 64, 3, 16, 32))
        return reshape_output_1


class Reshape1253(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(4, 16, 64, 32))
        return reshape_output_1


class Reshape1254(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 64, 32))
        return reshape_output_1


class Reshape1255(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(4, 16, 64, 64))
        return reshape_output_1


class Reshape1256(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(225, 16))
        return reshape_output_1


class Reshape1257(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 64, 16))
        return reshape_output_1


class Reshape1258(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 64, 64))
        return reshape_output_1


class Reshape1259(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 16, 64, 64))
        return reshape_output_1


class Reshape1260(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 32, 64))
        return reshape_output_1


class Reshape1261(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 2, 2, 8, 8, 512))
        return reshape_output_1


class Reshape1262(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 2048))
        return reshape_output_1


class Reshape1263(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 8, 1024))
        return reshape_output_1


class Reshape1264(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 64, 1024))
        return reshape_output_1


class Reshape1265(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 8, 1, 8, 1024))
        return reshape_output_1


class Reshape1266(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 1024))
        return reshape_output_1


class Reshape1267(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 64, 3, 32, 32))
        return reshape_output_1


class Reshape1268(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 64, 32))
        return reshape_output_1


class Reshape1269(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 64, 32))
        return reshape_output_1


class Reshape1270(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 32, 64, 64))
        return reshape_output_1


class Reshape1271(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(225, 32))
        return reshape_output_1


class Reshape1272(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(32, 64, 64))
        return reshape_output_1


class Reshape1273(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 1, 8, 8, 1024))
        return reshape_output_1


class Reshape1274(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 8, 4096))
        return reshape_output_1


class Reshape1275(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 4096))
        return reshape_output_1


class Reshape1276(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(61, 1024))
        return reshape_output_1


class Reshape1277(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 61, 64))
        return reshape_output_1


class Reshape1278(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 61, 61))
        return reshape_output_1


class Reshape1279(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 61, 61))
        return reshape_output_1


class Reshape1280(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 64, 61))
        return reshape_output_1


class Reshape1281(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 61, 64))
        return reshape_output_1


class Reshape1282(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 61, 2816))
        return reshape_output_1


class Reshape1283(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 1, 61))
        return reshape_output_1


class Reshape1284(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 1, 61))
        return reshape_output_1


class Reshape1285(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 2816))
        return reshape_output_1


class Reshape1286(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(512, 80, 3, 1))
        return reshape_output_1


class Reshape1287(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 512, 3000))
        return reshape_output_1


class Reshape1288(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 512, 3000, 1))
        return reshape_output_1


class Reshape1289(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(512, 512, 3, 1))
        return reshape_output_1


class Reshape1290(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 512, 1500))
        return reshape_output_1


class Reshape1291(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1500, 512))
        return reshape_output_1


class Reshape1292(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1500, 8, 64))
        return reshape_output_1


class Reshape1293(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1500, 512))
        return reshape_output_1


class Reshape1294(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 1500, 64))
        return reshape_output_1


class Reshape1295(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 1500, 1500))
        return reshape_output_1


class Reshape1296(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 1500, 1500))
        return reshape_output_1


class Reshape1297(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 64, 1500))
        return reshape_output_1


class Reshape1298(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 1500, 64))
        return reshape_output_1


class Reshape1299(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 8, 1, 1500))
        return reshape_output_1


class Reshape1300(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(8, 1, 1500))
        return reshape_output_1


class Reshape1301(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 27, 27, 12))
        return reshape_output_1


class Reshape1302(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(729, 12))
        return reshape_output_1


class Reshape1303(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(197, 197, 12))
        return reshape_output_1


class Reshape1304(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 7))
        return reshape_output_1


class Reshape1305(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(14, 512))
        return reshape_output_1


class Reshape1306(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 7, 8, 64))
        return reshape_output_1


class Reshape1307(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 7, 512))
        return reshape_output_1


class Reshape1308(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 7, 64))
        return reshape_output_1


class Reshape1309(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 8, 7, 7))
        return reshape_output_1


class Reshape1310(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 7, 7))
        return reshape_output_1


class Reshape1311(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 8, 7, 64))
        return reshape_output_1


class Reshape1312(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(2, 7, 2048))
        return reshape_output_1


class Reshape1313(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(14, 2048))
        return reshape_output_1


class Reshape1314(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 384, 196, 1))
        return reshape_output_1


class Reshape1315(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(197, 384))
        return reshape_output_1


class Reshape1316(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 197, 6, 64))
        return reshape_output_1


class Reshape1317(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 197, 384))
        return reshape_output_1


class Reshape1318(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(6, 197, 64))
        return reshape_output_1


class Reshape1319(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 6, 197, 197))
        return reshape_output_1


class Reshape1320(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(6, 197, 197))
        return reshape_output_1


class Reshape1321(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(6, 64, 197))
        return reshape_output_1


class Reshape1322(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 6, 197, 64))
        return reshape_output_1


class Reshape1323(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 256, 3072))
        return reshape_output_1


class Reshape1324(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(256, 3072))
        return reshape_output_1


class Reshape1325(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 576, 1, 1))
        return reshape_output_1


class Reshape1326(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 512, 1, 512))
        return reshape_output_1


class Reshape1327(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 50176, 256))
        return reshape_output_1


class Reshape1328(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(50176, 512))
        return reshape_output_1


class Reshape1329(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 50176, 1, 512))
        return reshape_output_1


class Reshape1330(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 50176, 512))
        return reshape_output_1


class Reshape1331(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1088, 1, 1))
        return reshape_output_1


class Reshape1332(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 400, 1, 1))
        return reshape_output_1


class Reshape1333(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 5776))
        return reshape_output_1


class Reshape1334(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 2166))
        return reshape_output_1


class Reshape1335(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 600))
        return reshape_output_1


class Reshape1336(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 150))
        return reshape_output_1


class Reshape1337(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 4, 36))
        return reshape_output_1


class Reshape1338(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 81, 5776))
        return reshape_output_1


class Reshape1339(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 81, 2166))
        return reshape_output_1


class Reshape1340(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 81, 600))
        return reshape_output_1


class Reshape1341(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 81, 150))
        return reshape_output_1


class Reshape1342(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 81, 36))
        return reshape_output_1


class Reshape1343(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 81, 4))
        return reshape_output_1


class Reshape1344(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(50, 768))
        return reshape_output_1


class Reshape1345(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(50, 1, 2304))
        return reshape_output_1


class Reshape1346(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(50, 1, 3, 768))
        return reshape_output_1


class Reshape1347(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(50, 12, 64))
        return reshape_output_1


class Reshape1348(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 50, 64))
        return reshape_output_1


class Reshape1349(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 50, 64))
        return reshape_output_1


class Reshape1350(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 50, 50))
        return reshape_output_1


class Reshape1351(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 50, 50))
        return reshape_output_1


class Reshape1352(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(50, 1, 768))
        return reshape_output_1


class Reshape1353(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(384, 80, 3, 1))
        return reshape_output_1


class Reshape1354(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 384, 3000))
        return reshape_output_1


class Reshape1355(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 384, 3000, 1))
        return reshape_output_1


class Reshape1356(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(384, 384, 3, 1))
        return reshape_output_1


class Reshape1357(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 384, 1500))
        return reshape_output_1


class Reshape1358(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1500, 384))
        return reshape_output_1


class Reshape1359(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1500, 6, 64))
        return reshape_output_1


class Reshape1360(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1500, 384))
        return reshape_output_1


class Reshape1361(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(6, 1500, 64))
        return reshape_output_1


class Reshape1362(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 6, 1500, 1500))
        return reshape_output_1


class Reshape1363(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(6, 1500, 1500))
        return reshape_output_1


class Reshape1364(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(6, 64, 1500))
        return reshape_output_1


class Reshape1365(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 6, 1500, 64))
        return reshape_output_1


class Reshape1366(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 6, 1, 1500))
        return reshape_output_1


class Reshape1367(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(6, 1, 1500))
        return reshape_output_1


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (
        Reshape0,
        [((1, 1000, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_dla_dla102_visual_bb_torchvision",
                "pt_dla_dla102_visual_bb_torchvision",
                "onnx_dla_dla102x2_visual_bb_torchvision",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "onnx_dla_dla169_visual_bb_torchvision",
                "pt_dla_dla169_visual_bb_torchvision",
                "onnx_dla_dla34_visual_bb_torchvision",
                "pt_dla_dla34_visual_bb_torchvision",
                "onnx_dla_dla46_c_visual_bb_torchvision",
                "onnx_dla_dla60_visual_bb_torchvision",
                "pt_dla_dla60_visual_bb_torchvision",
                "onnx_dla_dla60x_visual_bb_torchvision",
                "pt_dla_dla60x_visual_bb_torchvision",
                "onnx_dla_dla60x_c_visual_bb_torchvision",
                "pt_dla_dla60x_c_visual_bb_torchvision",
                "onnx_dla_dla102x_visual_bb_torchvision",
                "pt_dla_dla102x_visual_bb_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 1000)"},
        },
    ),
    (
        Reshape1,
        [((1, 1000, 1, 1), torch.float32)],
        {"model_names": ["pt_dla_dla34_in1k_img_cls_timm"], "pcc": 0.99, "args": {"shape": "(1, 1000, 1, 1)"}},
    ),
    (
        Reshape2,
        [((1, 1280, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_lite0_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_050_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm",
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 1280)"},
        },
    ),
    (
        Reshape3,
        [((1, 1280, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 1280, 1, 1)"},
        },
    ),
    (
        Reshape4,
        [((1, 13, 384), torch.float32)],
        {
            "model_names": ["onnx_minilm_sentence_transformers_all_minilm_l6_v2_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(13, 384)"},
        },
    ),
    (
        Reshape5,
        [((1, 13, 384), torch.float32)],
        {
            "model_names": ["onnx_minilm_sentence_transformers_all_minilm_l6_v2_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 13, 12, 32)"},
        },
    ),
    (
        Reshape6,
        [((13, 384), torch.float32)],
        {
            "model_names": ["onnx_minilm_sentence_transformers_all_minilm_l6_v2_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 13, 384)"},
        },
    ),
    (
        Reshape7,
        [((1, 12, 13, 32), torch.float32)],
        {
            "model_names": ["onnx_minilm_sentence_transformers_all_minilm_l6_v2_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(12, 13, 32)"},
        },
    ),
    (
        Reshape8,
        [((1, 12, 32, 13), torch.float32)],
        {
            "model_names": ["onnx_minilm_sentence_transformers_all_minilm_l6_v2_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(12, 32, 13)"},
        },
    ),
    (
        Reshape9,
        [((12, 13, 13), torch.float32)],
        {
            "model_names": ["onnx_minilm_sentence_transformers_all_minilm_l6_v2_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 12, 13, 13)"},
        },
    ),
    (
        Reshape10,
        [((1, 12, 13, 13), torch.float32)],
        {
            "model_names": ["onnx_minilm_sentence_transformers_all_minilm_l6_v2_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(12, 13, 13)"},
        },
    ),
    (
        Reshape11,
        [((12, 13, 32), torch.float32)],
        {
            "model_names": ["onnx_minilm_sentence_transformers_all_minilm_l6_v2_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 12, 13, 32)"},
        },
    ),
    (
        Reshape4,
        [((1, 13, 12, 32), torch.float32)],
        {
            "model_names": ["onnx_minilm_sentence_transformers_all_minilm_l6_v2_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(13, 384)"},
        },
    ),
    (
        Reshape12,
        [((1, 1, 384), torch.float32)],
        {
            "model_names": [
                "onnx_minilm_sentence_transformers_all_minilm_l6_v2_seq_cls_hf",
                "pt_deit_facebook_deit_small_patch16_224_img_cls_hf",
                "pt_whisper_openai_whisper_tiny_speech_recognition_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 384)"},
        },
    ),
    (
        Reshape13,
        [((1, 1, 384), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 1, 6, 64)"},
        },
    ),
    (
        Reshape14,
        [((1, 64, 128, 128), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b3_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 64, 16384)"},
        },
    ),
    (
        Reshape15,
        [((1, 64, 128, 128), torch.float32)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 64, 16384, 1)"},
        },
    ),
    (
        Reshape16,
        [((1, 64, 128, 128), torch.float32)],
        {
            "model_names": [
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
                "pt_albert_xxlarge_v2_token_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(64, 128, 128)"},
        },
    ),
    (
        Reshape17,
        [((1, 16384, 64), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b3_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 16384, 1, 64)"},
        },
    ),
    (
        Reshape18,
        [((1, 16384, 64), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b3_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 128, 128, 64)"},
        },
    ),
    (
        Reshape19,
        [((1, 64, 16384), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b3_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 64, 128, 128)"},
        },
    ),
    (
        Reshape20,
        [((1, 64, 16, 16), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b3_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 64, 256)"},
        },
    ),
    (
        Reshape21,
        [((1, 256, 64), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b3_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(256, 64)"},
        },
    ),
    (
        Reshape22,
        [((1, 256, 64), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b3_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 256, 1, 64)"},
        },
    ),
    (
        Reshape23,
        [((1, 256, 64), torch.float32)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 256, 2, 32)"},
        },
    ),
    (
        Reshape24,
        [((256, 64), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b3_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 256, 64)"},
        },
    ),
    (
        Reshape20,
        [((1, 1, 64, 256), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b3_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 64, 256)"},
        },
    ),
    (
        Reshape25,
        [((1, 16384, 256), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b3_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 1, 16384, 256)"},
        },
    ),
    (
        Reshape26,
        [((1, 1, 16384, 256), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b3_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 16384, 256)"},
        },
    ),
    (
        Reshape27,
        [((1, 256, 16384), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b3_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 256, 128, 128)"},
        },
    ),
    (
        Reshape28,
        [((1, 256, 128, 128), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b3_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 256, 16384)"},
        },
    ),
    (
        Reshape29,
        [((1, 256, 128, 128), torch.float32)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 256, 16384, 1)"},
        },
    ),
    (
        Reshape30,
        [((1, 128, 64, 64), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b3_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 128, 4096)"},
        },
    ),
    (
        Reshape31,
        [((1, 128, 64, 64), torch.float32)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_albert_xxlarge_v2_token_cls_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 128, 4096, 1)"},
        },
    ),
    (
        Reshape32,
        [((1, 4096, 128), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b3_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 4096, 2, 64)"},
        },
    ),
    (
        Reshape33,
        [((1, 4096, 128), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b3_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 64, 64, 128)"},
        },
    ),
    (
        Reshape34,
        [((1, 2, 4096, 64), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b3_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(2, 4096, 64)"},
        },
    ),
    (
        Reshape35,
        [((1, 128, 4096), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b3_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_albert_xxlarge_v2_token_cls_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 128, 64, 64)"},
        },
    ),
    (
        Reshape36,
        [((1, 128, 4096), torch.float32)],
        {
            "model_names": [
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
                "pt_albert_xxlarge_v2_token_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(128, 4096)"},
        },
    ),
    (
        Reshape37,
        [((1, 128, 16, 16), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b3_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 128, 256)"},
        },
    ),
    (
        Reshape38,
        [((1, 256, 128), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b3_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(256, 128)"},
        },
    ),
    (
        Reshape39,
        [((1, 256, 128), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b3_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 256, 2, 64)"},
        },
    ),
    (
        Reshape40,
        [((256, 128), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b3_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 256, 128)"},
        },
    ),
    (
        Reshape41,
        [((1, 2, 64, 256), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b3_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(2, 64, 256)"},
        },
    ),
    (
        Reshape42,
        [((2, 4096, 256), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b3_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 2, 4096, 256)"},
        },
    ),
    (
        Reshape43,
        [((1, 2, 4096, 256), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b3_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(2, 4096, 256)"},
        },
    ),
    (
        Reshape44,
        [((1, 2, 256, 64), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b3_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(2, 256, 64)"},
        },
    ),
    (
        Reshape45,
        [((2, 4096, 64), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b3_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 2, 4096, 64)"},
        },
    ),
    (
        Reshape46,
        [((1, 4096, 2, 64), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b3_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(4096, 128)"},
        },
    ),
    (
        Reshape47,
        [((4096, 128), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b3_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 4096, 128)"},
        },
    ),
    (
        Reshape48,
        [((4096, 128), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(64, 64, 128)"}},
    ),
    (
        Reshape33,
        [((4096, 128), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(1, 64, 64, 128)"}},
    ),
    (
        Reshape49,
        [((1, 512, 4096), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b3_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 512, 64, 64)"},
        },
    ),
    (
        Reshape50,
        [((1, 512, 64, 64), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b3_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 512, 4096)"},
        },
    ),
    (
        Reshape51,
        [((1, 512, 64, 64), torch.float32)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 512, 4096, 1)"},
        },
    ),
    (
        Reshape52,
        [((1, 320, 32, 32), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b3_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 320, 1024)"},
        },
    ),
    (
        Reshape53,
        [((1, 320, 32, 32), torch.float32)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 320, 1024, 1)"},
        },
    ),
    (
        Reshape54,
        [((1, 1024, 320), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b3_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 1024, 5, 64)"},
        },
    ),
    (
        Reshape55,
        [((1, 1024, 320), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b3_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 32, 32, 320)"},
        },
    ),
    (
        Reshape56,
        [((1, 5, 1024, 64), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b3_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(5, 1024, 64)"},
        },
    ),
    (
        Reshape57,
        [((1, 320, 1024), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b3_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 320, 32, 32)"},
        },
    ),
    (
        Reshape58,
        [((1, 320, 16, 16), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b3_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 320, 256)"},
        },
    ),
    (
        Reshape59,
        [((1, 256, 320), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b3_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(256, 320)"},
        },
    ),
    (
        Reshape60,
        [((1, 256, 320), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b3_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 256, 5, 64)"},
        },
    ),
    (
        Reshape61,
        [((256, 320), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b3_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 256, 320)"},
        },
    ),
    (
        Reshape62,
        [((1, 5, 64, 256), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b3_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(5, 64, 256)"},
        },
    ),
    (
        Reshape63,
        [((5, 1024, 256), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b3_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 5, 1024, 256)"},
        },
    ),
    (
        Reshape64,
        [((1, 5, 1024, 256), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b3_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(5, 1024, 256)"},
        },
    ),
    (
        Reshape65,
        [((1, 5, 256, 64), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b3_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(5, 256, 64)"},
        },
    ),
    (
        Reshape66,
        [((5, 1024, 64), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b3_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 5, 1024, 64)"},
        },
    ),
    (
        Reshape67,
        [((1, 1024, 5, 64), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b3_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1024, 320)"},
        },
    ),
    (
        Reshape68,
        [((1024, 320), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b3_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 1024, 320)"},
        },
    ),
    (
        Reshape69,
        [((1, 1280, 1024), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b3_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 1280, 32, 32)"},
        },
    ),
    (
        Reshape70,
        [((1, 1280, 32, 32), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b3_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 1280, 1024)"},
        },
    ),
    (
        Reshape71,
        [((1, 1280, 32, 32), torch.float32)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 1280, 1024, 1)"},
        },
    ),
    (
        Reshape72,
        [((1, 512, 16, 16), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b3_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 512, 256)"},
        },
    ),
    (
        Reshape73,
        [((1, 512, 16, 16), torch.float32)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 512, 256, 1)"},
        },
    ),
    (
        Reshape74,
        [((1, 256, 512), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b3_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(256, 512)"},
        },
    ),
    (
        Reshape75,
        [((1, 256, 512), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b3_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 256, 8, 64)"},
        },
    ),
    (
        Reshape76,
        [((1, 256, 512), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b3_img_cls_hf",
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 256, 512)"},
        },
    ),
    (
        Reshape77,
        [((1, 256, 512), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 16, 16, 512)"},
        },
    ),
    (
        Reshape78,
        [((1, 256, 512), torch.float32)],
        {"model_names": ["pt_mlp_mixer_base_img_cls_github"], "pcc": 0.99, "args": {"shape": "(1, 256, 512, 1)"}},
    ),
    (
        Reshape76,
        [((256, 512), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b3_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_mlp_mixer_base_img_cls_github",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 256, 512)"},
        },
    ),
    (
        Reshape75,
        [((256, 512), torch.float32)],
        {
            "model_names": ["pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 256, 8, 64)"},
        },
    ),
    (
        Reshape74,
        [((256, 512), torch.float32)],
        {"model_names": ["pt_opt_facebook_opt_350m_clm_hf"], "pcc": 0.99, "args": {"shape": "(256, 512)"}},
    ),
    (
        Reshape77,
        [((256, 512), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(1, 16, 16, 512)"}},
    ),
    (
        Reshape79,
        [((256, 512), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(4, 64, 512)"}},
    ),
    (
        Reshape80,
        [((1, 8, 256, 64), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b3_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(8, 256, 64)"},
        },
    ),
    (
        Reshape81,
        [((1, 8, 64, 256), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b3_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(8, 64, 256)"},
        },
    ),
    (
        Reshape82,
        [((8, 256, 256), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b3_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_perceiverio_deepmind_language_perceiver_mlm_hf",
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 8, 256, 256)"},
        },
    ),
    (
        Reshape83,
        [((1, 8, 256, 256), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b3_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_perceiverio_deepmind_language_perceiver_mlm_hf",
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(8, 256, 256)"},
        },
    ),
    (
        Reshape84,
        [((8, 256, 64), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b3_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 8, 256, 64)"},
        },
    ),
    (
        Reshape74,
        [((1, 256, 8, 64), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b3_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(256, 512)"},
        },
    ),
    (
        Reshape85,
        [((1, 2048, 256), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b3_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 2048, 16, 16)"},
        },
    ),
    (
        Reshape86,
        [((1, 2048, 256), torch.float32)],
        {
            "model_names": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 2048, 8, 32)"},
        },
    ),
    (
        Reshape87,
        [((1, 2048, 16, 16), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b3_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 2048, 256)"},
        },
    ),
    (
        Reshape88,
        [((1, 2048, 16, 16), torch.float32)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 2048, 256, 1)"},
        },
    ),
    (
        Reshape89,
        [((1, 256, 256), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 256, 16, 16)"},
        },
    ),
    (
        Reshape90,
        [((1, 256, 256), torch.float32)],
        {
            "model_names": [
                "pt_perceiverio_deepmind_language_perceiver_mlm_hf",
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 256, 8, 32)"},
        },
    ),
    (
        Reshape91,
        [((1, 256, 256), torch.float32)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(256, 256)"},
        },
    ),
    (
        Reshape92,
        [((1, 256, 256), torch.float32)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf", "onnx_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 256, 256)"},
        },
    ),
    (
        Reshape93,
        [((1, 256, 256), torch.float32)],
        {
            "model_names": [
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 16, 16, 256)"},
        },
    ),
    (
        Reshape94,
        [((1, 256, 1024), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 256, 32, 32)"},
        },
    ),
    (
        Reshape95,
        [((1, 256, 1024), torch.float32)],
        {
            "model_names": [
                "pt_bart_facebook_bart_large_mnli_seq_cls_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
                "pt_opt_facebook_opt_350m_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(256, 1024)"},
        },
    ),
    (
        Reshape96,
        [((1, 256, 1024), torch.float32)],
        {
            "model_names": ["pt_bart_facebook_bart_large_mnli_seq_cls_hf", "pt_opt_facebook_opt_350m_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 256, 16, 64)"},
        },
    ),
    (
        Reshape97,
        [((1, 256, 4096), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 256, 64, 64)"},
        },
    ),
    (
        Reshape98,
        [((1, 1024, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "onnx_vovnet_v1_vovnet39_obj_det_torchhub",
                "onnx_vovnet_vovnet_v1_57_obj_det_torchhub",
                "pt_mobilenetv1_basic_img_cls_torchvision",
                "pt_vovnet_vovnet39_img_cls_osmr",
                "pt_vovnet_vovnet_v1_57_obj_det_torchhub",
                "pt_vovnet_vovnet57_img_cls_osmr",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 1024)"},
        },
    ),
    (
        Reshape99,
        [((1, 1024, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_googlenet_base_img_cls_torchvision",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_vovnet_ese_vovnet39b_obj_det_torchhub",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pd_mobilenetv1_basic_img_cls_paddlemodels",
                "pd_densenet_121_img_cls_paddlemodels",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_swin_swin_v2_b_img_cls_torchvision",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 1024, 1, 1)"},
        },
    ),
    (
        Reshape100,
        [((1, 14, 768), torch.float32)],
        {
            "model_names": ["pd_bert_bert_base_japanese_qa_padlenlp", "pt_albert_twmkn9_albert_base_v2_squad2_qa_hf"],
            "pcc": 0.99,
            "args": {"shape": "(14, 768)"},
        },
    ),
    (
        Reshape101,
        [((1, 14, 768), torch.float32)],
        {
            "model_names": ["pd_bert_bert_base_japanese_qa_padlenlp", "pt_albert_twmkn9_albert_base_v2_squad2_qa_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 14, 12, 64)"},
        },
    ),
    (
        Reshape102,
        [((14, 768), torch.float32)],
        {
            "model_names": ["pd_bert_bert_base_japanese_qa_padlenlp", "pt_albert_twmkn9_albert_base_v2_squad2_qa_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 14, 768)"},
        },
    ),
    (
        Reshape103,
        [((1, 12, 14, 64), torch.float32)],
        {
            "model_names": ["pd_bert_bert_base_japanese_qa_padlenlp", "pt_albert_twmkn9_albert_base_v2_squad2_qa_hf"],
            "pcc": 0.99,
            "args": {"shape": "(12, 14, 64)"},
        },
    ),
    (
        Reshape104,
        [((1, 12, 64, 14), torch.float32)],
        {
            "model_names": ["pd_bert_bert_base_japanese_qa_padlenlp", "pt_albert_twmkn9_albert_base_v2_squad2_qa_hf"],
            "pcc": 0.99,
            "args": {"shape": "(12, 64, 14)"},
        },
    ),
    (
        Reshape105,
        [((12, 14, 14), torch.float32)],
        {
            "model_names": ["pd_bert_bert_base_japanese_qa_padlenlp", "pt_albert_twmkn9_albert_base_v2_squad2_qa_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 12, 14, 14)"},
        },
    ),
    (
        Reshape106,
        [((1, 12, 14, 14), torch.float32)],
        {
            "model_names": ["pd_bert_bert_base_japanese_qa_padlenlp", "pt_albert_twmkn9_albert_base_v2_squad2_qa_hf"],
            "pcc": 0.99,
            "args": {"shape": "(12, 14, 14)"},
        },
    ),
    (
        Reshape107,
        [((12, 14, 64), torch.float32)],
        {
            "model_names": ["pd_bert_bert_base_japanese_qa_padlenlp", "pt_albert_twmkn9_albert_base_v2_squad2_qa_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 12, 14, 64)"},
        },
    ),
    (
        Reshape100,
        [((1, 14, 12, 64), torch.float32)],
        {"model_names": ["pd_bert_bert_base_japanese_qa_padlenlp"], "pcc": 0.99, "args": {"shape": "(14, 768)"}},
    ),
    (
        Reshape108,
        [((1, 14, 12, 64), torch.float32)],
        {
            "model_names": ["pt_albert_twmkn9_albert_base_v2_squad2_qa_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 14, 768, 1)"},
        },
    ),
    (
        Reshape109,
        [((1, 2048, 1, 1), torch.float32)],
        {
            "model_names": [
                "pd_resnet_50_img_cls_paddlemodels",
                "pt_resnext_resnext101_32x8d_img_cls_torchhub",
                "pt_xception_xception65_img_cls_timm",
                "pt_xception_xception71_tf_in1k_img_cls_timm",
                "pt_mobilenetv3_ssd_resnet101_img_cls_torchvision",
                "pt_resnet_resnet101_img_cls_torchvision",
                "pt_xception_xception_img_cls_timm",
                "pt_mobilenetv3_ssd_resnet152_img_cls_torchvision",
                "pt_resnet_50_img_cls_hf",
                "pt_resnet_resnet152_img_cls_torchvision",
                "pt_resnext_resnext50_32x4d_img_cls_torchhub",
                "pd_resnet_152_img_cls_paddlemodels",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_mobilenetv3_ssd_resnet50_img_cls_torchvision",
                "pt_resnet_resnet50_img_cls_torchvision",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_regnet_regnet_x_16gf_img_cls_torchvision",
                "pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub",
                "pt_wideresnet_wide_resnet50_2_img_cls_timm",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_xception_xception41_img_cls_timm",
                "pt_resnet_50_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 2048, 1, 1)"},
        },
    ),
    (
        Reshape110,
        [((1, 2048, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_hrnet_hrnetv2_w30_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w32_pose_estimation_osmr",
                "pt_resnext_resnext14_32x4d_img_cls_osmr",
                "jax_resnet_50_img_cls_hf",
                "pt_hrnet_hrnetv2_w44_pose_estimation_osmr",
                "pt_resnext_resnext50_32x4d_img_cls_osmr",
                "pt_hrnet_hrnetv2_w48_pose_estimation_osmr",
                "onnx_resnet_50_img_cls_hf",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "onnx_efficientnet_efficientnet_b5_img_cls_timm",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
                "pt_resnext_resnext101_64x4d_img_cls_osmr",
                "pt_hrnet_hrnetv2_w40_pose_estimation_osmr",
                "pt_resnext_resnext26_32x4d_img_cls_osmr",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 2048)"},
        },
    ),
    (
        Reshape111,
        [((1, 128, 768), torch.float32)],
        {
            "model_names": [
                "pt_albert_base_v1_mlm_hf",
                "pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf",
                "pt_albert_base_v1_token_cls_hf",
                "pt_albert_base_v2_mlm_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_roberta_xlm_roberta_base_mlm_hf",
                "onnx_bert_bert_base_uncased_mlm_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
                "pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder",
                "pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf",
                "pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf",
                "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "pt_albert_base_v2_token_cls_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder",
            ],
            "pcc": 0.99,
            "args": {"shape": "(128, 768)"},
        },
    ),
    (
        Reshape112,
        [((1, 128, 768), torch.float32)],
        {
            "model_names": [
                "pt_albert_base_v1_mlm_hf",
                "pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf",
                "pt_albert_base_v1_token_cls_hf",
                "pt_albert_base_v2_mlm_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_roberta_xlm_roberta_base_mlm_hf",
                "onnx_bert_bert_base_uncased_mlm_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
                "pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder",
                "pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf",
                "pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf",
                "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "pt_albert_base_v2_token_cls_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 128, 12, 64)"},
        },
    ),
    (
        Reshape113,
        [((128, 768), torch.float32)],
        {
            "model_names": [
                "pt_albert_base_v1_mlm_hf",
                "pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf",
                "pt_albert_base_v1_token_cls_hf",
                "pt_albert_base_v2_mlm_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_roberta_xlm_roberta_base_mlm_hf",
                "onnx_bert_bert_base_uncased_mlm_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
                "pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder",
                "pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf",
                "pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf",
                "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "pt_albert_base_v2_token_cls_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 128, 768)"},
        },
    ),
    (
        Reshape114,
        [((1, 12, 128, 64), torch.float32)],
        {
            "model_names": [
                "pt_albert_base_v1_mlm_hf",
                "pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf",
                "pt_albert_base_v1_token_cls_hf",
                "pt_albert_base_v2_mlm_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder",
                "pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_roberta_xlm_roberta_base_mlm_hf",
                "onnx_bert_bert_base_uncased_mlm_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
                "pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder",
                "pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf",
                "pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf",
                "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "pt_albert_base_v2_token_cls_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder",
            ],
            "pcc": 0.99,
            "args": {"shape": "(12, 128, 64)"},
        },
    ),
    (
        Reshape115,
        [((12, 128, 128), torch.float32)],
        {
            "model_names": [
                "pt_albert_base_v1_mlm_hf",
                "pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf",
                "pt_albert_base_v1_token_cls_hf",
                "pt_albert_base_v2_mlm_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder",
                "pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_roberta_xlm_roberta_base_mlm_hf",
                "onnx_bert_bert_base_uncased_mlm_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
                "pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder",
                "pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf",
                "pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf",
                "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "pt_albert_base_v2_token_cls_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 12, 128, 128)"},
        },
    ),
    (
        Reshape116,
        [((1, 12, 128, 128), torch.float32)],
        {
            "model_names": [
                "pt_albert_base_v1_mlm_hf",
                "pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf",
                "pt_albert_base_v1_token_cls_hf",
                "pt_albert_base_v2_mlm_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder",
                "pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_roberta_xlm_roberta_base_mlm_hf",
                "onnx_bert_bert_base_uncased_mlm_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
                "pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder",
                "pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf",
                "pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf",
                "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "pt_albert_base_v2_token_cls_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder",
            ],
            "pcc": 0.99,
            "args": {"shape": "(12, 128, 128)"},
        },
    ),
    (
        Reshape117,
        [((1, 12, 64, 128), torch.float32)],
        {
            "model_names": [
                "pt_albert_base_v1_mlm_hf",
                "pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf",
                "pt_albert_base_v1_token_cls_hf",
                "pt_albert_base_v2_mlm_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_roberta_xlm_roberta_base_mlm_hf",
                "onnx_bert_bert_base_uncased_mlm_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
                "pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder",
                "pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf",
                "pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf",
                "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "pt_albert_base_v2_token_cls_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder",
            ],
            "pcc": 0.99,
            "args": {"shape": "(12, 64, 128)"},
        },
    ),
    (
        Reshape118,
        [((1, 12, 64, 128), torch.float32)],
        {
            "model_names": ["pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 768, 128, 1)"},
        },
    ),
    (
        Reshape119,
        [((12, 128, 64), torch.float32)],
        {
            "model_names": [
                "pt_albert_base_v1_mlm_hf",
                "pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf",
                "pt_albert_base_v1_token_cls_hf",
                "pt_albert_base_v2_mlm_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder",
                "pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_roberta_xlm_roberta_base_mlm_hf",
                "onnx_bert_bert_base_uncased_mlm_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
                "pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder",
                "pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf",
                "pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf",
                "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "pt_albert_base_v2_token_cls_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 12, 128, 64)"},
        },
    ),
    (
        Reshape120,
        [((1, 128, 12, 64), torch.float32)],
        {
            "model_names": [
                "pt_albert_base_v1_mlm_hf",
                "pt_albert_base_v1_token_cls_hf",
                "pt_albert_base_v2_mlm_hf",
                "pt_albert_base_v2_token_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 128, 768, 1)"},
        },
    ),
    (
        Reshape111,
        [((1, 128, 12, 64), torch.float32)],
        {
            "model_names": [
                "pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_roberta_xlm_roberta_base_mlm_hf",
                "onnx_bert_bert_base_uncased_mlm_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
                "pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder",
                "pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf",
                "pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf",
                "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder",
            ],
            "pcc": 0.99,
            "args": {"shape": "(128, 768)"},
        },
    ),
    (
        Reshape121,
        [((1, 9, 768), torch.float32)],
        {
            "model_names": [
                "pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf",
                "pd_ernie_1_0_seq_cls_padlenlp",
                "pd_ernie_1_0_qa_padlenlp",
                "pd_bert_chinese_roberta_base_mlm_padlenlp",
                "pd_bert_bert_base_uncased_qa_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"shape": "(9, 768)"},
        },
    ),
    (
        Reshape122,
        [((1, 9, 768), torch.float32)],
        {
            "model_names": [
                "pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf",
                "pd_ernie_1_0_seq_cls_padlenlp",
                "pd_ernie_1_0_qa_padlenlp",
                "pd_bert_chinese_roberta_base_mlm_padlenlp",
                "pd_bert_bert_base_uncased_qa_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 9, 12, 64)"},
        },
    ),
    (
        Reshape123,
        [((9, 768), torch.float32)],
        {
            "model_names": [
                "pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf",
                "pd_ernie_1_0_seq_cls_padlenlp",
                "pd_ernie_1_0_qa_padlenlp",
                "pd_bert_chinese_roberta_base_mlm_padlenlp",
                "pd_bert_bert_base_uncased_qa_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 9, 768)"},
        },
    ),
    (
        Reshape124,
        [((1, 12, 9, 64), torch.float32)],
        {
            "model_names": [
                "pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf",
                "pd_ernie_1_0_seq_cls_padlenlp",
                "pd_ernie_1_0_qa_padlenlp",
                "pd_bert_chinese_roberta_base_mlm_padlenlp",
                "pd_bert_bert_base_uncased_qa_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"shape": "(12, 9, 64)"},
        },
    ),
    (
        Reshape125,
        [((12, 9, 9), torch.float32)],
        {
            "model_names": [
                "pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf",
                "pd_ernie_1_0_seq_cls_padlenlp",
                "pd_ernie_1_0_qa_padlenlp",
                "pd_bert_chinese_roberta_base_mlm_padlenlp",
                "pd_bert_bert_base_uncased_qa_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 12, 9, 9)"},
        },
    ),
    (
        Reshape126,
        [((1, 12, 9, 9), torch.float32)],
        {
            "model_names": [
                "pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf",
                "pd_ernie_1_0_seq_cls_padlenlp",
                "pd_ernie_1_0_qa_padlenlp",
                "pd_bert_chinese_roberta_base_mlm_padlenlp",
                "pd_bert_bert_base_uncased_qa_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"shape": "(12, 9, 9)"},
        },
    ),
    (
        Reshape127,
        [((1, 12, 64, 9), torch.float32)],
        {
            "model_names": [
                "pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf",
                "pd_ernie_1_0_seq_cls_padlenlp",
                "pd_ernie_1_0_qa_padlenlp",
                "pd_bert_chinese_roberta_base_mlm_padlenlp",
                "pd_bert_bert_base_uncased_qa_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"shape": "(12, 64, 9)"},
        },
    ),
    (
        Reshape128,
        [((12, 9, 64), torch.float32)],
        {
            "model_names": [
                "pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf",
                "pd_ernie_1_0_seq_cls_padlenlp",
                "pd_ernie_1_0_qa_padlenlp",
                "pd_bert_chinese_roberta_base_mlm_padlenlp",
                "pd_bert_bert_base_uncased_qa_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 12, 9, 64)"},
        },
    ),
    (
        Reshape129,
        [((1, 9, 12, 64), torch.float32)],
        {
            "model_names": ["pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 9, 768, 1)"},
        },
    ),
    (
        Reshape121,
        [((1, 9, 12, 64), torch.float32)],
        {
            "model_names": [
                "pd_ernie_1_0_seq_cls_padlenlp",
                "pd_ernie_1_0_qa_padlenlp",
                "pd_bert_chinese_roberta_base_mlm_padlenlp",
                "pd_bert_bert_base_uncased_qa_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"shape": "(9, 768)"},
        },
    ),
    (
        Reshape130,
        [((1, 1, 768), torch.float32)],
        {
            "model_names": [
                "pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf",
                "pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf",
                "pt_deit_facebook_deit_base_distilled_patch16_224_img_cls_hf",
                "pt_deit_facebook_deit_base_patch16_224_img_cls_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder",
                "pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf",
                "pt_t5_t5_base_text_gen_hf",
                "pt_vit_vit_b_16_img_cls_torchvision",
                "pt_whisper_openai_whisper_small_speech_recognition_hf",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder",
                "pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf",
                "onnx_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf",
                "pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf",
                "onnx_vit_base_google_vit_base_patch16_224_img_cls_hf",
                "pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf",
                "pt_t5_google_flan_t5_base_text_gen_hf",
                "pt_vit_google_vit_base_patch16_224_img_cls_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder",
                "pt_vit_vit_b_32_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 768)"},
        },
    ),
    (
        Reshape131,
        [((1, 1, 768), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 1, 12, 64)"},
        },
    ),
    (
        Reshape132,
        [((1, 256, 6, 6), torch.float32)],
        {"model_names": ["pt_alexnet_base_img_cls_osmr"], "pcc": 0.99, "args": {"shape": "(1, 9216)"}},
    ),
    (
        Reshape133,
        [((1, 256, 6, 6), torch.float32)],
        {
            "model_names": [
                "pt_alexnet_base_img_cls_torchhub",
                "pt_rcnn_base_obj_det_torchvision_rect_0",
                "pd_alexnet_base_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 9216, 1, 1)"},
        },
    ),
    (
        Reshape134,
        [((1, 768, 14, 14), torch.float32)],
        {
            "model_names": [
                "pt_deit_facebook_deit_base_distilled_patch16_224_img_cls_hf",
                "pt_deit_facebook_deit_base_patch16_224_img_cls_hf",
                "pt_mlp_mixer_mixer_b16_224_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_in21k_img_cls_timm",
                "pt_vit_google_vit_base_patch16_224_img_cls_hf",
                "pt_beit_microsoft_beit_base_patch16_224_img_cls_hf",
                "pt_mlp_mixer_mixer_b16_224_goog_in21k_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 768, 196, 1)"},
        },
    ),
    (
        Reshape135,
        [((1, 768, 14, 14), torch.float32)],
        {
            "model_names": [
                "pt_vit_vit_b_16_img_cls_torchvision",
                "onnx_vit_base_google_vit_base_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 768, 196)"},
        },
    ),
    (
        Reshape136,
        [((1, 197, 768), torch.float32)],
        {
            "model_names": [
                "pt_deit_facebook_deit_base_distilled_patch16_224_img_cls_hf",
                "pt_deit_facebook_deit_base_patch16_224_img_cls_hf",
                "onnx_vit_base_google_vit_base_patch16_224_img_cls_hf",
                "pt_vit_google_vit_base_patch16_224_img_cls_hf",
                "pt_beit_microsoft_beit_base_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(197, 768)"},
        },
    ),
    (
        Reshape137,
        [((1, 197, 768), torch.float32)],
        {
            "model_names": [
                "pt_deit_facebook_deit_base_distilled_patch16_224_img_cls_hf",
                "pt_deit_facebook_deit_base_patch16_224_img_cls_hf",
                "onnx_vit_base_google_vit_base_patch16_224_img_cls_hf",
                "pt_vit_google_vit_base_patch16_224_img_cls_hf",
                "pt_beit_microsoft_beit_base_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 197, 12, 64)"},
        },
    ),
    (
        Reshape138,
        [((197, 768), torch.float32)],
        {
            "model_names": [
                "pt_deit_facebook_deit_base_distilled_patch16_224_img_cls_hf",
                "pt_deit_facebook_deit_base_patch16_224_img_cls_hf",
                "onnx_vit_base_google_vit_base_patch16_224_img_cls_hf",
                "pt_vit_google_vit_base_patch16_224_img_cls_hf",
                "pt_beit_microsoft_beit_base_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 197, 768)"},
        },
    ),
    (
        Reshape139,
        [((197, 768), torch.float32)],
        {"model_names": ["pt_vit_vit_b_16_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(197, 1, 768)"}},
    ),
    (
        Reshape137,
        [((197, 768), torch.float32)],
        {
            "model_names": ["pt_beit_microsoft_beit_base_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 197, 12, 64)"},
        },
    ),
    (
        Reshape140,
        [((1, 12, 197, 64), torch.float32)],
        {
            "model_names": [
                "pt_deit_facebook_deit_base_distilled_patch16_224_img_cls_hf",
                "pt_deit_facebook_deit_base_patch16_224_img_cls_hf",
                "onnx_vit_base_google_vit_base_patch16_224_img_cls_hf",
                "pt_vit_google_vit_base_patch16_224_img_cls_hf",
                "pt_beit_microsoft_beit_base_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(12, 197, 64)"},
        },
    ),
    (
        Reshape141,
        [((12, 197, 197), torch.float32)],
        {
            "model_names": [
                "pt_deit_facebook_deit_base_distilled_patch16_224_img_cls_hf",
                "pt_deit_facebook_deit_base_patch16_224_img_cls_hf",
                "pt_vit_vit_b_16_img_cls_torchvision",
                "onnx_vit_base_google_vit_base_patch16_224_img_cls_hf",
                "pt_vit_google_vit_base_patch16_224_img_cls_hf",
                "pt_beit_microsoft_beit_base_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 12, 197, 197)"},
        },
    ),
    (
        Reshape142,
        [((1, 12, 197, 197), torch.float32)],
        {
            "model_names": [
                "pt_deit_facebook_deit_base_distilled_patch16_224_img_cls_hf",
                "pt_deit_facebook_deit_base_patch16_224_img_cls_hf",
                "pt_vit_vit_b_16_img_cls_torchvision",
                "onnx_vit_base_google_vit_base_patch16_224_img_cls_hf",
                "pt_vit_google_vit_base_patch16_224_img_cls_hf",
                "pt_beit_microsoft_beit_base_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(12, 197, 197)"},
        },
    ),
    (
        Reshape143,
        [((1, 12, 64, 197), torch.float32)],
        {
            "model_names": [
                "pt_deit_facebook_deit_base_distilled_patch16_224_img_cls_hf",
                "pt_deit_facebook_deit_base_patch16_224_img_cls_hf",
                "onnx_vit_base_google_vit_base_patch16_224_img_cls_hf",
                "pt_vit_google_vit_base_patch16_224_img_cls_hf",
                "pt_beit_microsoft_beit_base_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(12, 64, 197)"},
        },
    ),
    (
        Reshape144,
        [((12, 197, 64), torch.float32)],
        {
            "model_names": [
                "pt_deit_facebook_deit_base_distilled_patch16_224_img_cls_hf",
                "pt_deit_facebook_deit_base_patch16_224_img_cls_hf",
                "pt_vit_vit_b_16_img_cls_torchvision",
                "onnx_vit_base_google_vit_base_patch16_224_img_cls_hf",
                "pt_vit_google_vit_base_patch16_224_img_cls_hf",
                "pt_beit_microsoft_beit_base_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 12, 197, 64)"},
        },
    ),
    (
        Reshape140,
        [((12, 197, 64), torch.float32)],
        {"model_names": ["pt_vit_vit_b_16_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(12, 197, 64)"}},
    ),
    (
        Reshape136,
        [((1, 197, 12, 64), torch.float32)],
        {
            "model_names": [
                "pt_deit_facebook_deit_base_distilled_patch16_224_img_cls_hf",
                "pt_deit_facebook_deit_base_patch16_224_img_cls_hf",
                "onnx_vit_base_google_vit_base_patch16_224_img_cls_hf",
                "pt_vit_google_vit_base_patch16_224_img_cls_hf",
                "pt_beit_microsoft_beit_base_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(197, 768)"},
        },
    ),
    (
        Reshape145,
        [((1, 100, 256), torch.float32)],
        {
            "model_names": [
                "pt_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
                "pt_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(100, 256)"},
        },
    ),
    (
        Reshape146,
        [((1, 100, 256), torch.float32)],
        {
            "model_names": [
                "pt_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
                "pt_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 100, 8, 32)"},
        },
    ),
    (
        Reshape147,
        [((100, 256), torch.float32)],
        {
            "model_names": [
                "pt_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
                "pt_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 100, 256)"},
        },
    ),
    (
        Reshape148,
        [((1, 8, 100, 32), torch.float32)],
        {
            "model_names": [
                "pt_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
                "pt_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(8, 100, 32)"},
        },
    ),
    (
        Reshape149,
        [((8, 100, 32), torch.float32)],
        {
            "model_names": [
                "pt_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
                "pt_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 8, 100, 32)"},
        },
    ),
    (
        Reshape145,
        [((1, 100, 8, 32), torch.float32)],
        {
            "model_names": [
                "pt_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
                "pt_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(100, 256)"},
        },
    ),
    (
        Reshape150,
        [((64,), torch.float32)],
        {
            "model_names": [
                "pt_detr_facebook_detr_resnet_50_obj_det_hf",
                "pt_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 64, 1, 1)"},
        },
    ),
    (
        Reshape151,
        [((64,), torch.float32)],
        {"model_names": ["tf_resnet_resnet50_img_cls_keras"], "pcc": 0.99, "args": {"shape": "(64,)"}},
    ),
    (
        Reshape152,
        [((64,), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99, "args": {"shape": "(1, 1, 1, 64)"}},
    ),
    (
        Reshape153,
        [((256,), torch.float32)],
        {
            "model_names": [
                "pt_detr_facebook_detr_resnet_50_obj_det_hf",
                "pt_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 256, 1, 1)"},
        },
    ),
    (
        Reshape154,
        [((256,), torch.float32)],
        {"model_names": ["tf_resnet_resnet50_img_cls_keras"], "pcc": 0.99, "args": {"shape": "(256,)"}},
    ),
    (
        Reshape155,
        [((256,), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99, "args": {"shape": "(1, 1, 1, 256)"}},
    ),
    (
        Reshape156,
        [((128,), torch.float32)],
        {
            "model_names": [
                "pt_detr_facebook_detr_resnet_50_obj_det_hf",
                "pt_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 128, 1, 1)"},
        },
    ),
    (
        Reshape157,
        [((128,), torch.float32)],
        {"model_names": ["tf_resnet_resnet50_img_cls_keras"], "pcc": 0.99, "args": {"shape": "(128,)"}},
    ),
    (
        Reshape158,
        [((128,), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99, "args": {"shape": "(1, 1, 1, 128)"}},
    ),
    (
        Reshape159,
        [((512,), torch.float32)],
        {
            "model_names": [
                "pt_detr_facebook_detr_resnet_50_obj_det_hf",
                "pt_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 512, 1, 1)"},
        },
    ),
    (
        Reshape160,
        [((512,), torch.float32)],
        {"model_names": ["tf_resnet_resnet50_img_cls_keras"], "pcc": 0.99, "args": {"shape": "(512,)"}},
    ),
    (
        Reshape161,
        [((512,), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99, "args": {"shape": "(1, 1, 1, 512)"}},
    ),
    (
        Reshape99,
        [((1024,), torch.float32)],
        {
            "model_names": [
                "pt_detr_facebook_detr_resnet_50_obj_det_hf",
                "pt_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 1024, 1, 1)"},
        },
    ),
    (
        Reshape162,
        [((1024,), torch.float32)],
        {"model_names": ["tf_resnet_resnet50_img_cls_keras"], "pcc": 0.99, "args": {"shape": "(1024,)"}},
    ),
    (
        Reshape163,
        [((1024,), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99, "args": {"shape": "(1, 1, 1, 1024)"}},
    ),
    (
        Reshape109,
        [((2048,), torch.float32)],
        {
            "model_names": [
                "pt_detr_facebook_detr_resnet_50_obj_det_hf",
                "pt_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 2048, 1, 1)"},
        },
    ),
    (
        Reshape164,
        [((2048,), torch.float32)],
        {"model_names": ["tf_resnet_resnet50_img_cls_keras"], "pcc": 0.99, "args": {"shape": "(2048,)"}},
    ),
    (
        Reshape165,
        [((2048,), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99, "args": {"shape": "(1, 1, 1, 2048)"}},
    ),
    (
        Reshape166,
        [((1, 256, 14, 20), torch.float32)],
        {
            "model_names": [
                "pt_detr_facebook_detr_resnet_50_obj_det_hf",
                "pt_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 256, 280, 1)"},
        },
    ),
    (
        Reshape167,
        [((1, 256, 14, 20), torch.float32)],
        {
            "model_names": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 256, 280)"},
        },
    ),
    (
        Reshape168,
        [((1, 256, 14, 20), torch.float32)],
        {
            "model_names": [
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
                "pt_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 8, 32, 280)"},
        },
    ),
    (
        Reshape169,
        [((1, 280, 256), torch.float32)],
        {
            "model_names": [
                "pt_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
                "pt_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(280, 256)"},
        },
    ),
    (
        Reshape170,
        [((1, 280, 256), torch.float32)],
        {
            "model_names": [
                "pt_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
                "pt_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 280, 8, 32)"},
        },
    ),
    (
        Reshape171,
        [((280, 256), torch.float32)],
        {
            "model_names": [
                "pt_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
                "pt_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 280, 256)"},
        },
    ),
    (
        Reshape172,
        [((1, 8, 280, 32), torch.float32)],
        {
            "model_names": [
                "pt_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
                "pt_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(8, 280, 32)"},
        },
    ),
    (
        Reshape173,
        [((8, 280, 280), torch.float32)],
        {
            "model_names": [
                "pt_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
                "pt_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 8, 280, 280)"},
        },
    ),
    (
        Reshape174,
        [((1, 8, 280, 280), torch.float32)],
        {
            "model_names": [
                "pt_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
                "pt_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(8, 280, 280)"},
        },
    ),
    (
        Reshape175,
        [((8, 280, 32), torch.float32)],
        {
            "model_names": [
                "pt_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
                "pt_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 8, 280, 32)"},
        },
    ),
    (
        Reshape169,
        [((1, 280, 8, 32), torch.float32)],
        {
            "model_names": [
                "pt_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
                "pt_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(280, 256)"},
        },
    ),
    (
        Reshape176,
        [((8, 100, 280), torch.float32)],
        {
            "model_names": [
                "pt_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
                "pt_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 8, 100, 280)"},
        },
    ),
    (
        Reshape177,
        [((1, 8, 100, 280), torch.float32)],
        {
            "model_names": [
                "pt_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
                "pt_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(8, 100, 280)"},
        },
    ),
    (
        Reshape178,
        [((32, 1, 3, 3), torch.float32)],
        {
            "model_names": [
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm",
                "pt_mobilenetv1_basic_img_cls_torchvision",
                "pt_yolox_yolox_nano_obj_det_torchhub",
                "pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"shape": "(32, 1, 3, 3)"},
        },
    ),
    (
        Reshape179,
        [((144, 1, 3, 3), torch.float32)],
        {
            "model_names": [
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"shape": "(144, 1, 3, 3)"},
        },
    ),
    (
        Reshape180,
        [((192, 1, 3, 3), torch.float32)],
        {
            "model_names": [
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "args": {"shape": "(192, 1, 3, 3)"},
        },
    ),
    (
        Reshape181,
        [((192, 1, 5, 5), torch.float32)],
        {
            "model_names": [
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"shape": "(192, 1, 5, 5)"},
        },
    ),
    (
        Reshape182,
        [((288, 1, 5, 5), torch.float32)],
        {
            "model_names": [
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
            ],
            "pcc": 0.99,
            "args": {"shape": "(288, 1, 5, 5)"},
        },
    ),
    (
        Reshape183,
        [((288, 1, 3, 3), torch.float32)],
        {
            "model_names": [
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(288, 1, 3, 3)"},
        },
    ),
    (
        Reshape184,
        [((576, 1, 3, 3), torch.float32)],
        {
            "model_names": [
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"shape": "(576, 1, 3, 3)"},
        },
    ),
    (
        Reshape185,
        [((576, 1, 5, 5), torch.float32)],
        {
            "model_names": [
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
            ],
            "pcc": 0.99,
            "args": {"shape": "(576, 1, 5, 5)"},
        },
    ),
    (
        Reshape186,
        [((816, 1, 5, 5), torch.float32)],
        {
            "model_names": ["pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm"],
            "pcc": 0.99,
            "args": {"shape": "(816, 1, 5, 5)"},
        },
    ),
    (
        Reshape187,
        [((1392, 1, 5, 5), torch.float32)],
        {
            "model_names": ["pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm"],
            "pcc": 0.99,
            "args": {"shape": "(1392, 1, 5, 5)"},
        },
    ),
    (
        Reshape188,
        [((1392, 1, 3, 3), torch.float32)],
        {
            "model_names": ["pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm"],
            "pcc": 0.99,
            "args": {"shape": "(1392, 1, 3, 3)"},
        },
    ),
    (
        Reshape189,
        [((1, 207, 2304), torch.float32)],
        {"model_names": ["pt_gemma_google_gemma_2_2b_it_qa_hf"], "pcc": 0.99, "args": {"shape": "(207, 2304)"}},
    ),
    (
        Reshape190,
        [((207, 2048), torch.float32)],
        {"model_names": ["pt_gemma_google_gemma_2_2b_it_qa_hf"], "pcc": 0.99, "args": {"shape": "(1, 207, 8, 256)"}},
    ),
    (
        Reshape191,
        [((1, 8, 207, 256), torch.float32)],
        {"model_names": ["pt_gemma_google_gemma_2_2b_it_qa_hf"], "pcc": 0.99, "args": {"shape": "(8, 207, 256)"}},
    ),
    (
        Reshape192,
        [((207, 1024), torch.float32)],
        {"model_names": ["pt_gemma_google_gemma_2_2b_it_qa_hf"], "pcc": 0.99, "args": {"shape": "(1, 207, 4, 256)"}},
    ),
    (
        Reshape191,
        [((1, 4, 2, 207, 256), torch.float32)],
        {"model_names": ["pt_gemma_google_gemma_2_2b_it_qa_hf"], "pcc": 0.99, "args": {"shape": "(8, 207, 256)"}},
    ),
    (
        Reshape193,
        [((1, 4, 2, 207, 256), torch.float32)],
        {"model_names": ["pt_gemma_google_gemma_2_2b_it_qa_hf"], "pcc": 0.99, "args": {"shape": "(1, 8, 207, 256)"}},
    ),
    (
        Reshape194,
        [((8, 207, 207), torch.float32)],
        {"model_names": ["pt_gemma_google_gemma_2_2b_it_qa_hf"], "pcc": 0.99, "args": {"shape": "(1, 8, 207, 207)"}},
    ),
    (
        Reshape195,
        [((1, 8, 207, 207), torch.float32)],
        {"model_names": ["pt_gemma_google_gemma_2_2b_it_qa_hf"], "pcc": 0.99, "args": {"shape": "(8, 207, 207)"}},
    ),
    (
        Reshape196,
        [((1, 8, 256, 207), torch.float32)],
        {"model_names": ["pt_gemma_google_gemma_2_2b_it_qa_hf"], "pcc": 0.99, "args": {"shape": "(8, 256, 207)"}},
    ),
    (
        Reshape193,
        [((8, 207, 256), torch.float32)],
        {"model_names": ["pt_gemma_google_gemma_2_2b_it_qa_hf"], "pcc": 0.99, "args": {"shape": "(1, 8, 207, 256)"}},
    ),
    (
        Reshape197,
        [((1, 207, 8, 256), torch.float32)],
        {"model_names": ["pt_gemma_google_gemma_2_2b_it_qa_hf"], "pcc": 0.99, "args": {"shape": "(207, 2048)"}},
    ),
    (
        Reshape198,
        [((207, 2304), torch.float32)],
        {"model_names": ["pt_gemma_google_gemma_2_2b_it_qa_hf"], "pcc": 0.99, "args": {"shape": "(1, 207, 2304)"}},
    ),
    (
        Reshape199,
        [((207, 9216), torch.float32)],
        {"model_names": ["pt_gemma_google_gemma_2_2b_it_qa_hf"], "pcc": 0.99, "args": {"shape": "(1, 207, 9216)"}},
    ),
    (
        Reshape200,
        [((1, 1, 224, 224), torch.float32)],
        {"model_names": ["pt_googlenet_base_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(1, 224, 224)"}},
    ),
    (
        Reshape201,
        [((1, 1536, 1, 1), torch.float32)],
        {
            "model_names": ["pt_inception_v4_img_cls_osmr", "onnx_efficientnet_efficientnet_b3a_img_cls_timm"],
            "pcc": 0.99,
            "args": {"shape": "(1, 1536)"},
        },
    ),
    (
        Reshape202,
        [((1, 1536, 1, 1), torch.float32)],
        {
            "model_names": ["pt_inception_inception_v4_tf_in1k_img_cls_timm", "pt_inception_inception_v4_img_cls_timm"],
            "pcc": 0.99,
            "args": {"shape": "(1, 1536, 1, 1)"},
        },
    ),
    (
        Reshape203,
        [((1, 1024, 7, 7), torch.float32)],
        {
            "model_names": ["pt_mlp_mixer_mixer_l32_224_img_cls_timm"],
            "pcc": 0.99,
            "args": {"shape": "(1, 1024, 49, 1)"},
        },
    ),
    (
        Reshape204,
        [((96, 1, 3, 3), torch.float32)],
        {
            "model_names": [
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"shape": "(96, 1, 3, 3)"},
        },
    ),
    (
        Reshape205,
        [((384, 1, 3, 3), torch.float32)],
        {
            "model_names": [
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"shape": "(384, 1, 3, 3)"},
        },
    ),
    (
        Reshape206,
        [((960, 1, 3, 3), torch.float32)],
        {
            "model_names": [
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"shape": "(960, 1, 3, 3)"},
        },
    ),
    (
        Reshape207,
        [((1, 7), torch.int64)],
        {
            "model_names": [
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
                "pt_gpt_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 7)"},
        },
    ),
    (
        Reshape208,
        [((1, 7, 768), torch.float32)],
        {
            "model_names": [
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
                "pt_gpt_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(7, 768)"},
        },
    ),
    (
        Reshape209,
        [((1, 7, 768), torch.float32)],
        {
            "model_names": [
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
                "pt_gpt_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 7, 12, 64)"},
        },
    ),
    (
        Reshape210,
        [((1, 7, 768), torch.float32)],
        {
            "model_names": ["pt_nanogpt_financialsupport_nanogpt_text_gen_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 7, 768)"},
        },
    ),
    (
        Reshape210,
        [((7, 768), torch.float32)],
        {
            "model_names": [
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
                "pt_gpt_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 7, 768)"},
        },
    ),
    (
        Reshape211,
        [((1, 12, 7, 64), torch.float32)],
        {
            "model_names": [
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
                "pt_gpt_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(12, 7, 64)"},
        },
    ),
    (
        Reshape212,
        [((12, 7, 7), torch.float32)],
        {
            "model_names": [
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
                "pt_gpt_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 12, 7, 7)"},
        },
    ),
    (
        Reshape213,
        [((1, 12, 7, 7), torch.float32)],
        {
            "model_names": [
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
                "pt_gpt_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(12, 7, 7)"},
        },
    ),
    (
        Reshape214,
        [((1, 12, 64, 7), torch.float32)],
        {
            "model_names": [
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
                "pt_gpt_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(12, 64, 7)"},
        },
    ),
    (
        Reshape215,
        [((12, 7, 64), torch.float32)],
        {
            "model_names": [
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
                "pt_gpt_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 12, 7, 64)"},
        },
    ),
    (
        Reshape208,
        [((1, 7, 12, 64), torch.float32)],
        {
            "model_names": [
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
                "pt_gpt_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(7, 768)"},
        },
    ),
    (
        Reshape216,
        [((7, 3072), torch.float32)],
        {
            "model_names": [
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
                "pt_gpt_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 7, 3072)"},
        },
    ),
    (
        Reshape217,
        [((1, 7, 3072), torch.float32)],
        {
            "model_names": [
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
                "pt_gpt_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(7, 3072)"},
        },
    ),
    (
        Reshape218,
        [((1, 32), torch.int64)],
        {
            "model_names": [
                "pt_opt_facebook_opt_350m_qa_hf",
                "pt_opt_facebook_opt_1_3b_seq_cls_hf",
                "pt_opt_facebook_opt_350m_seq_cls_hf",
                "pt_opt_facebook_opt_1_3b_qa_hf",
                "pt_opt_facebook_opt_125m_qa_hf",
                "pt_opt_facebook_opt_125m_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 32)"},
        },
    ),
    (
        Reshape219,
        [((1, 32, 1024), torch.float32)],
        {
            "model_names": ["pt_opt_facebook_opt_350m_qa_hf", "pt_opt_facebook_opt_350m_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(32, 1024)"},
        },
    ),
    (
        Reshape220,
        [((1, 32, 1024), torch.float32)],
        {
            "model_names": ["pt_opt_facebook_opt_350m_qa_hf", "pt_opt_facebook_opt_350m_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 32, 16, 64)"},
        },
    ),
    (
        Reshape221,
        [((32, 1024), torch.float32)],
        {
            "model_names": ["pt_opt_facebook_opt_350m_qa_hf", "pt_opt_facebook_opt_350m_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 32, 1024)"},
        },
    ),
    (
        Reshape222,
        [((1, 16, 32, 64), torch.float32)],
        {
            "model_names": ["pt_opt_facebook_opt_350m_qa_hf", "pt_opt_facebook_opt_350m_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(16, 32, 64)"},
        },
    ),
    (
        Reshape223,
        [((16, 32, 32), torch.float32)],
        {
            "model_names": [
                "pt_opt_facebook_opt_350m_qa_hf",
                "pt_opt_facebook_opt_350m_seq_cls_hf",
                "pt_bloom_bigscience_bloom_1b1_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 16, 32, 32)"},
        },
    ),
    (
        Reshape224,
        [((1, 16, 32, 32), torch.float32)],
        {
            "model_names": [
                "pt_opt_facebook_opt_350m_qa_hf",
                "pt_opt_facebook_opt_350m_seq_cls_hf",
                "pt_bloom_bigscience_bloom_1b1_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(16, 32, 32)"},
        },
    ),
    (
        Reshape225,
        [((16, 32, 64), torch.float32)],
        {
            "model_names": ["pt_opt_facebook_opt_350m_qa_hf", "pt_opt_facebook_opt_350m_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 16, 32, 64)"},
        },
    ),
    (
        Reshape219,
        [((1, 32, 16, 64), torch.float32)],
        {
            "model_names": ["pt_opt_facebook_opt_350m_qa_hf", "pt_opt_facebook_opt_350m_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(32, 1024)"},
        },
    ),
    (
        Reshape226,
        [((32, 512), torch.float32)],
        {
            "model_names": ["pt_opt_facebook_opt_350m_qa_hf", "pt_opt_facebook_opt_350m_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(32, 512)"},
        },
    ),
    (
        Reshape227,
        [((32, 1), torch.float32)],
        {
            "model_names": [
                "pt_opt_facebook_opt_350m_qa_hf",
                "pt_opt_facebook_opt_1_3b_qa_hf",
                "pt_opt_facebook_opt_125m_qa_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 32, 1)"},
        },
    ),
    (
        Reshape163,
        [((1, 1, 1024), torch.float32)],
        {
            "model_names": [
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 1, 1, 1024)"},
        },
    ),
    (
        Reshape98,
        [((1, 1, 1024), torch.float32)],
        {
            "model_names": [
                "pt_vit_google_vit_large_patch16_224_img_cls_hf",
                "pt_whisper_openai_whisper_medium_speech_recognition_hf",
                "pt_vit_vit_l_16_img_cls_torchvision",
                "pt_t5_google_flan_t5_large_text_gen_hf",
                "pt_t5_t5_large_text_gen_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 1024)"},
        },
    ),
    (
        Reshape228,
        [((1, 1, 1024), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_medium_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 1, 16, 64)"},
        },
    ),
    (
        Reshape229,
        [((1, 512, 322), torch.float32)],
        {
            "model_names": ["pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 512, 1, 322)"},
        },
    ),
    (
        Reshape230,
        [((1, 55, 55, 64), torch.float32)],
        {
            "model_names": ["pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 3025, 64)"},
        },
    ),
    (
        Reshape231,
        [((1, 3025, 322), torch.float32)],
        {
            "model_names": ["pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(3025, 322)"},
        },
    ),
    (
        Reshape232,
        [((1, 3025, 322), torch.float32)],
        {
            "model_names": ["pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 3025, 1, 322)"},
        },
    ),
    (
        Reshape233,
        [((3025, 322), torch.float32)],
        {
            "model_names": ["pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 3025, 322)"},
        },
    ),
    (
        Reshape234,
        [((1, 512, 3025), torch.float32)],
        {
            "model_names": ["pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 1, 512, 3025)"},
        },
    ),
    (
        Reshape235,
        [((1, 1, 512, 3025), torch.float32)],
        {
            "model_names": ["pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 512, 3025)"},
        },
    ),
    (
        Reshape236,
        [((1, 1, 322, 3025), torch.float32)],
        {
            "model_names": ["pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 322, 3025)"},
        },
    ),
    (
        Reshape237,
        [((1, 512, 1024), torch.float32)],
        {
            "model_names": [
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(512, 1024)"},
        },
    ),
    (
        Reshape238,
        [((1, 512, 1024), torch.float32)],
        {
            "model_names": [
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 512, 8, 128)"},
        },
    ),
    (
        Reshape239,
        [((1, 512, 1024), torch.float32)],
        {
            "model_names": [
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 512, 1, 1024)"},
        },
    ),
    (
        Reshape240,
        [((512, 1024), torch.float32)],
        {
            "model_names": [
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 512, 1024)"},
        },
    ),
    (
        Reshape241,
        [((1, 8, 512, 128), torch.float32)],
        {
            "model_names": [
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(8, 512, 128)"},
        },
    ),
    (
        Reshape242,
        [((8, 512, 512), torch.float32)],
        {
            "model_names": [
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 8, 512, 512)"},
        },
    ),
    (
        Reshape243,
        [((1, 8, 512, 512), torch.float32)],
        {
            "model_names": [
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(8, 512, 512)"},
        },
    ),
    (
        Reshape244,
        [((1, 8, 128, 512), torch.float32)],
        {
            "model_names": [
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(8, 128, 512)"},
        },
    ),
    (
        Reshape245,
        [((8, 512, 128), torch.float32)],
        {
            "model_names": [
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 8, 512, 128)"},
        },
    ),
    (
        Reshape237,
        [((1, 512, 8, 128), torch.float32)],
        {
            "model_names": [
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(512, 1024)"},
        },
    ),
    (
        Reshape161,
        [((1, 1, 512), torch.float32)],
        {
            "model_names": [
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 1, 1, 512)"},
        },
    ),
    (
        Reshape246,
        [((1, 1, 512), torch.float32)],
        {
            "model_names": [
                "pt_t5_google_flan_t5_small_text_gen_hf",
                "pt_t5_t5_small_text_gen_hf",
                "pt_whisper_openai_whisper_base_speech_recognition_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 512)"},
        },
    ),
    (
        Reshape247,
        [((1, 1, 512), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_base_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 1, 8, 64)"},
        },
    ),
    (
        Reshape248,
        [((1, 1, 1, 512), torch.float32)],
        {
            "model_names": [
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 1, 512)"},
        },
    ),
    (
        Reshape249,
        [((1, 1, 1024, 512), torch.float32)],
        {
            "model_names": [
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 1024, 512)"},
        },
    ),
    (
        Reshape0,
        [((1, 1, 1000), torch.float32)],
        {
            "model_names": [
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 1000)"},
        },
    ),
    (
        Reshape250,
        [((1, 12, 2048), torch.float32)],
        {
            "model_names": ["pt_phi1_microsoft_phi_1_token_cls_hf", "pt_phi_1_5_microsoft_phi_1_5_token_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(12, 2048)"},
        },
    ),
    (
        Reshape251,
        [((1, 12, 2048), torch.float32)],
        {
            "model_names": ["pt_phi1_microsoft_phi_1_token_cls_hf", "pt_phi_1_5_microsoft_phi_1_5_token_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 12, 32, 64)"},
        },
    ),
    (
        Reshape252,
        [((12, 2048), torch.float32)],
        {
            "model_names": ["pt_phi1_microsoft_phi_1_token_cls_hf", "pt_phi_1_5_microsoft_phi_1_5_token_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 12, 2048)"},
        },
    ),
    (
        Reshape253,
        [((1, 32, 12, 64), torch.float32)],
        {
            "model_names": ["pt_phi1_microsoft_phi_1_token_cls_hf", "pt_phi_1_5_microsoft_phi_1_5_token_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(32, 12, 64)"},
        },
    ),
    (
        Reshape254,
        [((1, 32, 12, 64), torch.float32)],
        {
            "model_names": ["pt_opt_facebook_opt_125m_qa_hf", "pt_opt_facebook_opt_125m_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(32, 768)"},
        },
    ),
    (
        Reshape255,
        [((32, 12, 12), torch.float32)],
        {
            "model_names": ["pt_phi1_microsoft_phi_1_token_cls_hf", "pt_phi_1_5_microsoft_phi_1_5_token_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 32, 12, 12)"},
        },
    ),
    (
        Reshape256,
        [((1, 32, 12, 12), torch.float32)],
        {
            "model_names": ["pt_phi1_microsoft_phi_1_token_cls_hf", "pt_phi_1_5_microsoft_phi_1_5_token_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(32, 12, 12)"},
        },
    ),
    (
        Reshape257,
        [((1, 32, 64, 12), torch.float32)],
        {
            "model_names": ["pt_phi1_microsoft_phi_1_token_cls_hf", "pt_phi_1_5_microsoft_phi_1_5_token_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(32, 64, 12)"},
        },
    ),
    (
        Reshape258,
        [((32, 12, 64), torch.float32)],
        {
            "model_names": ["pt_phi1_microsoft_phi_1_token_cls_hf", "pt_phi_1_5_microsoft_phi_1_5_token_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 32, 12, 64)"},
        },
    ),
    (
        Reshape250,
        [((1, 12, 32, 64), torch.float32)],
        {
            "model_names": ["pt_phi1_microsoft_phi_1_token_cls_hf", "pt_phi_1_5_microsoft_phi_1_5_token_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(12, 2048)"},
        },
    ),
    (
        Reshape259,
        [((1, 12, 32, 64), torch.float32)],
        {
            "model_names": ["pt_opt_facebook_opt_125m_qa_hf", "pt_opt_facebook_opt_125m_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(12, 32, 64)"},
        },
    ),
    (
        Reshape260,
        [((12, 8192), torch.float32)],
        {
            "model_names": ["pt_phi1_microsoft_phi_1_token_cls_hf", "pt_phi_1_5_microsoft_phi_1_5_token_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 12, 8192)"},
        },
    ),
    (
        Reshape261,
        [((1, 35, 1536), torch.float32)],
        {"model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf"], "pcc": 0.99, "args": {"shape": "(35, 1536)"}},
    ),
    (
        Reshape262,
        [((1, 35, 1536), torch.float32)],
        {
            "model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 35, 12, 128)"},
        },
    ),
    (
        Reshape263,
        [((35, 1536), torch.float32)],
        {
            "model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 35, 1536)"},
        },
    ),
    (
        Reshape264,
        [((1, 12, 35, 128), torch.float32)],
        {
            "model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(12, 35, 128)"},
        },
    ),
    (
        Reshape265,
        [((35, 256), torch.float32)],
        {
            "model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 35, 256)"},
        },
    ),
    (
        Reshape266,
        [((1, 35, 256), torch.float32)],
        {
            "model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 35, 2, 128)"},
        },
    ),
    (
        Reshape264,
        [((1, 2, 6, 35, 128), torch.float32)],
        {
            "model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(12, 35, 128)"},
        },
    ),
    (
        Reshape267,
        [((1, 2, 6, 35, 128), torch.float32)],
        {
            "model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 12, 35, 128)"},
        },
    ),
    (
        Reshape268,
        [((12, 35, 35), torch.float32)],
        {
            "model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 12, 35, 35)"},
        },
    ),
    (
        Reshape269,
        [((1, 12, 35, 35), torch.float32)],
        {
            "model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(12, 35, 35)"},
        },
    ),
    (
        Reshape270,
        [((1, 12, 128, 35), torch.float32)],
        {
            "model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(12, 128, 35)"},
        },
    ),
    (
        Reshape267,
        [((12, 35, 128), torch.float32)],
        {
            "model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 12, 35, 128)"},
        },
    ),
    (
        Reshape261,
        [((1, 35, 12, 128), torch.float32)],
        {"model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf"], "pcc": 0.99, "args": {"shape": "(35, 1536)"}},
    ),
    (
        Reshape271,
        [((35, 8960), torch.float32)],
        {
            "model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 35, 8960)"},
        },
    ),
    (
        Reshape272,
        [((1, 2520, 1, 1), torch.float32)],
        {
            "model_names": ["pt_regnet_regnet_x_32gf_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(1, 2520, 1, 1)"},
        },
    ),
    (
        Reshape273,
        [((1, 440, 1, 1), torch.float32)],
        {
            "model_names": ["pt_regnet_regnet_y_400mf_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(1, 440, 1, 1)"},
        },
    ),
    (
        Reshape274,
        [((1, 64, 64, 96), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 8, 8, 8, 8, 96)"},
        },
    ),
    (
        Reshape275,
        [((1, 64, 64, 96), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(4096, 96)"},
        },
    ),
    (
        Reshape276,
        [((1, 64, 64, 96), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 4096, 96)"},
        },
    ),
    (
        Reshape275,
        [((1, 8, 8, 8, 8, 96), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(4096, 96)"},
        },
    ),
    (
        Reshape277,
        [((1, 8, 8, 8, 8, 96), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 64, 64, 96)"},
        },
    ),
    (
        Reshape276,
        [((1, 8, 8, 8, 8, 96), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 4096, 96)"},
        },
    ),
    (
        Reshape278,
        [((4096, 288), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(64, 64, 288)"},
        },
    ),
    (
        Reshape279,
        [((64, 64, 288), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(64, 64, 3, 3, 32)"},
        },
    ),
    (
        Reshape280,
        [((1, 64, 3, 64, 32), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(64, 3, 64, 32)"},
        },
    ),
    (
        Reshape281,
        [((64, 3, 64, 32), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(192, 64, 32)"},
        },
    ),
    (
        Reshape282,
        [((192, 64, 64), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(64, 3, 64, 64)"},
        },
    ),
    (
        Reshape283,
        [((1, 15, 15, 2), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "pt_swin_swin_v2_b_img_cls_torchvision",
                "pt_swin_swin_v2_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(225, 2)"},
        },
    ),
    (
        Reshape284,
        [((225, 512), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_b_img_cls_torchvision",
                "pt_swin_swin_v2_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 15, 15, 512)"},
        },
    ),
    (
        Reshape285,
        [((1, 15, 15, 512), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_b_img_cls_torchvision",
                "pt_swin_swin_v2_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(225, 512)"},
        },
    ),
    (
        Reshape286,
        [((225, 3), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(225, 3)"},
        },
    ),
    (
        Reshape287,
        [((4096, 3), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(64, 64, 3)"},
        },
    ),
    (
        Reshape288,
        [((64, 3, 64, 64), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(192, 64, 64)"},
        },
    ),
    (
        Reshape289,
        [((64, 3, 64, 64), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 64, 3, 64, 64)"},
        },
    ),
    (
        Reshape290,
        [((64, 3, 32, 64), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(192, 32, 64)"},
        },
    ),
    (
        Reshape280,
        [((192, 64, 32), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(64, 3, 64, 32)"},
        },
    ),
    (
        Reshape275,
        [((64, 64, 3, 32), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(4096, 96)"},
        },
    ),
    (
        Reshape291,
        [((4096, 96), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(64, 64, 96)"},
        },
    ),
    (
        Reshape277,
        [((4096, 96), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(1, 64, 64, 96)"},
        },
    ),
    (
        Reshape292,
        [((4096, 96), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(64, 64, 3, 32)"},
        },
    ),
    (
        Reshape274,
        [((64, 64, 96), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 8, 8, 8, 8, 96)"},
        },
    ),
    (
        Reshape292,
        [((64, 64, 96), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(64, 64, 3, 32)"},
        },
    ),
    (
        Reshape293,
        [((4096, 384), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(1, 64, 64, 384)"},
        },
    ),
    (
        Reshape294,
        [((4096, 384), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(64, 64, 384)"}},
    ),
    (
        Reshape295,
        [((1, 64, 64, 384), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(4096, 384)"},
        },
    ),
    (
        Reshape282,
        [((1, 64, 3, 64, 64), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(64, 3, 64, 64)"},
        },
    ),
    (
        Reshape296,
        [((1, 32, 32, 384), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1024, 384)"},
        },
    ),
    (
        Reshape297,
        [((1024, 192), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(1, 32, 32, 192)"},
        },
    ),
    (
        Reshape298,
        [((1024, 192), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(16, 64, 192)"},
        },
    ),
    (
        Reshape299,
        [((1024, 192), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 1024, 192)"},
        },
    ),
    (
        Reshape300,
        [((1024, 192), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(16, 64, 6, 32)"},
        },
    ),
    (
        Reshape301,
        [((1, 32, 32, 192), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 4, 8, 4, 8, 192)"},
        },
    ),
    (
        Reshape302,
        [((1, 32, 32, 192), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(1024, 192)"},
        },
    ),
    (
        Reshape299,
        [((1, 32, 32, 192), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 1024, 192)"},
        },
    ),
    (
        Reshape302,
        [((1, 4, 4, 8, 8, 192), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1024, 192)"},
        },
    ),
    (
        Reshape303,
        [((1024, 576), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(16, 64, 576)"},
        },
    ),
    (
        Reshape304,
        [((16, 64, 576), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(16, 64, 3, 6, 32)"},
        },
    ),
    (
        Reshape305,
        [((1, 16, 6, 64, 32), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(16, 6, 64, 32)"},
        },
    ),
    (
        Reshape306,
        [((16, 6, 64, 32), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(96, 64, 32)"},
        },
    ),
    (
        Reshape307,
        [((96, 64, 64), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(16, 6, 64, 64)"},
        },
    ),
    (
        Reshape308,
        [((225, 6), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(225, 6)"},
        },
    ),
    (
        Reshape309,
        [((4096, 6), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(64, 64, 6)"},
        },
    ),
    (
        Reshape310,
        [((16, 6, 64, 64), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(96, 64, 64)"},
        },
    ),
    (
        Reshape311,
        [((16, 6, 64, 64), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 16, 6, 64, 64)"},
        },
    ),
    (
        Reshape312,
        [((16, 6, 32, 64), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(96, 32, 64)"},
        },
    ),
    (
        Reshape305,
        [((96, 64, 32), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(16, 6, 64, 32)"},
        },
    ),
    (
        Reshape302,
        [((16, 64, 6, 32), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1024, 192)"},
        },
    ),
    (
        Reshape313,
        [((16, 64, 192), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 4, 4, 8, 8, 192)"},
        },
    ),
    (
        Reshape300,
        [((16, 64, 192), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(16, 64, 6, 32)"},
        },
    ),
    (
        Reshape297,
        [((1, 4, 8, 4, 8, 192), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 32, 32, 192)"},
        },
    ),
    (
        Reshape299,
        [((1, 4, 8, 4, 8, 192), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 1024, 192)"},
        },
    ),
    (
        Reshape314,
        [((1024, 768), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(1, 32, 32, 768)"},
        },
    ),
    (
        Reshape315,
        [((1024, 768), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(16, 64, 768)"}},
    ),
    (
        Reshape316,
        [((1, 32, 32, 768), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(1024, 768)"},
        },
    ),
    (
        Reshape307,
        [((1, 16, 6, 64, 64), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(16, 6, 64, 64)"},
        },
    ),
    (
        Reshape317,
        [((1, 16, 16, 768), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(256, 768)"},
        },
    ),
    (
        Reshape318,
        [((256, 384), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(1, 16, 16, 384)"},
        },
    ),
    (
        Reshape319,
        [((256, 384), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(4, 64, 384)"},
        },
    ),
    (
        Reshape320,
        [((256, 384), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 256, 384)"},
        },
    ),
    (
        Reshape321,
        [((256, 384), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(4, 64, 12, 32)"},
        },
    ),
    (
        Reshape322,
        [((1, 16, 16, 384), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 2, 8, 2, 8, 384)"},
        },
    ),
    (
        Reshape323,
        [((1, 16, 16, 384), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(256, 384)"},
        },
    ),
    (
        Reshape320,
        [((1, 16, 16, 384), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 256, 384)"},
        },
    ),
    (
        Reshape323,
        [((1, 2, 2, 8, 8, 384), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(256, 384)"},
        },
    ),
    (
        Reshape324,
        [((256, 1152), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(4, 64, 1152)"},
        },
    ),
    (
        Reshape325,
        [((4, 64, 1152), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(4, 64, 3, 12, 32)"},
        },
    ),
    (
        Reshape326,
        [((1, 4, 12, 64, 32), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(4, 12, 64, 32)"},
        },
    ),
    (
        Reshape327,
        [((4, 12, 64, 32), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(48, 64, 32)"},
        },
    ),
    (
        Reshape328,
        [((48, 64, 64), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(4, 12, 64, 64)"},
        },
    ),
    (
        Reshape329,
        [((225, 12), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(225, 12)"},
        },
    ),
    (
        Reshape330,
        [((4096, 12), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(64, 64, 12)"},
        },
    ),
    (
        Reshape331,
        [((4, 12, 64, 64), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(48, 64, 64)"},
        },
    ),
    (
        Reshape332,
        [((4, 12, 64, 64), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 4, 12, 64, 64)"},
        },
    ),
    (
        Reshape333,
        [((4, 12, 32, 64), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(48, 32, 64)"},
        },
    ),
    (
        Reshape326,
        [((48, 64, 32), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(4, 12, 64, 32)"},
        },
    ),
    (
        Reshape323,
        [((4, 64, 12, 32), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(256, 384)"},
        },
    ),
    (
        Reshape334,
        [((4, 64, 384), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 2, 2, 8, 8, 384)"},
        },
    ),
    (
        Reshape321,
        [((4, 64, 384), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(4, 64, 12, 32)"},
        },
    ),
    (
        Reshape318,
        [((1, 2, 8, 2, 8, 384), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 16, 16, 384)"},
        },
    ),
    (
        Reshape320,
        [((1, 2, 8, 2, 8, 384), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 256, 384)"},
        },
    ),
    (
        Reshape335,
        [((256, 1536), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(1, 16, 16, 1536)"},
        },
    ),
    (
        Reshape336,
        [((256, 1536), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(4, 64, 1536)"}},
    ),
    (
        Reshape337,
        [((1, 16, 16, 1536), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(256, 1536)"},
        },
    ),
    (
        Reshape328,
        [((1, 4, 12, 64, 64), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(4, 12, 64, 64)"},
        },
    ),
    (
        Reshape338,
        [((1, 8, 8, 1536), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(64, 1536)"},
        },
    ),
    (
        Reshape339,
        [((64, 768), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(1, 8, 8, 768)"},
        },
    ),
    (
        Reshape340,
        [((64, 768), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 64, 768)"},
        },
    ),
    (
        Reshape341,
        [((64, 768), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 64, 24, 32)"},
        },
    ),
    (
        Reshape342,
        [((1, 8, 8, 768), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 1, 8, 1, 8, 768)"},
        },
    ),
    (
        Reshape343,
        [((1, 8, 8, 768), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(64, 768)"},
        },
    ),
    (
        Reshape343,
        [((1, 1, 1, 8, 8, 768), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(64, 768)"},
        },
    ),
    (
        Reshape344,
        [((64, 2304), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(1, 64, 2304)"},
        },
    ),
    (
        Reshape345,
        [((1, 64, 2304), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(1, 64, 3, 24, 32)"},
        },
    ),
    (
        Reshape346,
        [((1, 1, 24, 64, 32), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(1, 24, 64, 32)"},
        },
    ),
    (
        Reshape347,
        [((1, 24, 64, 32), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(24, 64, 32)"},
        },
    ),
    (
        Reshape348,
        [((24, 64, 64), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 24, 64, 64)"},
        },
    ),
    (
        Reshape349,
        [((225, 24), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(225, 24)"},
        },
    ),
    (
        Reshape350,
        [((4096, 24), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(64, 64, 24)"},
        },
    ),
    (
        Reshape351,
        [((1, 24, 64, 64), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(24, 64, 64)"},
        },
    ),
    (
        Reshape352,
        [((1, 24, 32, 64), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(24, 32, 64)"},
        },
    ),
    (
        Reshape346,
        [((24, 64, 32), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 24, 64, 32)"},
        },
    ),
    (
        Reshape343,
        [((1, 64, 24, 32), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(64, 768)"},
        },
    ),
    (
        Reshape353,
        [((1, 64, 768), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(1, 1, 1, 8, 8, 768)"},
        },
    ),
    (
        Reshape339,
        [((1, 64, 768), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 8, 8, 768)"},
        },
    ),
    (
        Reshape341,
        [((1, 64, 768), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 64, 24, 32)"},
        },
    ),
    (
        Reshape340,
        [((1, 64, 768), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 64, 768)"},
        },
    ),
    (
        Reshape339,
        [((1, 1, 8, 1, 8, 768), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(1, 8, 8, 768)"},
        },
    ),
    (
        Reshape354,
        [((64, 3072), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(1, 8, 8, 3072)"},
        },
    ),
    (
        Reshape355,
        [((64, 3072), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(1, 64, 3072)"}},
    ),
    (
        Reshape356,
        [((1, 8, 8, 3072), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(64, 3072)"},
        },
    ),
    (
        Reshape357,
        [((1, 768, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_swin_v2_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 768, 1, 1)"},
        },
    ),
    (
        Reshape358,
        [((1, 1), torch.int64)],
        {
            "model_names": [
                "pt_t5_google_flan_t5_small_text_gen_hf",
                "pt_whisper_openai_whisper_medium_speech_recognition_hf",
                "pt_t5_t5_base_text_gen_hf",
                "pt_whisper_openai_whisper_small_speech_recognition_hf",
                "pt_t5_t5_small_text_gen_hf",
                "pt_t5_google_flan_t5_base_text_gen_hf",
                "pt_t5_google_flan_t5_large_text_gen_hf",
                "pt_whisper_openai_whisper_base_speech_recognition_hf",
                "pt_t5_t5_large_text_gen_hf",
                "pt_whisper_openai_whisper_tiny_speech_recognition_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 1)"},
        },
    ),
    (
        Reshape13,
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
        Reshape359,
        [((1, 384), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 1, 384)"},
        },
    ),
    (
        Reshape360,
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
        Reshape361,
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
        Reshape362,
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
        Reshape363,
        [((1, 6, 64, 1), torch.float32)],
        {
            "model_names": [
                "pt_t5_google_flan_t5_small_text_gen_hf",
                "pt_whisper_openai_whisper_tiny_speech_recognition_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(6, 64, 1)"},
        },
    ),
    (
        Reshape364,
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
        Reshape12,
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
        Reshape248,
        [((1, 512), torch.float32)],
        {
            "model_names": [
                "pt_t5_google_flan_t5_small_text_gen_hf",
                "pt_t5_t5_small_text_gen_hf",
                "pt_whisper_openai_whisper_base_speech_recognition_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 1, 512)"},
        },
    ),
    (
        Reshape247,
        [((1, 512), torch.float32)],
        {
            "model_names": ["pt_t5_t5_small_text_gen_hf", "pt_whisper_openai_whisper_base_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 1, 8, 64)"},
        },
    ),
    (
        Reshape365,
        [((1, 61), torch.int64)],
        {
            "model_names": [
                "pt_t5_google_flan_t5_small_text_gen_hf",
                "pt_t5_t5_base_text_gen_hf",
                "pt_t5_t5_small_text_gen_hf",
                "pt_t5_google_flan_t5_base_text_gen_hf",
                "pt_t5_google_flan_t5_large_text_gen_hf",
                "pt_t5_t5_large_text_gen_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 61)"},
        },
    ),
    (
        Reshape366,
        [((1, 61, 512), torch.float32)],
        {
            "model_names": ["pt_t5_google_flan_t5_small_text_gen_hf", "pt_t5_t5_small_text_gen_hf"],
            "pcc": 0.99,
            "args": {"shape": "(61, 512)"},
        },
    ),
    (
        Reshape367,
        [((61, 384), torch.float32)],
        {"model_names": ["pt_t5_google_flan_t5_small_text_gen_hf"], "pcc": 0.99, "args": {"shape": "(1, 61, 6, 64)"}},
    ),
    (
        Reshape368,
        [((1, 6, 61, 64), torch.float32)],
        {"model_names": ["pt_t5_google_flan_t5_small_text_gen_hf"], "pcc": 0.99, "args": {"shape": "(6, 61, 64)"}},
    ),
    (
        Reshape369,
        [((6, 61, 61), torch.float32)],
        {"model_names": ["pt_t5_google_flan_t5_small_text_gen_hf"], "pcc": 0.99, "args": {"shape": "(1, 6, 61, 61)"}},
    ),
    (
        Reshape370,
        [((1, 6, 61, 61), torch.float32)],
        {"model_names": ["pt_t5_google_flan_t5_small_text_gen_hf"], "pcc": 0.99, "args": {"shape": "(6, 61, 61)"}},
    ),
    (
        Reshape371,
        [((1, 6, 64, 61), torch.float32)],
        {"model_names": ["pt_t5_google_flan_t5_small_text_gen_hf"], "pcc": 0.99, "args": {"shape": "(6, 64, 61)"}},
    ),
    (
        Reshape372,
        [((6, 61, 64), torch.float32)],
        {"model_names": ["pt_t5_google_flan_t5_small_text_gen_hf"], "pcc": 0.99, "args": {"shape": "(1, 6, 61, 64)"}},
    ),
    (
        Reshape373,
        [((1, 61, 6, 64), torch.float32)],
        {"model_names": ["pt_t5_google_flan_t5_small_text_gen_hf"], "pcc": 0.99, "args": {"shape": "(61, 384)"}},
    ),
    (
        Reshape374,
        [((61, 512), torch.float32)],
        {
            "model_names": ["pt_t5_google_flan_t5_small_text_gen_hf", "pt_t5_t5_small_text_gen_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 61, 512)"},
        },
    ),
    (
        Reshape375,
        [((61, 512), torch.float32)],
        {"model_names": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99, "args": {"shape": "(1, 61, 8, 64)"}},
    ),
    (
        Reshape376,
        [((61, 1024), torch.float32)],
        {
            "model_names": [
                "pt_t5_google_flan_t5_small_text_gen_hf",
                "pt_t5_google_flan_t5_large_text_gen_hf",
                "pt_t5_t5_large_text_gen_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 61, 1024)"},
        },
    ),
    (
        Reshape377,
        [((61, 1024), torch.float32)],
        {
            "model_names": ["pt_t5_google_flan_t5_large_text_gen_hf", "pt_t5_t5_large_text_gen_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 61, 16, 64)"},
        },
    ),
    (
        Reshape378,
        [((6, 1, 61), torch.float32)],
        {"model_names": ["pt_t5_google_flan_t5_small_text_gen_hf"], "pcc": 0.99, "args": {"shape": "(1, 6, 1, 61)"}},
    ),
    (
        Reshape379,
        [((1, 6, 1, 61), torch.float32)],
        {"model_names": ["pt_t5_google_flan_t5_small_text_gen_hf"], "pcc": 0.99, "args": {"shape": "(6, 1, 61)"}},
    ),
    (
        Reshape380,
        [((1, 1024), torch.float32)],
        {
            "model_names": [
                "pt_t5_google_flan_t5_small_text_gen_hf",
                "pt_whisper_openai_whisper_medium_speech_recognition_hf",
                "pt_t5_google_flan_t5_large_text_gen_hf",
                "pt_t5_t5_large_text_gen_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 1, 1024)"},
        },
    ),
    (
        Reshape228,
        [((1, 1024), torch.float32)],
        {
            "model_names": [
                "pt_whisper_openai_whisper_medium_speech_recognition_hf",
                "pt_t5_google_flan_t5_large_text_gen_hf",
                "pt_t5_t5_large_text_gen_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 1, 16, 64)"},
        },
    ),
    (
        Reshape381,
        [((1, 512, 7, 7), torch.float32)],
        {
            "model_names": [
                "pt_vgg_19_obj_det_hf",
                "pt_vgg_vgg11_bn_img_cls_torchvision",
                "pt_vgg_vgg13_img_cls_torchvision",
                "pt_vgg_vgg16_img_cls_torchvision",
                "pt_vgg_vgg16_bn_img_cls_torchvision",
                "pt_vgg_vgg11_img_cls_torchvision",
                "pt_vgg_vgg13_bn_img_cls_torchvision",
                "pt_vgg_vgg19_bn_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 25088, 1, 1)"},
        },
    ),
    (
        Reshape382,
        [((1, 512, 7, 7), torch.float32)],
        {
            "model_names": [
                "pt_vgg_bn_vgg19_obj_det_osmr",
                "pt_vgg_bn_vgg19b_obj_det_osmr",
                "pt_vgg_vgg16_obj_det_osmr",
                "pt_vgg_vgg19_obj_det_osmr",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 25088)"},
        },
    ),
    (
        Reshape383,
        [((1, 512, 7, 7), torch.float32)],
        {"model_names": ["pt_mlp_mixer_mixer_s32_224_img_cls_timm"], "pcc": 0.99, "args": {"shape": "(1, 512, 49, 1)"}},
    ),
    (
        Reshape384,
        [((1, 1024, 14, 14), torch.float32)],
        {
            "model_names": [
                "pt_vit_google_vit_large_patch16_224_img_cls_hf",
                "pt_beit_microsoft_beit_large_patch16_224_img_cls_hf",
                "pt_mlp_mixer_mixer_l16_224_img_cls_timm",
                "pt_mlp_mixer_mixer_l16_224_in21k_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 1024, 196, 1)"},
        },
    ),
    (
        Reshape385,
        [((1, 1024, 14, 14), torch.float32)],
        {"model_names": ["pt_vit_vit_l_16_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(1, 1024, 196)"}},
    ),
    (
        Reshape386,
        [((1, 197, 1024), torch.float32)],
        {
            "model_names": [
                "pt_vit_google_vit_large_patch16_224_img_cls_hf",
                "pt_beit_microsoft_beit_large_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(197, 1024)"},
        },
    ),
    (
        Reshape387,
        [((1, 197, 1024), torch.float32)],
        {
            "model_names": [
                "pt_vit_google_vit_large_patch16_224_img_cls_hf",
                "pt_beit_microsoft_beit_large_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 197, 16, 64)"},
        },
    ),
    (
        Reshape388,
        [((197, 1024), torch.float32)],
        {
            "model_names": [
                "pt_vit_google_vit_large_patch16_224_img_cls_hf",
                "pt_beit_microsoft_beit_large_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 197, 1024)"},
        },
    ),
    (
        Reshape387,
        [((197, 1024), torch.float32)],
        {
            "model_names": ["pt_beit_microsoft_beit_large_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 197, 16, 64)"},
        },
    ),
    (
        Reshape389,
        [((197, 1024), torch.float32)],
        {"model_names": ["pt_vit_vit_l_16_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(197, 1, 1024)"}},
    ),
    (
        Reshape390,
        [((1, 16, 197, 64), torch.float32)],
        {
            "model_names": [
                "pt_vit_google_vit_large_patch16_224_img_cls_hf",
                "pt_beit_microsoft_beit_large_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(16, 197, 64)"},
        },
    ),
    (
        Reshape391,
        [((16, 197, 197), torch.float32)],
        {
            "model_names": [
                "pt_vit_google_vit_large_patch16_224_img_cls_hf",
                "pt_beit_microsoft_beit_large_patch16_224_img_cls_hf",
                "pt_vit_vit_l_16_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 16, 197, 197)"},
        },
    ),
    (
        Reshape392,
        [((1, 16, 197, 197), torch.float32)],
        {
            "model_names": [
                "pt_vit_google_vit_large_patch16_224_img_cls_hf",
                "pt_beit_microsoft_beit_large_patch16_224_img_cls_hf",
                "pt_vit_vit_l_16_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(16, 197, 197)"},
        },
    ),
    (
        Reshape393,
        [((1, 16, 64, 197), torch.float32)],
        {
            "model_names": [
                "pt_vit_google_vit_large_patch16_224_img_cls_hf",
                "pt_beit_microsoft_beit_large_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(16, 64, 197)"},
        },
    ),
    (
        Reshape394,
        [((16, 197, 64), torch.float32)],
        {
            "model_names": [
                "pt_vit_google_vit_large_patch16_224_img_cls_hf",
                "pt_beit_microsoft_beit_large_patch16_224_img_cls_hf",
                "pt_vit_vit_l_16_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 16, 197, 64)"},
        },
    ),
    (
        Reshape390,
        [((16, 197, 64), torch.float32)],
        {"model_names": ["pt_vit_vit_l_16_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(16, 197, 64)"}},
    ),
    (
        Reshape386,
        [((1, 197, 16, 64), torch.float32)],
        {
            "model_names": [
                "pt_vit_google_vit_large_patch16_224_img_cls_hf",
                "pt_beit_microsoft_beit_large_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(197, 1024)"},
        },
    ),
    (
        Reshape395,
        [((1, 16, 1, 64), torch.float32)],
        {
            "model_names": [
                "pt_whisper_openai_whisper_medium_speech_recognition_hf",
                "pt_t5_google_flan_t5_large_text_gen_hf",
                "pt_t5_t5_large_text_gen_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(16, 1, 64)"},
        },
    ),
    (
        Reshape396,
        [((16, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_whisper_openai_whisper_medium_speech_recognition_hf",
                "pt_t5_google_flan_t5_large_text_gen_hf",
                "pt_t5_t5_large_text_gen_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 16, 1, 1)"},
        },
    ),
    (
        Reshape397,
        [((1, 16, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_whisper_openai_whisper_medium_speech_recognition_hf",
                "pt_t5_google_flan_t5_large_text_gen_hf",
                "pt_t5_t5_large_text_gen_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(16, 1, 1)"},
        },
    ),
    (
        Reshape398,
        [((1, 16, 1, 1), torch.float32)],
        {"model_names": ["pt_ssd300_resnet50_base_img_cls_torchhub"], "pcc": 0.99, "args": {"shape": "(1, 4, 4)"}},
    ),
    (
        Reshape399,
        [((1, 16, 64, 1), torch.float32)],
        {
            "model_names": [
                "pt_whisper_openai_whisper_medium_speech_recognition_hf",
                "pt_t5_google_flan_t5_large_text_gen_hf",
                "pt_t5_t5_large_text_gen_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(16, 64, 1)"},
        },
    ),
    (
        Reshape400,
        [((16, 1, 64), torch.float32)],
        {
            "model_names": [
                "pt_whisper_openai_whisper_medium_speech_recognition_hf",
                "pt_t5_google_flan_t5_large_text_gen_hf",
                "pt_t5_t5_large_text_gen_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 16, 1, 64)"},
        },
    ),
    (
        Reshape98,
        [((1, 1, 16, 64), torch.float32)],
        {
            "model_names": [
                "pt_whisper_openai_whisper_medium_speech_recognition_hf",
                "pt_t5_google_flan_t5_large_text_gen_hf",
                "pt_t5_t5_large_text_gen_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 1024)"},
        },
    ),
    (
        Reshape401,
        [((1, 80, 3000), torch.float32)],
        {
            "model_names": [
                "pt_whisper_openai_whisper_medium_speech_recognition_hf",
                "pt_whisper_openai_whisper_small_speech_recognition_hf",
                "pt_whisper_openai_whisper_base_speech_recognition_hf",
                "pt_whisper_openai_whisper_tiny_speech_recognition_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 80, 3000, 1)"},
        },
    ),
    (
        Reshape402,
        [((1024, 80, 3), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_medium_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1024, 80, 3, 1)"},
        },
    ),
    (
        Reshape403,
        [((1, 1024, 3000, 1), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_medium_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 1024, 3000)"},
        },
    ),
    (
        Reshape404,
        [((1, 1024, 3000), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_medium_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 1024, 3000, 1)"},
        },
    ),
    (
        Reshape405,
        [((1024, 1024, 3), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_medium_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1024, 1024, 3, 1)"},
        },
    ),
    (
        Reshape406,
        [((1, 1024, 1500, 1), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_medium_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 1024, 1500)"},
        },
    ),
    (
        Reshape407,
        [((1, 1500, 1024), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_medium_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1500, 1024)"},
        },
    ),
    (
        Reshape408,
        [((1, 1500, 1024), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_medium_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 1500, 16, 64)"},
        },
    ),
    (
        Reshape409,
        [((1500, 1024), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_medium_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 1500, 1024)"},
        },
    ),
    (
        Reshape408,
        [((1500, 1024), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_medium_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 1500, 16, 64)"},
        },
    ),
    (
        Reshape410,
        [((1, 16, 1500, 64), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_medium_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(16, 1500, 64)"},
        },
    ),
    (
        Reshape411,
        [((16, 1500, 1500), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_medium_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 16, 1500, 1500)"},
        },
    ),
    (
        Reshape412,
        [((1, 16, 1500, 1500), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_medium_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(16, 1500, 1500)"},
        },
    ),
    (
        Reshape413,
        [((1, 16, 64, 1500), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_medium_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(16, 64, 1500)"},
        },
    ),
    (
        Reshape414,
        [((16, 1500, 64), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_medium_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 16, 1500, 64)"},
        },
    ),
    (
        Reshape407,
        [((1, 1500, 16, 64), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_medium_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1500, 1024)"},
        },
    ),
    (
        Reshape415,
        [((16, 1, 1500), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_medium_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 16, 1, 1500)"},
        },
    ),
    (
        Reshape416,
        [((1, 16, 1, 1500), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_medium_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(16, 1, 1500)"},
        },
    ),
    (
        Reshape417,
        [((64, 1, 3, 3), torch.float32)],
        {
            "model_names": [
                "pt_xception_xception65_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_xception_xception71_tf_in1k_img_cls_timm",
                "pt_xception_xception_img_cls_timm",
                "pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
                "pt_mobilenetv1_basic_img_cls_torchvision",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_yolox_yolox_nano_obj_det_torchhub",
                "pt_xception_xception41_img_cls_timm",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
                "pt_xception_xception71_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"shape": "(64, 1, 3, 3)"},
        },
    ),
    (
        Reshape418,
        [((128, 1, 3, 3), torch.float32)],
        {
            "model_names": [
                "pt_xception_xception65_img_cls_timm",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_xception_xception71_tf_in1k_img_cls_timm",
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_xception_xception_img_cls_timm",
                "pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pt_mobilenetv1_basic_img_cls_torchvision",
                "pt_yolox_yolox_nano_obj_det_torchhub",
                "pt_xception_xception41_img_cls_timm",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
                "pt_xception_xception71_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"shape": "(128, 1, 3, 3)"},
        },
    ),
    (
        Reshape419,
        [((256, 1, 3, 3), torch.float32)],
        {
            "model_names": [
                "pt_xception_xception65_img_cls_timm",
                "pt_xception_xception71_tf_in1k_img_cls_timm",
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_xception_xception_img_cls_timm",
                "pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf",
                "pt_mobilenetv1_basic_img_cls_torchvision",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"shape": "(256, 1, 3, 3)"},
        },
    ),
    (
        Reshape420,
        [((728, 1, 3, 3), torch.float32)],
        {
            "model_names": [
                "pt_xception_xception65_img_cls_timm",
                "pt_xception_xception71_tf_in1k_img_cls_timm",
                "pt_xception_xception_img_cls_timm",
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"shape": "(728, 1, 3, 3)"},
        },
    ),
    (
        Reshape421,
        [((1024, 1, 3, 3), torch.float32)],
        {
            "model_names": [
                "pt_xception_xception65_img_cls_timm",
                "pt_xception_xception71_tf_in1k_img_cls_timm",
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_xception_xception_img_cls_timm",
                "pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pt_mobilenetv1_basic_img_cls_torchvision",
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1024, 1, 3, 3)"},
        },
    ),
    (
        Reshape422,
        [((1536, 1, 3, 3), torch.float32)],
        {
            "model_names": [
                "pt_xception_xception65_img_cls_timm",
                "pt_xception_xception71_tf_in1k_img_cls_timm",
                "pt_xception_xception_img_cls_timm",
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1536, 1, 3, 3)"},
        },
    ),
    (
        Reshape423,
        [((1, 3, 85, 40, 40), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5s_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_640x640",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 255, 40, 40)"},
        },
    ),
    (
        Reshape424,
        [((1, 255, 40, 40), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5s_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_640x640",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 1, 255, 1600)"},
        },
    ),
    (
        Reshape425,
        [((1, 1, 255, 1600), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5s_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_640x640",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 3, 85, 1600)"},
        },
    ),
    (
        Reshape426,
        [((1, 3, 1600, 85), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5s_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_640x640",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 4800, 85)"},
        },
    ),
    (
        Reshape427,
        [((1, 3, 85, 20, 20), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5s_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_640x640",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 255, 20, 20)"},
        },
    ),
    (
        Reshape428,
        [((1, 255, 20, 20), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5s_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_640x640",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 1, 255, 400)"},
        },
    ),
    (
        Reshape429,
        [((1, 1, 255, 400), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5s_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_640x640",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 3, 85, 400)"},
        },
    ),
    (
        Reshape430,
        [((1, 3, 400, 85), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5s_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_640x640",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 1200, 85)"},
        },
    ),
    (
        Reshape431,
        [((1, 3, 85, 10, 10), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5s_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_320x320",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 255, 10, 10)"},
        },
    ),
    (
        Reshape432,
        [((1, 255, 10, 10), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5s_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_320x320",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 1, 255, 100)"},
        },
    ),
    (
        Reshape433,
        [((1, 1, 255, 100), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5s_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_320x320",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 3, 85, 100)"},
        },
    ),
    (
        Reshape434,
        [((1, 3, 100, 85), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5s_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_320x320",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 300, 85)"},
        },
    ),
    (
        Reshape435,
        [((1, 3, 85, 80, 80), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5s_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_640x640",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 255, 80, 80)"},
        },
    ),
    (
        Reshape436,
        [((1, 255, 80, 80), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5s_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_640x640",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 1, 255, 6400)"},
        },
    ),
    (
        Reshape437,
        [((1, 1, 255, 6400), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5s_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_640x640",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 3, 85, 6400)"},
        },
    ),
    (
        Reshape438,
        [((1, 3, 6400, 85), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5s_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_640x640",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 19200, 85)"},
        },
    ),
    (
        Reshape439,
        [((1, 4, 56, 80), torch.float32)],
        {
            "model_names": ["pt_yolo_v6_yolov6n_obj_det_torchhub", "pt_yolo_v6_yolov6s_obj_det_torchhub"],
            "pcc": 0.99,
            "args": {"shape": "(1, 4, 4480)"},
        },
    ),
    (
        Reshape440,
        [((1, 4, 28, 40), torch.float32)],
        {
            "model_names": ["pt_yolo_v6_yolov6n_obj_det_torchhub", "pt_yolo_v6_yolov6s_obj_det_torchhub"],
            "pcc": 0.99,
            "args": {"shape": "(1, 4, 1120)"},
        },
    ),
    (
        Reshape441,
        [((1, 4, 14, 20), torch.float32)],
        {
            "model_names": ["pt_yolo_v6_yolov6n_obj_det_torchhub", "pt_yolo_v6_yolov6s_obj_det_torchhub"],
            "pcc": 0.99,
            "args": {"shape": "(1, 4, 280)"},
        },
    ),
    (
        Reshape442,
        [((1, 80, 56, 80), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v6_yolov6n_obj_det_torchhub",
                "pt_yolo_v6_yolov6s_obj_det_torchhub",
                "pt_yolo_v6_yolov6l_obj_det_torchhub",
                "pt_yolo_v6_yolov6m_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 80, 4480)"},
        },
    ),
    (
        Reshape443,
        [((1, 80, 28, 40), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v6_yolov6n_obj_det_torchhub",
                "pt_yolo_v6_yolov6s_obj_det_torchhub",
                "pt_yolo_v6_yolov6l_obj_det_torchhub",
                "pt_yolo_v6_yolov6m_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 80, 1120)"},
        },
    ),
    (
        Reshape444,
        [((1, 80, 14, 20), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v6_yolov6n_obj_det_torchhub",
                "pt_yolo_v6_yolov6s_obj_det_torchhub",
                "pt_yolo_v6_yolov6l_obj_det_torchhub",
                "pt_yolo_v6_yolov6m_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 80, 280)"},
        },
    ),
    (
        Reshape445,
        [((1, 85, 52, 52), torch.float32)],
        {
            "model_names": ["pt_yolox_yolox_tiny_obj_det_torchhub", "pt_yolox_yolox_nano_obj_det_torchhub"],
            "pcc": 0.99,
            "args": {"shape": "(1, 85, 2704, 1)"},
        },
    ),
    (
        Reshape446,
        [((1, 85, 26, 26), torch.float32)],
        {
            "model_names": ["pt_yolox_yolox_tiny_obj_det_torchhub", "pt_yolox_yolox_nano_obj_det_torchhub"],
            "pcc": 0.99,
            "args": {"shape": "(1, 85, 676, 1)"},
        },
    ),
    (
        Reshape447,
        [((1, 85, 13, 13), torch.float32)],
        {
            "model_names": ["pt_yolox_yolox_tiny_obj_det_torchhub", "pt_yolox_yolox_nano_obj_det_torchhub"],
            "pcc": 0.99,
            "args": {"shape": "(1, 85, 169, 1)"},
        },
    ),
    (
        Reshape448,
        [((8, 100, 100), torch.float32)],
        {
            "model_names": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(8, 100, 100)"},
        },
    ),
    (
        Reshape449,
        [((100, 92), torch.float32)],
        {
            "model_names": ["onnx_detr_facebook_detr_resnet_50_obj_det_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 100, 92)"},
        },
    ),
    (
        Reshape450,
        [((1, 768, 256), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 768, 16, 16)"},
        },
    ),
    (
        Reshape451,
        [((1, 768, 1024), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 768, 32, 32)"},
        },
    ),
    (
        Reshape452,
        [((1, 768, 4096), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 768, 64, 64)"},
        },
    ),
    (
        Reshape453,
        [((1, 768, 16384), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 768, 128, 128)"},
        },
    ),
    (
        Reshape454,
        [((1, 11, 768), torch.float32)],
        {
            "model_names": [
                "pd_bert_chinese_roberta_base_qa_padlenlp",
                "pd_bert_chinese_roberta_base_seq_cls_padlenlp",
                "pd_roberta_rbt4_ch_clm_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"shape": "(11, 768)"},
        },
    ),
    (
        Reshape455,
        [((1, 11, 768), torch.float32)],
        {
            "model_names": [
                "pd_bert_chinese_roberta_base_qa_padlenlp",
                "pd_bert_chinese_roberta_base_seq_cls_padlenlp",
                "pd_roberta_rbt4_ch_clm_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 11, 12, 64)"},
        },
    ),
    (
        Reshape456,
        [((11, 768), torch.float32)],
        {
            "model_names": [
                "pd_bert_chinese_roberta_base_qa_padlenlp",
                "pd_bert_chinese_roberta_base_seq_cls_padlenlp",
                "pd_roberta_rbt4_ch_clm_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 11, 768)"},
        },
    ),
    (
        Reshape457,
        [((1, 12, 11, 64), torch.float32)],
        {
            "model_names": [
                "pd_bert_chinese_roberta_base_qa_padlenlp",
                "pd_bert_chinese_roberta_base_seq_cls_padlenlp",
                "pd_roberta_rbt4_ch_clm_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"shape": "(12, 11, 64)"},
        },
    ),
    (
        Reshape458,
        [((1, 12, 64, 11), torch.float32)],
        {
            "model_names": [
                "pd_bert_chinese_roberta_base_qa_padlenlp",
                "pd_bert_chinese_roberta_base_seq_cls_padlenlp",
                "pd_roberta_rbt4_ch_clm_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"shape": "(12, 64, 11)"},
        },
    ),
    (
        Reshape459,
        [((12, 11, 11), torch.float32)],
        {
            "model_names": [
                "pd_bert_chinese_roberta_base_qa_padlenlp",
                "pd_albert_chinese_tiny_mlm_padlenlp",
                "pd_bert_chinese_roberta_base_seq_cls_padlenlp",
                "pd_roberta_rbt4_ch_clm_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 12, 11, 11)"},
        },
    ),
    (
        Reshape460,
        [((1, 12, 11, 11), torch.float32)],
        {
            "model_names": [
                "pd_bert_chinese_roberta_base_qa_padlenlp",
                "pd_albert_chinese_tiny_mlm_padlenlp",
                "pd_bert_chinese_roberta_base_seq_cls_padlenlp",
                "pd_roberta_rbt4_ch_clm_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"shape": "(12, 11, 11)"},
        },
    ),
    (
        Reshape461,
        [((12, 11, 64), torch.float32)],
        {
            "model_names": [
                "pd_bert_chinese_roberta_base_qa_padlenlp",
                "pd_bert_chinese_roberta_base_seq_cls_padlenlp",
                "pd_roberta_rbt4_ch_clm_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 12, 11, 64)"},
        },
    ),
    (
        Reshape454,
        [((1, 11, 12, 64), torch.float32)],
        {
            "model_names": [
                "pd_bert_chinese_roberta_base_qa_padlenlp",
                "pd_bert_chinese_roberta_base_seq_cls_padlenlp",
                "pd_roberta_rbt4_ch_clm_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"shape": "(11, 768)"},
        },
    ),
    (
        Reshape462,
        [((256, 1024), torch.float32)],
        {
            "model_names": [
                "pt_bart_facebook_bart_large_mnli_seq_cls_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
                "pt_opt_facebook_opt_350m_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 256, 1024)"},
        },
    ),
    (
        Reshape463,
        [((256, 1024), torch.float32)],
        {
            "model_names": ["pt_codegen_salesforce_codegen_350m_nl_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 256, 4, 256)"},
        },
    ),
    (
        Reshape464,
        [((1, 16, 256, 64), torch.float32)],
        {
            "model_names": [
                "pt_bart_facebook_bart_large_mnli_seq_cls_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
                "pt_opt_facebook_opt_350m_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(16, 256, 64)"},
        },
    ),
    (
        Reshape465,
        [((16, 256, 256), torch.float32)],
        {
            "model_names": [
                "pt_bart_facebook_bart_large_mnli_seq_cls_hf",
                "pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf",
                "pt_xglm_facebook_xglm_1_7b_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
                "pt_opt_facebook_opt_350m_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 16, 256, 256)"},
        },
    ),
    (
        Reshape466,
        [((1, 16, 256, 256), torch.float32)],
        {
            "model_names": [
                "pt_bart_facebook_bart_large_mnli_seq_cls_hf",
                "pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf",
                "pt_xglm_facebook_xglm_1_7b_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
                "pt_opt_facebook_opt_350m_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(16, 256, 256)"},
        },
    ),
    (
        Reshape467,
        [((16, 256, 64), torch.float32)],
        {
            "model_names": [
                "pt_bart_facebook_bart_large_mnli_seq_cls_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
                "pt_opt_facebook_opt_350m_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 16, 256, 64)"},
        },
    ),
    (
        Reshape95,
        [((1, 256, 16, 64), torch.float32)],
        {
            "model_names": [
                "pt_bart_facebook_bart_large_mnli_seq_cls_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
                "pt_opt_facebook_opt_350m_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(256, 1024)"},
        },
    ),
    (
        Reshape468,
        [((1, 256), torch.int64)],
        {
            "model_names": [
                "pt_bart_facebook_bart_large_mnli_seq_cls_hf",
                "pt_xglm_facebook_xglm_1_7b_clm_hf",
                "pt_opt_facebook_opt_350m_clm_hf",
                "pt_gpt_gpt2_text_gen_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 256)"},
        },
    ),
    (
        Reshape469,
        [((1, 128, 1024), torch.float32)],
        {
            "model_names": [
                "pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf",
                "pt_albert_large_v1_token_cls_hf",
                "pt_albert_large_v2_mlm_hf",
                "pt_albert_large_v2_token_cls_hf",
                "pt_albert_large_v1_mlm_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(128, 1024)"},
        },
    ),
    (
        Reshape470,
        [((1, 128, 1024), torch.float32)],
        {
            "model_names": [
                "pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf",
                "pt_albert_large_v1_token_cls_hf",
                "pt_albert_large_v2_mlm_hf",
                "pt_albert_large_v2_token_cls_hf",
                "pt_albert_large_v1_mlm_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 128, 16, 64)"},
        },
    ),
    (
        Reshape471,
        [((128, 1024), torch.float32)],
        {
            "model_names": [
                "pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf",
                "pt_albert_large_v1_token_cls_hf",
                "pt_albert_large_v2_mlm_hf",
                "pt_albert_large_v2_token_cls_hf",
                "pt_albert_large_v1_mlm_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 128, 1024)"},
        },
    ),
    (
        Reshape472,
        [((1, 16, 128, 64), torch.float32)],
        {
            "model_names": [
                "pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf",
                "pt_albert_large_v1_token_cls_hf",
                "pt_albert_large_v2_mlm_hf",
                "pt_albert_large_v2_token_cls_hf",
                "pt_albert_large_v1_mlm_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(16, 128, 64)"},
        },
    ),
    (
        Reshape473,
        [((16, 128, 128), torch.float32)],
        {
            "model_names": [
                "pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf",
                "pt_albert_large_v1_token_cls_hf",
                "pt_albert_large_v2_mlm_hf",
                "pt_albert_large_v2_token_cls_hf",
                "pt_albert_xlarge_v1_mlm_hf",
                "pt_albert_large_v1_mlm_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 16, 128, 128)"},
        },
    ),
    (
        Reshape474,
        [((1, 16, 128, 128), torch.float32)],
        {
            "model_names": [
                "pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf",
                "pt_albert_large_v1_token_cls_hf",
                "pt_albert_large_v2_mlm_hf",
                "pt_albert_large_v2_token_cls_hf",
                "pt_albert_xlarge_v1_mlm_hf",
                "pt_albert_large_v1_mlm_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(16, 128, 128)"},
        },
    ),
    (
        Reshape475,
        [((1, 16, 64, 128), torch.float32)],
        {
            "model_names": [
                "pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf",
                "pt_albert_large_v1_token_cls_hf",
                "pt_albert_large_v2_mlm_hf",
                "pt_albert_large_v2_token_cls_hf",
                "pt_albert_large_v1_mlm_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(16, 64, 128)"},
        },
    ),
    (
        Reshape476,
        [((16, 128, 64), torch.float32)],
        {
            "model_names": [
                "pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf",
                "pt_albert_large_v1_token_cls_hf",
                "pt_albert_large_v2_mlm_hf",
                "pt_albert_large_v2_token_cls_hf",
                "pt_albert_large_v1_mlm_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 16, 128, 64)"},
        },
    ),
    (
        Reshape469,
        [((1, 128, 16, 64), torch.float32)],
        {
            "model_names": ["pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(128, 1024)"},
        },
    ),
    (
        Reshape477,
        [((1, 128, 16, 64), torch.float32)],
        {
            "model_names": [
                "pt_albert_large_v1_token_cls_hf",
                "pt_albert_large_v2_mlm_hf",
                "pt_albert_large_v2_token_cls_hf",
                "pt_albert_large_v1_mlm_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 128, 1024, 1)"},
        },
    ),
    (
        Reshape478,
        [((144, 1, 5, 5), torch.float32)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
            ],
            "pcc": 0.99,
            "args": {"shape": "(144, 1, 5, 5)"},
        },
    ),
    (
        Reshape479,
        [((240, 1, 5, 5), torch.float32)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
            ],
            "pcc": 0.99,
            "args": {"shape": "(240, 1, 5, 5)"},
        },
    ),
    (
        Reshape480,
        [((240, 1, 3, 3), torch.float32)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"shape": "(240, 1, 3, 3)"},
        },
    ),
    (
        Reshape481,
        [((480, 1, 3, 3), torch.float32)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"shape": "(480, 1, 3, 3)"},
        },
    ),
    (
        Reshape482,
        [((480, 1, 5, 5), torch.float32)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"shape": "(480, 1, 5, 5)"},
        },
    ),
    (
        Reshape483,
        [((672, 1, 5, 5), torch.float32)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"shape": "(672, 1, 5, 5)"},
        },
    ),
    (
        Reshape484,
        [((1152, 1, 5, 5), torch.float32)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1152, 1, 5, 5)"},
        },
    ),
    (
        Reshape485,
        [((1152, 1, 3, 3), torch.float32)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1152, 1, 3, 3)"},
        },
    ),
    (
        Reshape486,
        [((336, 1, 5, 5), torch.float32)],
        {
            "model_names": [
                "pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"shape": "(336, 1, 5, 5)"},
        },
    ),
    (
        Reshape487,
        [((336, 1, 3, 3), torch.float32)],
        {
            "model_names": [
                "pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"shape": "(336, 1, 3, 3)"},
        },
    ),
    (
        Reshape488,
        [((672, 1, 3, 3), torch.float32)],
        {
            "model_names": [
                "pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"shape": "(672, 1, 3, 3)"},
        },
    ),
    (
        Reshape489,
        [((960, 1, 5, 5), torch.float32)],
        {
            "model_names": [
                "pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"shape": "(960, 1, 5, 5)"},
        },
    ),
    (
        Reshape490,
        [((1632, 1, 5, 5), torch.float32)],
        {
            "model_names": [
                "pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1632, 1, 5, 5)"},
        },
    ),
    (
        Reshape491,
        [((1632, 1, 3, 3), torch.float32)],
        {
            "model_names": [
                "pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1632, 1, 3, 3)"},
        },
    ),
    (
        Reshape492,
        [((8, 1, 3, 3), torch.float32)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"shape": "(8, 1, 3, 3)"},
        },
    ),
    (
        Reshape493,
        [((24, 1, 3, 3), torch.float32)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"shape": "(24, 1, 3, 3)"},
        },
    ),
    (
        Reshape494,
        [((48, 1, 3, 3), torch.float32)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"shape": "(48, 1, 3, 3)"},
        },
    ),
    (
        Reshape495,
        [((12, 1, 3, 3), torch.float32)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"shape": "(12, 1, 3, 3)"},
        },
    ),
    (
        Reshape496,
        [((16, 1, 3, 3), torch.float32)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_yolox_yolox_nano_obj_det_torchhub",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
            ],
            "pcc": 0.99,
            "args": {"shape": "(16, 1, 3, 3)"},
        },
    ),
    (
        Reshape497,
        [((36, 1, 3, 3), torch.float32)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"shape": "(36, 1, 3, 3)"},
        },
    ),
    (
        Reshape498,
        [((72, 1, 5, 5), torch.float32)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"shape": "(72, 1, 5, 5)"},
        },
    ),
    (
        Reshape499,
        [((20, 1, 3, 3), torch.float32)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"shape": "(20, 1, 3, 3)"},
        },
    ),
    (
        Reshape500,
        [((24, 1, 5, 5), torch.float32)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"shape": "(24, 1, 5, 5)"},
        },
    ),
    (
        Reshape501,
        [((60, 1, 3, 3), torch.float32)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"shape": "(60, 1, 3, 3)"},
        },
    ),
    (
        Reshape502,
        [((120, 1, 3, 3), torch.float32)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"shape": "(120, 1, 3, 3)"},
        },
    ),
    (
        Reshape503,
        [((40, 1, 3, 3), torch.float32)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"shape": "(40, 1, 3, 3)"},
        },
    ),
    (
        Reshape504,
        [((100, 1, 3, 3), torch.float32)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"shape": "(100, 1, 3, 3)"},
        },
    ),
    (
        Reshape505,
        [((92, 1, 3, 3), torch.float32)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"shape": "(92, 1, 3, 3)"},
        },
    ),
    (
        Reshape506,
        [((56, 1, 3, 3), torch.float32)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"shape": "(56, 1, 3, 3)"},
        },
    ),
    (
        Reshape507,
        [((80, 1, 3, 3), torch.float32)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"shape": "(80, 1, 3, 3)"},
        },
    ),
    (
        Reshape508,
        [((112, 1, 5, 5), torch.float32)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"shape": "(112, 1, 5, 5)"},
        },
    ),
    (
        Reshape509,
        [((7, 2), torch.float32)],
        {
            "model_names": ["pt_gpt_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(7, 2)"},
        },
    ),
    (
        Reshape510,
        [((1, 2), torch.float32)],
        {
            "model_names": [
                "pt_gpt_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
                "pt_opt_facebook_opt_1_3b_seq_cls_hf",
                "pt_opt_facebook_opt_350m_seq_cls_hf",
                "pt_opt_facebook_opt_125m_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 2)"},
        },
    ),
    (
        Reshape511,
        [((1, 3, 256, 256), torch.float32)],
        {"model_names": ["pt_mlp_mixer_base_img_cls_github"], "pcc": 0.99, "args": {"shape": "(1, 3, 16, 16, 16, 16)"}},
    ),
    (
        Reshape317,
        [((1, 16, 16, 16, 16, 3), torch.float32)],
        {"model_names": ["pt_mlp_mixer_base_img_cls_github"], "pcc": 0.99, "args": {"shape": "(256, 768)"}},
    ),
    (
        Reshape512,
        [((1024, 256, 1), torch.float32)],
        {"model_names": ["pt_mlp_mixer_base_img_cls_github"], "pcc": 0.99, "args": {"shape": "(1024, 256, 1, 1)"}},
    ),
    (
        Reshape249,
        [((1, 1024, 512, 1), torch.float32)],
        {"model_names": ["pt_mlp_mixer_base_img_cls_github"], "pcc": 0.99, "args": {"shape": "(1, 1024, 512)"}},
    ),
    (
        Reshape513,
        [((1, 1024, 512), torch.float32)],
        {"model_names": ["pt_mlp_mixer_base_img_cls_github"], "pcc": 0.99, "args": {"shape": "(1, 1024, 512, 1)"}},
    ),
    (
        Reshape514,
        [((256, 1024, 1), torch.float32)],
        {"model_names": ["pt_mlp_mixer_base_img_cls_github"], "pcc": 0.99, "args": {"shape": "(256, 1024, 1, 1)"}},
    ),
    (
        Reshape76,
        [((1, 256, 512, 1), torch.float32)],
        {"model_names": ["pt_mlp_mixer_base_img_cls_github"], "pcc": 0.99, "args": {"shape": "(1, 256, 512)"}},
    ),
    (
        Reshape515,
        [((1, 512, 14, 14), torch.float32)],
        {
            "model_names": ["pt_mlp_mixer_mixer_s16_224_img_cls_timm"],
            "pcc": 0.99,
            "args": {"shape": "(1, 512, 196, 1)"},
        },
    ),
    (
        Reshape516,
        [((72, 1, 3, 3), torch.float32)],
        {
            "model_names": [
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
            ],
            "pcc": 0.99,
            "args": {"shape": "(72, 1, 3, 3)"},
        },
    ),
    (
        Reshape517,
        [((120, 1, 5, 5), torch.float32)],
        {
            "model_names": [
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
            ],
            "pcc": 0.99,
            "args": {"shape": "(120, 1, 5, 5)"},
        },
    ),
    (
        Reshape518,
        [((200, 1, 3, 3), torch.float32)],
        {
            "model_names": [
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"shape": "(200, 1, 3, 3)"},
        },
    ),
    (
        Reshape519,
        [((184, 1, 3, 3), torch.float32)],
        {
            "model_names": [
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"shape": "(184, 1, 3, 3)"},
        },
    ),
    (
        Reshape520,
        [((1, 960, 1, 1), torch.float32)],
        {
            "model_names": ["pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub"],
            "pcc": 0.99,
            "args": {"shape": "(1, 960, 1, 1)"},
        },
    ),
    (
        Reshape521,
        [((1, 32, 2048), torch.float32)],
        {
            "model_names": ["pt_opt_facebook_opt_1_3b_seq_cls_hf", "pt_opt_facebook_opt_1_3b_qa_hf"],
            "pcc": 0.99,
            "args": {"shape": "(32, 2048)"},
        },
    ),
    (
        Reshape522,
        [((1, 32, 2048), torch.float32)],
        {
            "model_names": ["pt_opt_facebook_opt_1_3b_seq_cls_hf", "pt_opt_facebook_opt_1_3b_qa_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 32, 32, 64)"},
        },
    ),
    (
        Reshape523,
        [((32, 2048), torch.float32)],
        {
            "model_names": ["pt_opt_facebook_opt_1_3b_seq_cls_hf", "pt_opt_facebook_opt_1_3b_qa_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 32, 2048)"},
        },
    ),
    (
        Reshape524,
        [((1, 32, 32, 64), torch.float32)],
        {
            "model_names": [
                "pt_opt_facebook_opt_1_3b_seq_cls_hf",
                "pt_opt_facebook_opt_1_3b_qa_hf",
                "pt_swin_swin_v2_b_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(32, 32, 64)"},
        },
    ),
    (
        Reshape521,
        [((1, 32, 32, 64), torch.float32)],
        {
            "model_names": ["pt_opt_facebook_opt_1_3b_seq_cls_hf", "pt_opt_facebook_opt_1_3b_qa_hf"],
            "pcc": 0.99,
            "args": {"shape": "(32, 2048)"},
        },
    ),
    (
        Reshape525,
        [((32, 32, 32), torch.float32)],
        {
            "model_names": ["pt_opt_facebook_opt_1_3b_seq_cls_hf", "pt_opt_facebook_opt_1_3b_qa_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 32, 32, 32)"},
        },
    ),
    (
        Reshape526,
        [((1, 32, 32, 32), torch.float32)],
        {
            "model_names": ["pt_opt_facebook_opt_1_3b_seq_cls_hf", "pt_opt_facebook_opt_1_3b_qa_hf"],
            "pcc": 0.99,
            "args": {"shape": "(32, 32, 32)"},
        },
    ),
    (
        Reshape522,
        [((32, 32, 64), torch.float32)],
        {
            "model_names": ["pt_opt_facebook_opt_1_3b_seq_cls_hf", "pt_opt_facebook_opt_1_3b_qa_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 32, 32, 64)"},
        },
    ),
    (
        Reshape527,
        [((32, 2), torch.float32)],
        {
            "model_names": [
                "pt_opt_facebook_opt_1_3b_seq_cls_hf",
                "pt_opt_facebook_opt_350m_seq_cls_hf",
                "pt_opt_facebook_opt_125m_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(32, 2)"},
        },
    ),
    (
        Reshape528,
        [((1, 512, 261), torch.float32)],
        {
            "model_names": ["pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 512, 1, 261)"},
        },
    ),
    (
        Reshape529,
        [((1, 224, 224, 3), torch.float32)],
        {
            "model_names": ["pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 50176, 3)"},
        },
    ),
    (
        Reshape530,
        [((1, 50176, 261), torch.float32)],
        {
            "model_names": ["pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(50176, 261)"},
        },
    ),
    (
        Reshape531,
        [((1, 50176, 261), torch.float32)],
        {
            "model_names": ["pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 50176, 1, 261)"},
        },
    ),
    (
        Reshape532,
        [((50176, 261), torch.float32)],
        {
            "model_names": ["pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 50176, 261)"},
        },
    ),
    (
        Reshape533,
        [((1, 512, 50176), torch.float32)],
        {
            "model_names": [
                "pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 1, 512, 50176)"},
        },
    ),
    (
        Reshape534,
        [((1, 1, 512, 50176), torch.float32)],
        {
            "model_names": [
                "pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 512, 50176)"},
        },
    ),
    (
        Reshape535,
        [((1, 1, 261, 50176), torch.float32)],
        {
            "model_names": ["pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 261, 50176)"},
        },
    ),
    (
        Reshape536,
        [((1, 1008, 1, 1), torch.float32)],
        {
            "model_names": ["pt_regnet_regnet_x_3_2gf_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(1, 1008, 1, 1)"},
        },
    ),
    (
        Reshape537,
        [((1, 784, 1, 1), torch.float32)],
        {
            "model_names": ["pt_regnet_regnet_y_800mf_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(1, 784, 1, 1)"},
        },
    ),
    (
        Reshape538,
        [((1, 768, 128), torch.float32)],
        {
            "model_names": ["pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 12, 64, 128)"},
        },
    ),
    (
        Reshape117,
        [((1, 768, 128), torch.float32)],
        {
            "model_names": ["pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(12, 64, 128)"},
        },
    ),
    (
        Reshape539,
        [((768, 768, 1), torch.float32)],
        {
            "model_names": ["pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(768, 768, 1, 1)"},
        },
    ),
    (
        Reshape540,
        [((1, 768, 128, 1), torch.float32)],
        {
            "model_names": ["pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 768, 128)"},
        },
    ),
    (
        Reshape131,
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
        Reshape541,
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
        Reshape542,
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
        Reshape543,
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
        Reshape544,
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
        Reshape545,
        [((1, 12, 64, 1), torch.float32)],
        {
            "model_names": [
                "pt_t5_t5_base_text_gen_hf",
                "pt_whisper_openai_whisper_small_speech_recognition_hf",
                "pt_t5_google_flan_t5_base_text_gen_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(12, 64, 1)"},
        },
    ),
    (
        Reshape546,
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
        Reshape130,
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
        Reshape547,
        [((1, 61, 768), torch.float32)],
        {
            "model_names": ["pt_t5_t5_base_text_gen_hf", "pt_t5_google_flan_t5_base_text_gen_hf"],
            "pcc": 0.99,
            "args": {"shape": "(61, 768)"},
        },
    ),
    (
        Reshape548,
        [((61, 768), torch.float32)],
        {
            "model_names": ["pt_t5_t5_base_text_gen_hf", "pt_t5_google_flan_t5_base_text_gen_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 61, 12, 64)"},
        },
    ),
    (
        Reshape549,
        [((61, 768), torch.float32)],
        {
            "model_names": ["pt_t5_t5_base_text_gen_hf", "pt_t5_google_flan_t5_base_text_gen_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 61, 768)"},
        },
    ),
    (
        Reshape550,
        [((1, 12, 61, 64), torch.float32)],
        {
            "model_names": ["pt_t5_t5_base_text_gen_hf", "pt_t5_google_flan_t5_base_text_gen_hf"],
            "pcc": 0.99,
            "args": {"shape": "(12, 61, 64)"},
        },
    ),
    (
        Reshape551,
        [((12, 61, 61), torch.float32)],
        {
            "model_names": ["pt_t5_t5_base_text_gen_hf", "pt_t5_google_flan_t5_base_text_gen_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 12, 61, 61)"},
        },
    ),
    (
        Reshape552,
        [((1, 12, 61, 61), torch.float32)],
        {
            "model_names": ["pt_t5_t5_base_text_gen_hf", "pt_t5_google_flan_t5_base_text_gen_hf"],
            "pcc": 0.99,
            "args": {"shape": "(12, 61, 61)"},
        },
    ),
    (
        Reshape553,
        [((1, 12, 64, 61), torch.float32)],
        {
            "model_names": ["pt_t5_t5_base_text_gen_hf", "pt_t5_google_flan_t5_base_text_gen_hf"],
            "pcc": 0.99,
            "args": {"shape": "(12, 64, 61)"},
        },
    ),
    (
        Reshape554,
        [((12, 61, 64), torch.float32)],
        {
            "model_names": ["pt_t5_t5_base_text_gen_hf", "pt_t5_google_flan_t5_base_text_gen_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 12, 61, 64)"},
        },
    ),
    (
        Reshape547,
        [((1, 61, 12, 64), torch.float32)],
        {
            "model_names": ["pt_t5_t5_base_text_gen_hf", "pt_t5_google_flan_t5_base_text_gen_hf"],
            "pcc": 0.99,
            "args": {"shape": "(61, 768)"},
        },
    ),
    (
        Reshape555,
        [((12, 1, 61), torch.float32)],
        {
            "model_names": ["pt_t5_t5_base_text_gen_hf", "pt_t5_google_flan_t5_base_text_gen_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 12, 1, 61)"},
        },
    ),
    (
        Reshape556,
        [((1, 12, 1, 61), torch.float32)],
        {
            "model_names": ["pt_t5_t5_base_text_gen_hf", "pt_t5_google_flan_t5_base_text_gen_hf"],
            "pcc": 0.99,
            "args": {"shape": "(12, 1, 61)"},
        },
    ),
    (
        Reshape557,
        [((1, 4096, 1, 1), torch.float32)],
        {"model_names": ["pt_vgg_vgg19_bn_obj_det_timm"], "pcc": 0.99, "args": {"shape": "(1, 4096, 1, 1)"}},
    ),
    (
        Reshape136,
        [((197, 1, 768), torch.float32)],
        {"model_names": ["pt_vit_vit_b_16_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(197, 768)"}},
    ),
    (
        Reshape558,
        [((197, 2304), torch.float32)],
        {"model_names": ["pt_vit_vit_b_16_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(197, 1, 2304)"}},
    ),
    (
        Reshape559,
        [((197, 1, 2304), torch.float32)],
        {"model_names": ["pt_vit_vit_b_16_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(197, 1, 3, 768)"}},
    ),
    (
        Reshape560,
        [((1, 197, 1, 768), torch.float32)],
        {"model_names": ["pt_vit_vit_b_16_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(197, 12, 64)"}},
    ),
    (
        Reshape136,
        [((197, 1, 12, 64), torch.float32)],
        {"model_names": ["pt_vit_vit_b_16_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(197, 768)"}},
    ),
    (
        Reshape561,
        [((160, 1, 3, 3), torch.float32)],
        {
            "model_names": [
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "args": {"shape": "(160, 1, 3, 3)"},
        },
    ),
    (
        Reshape562,
        [((224, 1, 3, 3), torch.float32)],
        {
            "model_names": [
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "args": {"shape": "(224, 1, 3, 3)"},
        },
    ),
    (
        Reshape563,
        [((768, 80, 3), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(768, 80, 3, 1)"},
        },
    ),
    (
        Reshape564,
        [((1, 768, 3000, 1), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 768, 3000)"},
        },
    ),
    (
        Reshape565,
        [((1, 768, 3000), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 768, 3000, 1)"},
        },
    ),
    (
        Reshape566,
        [((768, 768, 3), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(768, 768, 3, 1)"},
        },
    ),
    (
        Reshape567,
        [((1, 768, 1500, 1), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 768, 1500)"},
        },
    ),
    (
        Reshape568,
        [((1, 1500, 768), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1500, 768)"},
        },
    ),
    (
        Reshape569,
        [((1, 1500, 768), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 1500, 12, 64)"},
        },
    ),
    (
        Reshape570,
        [((1500, 768), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 1500, 768)"},
        },
    ),
    (
        Reshape569,
        [((1500, 768), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 1500, 12, 64)"},
        },
    ),
    (
        Reshape571,
        [((1, 12, 1500, 64), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(12, 1500, 64)"},
        },
    ),
    (
        Reshape572,
        [((12, 1500, 1500), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 12, 1500, 1500)"},
        },
    ),
    (
        Reshape573,
        [((1, 12, 1500, 1500), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(12, 1500, 1500)"},
        },
    ),
    (
        Reshape574,
        [((1, 12, 64, 1500), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(12, 64, 1500)"},
        },
    ),
    (
        Reshape575,
        [((12, 1500, 64), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 12, 1500, 64)"},
        },
    ),
    (
        Reshape568,
        [((1, 1500, 12, 64), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1500, 768)"},
        },
    ),
    (
        Reshape576,
        [((12, 1, 1500), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 12, 1, 1500)"},
        },
    ),
    (
        Reshape577,
        [((1, 12, 1, 1500), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(12, 1, 1500)"},
        },
    ),
    (
        Reshape578,
        [((1, 85, 80, 80), torch.float32)],
        {
            "model_names": [
                "pt_yolox_yolox_x_obj_det_torchhub",
                "pt_yolox_yolox_m_obj_det_torchhub",
                "pt_yolox_yolox_s_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 85, 6400, 1)"},
        },
    ),
    (
        Reshape579,
        [((1, 85, 40, 40), torch.float32)],
        {
            "model_names": [
                "pt_yolox_yolox_x_obj_det_torchhub",
                "pt_yolox_yolox_m_obj_det_torchhub",
                "pt_yolox_yolox_s_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 85, 1600, 1)"},
        },
    ),
    (
        Reshape580,
        [((1, 85, 20, 20), torch.float32)],
        {
            "model_names": [
                "pt_yolox_yolox_x_obj_det_torchhub",
                "pt_yolox_yolox_m_obj_det_torchhub",
                "pt_yolox_yolox_s_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 85, 400, 1)"},
        },
    ),
    (
        Reshape0,
        [((1000,), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99, "args": {"shape": "(1, 1000)"}},
    ),
    (
        Reshape581,
        [((100, 251), torch.float32)],
        {
            "model_names": [
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
                "pt_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 100, 251)"},
        },
    ),
    (
        Reshape582,
        [((1, 100, 32, 107, 160), torch.float32)],
        {
            "model_names": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"shape": "(100, 32, 107, 160)"},
        },
    ),
    (
        Reshape583,
        [((1, 100, 32, 107, 160), torch.float32)],
        {
            "model_names": ["pt_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"shape": "(100, 1, 32, 107, 160)"},
        },
    ),
    (
        Reshape584,
        [((1, 100, 64, 54, 80), torch.float32)],
        {
            "model_names": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"shape": "(100, 64, 54, 80)"},
        },
    ),
    (
        Reshape585,
        [((1, 100, 64, 54, 80), torch.float32)],
        {
            "model_names": ["pt_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"shape": "(100, 1, 64, 54, 80)"},
        },
    ),
    (
        Reshape586,
        [((1, 100, 128, 27, 40), torch.float32)],
        {
            "model_names": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"shape": "(100, 128, 27, 40)"},
        },
    ),
    (
        Reshape587,
        [((1, 100, 128, 27, 40), torch.float32)],
        {
            "model_names": ["pt_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"shape": "(100, 1, 128, 27, 40)"},
        },
    ),
    (
        Reshape588,
        [((1, 100, 256, 14, 20), torch.float32)],
        {
            "model_names": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"shape": "(100, 256, 14, 20)"},
        },
    ),
    (
        Reshape589,
        [((1, 100, 256, 14, 20), torch.float32)],
        {
            "model_names": ["pt_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"shape": "(100, 1, 256, 14, 20)"},
        },
    ),
    (
        Reshape590,
        [((1, 256, 280), torch.float32)],
        {
            "model_names": [
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
                "pt_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 256, 14, 20)"},
        },
    ),
    (
        Reshape591,
        [((1, 100, 8, 280), torch.float32)],
        {
            "model_names": [
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
                "pt_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 100, 8, 14, 20)"},
        },
    ),
    (
        Reshape592,
        [((1, 100, 8, 14, 20), torch.float32)],
        {
            "model_names": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 100, 2240)"},
        },
    ),
    (
        Reshape593,
        [((1, 100, 8, 14, 20), torch.float32)],
        {
            "model_names": ["pt_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 100, 2240, 1, 1)"},
        },
    ),
    (
        Reshape594,
        [((1, 100, 8, 14, 20), torch.float32)],
        {
            "model_names": ["pt_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"shape": "(100, 1, 8, 14, 20)"},
        },
    ),
    (
        Reshape595,
        [((1, 100, 2240), torch.float32)],
        {
            "model_names": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"shape": "(100, 8, 14, 20)"},
        },
    ),
    (
        Reshape591,
        [((1, 100, 2240), torch.float32)],
        {
            "model_names": ["pt_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 100, 8, 14, 20)"},
        },
    ),
    (
        Reshape596,
        [((100, 264, 14, 20), torch.float32)],
        {
            "model_names": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"shape": "(100, 8, 9240)"},
        },
    ),
    (
        Reshape597,
        [((100, 264, 14, 20), torch.float32)],
        {
            "model_names": ["pt_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"shape": "(100, 8, 33, 280)"},
        },
    ),
    (
        Reshape598,
        [((100, 8, 9240), torch.float32)],
        {
            "model_names": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"shape": "(100, 264, 14, 20)"},
        },
    ),
    (
        Reshape599,
        [((100, 128, 14, 20), torch.float32)],
        {
            "model_names": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"shape": "(100, 8, 4480)"},
        },
    ),
    (
        Reshape600,
        [((100, 128, 14, 20), torch.float32)],
        {
            "model_names": ["pt_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"shape": "(100, 8, 16, 280)"},
        },
    ),
    (
        Reshape601,
        [((100, 8, 4480), torch.float32)],
        {
            "model_names": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"shape": "(100, 128, 14, 20)"},
        },
    ),
    (
        Reshape602,
        [((100, 64, 27, 40), torch.float32)],
        {
            "model_names": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"shape": "(100, 8, 8640)"},
        },
    ),
    (
        Reshape603,
        [((100, 64, 27, 40), torch.float32)],
        {
            "model_names": ["pt_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"shape": "(100, 8, 8, 1080)"},
        },
    ),
    (
        Reshape604,
        [((100, 8, 8640), torch.float32)],
        {
            "model_names": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"shape": "(100, 64, 27, 40)"},
        },
    ),
    (
        Reshape605,
        [((100, 32, 54, 80), torch.float32)],
        {
            "model_names": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"shape": "(100, 8, 17280)"},
        },
    ),
    (
        Reshape606,
        [((100, 32, 54, 80), torch.float32)],
        {
            "model_names": ["pt_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"shape": "(100, 8, 4, 4320)"},
        },
    ),
    (
        Reshape607,
        [((100, 8, 17280), torch.float32)],
        {
            "model_names": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"shape": "(100, 32, 54, 80)"},
        },
    ),
    (
        Reshape608,
        [((100, 16, 107, 160), torch.float32)],
        {
            "model_names": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"shape": "(100, 8, 34240)"},
        },
    ),
    (
        Reshape609,
        [((100, 16, 107, 160), torch.float32)],
        {
            "model_names": ["pt_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"shape": "(100, 8, 2, 17120)"},
        },
    ),
    (
        Reshape610,
        [((100, 8, 34240), torch.float32)],
        {
            "model_names": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"shape": "(100, 16, 107, 160)"},
        },
    ),
    (
        Reshape611,
        [((100, 1, 107, 160), torch.float32)],
        {
            "model_names": [
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
                "pt_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 100, 107, 160)"},
        },
    ),
    (
        Reshape612,
        [((729, 16), torch.float32)],
        {
            "model_names": ["pt_beit_microsoft_beit_large_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 27, 27, 16)"},
        },
    ),
    (
        Reshape613,
        [((1, 27, 27, 16), torch.float32)],
        {
            "model_names": ["pt_beit_microsoft_beit_large_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(729, 16)"},
        },
    ),
    (
        Reshape614,
        [((38809, 16), torch.float32)],
        {
            "model_names": ["pt_beit_microsoft_beit_large_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(197, 197, 16)"},
        },
    ),
    (
        Reshape615,
        [((1, 384, 1024), torch.float32)],
        {
            "model_names": [
                "pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf",
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                "onnx_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(384, 1024)"},
        },
    ),
    (
        Reshape616,
        [((1, 384, 1024), torch.float32)],
        {
            "model_names": [
                "pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf",
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                "onnx_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 384, 16, 64)"},
        },
    ),
    (
        Reshape617,
        [((384, 1024), torch.float32)],
        {
            "model_names": [
                "pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf",
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                "onnx_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 384, 1024)"},
        },
    ),
    (
        Reshape618,
        [((1, 16, 384, 64), torch.float32)],
        {
            "model_names": [
                "pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf",
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                "onnx_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(16, 384, 64)"},
        },
    ),
    (
        Reshape619,
        [((16, 384, 384), torch.float32)],
        {
            "model_names": [
                "pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf",
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                "onnx_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 16, 384, 384)"},
        },
    ),
    (
        Reshape620,
        [((1, 16, 384, 384), torch.float32)],
        {
            "model_names": [
                "pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf",
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                "onnx_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(16, 384, 384)"},
        },
    ),
    (
        Reshape621,
        [((1, 16, 64, 384), torch.float32)],
        {
            "model_names": [
                "pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf",
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                "onnx_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(16, 64, 384)"},
        },
    ),
    (
        Reshape622,
        [((16, 384, 64), torch.float32)],
        {
            "model_names": [
                "pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf",
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                "onnx_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 16, 384, 64)"},
        },
    ),
    (
        Reshape615,
        [((1, 384, 16, 64), torch.float32)],
        {
            "model_names": [
                "pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf",
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                "onnx_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(384, 1024)"},
        },
    ),
    (
        Reshape623,
        [((384, 1), torch.float32)],
        {
            "model_names": [
                "pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf",
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                "pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf",
                "onnx_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 384, 1)"},
        },
    ),
    (
        Reshape624,
        [((1, 192, 14, 14), torch.float32)],
        {
            "model_names": ["pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 192, 196, 1)"},
        },
    ),
    (
        Reshape625,
        [((1, 197, 192), torch.float32)],
        {
            "model_names": ["pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(197, 192)"},
        },
    ),
    (
        Reshape626,
        [((1, 197, 192), torch.float32)],
        {
            "model_names": ["pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 197, 3, 64)"},
        },
    ),
    (
        Reshape627,
        [((197, 192), torch.float32)],
        {
            "model_names": ["pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 197, 192)"},
        },
    ),
    (
        Reshape628,
        [((1, 3, 197, 64), torch.float32)],
        {
            "model_names": ["pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(3, 197, 64)"},
        },
    ),
    (
        Reshape629,
        [((3, 197, 197), torch.float32)],
        {
            "model_names": ["pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 3, 197, 197)"},
        },
    ),
    (
        Reshape630,
        [((1, 3, 197, 197), torch.float32)],
        {
            "model_names": ["pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(3, 197, 197)"},
        },
    ),
    (
        Reshape631,
        [((1, 3, 64, 197), torch.float32)],
        {
            "model_names": ["pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(3, 64, 197)"},
        },
    ),
    (
        Reshape632,
        [((3, 197, 64), torch.float32)],
        {
            "model_names": ["pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 3, 197, 64)"},
        },
    ),
    (
        Reshape625,
        [((1, 197, 3, 64), torch.float32)],
        {
            "model_names": ["pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(197, 192)"},
        },
    ),
    (
        Reshape633,
        [((1, 1, 192), torch.float32)],
        {
            "model_names": [
                "pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf",
                "pt_yolos_hustvl_yolos_tiny_obj_det_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 192)"},
        },
    ),
    (
        Reshape634,
        [((72, 1, 1, 5), torch.float32)],
        {
            "model_names": ["pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "args": {"shape": "(72, 1, 1, 5)"},
        },
    ),
    (
        Reshape635,
        [((72, 1, 5, 1), torch.float32)],
        {
            "model_names": ["pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "args": {"shape": "(72, 1, 5, 1)"},
        },
    ),
    (
        Reshape636,
        [((120, 1, 1, 5), torch.float32)],
        {
            "model_names": ["pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "args": {"shape": "(120, 1, 1, 5)"},
        },
    ),
    (
        Reshape637,
        [((120, 1, 5, 1), torch.float32)],
        {
            "model_names": ["pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "args": {"shape": "(120, 1, 5, 1)"},
        },
    ),
    (
        Reshape638,
        [((240, 1, 1, 5), torch.float32)],
        {
            "model_names": ["pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "args": {"shape": "(240, 1, 1, 5)"},
        },
    ),
    (
        Reshape639,
        [((240, 1, 5, 1), torch.float32)],
        {
            "model_names": ["pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "args": {"shape": "(240, 1, 5, 1)"},
        },
    ),
    (
        Reshape640,
        [((200, 1, 1, 5), torch.float32)],
        {
            "model_names": ["pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "args": {"shape": "(200, 1, 1, 5)"},
        },
    ),
    (
        Reshape641,
        [((200, 1, 5, 1), torch.float32)],
        {
            "model_names": ["pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "args": {"shape": "(200, 1, 5, 1)"},
        },
    ),
    (
        Reshape642,
        [((184, 1, 1, 5), torch.float32)],
        {
            "model_names": ["pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "args": {"shape": "(184, 1, 1, 5)"},
        },
    ),
    (
        Reshape643,
        [((184, 1, 5, 1), torch.float32)],
        {
            "model_names": ["pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "args": {"shape": "(184, 1, 5, 1)"},
        },
    ),
    (
        Reshape644,
        [((480, 1, 1, 5), torch.float32)],
        {
            "model_names": ["pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "args": {"shape": "(480, 1, 1, 5)"},
        },
    ),
    (
        Reshape645,
        [((480, 1, 5, 1), torch.float32)],
        {
            "model_names": ["pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "args": {"shape": "(480, 1, 5, 1)"},
        },
    ),
    (
        Reshape646,
        [((672, 1, 1, 5), torch.float32)],
        {
            "model_names": ["pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "args": {"shape": "(672, 1, 1, 5)"},
        },
    ),
    (
        Reshape647,
        [((672, 1, 5, 1), torch.float32)],
        {
            "model_names": ["pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "args": {"shape": "(672, 1, 5, 1)"},
        },
    ),
    (
        Reshape648,
        [((960, 1, 1, 5), torch.float32)],
        {
            "model_names": ["pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "args": {"shape": "(960, 1, 1, 5)"},
        },
    ),
    (
        Reshape649,
        [((960, 1, 5, 1), torch.float32)],
        {
            "model_names": ["pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "args": {"shape": "(960, 1, 5, 1)"},
        },
    ),
    (
        Reshape650,
        [((1, 256, 2048), torch.float32)],
        {
            "model_names": [
                "pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf",
                "pt_xglm_facebook_xglm_1_7b_clm_hf",
                "pt_phi1_microsoft_phi_1_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(256, 2048)"},
        },
    ),
    (
        Reshape651,
        [((1, 256, 2048), torch.float32)],
        {"model_names": ["pt_xglm_facebook_xglm_1_7b_clm_hf"], "pcc": 0.99, "args": {"shape": "(1, 256, 16, 128)"}},
    ),
    (
        Reshape652,
        [((1, 256, 2048), torch.float32)],
        {"model_names": ["pt_phi1_microsoft_phi_1_seq_cls_hf"], "pcc": 0.99, "args": {"shape": "(1, 256, 32, 64)"}},
    ),
    (
        Reshape651,
        [((256, 2048), torch.float32)],
        {
            "model_names": ["pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 256, 16, 128)"},
        },
    ),
    (
        Reshape653,
        [((256, 2048), torch.float32)],
        {
            "model_names": [
                "pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf",
                "pt_xglm_facebook_xglm_1_7b_clm_hf",
                "pt_phi1_microsoft_phi_1_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 256, 2048)"},
        },
    ),
    (
        Reshape652,
        [((256, 2048), torch.float32)],
        {
            "model_names": ["pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 256, 32, 64)"},
        },
    ),
    (
        Reshape654,
        [((256, 2048), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(1, 16, 16, 2048)"}},
    ),
    (
        Reshape655,
        [((1, 16, 256, 128), torch.float32)],
        {
            "model_names": ["pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf", "pt_xglm_facebook_xglm_1_7b_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(16, 256, 128)"},
        },
    ),
    (
        Reshape656,
        [((1, 16, 128, 256), torch.float32)],
        {"model_names": ["pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf"], "pcc": 0.99, "args": {"shape": "(16, 128, 256)"}},
    ),
    (
        Reshape657,
        [((16, 256, 128), torch.float32)],
        {
            "model_names": ["pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf", "pt_xglm_facebook_xglm_1_7b_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 16, 256, 128)"},
        },
    ),
    (
        Reshape650,
        [((1, 256, 16, 128), torch.float32)],
        {
            "model_names": ["pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf", "pt_xglm_facebook_xglm_1_7b_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(256, 2048)"},
        },
    ),
    (
        Reshape133,
        [((1, 64, 12, 12), torch.float32)],
        {"model_names": ["pt_mnist_base_img_cls_github"], "pcc": 0.99, "args": {"shape": "(1, 9216, 1, 1)"}},
    ),
    (
        Reshape658,
        [((1, 8, 2048, 32), torch.float32)],
        {
            "model_names": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(8, 2048, 32)"},
        },
    ),
    (
        Reshape659,
        [((1, 8, 256, 32), torch.float32)],
        {
            "model_names": [
                "pt_perceiverio_deepmind_language_perceiver_mlm_hf",
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(8, 256, 32)"},
        },
    ),
    (
        Reshape660,
        [((1, 2048, 768), torch.float32)],
        {
            "model_names": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(2048, 768)"},
        },
    ),
    (
        Reshape87,
        [((2048, 256), torch.float32)],
        {
            "model_names": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 2048, 256)"},
        },
    ),
    (
        Reshape661,
        [((8, 256, 2048), torch.float32)],
        {
            "model_names": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 8, 256, 2048)"},
        },
    ),
    (
        Reshape662,
        [((1, 8, 256, 2048), torch.float32)],
        {
            "model_names": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(8, 256, 2048)"},
        },
    ),
    (
        Reshape663,
        [((2048, 1280), torch.float32)],
        {
            "model_names": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 2048, 1280)"},
        },
    ),
    (
        Reshape664,
        [((1, 2048, 1280), torch.float32)],
        {
            "model_names": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 2048, 8, 160)"},
        },
    ),
    (
        Reshape665,
        [((1, 8, 160, 2048), torch.float32)],
        {
            "model_names": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(8, 160, 2048)"},
        },
    ),
    (
        Reshape666,
        [((8, 256, 160), torch.float32)],
        {
            "model_names": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 8, 256, 160)"},
        },
    ),
    (
        Reshape667,
        [((1, 256, 8, 160), torch.float32)],
        {
            "model_names": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(256, 1280)"},
        },
    ),
    (
        Reshape668,
        [((256, 1280), torch.float32)],
        {
            "model_names": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 256, 1280)"},
        },
    ),
    (
        Reshape667,
        [((1, 256, 1280), torch.float32)],
        {
            "model_names": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(256, 1280)"},
        },
    ),
    (
        Reshape669,
        [((1, 256, 1280), torch.float32)],
        {
            "model_names": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 256, 8, 160)"},
        },
    ),
    (
        Reshape92,
        [((256, 256), torch.float32)],
        {
            "model_names": [
                "pt_perceiverio_deepmind_language_perceiver_mlm_hf",
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 256, 256)"},
        },
    ),
    (
        Reshape670,
        [((1, 8, 160, 256), torch.float32)],
        {
            "model_names": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(8, 160, 256)"},
        },
    ),
    (
        Reshape671,
        [((8, 2048, 256), torch.float32)],
        {
            "model_names": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 8, 2048, 256)"},
        },
    ),
    (
        Reshape672,
        [((1, 8, 2048, 256), torch.float32)],
        {
            "model_names": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(8, 2048, 256)"},
        },
    ),
    (
        Reshape673,
        [((256, 768), torch.float32)],
        {
            "model_names": [
                "pt_perceiverio_deepmind_language_perceiver_mlm_hf",
                "pt_gptneo_eleutherai_gpt_neo_125m_clm_hf",
                "pt_gpt_gpt2_text_gen_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 256, 768)"},
        },
    ),
    (
        Reshape674,
        [((256, 768), torch.float32)],
        {
            "model_names": ["pt_gptneo_eleutherai_gpt_neo_125m_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 256, 12, 64)"},
        },
    ),
    (
        Reshape675,
        [((1, 256, 768), torch.float32)],
        {
            "model_names": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 256, 8, 96)"},
        },
    ),
    (
        Reshape317,
        [((1, 256, 768), torch.float32)],
        {
            "model_names": ["pt_gptneo_eleutherai_gpt_neo_125m_clm_hf", "pt_gpt_gpt2_text_gen_hf"],
            "pcc": 0.99,
            "args": {"shape": "(256, 768)"},
        },
    ),
    (
        Reshape674,
        [((1, 256, 768), torch.float32)],
        {"model_names": ["pt_gpt_gpt2_text_gen_hf"], "pcc": 0.99, "args": {"shape": "(1, 256, 12, 64)"}},
    ),
    (
        Reshape676,
        [((1, 8, 96, 256), torch.float32)],
        {
            "model_names": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(8, 96, 256)"},
        },
    ),
    (
        Reshape677,
        [((8, 2048, 96), torch.float32)],
        {
            "model_names": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 8, 2048, 96)"},
        },
    ),
    (
        Reshape660,
        [((1, 2048, 8, 96), torch.float32)],
        {
            "model_names": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(2048, 768)"},
        },
    ),
    (
        Reshape678,
        [((2048, 768), torch.float32)],
        {
            "model_names": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 2048, 768)"},
        },
    ),
    (
        Reshape679,
        [((2048, 262), torch.float32)],
        {
            "model_names": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 2048, 262)"},
        },
    ),
    (
        Reshape680,
        [((1, 6, 1024), torch.float32)],
        {"model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"], "pcc": 0.99, "args": {"shape": "(6, 1024)"}},
    ),
    (
        Reshape681,
        [((1, 6, 1024), torch.float32)],
        {"model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"], "pcc": 0.99, "args": {"shape": "(1, 6, 16, 64)"}},
    ),
    (
        Reshape682,
        [((6, 1024), torch.float32)],
        {"model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"], "pcc": 0.99, "args": {"shape": "(1, 6, 1024)"}},
    ),
    (
        Reshape683,
        [((1, 16, 6, 64), torch.float32)],
        {"model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"], "pcc": 0.99, "args": {"shape": "(16, 6, 64)"}},
    ),
    (
        Reshape684,
        [((16, 6, 6), torch.float32)],
        {"model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"], "pcc": 0.99, "args": {"shape": "(1, 16, 6, 6)"}},
    ),
    (
        Reshape685,
        [((1, 16, 6, 6), torch.float32)],
        {"model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"], "pcc": 0.99, "args": {"shape": "(16, 6, 6)"}},
    ),
    (
        Reshape686,
        [((1, 16, 64, 6), torch.float32)],
        {"model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"], "pcc": 0.99, "args": {"shape": "(16, 64, 6)"}},
    ),
    (
        Reshape687,
        [((16, 6, 64), torch.float32)],
        {"model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"], "pcc": 0.99, "args": {"shape": "(1, 16, 6, 64)"}},
    ),
    (
        Reshape680,
        [((1, 6, 16, 64), torch.float32)],
        {"model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"], "pcc": 0.99, "args": {"shape": "(6, 1024)"}},
    ),
    (
        Reshape688,
        [((6, 2816), torch.float32)],
        {"model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"], "pcc": 0.99, "args": {"shape": "(1, 6, 2816)"}},
    ),
    (
        Reshape689,
        [((1, 1296, 1, 1), torch.float32)],
        {
            "model_names": ["pt_regnet_facebook_regnet_y_064_img_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 1296, 1, 1)"},
        },
    ),
    (
        Reshape690,
        [((1, 672, 1, 1), torch.float32)],
        {
            "model_names": ["pt_regnet_regnet_x_800mf_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(1, 672, 1, 1)"},
        },
    ),
    (
        Reshape691,
        [((1, 32, 128, 128), torch.float32)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 32, 16384, 1)"},
        },
    ),
    (
        Reshape692,
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
        Reshape693,
        [((1, 16384, 32), torch.float32)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 16384, 1, 32)"},
        },
    ),
    (
        Reshape694,
        [((1, 16384, 32), torch.float32)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 128, 128, 32)"},
        },
    ),
    (
        Reshape695,
        [((1, 32, 16384), torch.float32)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 32, 128, 128)"},
        },
    ),
    (
        Reshape696,
        [((1, 32, 16, 16), torch.float32)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 32, 256)"},
        },
    ),
    (
        Reshape697,
        [((1, 256, 32), torch.float32)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(256, 32)"},
        },
    ),
    (
        Reshape698,
        [((1, 256, 32), torch.float32)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 256, 1, 32)"},
        },
    ),
    (
        Reshape699,
        [((256, 32), torch.float32)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 256, 32)"},
        },
    ),
    (
        Reshape696,
        [((1, 1, 32, 256), torch.float32)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 32, 256)"},
        },
    ),
    (
        Reshape700,
        [((1, 128, 16384), torch.float32)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 128, 128, 128)"},
        },
    ),
    (
        Reshape701,
        [((1, 128, 128, 128), torch.float32)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 128, 16384, 1)"},
        },
    ),
    (
        Reshape702,
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
        Reshape703,
        [((1, 64, 64, 64), torch.float32)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 64, 4096, 1)"},
        },
    ),
    (
        Reshape704,
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
        Reshape705,
        [((1, 4096, 64), torch.float32)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 4096, 2, 32)"},
        },
    ),
    (
        Reshape706,
        [((1, 4096, 64), torch.float32)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 64, 64, 64)"},
        },
    ),
    (
        Reshape707,
        [((1, 2, 4096, 32), torch.float32)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(2, 4096, 32)"},
        },
    ),
    (
        Reshape706,
        [((1, 64, 4096), torch.float32)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 64, 64, 64)"},
        },
    ),
    (
        Reshape708,
        [((1, 2, 256, 32), torch.float32)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(2, 256, 32)"},
        },
    ),
    (
        Reshape709,
        [((1, 2, 32, 256), torch.float32)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(2, 32, 256)"},
        },
    ),
    (
        Reshape710,
        [((2, 4096, 32), torch.float32)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 2, 4096, 32)"},
        },
    ),
    (
        Reshape711,
        [((1, 4096, 2, 32), torch.float32)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(4096, 64)"},
        },
    ),
    (
        Reshape712,
        [((4096, 64), torch.float32)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 4096, 64)"},
        },
    ),
    (
        Reshape713,
        [((1, 256, 64, 64), torch.float32)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 256, 4096, 1)"},
        },
    ),
    (
        Reshape714,
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
        Reshape715,
        [((1, 160, 32, 32), torch.float32)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 160, 1024, 1)"},
        },
    ),
    (
        Reshape716,
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
        Reshape717,
        [((1, 1024, 160), torch.float32)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 1024, 5, 32)"},
        },
    ),
    (
        Reshape718,
        [((1, 1024, 160), torch.float32)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 32, 32, 160)"},
        },
    ),
    (
        Reshape719,
        [((1, 5, 1024, 32), torch.float32)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(5, 1024, 32)"},
        },
    ),
    (
        Reshape720,
        [((1, 160, 1024), torch.float32)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 160, 32, 32)"},
        },
    ),
    (
        Reshape721,
        [((1, 160, 16, 16), torch.float32)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 160, 256)"},
        },
    ),
    (
        Reshape722,
        [((1, 256, 160), torch.float32)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(256, 160)"},
        },
    ),
    (
        Reshape723,
        [((1, 256, 160), torch.float32)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 256, 5, 32)"},
        },
    ),
    (
        Reshape724,
        [((256, 160), torch.float32)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 256, 160)"},
        },
    ),
    (
        Reshape725,
        [((1, 5, 256, 32), torch.float32)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(5, 256, 32)"},
        },
    ),
    (
        Reshape726,
        [((1, 5, 32, 256), torch.float32)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(5, 32, 256)"},
        },
    ),
    (
        Reshape727,
        [((5, 1024, 32), torch.float32)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 5, 1024, 32)"},
        },
    ),
    (
        Reshape728,
        [((1, 1024, 5, 32), torch.float32)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1024, 160)"},
        },
    ),
    (
        Reshape729,
        [((1024, 160), torch.float32)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 1024, 160)"},
        },
    ),
    (
        Reshape730,
        [((1, 640, 1024), torch.float32)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 640, 32, 32)"},
        },
    ),
    (
        Reshape731,
        [((640, 1, 3, 3), torch.float32)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(640, 1, 3, 3)"},
        },
    ),
    (
        Reshape732,
        [((1, 640, 32, 32), torch.float32)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 640, 1024, 1)"},
        },
    ),
    (
        Reshape733,
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
        Reshape734,
        [((1, 256, 16, 16), torch.float32)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 256, 256, 1)"},
        },
    ),
    (
        Reshape92,
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
        Reshape735,
        [((1, 8, 32, 256), torch.float32)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(8, 32, 256)"},
        },
    ),
    (
        Reshape736,
        [((8, 256, 32), torch.float32)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 8, 256, 32)"},
        },
    ),
    (
        Reshape91,
        [((1, 256, 8, 32), torch.float32)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(256, 256)"},
        },
    ),
    (
        Reshape737,
        [((1, 1024, 256), torch.float32)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 1024, 16, 16)"},
        },
    ),
    (
        Reshape738,
        [((1, 1024, 16, 16), torch.float32)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 1024, 256, 1)"},
        },
    ),
    (
        Reshape739,
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
        Reshape740,
        [((1, 96, 56, 56), torch.float32)],
        {
            "model_names": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 96, 3136, 1)"},
        },
    ),
    (
        Reshape741,
        [((1, 3136, 96), torch.float32)],
        {
            "model_names": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 8, 7, 8, 7, 96)"},
        },
    ),
    (
        Reshape742,
        [((1, 3136, 96), torch.float32)],
        {
            "model_names": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 56, 56, 96)"},
        },
    ),
    (
        Reshape743,
        [((1, 8, 8, 7, 7, 96), torch.float32)],
        {
            "model_names": [
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(3136, 96)"},
        },
    ),
    (
        Reshape744,
        [((3136, 96), torch.float32)],
        {
            "model_names": [
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(64, 49, 96)"},
        },
    ),
    (
        Reshape742,
        [((3136, 96), torch.float32)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(1, 56, 56, 96)"},
        },
    ),
    (
        Reshape745,
        [((64, 49, 96), torch.float32)],
        {
            "model_names": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(64, 49, 3, 32)"},
        },
    ),
    (
        Reshape746,
        [((64, 49, 96), torch.float32)],
        {
            "model_names": [
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 8, 8, 7, 7, 96)"},
        },
    ),
    (
        Reshape747,
        [((64, 3, 49, 32), torch.float32)],
        {
            "model_names": [
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(192, 49, 32)"},
        },
    ),
    (
        Reshape748,
        [((192, 49, 49), torch.float32)],
        {
            "model_names": [
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(64, 3, 49, 49)"},
        },
    ),
    (
        Reshape749,
        [((49, 49), torch.int64)],
        {
            "model_names": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(2401,)"},
        },
    ),
    (
        Reshape750,
        [((2401, 3), torch.float32)],
        {
            "model_names": [
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(49, 49, 3)"},
        },
    ),
    (
        Reshape751,
        [((64, 3, 49, 49), torch.float32)],
        {
            "model_names": [
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(192, 49, 49)"},
        },
    ),
    (
        Reshape752,
        [((64, 3, 49, 49), torch.float32)],
        {
            "model_names": [
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 64, 3, 49, 49)"},
        },
    ),
    (
        Reshape753,
        [((64, 3, 32, 49), torch.float32)],
        {
            "model_names": [
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(192, 32, 49)"},
        },
    ),
    (
        Reshape754,
        [((192, 49, 32), torch.float32)],
        {
            "model_names": [
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(64, 3, 49, 32)"},
        },
    ),
    (
        Reshape743,
        [((64, 49, 3, 32), torch.float32)],
        {
            "model_names": [
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(3136, 96)"},
        },
    ),
    (
        Reshape755,
        [((1, 8, 7, 8, 7, 96), torch.float32)],
        {
            "model_names": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 3136, 96)"},
        },
    ),
    (
        Reshape742,
        [((1, 8, 7, 8, 7, 96), torch.float32)],
        {
            "model_names": [
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 56, 56, 96)"},
        },
    ),
    (
        Reshape741,
        [((1, 56, 56, 96), torch.float32)],
        {
            "model_names": [
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 8, 7, 8, 7, 96)"},
        },
    ),
    (
        Reshape755,
        [((1, 56, 56, 96), torch.float32)],
        {
            "model_names": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 3136, 96)"},
        },
    ),
    (
        Reshape743,
        [((1, 56, 56, 96), torch.float32)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(3136, 96)"},
        },
    ),
    (
        Reshape748,
        [((1, 64, 3, 49, 49), torch.float32)],
        {
            "model_names": [
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(64, 3, 49, 49)"},
        },
    ),
    (
        Reshape756,
        [((1, 28, 28, 384), torch.float32)],
        {
            "model_names": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 784, 384)"},
        },
    ),
    (
        Reshape757,
        [((1, 28, 28, 384), torch.float32)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(784, 384)"},
        },
    ),
    (
        Reshape758,
        [((1, 784, 192), torch.float32)],
        {
            "model_names": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 4, 7, 4, 7, 192)"},
        },
    ),
    (
        Reshape759,
        [((1, 784, 192), torch.float32)],
        {
            "model_names": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 28, 28, 192)"},
        },
    ),
    (
        Reshape760,
        [((1, 4, 4, 7, 7, 192), torch.float32)],
        {
            "model_names": [
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(784, 192)"},
        },
    ),
    (
        Reshape761,
        [((784, 192), torch.float32)],
        {
            "model_names": [
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(16, 49, 192)"},
        },
    ),
    (
        Reshape759,
        [((784, 192), torch.float32)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(1, 28, 28, 192)"},
        },
    ),
    (
        Reshape762,
        [((16, 49, 192), torch.float32)],
        {
            "model_names": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(16, 49, 6, 32)"},
        },
    ),
    (
        Reshape763,
        [((16, 49, 192), torch.float32)],
        {
            "model_names": [
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 4, 4, 7, 7, 192)"},
        },
    ),
    (
        Reshape764,
        [((16, 6, 49, 32), torch.float32)],
        {
            "model_names": [
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(96, 49, 32)"},
        },
    ),
    (
        Reshape765,
        [((96, 49, 49), torch.float32)],
        {
            "model_names": [
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(16, 6, 49, 49)"},
        },
    ),
    (
        Reshape766,
        [((2401, 6), torch.float32)],
        {
            "model_names": [
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(49, 49, 6)"},
        },
    ),
    (
        Reshape767,
        [((16, 6, 49, 49), torch.float32)],
        {
            "model_names": [
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(96, 49, 49)"},
        },
    ),
    (
        Reshape768,
        [((16, 6, 49, 49), torch.float32)],
        {
            "model_names": [
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 16, 6, 49, 49)"},
        },
    ),
    (
        Reshape769,
        [((16, 6, 32, 49), torch.float32)],
        {
            "model_names": [
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(96, 32, 49)"},
        },
    ),
    (
        Reshape770,
        [((96, 49, 32), torch.float32)],
        {
            "model_names": [
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(16, 6, 49, 32)"},
        },
    ),
    (
        Reshape760,
        [((16, 49, 6, 32), torch.float32)],
        {
            "model_names": [
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(784, 192)"},
        },
    ),
    (
        Reshape771,
        [((1, 4, 7, 4, 7, 192), torch.float32)],
        {
            "model_names": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 784, 192)"},
        },
    ),
    (
        Reshape759,
        [((1, 4, 7, 4, 7, 192), torch.float32)],
        {
            "model_names": [
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 28, 28, 192)"},
        },
    ),
    (
        Reshape758,
        [((1, 28, 28, 192), torch.float32)],
        {
            "model_names": [
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 4, 7, 4, 7, 192)"},
        },
    ),
    (
        Reshape771,
        [((1, 28, 28, 192), torch.float32)],
        {
            "model_names": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 784, 192)"},
        },
    ),
    (
        Reshape760,
        [((1, 28, 28, 192), torch.float32)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(784, 192)"},
        },
    ),
    (
        Reshape765,
        [((1, 16, 6, 49, 49), torch.float32)],
        {
            "model_names": [
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(16, 6, 49, 49)"},
        },
    ),
    (
        Reshape772,
        [((1, 14, 14, 768), torch.float32)],
        {
            "model_names": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 196, 768)"},
        },
    ),
    (
        Reshape773,
        [((1, 14, 14, 768), torch.float32)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(196, 768)"},
        },
    ),
    (
        Reshape774,
        [((1, 196, 384), torch.float32)],
        {
            "model_names": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 2, 7, 2, 7, 384)"},
        },
    ),
    (
        Reshape775,
        [((1, 196, 384), torch.float32)],
        {
            "model_names": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 14, 14, 384)"},
        },
    ),
    (
        Reshape776,
        [((1, 2, 2, 7, 7, 384), torch.float32)],
        {
            "model_names": [
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(196, 384)"},
        },
    ),
    (
        Reshape777,
        [((196, 384), torch.float32)],
        {
            "model_names": [
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(4, 49, 384)"},
        },
    ),
    (
        Reshape775,
        [((196, 384), torch.float32)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(1, 14, 14, 384)"},
        },
    ),
    (
        Reshape778,
        [((4, 49, 384), torch.float32)],
        {
            "model_names": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(4, 49, 12, 32)"},
        },
    ),
    (
        Reshape779,
        [((4, 49, 384), torch.float32)],
        {
            "model_names": [
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 2, 2, 7, 7, 384)"},
        },
    ),
    (
        Reshape780,
        [((4, 12, 49, 32), torch.float32)],
        {
            "model_names": [
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(48, 49, 32)"},
        },
    ),
    (
        Reshape781,
        [((48, 49, 49), torch.float32)],
        {
            "model_names": [
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(4, 12, 49, 49)"},
        },
    ),
    (
        Reshape782,
        [((2401, 12), torch.float32)],
        {
            "model_names": [
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(49, 49, 12)"},
        },
    ),
    (
        Reshape783,
        [((4, 12, 49, 49), torch.float32)],
        {
            "model_names": [
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(48, 49, 49)"},
        },
    ),
    (
        Reshape784,
        [((4, 12, 49, 49), torch.float32)],
        {
            "model_names": [
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 4, 12, 49, 49)"},
        },
    ),
    (
        Reshape785,
        [((4, 12, 32, 49), torch.float32)],
        {
            "model_names": [
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(48, 32, 49)"},
        },
    ),
    (
        Reshape786,
        [((48, 49, 32), torch.float32)],
        {
            "model_names": [
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(4, 12, 49, 32)"},
        },
    ),
    (
        Reshape776,
        [((4, 49, 12, 32), torch.float32)],
        {
            "model_names": [
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(196, 384)"},
        },
    ),
    (
        Reshape787,
        [((1, 2, 7, 2, 7, 384), torch.float32)],
        {
            "model_names": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 196, 384)"},
        },
    ),
    (
        Reshape775,
        [((1, 2, 7, 2, 7, 384), torch.float32)],
        {
            "model_names": [
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 14, 14, 384)"},
        },
    ),
    (
        Reshape774,
        [((1, 14, 14, 384), torch.float32)],
        {
            "model_names": [
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 2, 7, 2, 7, 384)"},
        },
    ),
    (
        Reshape787,
        [((1, 14, 14, 384), torch.float32)],
        {
            "model_names": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 196, 384)"},
        },
    ),
    (
        Reshape776,
        [((1, 14, 14, 384), torch.float32)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(196, 384)"},
        },
    ),
    (
        Reshape781,
        [((1, 4, 12, 49, 49), torch.float32)],
        {
            "model_names": [
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(4, 12, 49, 49)"},
        },
    ),
    (
        Reshape788,
        [((1, 7, 7, 1536), torch.float32)],
        {
            "model_names": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 49, 1536)"},
        },
    ),
    (
        Reshape789,
        [((1, 7, 7, 1536), torch.float32)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(49, 1536)"},
        },
    ),
    (
        Reshape790,
        [((1, 49, 768), torch.float32)],
        {
            "model_names": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(49, 768)"},
        },
    ),
    (
        Reshape791,
        [((1, 49, 768), torch.float32)],
        {
            "model_names": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 49, 24, 32)"},
        },
    ),
    (
        Reshape792,
        [((1, 49, 768), torch.float32)],
        {
            "model_names": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 49, 768)"},
        },
    ),
    (
        Reshape793,
        [((1, 49, 768), torch.float32)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(1, 1, 1, 7, 7, 768)"},
        },
    ),
    (
        Reshape792,
        [((49, 768), torch.float32)],
        {
            "model_names": [
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 49, 768)"},
        },
    ),
    (
        Reshape794,
        [((49, 768), torch.float32)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(1, 7, 7, 768)"},
        },
    ),
    (
        Reshape795,
        [((1, 24, 49, 32), torch.float32)],
        {
            "model_names": [
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(24, 49, 32)"},
        },
    ),
    (
        Reshape796,
        [((24, 49, 49), torch.float32)],
        {
            "model_names": [
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 24, 49, 49)"},
        },
    ),
    (
        Reshape797,
        [((2401, 24), torch.float32)],
        {
            "model_names": [
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(49, 49, 24)"},
        },
    ),
    (
        Reshape798,
        [((1, 24, 49, 49), torch.float32)],
        {
            "model_names": [
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(24, 49, 49)"},
        },
    ),
    (
        Reshape799,
        [((1, 24, 32, 49), torch.float32)],
        {
            "model_names": [
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(24, 32, 49)"},
        },
    ),
    (
        Reshape800,
        [((24, 49, 32), torch.float32)],
        {
            "model_names": [
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 24, 49, 32)"},
        },
    ),
    (
        Reshape790,
        [((1, 49, 24, 32), torch.float32)],
        {
            "model_names": [
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(49, 768)"},
        },
    ),
    (
        Reshape801,
        [((1, 768, 1), torch.float32)],
        {
            "model_names": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 768, 1)"},
        },
    ),
    (
        Reshape130,
        [((1, 768, 1), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 768)"},
        },
    ),
    (
        Reshape802,
        [((1, 8, 1, 64), torch.float32)],
        {
            "model_names": ["pt_t5_t5_small_text_gen_hf", "pt_whisper_openai_whisper_base_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(8, 1, 64)"},
        },
    ),
    (
        Reshape803,
        [((8, 1, 1), torch.float32)],
        {
            "model_names": ["pt_t5_t5_small_text_gen_hf", "pt_whisper_openai_whisper_base_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 8, 1, 1)"},
        },
    ),
    (
        Reshape804,
        [((1, 8, 1, 1), torch.float32)],
        {
            "model_names": ["pt_t5_t5_small_text_gen_hf", "pt_whisper_openai_whisper_base_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(8, 1, 1)"},
        },
    ),
    (
        Reshape805,
        [((1, 8, 64, 1), torch.float32)],
        {
            "model_names": ["pt_t5_t5_small_text_gen_hf", "pt_whisper_openai_whisper_base_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(8, 64, 1)"},
        },
    ),
    (
        Reshape806,
        [((8, 1, 64), torch.float32)],
        {
            "model_names": ["pt_t5_t5_small_text_gen_hf", "pt_whisper_openai_whisper_base_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 8, 1, 64)"},
        },
    ),
    (
        Reshape246,
        [((1, 1, 8, 64), torch.float32)],
        {
            "model_names": ["pt_t5_t5_small_text_gen_hf", "pt_whisper_openai_whisper_base_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 512)"},
        },
    ),
    (
        Reshape807,
        [((1, 8, 61, 64), torch.float32)],
        {"model_names": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99, "args": {"shape": "(8, 61, 64)"}},
    ),
    (
        Reshape808,
        [((8, 61, 61), torch.float32)],
        {"model_names": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99, "args": {"shape": "(1, 8, 61, 61)"}},
    ),
    (
        Reshape809,
        [((1, 8, 61, 61), torch.float32)],
        {"model_names": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99, "args": {"shape": "(8, 61, 61)"}},
    ),
    (
        Reshape810,
        [((1, 8, 64, 61), torch.float32)],
        {"model_names": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99, "args": {"shape": "(8, 64, 61)"}},
    ),
    (
        Reshape811,
        [((8, 61, 64), torch.float32)],
        {"model_names": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99, "args": {"shape": "(1, 8, 61, 64)"}},
    ),
    (
        Reshape366,
        [((1, 61, 8, 64), torch.float32)],
        {"model_names": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99, "args": {"shape": "(61, 512)"}},
    ),
    (
        Reshape812,
        [((8, 1, 61), torch.float32)],
        {"model_names": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99, "args": {"shape": "(1, 8, 1, 61)"}},
    ),
    (
        Reshape813,
        [((1, 8, 1, 61), torch.float32)],
        {"model_names": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99, "args": {"shape": "(8, 1, 61)"}},
    ),
    (
        Reshape814,
        [((1, 1280, 37, 37), torch.float32)],
        {"model_names": ["pt_vit_vit_h_14_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(1, 1280, 1369)"}},
    ),
    (
        Reshape815,
        [((1370, 1, 1280), torch.float32)],
        {"model_names": ["pt_vit_vit_h_14_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(1370, 1280)"}},
    ),
    (
        Reshape816,
        [((1370, 3840), torch.float32)],
        {"model_names": ["pt_vit_vit_h_14_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(1370, 1, 3840)"}},
    ),
    (
        Reshape817,
        [((1370, 1, 3840), torch.float32)],
        {"model_names": ["pt_vit_vit_h_14_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(1370, 1, 3, 1280)"}},
    ),
    (
        Reshape818,
        [((1, 1370, 1, 1280), torch.float32)],
        {"model_names": ["pt_vit_vit_h_14_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(1370, 16, 80)"}},
    ),
    (
        Reshape819,
        [((16, 1370, 80), torch.float32)],
        {"model_names": ["pt_vit_vit_h_14_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(16, 1370, 80)"}},
    ),
    (
        Reshape820,
        [((16, 1370, 80), torch.float32)],
        {"model_names": ["pt_vit_vit_h_14_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(1, 16, 1370, 80)"}},
    ),
    (
        Reshape821,
        [((16, 1370, 1370), torch.float32)],
        {"model_names": ["pt_vit_vit_h_14_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(1, 16, 1370, 1370)"}},
    ),
    (
        Reshape822,
        [((1, 16, 1370, 1370), torch.float32)],
        {"model_names": ["pt_vit_vit_h_14_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(16, 1370, 1370)"}},
    ),
    (
        Reshape815,
        [((1370, 1, 16, 80), torch.float32)],
        {"model_names": ["pt_vit_vit_h_14_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(1370, 1280)"}},
    ),
    (
        Reshape823,
        [((1370, 1280), torch.float32)],
        {"model_names": ["pt_vit_vit_h_14_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(1370, 1, 1280)"}},
    ),
    (
        Reshape2,
        [((1, 1, 1280), torch.float32)],
        {"model_names": ["pt_vit_vit_h_14_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(1, 1280)"}},
    ),
    (
        Reshape510,
        [((1, 2), torch.int64)],
        {
            "model_names": [
                "pt_whisper_openai_whisper_large_v3_clm_hf",
                "pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 2)"},
        },
    ),
    (
        Reshape824,
        [((1, 2, 1280), torch.float32)],
        {
            "model_names": [
                "pt_whisper_openai_whisper_large_v3_clm_hf",
                "pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(2, 1280)"},
        },
    ),
    (
        Reshape825,
        [((1, 2, 1280), torch.float32)],
        {
            "model_names": [
                "pt_whisper_openai_whisper_large_v3_clm_hf",
                "pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 2, 20, 64)"},
        },
    ),
    (
        Reshape826,
        [((2, 1280), torch.float32)],
        {
            "model_names": [
                "pt_whisper_openai_whisper_large_v3_clm_hf",
                "pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 2, 1280)"},
        },
    ),
    (
        Reshape825,
        [((2, 1280), torch.float32)],
        {
            "model_names": [
                "pt_whisper_openai_whisper_large_v3_clm_hf",
                "pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 2, 20, 64)"},
        },
    ),
    (
        Reshape827,
        [((1, 20, 2, 64), torch.float32)],
        {
            "model_names": [
                "pt_whisper_openai_whisper_large_v3_clm_hf",
                "pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(20, 2, 64)"},
        },
    ),
    (
        Reshape828,
        [((20, 2, 2), torch.float32)],
        {
            "model_names": [
                "pt_whisper_openai_whisper_large_v3_clm_hf",
                "pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 20, 2, 2)"},
        },
    ),
    (
        Reshape829,
        [((1, 20, 2, 2), torch.float32)],
        {
            "model_names": [
                "pt_whisper_openai_whisper_large_v3_clm_hf",
                "pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(20, 2, 2)"},
        },
    ),
    (
        Reshape830,
        [((1, 20, 64, 2), torch.float32)],
        {
            "model_names": [
                "pt_whisper_openai_whisper_large_v3_clm_hf",
                "pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(20, 64, 2)"},
        },
    ),
    (
        Reshape831,
        [((20, 2, 64), torch.float32)],
        {
            "model_names": [
                "pt_whisper_openai_whisper_large_v3_clm_hf",
                "pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 20, 2, 64)"},
        },
    ),
    (
        Reshape824,
        [((1, 2, 20, 64), torch.float32)],
        {
            "model_names": [
                "pt_whisper_openai_whisper_large_v3_clm_hf",
                "pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(2, 1280)"},
        },
    ),
    (
        Reshape832,
        [((1, 128, 3000), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_large_v3_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 128, 3000, 1)"},
        },
    ),
    (
        Reshape833,
        [((1280, 128, 3), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_large_v3_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1280, 128, 3, 1)"},
        },
    ),
    (
        Reshape834,
        [((1, 1280, 3000, 1), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_large_v3_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 1280, 3000)"},
        },
    ),
    (
        Reshape835,
        [((1, 1280, 3000), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_large_v3_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 1280, 3000, 1)"},
        },
    ),
    (
        Reshape836,
        [((1280, 1280, 3), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_large_v3_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1280, 1280, 3, 1)"},
        },
    ),
    (
        Reshape837,
        [((1, 1280, 1500, 1), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_large_v3_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 1280, 1500)"},
        },
    ),
    (
        Reshape838,
        [((1, 1500, 1280), torch.float32)],
        {
            "model_names": [
                "pt_whisper_openai_whisper_large_v3_clm_hf",
                "pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1500, 1280)"},
        },
    ),
    (
        Reshape839,
        [((1, 1500, 1280), torch.float32)],
        {
            "model_names": [
                "pt_whisper_openai_whisper_large_v3_clm_hf",
                "pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 1500, 20, 64)"},
        },
    ),
    (
        Reshape840,
        [((1500, 1280), torch.float32)],
        {
            "model_names": [
                "pt_whisper_openai_whisper_large_v3_clm_hf",
                "pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 1500, 1280)"},
        },
    ),
    (
        Reshape839,
        [((1500, 1280), torch.float32)],
        {
            "model_names": [
                "pt_whisper_openai_whisper_large_v3_clm_hf",
                "pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 1500, 20, 64)"},
        },
    ),
    (
        Reshape841,
        [((1, 20, 1500, 64), torch.float32)],
        {
            "model_names": [
                "pt_whisper_openai_whisper_large_v3_clm_hf",
                "pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(20, 1500, 64)"},
        },
    ),
    (
        Reshape842,
        [((20, 1500, 1500), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_large_v3_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 20, 1500, 1500)"},
        },
    ),
    (
        Reshape843,
        [((1, 20, 1500, 1500), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_large_v3_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(20, 1500, 1500)"},
        },
    ),
    (
        Reshape844,
        [((1, 20, 64, 1500), torch.float32)],
        {
            "model_names": [
                "pt_whisper_openai_whisper_large_v3_clm_hf",
                "pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(20, 64, 1500)"},
        },
    ),
    (
        Reshape845,
        [((20, 1500, 64), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_large_v3_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 20, 1500, 64)"},
        },
    ),
    (
        Reshape838,
        [((1, 1500, 20, 64), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_large_v3_clm_hf"], "pcc": 0.99, "args": {"shape": "(1500, 1280)"}},
    ),
    (
        Reshape846,
        [((20, 2, 1500), torch.float32)],
        {
            "model_names": [
                "pt_whisper_openai_whisper_large_v3_clm_hf",
                "pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 20, 2, 1500)"},
        },
    ),
    (
        Reshape847,
        [((1, 20, 2, 1500), torch.float32)],
        {
            "model_names": [
                "pt_whisper_openai_whisper_large_v3_clm_hf",
                "pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(20, 2, 1500)"},
        },
    ),
    (
        Reshape848,
        [((1, 3, 85, 60, 60), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5m_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_480x480",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 255, 60, 60)"},
        },
    ),
    (
        Reshape849,
        [((1, 255, 60, 60), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5m_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_480x480",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 1, 255, 3600)"},
        },
    ),
    (
        Reshape850,
        [((1, 1, 255, 3600), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5m_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_480x480",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 3, 85, 3600)"},
        },
    ),
    (
        Reshape851,
        [((1, 3, 3600, 85), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5m_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_480x480",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 10800, 85)"},
        },
    ),
    (
        Reshape852,
        [((1, 3, 85, 30, 30), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5m_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_480x480",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 255, 30, 30)"},
        },
    ),
    (
        Reshape853,
        [((1, 255, 30, 30), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5m_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_480x480",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 1, 255, 900)"},
        },
    ),
    (
        Reshape854,
        [((1, 1, 255, 900), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5m_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_480x480",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 3, 85, 900)"},
        },
    ),
    (
        Reshape855,
        [((1, 3, 900, 85), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5m_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_480x480",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 2700, 85)"},
        },
    ),
    (
        Reshape856,
        [((1, 3, 85, 15, 15), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5m_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_480x480",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 255, 15, 15)"},
        },
    ),
    (
        Reshape857,
        [((1, 255, 15, 15), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5m_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_480x480",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 1, 255, 225)"},
        },
    ),
    (
        Reshape858,
        [((1, 1, 255, 225), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5m_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_480x480",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 3, 85, 225)"},
        },
    ),
    (
        Reshape859,
        [((1, 3, 225, 85), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5m_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_480x480",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 675, 85)"},
        },
    ),
    (
        Reshape860,
        [((1, 192, 32, 42), torch.float32)],
        {"model_names": ["pt_yolos_hustvl_yolos_tiny_obj_det_hf"], "pcc": 0.99, "args": {"shape": "(1, 192, 1344, 1)"}},
    ),
    (
        Reshape861,
        [((1, 192, 4150), torch.float32)],
        {"model_names": ["pt_yolos_hustvl_yolos_tiny_obj_det_hf"], "pcc": 0.99, "args": {"shape": "(1, 192, 50, 83)"}},
    ),
    (
        Reshape862,
        [((1, 1445, 192), torch.float32)],
        {"model_names": ["pt_yolos_hustvl_yolos_tiny_obj_det_hf"], "pcc": 0.99, "args": {"shape": "(1445, 192)"}},
    ),
    (
        Reshape863,
        [((1, 1445, 192), torch.float32)],
        {"model_names": ["pt_yolos_hustvl_yolos_tiny_obj_det_hf"], "pcc": 0.99, "args": {"shape": "(1, 1445, 3, 64)"}},
    ),
    (
        Reshape864,
        [((1445, 192), torch.float32)],
        {"model_names": ["pt_yolos_hustvl_yolos_tiny_obj_det_hf"], "pcc": 0.99, "args": {"shape": "(1, 1445, 192)"}},
    ),
    (
        Reshape865,
        [((1, 3, 1445, 64), torch.float32)],
        {"model_names": ["pt_yolos_hustvl_yolos_tiny_obj_det_hf"], "pcc": 0.99, "args": {"shape": "(3, 1445, 64)"}},
    ),
    (
        Reshape866,
        [((3, 1445, 1445), torch.float32)],
        {
            "model_names": ["pt_yolos_hustvl_yolos_tiny_obj_det_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 3, 1445, 1445)"},
        },
    ),
    (
        Reshape867,
        [((1, 3, 1445, 1445), torch.float32)],
        {"model_names": ["pt_yolos_hustvl_yolos_tiny_obj_det_hf"], "pcc": 0.99, "args": {"shape": "(3, 1445, 1445)"}},
    ),
    (
        Reshape868,
        [((1, 3, 64, 1445), torch.float32)],
        {"model_names": ["pt_yolos_hustvl_yolos_tiny_obj_det_hf"], "pcc": 0.99, "args": {"shape": "(3, 64, 1445)"}},
    ),
    (
        Reshape869,
        [((3, 1445, 64), torch.float32)],
        {"model_names": ["pt_yolos_hustvl_yolos_tiny_obj_det_hf"], "pcc": 0.99, "args": {"shape": "(1, 3, 1445, 64)"}},
    ),
    (
        Reshape862,
        [((1, 1445, 3, 64), torch.float32)],
        {"model_names": ["pt_yolos_hustvl_yolos_tiny_obj_det_hf"], "pcc": 0.99, "args": {"shape": "(1445, 192)"}},
    ),
    (
        Reshape870,
        [((1, 100, 192), torch.float32)],
        {"model_names": ["pt_yolos_hustvl_yolos_tiny_obj_det_hf"], "pcc": 0.99, "args": {"shape": "(100, 192)"}},
    ),
    (
        Reshape871,
        [((100, 192), torch.float32)],
        {"model_names": ["pt_yolos_hustvl_yolos_tiny_obj_det_hf"], "pcc": 0.99, "args": {"shape": "(1, 100, 192)"}},
    ),
    (
        Reshape872,
        [((1, 144, 80, 80), torch.float32)],
        {
            "model_names": [
                "pt_yolov8_default_obj_det_github",
                "onnx_yolov10_default_obj_det_github",
                "pt_yolov9_default_obj_det_github",
                "onnx_yolov8_default_obj_det_github",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 144, 6400)"},
        },
    ),
    (
        Reshape873,
        [((1, 144, 40, 40), torch.float32)],
        {
            "model_names": [
                "pt_yolov8_default_obj_det_github",
                "onnx_yolov10_default_obj_det_github",
                "pt_yolov9_default_obj_det_github",
                "onnx_yolov8_default_obj_det_github",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 144, 1600)"},
        },
    ),
    (
        Reshape874,
        [((1, 144, 20, 20), torch.float32)],
        {
            "model_names": [
                "pt_yolov8_default_obj_det_github",
                "onnx_yolov10_default_obj_det_github",
                "pt_yolov9_default_obj_det_github",
                "onnx_yolov8_default_obj_det_github",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 144, 400)"},
        },
    ),
    (
        Reshape875,
        [((1, 64, 8400), torch.float32)],
        {
            "model_names": [
                "pt_yolov8_default_obj_det_github",
                "onnx_yolov10_default_obj_det_github",
                "pt_yolov9_default_obj_det_github",
                "onnx_yolov8_default_obj_det_github",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 4, 16, 8400)"},
        },
    ),
    (
        Reshape876,
        [((1, 1, 4, 8400), torch.float32)],
        {
            "model_names": [
                "pt_yolov8_default_obj_det_github",
                "onnx_yolov10_default_obj_det_github",
                "pt_yolov9_default_obj_det_github",
                "onnx_yolov8_default_obj_det_github",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 4, 8400)"},
        },
    ),
    (
        Reshape877,
        [((1, 1408, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2a_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 1408)"},
        },
    ),
    (
        Reshape878,
        [((1, 96, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 96, 4096)"},
        },
    ),
    (
        Reshape277,
        [((1, 4096, 96), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 64, 64, 96)"},
        },
    ),
    (
        Reshape297,
        [((1, 1024, 192), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 32, 32, 192)"},
        },
    ),
    (
        Reshape318,
        [((1, 256, 384), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 16, 16, 384)"},
        },
    ),
    (
        Reshape879,
        [((1, 256, 20, 20), torch.float32)],
        {"model_names": ["onnx_yolov10_default_obj_det_github"], "pcc": 0.99, "args": {"shape": "(1, 2, 128, 400)"}},
    ),
    (
        Reshape880,
        [((1, 2, 64, 400), torch.float32)],
        {"model_names": ["onnx_yolov10_default_obj_det_github"], "pcc": 0.99, "args": {"shape": "(2, 64, 400)"}},
    ),
    (
        Reshape881,
        [((1, 2, 64, 400), torch.float32)],
        {"model_names": ["onnx_yolov10_default_obj_det_github"], "pcc": 0.99, "args": {"shape": "(1, 128, 20, 20)"}},
    ),
    (
        Reshape882,
        [((1, 2, 400, 32), torch.float32)],
        {"model_names": ["onnx_yolov10_default_obj_det_github"], "pcc": 0.99, "args": {"shape": "(2, 400, 32)"}},
    ),
    (
        Reshape883,
        [((1, 2, 32, 400), torch.float32)],
        {"model_names": ["onnx_yolov10_default_obj_det_github"], "pcc": 0.99, "args": {"shape": "(2, 32, 400)"}},
    ),
    (
        Reshape884,
        [((2, 400, 400), torch.float32)],
        {"model_names": ["onnx_yolov10_default_obj_det_github"], "pcc": 0.99, "args": {"shape": "(1, 2, 400, 400)"}},
    ),
    (
        Reshape885,
        [((1, 2, 400, 400), torch.float32)],
        {"model_names": ["onnx_yolov10_default_obj_det_github"], "pcc": 0.99, "args": {"shape": "(2, 400, 400)"}},
    ),
    (
        Reshape881,
        [((2, 64, 400), torch.float32)],
        {"model_names": ["onnx_yolov10_default_obj_det_github"], "pcc": 0.99, "args": {"shape": "(1, 128, 20, 20)"}},
    ),
    (
        Reshape886,
        [((1, 128, 2048), torch.float32)],
        {"model_names": ["pt_albert_xlarge_v1_mlm_hf"], "pcc": 0.99, "args": {"shape": "(128, 2048)"}},
    ),
    (
        Reshape887,
        [((1, 128, 2048), torch.float32)],
        {"model_names": ["pt_albert_xlarge_v1_mlm_hf"], "pcc": 0.99, "args": {"shape": "(1, 128, 16, 128)"}},
    ),
    (
        Reshape888,
        [((128, 2048), torch.float32)],
        {"model_names": ["pt_albert_xlarge_v1_mlm_hf"], "pcc": 0.99, "args": {"shape": "(1, 128, 2048)"}},
    ),
    (
        Reshape889,
        [((1, 128, 16, 128), torch.float32)],
        {"model_names": ["pt_albert_xlarge_v1_mlm_hf"], "pcc": 0.99, "args": {"shape": "(1, 128, 2048, 1)"}},
    ),
    (
        Reshape158,
        [((1, 128), torch.bool)],
        {
            "model_names": [
                "pt_distilbert_distilbert_base_cased_mlm_hf",
                "pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf",
                "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 1, 1, 128)"},
        },
    ),
    (
        Reshape890,
        [((2688, 1, 3, 3), torch.float32)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"shape": "(2688, 1, 3, 3)"},
        },
    ),
    (
        Reshape891,
        [((1, 1792, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 1792, 1, 1)"},
        },
    ),
    (
        Reshape892,
        [((1, 1792, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b4_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_140_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 1792)"},
        },
    ),
    (
        Reshape893,
        [((1, 12, 256, 64), torch.float32)],
        {
            "model_names": ["pt_gptneo_eleutherai_gpt_neo_125m_clm_hf", "pt_gpt_gpt2_text_gen_hf"],
            "pcc": 0.99,
            "args": {"shape": "(12, 256, 64)"},
        },
    ),
    (
        Reshape894,
        [((12, 256, 256), torch.float32)],
        {
            "model_names": ["pt_gptneo_eleutherai_gpt_neo_125m_clm_hf", "pt_gpt_gpt2_text_gen_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 12, 256, 256)"},
        },
    ),
    (
        Reshape895,
        [((1, 12, 256, 256), torch.float32)],
        {
            "model_names": ["pt_gptneo_eleutherai_gpt_neo_125m_clm_hf", "pt_gpt_gpt2_text_gen_hf"],
            "pcc": 0.99,
            "args": {"shape": "(12, 256, 256)"},
        },
    ),
    (
        Reshape896,
        [((1, 12, 64, 256), torch.float32)],
        {
            "model_names": ["pt_gptneo_eleutherai_gpt_neo_125m_clm_hf", "pt_gpt_gpt2_text_gen_hf"],
            "pcc": 0.99,
            "args": {"shape": "(12, 64, 256)"},
        },
    ),
    (
        Reshape897,
        [((12, 256, 64), torch.float32)],
        {
            "model_names": ["pt_gptneo_eleutherai_gpt_neo_125m_clm_hf", "pt_gpt_gpt2_text_gen_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 12, 256, 64)"},
        },
    ),
    (
        Reshape317,
        [((1, 256, 12, 64), torch.float32)],
        {
            "model_names": ["pt_gptneo_eleutherai_gpt_neo_125m_clm_hf", "pt_gpt_gpt2_text_gen_hf"],
            "pcc": 0.99,
            "args": {"shape": "(256, 768)"},
        },
    ),
    (
        Reshape898,
        [((1, 32, 256, 64), torch.float32)],
        {
            "model_names": ["pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf", "pt_phi1_microsoft_phi_1_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(32, 256, 64)"},
        },
    ),
    (
        Reshape898,
        [((1, 8, 4, 256, 64), torch.float32)],
        {
            "model_names": ["pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(32, 256, 64)"},
        },
    ),
    (
        Reshape899,
        [((1, 8, 4, 256, 64), torch.float32)],
        {
            "model_names": ["pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 32, 256, 64)"},
        },
    ),
    (
        Reshape900,
        [((32, 256, 256), torch.float32)],
        {
            "model_names": [
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf",
                "pt_phi2_microsoft_phi_2_clm_hf",
                "pt_phi1_microsoft_phi_1_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 32, 256, 256)"},
        },
    ),
    (
        Reshape901,
        [((1, 32, 256, 256), torch.float32)],
        {
            "model_names": [
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf",
                "pt_phi2_microsoft_phi_2_clm_hf",
                "pt_phi1_microsoft_phi_1_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(32, 256, 256)"},
        },
    ),
    (
        Reshape902,
        [((1, 32, 64, 256), torch.float32)],
        {
            "model_names": ["pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf", "pt_phi1_microsoft_phi_1_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(32, 64, 256)"},
        },
    ),
    (
        Reshape899,
        [((32, 256, 64), torch.float32)],
        {
            "model_names": ["pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf", "pt_phi1_microsoft_phi_1_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 32, 256, 64)"},
        },
    ),
    (
        Reshape650,
        [((1, 256, 32, 64), torch.float32)],
        {
            "model_names": ["pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf", "pt_phi1_microsoft_phi_1_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(256, 2048)"},
        },
    ),
    (
        Reshape903,
        [((256, 8192), torch.float32)],
        {
            "model_names": ["pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf", "pt_phi1_microsoft_phi_1_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 256, 8192)"},
        },
    ),
    (
        Reshape904,
        [((512, 1, 3, 3), torch.float32)],
        {
            "model_names": [
                "pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf",
                "pt_mobilenetv1_basic_img_cls_torchvision",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(512, 1, 3, 3)"},
        },
    ),
    (
        Reshape905,
        [((1, 256, 2560), torch.float32)],
        {"model_names": ["pt_phi2_microsoft_phi_2_clm_hf"], "pcc": 0.99, "args": {"shape": "(256, 2560)"}},
    ),
    (
        Reshape906,
        [((1, 256, 2560), torch.float32)],
        {"model_names": ["pt_phi2_microsoft_phi_2_clm_hf"], "pcc": 0.99, "args": {"shape": "(1, 256, 32, 80)"}},
    ),
    (
        Reshape907,
        [((256, 2560), torch.float32)],
        {"model_names": ["pt_phi2_microsoft_phi_2_clm_hf"], "pcc": 0.99, "args": {"shape": "(1, 256, 2560)"}},
    ),
    (
        Reshape908,
        [((1, 32, 256, 80), torch.float32)],
        {"model_names": ["pt_phi2_microsoft_phi_2_clm_hf"], "pcc": 0.99, "args": {"shape": "(32, 256, 80)"}},
    ),
    (
        Reshape909,
        [((1, 32, 80, 256), torch.float32)],
        {"model_names": ["pt_phi2_microsoft_phi_2_clm_hf"], "pcc": 0.99, "args": {"shape": "(32, 80, 256)"}},
    ),
    (
        Reshape910,
        [((32, 256, 80), torch.float32)],
        {"model_names": ["pt_phi2_microsoft_phi_2_clm_hf"], "pcc": 0.99, "args": {"shape": "(1, 32, 256, 80)"}},
    ),
    (
        Reshape905,
        [((1, 256, 32, 80), torch.float32)],
        {"model_names": ["pt_phi2_microsoft_phi_2_clm_hf"], "pcc": 0.99, "args": {"shape": "(256, 2560)"}},
    ),
    (
        Reshape911,
        [((256, 10240), torch.float32)],
        {"model_names": ["pt_phi2_microsoft_phi_2_clm_hf"], "pcc": 0.99, "args": {"shape": "(1, 256, 10240)"}},
    ),
    (
        Reshape912,
        [((1, 29, 1024), torch.float32)],
        {"model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_chat_clm_hf"], "pcc": 0.99, "args": {"shape": "(29, 1024)"}},
    ),
    (
        Reshape913,
        [((1, 29, 1024), torch.float32)],
        {
            "model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_chat_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 29, 16, 64)"},
        },
    ),
    (
        Reshape914,
        [((29, 1024), torch.float32)],
        {"model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_chat_clm_hf"], "pcc": 0.99, "args": {"shape": "(1, 29, 1024)"}},
    ),
    (
        Reshape915,
        [((1, 16, 29, 64), torch.float32)],
        {"model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_chat_clm_hf"], "pcc": 0.99, "args": {"shape": "(16, 29, 64)"}},
    ),
    (
        Reshape916,
        [((16, 29, 29), torch.float32)],
        {
            "model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_chat_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 16, 29, 29)"},
        },
    ),
    (
        Reshape917,
        [((1, 16, 29, 29), torch.float32)],
        {"model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_chat_clm_hf"], "pcc": 0.99, "args": {"shape": "(16, 29, 29)"}},
    ),
    (
        Reshape918,
        [((1, 16, 64, 29), torch.float32)],
        {"model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_chat_clm_hf"], "pcc": 0.99, "args": {"shape": "(16, 64, 29)"}},
    ),
    (
        Reshape919,
        [((16, 29, 64), torch.float32)],
        {
            "model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_chat_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 16, 29, 64)"},
        },
    ),
    (
        Reshape912,
        [((1, 29, 16, 64), torch.float32)],
        {"model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_chat_clm_hf"], "pcc": 0.99, "args": {"shape": "(29, 1024)"}},
    ),
    (
        Reshape920,
        [((29, 2816), torch.float32)],
        {"model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_chat_clm_hf"], "pcc": 0.99, "args": {"shape": "(1, 29, 2816)"}},
    ),
    (
        Reshape921,
        [((1, 2016, 1, 1), torch.float32)],
        {
            "model_names": ["pt_regnet_facebook_regnet_y_080_img_cls_hf", "pt_regnet_regnet_y_8gf_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(1, 2016, 1, 1)"},
        },
    ),
    (
        Reshape922,
        [((1, 1920, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_regnet_regnet_x_8gf_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 1920, 1, 1)"},
        },
    ),
    (
        Reshape923,
        [((1280, 1, 3, 3), torch.float32)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1280, 1, 3, 3)"},
        },
    ),
    (
        Reshape924,
        [((2048, 1, 3, 3), torch.float32)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(2048, 1, 3, 3)"},
        },
    ),
    (
        Reshape925,
        [((8, 1), torch.int64)],
        {
            "model_names": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"],
            "pcc": 0.99,
            "args": {"shape": "(2, 4, 1)"},
        },
    ),
    (
        Reshape926,
        [((2, 1, 1), torch.int64)],
        {
            "model_names": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"],
            "pcc": 0.99,
            "args": {"shape": "(2, 1)"},
        },
    ),
    (
        Reshape201,
        [((1, 1536), torch.float32)],
        {
            "model_names": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 1536)"},
        },
    ),
    (
        Reshape927,
        [((2, 1, 1536), torch.float32)],
        {
            "model_names": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"],
            "pcc": 0.99,
            "args": {"shape": "(2, 1536)"},
        },
    ),
    (
        Reshape928,
        [((2, 1, 1536), torch.float32)],
        {
            "model_names": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"],
            "pcc": 0.99,
            "args": {"shape": "(2, 1, 24, 64)"},
        },
    ),
    (
        Reshape929,
        [((2, 1536), torch.float32)],
        {
            "model_names": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"],
            "pcc": 0.99,
            "args": {"shape": "(2, 1, 1536)"},
        },
    ),
    (
        Reshape928,
        [((2, 1536), torch.float32)],
        {
            "model_names": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"],
            "pcc": 0.99,
            "args": {"shape": "(2, 1, 24, 64)"},
        },
    ),
    (
        Reshape930,
        [((2, 24, 1, 64), torch.float32)],
        {
            "model_names": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"],
            "pcc": 0.99,
            "args": {"shape": "(48, 1, 64)"},
        },
    ),
    (
        Reshape931,
        [((48, 1, 64), torch.float32)],
        {
            "model_names": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"],
            "pcc": 0.99,
            "args": {"shape": "(2, 24, 1, 64)"},
        },
    ),
    (
        Reshape927,
        [((2, 1, 24, 64), torch.float32)],
        {
            "model_names": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"],
            "pcc": 0.99,
            "args": {"shape": "(2, 1536)"},
        },
    ),
    (
        Reshape932,
        [((2, 13), torch.int64)],
        {
            "model_names": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"],
            "pcc": 0.99,
            "args": {"shape": "(2, 13)"},
        },
    ),
    (
        Reshape933,
        [((2, 13, 768), torch.float32)],
        {
            "model_names": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"],
            "pcc": 0.99,
            "args": {"shape": "(26, 768)"},
        },
    ),
    (
        Reshape934,
        [((26, 768), torch.float32)],
        {
            "model_names": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"],
            "pcc": 0.99,
            "args": {"shape": "(2, 13, 12, 64)"},
        },
    ),
    (
        Reshape935,
        [((26, 768), torch.float32)],
        {
            "model_names": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"],
            "pcc": 0.99,
            "args": {"shape": "(2, 13, 768)"},
        },
    ),
    (
        Reshape936,
        [((2, 12, 13, 64), torch.float32)],
        {
            "model_names": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"],
            "pcc": 0.99,
            "args": {"shape": "(24, 13, 64)"},
        },
    ),
    (
        Reshape937,
        [((24, 13, 13), torch.float32)],
        {
            "model_names": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"],
            "pcc": 0.99,
            "args": {"shape": "(2, 12, 13, 13)"},
        },
    ),
    (
        Reshape938,
        [((2, 12, 13, 13), torch.float32)],
        {
            "model_names": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"],
            "pcc": 0.99,
            "args": {"shape": "(24, 13, 13)"},
        },
    ),
    (
        Reshape939,
        [((2, 12, 64, 13), torch.float32)],
        {
            "model_names": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"],
            "pcc": 0.99,
            "args": {"shape": "(24, 64, 13)"},
        },
    ),
    (
        Reshape940,
        [((24, 13, 64), torch.float32)],
        {
            "model_names": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"],
            "pcc": 0.99,
            "args": {"shape": "(2, 12, 13, 64)"},
        },
    ),
    (
        Reshape933,
        [((2, 13, 12, 64), torch.float32)],
        {
            "model_names": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"],
            "pcc": 0.99,
            "args": {"shape": "(26, 768)"},
        },
    ),
    (
        Reshape941,
        [((26, 3072), torch.float32)],
        {
            "model_names": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"],
            "pcc": 0.99,
            "args": {"shape": "(2, 13, 3072)"},
        },
    ),
    (
        Reshape942,
        [((2, 13, 3072), torch.float32)],
        {
            "model_names": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"],
            "pcc": 0.99,
            "args": {"shape": "(26, 3072)"},
        },
    ),
    (
        Reshape943,
        [((26, 1536), torch.float32)],
        {
            "model_names": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"],
            "pcc": 0.99,
            "args": {"shape": "(2, 13, 1536)"},
        },
    ),
    (
        Reshape944,
        [((26, 1536), torch.float32)],
        {
            "model_names": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"],
            "pcc": 0.99,
            "args": {"shape": "(2, 13, 24, 64)"},
        },
    ),
    (
        Reshape945,
        [((2, 13, 1536), torch.float32)],
        {
            "model_names": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"],
            "pcc": 0.99,
            "args": {"shape": "(26, 1536)"},
        },
    ),
    (
        Reshape946,
        [((2, 24, 13, 64), torch.float32)],
        {
            "model_names": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"],
            "pcc": 0.99,
            "args": {"shape": "(48, 13, 64)"},
        },
    ),
    (
        Reshape947,
        [((48, 1, 13), torch.float32)],
        {
            "model_names": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"],
            "pcc": 0.99,
            "args": {"shape": "(2, 24, 1, 13)"},
        },
    ),
    (
        Reshape948,
        [((2, 24, 1, 13), torch.float32)],
        {
            "model_names": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"],
            "pcc": 0.99,
            "args": {"shape": "(48, 1, 13)"},
        },
    ),
    (
        Reshape949,
        [((2, 6144), torch.float32)],
        {
            "model_names": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"],
            "pcc": 0.99,
            "args": {"shape": "(2, 1, 6144)"},
        },
    ),
    (
        Reshape950,
        [((2, 1, 6144), torch.float32)],
        {
            "model_names": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"],
            "pcc": 0.99,
            "args": {"shape": "(2, 6144)"},
        },
    ),
    (
        Reshape951,
        [((2, 2048), torch.float32)],
        {
            "model_names": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"],
            "pcc": 0.99,
            "args": {"shape": "(2, 1, 2048)"},
        },
    ),
    (
        Reshape952,
        [((2, 4, 1, 2048), torch.float32)],
        {
            "model_names": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"],
            "pcc": 0.99,
            "args": {"shape": "(8, 1, 2048)"},
        },
    ),
    (
        Reshape386,
        [((197, 1, 1024), torch.float32)],
        {"model_names": ["pt_vit_vit_l_16_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(197, 1024)"}},
    ),
    (
        Reshape953,
        [((197, 3072), torch.float32)],
        {"model_names": ["pt_vit_vit_l_16_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(197, 1, 3072)"}},
    ),
    (
        Reshape954,
        [((197, 1, 3072), torch.float32)],
        {"model_names": ["pt_vit_vit_l_16_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(197, 1, 3, 1024)"}},
    ),
    (
        Reshape955,
        [((1, 197, 1, 1024), torch.float32)],
        {"model_names": ["pt_vit_vit_l_16_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(197, 16, 64)"}},
    ),
    (
        Reshape386,
        [((197, 1, 16, 64), torch.float32)],
        {"model_names": ["pt_vit_vit_l_16_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(197, 1024)"}},
    ),
    (
        Reshape956,
        [((1, 6, 768), torch.float32)],
        {
            "model_names": ["onnx_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf"],
            "pcc": 0.99,
            "args": {"shape": "(6, 768)"},
        },
    ),
    (
        Reshape957,
        [((1, 6, 768), torch.float32)],
        {
            "model_names": ["onnx_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 6, 12, 64)"},
        },
    ),
    (
        Reshape958,
        [((6, 768), torch.float32)],
        {
            "model_names": ["onnx_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 6, 768)"},
        },
    ),
    (
        Reshape959,
        [((1, 12, 6, 64), torch.float32)],
        {
            "model_names": ["onnx_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf"],
            "pcc": 0.99,
            "args": {"shape": "(12, 6, 64)"},
        },
    ),
    (
        Reshape960,
        [((1, 12, 64, 6), torch.float32)],
        {
            "model_names": ["onnx_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf"],
            "pcc": 0.99,
            "args": {"shape": "(12, 64, 6)"},
        },
    ),
    (
        Reshape961,
        [((12, 6, 6), torch.float32)],
        {
            "model_names": ["onnx_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 12, 6, 6)"},
        },
    ),
    (
        Reshape962,
        [((1, 12, 6, 6), torch.float32)],
        {
            "model_names": ["onnx_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf"],
            "pcc": 0.99,
            "args": {"shape": "(12, 6, 6)"},
        },
    ),
    (
        Reshape963,
        [((12, 6, 64), torch.float32)],
        {
            "model_names": ["onnx_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 12, 6, 64)"},
        },
    ),
    (
        Reshape956,
        [((1, 6, 12, 64), torch.float32)],
        {
            "model_names": ["onnx_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf"],
            "pcc": 0.99,
            "args": {"shape": "(6, 768)"},
        },
    ),
    (
        Reshape964,
        [((1, 8, 768), torch.float32)],
        {
            "model_names": [
                "pd_blip_text_salesforce_blip_image_captioning_base_text_enc_padlenlp",
                "pd_bert_bert_base_uncased_seq_cls_padlenlp",
                "pd_chineseclip_text_ofa_sys_chinese_clip_vit_base_patch16_text_enc_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"shape": "(8, 768)"},
        },
    ),
    (
        Reshape965,
        [((1, 8, 768), torch.float32)],
        {
            "model_names": [
                "pd_blip_text_salesforce_blip_image_captioning_base_text_enc_padlenlp",
                "pd_bert_bert_base_uncased_seq_cls_padlenlp",
                "pd_chineseclip_text_ofa_sys_chinese_clip_vit_base_patch16_text_enc_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 8, 12, 64)"},
        },
    ),
    (
        Reshape966,
        [((8, 768), torch.float32)],
        {
            "model_names": [
                "pd_blip_text_salesforce_blip_image_captioning_base_text_enc_padlenlp",
                "pd_bert_bert_base_uncased_seq_cls_padlenlp",
                "pd_chineseclip_text_ofa_sys_chinese_clip_vit_base_patch16_text_enc_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 8, 768)"},
        },
    ),
    (
        Reshape967,
        [((1, 12, 8, 64), torch.float32)],
        {
            "model_names": [
                "pd_blip_text_salesforce_blip_image_captioning_base_text_enc_padlenlp",
                "pd_bert_bert_base_uncased_seq_cls_padlenlp",
                "pd_chineseclip_text_ofa_sys_chinese_clip_vit_base_patch16_text_enc_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"shape": "(12, 8, 64)"},
        },
    ),
    (
        Reshape968,
        [((1, 12, 64, 8), torch.float32)],
        {
            "model_names": [
                "pd_blip_text_salesforce_blip_image_captioning_base_text_enc_padlenlp",
                "pd_bert_bert_base_uncased_seq_cls_padlenlp",
                "pd_chineseclip_text_ofa_sys_chinese_clip_vit_base_patch16_text_enc_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"shape": "(12, 64, 8)"},
        },
    ),
    (
        Reshape969,
        [((12, 8, 8), torch.float32)],
        {
            "model_names": [
                "pd_blip_text_salesforce_blip_image_captioning_base_text_enc_padlenlp",
                "pd_bert_bert_base_uncased_seq_cls_padlenlp",
                "pd_chineseclip_text_ofa_sys_chinese_clip_vit_base_patch16_text_enc_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 12, 8, 8)"},
        },
    ),
    (
        Reshape970,
        [((1, 12, 8, 8), torch.float32)],
        {
            "model_names": [
                "pd_blip_text_salesforce_blip_image_captioning_base_text_enc_padlenlp",
                "pd_bert_bert_base_uncased_seq_cls_padlenlp",
                "pd_chineseclip_text_ofa_sys_chinese_clip_vit_base_patch16_text_enc_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"shape": "(12, 8, 8)"},
        },
    ),
    (
        Reshape971,
        [((12, 8, 64), torch.float32)],
        {
            "model_names": [
                "pd_blip_text_salesforce_blip_image_captioning_base_text_enc_padlenlp",
                "pd_bert_bert_base_uncased_seq_cls_padlenlp",
                "pd_chineseclip_text_ofa_sys_chinese_clip_vit_base_patch16_text_enc_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 12, 8, 64)"},
        },
    ),
    (
        Reshape964,
        [((1, 8, 12, 64), torch.float32)],
        {
            "model_names": [
                "pd_blip_text_salesforce_blip_image_captioning_base_text_enc_padlenlp",
                "pd_bert_bert_base_uncased_seq_cls_padlenlp",
                "pd_chineseclip_text_ofa_sys_chinese_clip_vit_base_patch16_text_enc_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"shape": "(8, 768)"},
        },
    ),
    (
        Reshape972,
        [((1, 10, 768), torch.float32)],
        {"model_names": ["pd_bert_bert_base_japanese_mlm_padlenlp"], "pcc": 0.99, "args": {"shape": "(10, 768)"}},
    ),
    (
        Reshape973,
        [((1, 10, 768), torch.float32)],
        {"model_names": ["pd_bert_bert_base_japanese_mlm_padlenlp"], "pcc": 0.99, "args": {"shape": "(1, 10, 12, 64)"}},
    ),
    (
        Reshape974,
        [((10, 768), torch.float32)],
        {"model_names": ["pd_bert_bert_base_japanese_mlm_padlenlp"], "pcc": 0.99, "args": {"shape": "(1, 10, 768)"}},
    ),
    (
        Reshape975,
        [((1, 12, 10, 64), torch.float32)],
        {"model_names": ["pd_bert_bert_base_japanese_mlm_padlenlp"], "pcc": 0.99, "args": {"shape": "(12, 10, 64)"}},
    ),
    (
        Reshape976,
        [((1, 12, 64, 10), torch.float32)],
        {"model_names": ["pd_bert_bert_base_japanese_mlm_padlenlp"], "pcc": 0.99, "args": {"shape": "(12, 64, 10)"}},
    ),
    (
        Reshape977,
        [((12, 10, 10), torch.float32)],
        {"model_names": ["pd_bert_bert_base_japanese_mlm_padlenlp"], "pcc": 0.99, "args": {"shape": "(1, 12, 10, 10)"}},
    ),
    (
        Reshape978,
        [((1, 12, 10, 10), torch.float32)],
        {"model_names": ["pd_bert_bert_base_japanese_mlm_padlenlp"], "pcc": 0.99, "args": {"shape": "(12, 10, 10)"}},
    ),
    (
        Reshape979,
        [((12, 10, 64), torch.float32)],
        {"model_names": ["pd_bert_bert_base_japanese_mlm_padlenlp"], "pcc": 0.99, "args": {"shape": "(1, 12, 10, 64)"}},
    ),
    (
        Reshape972,
        [((1, 10, 12, 64), torch.float32)],
        {"model_names": ["pd_bert_bert_base_japanese_mlm_padlenlp"], "pcc": 0.99, "args": {"shape": "(10, 768)"}},
    ),
    (
        Reshape980,
        [((25, 1, 2, 48), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"shape": "(25, 1, 96)"},
        },
    ),
    (
        Reshape30,
        [((128, 4096), torch.float32)],
        {
            "model_names": [
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
                "pt_albert_xxlarge_v2_token_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 128, 4096)"},
        },
    ),
    (
        Reshape981,
        [((1, 64, 128, 64), torch.float32)],
        {
            "model_names": [
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
                "pt_albert_xxlarge_v2_token_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(64, 128, 64)"},
        },
    ),
    (
        Reshape19,
        [((64, 128, 128), torch.float32)],
        {
            "model_names": [
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
                "pt_albert_xxlarge_v2_token_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 64, 128, 128)"},
        },
    ),
    (
        Reshape48,
        [((1, 64, 64, 128), torch.float32)],
        {
            "model_names": [
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
                "pt_albert_xxlarge_v2_token_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(64, 64, 128)"},
        },
    ),
    (
        Reshape982,
        [((1, 64, 64, 128), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(1, 8, 8, 8, 8, 128)"},
        },
    ),
    (
        Reshape46,
        [((1, 64, 64, 128), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(4096, 128)"}},
    ),
    (
        Reshape983,
        [((64, 128, 64), torch.float32)],
        {
            "model_names": [
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
                "pt_albert_xxlarge_v2_token_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 64, 128, 64)"},
        },
    ),
    (
        Reshape984,
        [((1, 32, 4608), torch.float32)],
        {"model_names": ["pt_bloom_bigscience_bloom_1b1_clm_hf"], "pcc": 0.99, "args": {"shape": "(1, 32, 16, 3, 96)"}},
    ),
    (
        Reshape985,
        [((1, 32, 16, 1, 96), torch.float32)],
        {"model_names": ["pt_bloom_bigscience_bloom_1b1_clm_hf"], "pcc": 0.99, "args": {"shape": "(1, 32, 16, 96)"}},
    ),
    (
        Reshape986,
        [((1, 16, 32, 96), torch.float32)],
        {"model_names": ["pt_bloom_bigscience_bloom_1b1_clm_hf"], "pcc": 0.99, "args": {"shape": "(16, 32, 96)"}},
    ),
    (
        Reshape987,
        [((1, 16, 32), torch.float32)],
        {"model_names": ["pt_bloom_bigscience_bloom_1b1_clm_hf"], "pcc": 0.99, "args": {"shape": "(16, 1, 32)"}},
    ),
    (
        Reshape988,
        [((16, 32, 96), torch.float32)],
        {"model_names": ["pt_bloom_bigscience_bloom_1b1_clm_hf"], "pcc": 0.99, "args": {"shape": "(1, 16, 32, 96)"}},
    ),
    (
        Reshape989,
        [((1, 32, 16, 96), torch.float32)],
        {"model_names": ["pt_bloom_bigscience_bloom_1b1_clm_hf"], "pcc": 0.99, "args": {"shape": "(32, 1536)"}},
    ),
    (
        Reshape990,
        [((32, 1536), torch.float32)],
        {"model_names": ["pt_bloom_bigscience_bloom_1b1_clm_hf"], "pcc": 0.99, "args": {"shape": "(1, 32, 1536)"}},
    ),
    (
        Reshape96,
        [((1, 256, 4, 256), torch.float32)],
        {
            "model_names": ["pt_codegen_salesforce_codegen_350m_nl_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 256, 16, 64)"},
        },
    ),
    (
        Reshape991,
        [((1, 256, 16, 16, 2), torch.float32)],
        {
            "model_names": ["pt_codegen_salesforce_codegen_350m_nl_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 256, 16, 32, 1)"},
        },
    ),
    (
        Reshape992,
        [((1, 16, 64, 256), torch.float32)],
        {
            "model_names": ["pt_codegen_salesforce_codegen_350m_nl_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(16, 64, 256)"},
        },
    ),
    (
        Reshape714,
        [((256, 4096), torch.float32)],
        {
            "model_names": ["pt_codegen_salesforce_codegen_350m_nl_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 256, 4096)"},
        },
    ),
    (
        Reshape993,
        [((1, 1664, 1, 1), torch.float32)],
        {
            "model_names": ["pt_densenet_densenet169_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(1, 1664, 1, 1)"},
        },
    ),
    (
        Reshape994,
        [((1, 384, 768), torch.float32)],
        {
            "model_names": ["pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf"],
            "pcc": 0.99,
            "args": {"shape": "(384, 768)"},
        },
    ),
    (
        Reshape995,
        [((1, 384, 768), torch.float32)],
        {
            "model_names": ["pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 384, 12, 64)"},
        },
    ),
    (
        Reshape996,
        [((384, 768), torch.float32)],
        {
            "model_names": ["pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 384, 768)"},
        },
    ),
    (
        Reshape997,
        [((1, 12, 384, 64), torch.float32)],
        {
            "model_names": ["pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf"],
            "pcc": 0.99,
            "args": {"shape": "(12, 384, 64)"},
        },
    ),
    (
        Reshape998,
        [((12, 384, 384), torch.float32)],
        {
            "model_names": ["pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 12, 384, 384)"},
        },
    ),
    (
        Reshape999,
        [((1, 384), torch.bool)],
        {
            "model_names": ["pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 1, 1, 384)"},
        },
    ),
    (
        Reshape1000,
        [((1, 12, 384, 384), torch.float32)],
        {
            "model_names": ["pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf"],
            "pcc": 0.99,
            "args": {"shape": "(12, 384, 384)"},
        },
    ),
    (
        Reshape1001,
        [((1, 12, 64, 384), torch.float32)],
        {
            "model_names": ["pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf"],
            "pcc": 0.99,
            "args": {"shape": "(12, 64, 384)"},
        },
    ),
    (
        Reshape1002,
        [((12, 384, 64), torch.float32)],
        {
            "model_names": ["pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 12, 384, 64)"},
        },
    ),
    (
        Reshape994,
        [((1, 384, 12, 64), torch.float32)],
        {
            "model_names": ["pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf"],
            "pcc": 0.99,
            "args": {"shape": "(384, 768)"},
        },
    ),
    (
        Reshape1003,
        [((2048, 1, 4), torch.float32)],
        {"model_names": ["pt_mamba_state_spaces_mamba_370m_hf_clm_hf"], "pcc": 0.99, "args": {"shape": "(2048, 1, 4)"}},
    ),
    (
        Reshape1004,
        [((1, 6, 2048), torch.float32)],
        {"model_names": ["pt_mamba_state_spaces_mamba_370m_hf_clm_hf"], "pcc": 0.99, "args": {"shape": "(6, 2048)"}},
    ),
    (
        Reshape1005,
        [((6, 64), torch.float32)],
        {"model_names": ["pt_mamba_state_spaces_mamba_370m_hf_clm_hf"], "pcc": 0.99, "args": {"shape": "(1, 6, 64)"}},
    ),
    (
        Reshape1006,
        [((1, 2048, 1, 16), torch.float32)],
        {
            "model_names": ["pt_mamba_state_spaces_mamba_370m_hf_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 2048, 16)"},
        },
    ),
    (
        Reshape1007,
        [((6, 16), torch.float32)],
        {
            "model_names": ["pt_mamba_state_spaces_mamba_370m_hf_clm_hf", "pt_mamba_state_spaces_mamba_790m_hf_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 6, 16)"},
        },
    ),
    (
        Reshape1008,
        [((1, 1, 16), torch.float32)],
        {
            "model_names": ["pt_mamba_state_spaces_mamba_370m_hf_clm_hf", "pt_mamba_state_spaces_mamba_790m_hf_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 16)"},
        },
    ),
    (
        Reshape110,
        [((1, 2048, 1), torch.float32)],
        {"model_names": ["pt_mamba_state_spaces_mamba_370m_hf_clm_hf"], "pcc": 0.99, "args": {"shape": "(1, 2048)"}},
    ),
    (
        Reshape1009,
        [((1, 768, 7, 7), torch.float32)],
        {"model_names": ["pt_mlp_mixer_mixer_b32_224_img_cls_timm"], "pcc": 0.99, "args": {"shape": "(1, 768, 49, 1)"}},
    ),
    (
        Reshape1010,
        [((1, 768, 7, 7), torch.float32)],
        {"model_names": ["pt_vit_vit_b_32_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(1, 768, 49)"}},
    ),
    (
        Reshape1011,
        [((256, 50272), torch.float32)],
        {"model_names": ["pt_opt_facebook_opt_350m_clm_hf"], "pcc": 0.99, "args": {"shape": "(1, 256, 50272)"}},
    ),
    (
        Reshape1012,
        [((1, 39, 1536), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf"], "pcc": 0.99, "args": {"shape": "(39, 1536)"}},
    ),
    (
        Reshape1013,
        [((1, 39, 1536), torch.float32)],
        {
            "model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 39, 12, 128)"},
        },
    ),
    (
        Reshape1014,
        [((39, 1536), torch.float32)],
        {
            "model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 39, 1536)"},
        },
    ),
    (
        Reshape1015,
        [((1, 12, 39, 128), torch.float32)],
        {
            "model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(12, 39, 128)"},
        },
    ),
    (
        Reshape1016,
        [((39, 256), torch.float32)],
        {
            "model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 39, 256)"},
        },
    ),
    (
        Reshape1017,
        [((1, 39, 256), torch.float32)],
        {
            "model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 39, 2, 128)"},
        },
    ),
    (
        Reshape1015,
        [((1, 2, 6, 39, 128), torch.float32)],
        {
            "model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(12, 39, 128)"},
        },
    ),
    (
        Reshape1018,
        [((1, 2, 6, 39, 128), torch.float32)],
        {
            "model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 12, 39, 128)"},
        },
    ),
    (
        Reshape1019,
        [((12, 39, 39), torch.float32)],
        {
            "model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 12, 39, 39)"},
        },
    ),
    (
        Reshape1020,
        [((1, 12, 39, 39), torch.float32)],
        {
            "model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(12, 39, 39)"},
        },
    ),
    (
        Reshape1021,
        [((1, 12, 128, 39), torch.float32)],
        {
            "model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(12, 128, 39)"},
        },
    ),
    (
        Reshape1018,
        [((12, 39, 128), torch.float32)],
        {
            "model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 12, 39, 128)"},
        },
    ),
    (
        Reshape1012,
        [((1, 39, 12, 128), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf"], "pcc": 0.99, "args": {"shape": "(39, 1536)"}},
    ),
    (
        Reshape1022,
        [((39, 8960), torch.float32)],
        {
            "model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 39, 8960)"},
        },
    ),
    (
        Reshape1023,
        [((1, 3712, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_regnet_facebook_regnet_y_320_img_cls_hf",
                "pt_regnet_regnet_y_32gf_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 3712, 1, 1)"},
        },
    ),
    (
        Reshape1024,
        [((1, 888, 1, 1), torch.float32)],
        {
            "model_names": ["pt_regnet_regnet_y_1_6gf_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(1, 888, 1, 1)"},
        },
    ),
    (
        Reshape1025,
        [((3136, 288), torch.float32)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(64, 49, 288)"},
        },
    ),
    (
        Reshape1026,
        [((64, 49, 288), torch.float32)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(64, 49, 3, 3, 32)"},
        },
    ),
    (
        Reshape754,
        [((1, 64, 3, 49, 32), torch.float32)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(64, 3, 49, 32)"},
        },
    ),
    (
        Reshape747,
        [((1, 64, 3, 49, 32), torch.float32)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(192, 49, 32)"},
        },
    ),
    (
        Reshape1027,
        [((3136, 384), torch.float32)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(1, 56, 56, 384)"},
        },
    ),
    (
        Reshape1028,
        [((1, 56, 56, 384), torch.float32)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(3136, 384)"},
        },
    ),
    (
        Reshape1029,
        [((784, 576), torch.float32)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(16, 49, 576)"},
        },
    ),
    (
        Reshape1030,
        [((16, 49, 576), torch.float32)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(16, 49, 3, 6, 32)"},
        },
    ),
    (
        Reshape770,
        [((1, 16, 6, 49, 32), torch.float32)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(16, 6, 49, 32)"},
        },
    ),
    (
        Reshape764,
        [((1, 16, 6, 49, 32), torch.float32)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(96, 49, 32)"},
        },
    ),
    (
        Reshape1031,
        [((784, 768), torch.float32)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(1, 28, 28, 768)"},
        },
    ),
    (
        Reshape1032,
        [((1, 28, 28, 768), torch.float32)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(784, 768)"},
        },
    ),
    (
        Reshape1033,
        [((196, 1152), torch.float32)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(4, 49, 1152)"},
        },
    ),
    (
        Reshape1034,
        [((4, 49, 1152), torch.float32)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(4, 49, 3, 12, 32)"},
        },
    ),
    (
        Reshape786,
        [((1, 4, 12, 49, 32), torch.float32)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(4, 12, 49, 32)"},
        },
    ),
    (
        Reshape780,
        [((1, 4, 12, 49, 32), torch.float32)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(48, 49, 32)"},
        },
    ),
    (
        Reshape1035,
        [((196, 1536), torch.float32)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(1, 14, 14, 1536)"},
        },
    ),
    (
        Reshape1036,
        [((1, 14, 14, 1536), torch.float32)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(196, 1536)"},
        },
    ),
    (
        Reshape1037,
        [((1, 7, 7, 768), torch.float32)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(1, 1, 7, 1, 7, 768)"},
        },
    ),
    (
        Reshape790,
        [((1, 7, 7, 768), torch.float32)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(49, 768)"},
        },
    ),
    (
        Reshape790,
        [((1, 1, 1, 7, 7, 768), torch.float32)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(49, 768)"},
        },
    ),
    (
        Reshape1038,
        [((49, 2304), torch.float32)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(1, 49, 2304)"},
        },
    ),
    (
        Reshape1039,
        [((1, 49, 2304), torch.float32)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(1, 49, 3, 24, 32)"},
        },
    ),
    (
        Reshape800,
        [((1, 1, 24, 49, 32), torch.float32)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(1, 24, 49, 32)"},
        },
    ),
    (
        Reshape795,
        [((1, 1, 24, 49, 32), torch.float32)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(24, 49, 32)"},
        },
    ),
    (
        Reshape794,
        [((1, 1, 7, 1, 7, 768), torch.float32)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(1, 7, 7, 768)"},
        },
    ),
    (
        Reshape1040,
        [((49, 3072), torch.float32)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(1, 7, 7, 3072)"},
        },
    ),
    (
        Reshape1041,
        [((1, 7, 7, 3072), torch.float32)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(49, 3072)"},
        },
    ),
    (
        Reshape1042,
        [((1, 201, 768), torch.float32)],
        {
            "model_names": ["pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf"],
            "pcc": 0.99,
            "args": {"shape": "(201, 768)"},
        },
    ),
    (
        Reshape1043,
        [((1, 201, 768), torch.float32)],
        {
            "model_names": ["pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 201, 12, 64)"},
        },
    ),
    (
        Reshape1044,
        [((201, 768), torch.float32)],
        {
            "model_names": ["pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 201, 768)"},
        },
    ),
    (
        Reshape1045,
        [((1, 12, 201, 64), torch.float32)],
        {
            "model_names": ["pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf"],
            "pcc": 0.99,
            "args": {"shape": "(12, 201, 64)"},
        },
    ),
    (
        Reshape1046,
        [((12, 201, 201), torch.float32)],
        {
            "model_names": ["pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 12, 201, 201)"},
        },
    ),
    (
        Reshape1047,
        [((1, 12, 201, 201), torch.float32)],
        {
            "model_names": ["pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf"],
            "pcc": 0.99,
            "args": {"shape": "(12, 201, 201)"},
        },
    ),
    (
        Reshape1048,
        [((1, 12, 64, 201), torch.float32)],
        {
            "model_names": ["pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf"],
            "pcc": 0.99,
            "args": {"shape": "(12, 64, 201)"},
        },
    ),
    (
        Reshape1049,
        [((12, 201, 64), torch.float32)],
        {
            "model_names": ["pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 12, 201, 64)"},
        },
    ),
    (
        Reshape1042,
        [((1, 201, 12, 64), torch.float32)],
        {
            "model_names": ["pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf"],
            "pcc": 0.99,
            "args": {"shape": "(201, 768)"},
        },
    ),
    (
        Reshape246,
        [((1, 512, 1, 1), torch.float32)],
        {"model_names": ["pt_vovnet_vovnet27s_img_cls_osmr"], "pcc": 0.99, "args": {"shape": "(1, 512)"}},
    ),
    (
        Reshape159,
        [((1, 512, 1, 1), torch.float32)],
        {
            "model_names": ["pd_resnet_18_img_cls_paddlemodels", "pd_resnet_34_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {"shape": "(1, 512, 1, 1)"},
        },
    ),
    (
        Reshape1050,
        [((1, 11, 312), torch.float32)],
        {"model_names": ["pd_albert_chinese_tiny_mlm_padlenlp"], "pcc": 0.99, "args": {"shape": "(11, 312)"}},
    ),
    (
        Reshape1051,
        [((1, 11, 312), torch.float32)],
        {"model_names": ["pd_albert_chinese_tiny_mlm_padlenlp"], "pcc": 0.99, "args": {"shape": "(1, 11, 12, 26)"}},
    ),
    (
        Reshape1052,
        [((11, 312), torch.float32)],
        {"model_names": ["pd_albert_chinese_tiny_mlm_padlenlp"], "pcc": 0.99, "args": {"shape": "(1, 11, 312)"}},
    ),
    (
        Reshape1053,
        [((1, 12, 11, 26), torch.float32)],
        {"model_names": ["pd_albert_chinese_tiny_mlm_padlenlp"], "pcc": 0.99, "args": {"shape": "(12, 11, 26)"}},
    ),
    (
        Reshape1054,
        [((1, 12, 26, 11), torch.float32)],
        {"model_names": ["pd_albert_chinese_tiny_mlm_padlenlp"], "pcc": 0.99, "args": {"shape": "(12, 26, 11)"}},
    ),
    (
        Reshape1055,
        [((12, 11, 26), torch.float32)],
        {"model_names": ["pd_albert_chinese_tiny_mlm_padlenlp"], "pcc": 0.99, "args": {"shape": "(1, 12, 11, 26)"}},
    ),
    (
        Reshape1050,
        [((1, 11, 12, 26), torch.float32)],
        {"model_names": ["pd_albert_chinese_tiny_mlm_padlenlp"], "pcc": 0.99, "args": {"shape": "(11, 312)"}},
    ),
    (
        Reshape1056,
        [((1, 15, 768), torch.float32)],
        {"model_names": ["pd_bert_bert_base_japanese_seq_cls_padlenlp"], "pcc": 0.99, "args": {"shape": "(15, 768)"}},
    ),
    (
        Reshape1057,
        [((1, 15, 768), torch.float32)],
        {
            "model_names": ["pd_bert_bert_base_japanese_seq_cls_padlenlp"],
            "pcc": 0.99,
            "args": {"shape": "(1, 15, 12, 64)"},
        },
    ),
    (
        Reshape1058,
        [((15, 768), torch.float32)],
        {
            "model_names": ["pd_bert_bert_base_japanese_seq_cls_padlenlp"],
            "pcc": 0.99,
            "args": {"shape": "(1, 15, 768)"},
        },
    ),
    (
        Reshape1059,
        [((1, 12, 15, 64), torch.float32)],
        {
            "model_names": ["pd_bert_bert_base_japanese_seq_cls_padlenlp"],
            "pcc": 0.99,
            "args": {"shape": "(12, 15, 64)"},
        },
    ),
    (
        Reshape1060,
        [((1, 12, 64, 15), torch.float32)],
        {
            "model_names": ["pd_bert_bert_base_japanese_seq_cls_padlenlp"],
            "pcc": 0.99,
            "args": {"shape": "(12, 64, 15)"},
        },
    ),
    (
        Reshape1061,
        [((12, 15, 15), torch.float32)],
        {
            "model_names": ["pd_bert_bert_base_japanese_seq_cls_padlenlp"],
            "pcc": 0.99,
            "args": {"shape": "(1, 12, 15, 15)"},
        },
    ),
    (
        Reshape1062,
        [((1, 12, 15, 15), torch.float32)],
        {
            "model_names": ["pd_bert_bert_base_japanese_seq_cls_padlenlp"],
            "pcc": 0.99,
            "args": {"shape": "(12, 15, 15)"},
        },
    ),
    (
        Reshape1063,
        [((12, 15, 64), torch.float32)],
        {
            "model_names": ["pd_bert_bert_base_japanese_seq_cls_padlenlp"],
            "pcc": 0.99,
            "args": {"shape": "(1, 12, 15, 64)"},
        },
    ),
    (
        Reshape1056,
        [((1, 15, 12, 64), torch.float32)],
        {"model_names": ["pd_bert_bert_base_japanese_seq_cls_padlenlp"], "pcc": 0.99, "args": {"shape": "(15, 768)"}},
    ),
    (
        Reshape1064,
        [((1, 120, 1, 12), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v4_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 120, 12, 1)"},
        },
    ),
    (
        Reshape1065,
        [((1, 12, 360), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v4_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 12, 3, 8, 15)"},
        },
    ),
    (
        Reshape1066,
        [((1, 8, 12, 15), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v4_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"shape": "(8, 12, 15)"},
        },
    ),
    (
        Reshape1067,
        [((1, 8, 15, 12), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v4_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"shape": "(8, 15, 12)"},
        },
    ),
    (
        Reshape1068,
        [((8, 12, 12), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v4_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 8, 12, 12)"},
        },
    ),
    (
        Reshape1069,
        [((1, 8, 12, 12), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v4_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"shape": "(8, 12, 12)"},
        },
    ),
    (
        Reshape1070,
        [((8, 12, 15), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v4_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 8, 12, 15)"},
        },
    ),
    (
        Reshape1071,
        [((1, 12, 8, 15), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v4_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"shape": "(12, 120)"},
        },
    ),
    (
        Reshape1072,
        [((12, 120), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v4_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 12, 120)"},
        },
    ),
    (
        Reshape1073,
        [((1, 12, 120), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v4_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"shape": "(1, 1, 12, 120)"},
        },
    ),
    (
        Reshape1074,
        [((1, 522, 2048), torch.float32)],
        {"model_names": ["pt_falcon3_tiiuae_falcon3_1b_base_clm_hf"], "pcc": 0.99, "args": {"shape": "(522, 2048)"}},
    ),
    (
        Reshape1075,
        [((522, 2048), torch.float32)],
        {
            "model_names": ["pt_falcon3_tiiuae_falcon3_1b_base_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 522, 8, 256)"},
        },
    ),
    (
        Reshape1076,
        [((522, 2048), torch.float32)],
        {"model_names": ["pt_falcon3_tiiuae_falcon3_1b_base_clm_hf"], "pcc": 0.99, "args": {"shape": "(1, 522, 2048)"}},
    ),
    (
        Reshape1077,
        [((1, 8, 522, 256), torch.float32)],
        {"model_names": ["pt_falcon3_tiiuae_falcon3_1b_base_clm_hf"], "pcc": 0.99, "args": {"shape": "(8, 522, 256)"}},
    ),
    (
        Reshape1078,
        [((522, 1024), torch.float32)],
        {
            "model_names": ["pt_falcon3_tiiuae_falcon3_1b_base_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 522, 4, 256)"},
        },
    ),
    (
        Reshape1077,
        [((1, 4, 2, 522, 256), torch.float32)],
        {"model_names": ["pt_falcon3_tiiuae_falcon3_1b_base_clm_hf"], "pcc": 0.99, "args": {"shape": "(8, 522, 256)"}},
    ),
    (
        Reshape1079,
        [((1, 4, 2, 522, 256), torch.float32)],
        {
            "model_names": ["pt_falcon3_tiiuae_falcon3_1b_base_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 8, 522, 256)"},
        },
    ),
    (
        Reshape1080,
        [((8, 522, 522), torch.float32)],
        {
            "model_names": ["pt_falcon3_tiiuae_falcon3_1b_base_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 8, 522, 522)"},
        },
    ),
    (
        Reshape1081,
        [((1, 8, 522, 522), torch.float32)],
        {"model_names": ["pt_falcon3_tiiuae_falcon3_1b_base_clm_hf"], "pcc": 0.99, "args": {"shape": "(8, 522, 522)"}},
    ),
    (
        Reshape1082,
        [((1, 8, 256, 522), torch.float32)],
        {"model_names": ["pt_falcon3_tiiuae_falcon3_1b_base_clm_hf"], "pcc": 0.99, "args": {"shape": "(8, 256, 522)"}},
    ),
    (
        Reshape1079,
        [((8, 522, 256), torch.float32)],
        {
            "model_names": ["pt_falcon3_tiiuae_falcon3_1b_base_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 8, 522, 256)"},
        },
    ),
    (
        Reshape1074,
        [((1, 522, 8, 256), torch.float32)],
        {"model_names": ["pt_falcon3_tiiuae_falcon3_1b_base_clm_hf"], "pcc": 0.99, "args": {"shape": "(522, 2048)"}},
    ),
    (
        Reshape1083,
        [((522, 8192), torch.float32)],
        {"model_names": ["pt_falcon3_tiiuae_falcon3_1b_base_clm_hf"], "pcc": 0.99, "args": {"shape": "(1, 522, 8192)"}},
    ),
    (
        Reshape1084,
        [((1, 64, 120, 160), torch.float32)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 64, 19200, 1)"},
        },
    ),
    (
        Reshape1085,
        [((1, 19200, 64), torch.float32)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 19200, 1, 64)"},
        },
    ),
    (
        Reshape1086,
        [((1, 19200, 64), torch.float32)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 120, 160, 64)"},
        },
    ),
    (
        Reshape1087,
        [((1, 64, 19200), torch.float32)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 64, 120, 160)"},
        },
    ),
    (
        Reshape1088,
        [((1, 64, 15, 20), torch.float32)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 64, 300)"},
        },
    ),
    (
        Reshape1089,
        [((1, 300, 64), torch.float32)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"shape": "(300, 64)"},
        },
    ),
    (
        Reshape1090,
        [((1, 300, 64), torch.float32)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 300, 1, 64)"},
        },
    ),
    (
        Reshape1091,
        [((300, 64), torch.float32)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 300, 64)"},
        },
    ),
    (
        Reshape1092,
        [((1, 19200, 300), torch.float32)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 1, 19200, 300)"},
        },
    ),
    (
        Reshape1093,
        [((1, 1, 19200, 300), torch.float32)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 19200, 300)"},
        },
    ),
    (
        Reshape1088,
        [((1, 1, 64, 300), torch.float32)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 64, 300)"},
        },
    ),
    (
        Reshape1094,
        [((1, 256, 19200), torch.float32)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 256, 120, 160)"},
        },
    ),
    (
        Reshape1095,
        [((1, 256, 120, 160), torch.float32)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 256, 19200, 1)"},
        },
    ),
    (
        Reshape1096,
        [((1, 128, 60, 80), torch.float32)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 128, 4800, 1)"},
        },
    ),
    (
        Reshape1097,
        [((1, 4800, 128), torch.float32)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 4800, 2, 64)"},
        },
    ),
    (
        Reshape1098,
        [((1, 4800, 128), torch.float32)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 60, 80, 128)"},
        },
    ),
    (
        Reshape1099,
        [((1, 2, 4800, 64), torch.float32)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"shape": "(2, 4800, 64)"},
        },
    ),
    (
        Reshape1100,
        [((1, 128, 4800), torch.float32)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 128, 60, 80)"},
        },
    ),
    (
        Reshape1101,
        [((1, 128, 15, 20), torch.float32)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 128, 300)"},
        },
    ),
    (
        Reshape1102,
        [((1, 300, 128), torch.float32)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"shape": "(300, 128)"},
        },
    ),
    (
        Reshape1103,
        [((1, 300, 128), torch.float32)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 300, 2, 64)"},
        },
    ),
    (
        Reshape1104,
        [((300, 128), torch.float32)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 300, 128)"},
        },
    ),
    (
        Reshape1105,
        [((1, 2, 300, 64), torch.float32)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"shape": "(2, 300, 64)"},
        },
    ),
    (
        Reshape1106,
        [((2, 4800, 300), torch.float32)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 2, 4800, 300)"},
        },
    ),
    (
        Reshape1107,
        [((1, 2, 4800, 300), torch.float32)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"shape": "(2, 4800, 300)"},
        },
    ),
    (
        Reshape1108,
        [((1, 2, 64, 300), torch.float32)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"shape": "(2, 64, 300)"},
        },
    ),
    (
        Reshape1109,
        [((2, 4800, 64), torch.float32)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 2, 4800, 64)"},
        },
    ),
    (
        Reshape1110,
        [((1, 4800, 2, 64), torch.float32)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"shape": "(4800, 128)"},
        },
    ),
    (
        Reshape1111,
        [((4800, 128), torch.float32)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 4800, 128)"},
        },
    ),
    (
        Reshape1112,
        [((1, 512, 4800), torch.float32)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 512, 60, 80)"},
        },
    ),
    (
        Reshape1113,
        [((1, 512, 60, 80), torch.float32)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 512, 4800, 1)"},
        },
    ),
    (
        Reshape1114,
        [((1, 320, 30, 40), torch.float32)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 320, 1200, 1)"},
        },
    ),
    (
        Reshape1115,
        [((1, 1200, 320), torch.float32)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 1200, 5, 64)"},
        },
    ),
    (
        Reshape1116,
        [((1, 1200, 320), torch.float32)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 30, 40, 320)"},
        },
    ),
    (
        Reshape1117,
        [((1, 5, 1200, 64), torch.float32)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"shape": "(5, 1200, 64)"},
        },
    ),
    (
        Reshape1118,
        [((1, 320, 1200), torch.float32)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 320, 30, 40)"},
        },
    ),
    (
        Reshape1119,
        [((1, 320, 15, 20), torch.float32)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 320, 300)"},
        },
    ),
    (
        Reshape1120,
        [((1, 300, 320), torch.float32)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"shape": "(300, 320)"},
        },
    ),
    (
        Reshape1121,
        [((1, 300, 320), torch.float32)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 300, 5, 64)"},
        },
    ),
    (
        Reshape1122,
        [((300, 320), torch.float32)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 300, 320)"},
        },
    ),
    (
        Reshape1123,
        [((1, 5, 300, 64), torch.float32)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"shape": "(5, 300, 64)"},
        },
    ),
    (
        Reshape1124,
        [((5, 1200, 300), torch.float32)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 5, 1200, 300)"},
        },
    ),
    (
        Reshape1125,
        [((1, 5, 1200, 300), torch.float32)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"shape": "(5, 1200, 300)"},
        },
    ),
    (
        Reshape1126,
        [((1, 5, 64, 300), torch.float32)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"shape": "(5, 64, 300)"},
        },
    ),
    (
        Reshape1127,
        [((5, 1200, 64), torch.float32)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 5, 1200, 64)"},
        },
    ),
    (
        Reshape1128,
        [((1, 1200, 5, 64), torch.float32)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1200, 320)"},
        },
    ),
    (
        Reshape1129,
        [((1200, 320), torch.float32)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 1200, 320)"},
        },
    ),
    (
        Reshape1130,
        [((1, 1280, 1200), torch.float32)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 1280, 30, 40)"},
        },
    ),
    (
        Reshape1131,
        [((1, 1280, 30, 40), torch.float32)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 1280, 1200, 1)"},
        },
    ),
    (
        Reshape1132,
        [((1, 512, 15, 20), torch.float32)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 512, 300, 1)"},
        },
    ),
    (
        Reshape1133,
        [((1, 300, 512), torch.float32)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"shape": "(300, 512)"},
        },
    ),
    (
        Reshape1134,
        [((1, 300, 512), torch.float32)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 300, 8, 64)"},
        },
    ),
    (
        Reshape1135,
        [((1, 300, 512), torch.float32)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 15, 20, 512)"},
        },
    ),
    (
        Reshape1136,
        [((300, 512), torch.float32)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 300, 512)"},
        },
    ),
    (
        Reshape1137,
        [((1, 8, 300, 64), torch.float32)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"shape": "(8, 300, 64)"},
        },
    ),
    (
        Reshape1138,
        [((8, 300, 300), torch.float32)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 8, 300, 300)"},
        },
    ),
    (
        Reshape1139,
        [((1, 8, 300, 300), torch.float32)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"shape": "(8, 300, 300)"},
        },
    ),
    (
        Reshape1140,
        [((1, 8, 64, 300), torch.float32)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"shape": "(8, 64, 300)"},
        },
    ),
    (
        Reshape1141,
        [((8, 300, 64), torch.float32)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 8, 300, 64)"},
        },
    ),
    (
        Reshape1133,
        [((1, 300, 8, 64), torch.float32)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"shape": "(300, 512)"},
        },
    ),
    (
        Reshape1142,
        [((1, 2048, 300), torch.float32)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 2048, 15, 20)"},
        },
    ),
    (
        Reshape1143,
        [((1, 2048, 15, 20), torch.float32)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 2048, 300, 1)"},
        },
    ),
    (
        Reshape1144,
        [((1, 1, 30, 40), torch.float32)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 30, 40)"},
        },
    ),
    (
        Reshape1145,
        [((1, 1, 60, 80), torch.float32)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 60, 80)"},
        },
    ),
    (
        Reshape1146,
        [((1, 1, 120, 160), torch.float32)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 120, 160)"},
        },
    ),
    (
        Reshape1147,
        [((3072, 1, 4), torch.float32)],
        {"model_names": ["pt_mamba_state_spaces_mamba_790m_hf_clm_hf"], "pcc": 0.99, "args": {"shape": "(3072, 1, 4)"}},
    ),
    (
        Reshape1148,
        [((1, 6, 3072), torch.float32)],
        {"model_names": ["pt_mamba_state_spaces_mamba_790m_hf_clm_hf"], "pcc": 0.99, "args": {"shape": "(6, 3072)"}},
    ),
    (
        Reshape1149,
        [((6, 96), torch.float32)],
        {"model_names": ["pt_mamba_state_spaces_mamba_790m_hf_clm_hf"], "pcc": 0.99, "args": {"shape": "(1, 6, 96)"}},
    ),
    (
        Reshape1150,
        [((1, 3072, 1, 16), torch.float32)],
        {
            "model_names": ["pt_mamba_state_spaces_mamba_790m_hf_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 3072, 16)"},
        },
    ),
    (
        Reshape1151,
        [((1, 3072, 1), torch.float32)],
        {"model_names": ["pt_mamba_state_spaces_mamba_790m_hf_clm_hf"], "pcc": 0.99, "args": {"shape": "(1, 3072)"}},
    ),
    (
        Reshape1152,
        [((1, 7, 2048), torch.float32)],
        {"model_names": ["pt_phi1_microsoft_phi_1_clm_hf"], "pcc": 0.99, "args": {"shape": "(7, 2048)"}},
    ),
    (
        Reshape1153,
        [((1, 7, 2048), torch.float32)],
        {"model_names": ["pt_phi1_microsoft_phi_1_clm_hf"], "pcc": 0.99, "args": {"shape": "(1, 7, 32, 64)"}},
    ),
    (
        Reshape1154,
        [((7, 2048), torch.float32)],
        {"model_names": ["pt_phi1_microsoft_phi_1_clm_hf"], "pcc": 0.99, "args": {"shape": "(1, 7, 2048)"}},
    ),
    (
        Reshape1155,
        [((1, 32, 7, 64), torch.float32)],
        {"model_names": ["pt_phi1_microsoft_phi_1_clm_hf"], "pcc": 0.99, "args": {"shape": "(32, 7, 64)"}},
    ),
    (
        Reshape1156,
        [((32, 7, 7), torch.float32)],
        {"model_names": ["pt_phi1_microsoft_phi_1_clm_hf"], "pcc": 0.99, "args": {"shape": "(1, 32, 7, 7)"}},
    ),
    (
        Reshape1157,
        [((1, 32, 7, 7), torch.float32)],
        {"model_names": ["pt_phi1_microsoft_phi_1_clm_hf"], "pcc": 0.99, "args": {"shape": "(32, 7, 7)"}},
    ),
    (
        Reshape1158,
        [((1, 32, 64, 7), torch.float32)],
        {"model_names": ["pt_phi1_microsoft_phi_1_clm_hf"], "pcc": 0.99, "args": {"shape": "(32, 64, 7)"}},
    ),
    (
        Reshape1159,
        [((32, 7, 64), torch.float32)],
        {"model_names": ["pt_phi1_microsoft_phi_1_clm_hf"], "pcc": 0.99, "args": {"shape": "(1, 32, 7, 64)"}},
    ),
    (
        Reshape1152,
        [((1, 7, 32, 64), torch.float32)],
        {"model_names": ["pt_phi1_microsoft_phi_1_clm_hf"], "pcc": 0.99, "args": {"shape": "(7, 2048)"}},
    ),
    (
        Reshape1160,
        [((7, 8192), torch.float32)],
        {"model_names": ["pt_phi1_microsoft_phi_1_clm_hf"], "pcc": 0.99, "args": {"shape": "(1, 7, 8192)"}},
    ),
    (
        Reshape1161,
        [((1, 29, 1536), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99, "args": {"shape": "(29, 1536)"}},
    ),
    (
        Reshape1162,
        [((1, 29, 1536), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99, "args": {"shape": "(1, 29, 12, 128)"}},
    ),
    (
        Reshape1163,
        [((29, 1536), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99, "args": {"shape": "(1, 29, 1536)"}},
    ),
    (
        Reshape1164,
        [((1, 12, 29, 128), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99, "args": {"shape": "(12, 29, 128)"}},
    ),
    (
        Reshape1165,
        [((29, 256), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99, "args": {"shape": "(1, 29, 256)"}},
    ),
    (
        Reshape1166,
        [((1, 29, 256), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99, "args": {"shape": "(1, 29, 2, 128)"}},
    ),
    (
        Reshape1164,
        [((1, 2, 6, 29, 128), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99, "args": {"shape": "(12, 29, 128)"}},
    ),
    (
        Reshape1167,
        [((1, 2, 6, 29, 128), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99, "args": {"shape": "(1, 12, 29, 128)"}},
    ),
    (
        Reshape1168,
        [((12, 29, 29), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99, "args": {"shape": "(1, 12, 29, 29)"}},
    ),
    (
        Reshape1169,
        [((1, 12, 29, 29), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99, "args": {"shape": "(12, 29, 29)"}},
    ),
    (
        Reshape1170,
        [((1, 12, 128, 29), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99, "args": {"shape": "(12, 128, 29)"}},
    ),
    (
        Reshape1167,
        [((12, 29, 128), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99, "args": {"shape": "(1, 12, 29, 128)"}},
    ),
    (
        Reshape1161,
        [((1, 29, 12, 128), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99, "args": {"shape": "(29, 1536)"}},
    ),
    (
        Reshape1171,
        [((29, 8960), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99, "args": {"shape": "(1, 29, 8960)"}},
    ),
    (
        Reshape1172,
        [((61, 2048), torch.float32)],
        {"model_names": ["pt_t5_google_flan_t5_base_text_gen_hf"], "pcc": 0.99, "args": {"shape": "(1, 61, 2048)"}},
    ),
    (
        Reshape1173,
        [((1, 2048), torch.float32)],
        {"model_names": ["pt_t5_google_flan_t5_base_text_gen_hf"], "pcc": 0.99, "args": {"shape": "(1, 1, 2048)"}},
    ),
    (
        Reshape1174,
        [((1, 68, 56, 80), torch.float32)],
        {
            "model_names": ["pt_yolo_v6_yolov6l_obj_det_torchhub", "pt_yolo_v6_yolov6m_obj_det_torchhub"],
            "pcc": 0.99,
            "args": {"shape": "(1, 4, 17, 4480)"},
        },
    ),
    (
        Reshape439,
        [((1, 1, 4, 4480), torch.float32)],
        {
            "model_names": ["pt_yolo_v6_yolov6l_obj_det_torchhub", "pt_yolo_v6_yolov6m_obj_det_torchhub"],
            "pcc": 0.99,
            "args": {"shape": "(1, 4, 4480)"},
        },
    ),
    (
        Reshape1175,
        [((1, 68, 28, 40), torch.float32)],
        {
            "model_names": ["pt_yolo_v6_yolov6l_obj_det_torchhub", "pt_yolo_v6_yolov6m_obj_det_torchhub"],
            "pcc": 0.99,
            "args": {"shape": "(1, 4, 17, 1120)"},
        },
    ),
    (
        Reshape440,
        [((1, 1, 4, 1120), torch.float32)],
        {
            "model_names": ["pt_yolo_v6_yolov6l_obj_det_torchhub", "pt_yolo_v6_yolov6m_obj_det_torchhub"],
            "pcc": 0.99,
            "args": {"shape": "(1, 4, 1120)"},
        },
    ),
    (
        Reshape1176,
        [((1, 68, 14, 20), torch.float32)],
        {
            "model_names": ["pt_yolo_v6_yolov6l_obj_det_torchhub", "pt_yolo_v6_yolov6m_obj_det_torchhub"],
            "pcc": 0.99,
            "args": {"shape": "(1, 4, 17, 280)"},
        },
    ),
    (
        Reshape441,
        [((1, 1, 4, 280), torch.float32)],
        {
            "model_names": ["pt_yolo_v6_yolov6l_obj_det_torchhub", "pt_yolo_v6_yolov6m_obj_det_torchhub"],
            "pcc": 0.99,
            "args": {"shape": "(1, 4, 280)"},
        },
    ),
    (
        Reshape1177,
        [((1, 128, 3, 3), torch.float32)],
        {"model_names": ["pd_googlenet_base_img_cls_paddlemodels"], "pcc": 0.99, "args": {"shape": "(1, 1152, 1, 1)"}},
    ),
    (
        Reshape1178,
        [((14, 1), torch.float32)],
        {"model_names": ["pt_albert_twmkn9_albert_base_v2_squad2_qa_hf"], "pcc": 0.99, "args": {"shape": "(1, 14, 1)"}},
    ),
    (
        Reshape1179,
        [((1, 588, 2048), torch.float32)],
        {
            "model_names": ["pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf"],
            "pcc": 0.99,
            "args": {"shape": "(588, 2048)"},
        },
    ),
    (
        Reshape1180,
        [((588, 2048), torch.float32)],
        {
            "model_names": ["pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 588, 16, 128)"},
        },
    ),
    (
        Reshape1181,
        [((588, 2048), torch.float32)],
        {
            "model_names": ["pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 588, 2048)"},
        },
    ),
    (
        Reshape1182,
        [((1, 16, 588, 128), torch.float32)],
        {
            "model_names": ["pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf"],
            "pcc": 0.99,
            "args": {"shape": "(16, 588, 128)"},
        },
    ),
    (
        Reshape1183,
        [((16, 588, 588), torch.float32)],
        {
            "model_names": ["pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 16, 588, 588)"},
        },
    ),
    (
        Reshape1184,
        [((1, 16, 588, 588), torch.float32)],
        {
            "model_names": ["pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf"],
            "pcc": 0.99,
            "args": {"shape": "(16, 588, 588)"},
        },
    ),
    (
        Reshape1185,
        [((1, 16, 128, 588), torch.float32)],
        {
            "model_names": ["pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf"],
            "pcc": 0.99,
            "args": {"shape": "(16, 128, 588)"},
        },
    ),
    (
        Reshape1186,
        [((16, 588, 128), torch.float32)],
        {
            "model_names": ["pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 16, 588, 128)"},
        },
    ),
    (
        Reshape1179,
        [((1, 588, 16, 128), torch.float32)],
        {
            "model_names": ["pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf"],
            "pcc": 0.99,
            "args": {"shape": "(588, 2048)"},
        },
    ),
    (
        Reshape1187,
        [((588, 5504), torch.float32)],
        {
            "model_names": ["pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 588, 5504)"},
        },
    ),
    (
        Reshape1188,
        [((528, 1, 3, 3), torch.float32)],
        {
            "model_names": ["pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm"],
            "pcc": 0.99,
            "args": {"shape": "(528, 1, 3, 3)"},
        },
    ),
    (
        Reshape1189,
        [((528, 1, 5, 5), torch.float32)],
        {
            "model_names": ["pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm"],
            "pcc": 0.99,
            "args": {"shape": "(528, 1, 5, 5)"},
        },
    ),
    (
        Reshape1190,
        [((720, 1, 5, 5), torch.float32)],
        {
            "model_names": ["pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm"],
            "pcc": 0.99,
            "args": {"shape": "(720, 1, 5, 5)"},
        },
    ),
    (
        Reshape1191,
        [((1248, 1, 5, 5), torch.float32)],
        {
            "model_names": ["pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm"],
            "pcc": 0.99,
            "args": {"shape": "(1248, 1, 5, 5)"},
        },
    ),
    (
        Reshape1192,
        [((1248, 1, 3, 3), torch.float32)],
        {
            "model_names": ["pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm"],
            "pcc": 0.99,
            "args": {"shape": "(1248, 1, 3, 3)"},
        },
    ),
    (
        Reshape1193,
        [((1, 768, 8, 32), torch.float32)],
        {
            "model_names": ["pt_mgp_alibaba_damo_mgp_str_base_scene_text_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 768, 256, 1)"},
        },
    ),
    (
        Reshape1194,
        [((1, 257, 2304), torch.float32)],
        {
            "model_names": ["pt_mgp_alibaba_damo_mgp_str_base_scene_text_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 257, 3, 12, 64)"},
        },
    ),
    (
        Reshape1195,
        [((1, 1, 12, 257, 64), torch.float32)],
        {
            "model_names": ["pt_mgp_alibaba_damo_mgp_str_base_scene_text_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(12, 257, 64)"},
        },
    ),
    (
        Reshape1196,
        [((1, 1, 12, 257, 64), torch.float32)],
        {
            "model_names": ["pt_mgp_alibaba_damo_mgp_str_base_scene_text_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 12, 257, 64)"},
        },
    ),
    (
        Reshape1197,
        [((12, 257, 257), torch.float32)],
        {
            "model_names": ["pt_mgp_alibaba_damo_mgp_str_base_scene_text_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 12, 257, 257)"},
        },
    ),
    (
        Reshape1198,
        [((1, 12, 257, 257), torch.float32)],
        {
            "model_names": ["pt_mgp_alibaba_damo_mgp_str_base_scene_text_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(12, 257, 257)"},
        },
    ),
    (
        Reshape1199,
        [((1, 12, 64, 257), torch.float32)],
        {
            "model_names": ["pt_mgp_alibaba_damo_mgp_str_base_scene_text_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(12, 64, 257)"},
        },
    ),
    (
        Reshape1196,
        [((12, 257, 64), torch.float32)],
        {
            "model_names": ["pt_mgp_alibaba_damo_mgp_str_base_scene_text_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 12, 257, 64)"},
        },
    ),
    (
        Reshape1200,
        [((1, 257, 12, 64), torch.float32)],
        {
            "model_names": ["pt_mgp_alibaba_damo_mgp_str_base_scene_text_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(257, 768)"},
        },
    ),
    (
        Reshape1201,
        [((257, 768), torch.float32)],
        {
            "model_names": ["pt_mgp_alibaba_damo_mgp_str_base_scene_text_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 257, 768)"},
        },
    ),
    (
        Reshape1202,
        [((1, 27, 257, 1), torch.float32)],
        {
            "model_names": ["pt_mgp_alibaba_damo_mgp_str_base_scene_text_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 27, 257, 1)"},
        },
    ),
    (
        Reshape1203,
        [((1, 768, 257, 1), torch.float32)],
        {
            "model_names": ["pt_mgp_alibaba_damo_mgp_str_base_scene_text_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 768, 257, 1)"},
        },
    ),
    (
        Reshape1204,
        [((432, 1, 3, 3), torch.float32)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(432, 1, 3, 3)"},
        },
    ),
    (
        Reshape1205,
        [((720, 1, 3, 3), torch.float32)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(720, 1, 3, 3)"},
        },
    ),
    (
        Reshape1206,
        [((88, 1, 3, 3), torch.float32)],
        {
            "model_names": [
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
            ],
            "pcc": 0.99,
            "args": {"shape": "(88, 1, 3, 3)"},
        },
    ),
    (
        Reshape1207,
        [((96, 1, 5, 5), torch.float32)],
        {
            "model_names": [
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
            ],
            "pcc": 0.99,
            "args": {"shape": "(96, 1, 5, 5)"},
        },
    ),
    (
        Reshape254,
        [((1, 32, 768), torch.float32)],
        {
            "model_names": ["pt_opt_facebook_opt_125m_qa_hf", "pt_opt_facebook_opt_125m_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(32, 768)"},
        },
    ),
    (
        Reshape258,
        [((1, 32, 768), torch.float32)],
        {
            "model_names": ["pt_opt_facebook_opt_125m_qa_hf", "pt_opt_facebook_opt_125m_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 32, 12, 64)"},
        },
    ),
    (
        Reshape1208,
        [((32, 768), torch.float32)],
        {
            "model_names": ["pt_opt_facebook_opt_125m_qa_hf", "pt_opt_facebook_opt_125m_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 32, 768)"},
        },
    ),
    (
        Reshape1209,
        [((12, 32, 32), torch.float32)],
        {
            "model_names": ["pt_opt_facebook_opt_125m_qa_hf", "pt_opt_facebook_opt_125m_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 12, 32, 32)"},
        },
    ),
    (
        Reshape1210,
        [((1, 12, 32, 32), torch.float32)],
        {
            "model_names": ["pt_opt_facebook_opt_125m_qa_hf", "pt_opt_facebook_opt_125m_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(12, 32, 32)"},
        },
    ),
    (
        Reshape251,
        [((12, 32, 64), torch.float32)],
        {
            "model_names": ["pt_opt_facebook_opt_125m_qa_hf", "pt_opt_facebook_opt_125m_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 12, 32, 64)"},
        },
    ),
    (
        Reshape510,
        [((1, 1, 2), torch.float32)],
        {"model_names": ["pt_phi1_microsoft_phi_1_seq_cls_hf"], "pcc": 0.99, "args": {"shape": "(1, 2)"}},
    ),
    (
        Reshape1211,
        [((1, 35, 896), torch.float32)],
        {"model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf"], "pcc": 0.99, "args": {"shape": "(35, 896)"}},
    ),
    (
        Reshape1212,
        [((1, 35, 896), torch.float32)],
        {
            "model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 35, 14, 64)"},
        },
    ),
    (
        Reshape1213,
        [((35, 896), torch.float32)],
        {
            "model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 35, 896)"},
        },
    ),
    (
        Reshape1214,
        [((1, 14, 35, 64), torch.float32)],
        {
            "model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(14, 35, 64)"},
        },
    ),
    (
        Reshape1215,
        [((35, 128), torch.float32)],
        {
            "model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 35, 128)"},
        },
    ),
    (
        Reshape1216,
        [((1, 35, 128), torch.float32)],
        {
            "model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 35, 2, 64)"},
        },
    ),
    (
        Reshape1214,
        [((1, 2, 7, 35, 64), torch.float32)],
        {
            "model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(14, 35, 64)"},
        },
    ),
    (
        Reshape1217,
        [((1, 2, 7, 35, 64), torch.float32)],
        {
            "model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 14, 35, 64)"},
        },
    ),
    (
        Reshape1218,
        [((14, 35, 35), torch.float32)],
        {
            "model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 14, 35, 35)"},
        },
    ),
    (
        Reshape1219,
        [((1, 14, 35, 35), torch.float32)],
        {
            "model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(14, 35, 35)"},
        },
    ),
    (
        Reshape1220,
        [((1, 14, 64, 35), torch.float32)],
        {
            "model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(14, 64, 35)"},
        },
    ),
    (
        Reshape1217,
        [((14, 35, 64), torch.float32)],
        {
            "model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 14, 35, 64)"},
        },
    ),
    (
        Reshape1211,
        [((1, 35, 14, 64), torch.float32)],
        {"model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf"], "pcc": 0.99, "args": {"shape": "(35, 896)"}},
    ),
    (
        Reshape1221,
        [((35, 4864), torch.float32)],
        {
            "model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 35, 4864)"},
        },
    ),
    (
        Reshape1222,
        [((1, 912, 1, 1), torch.float32)],
        {
            "model_names": ["pt_regnet_regnet_x_1_6gf_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(1, 912, 1, 1)"},
        },
    ),
    (
        Reshape1223,
        [((1, 1512, 1, 1), torch.float32)],
        {
            "model_names": ["pt_regnet_regnet_y_3_2gf_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(1, 1512, 1, 1)"},
        },
    ),
    (
        Reshape46,
        [((1, 8, 8, 8, 8, 128), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(4096, 128)"}},
    ),
    (
        Reshape33,
        [((1, 8, 8, 8, 8, 128), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(1, 64, 64, 128)"}},
    ),
    (
        Reshape1224,
        [((64, 64, 384), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(64, 64, 3, 4, 32)"},
        },
    ),
    (
        Reshape1225,
        [((1, 64, 4, 64, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(64, 4, 64, 32)"}},
    ),
    (
        Reshape1226,
        [((64, 4, 64, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(256, 64, 32)"}},
    ),
    (
        Reshape1227,
        [((256, 64, 64), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(64, 4, 64, 64)"}},
    ),
    (
        Reshape1228,
        [((225, 4), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(225, 4)"}},
    ),
    (
        Reshape1229,
        [((4096, 4), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(64, 64, 4)"}},
    ),
    (
        Reshape1230,
        [((64, 4, 64, 64), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(256, 64, 64)"}},
    ),
    (
        Reshape1231,
        [((64, 4, 64, 64), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(1, 64, 4, 64, 64)"},
        },
    ),
    (
        Reshape1232,
        [((64, 4, 32, 64), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(256, 32, 64)"}},
    ),
    (
        Reshape1225,
        [((256, 64, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(64, 4, 64, 32)"}},
    ),
    (
        Reshape46,
        [((64, 64, 4, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(4096, 128)"}},
    ),
    (
        Reshape982,
        [((64, 64, 128), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(1, 8, 8, 8, 8, 128)"},
        },
    ),
    (
        Reshape1233,
        [((4096, 512), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(1, 64, 64, 512)"}},
    ),
    (
        Reshape1234,
        [((1, 64, 64, 512), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(4096, 512)"}},
    ),
    (
        Reshape1227,
        [((1, 64, 4, 64, 64), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(64, 4, 64, 64)"}},
    ),
    (
        Reshape1235,
        [((1, 32, 32, 512), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(1024, 512)"}},
    ),
    (
        Reshape1236,
        [((1024, 256), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(1, 32, 32, 256)"}},
    ),
    (
        Reshape992,
        [((1024, 256), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(16, 64, 256)"}},
    ),
    (
        Reshape1237,
        [((1, 32, 32, 256), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(1, 4, 8, 4, 8, 256)"},
        },
    ),
    (
        Reshape1238,
        [((1, 32, 32, 256), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(1024, 256)"}},
    ),
    (
        Reshape1238,
        [((1, 4, 4, 8, 8, 256), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(1024, 256)"}},
    ),
    (
        Reshape1239,
        [((16, 64, 768), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(16, 64, 3, 8, 32)"},
        },
    ),
    (
        Reshape1240,
        [((1, 16, 8, 64, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(16, 8, 64, 32)"}},
    ),
    (
        Reshape1241,
        [((16, 8, 64, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(128, 64, 32)"}},
    ),
    (
        Reshape1242,
        [((128, 64, 64), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(16, 8, 64, 64)"}},
    ),
    (
        Reshape1243,
        [((225, 8), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(225, 8)"}},
    ),
    (
        Reshape1244,
        [((4096, 8), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(64, 64, 8)"}},
    ),
    (
        Reshape1245,
        [((16, 8, 64, 64), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(128, 64, 64)"}},
    ),
    (
        Reshape1246,
        [((16, 8, 64, 64), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(1, 16, 8, 64, 64)"},
        },
    ),
    (
        Reshape1247,
        [((16, 8, 32, 64), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(128, 32, 64)"}},
    ),
    (
        Reshape1240,
        [((128, 64, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(16, 8, 64, 32)"}},
    ),
    (
        Reshape1238,
        [((16, 64, 8, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(1024, 256)"}},
    ),
    (
        Reshape1248,
        [((16, 64, 256), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(1, 4, 4, 8, 8, 256)"},
        },
    ),
    (
        Reshape1236,
        [((1, 4, 8, 4, 8, 256), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(1, 32, 32, 256)"}},
    ),
    (
        Reshape1249,
        [((1024, 1024), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(1, 32, 32, 1024)"}},
    ),
    (
        Reshape1250,
        [((1, 32, 32, 1024), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(1024, 1024)"}},
    ),
    (
        Reshape1242,
        [((1, 16, 8, 64, 64), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(16, 8, 64, 64)"}},
    ),
    (
        Reshape95,
        [((1, 16, 16, 1024), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(256, 1024)"}},
    ),
    (
        Reshape1251,
        [((1, 16, 16, 512), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(1, 2, 8, 2, 8, 512)"},
        },
    ),
    (
        Reshape74,
        [((1, 16, 16, 512), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(256, 512)"}},
    ),
    (
        Reshape74,
        [((1, 2, 2, 8, 8, 512), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(256, 512)"}},
    ),
    (
        Reshape1252,
        [((4, 64, 1536), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(4, 64, 3, 16, 32)"},
        },
    ),
    (
        Reshape1253,
        [((1, 4, 16, 64, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(4, 16, 64, 32)"}},
    ),
    (
        Reshape1254,
        [((4, 16, 64, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(64, 64, 32)"}},
    ),
    (
        Reshape1255,
        [((64, 64, 64), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(4, 16, 64, 64)"}},
    ),
    (
        Reshape1256,
        [((225, 16), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(225, 16)"}},
    ),
    (
        Reshape1257,
        [((4096, 16), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(64, 64, 16)"}},
    ),
    (
        Reshape1258,
        [((4, 16, 64, 64), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(64, 64, 64)"}},
    ),
    (
        Reshape1259,
        [((4, 16, 64, 64), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(1, 4, 16, 64, 64)"},
        },
    ),
    (
        Reshape1260,
        [((4, 16, 32, 64), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(64, 32, 64)"}},
    ),
    (
        Reshape1253,
        [((64, 64, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(4, 16, 64, 32)"}},
    ),
    (
        Reshape74,
        [((4, 64, 16, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(256, 512)"}},
    ),
    (
        Reshape1261,
        [((4, 64, 512), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(1, 2, 2, 8, 8, 512)"},
        },
    ),
    (
        Reshape77,
        [((1, 2, 8, 2, 8, 512), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(1, 16, 16, 512)"}},
    ),
    (
        Reshape650,
        [((1, 16, 16, 2048), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(256, 2048)"}},
    ),
    (
        Reshape1255,
        [((1, 4, 16, 64, 64), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(4, 16, 64, 64)"}},
    ),
    (
        Reshape1262,
        [((1, 8, 8, 2048), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(64, 2048)"}},
    ),
    (
        Reshape1263,
        [((64, 1024), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(1, 8, 8, 1024)"}},
    ),
    (
        Reshape1264,
        [((64, 1024), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(1, 64, 1024)"}},
    ),
    (
        Reshape1265,
        [((1, 8, 8, 1024), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(1, 1, 8, 1, 8, 1024)"},
        },
    ),
    (
        Reshape1266,
        [((1, 8, 8, 1024), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(64, 1024)"}},
    ),
    (
        Reshape1266,
        [((1, 1, 1, 8, 8, 1024), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(64, 1024)"}},
    ),
    (
        Reshape1267,
        [((1, 64, 3072), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(1, 64, 3, 32, 32)"},
        },
    ),
    (
        Reshape1268,
        [((1, 1, 32, 64, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(1, 32, 64, 32)"}},
    ),
    (
        Reshape1269,
        [((1, 32, 64, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(32, 64, 32)"}},
    ),
    (
        Reshape1270,
        [((32, 64, 64), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(1, 32, 64, 64)"}},
    ),
    (
        Reshape1271,
        [((225, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(225, 32)"}},
    ),
    (
        Reshape1254,
        [((4096, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(64, 64, 32)"}},
    ),
    (
        Reshape1272,
        [((1, 32, 64, 64), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(32, 64, 64)"}},
    ),
    (
        Reshape1268,
        [((32, 64, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(1, 32, 64, 32)"}},
    ),
    (
        Reshape1266,
        [((1, 64, 32, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(64, 1024)"}},
    ),
    (
        Reshape1273,
        [((1, 64, 1024), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(1, 1, 1, 8, 8, 1024)"},
        },
    ),
    (
        Reshape1263,
        [((1, 1, 8, 1, 8, 1024), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(1, 8, 8, 1024)"}},
    ),
    (
        Reshape1274,
        [((64, 4096), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(1, 8, 8, 4096)"}},
    ),
    (
        Reshape1275,
        [((1, 8, 8, 4096), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(64, 4096)"}},
    ),
    (
        Reshape1276,
        [((1, 61, 1024), torch.float32)],
        {
            "model_names": ["pt_t5_google_flan_t5_large_text_gen_hf", "pt_t5_t5_large_text_gen_hf"],
            "pcc": 0.99,
            "args": {"shape": "(61, 1024)"},
        },
    ),
    (
        Reshape1277,
        [((1, 16, 61, 64), torch.float32)],
        {
            "model_names": ["pt_t5_google_flan_t5_large_text_gen_hf", "pt_t5_t5_large_text_gen_hf"],
            "pcc": 0.99,
            "args": {"shape": "(16, 61, 64)"},
        },
    ),
    (
        Reshape1278,
        [((16, 61, 61), torch.float32)],
        {
            "model_names": ["pt_t5_google_flan_t5_large_text_gen_hf", "pt_t5_t5_large_text_gen_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 16, 61, 61)"},
        },
    ),
    (
        Reshape1279,
        [((1, 16, 61, 61), torch.float32)],
        {
            "model_names": ["pt_t5_google_flan_t5_large_text_gen_hf", "pt_t5_t5_large_text_gen_hf"],
            "pcc": 0.99,
            "args": {"shape": "(16, 61, 61)"},
        },
    ),
    (
        Reshape1280,
        [((1, 16, 64, 61), torch.float32)],
        {
            "model_names": ["pt_t5_google_flan_t5_large_text_gen_hf", "pt_t5_t5_large_text_gen_hf"],
            "pcc": 0.99,
            "args": {"shape": "(16, 64, 61)"},
        },
    ),
    (
        Reshape1281,
        [((16, 61, 64), torch.float32)],
        {
            "model_names": ["pt_t5_google_flan_t5_large_text_gen_hf", "pt_t5_t5_large_text_gen_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 16, 61, 64)"},
        },
    ),
    (
        Reshape1276,
        [((1, 61, 16, 64), torch.float32)],
        {
            "model_names": ["pt_t5_google_flan_t5_large_text_gen_hf", "pt_t5_t5_large_text_gen_hf"],
            "pcc": 0.99,
            "args": {"shape": "(61, 1024)"},
        },
    ),
    (
        Reshape1282,
        [((61, 2816), torch.float32)],
        {"model_names": ["pt_t5_google_flan_t5_large_text_gen_hf"], "pcc": 0.99, "args": {"shape": "(1, 61, 2816)"}},
    ),
    (
        Reshape1283,
        [((16, 1, 61), torch.float32)],
        {
            "model_names": ["pt_t5_google_flan_t5_large_text_gen_hf", "pt_t5_t5_large_text_gen_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 16, 1, 61)"},
        },
    ),
    (
        Reshape1284,
        [((1, 16, 1, 61), torch.float32)],
        {
            "model_names": ["pt_t5_google_flan_t5_large_text_gen_hf", "pt_t5_t5_large_text_gen_hf"],
            "pcc": 0.99,
            "args": {"shape": "(16, 1, 61)"},
        },
    ),
    (
        Reshape1285,
        [((1, 2816), torch.float32)],
        {"model_names": ["pt_t5_google_flan_t5_large_text_gen_hf"], "pcc": 0.99, "args": {"shape": "(1, 1, 2816)"}},
    ),
    (
        Reshape1286,
        [((512, 80, 3), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_base_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(512, 80, 3, 1)"},
        },
    ),
    (
        Reshape1287,
        [((1, 512, 3000, 1), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_base_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 512, 3000)"},
        },
    ),
    (
        Reshape1288,
        [((1, 512, 3000), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_base_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 512, 3000, 1)"},
        },
    ),
    (
        Reshape1289,
        [((512, 512, 3), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_base_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(512, 512, 3, 1)"},
        },
    ),
    (
        Reshape1290,
        [((1, 512, 1500, 1), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_base_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 512, 1500)"},
        },
    ),
    (
        Reshape1291,
        [((1, 1500, 512), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_base_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1500, 512)"},
        },
    ),
    (
        Reshape1292,
        [((1, 1500, 512), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_base_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 1500, 8, 64)"},
        },
    ),
    (
        Reshape1293,
        [((1500, 512), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_base_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 1500, 512)"},
        },
    ),
    (
        Reshape1292,
        [((1500, 512), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_base_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 1500, 8, 64)"},
        },
    ),
    (
        Reshape1294,
        [((1, 8, 1500, 64), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_base_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(8, 1500, 64)"},
        },
    ),
    (
        Reshape1295,
        [((8, 1500, 1500), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_base_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 8, 1500, 1500)"},
        },
    ),
    (
        Reshape1296,
        [((1, 8, 1500, 1500), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_base_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(8, 1500, 1500)"},
        },
    ),
    (
        Reshape1297,
        [((1, 8, 64, 1500), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_base_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(8, 64, 1500)"},
        },
    ),
    (
        Reshape1298,
        [((8, 1500, 64), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_base_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 8, 1500, 64)"},
        },
    ),
    (
        Reshape1291,
        [((1, 1500, 8, 64), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_base_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1500, 512)"},
        },
    ),
    (
        Reshape1299,
        [((8, 1, 1500), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_base_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 8, 1, 1500)"},
        },
    ),
    (
        Reshape1300,
        [((1, 8, 1, 1500), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_base_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(8, 1, 1500)"},
        },
    ),
    (
        Reshape1301,
        [((729, 12), torch.float32)],
        {
            "model_names": ["pt_beit_microsoft_beit_base_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 27, 27, 12)"},
        },
    ),
    (
        Reshape1302,
        [((1, 27, 27, 12), torch.float32)],
        {
            "model_names": ["pt_beit_microsoft_beit_base_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(729, 12)"},
        },
    ),
    (
        Reshape1303,
        [((38809, 12), torch.float32)],
        {
            "model_names": ["pt_beit_microsoft_beit_base_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(197, 197, 12)"},
        },
    ),
    (
        Reshape1304,
        [((2, 7), torch.int64)],
        {
            "model_names": ["pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text"],
            "pcc": 0.99,
            "args": {"shape": "(2, 7)"},
        },
    ),
    (
        Reshape1305,
        [((2, 7, 512), torch.float32)],
        {
            "model_names": ["pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text"],
            "pcc": 0.99,
            "args": {"shape": "(14, 512)"},
        },
    ),
    (
        Reshape1306,
        [((2, 7, 512), torch.float32)],
        {
            "model_names": ["pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text"],
            "pcc": 0.99,
            "args": {"shape": "(2, 7, 8, 64)"},
        },
    ),
    (
        Reshape1307,
        [((14, 512), torch.float32)],
        {
            "model_names": ["pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text"],
            "pcc": 0.99,
            "args": {"shape": "(2, 7, 512)"},
        },
    ),
    (
        Reshape1308,
        [((2, 8, 7, 64), torch.float32)],
        {
            "model_names": ["pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text"],
            "pcc": 0.99,
            "args": {"shape": "(16, 7, 64)"},
        },
    ),
    (
        Reshape1309,
        [((16, 7, 7), torch.float32)],
        {
            "model_names": ["pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text"],
            "pcc": 0.99,
            "args": {"shape": "(2, 8, 7, 7)"},
        },
    ),
    (
        Reshape1309,
        [((2, 8, 7, 7), torch.float32)],
        {
            "model_names": ["pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text"],
            "pcc": 0.99,
            "args": {"shape": "(2, 8, 7, 7)"},
        },
    ),
    (
        Reshape1310,
        [((2, 8, 7, 7), torch.float32)],
        {
            "model_names": ["pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text"],
            "pcc": 0.99,
            "args": {"shape": "(16, 7, 7)"},
        },
    ),
    (
        Reshape1311,
        [((16, 7, 64), torch.float32)],
        {
            "model_names": ["pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text"],
            "pcc": 0.99,
            "args": {"shape": "(2, 8, 7, 64)"},
        },
    ),
    (
        Reshape1305,
        [((2, 7, 8, 64), torch.float32)],
        {
            "model_names": ["pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text"],
            "pcc": 0.99,
            "args": {"shape": "(14, 512)"},
        },
    ),
    (
        Reshape1312,
        [((14, 2048), torch.float32)],
        {
            "model_names": ["pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text"],
            "pcc": 0.99,
            "args": {"shape": "(2, 7, 2048)"},
        },
    ),
    (
        Reshape1313,
        [((2, 7, 2048), torch.float32)],
        {
            "model_names": ["pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text"],
            "pcc": 0.99,
            "args": {"shape": "(14, 2048)"},
        },
    ),
    (
        Reshape1314,
        [((1, 384, 14, 14), torch.float32)],
        {
            "model_names": ["pt_deit_facebook_deit_small_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 384, 196, 1)"},
        },
    ),
    (
        Reshape1315,
        [((1, 197, 384), torch.float32)],
        {
            "model_names": ["pt_deit_facebook_deit_small_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(197, 384)"},
        },
    ),
    (
        Reshape1316,
        [((1, 197, 384), torch.float32)],
        {
            "model_names": ["pt_deit_facebook_deit_small_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 197, 6, 64)"},
        },
    ),
    (
        Reshape1317,
        [((197, 384), torch.float32)],
        {
            "model_names": ["pt_deit_facebook_deit_small_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 197, 384)"},
        },
    ),
    (
        Reshape1318,
        [((1, 6, 197, 64), torch.float32)],
        {
            "model_names": ["pt_deit_facebook_deit_small_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(6, 197, 64)"},
        },
    ),
    (
        Reshape1319,
        [((6, 197, 197), torch.float32)],
        {
            "model_names": ["pt_deit_facebook_deit_small_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 6, 197, 197)"},
        },
    ),
    (
        Reshape1320,
        [((1, 6, 197, 197), torch.float32)],
        {
            "model_names": ["pt_deit_facebook_deit_small_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(6, 197, 197)"},
        },
    ),
    (
        Reshape1321,
        [((1, 6, 64, 197), torch.float32)],
        {
            "model_names": ["pt_deit_facebook_deit_small_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(6, 64, 197)"},
        },
    ),
    (
        Reshape1322,
        [((6, 197, 64), torch.float32)],
        {
            "model_names": ["pt_deit_facebook_deit_small_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 6, 197, 64)"},
        },
    ),
    (
        Reshape1315,
        [((1, 197, 6, 64), torch.float32)],
        {
            "model_names": ["pt_deit_facebook_deit_small_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(197, 384)"},
        },
    ),
    (
        Reshape598,
        [((100, 8, 33, 280), torch.float32)],
        {
            "model_names": ["pt_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"shape": "(100, 264, 14, 20)"},
        },
    ),
    (
        Reshape601,
        [((100, 8, 16, 280), torch.float32)],
        {
            "model_names": ["pt_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"shape": "(100, 128, 14, 20)"},
        },
    ),
    (
        Reshape604,
        [((100, 8, 8, 1080), torch.float32)],
        {
            "model_names": ["pt_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"shape": "(100, 64, 27, 40)"},
        },
    ),
    (
        Reshape607,
        [((100, 8, 4, 4320), torch.float32)],
        {
            "model_names": ["pt_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"shape": "(100, 32, 54, 80)"},
        },
    ),
    (
        Reshape610,
        [((100, 8, 2, 17120), torch.float32)],
        {
            "model_names": ["pt_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"shape": "(100, 16, 107, 160)"},
        },
    ),
    (
        Reshape468,
        [((1, 256), torch.float32)],
        {"model_names": ["pt_gpt_gpt2_text_gen_hf"], "pcc": 0.99, "args": {"shape": "(1, 256)"}},
    ),
    (
        Reshape1323,
        [((256, 3072), torch.float32)],
        {"model_names": ["pt_gpt_gpt2_text_gen_hf"], "pcc": 0.99, "args": {"shape": "(1, 256, 3072)"}},
    ),
    (
        Reshape1324,
        [((1, 256, 3072), torch.float32)],
        {"model_names": ["pt_gpt_gpt2_text_gen_hf"], "pcc": 0.99, "args": {"shape": "(256, 3072)"}},
    ),
    (
        Reshape1325,
        [((1, 576, 1, 1), torch.float32)],
        {
            "model_names": ["pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"],
            "pcc": 0.99,
            "args": {"shape": "(1, 576, 1, 1)"},
        },
    ),
    (
        Reshape1326,
        [((1, 512, 512), torch.float32)],
        {
            "model_names": ["pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 512, 1, 512)"},
        },
    ),
    (
        Reshape1327,
        [((1, 224, 224, 256), torch.float32)],
        {
            "model_names": ["pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 50176, 256)"},
        },
    ),
    (
        Reshape1328,
        [((1, 50176, 512), torch.float32)],
        {
            "model_names": ["pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(50176, 512)"},
        },
    ),
    (
        Reshape1329,
        [((1, 50176, 512), torch.float32)],
        {
            "model_names": ["pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 50176, 1, 512)"},
        },
    ),
    (
        Reshape1330,
        [((50176, 512), torch.float32)],
        {
            "model_names": ["pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 50176, 512)"},
        },
    ),
    (
        Reshape1331,
        [((1, 1088, 1, 1), torch.float32)],
        {
            "model_names": ["pt_regnet_facebook_regnet_y_040_img_cls_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 1088, 1, 1)"},
        },
    ),
    (
        Reshape1332,
        [((1, 400, 1, 1), torch.float32)],
        {
            "model_names": ["pt_regnet_regnet_x_400mf_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"shape": "(1, 400, 1, 1)"},
        },
    ),
    (
        Reshape1333,
        [((1, 16, 38, 38), torch.float32)],
        {"model_names": ["pt_ssd300_resnet50_base_img_cls_torchhub"], "pcc": 0.99, "args": {"shape": "(1, 4, 5776)"}},
    ),
    (
        Reshape1334,
        [((1, 24, 19, 19), torch.float32)],
        {"model_names": ["pt_ssd300_resnet50_base_img_cls_torchhub"], "pcc": 0.99, "args": {"shape": "(1, 4, 2166)"}},
    ),
    (
        Reshape1335,
        [((1, 24, 10, 10), torch.float32)],
        {"model_names": ["pt_ssd300_resnet50_base_img_cls_torchhub"], "pcc": 0.99, "args": {"shape": "(1, 4, 600)"}},
    ),
    (
        Reshape1336,
        [((1, 24, 5, 5), torch.float32)],
        {"model_names": ["pt_ssd300_resnet50_base_img_cls_torchhub"], "pcc": 0.99, "args": {"shape": "(1, 4, 150)"}},
    ),
    (
        Reshape1337,
        [((1, 16, 3, 3), torch.float32)],
        {"model_names": ["pt_ssd300_resnet50_base_img_cls_torchhub"], "pcc": 0.99, "args": {"shape": "(1, 4, 36)"}},
    ),
    (
        Reshape1338,
        [((1, 324, 38, 38), torch.float32)],
        {"model_names": ["pt_ssd300_resnet50_base_img_cls_torchhub"], "pcc": 0.99, "args": {"shape": "(1, 81, 5776)"}},
    ),
    (
        Reshape1339,
        [((1, 486, 19, 19), torch.float32)],
        {"model_names": ["pt_ssd300_resnet50_base_img_cls_torchhub"], "pcc": 0.99, "args": {"shape": "(1, 81, 2166)"}},
    ),
    (
        Reshape1340,
        [((1, 486, 10, 10), torch.float32)],
        {"model_names": ["pt_ssd300_resnet50_base_img_cls_torchhub"], "pcc": 0.99, "args": {"shape": "(1, 81, 600)"}},
    ),
    (
        Reshape1341,
        [((1, 486, 5, 5), torch.float32)],
        {"model_names": ["pt_ssd300_resnet50_base_img_cls_torchhub"], "pcc": 0.99, "args": {"shape": "(1, 81, 150)"}},
    ),
    (
        Reshape1342,
        [((1, 324, 3, 3), torch.float32)],
        {"model_names": ["pt_ssd300_resnet50_base_img_cls_torchhub"], "pcc": 0.99, "args": {"shape": "(1, 81, 36)"}},
    ),
    (
        Reshape1343,
        [((1, 324, 1, 1), torch.float32)],
        {"model_names": ["pt_ssd300_resnet50_base_img_cls_torchhub"], "pcc": 0.99, "args": {"shape": "(1, 81, 4)"}},
    ),
    (
        Reshape1344,
        [((50, 1, 768), torch.float32)],
        {"model_names": ["pt_vit_vit_b_32_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(50, 768)"}},
    ),
    (
        Reshape1345,
        [((50, 2304), torch.float32)],
        {"model_names": ["pt_vit_vit_b_32_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(50, 1, 2304)"}},
    ),
    (
        Reshape1346,
        [((50, 1, 2304), torch.float32)],
        {"model_names": ["pt_vit_vit_b_32_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(50, 1, 3, 768)"}},
    ),
    (
        Reshape1347,
        [((1, 50, 1, 768), torch.float32)],
        {"model_names": ["pt_vit_vit_b_32_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(50, 12, 64)"}},
    ),
    (
        Reshape1348,
        [((12, 50, 64), torch.float32)],
        {"model_names": ["pt_vit_vit_b_32_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(12, 50, 64)"}},
    ),
    (
        Reshape1349,
        [((12, 50, 64), torch.float32)],
        {"model_names": ["pt_vit_vit_b_32_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(1, 12, 50, 64)"}},
    ),
    (
        Reshape1350,
        [((12, 50, 50), torch.float32)],
        {"model_names": ["pt_vit_vit_b_32_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(1, 12, 50, 50)"}},
    ),
    (
        Reshape1351,
        [((1, 12, 50, 50), torch.float32)],
        {"model_names": ["pt_vit_vit_b_32_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(12, 50, 50)"}},
    ),
    (
        Reshape1344,
        [((50, 1, 12, 64), torch.float32)],
        {"model_names": ["pt_vit_vit_b_32_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(50, 768)"}},
    ),
    (
        Reshape1352,
        [((50, 768), torch.float32)],
        {"model_names": ["pt_vit_vit_b_32_img_cls_torchvision"], "pcc": 0.99, "args": {"shape": "(50, 1, 768)"}},
    ),
    (
        Reshape1353,
        [((384, 80, 3), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(384, 80, 3, 1)"},
        },
    ),
    (
        Reshape1354,
        [((1, 384, 3000, 1), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 384, 3000)"},
        },
    ),
    (
        Reshape1355,
        [((1, 384, 3000), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 384, 3000, 1)"},
        },
    ),
    (
        Reshape1356,
        [((384, 384, 3), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(384, 384, 3, 1)"},
        },
    ),
    (
        Reshape1357,
        [((1, 384, 1500, 1), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 384, 1500)"},
        },
    ),
    (
        Reshape1358,
        [((1, 1500, 384), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1500, 384)"},
        },
    ),
    (
        Reshape1359,
        [((1, 1500, 384), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 1500, 6, 64)"},
        },
    ),
    (
        Reshape1360,
        [((1500, 384), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 1500, 384)"},
        },
    ),
    (
        Reshape1359,
        [((1500, 384), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 1500, 6, 64)"},
        },
    ),
    (
        Reshape1361,
        [((1, 6, 1500, 64), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(6, 1500, 64)"},
        },
    ),
    (
        Reshape1362,
        [((6, 1500, 1500), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 6, 1500, 1500)"},
        },
    ),
    (
        Reshape1363,
        [((1, 6, 1500, 1500), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(6, 1500, 1500)"},
        },
    ),
    (
        Reshape1364,
        [((1, 6, 64, 1500), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(6, 64, 1500)"},
        },
    ),
    (
        Reshape1365,
        [((6, 1500, 64), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 6, 1500, 64)"},
        },
    ),
    (
        Reshape1358,
        [((1, 1500, 6, 64), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1500, 384)"},
        },
    ),
    (
        Reshape1366,
        [((6, 1, 1500), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 6, 1, 1500)"},
        },
    ),
    (
        Reshape1367,
        [((1, 6, 1, 1500), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(6, 1, 1500)"},
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

    compiled_model = compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model, VerifyConfig(value_checker=AutomaticValueChecker(pcc=pcc)))
