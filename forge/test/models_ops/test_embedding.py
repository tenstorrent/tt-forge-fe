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


class Embedding0(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding0.weight_1",
            forge.Parameter(*(30000, 128), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding0.weight_1"))
        return embedding_output_1


class Embedding1(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, embedding_input_0, embedding_input_1):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, embedding_input_1)
        return embedding_output_1


class Embedding2(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding2.weight_1",
            forge.Parameter(*(50257, 768), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding2.weight_1"))
        return embedding_output_1


class Embedding3(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding3.weight_1",
            forge.Parameter(*(2, 128), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding3.weight_1"))
        return embedding_output_1


class Embedding4(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding4.weight_1",
            forge.Parameter(*(512, 128), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding4.weight_1"))
        return embedding_output_1


class Embedding5(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding5.weight_1",
            forge.Parameter(*(28996, 1024), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding5.weight_1"))
        return embedding_output_1


class Embedding6(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding6.weight_1",
            forge.Parameter(*(2, 1024), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding6.weight_1"))
        return embedding_output_1


class Embedding7(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding7.weight_1",
            forge.Parameter(*(512, 1024), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding7.weight_1"))
        return embedding_output_1


class Embedding8(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding8.weight_1",
            forge.Parameter(*(51200, 1024), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding8.weight_1"))
        return embedding_output_1


class Embedding9(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding9.weight_1",
            forge.Parameter(*(50272, 768), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding9.weight_1"))
        return embedding_output_1


class Embedding10(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding10.weight_1",
            forge.Parameter(*(2050, 768), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding10.weight_1"))
        return embedding_output_1


class Embedding11(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding11.weight_1",
            forge.Parameter(*(50272, 512), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding11.weight_1"))
        return embedding_output_1


class Embedding12(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding12.weight_1",
            forge.Parameter(*(2050, 1024), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding12.weight_1"))
        return embedding_output_1


class Embedding13(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding13.weight_1",
            forge.Parameter(*(262, 768), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding13.weight_1"))
        return embedding_output_1


class Embedding14(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding14.weight_1",
            forge.Parameter(*(51200, 2048), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding14.weight_1"))
        return embedding_output_1


class Embedding15(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding15.weight_1",
            forge.Parameter(*(151936, 896), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding15.weight_1"))
        return embedding_output_1


class Embedding16(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding16.weight_1",
            forge.Parameter(*(2049, 1536), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding16.weight_1"))
        return embedding_output_1


class Embedding17(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding17.weight_1",
            forge.Parameter(*(32128, 768), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding17.weight_1"))
        return embedding_output_1


class Embedding18(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding18.weight_1",
            forge.Parameter(*(30522, 384), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding18.weight_1"))
        return embedding_output_1


class Embedding19(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding19.weight_1",
            forge.Parameter(*(51865, 384), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding19.weight_1"))
        return embedding_output_1


class Embedding20(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding20.weight_1",
            forge.Parameter(*(32000, 768), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding20.weight_1"))
        return embedding_output_1


class Embedding21(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding21.weight_1",
            forge.Parameter(*(512, 768), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding21.weight_1"))
        return embedding_output_1


class Embedding22(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding22.weight_1",
            forge.Parameter(*(2, 768), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding22.weight_1"))
        return embedding_output_1


class Embedding23(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding23.weight_1",
            forge.Parameter(*(30522, 768), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding23.weight_1"))
        return embedding_output_1


class Embedding24(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding24.weight_1",
            forge.Parameter(*(32256, 2048), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding24.weight_1"))
        return embedding_output_1


class Embedding25(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding25.weight_1",
            forge.Parameter(*(50265, 768), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding25.weight_1"))
        return embedding_output_1


class Embedding26(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding26.weight_1",
            forge.Parameter(*(1, 768), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding26.weight_1"))
        return embedding_output_1


class Embedding27(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding27.weight_1",
            forge.Parameter(*(514, 768), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding27.weight_1"))
        return embedding_output_1


class Embedding28(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding28.weight_1",
            forge.Parameter(*(32128, 512), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding28.weight_1"))
        return embedding_output_1


class Embedding29(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding29.weight_1",
            forge.Parameter(*(51865, 512), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding29.weight_1"))
        return embedding_output_1


class Embedding30(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding30.weight_1",
            forge.Parameter(*(119547, 768), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding30.weight_1"))
        return embedding_output_1


class Embedding31(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding31.weight_1",
            forge.Parameter(*(28996, 768), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding31.weight_1"))
        return embedding_output_1


class Embedding32(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding32.weight_1",
            forge.Parameter(*(151936, 1536), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding32.weight_1"))
        return embedding_output_1


class Embedding33(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding33.weight_1",
            forge.Parameter(*(30528, 768), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding33.weight_1"))
        return embedding_output_1


class Embedding34(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("embedding34_const_1", shape=(21128, 128), dtype=torch.float32)

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_constant("embedding34_const_1"))
        return embedding_output_1


class Embedding35(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding35.weight_1",
            forge.Parameter(*(131072, 2048), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding35.weight_1"))
        return embedding_output_1


class Embedding36(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding36.weight_1",
            forge.Parameter(*(50257, 2048), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding36.weight_1"))
        return embedding_output_1


class Embedding37(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding37.weight_1",
            forge.Parameter(*(30524, 768), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding37.weight_1"))
        return embedding_output_1


class Embedding38(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding38.weight_1",
            forge.Parameter(*(21128, 768), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding38.weight_1"))
        return embedding_output_1


class Embedding39(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("embedding39_const_1", shape=(18000, 768), dtype=torch.float32)

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_constant("embedding39_const_1"))
        return embedding_output_1


class Embedding40(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding40.weight_1",
            forge.Parameter(*(513, 768), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding40.weight_1"))
        return embedding_output_1


class Embedding41(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("embedding41_const_1", shape=(21128, 768), dtype=torch.float32)

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_constant("embedding41_const_1"))
        return embedding_output_1


class Embedding42(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding42.weight_1",
            forge.Parameter(*(151936, 1024), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding42.weight_1"))
        return embedding_output_1


class Embedding43(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding43.weight_1",
            forge.Parameter(*(151665, 2560), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding43.weight_1"))
        return embedding_output_1


class Embedding44(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding44.weight_1",
            forge.Parameter(*(32128, 1024), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding44.weight_1"))
        return embedding_output_1


class Embedding45(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding45.weight_1",
            forge.Parameter(*(50265, 1024), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding45.weight_1"))
        return embedding_output_1


class Embedding46(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding46.weight_1",
            forge.Parameter(*(30522, 1024), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding46.weight_1"))
        return embedding_output_1


class Embedding47(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding47.weight_1",
            forge.Parameter(*(250880, 1536), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding47.weight_1"))
        return embedding_output_1


class Embedding48(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding48.weight_1",
            forge.Parameter(*(50272, 2048), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding48.weight_1"))
        return embedding_output_1


class Embedding49(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding49.weight_1",
            forge.Parameter(*(2050, 2048), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding49.weight_1"))
        return embedding_output_1


class Embedding50(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding50.weight_1",
            forge.Parameter(*(81, 768), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding50.weight_1"))
        return embedding_output_1


class Embedding51(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding51.weight_1",
            forge.Parameter(*(256008, 2048), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding51.weight_1"))
        return embedding_output_1


class Embedding52(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding52.weight_1",
            forge.Parameter(*(128256, 2048), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding52.weight_1"))
        return embedding_output_1


class Embedding53(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding53.weight_1",
            forge.Parameter(*(51200, 2560), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding53.weight_1"))
        return embedding_output_1


class Embedding54(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding54.weight_1",
            forge.Parameter(*(250002, 768), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding54.weight_1"))
        return embedding_output_1


class Embedding55(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding55.weight_1",
            forge.Parameter(*(49408, 512), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding55.weight_1"))
        return embedding_output_1


class Embedding56(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding56.weight_1",
            forge.Parameter(*(77, 512), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding56.weight_1"))
        return embedding_output_1


class Embedding57(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding57.weight_1",
            forge.Parameter(*(151936, 2048), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding57.weight_1"))
        return embedding_output_1


class Embedding58(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding58.weight_1",
            forge.Parameter(*(51865, 768), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding58.weight_1"))
        return embedding_output_1


class Embedding59(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding59.weight_1",
            forge.Parameter(*(2049, 1024), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding59.weight_1"))
        return embedding_output_1


class Embedding60(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding60.weight_1",
            forge.Parameter(*(32064, 4096), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding60.weight_1"))
        return embedding_output_1


class Embedding61(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding61.weight_1",
            forge.Parameter(*(32000, 4096), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding61.weight_1"))
        return embedding_output_1


class Embedding62(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding62.weight_1",
            forge.Parameter(*(131072, 4096), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding62.weight_1"))
        return embedding_output_1


class Embedding63(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding63.weight_1",
            forge.Parameter(*(32768, 4096), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding63.weight_1"))
        return embedding_output_1


class Embedding64(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding64.weight_1",
            forge.Parameter(*(32064, 3072), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding64.weight_1"))
        return embedding_output_1


class Embedding65(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding65.weight_1",
            forge.Parameter(*(2049, 2048), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding65.weight_1"))
        return embedding_output_1


class Embedding66(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding66.weight_1",
            forge.Parameter(*(51866, 1280), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding66.weight_1"))
        return embedding_output_1


class Embedding67(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding67.weight_1",
            forge.Parameter(*(256008, 1024), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding67.weight_1"))
        return embedding_output_1


class Embedding68(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding68.weight_1",
            forge.Parameter(*(102400, 4096), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding68.weight_1"))
        return embedding_output_1


class Embedding69(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding69.weight_1",
            forge.Parameter(*(131072, 3072), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding69.weight_1"))
        return embedding_output_1


class Embedding70(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding70.weight_1",
            forge.Parameter(*(65024, 4544), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding70.weight_1"))
        return embedding_output_1


class Embedding71(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding71.weight_1",
            forge.Parameter(*(256000, 2048), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding71.weight_1"))
        return embedding_output_1


class Embedding72(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding72.weight_1",
            forge.Parameter(*(256000, 3072), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding72.weight_1"))
        return embedding_output_1


class Embedding73(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding73.weight_1",
            forge.Parameter(*(50257, 2560), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding73.weight_1"))
        return embedding_output_1


class Embedding74(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding74.weight_1",
            forge.Parameter(*(128256, 4096), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding74.weight_1"))
        return embedding_output_1


class Embedding75(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding75.weight_1",
            forge.Parameter(*(128256, 3072), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding75.weight_1"))
        return embedding_output_1


class Embedding76(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding76.weight_1",
            forge.Parameter(*(100352, 5120), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding76.weight_1"))
        return embedding_output_1


class Embedding77(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding77.weight_1",
            forge.Parameter(*(152064, 3584), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding77.weight_1"))
        return embedding_output_1


class Embedding78(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding78.weight_1",
            forge.Parameter(*(151936, 2560), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding78.weight_1"))
        return embedding_output_1


class Embedding79(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding79.weight_1",
            forge.Parameter(*(151669, 1024), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding79.weight_1"))
        return embedding_output_1


class Embedding80(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding80.weight_1",
            forge.Parameter(*(51865, 1280), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding80.weight_1"))
        return embedding_output_1


class Embedding81(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding81.weight_1",
            forge.Parameter(*(51865, 1024), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding81.weight_1"))
        return embedding_output_1


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (
        Embedding0,
        [((1, 128), torch.int64)],
        {
            "model_names": [
                "onnx_albert_xxlarge_v1_mlm_hf",
                "pt_albert_base_v1_mlm_hf",
                "pt_albert_base_v1_token_cls_hf",
                "onnx_albert_xlarge_v2_mlm_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
                "pt_albert_xxlarge_v2_token_cls_hf",
                "onnx_albert_large_v1_mlm_hf",
                "pt_albert_xlarge_v1_mlm_hf",
                "pt_albert_xlarge_v1_token_cls_hf",
                "onnx_albert_large_v2_mlm_hf",
                "pt_albert_xlarge_v2_mlm_hf",
                "pt_albert_xlarge_v2_token_cls_hf",
                "onnx_albert_base_v2_mlm_hf",
                "pt_albert_large_v2_mlm_hf",
                "pt_albert_large_v2_token_cls_hf",
                "pt_albert_base_v2_mlm_hf",
                "pt_albert_base_v2_token_cls_hf",
                "onnx_albert_base_v1_mlm_hf",
                "pt_albert_large_v1_mlm_hf",
                "pt_albert_large_v1_token_cls_hf",
                "onnx_albert_xxlarge_v2_mlm_hf",
                "onnx_albert_xlarge_v1_mlm_hf",
                "pt_albert_xxlarge_v1_mlm_hf",
                "pt_albert_xxlarge_v1_token_cls_hf",
            ],
            "pcc": 0.99,
            "max_int": 29999,
        },
    ),
    (
        Embedding1,
        [((1, 128), torch.int64), ((2, 128), torch.float32)],
        {
            "model_names": [
                "onnx_albert_xxlarge_v1_mlm_hf",
                "onnx_albert_xlarge_v2_mlm_hf",
                "onnx_albert_large_v1_mlm_hf",
                "onnx_albert_large_v2_mlm_hf",
                "onnx_albert_base_v2_mlm_hf",
                "onnx_albert_base_v1_mlm_hf",
                "onnx_albert_xxlarge_v2_mlm_hf",
                "onnx_albert_xlarge_v1_mlm_hf",
            ],
            "pcc": 0.99,
            "max_int": 1,
        },
    ),
    (
        Embedding1,
        [((1, 128), torch.int64), ((512, 128), torch.float32)],
        {
            "model_names": [
                "onnx_albert_xxlarge_v1_mlm_hf",
                "onnx_albert_xlarge_v2_mlm_hf",
                "onnx_albert_large_v1_mlm_hf",
                "onnx_albert_large_v2_mlm_hf",
                "onnx_albert_base_v2_mlm_hf",
                "onnx_albert_base_v1_mlm_hf",
                "onnx_albert_xxlarge_v2_mlm_hf",
                "onnx_albert_xlarge_v1_mlm_hf",
            ],
            "pcc": 0.99,
            "max_int": 511,
        },
    ),
    (
        Embedding2,
        [((1, 7), torch.int64)],
        {
            "model_names": [
                "onnx_gpt_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
                "pt_gpt_gpt2_sequence_classification_seq_cls_hf",
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
                "onnx_nanogpt_financialsupport_nanogpt_text_gen_hf",
            ],
            "pcc": 0.99,
            "max_int": 50256,
        },
    ),
    (
        Embedding1,
        [((1, 7), torch.int64), ((1024, 768), torch.float32)],
        {
            "model_names": [
                "onnx_gpt_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
                "pt_gpt_gpt2_sequence_classification_seq_cls_hf",
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
                "onnx_nanogpt_financialsupport_nanogpt_text_gen_hf",
            ],
            "pcc": 0.99,
            "max_int": 1023,
        },
    ),
    (Embedding0, [((1, 14), torch.int64)], {"model_names": ["pt_albert_squad2_qa_hf"], "pcc": 0.99, "max_int": 29999}),
    (Embedding3, [((1, 14), torch.int64)], {"model_names": ["pt_albert_squad2_qa_hf"], "pcc": 0.99, "max_int": 1}),
    (Embedding4, [((1, 14), torch.int64)], {"model_names": ["pt_albert_squad2_qa_hf"], "pcc": 0.99, "max_int": 511}),
    (
        Embedding5,
        [((1, 128), torch.int64)],
        {
            "model_names": ["pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf"],
            "pcc": 0.99,
            "max_int": 28995,
        },
    ),
    (
        Embedding6,
        [((1, 128), torch.int64)],
        {
            "model_names": ["pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf"],
            "pcc": 0.99,
            "max_int": 1,
        },
    ),
    (
        Embedding7,
        [((1, 128), torch.int64)],
        {
            "model_names": ["pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf"],
            "pcc": 0.99,
            "max_int": 511,
        },
    ),
    (
        Embedding8,
        [((1, 256), torch.int64)],
        {
            "model_names": [
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
            ],
            "pcc": 0.99,
            "max_int": 51199,
        },
    ),
    (
        Embedding9,
        [((1, 32), torch.int64)],
        {
            "model_names": ["pt_opt_facebook_opt_125m_seq_cls_hf", "pt_opt_facebook_opt_125m_qa_hf"],
            "pcc": 0.99,
            "max_int": 50271,
        },
    ),
    (
        Embedding10,
        [((1, 32), torch.int64)],
        {
            "model_names": ["pt_opt_facebook_opt_125m_seq_cls_hf", "pt_opt_facebook_opt_125m_qa_hf"],
            "pcc": 0.99,
            "max_int": 2049,
        },
    ),
    (
        Embedding11,
        [((1, 32), torch.int64)],
        {
            "model_names": ["pt_opt_facebook_opt_350m_qa_hf", "pt_opt_facebook_opt_350m_seq_cls_hf"],
            "pcc": 0.99,
            "max_int": 50271,
        },
    ),
    (
        Embedding12,
        [((1, 32), torch.int64)],
        {
            "model_names": ["pt_opt_facebook_opt_350m_qa_hf", "pt_opt_facebook_opt_350m_seq_cls_hf"],
            "pcc": 0.99,
            "max_int": 2049,
        },
    ),
    (
        Embedding13,
        [((1, 2048), torch.int64)],
        {
            "model_names": [
                "pt_perceiverio_deepmind_language_perceiver_mlm_hf",
                "onnx_perceiverio_deepmind_language_perceiver_mlm_hf",
            ],
            "pcc": 0.99,
            "max_int": 261,
        },
    ),
    (
        Embedding1,
        [((2048,), torch.int64), ((2048, 768), torch.float32)],
        {"model_names": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"], "pcc": 0.99, "max_int": 2047},
    ),
    (
        Embedding14,
        [((1, 256), torch.int64)],
        {
            "model_names": ["pt_phi_1_5_microsoft_phi_1_5_clm_hf", "pt_phi1_microsoft_phi_1_clm_hf"],
            "pcc": 0.99,
            "max_int": 51199,
        },
    ),
    (
        Embedding15,
        [((1, 35), torch.int64)],
        {"model_names": ["pt_qwen_coder_0_5b_clm_hf"], "pcc": 0.99, "max_int": 151935},
    ),
    (
        Embedding16,
        [((1, 1), torch.int64)],
        {"model_names": ["pt_stereo_medium_music_generation_hf"], "pcc": 0.99, "max_int": 2048},
    ),
    (
        Embedding17,
        [((1, 25), torch.int64)],
        {
            "model_names": [
                "pt_stereo_medium_music_generation_hf",
                "pt_stereo_small_music_generation_hf",
                "pt_stereo_large_music_generation_hf",
            ],
            "pcc": 0.99,
            "max_int": 32127,
        },
    ),
    (
        Embedding1,
        [((25, 25), torch.int32), ((32, 12), torch.float32)],
        {
            "model_names": [
                "pt_stereo_medium_music_generation_hf",
                "pt_stereo_small_music_generation_hf",
                "pt_stereo_large_music_generation_hf",
            ],
            "pcc": 0.99,
            "max_int": 31,
        },
    ),
    (
        Embedding18,
        [((1, 13), torch.int64)],
        {
            "model_names": ["onnx_minilm_sentence_transformers_all_minilm_l6_v2_seq_cls_hf"],
            "pcc": 0.99,
            "max_int": 30521,
        },
    ),
    (
        Embedding1,
        [((1, 13), torch.int64), ((2, 384), torch.float32)],
        {"model_names": ["onnx_minilm_sentence_transformers_all_minilm_l6_v2_seq_cls_hf"], "pcc": 0.99, "max_int": 1},
    ),
    (
        Embedding1,
        [((1, 13), torch.int64), ((512, 384), torch.float32)],
        {"model_names": ["onnx_minilm_sentence_transformers_all_minilm_l6_v2_seq_cls_hf"], "pcc": 0.99, "max_int": 511},
    ),
    (
        Embedding19,
        [((1, 1), torch.int64)],
        {"model_names": ["onnx_whisper_openai_whisper_tiny_speech_recognition_hf"], "pcc": 0.99, "max_int": 51864},
    ),
    (
        Embedding1,
        [((1, 1), torch.int64), ((448, 384), torch.float32)],
        {"model_names": ["onnx_whisper_openai_whisper_tiny_speech_recognition_hf"], "pcc": 0.99, "max_int": 447},
    ),
    (
        Embedding20,
        [((1, 14), torch.int64)],
        {"model_names": ["pd_bert_bert_base_japanese_qa_padlenlp"], "pcc": 0.99, "max_int": 31999},
    ),
    (
        Embedding21,
        [((1, 14), torch.int64)],
        {"model_names": ["pd_bert_bert_base_japanese_qa_padlenlp"], "pcc": 0.99, "max_int": 511},
    ),
    (
        Embedding22,
        [((1, 14), torch.int64)],
        {"model_names": ["pd_bert_bert_base_japanese_qa_padlenlp"], "pcc": 0.99, "max_int": 1},
    ),
    (
        Embedding23,
        [((1, 9), torch.int64)],
        {
            "model_names": ["pd_bert_bert_base_uncased_qa_padlenlp", "pd_bert_bert_base_uncased_mlm_padlenlp"],
            "pcc": 0.99,
            "max_int": 30521,
        },
    ),
    (
        Embedding21,
        [((1, 9), torch.int64)],
        {
            "model_names": [
                "pd_bert_bert_base_uncased_qa_padlenlp",
                "pd_bert_chinese_roberta_base_mlm_padlenlp",
                "pd_bert_bert_base_uncased_mlm_padlenlp",
                "pd_roberta_rbt4_ch_seq_cls_padlenlp",
            ],
            "pcc": 0.99,
            "max_int": 511,
        },
    ),
    (
        Embedding22,
        [((1, 9), torch.int64)],
        {
            "model_names": [
                "pd_bert_bert_base_uncased_qa_padlenlp",
                "pd_ernie_1_0_mlm_padlenlp",
                "pd_ernie_1_0_qa_padlenlp",
                "pd_bert_chinese_roberta_base_mlm_padlenlp",
                "pd_bert_bert_base_uncased_mlm_padlenlp",
                "pd_ernie_1_0_seq_cls_padlenlp",
            ],
            "pcc": 0.99,
            "max_int": 1,
        },
    ),
    (
        Embedding3,
        [((1, 128), torch.int64)],
        {
            "model_names": [
                "pt_albert_base_v1_mlm_hf",
                "pt_albert_base_v1_token_cls_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
                "pt_albert_xxlarge_v2_token_cls_hf",
                "pt_albert_xlarge_v1_mlm_hf",
                "pt_albert_xlarge_v1_token_cls_hf",
                "pt_albert_xlarge_v2_mlm_hf",
                "pt_albert_xlarge_v2_token_cls_hf",
                "pt_albert_large_v2_mlm_hf",
                "pt_albert_large_v2_token_cls_hf",
                "pt_albert_base_v2_mlm_hf",
                "pt_albert_base_v2_token_cls_hf",
                "pt_albert_large_v1_mlm_hf",
                "pt_albert_large_v1_token_cls_hf",
                "pt_albert_xxlarge_v1_mlm_hf",
                "pt_albert_xxlarge_v1_token_cls_hf",
            ],
            "pcc": 0.99,
            "max_int": 1,
        },
    ),
    (
        Embedding4,
        [((1, 128), torch.int64)],
        {
            "model_names": [
                "pt_albert_base_v1_mlm_hf",
                "pt_albert_base_v1_token_cls_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
                "pt_albert_xxlarge_v2_token_cls_hf",
                "pt_albert_xlarge_v1_mlm_hf",
                "pt_albert_xlarge_v1_token_cls_hf",
                "pt_albert_xlarge_v2_mlm_hf",
                "pt_albert_xlarge_v2_token_cls_hf",
                "pt_albert_large_v2_mlm_hf",
                "pt_albert_large_v2_token_cls_hf",
                "pt_albert_base_v2_mlm_hf",
                "pt_albert_base_v2_token_cls_hf",
                "pt_albert_large_v1_mlm_hf",
                "pt_albert_large_v1_token_cls_hf",
                "pt_albert_xxlarge_v1_mlm_hf",
                "pt_albert_xxlarge_v1_token_cls_hf",
            ],
            "pcc": 0.99,
            "max_int": 511,
        },
    ),
    (
        Embedding24,
        [((1, 588), torch.int64)],
        {"model_names": ["pt_deepseek_1_3b_instruct_qa_hf"], "pcc": 0.99, "max_int": 32255},
    ),
    (
        Embedding14,
        [((1, 12), torch.int64)],
        {
            "model_names": ["pt_phi_1_5_microsoft_phi_1_5_token_cls_hf", "pt_phi1_microsoft_phi_1_token_cls_hf"],
            "pcc": 0.99,
            "max_int": 51199,
        },
    ),
    (
        Embedding25,
        [((1, 128), torch.int64)],
        {
            "model_names": [
                "pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf",
                "onnx_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf",
            ],
            "pcc": 0.99,
            "max_int": 50264,
        },
    ),
    (
        Embedding26,
        [((1, 128), torch.int64)],
        {"model_names": ["pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf"], "pcc": 0.99, "max_int": 0},
    ),
    (
        Embedding27,
        [((1, 128), torch.int64)],
        {
            "model_names": [
                "pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf",
                "onnx_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf",
            ],
            "pcc": 0.99,
            "max_int": 513,
        },
    ),
    (
        Embedding23,
        [((1, 128), torch.int64)],
        {
            "model_names": [
                "onnx_bert_bert_base_uncased_mlm_hf",
                "pt_bert_bert_base_uncased_mlm_hf",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder",
                "pt_distilbert_distilbert_base_uncased_mlm_hf",
                "pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder",
                "pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder",
            ],
            "pcc": 0.99,
            "max_int": 30521,
        },
    ),
    (
        Embedding1,
        [((1, 128), torch.int64), ((2, 768), torch.float32)],
        {
            "model_names": [
                "onnx_bert_bert_base_uncased_mlm_hf",
                "onnx_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf",
                "pt_squeezebert_squeezebert_mnli_seq_cls_hf",
            ],
            "pcc": 0.99,
            "max_int": 1,
        },
    ),
    (
        Embedding1,
        [((1, 128), torch.int64), ((512, 768), torch.float32)],
        {
            "model_names": [
                "onnx_bert_bert_base_uncased_mlm_hf",
                "onnx_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf",
                "onnx_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
            ],
            "pcc": 0.99,
            "max_int": 511,
        },
    ),
    (
        Embedding28,
        [((1, 1), torch.int64)],
        {"model_names": ["onnx_t5_t5_small_text_gen_hf"], "pcc": 0.99, "max_int": 32127},
    ),
    (
        Embedding1,
        [((1, 1), torch.int64), ((32, 8), torch.float32)],
        {"model_names": ["onnx_t5_t5_small_text_gen_hf"], "pcc": 0.99, "max_int": 31},
    ),
    (
        Embedding28,
        [((1, 61), torch.int64)],
        {
            "model_names": [
                "onnx_t5_t5_small_text_gen_hf",
                "pt_t5_t5_small_text_gen_hf",
                "pt_t5_google_flan_t5_small_text_gen_hf",
            ],
            "pcc": 0.99,
            "max_int": 32127,
        },
    ),
    (
        Embedding1,
        [((61, 61), torch.int64), ((32, 8), torch.float32)],
        {"model_names": ["onnx_t5_t5_small_text_gen_hf"], "pcc": 0.99, "max_int": 31},
    ),
    (
        Embedding29,
        [((1, 1), torch.int64)],
        {"model_names": ["onnx_whisper_openai_whisper_base_speech_recognition_hf"], "pcc": 0.99, "max_int": 51864},
    ),
    (
        Embedding1,
        [((1, 1), torch.int64), ((448, 512), torch.float32)],
        {"model_names": ["onnx_whisper_openai_whisper_base_speech_recognition_hf"], "pcc": 0.99, "max_int": 447},
    ),
    (
        Embedding8,
        [((1, 5), torch.int64)],
        {
            "model_names": [
                "pt_codegen_salesforce_codegen_350m_mono_clm_hf",
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
            ],
            "pcc": 0.99,
            "max_int": 51199,
        },
    ),
    (
        Embedding30,
        [((1, 128), torch.int64)],
        {
            "model_names": [
                "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "onnx_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "pt_distilbert_distilbert_base_multilingual_cased_mlm_hf",
            ],
            "pcc": 0.99,
            "max_int": 119546,
        },
    ),
    (
        Embedding21,
        [((1, 128), torch.int64)],
        {
            "model_names": [
                "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "pt_bert_bert_base_uncased_mlm_hf",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_squeezebert_squeezebert_mnli_seq_cls_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
                "pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder",
                "pt_distilbert_distilbert_base_uncased_mlm_hf",
                "pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder",
                "pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf",
                "pt_distilbert_distilbert_base_multilingual_cased_mlm_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder",
            ],
            "pcc": 0.99,
            "max_int": 511,
        },
    ),
    (
        Embedding31,
        [((1, 384), torch.int64)],
        {"model_names": ["pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf"], "pcc": 0.99, "max_int": 28995},
    ),
    (
        Embedding21,
        [((1, 384), torch.int64)],
        {"model_names": ["pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf"], "pcc": 0.99, "max_int": 511},
    ),
    (
        Embedding9,
        [((1, 256), torch.int64)],
        {"model_names": ["pt_opt_facebook_opt_125m_clm_hf"], "pcc": 0.99, "max_int": 50271},
    ),
    (
        Embedding10,
        [((1, 256), torch.int64)],
        {"model_names": ["pt_opt_facebook_opt_125m_clm_hf"], "pcc": 0.99, "max_int": 2049},
    ),
    (
        Embedding32,
        [((1, 29), torch.int64)],
        {"model_names": ["pt_qwen_v2_1_5b_clm_hf"], "pcc": 0.99, "max_int": 151935},
    ),
    (
        Embedding33,
        [((1, 128), torch.int64)],
        {
            "model_names": [
                "onnx_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf",
                "pt_squeezebert_squeezebert_mnli_seq_cls_hf",
            ],
            "pcc": 0.99,
            "max_int": 30527,
        },
    ),
    (
        Embedding34,
        [((1, 11), torch.int64)],
        {"model_names": ["pd_albert_chinese_tiny_mlm_padlenlp"], "pcc": 0.99, "max_int": 21127},
    ),
    (
        Embedding3,
        [((1, 11), torch.int64)],
        {"model_names": ["pd_albert_chinese_tiny_mlm_padlenlp"], "pcc": 0.99, "max_int": 1},
    ),
    (
        Embedding4,
        [((1, 11), torch.int64)],
        {"model_names": ["pd_albert_chinese_tiny_mlm_padlenlp"], "pcc": 0.99, "max_int": 511},
    ),
    (
        Embedding20,
        [((1, 15), torch.int64)],
        {"model_names": ["pd_bert_bert_base_japanese_seq_cls_padlenlp"], "pcc": 0.99, "max_int": 31999},
    ),
    (
        Embedding21,
        [((1, 15), torch.int64)],
        {"model_names": ["pd_bert_bert_base_japanese_seq_cls_padlenlp"], "pcc": 0.99, "max_int": 511},
    ),
    (
        Embedding22,
        [((1, 15), torch.int64)],
        {"model_names": ["pd_bert_bert_base_japanese_seq_cls_padlenlp"], "pcc": 0.99, "max_int": 1},
    ),
    (
        Embedding23,
        [((1, 8), torch.int64)],
        {"model_names": ["pd_bert_bert_base_uncased_seq_cls_padlenlp"], "pcc": 0.99, "max_int": 30521},
    ),
    (
        Embedding21,
        [((1, 8), torch.int64)],
        {
            "model_names": [
                "pd_bert_bert_base_uncased_seq_cls_padlenlp",
                "pd_blip_text_salesforce_blip_image_captioning_base_text_enc_padlenlp",
            ],
            "pcc": 0.99,
            "max_int": 511,
        },
    ),
    (
        Embedding22,
        [((1, 8), torch.int64)],
        {"model_names": ["pd_bert_bert_base_uncased_seq_cls_padlenlp"], "pcc": 0.99, "max_int": 1},
    ),
    (
        Embedding23,
        [((1, 25), torch.int64)],
        {
            "model_names": [
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader",
            ],
            "pcc": 0.99,
            "max_int": 30521,
        },
    ),
    (
        Embedding22,
        [((1, 25), torch.int64)],
        {
            "model_names": [
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader",
            ],
            "pcc": 0.99,
            "max_int": 1,
        },
    ),
    (
        Embedding21,
        [((1, 25), torch.int64)],
        {
            "model_names": [
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader",
            ],
            "pcc": 0.99,
            "max_int": 511,
        },
    ),
    (
        Embedding35,
        [((1, 522), torch.int64)],
        {"model_names": ["pt_falcon3_tiiuae_falcon3_1b_base_clm_hf"], "pcc": 0.99, "max_int": 131071},
    ),
    (
        Embedding36,
        [((1, 5), torch.int64)],
        {"model_names": ["pt_gptneo_gpt_neo_1_3b_seq_cls_hf"], "pcc": 0.99, "max_int": 50256},
    ),
    (
        Embedding1,
        [((1, 5), torch.int64), ((2048, 2048), torch.float32)],
        {"model_names": ["pt_gptneo_gpt_neo_1_3b_seq_cls_hf"], "pcc": 0.99, "max_int": 2047},
    ),
    (
        Embedding37,
        [((1, 8), torch.int64)],
        {
            "model_names": ["pd_blip_text_salesforce_blip_image_captioning_base_text_enc_padlenlp"],
            "pcc": 0.99,
            "max_int": 30523,
        },
    ),
    (
        Embedding38,
        [((4, 5), torch.int64)],
        {
            "model_names": ["pd_chineseclip_ofa_sys_chinese_clip_vit_base_patch16_img_text_pairing_padlenlp"],
            "pcc": 0.99,
            "max_int": 21127,
        },
    ),
    (
        Embedding1,
        [((4, 5), torch.int64), ((512, 768), torch.float32)],
        {
            "model_names": ["pd_chineseclip_ofa_sys_chinese_clip_vit_base_patch16_img_text_pairing_padlenlp"],
            "pcc": 0.99,
            "max_int": 511,
        },
    ),
    (
        Embedding1,
        [((4, 5), torch.int64), ((2, 768), torch.float32)],
        {
            "model_names": ["pd_chineseclip_ofa_sys_chinese_clip_vit_base_patch16_img_text_pairing_padlenlp"],
            "pcc": 0.99,
            "max_int": 1,
        },
    ),
    (
        Embedding39,
        [((1, 9), torch.int64)],
        {
            "model_names": ["pd_ernie_1_0_mlm_padlenlp", "pd_ernie_1_0_qa_padlenlp", "pd_ernie_1_0_seq_cls_padlenlp"],
            "pcc": 0.99,
            "max_int": 17999,
        },
    ),
    (
        Embedding40,
        [((1, 9), torch.int64)],
        {
            "model_names": ["pd_ernie_1_0_mlm_padlenlp", "pd_ernie_1_0_qa_padlenlp", "pd_ernie_1_0_seq_cls_padlenlp"],
            "pcc": 0.99,
            "max_int": 512,
        },
    ),
    (
        Embedding41,
        [((1, 11), torch.int64)],
        {"model_names": ["pd_roberta_rbt4_ch_clm_padlenlp"], "pcc": 0.99, "max_int": 21127},
    ),
    (
        Embedding21,
        [((1, 11), torch.int64)],
        {
            "model_names": [
                "pd_roberta_rbt4_ch_clm_padlenlp",
                "pd_bert_chinese_roberta_base_qa_padlenlp",
                "pd_bert_chinese_roberta_base_seq_cls_padlenlp",
            ],
            "pcc": 0.99,
            "max_int": 511,
        },
    ),
    (
        Embedding1,
        [((1, 11), torch.int64), ((2, 768), torch.float32)],
        {"model_names": ["pd_roberta_rbt4_ch_clm_padlenlp"], "pcc": 0.99, "max_int": 1},
    ),
    (
        Embedding22,
        [((1, 128), torch.int64)],
        {
            "model_names": [
                "pt_bert_bert_base_uncased_mlm_hf",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder",
                "pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder",
            ],
            "pcc": 0.99,
            "max_int": 1,
        },
    ),
    (Embedding42, [((1, 6), torch.int64)], {"model_names": ["pt_qwen1_5_0_5b_clm_hf"], "pcc": 0.99, "max_int": 151935}),
    (
        Embedding43,
        [((4, 31), torch.int64)],
        {
            "model_names": ["pt_qwen_v3_embedding_4b_sentence_embed_gen_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "max_int": 151664,
        },
    ),
    (
        Embedding44,
        [((1, 513), torch.int64)],
        {
            "model_names": ["pt_t5_google_flan_t5_large_text_gen_hf", "pt_t5_t5_large_text_gen_hf"],
            "pcc": 0.99,
            "max_int": 32127,
        },
    ),
    (
        Embedding1,
        [((513, 513), torch.int32), ((32, 16), torch.float32)],
        {
            "model_names": ["pt_t5_google_flan_t5_large_text_gen_hf", "pt_t5_t5_large_text_gen_hf"],
            "pcc": 0.99,
            "max_int": 31,
        },
    ),
    (
        Embedding44,
        [((1, 61), torch.int64)],
        {
            "model_names": ["pt_t5_google_flan_t5_large_text_gen_hf", "pt_t5_t5_large_text_gen_hf"],
            "pcc": 0.99,
            "max_int": 32127,
        },
    ),
    (
        Embedding1,
        [((61, 61), torch.int32), ((32, 16), torch.float32)],
        {
            "model_names": ["pt_t5_google_flan_t5_large_text_gen_hf", "pt_t5_t5_large_text_gen_hf"],
            "pcc": 0.99,
            "max_int": 31,
        },
    ),
    (
        Embedding45,
        [((1, 256), torch.int64)],
        {"model_names": ["pt_bart_large_seq_cls_hf"], "pcc": 0.99, "max_int": 50264},
    ),
    (
        Embedding1,
        [((1, 256), torch.int64), ((1026, 1024), torch.float32)],
        {"model_names": ["pt_bart_large_seq_cls_hf"], "pcc": 0.99, "max_int": 1025},
    ),
    (
        Embedding20,
        [((1, 16), torch.int64)],
        {
            "model_names": ["pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf"],
            "pcc": 0.99,
            "max_int": 31999,
        },
    ),
    (
        Embedding22,
        [((1, 16), torch.int64)],
        {
            "model_names": ["pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf"],
            "pcc": 0.99,
            "max_int": 1,
        },
    ),
    (
        Embedding21,
        [((1, 16), torch.int64)],
        {
            "model_names": ["pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf"],
            "pcc": 0.99,
            "max_int": 511,
        },
    ),
    (
        Embedding46,
        [((1, 384), torch.int64)],
        {
            "model_names": [
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                "onnx_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
            ],
            "pcc": 0.99,
            "max_int": 30521,
        },
    ),
    (
        Embedding6,
        [((1, 384), torch.int64)],
        {
            "model_names": [
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                "pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf",
            ],
            "pcc": 0.99,
            "max_int": 1,
        },
    ),
    (
        Embedding7,
        [((1, 384), torch.int64)],
        {
            "model_names": [
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                "pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf",
            ],
            "pcc": 0.99,
            "max_int": 511,
        },
    ),
    (
        Embedding47,
        [((1, 32), torch.int64)],
        {"model_names": ["pt_bloom_default_clm_hf"], "pcc": 0.99, "max_int": 250879},
    ),
    (
        Embedding31,
        [((1, 128), torch.int64)],
        {"model_names": ["pt_distilbert_distilbert_base_cased_mlm_hf"], "pcc": 0.99, "max_int": 28995},
    ),
    (
        Embedding2,
        [((1, 5), torch.int64)],
        {"model_names": ["pt_gptneo_gpt_neo_125m_seq_cls_hf"], "pcc": 0.99, "max_int": 50256},
    ),
    (
        Embedding1,
        [((1, 5), torch.int64), ((2048, 768), torch.float32)],
        {"model_names": ["pt_gptneo_gpt_neo_125m_seq_cls_hf"], "pcc": 0.99, "max_int": 2047},
    ),
    (
        Embedding48,
        [((1, 256), torch.int64)],
        {"model_names": ["pt_opt_facebook_opt_1_3b_clm_hf"], "pcc": 0.99, "max_int": 50271},
    ),
    (
        Embedding49,
        [((1, 256), torch.int64)],
        {"model_names": ["pt_opt_facebook_opt_1_3b_clm_hf"], "pcc": 0.99, "max_int": 2049},
    ),
    (
        Embedding48,
        [((1, 32), torch.int64)],
        {
            "model_names": ["pt_opt_facebook_opt_1_3b_qa_hf", "pt_opt_facebook_opt_1_3b_seq_cls_hf"],
            "pcc": 0.99,
            "max_int": 50271,
        },
    ),
    (
        Embedding49,
        [((1, 32), torch.int64)],
        {
            "model_names": ["pt_opt_facebook_opt_1_3b_qa_hf", "pt_opt_facebook_opt_1_3b_seq_cls_hf"],
            "pcc": 0.99,
            "max_int": 2049,
        },
    ),
    (
        Embedding50,
        [((1, 24), torch.int64)],
        {"model_names": ["pt_speecht5_tts_tts_text_to_speech_hf"], "pcc": 0.99, "max_int": 80},
    ),
    (
        Embedding1,
        [((24, 24), torch.int64), ((320, 64), torch.float32)],
        {"model_names": ["pt_speecht5_tts_tts_text_to_speech_hf"], "pcc": 0.99, "max_int": 319},
    ),
    (
        Embedding51,
        [((1, 256), torch.int64)],
        {"model_names": ["pt_xglm_xglm_1_7b_clm_hf"], "pcc": 0.99, "max_int": 256007},
    ),
    (
        Embedding5,
        [((1, 384), torch.int64)],
        {
            "model_names": ["pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf"],
            "pcc": 0.99,
            "max_int": 28995,
        },
    ),
    (
        Embedding2,
        [((1, 256), torch.int64)],
        {"model_names": ["pt_gptneo_gpt_neo_125m_clm_hf", "pt_gpt_gpt2_text_gen_hf"], "pcc": 0.99, "max_int": 50256},
    ),
    (
        Embedding1,
        [((1, 256), torch.int64), ((2048, 768), torch.float32)],
        {"model_names": ["pt_gptneo_gpt_neo_125m_clm_hf"], "pcc": 0.99, "max_int": 2047},
    ),
    (
        Embedding52,
        [((1, 4), torch.int64)],
        {
            "model_names": ["pt_llama3_llama_3_2_1b_seq_cls_hf", "pt_llama3_llama_3_2_1b_instruct_seq_cls_hf"],
            "pcc": 0.99,
            "max_int": 128255,
        },
    ),
    (
        Embedding53,
        [((1, 11), torch.int64)],
        {
            "model_names": ["pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf", "pt_phi2_microsoft_phi_2_seq_cls_hf"],
            "pcc": 0.99,
            "max_int": 51199,
        },
    ),
    (
        Embedding42,
        [((1, 128), torch.int64)],
        {"model_names": ["pt_qwen_v3_0_6b_clm_hf"], "pcc": 0.99, "max_int": 151935},
    ),
    (
        Embedding54,
        [((1, 10), torch.int64)],
        {"model_names": ["pt_roberta_xlm_base_mlm_hf"], "pcc": 0.99, "max_int": 250001},
    ),
    (Embedding26, [((1, 10), torch.int64)], {"model_names": ["pt_roberta_xlm_base_mlm_hf"], "pcc": 0.99, "max_int": 0}),
    (
        Embedding27,
        [((1, 10), torch.int64)],
        {"model_names": ["pt_roberta_xlm_base_mlm_hf"], "pcc": 0.99, "max_int": 513},
    ),
    (
        Embedding1,
        [((1, 128), torch.int64), ((1, 768), torch.float32)],
        {
            "model_names": ["onnx_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf"],
            "pcc": 0.99,
            "max_int": 0,
        },
    ),
    (
        Embedding55,
        [((2, 7), torch.int64)],
        {
            "model_names": [
                "pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text",
                "onnx_clip_openai_clip_vit_base_patch32_text_gen_hf_text",
            ],
            "pcc": 0.99,
            "max_int": 49407,
        },
    ),
    (
        Embedding56,
        [((1, 7), torch.int64)],
        {"model_names": ["pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text"], "pcc": 0.99, "max_int": 76},
    ),
    (
        Embedding36,
        [((1, 256), torch.int64)],
        {"model_names": ["pt_gptneo_gpt_neo_1_3b_clm_hf"], "pcc": 0.99, "max_int": 50256},
    ),
    (
        Embedding1,
        [((1, 256), torch.int64), ((2048, 2048), torch.float32)],
        {"model_names": ["pt_gptneo_gpt_neo_1_3b_clm_hf"], "pcc": 0.99, "max_int": 2047},
    ),
    (
        Embedding14,
        [((1, 5), torch.int64)],
        {
            "model_names": ["pt_phi1_microsoft_phi_1_seq_cls_hf", "pt_phi_1_5_microsoft_phi_1_5_seq_cls_hf"],
            "pcc": 0.99,
            "max_int": 51199,
        },
    ),
    (
        Embedding57,
        [((1, 128), torch.int64)],
        {"model_names": ["pt_qwen_v3_1_7b_clm_hf"], "pcc": 0.99, "max_int": 151935},
    ),
    (
        Embedding17,
        [((1, 513), torch.int64)],
        {
            "model_names": ["pt_t5_t5_base_text_gen_hf", "pt_t5_google_flan_t5_base_text_gen_hf"],
            "pcc": 0.99,
            "max_int": 32127,
        },
    ),
    (
        Embedding1,
        [((513, 513), torch.int32), ((32, 12), torch.float32)],
        {
            "model_names": ["pt_t5_t5_base_text_gen_hf", "pt_t5_google_flan_t5_base_text_gen_hf"],
            "pcc": 0.99,
            "max_int": 31,
        },
    ),
    (
        Embedding17,
        [((1, 61), torch.int64)],
        {
            "model_names": ["pt_t5_t5_base_text_gen_hf", "pt_t5_google_flan_t5_base_text_gen_hf"],
            "pcc": 0.99,
            "max_int": 32127,
        },
    ),
    (
        Embedding1,
        [((61, 61), torch.int32), ((32, 12), torch.float32)],
        {
            "model_names": ["pt_t5_t5_base_text_gen_hf", "pt_t5_google_flan_t5_base_text_gen_hf"],
            "pcc": 0.99,
            "max_int": 31,
        },
    ),
    (
        Embedding58,
        [((1, 1), torch.int64)],
        {"model_names": ["onnx_whisper_openai_whisper_small_speech_recognition_hf"], "pcc": 0.99, "max_int": 51864},
    ),
    (
        Embedding1,
        [((1, 1), torch.int64), ((448, 768), torch.float32)],
        {"model_names": ["onnx_whisper_openai_whisper_small_speech_recognition_hf"], "pcc": 0.99, "max_int": 447},
    ),
    (
        Embedding20,
        [((1, 10), torch.int64)],
        {"model_names": ["pd_bert_bert_base_japanese_mlm_padlenlp"], "pcc": 0.99, "max_int": 31999},
    ),
    (
        Embedding21,
        [((1, 10), torch.int64)],
        {"model_names": ["pd_bert_bert_base_japanese_mlm_padlenlp"], "pcc": 0.99, "max_int": 511},
    ),
    (
        Embedding22,
        [((1, 10), torch.int64)],
        {"model_names": ["pd_bert_bert_base_japanese_mlm_padlenlp"], "pcc": 0.99, "max_int": 1},
    ),
    (
        Embedding38,
        [((1, 8), torch.int64)],
        {
            "model_names": ["pd_chineseclip_text_ofa_sys_chinese_clip_vit_base_patch16_text_enc_padlenlp"],
            "pcc": 0.99,
            "max_int": 21127,
        },
    ),
    (
        Embedding1,
        [((1, 8), torch.int64), ((512, 768), torch.float32)],
        {
            "model_names": ["pd_chineseclip_text_ofa_sys_chinese_clip_vit_base_patch16_text_enc_padlenlp"],
            "pcc": 0.99,
            "max_int": 511,
        },
    ),
    (
        Embedding1,
        [((1, 8), torch.int64), ((2, 768), torch.float32)],
        {
            "model_names": ["pd_chineseclip_text_ofa_sys_chinese_clip_vit_base_patch16_text_enc_padlenlp"],
            "pcc": 0.99,
            "max_int": 1,
        },
    ),
    (
        Embedding0,
        [((1, 9), torch.int64)],
        {"model_names": ["pt_albert_imdb_seq_cls_hf"], "pcc": 0.99, "max_int": 29999},
    ),
    (Embedding3, [((1, 9), torch.int64)], {"model_names": ["pt_albert_imdb_seq_cls_hf"], "pcc": 0.99, "max_int": 1}),
    (Embedding4, [((1, 9), torch.int64)], {"model_names": ["pt_albert_imdb_seq_cls_hf"], "pcc": 0.99, "max_int": 511}),
    (
        Embedding1,
        [((1, 256), torch.int64), ((1024, 768), torch.float32)],
        {"model_names": ["pt_gpt_gpt2_text_gen_hf"], "pcc": 0.99, "max_int": 1023},
    ),
    (
        Embedding11,
        [((1, 256), torch.int64)],
        {"model_names": ["pt_opt_facebook_opt_350m_clm_hf"], "pcc": 0.99, "max_int": 50271},
    ),
    (
        Embedding12,
        [((1, 256), torch.int64)],
        {"model_names": ["pt_opt_facebook_opt_350m_clm_hf"], "pcc": 0.99, "max_int": 2049},
    ),
    (
        Embedding32,
        [((1, 35), torch.int64)],
        {"model_names": ["pt_qwen_coder_1_5b_clm_hf"], "pcc": 0.99, "max_int": 151935},
    ),
    (
        Embedding59,
        [((1, 1), torch.int64)],
        {"model_names": ["pt_stereo_small_music_generation_hf"], "pcc": 0.99, "max_int": 2048},
    ),
    (
        Embedding28,
        [((1, 513), torch.int64)],
        {
            "model_names": ["pt_t5_t5_small_text_gen_hf", "pt_t5_google_flan_t5_small_text_gen_hf"],
            "pcc": 0.99,
            "max_int": 32127,
        },
    ),
    (
        Embedding1,
        [((513, 513), torch.int32), ((32, 8), torch.float32)],
        {"model_names": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99, "max_int": 31},
    ),
    (
        Embedding1,
        [((61, 61), torch.int32), ((32, 8), torch.float32)],
        {"model_names": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99, "max_int": 31},
    ),
    (
        Embedding20,
        [((1, 6), torch.int64)],
        {
            "model_names": ["onnx_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf"],
            "pcc": 0.99,
            "max_int": 31999,
        },
    ),
    (
        Embedding22,
        [((1, 6), torch.int64)],
        {
            "model_names": ["onnx_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf"],
            "pcc": 0.99,
            "max_int": 1,
        },
    ),
    (
        Embedding1,
        [((1, 6), torch.int64), ((512, 768), torch.float32)],
        {
            "model_names": ["onnx_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf"],
            "pcc": 0.99,
            "max_int": 511,
        },
    ),
    (
        Embedding1,
        [((1, 7), torch.int64), ((77, 512), torch.float32)],
        {"model_names": ["onnx_clip_openai_clip_vit_base_patch32_text_gen_hf_text"], "pcc": 0.99, "max_int": 76},
    ),
    (
        Embedding38,
        [((1, 9), torch.int64)],
        {"model_names": ["pd_bert_chinese_roberta_base_mlm_padlenlp"], "pcc": 0.99, "max_int": 21127},
    ),
    (
        Embedding38,
        [((1, 11), torch.int64)],
        {
            "model_names": [
                "pd_bert_chinese_roberta_base_qa_padlenlp",
                "pd_bert_chinese_roberta_base_seq_cls_padlenlp",
            ],
            "pcc": 0.99,
            "max_int": 21127,
        },
    ),
    (
        Embedding22,
        [((1, 11), torch.int64)],
        {
            "model_names": [
                "pd_bert_chinese_roberta_base_qa_padlenlp",
                "pd_bert_chinese_roberta_base_seq_cls_padlenlp",
            ],
            "pcc": 0.99,
            "max_int": 1,
        },
    ),
    (
        Embedding37,
        [((2, 4), torch.int64)],
        {
            "model_names": ["pd_blip_salesforce_blip_image_captioning_base_img_captioning_padlenlp"],
            "pcc": 0.99,
            "max_int": 30523,
        },
    ),
    (
        Embedding21,
        [((1, 4), torch.int64)],
        {
            "model_names": ["pd_blip_salesforce_blip_image_captioning_base_img_captioning_padlenlp"],
            "pcc": 0.99,
            "max_int": 511,
        },
    ),
    (
        Embedding42,
        [((1, 29), torch.int64)],
        {"model_names": ["pt_qwen1_5_0_5b_chat_clm_hf"], "pcc": 0.99, "max_int": 151935},
    ),
    (
        Embedding15,
        [((1, 39), torch.int64)],
        {"model_names": ["pt_qwen_v2_0_5b_instruct_clm_hf"], "pcc": 0.99, "max_int": 151935},
    ),
    (
        Embedding1,
        [((513, 513), torch.int32), ((32, 6), torch.float32)],
        {"model_names": ["pt_t5_google_flan_t5_small_text_gen_hf"], "pcc": 0.99, "max_int": 31},
    ),
    (
        Embedding1,
        [((61, 61), torch.int32), ((32, 6), torch.float32)],
        {"model_names": ["pt_t5_google_flan_t5_small_text_gen_hf"], "pcc": 0.99, "max_int": 31},
    ),
    (
        Embedding1,
        [((1, 577), torch.int64), ((577, 1024), torch.float32)],
        {"model_names": ["pt_llava_1_5_7b_cond_gen_hf"], "pcc": 0.99, "max_int": 576},
    ),
    (
        Embedding60,
        [((1, 596), torch.int64)],
        {"model_names": ["pt_llava_1_5_7b_cond_gen_hf"], "pcc": 0.99, "max_int": 32063},
    ),
    (
        Embedding61,
        [((1, 12), torch.int64)],
        {
            "model_names": ["pt_ministral_ministral_3b_instruct_clm_hf", "pt_mistral_7b_clm_hf"],
            "pcc": 0.99,
            "max_int": 31999,
        },
    ),
    (
        Embedding62,
        [((1, 12), torch.int64)],
        {"model_names": ["pt_ministral_ministral_8b_instruct_clm_hf"], "pcc": 0.99, "max_int": 131071},
    ),
    (
        Embedding63,
        [((1, 12), torch.int64)],
        {"model_names": ["pt_mistral_7b_instruct_v03_clm_hf"], "pcc": 0.99, "max_int": 32767},
    ),
    (
        Embedding64,
        [((1, 256), torch.int64)],
        {
            "model_names": [
                "pt_phi3_5_mini_instruct_clm_hf",
                "pt_phi3_microsoft_phi_3_mini_128k_instruct_clm_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "max_int": 32063,
        },
    ),
    (
        Embedding65,
        [((1, 1), torch.int64)],
        {"model_names": ["pt_stereo_large_music_generation_hf"], "pcc": 0.99, "max_int": 2048},
    ),
    (
        Embedding41,
        [((1, 9), torch.int64)],
        {"model_names": ["pd_roberta_rbt4_ch_seq_cls_padlenlp"], "pcc": 0.99, "max_int": 21127},
    ),
    (
        Embedding1,
        [((1, 9), torch.int64), ((2, 768), torch.float32)],
        {"model_names": ["pd_roberta_rbt4_ch_seq_cls_padlenlp"], "pcc": 0.99, "max_int": 1},
    ),
    (
        Embedding66,
        [((1, 2), torch.int64)],
        {
            "model_names": [
                "pt_whisper_openai_whisper_large_v3_clm_hf",
                "pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf",
            ],
            "pcc": 0.99,
            "max_int": 51865,
        },
    ),
    (
        Embedding67,
        [((1, 256), torch.int64)],
        {"model_names": ["pt_xglm_xglm_564m_clm_hf"], "pcc": 0.99, "max_int": 256007},
    ),
    (
        Embedding52,
        [((1, 256), torch.int64)],
        {
            "model_names": ["pt_llama3_llama_3_2_1b_clm_hf", "pt_llama3_llama_3_2_1b_instruct_clm_hf"],
            "pcc": 0.99,
            "max_int": 128255,
        },
    ),
    (
        Embedding1,
        [((1, 384), torch.int64), ((2, 1024), torch.float32)],
        {"model_names": ["onnx_bert_phiyodr_bert_large_finetuned_squad2_qa_hf"], "pcc": 0.99, "max_int": 1},
    ),
    (
        Embedding1,
        [((1, 384), torch.int64), ((512, 1024), torch.float32)],
        {"model_names": ["onnx_bert_phiyodr_bert_large_finetuned_squad2_qa_hf"], "pcc": 0.99, "max_int": 511},
    ),
    (
        Embedding68,
        [((1, 1063), torch.int64)],
        {"model_names": ["pt_deepseek_7b_instruct_qa_hf"], "pcc": 0.99, "max_int": 102399},
    ),
    (
        Embedding69,
        [((1, 522), torch.int64)],
        {
            "model_names": ["pt_falcon3_tiiuae_falcon3_3b_base_clm_hf", "pt_falcon3_tiiuae_falcon3_7b_base_clm_hf"],
            "pcc": 0.99,
            "max_int": 131071,
        },
    ),
    (
        Embedding70,
        [((1, 6), torch.int64)],
        {"model_names": ["pt_falcon_tiiuae_falcon_7b_instruct_clm_hf"], "pcc": 0.99, "max_int": 65023},
    ),
    (
        Embedding71,
        [((1, 356), torch.int64)],
        {"model_names": ["pt_gemma_google_gemma_1_1_2b_it_qa_hf"], "pcc": 0.99, "max_int": 255999},
    ),
    (
        Embedding72,
        [((1, 356), torch.int64)],
        {"model_names": ["pt_gemma_google_gemma_1_1_7b_it_qa_hf"], "pcc": 0.99, "max_int": 255999},
    ),
    (
        Embedding71,
        [((1, 512), torch.int64)],
        {"model_names": ["pt_gemma_google_gemma_2b_text_gen_hf"], "pcc": 0.99, "max_int": 255999},
    ),
    (
        Embedding73,
        [((1, 256), torch.int64)],
        {"model_names": ["pt_gptneo_gpt_neo_2_7b_clm_hf"], "pcc": 0.99, "max_int": 50256},
    ),
    (
        Embedding1,
        [((1, 256), torch.int64), ((2048, 2560), torch.float32)],
        {"model_names": ["pt_gptneo_gpt_neo_2_7b_clm_hf"], "pcc": 0.99, "max_int": 2047},
    ),
    (
        Embedding73,
        [((1, 5), torch.int64)],
        {"model_names": ["pt_gptneo_gpt_neo_2_7b_seq_cls_hf"], "pcc": 0.99, "max_int": 50256},
    ),
    (
        Embedding1,
        [((1, 5), torch.int64), ((2048, 2560), torch.float32)],
        {"model_names": ["pt_gptneo_gpt_neo_2_7b_seq_cls_hf"], "pcc": 0.99, "max_int": 2047},
    ),
    (
        Embedding61,
        [((1, 256), torch.int64)],
        {"model_names": ["pt_llama3_huggyllama_7b_clm_hf"], "pcc": 0.99, "max_int": 31999},
    ),
    (
        Embedding61,
        [((1, 4), torch.int64)],
        {"model_names": ["pt_llama3_huggyllama_7b_seq_cls_hf"], "pcc": 0.99, "max_int": 31999},
    ),
    (
        Embedding74,
        [((1, 256), torch.int64)],
        {
            "model_names": [
                "pt_llama3_llama_3_1_8b_clm_hf",
                "pt_llama3_llama_3_1_8b_instruct_clm_hf",
                "pt_llama3_llama_3_8b_clm_hf",
                "pt_llama3_llama_3_8b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "max_int": 128255,
        },
    ),
    (
        Embedding74,
        [((1, 4), torch.int64)],
        {
            "model_names": [
                "pt_llama3_llama_3_1_8b_instruct_seq_cls_hf",
                "pt_llama3_llama_3_1_8b_seq_cls_hf",
                "pt_llama3_llama_3_8b_instruct_seq_cls_hf",
                "pt_llama3_llama_3_8b_seq_cls_hf",
            ],
            "pcc": 0.99,
            "max_int": 128255,
        },
    ),
    (
        Embedding75,
        [((1, 256), torch.int64)],
        {
            "model_names": ["pt_llama3_llama_3_2_3b_clm_hf", "pt_llama3_llama_3_2_3b_instruct_clm_hf"],
            "pcc": 0.99,
            "max_int": 128255,
        },
    ),
    (
        Embedding75,
        [((1, 4), torch.int64)],
        {
            "model_names": ["pt_llama3_llama_3_2_3b_instruct_seq_cls_hf", "pt_llama3_llama_3_2_3b_seq_cls_hf"],
            "pcc": 0.99,
            "max_int": 128255,
        },
    ),
    (
        Embedding53,
        [((1, 256), torch.int64)],
        {"model_names": ["pt_phi2_microsoft_phi_2_pytdml_clm_hf"], "pcc": 0.99, "max_int": 51199},
    ),
    (
        Embedding53,
        [((1, 12), torch.int64)],
        {
            "model_names": ["pt_phi2_microsoft_phi_2_pytdml_token_cls_hf", "pt_phi2_microsoft_phi_2_token_cls_hf"],
            "pcc": 0.99,
            "max_int": 51199,
        },
    ),
    (
        Embedding64,
        [((1, 5), torch.int64)],
        {
            "model_names": [
                "pt_phi3_microsoft_phi_3_mini_128k_instruct_seq_cls_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf",
            ],
            "pcc": 0.99,
            "max_int": 32063,
        },
    ),
    (
        Embedding64,
        [((1, 13), torch.int64)],
        {
            "model_names": [
                "pt_phi3_microsoft_phi_3_mini_128k_instruct_token_cls_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf",
            ],
            "pcc": 0.99,
            "max_int": 32063,
        },
    ),
    (
        Embedding76,
        [((1, 5), torch.int64)],
        {"model_names": ["pt_phi4_microsoft_phi_4_seq_cls_hf"], "pcc": 0.99, "max_int": 100351},
    ),
    (
        Embedding76,
        [((1, 12), torch.int64)],
        {"model_names": ["pt_phi4_microsoft_phi_4_token_cls_hf"], "pcc": 0.99, "max_int": 100351},
    ),
    (
        Embedding77,
        [((1, 13), torch.int64)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_7b_token_cls_hf"], "pcc": 0.99, "max_int": 152063},
    ),
    (Embedding78, [((1, 128), torch.int64)], {"model_names": ["pt_qwen_v3_4b_clm_hf"], "pcc": 0.99, "max_int": 151935}),
    (
        Embedding79,
        [((4, 31), torch.int64)],
        {
            "model_names": ["pt_qwen_v3_embedding_0_6b_sentence_embed_gen_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "max_int": 151668,
        },
    ),
    (
        Embedding29,
        [((1, 101), torch.int64)],
        {"model_names": ["pt_whisper_openai_whisper_base_speech_recognition_hf"], "pcc": 0.99, "max_int": 51864},
    ),
    (
        Embedding80,
        [((1, 101), torch.int64)],
        {"model_names": ["pt_whisper_openai_whisper_large_speech_recognition_hf"], "pcc": 0.99, "max_int": 51864},
    ),
    (
        Embedding81,
        [((1, 101), torch.int64)],
        {"model_names": ["pt_whisper_openai_whisper_medium_speech_recognition_hf"], "pcc": 0.99, "max_int": 51864},
    ),
    (
        Embedding58,
        [((1, 101), torch.int64)],
        {"model_names": ["pt_whisper_openai_whisper_small_speech_recognition_hf"], "pcc": 0.99, "max_int": 51864},
    ),
    (
        Embedding19,
        [((1, 101), torch.int64)],
        {"model_names": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"], "pcc": 0.99, "max_int": 51864},
    ),
]


@pytest.mark.nightly_models_ops
@pytest.mark.parametrize("forge_module_and_shapes_dtypes", forge_modules_and_shapes_dtypes_list, ids=ids_func)
@pytest.mark.parametrize("training_test", [False, True], ids=["inference", "training"])
def test_module(forge_module_and_shapes_dtypes, training_test):

    record_forge_op_name("Embedding")

    forge_module, operand_shapes_dtypes, metadata = forge_module_and_shapes_dtypes

    pcc = metadata.get("pcc")
    max_int = metadata.get("max_int")

    for metadata_name, metadata_value in metadata.items():
        if metadata_name in ["pcc", "max_int"]:
            continue
        elif metadata_name == "model_names":
            record_op_model_names(metadata_value)
        elif metadata_name == "args":
            record_forge_op_args(metadata_value)
        else:
            logger.warning(
                "No utility function available in forge property handler to record %s property", metadata_name
            )

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
