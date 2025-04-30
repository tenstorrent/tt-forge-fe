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


class Embedding0(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding0.weight_1",
            forge.Parameter(*(30522, 1024), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(32000, 768), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding2.weight_1"))
        return embedding_output_1


class Embedding3(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding3.weight_1",
            forge.Parameter(*(2, 768), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding3.weight_1"))
        return embedding_output_1


class Embedding4(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding4.weight_1",
            forge.Parameter(*(30522, 768), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding4.weight_1"))
        return embedding_output_1


class Embedding5(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding5.weight_1",
            forge.Parameter(*(30522, 384), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding5.weight_1"))
        return embedding_output_1


class Embedding6(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding6.weight_1",
            forge.Parameter(*(30524, 768), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding6.weight_1"))
        return embedding_output_1


class Embedding7(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding7.weight_1",
            forge.Parameter(*(512, 768), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding7.weight_1"))
        return embedding_output_1


class Embedding8(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding8.weight_1",
            forge.Parameter(*(21128, 768), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding8.weight_1"))
        return embedding_output_1


class Embedding9(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("embedding9_const_1", shape=(21128, 128), dtype=torch.float32)

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_constant("embedding9_const_1"))
        return embedding_output_1


class Embedding10(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding10.weight_1",
            forge.Parameter(*(2, 128), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding10.weight_1"))
        return embedding_output_1


class Embedding11(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding11.weight_1",
            forge.Parameter(*(512, 128), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding11.weight_1"))
        return embedding_output_1


class Embedding12(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("embedding12_const_1", shape=(18000, 768), dtype=torch.float32)

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_constant("embedding12_const_1"))
        return embedding_output_1


class Embedding13(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding13.weight_1",
            forge.Parameter(*(513, 768), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding13.weight_1"))
        return embedding_output_1


class Embedding14(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("embedding14_const_1", shape=(21128, 768), dtype=torch.float32)

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_constant("embedding14_const_1"))
        return embedding_output_1


class Embedding15(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding15.weight_1",
            forge.Parameter(*(2049, 2048), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding15.weight_1"))
        return embedding_output_1


class Embedding16(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding16.weight_1",
            forge.Parameter(*(32128, 768), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding16.weight_1"))
        return embedding_output_1


class Embedding17(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding17.weight_1",
            forge.Parameter(*(2049, 1536), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding17.weight_1"))
        return embedding_output_1


class Embedding18(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding18.weight_1",
            forge.Parameter(*(51865, 512), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding18.weight_1"))
        return embedding_output_1


class Embedding19(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding19.weight_1",
            forge.Parameter(*(51865, 1280), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding19.weight_1"))
        return embedding_output_1


class Embedding20(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding20.weight_1",
            forge.Parameter(*(51865, 768), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding20.weight_1"))
        return embedding_output_1


class Embedding21(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding21.weight_1",
            forge.Parameter(*(51865, 1024), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding21.weight_1"))
        return embedding_output_1


class Embedding22(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding22.weight_1",
            forge.Parameter(*(51865, 384), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding22.weight_1"))
        return embedding_output_1


class Embedding23(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding23.weight_1",
            forge.Parameter(*(51866, 1280), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding23.weight_1"))
        return embedding_output_1


class Embedding24(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding24.weight_1",
            forge.Parameter(*(49408, 512), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding24.weight_1"))
        return embedding_output_1


class Embedding25(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding25.weight_1",
            forge.Parameter(*(77, 512), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding25.weight_1"))
        return embedding_output_1


class Embedding26(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding26.weight_1",
            forge.Parameter(*(32256, 2048), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding26.weight_1"))
        return embedding_output_1


class Embedding27(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding27.weight_1",
            forge.Parameter(*(102400, 4096), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding27.weight_1"))
        return embedding_output_1


class Embedding28(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding28.weight_1",
            forge.Parameter(*(32064, 4096), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding28.weight_1"))
        return embedding_output_1


class Embedding29(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding29.weight_1",
            forge.Parameter(*(30000, 128), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding29.weight_1"))
        return embedding_output_1


class Embedding30(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding30.weight_1",
            forge.Parameter(*(50265, 1024), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding30.weight_1"))
        return embedding_output_1


class Embedding31(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding31.weight_1",
            forge.Parameter(*(28996, 1024), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding31.weight_1"))
        return embedding_output_1


class Embedding32(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding32.weight_1",
            forge.Parameter(*(2, 1024), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding32.weight_1"))
        return embedding_output_1


class Embedding33(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding33.weight_1",
            forge.Parameter(*(512, 1024), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding33.weight_1"))
        return embedding_output_1


class Embedding34(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding34.weight_1",
            forge.Parameter(*(250880, 1536), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding34.weight_1"))
        return embedding_output_1


class Embedding35(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding35.weight_1",
            forge.Parameter(*(51200, 1024), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding35.weight_1"))
        return embedding_output_1


class Embedding36(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding36.weight_1",
            forge.Parameter(*(119547, 768), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding36.weight_1"))
        return embedding_output_1


class Embedding37(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding37.weight_1",
            forge.Parameter(*(28996, 768), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding37.weight_1"))
        return embedding_output_1


class Embedding38(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding38.weight_1",
            forge.Parameter(*(131072, 3072), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding38.weight_1"))
        return embedding_output_1


class Embedding39(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding39.weight_1",
            forge.Parameter(*(131072, 2048), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding39.weight_1"))
        return embedding_output_1


class Embedding40(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding40.weight_1",
            forge.Parameter(*(65024, 4544), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding40.weight_1"))
        return embedding_output_1


class Embedding41(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding41.weight_1",
            forge.Parameter(*(256000, 2304), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding41.weight_1"))
        return embedding_output_1


class Embedding42(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding42.weight_1",
            forge.Parameter(*(256000, 3584), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding42.weight_1"))
        return embedding_output_1


class Embedding43(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding43.weight_1",
            forge.Parameter(*(256000, 2048), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding43.weight_1"))
        return embedding_output_1


class Embedding44(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding44.weight_1",
            forge.Parameter(*(256000, 3072), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding44.weight_1"))
        return embedding_output_1


class Embedding45(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding45.weight_1",
            forge.Parameter(*(50257, 768), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding45.weight_1"))
        return embedding_output_1


class Embedding46(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding46.weight_1",
            forge.Parameter(*(50257, 2560), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding46.weight_1"))
        return embedding_output_1


class Embedding47(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding47.weight_1",
            forge.Parameter(*(50257, 2048), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding47.weight_1"))
        return embedding_output_1


class Embedding48(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding48.weight_1",
            forge.Parameter(*(32000, 4096), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding48.weight_1"))
        return embedding_output_1


class Embedding49(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding49.weight_1",
            forge.Parameter(*(128256, 3072), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding49.weight_1"))
        return embedding_output_1


class Embedding50(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding50.weight_1",
            forge.Parameter(*(128256, 2048), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding50.weight_1"))
        return embedding_output_1


class Embedding51(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding51.weight_1",
            forge.Parameter(*(128256, 4096), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding51.weight_1"))
        return embedding_output_1


class Embedding52(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding52.weight_1",
            forge.Parameter(*(131072, 4096), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding52.weight_1"))
        return embedding_output_1


class Embedding53(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding53.weight_1",
            forge.Parameter(*(32768, 4096), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding53.weight_1"))
        return embedding_output_1


class Embedding54(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding54.weight_1",
            forge.Parameter(*(50272, 768), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding54.weight_1"))
        return embedding_output_1


class Embedding55(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding55.weight_1",
            forge.Parameter(*(2050, 768), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding55.weight_1"))
        return embedding_output_1


class Embedding56(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding56.weight_1",
            forge.Parameter(*(50272, 512), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding56.weight_1"))
        return embedding_output_1


class Embedding57(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding57.weight_1",
            forge.Parameter(*(2050, 1024), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding57.weight_1"))
        return embedding_output_1


class Embedding58(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding58.weight_1",
            forge.Parameter(*(50272, 2048), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding58.weight_1"))
        return embedding_output_1


class Embedding59(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding59.weight_1",
            forge.Parameter(*(2050, 2048), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding59.weight_1"))
        return embedding_output_1


class Embedding60(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding60.weight_1",
            forge.Parameter(*(262, 768), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding60.weight_1"))
        return embedding_output_1


class Embedding61(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding61.weight_1",
            forge.Parameter(*(51200, 2048), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding61.weight_1"))
        return embedding_output_1


class Embedding62(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding62.weight_1",
            forge.Parameter(*(51200, 2560), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding62.weight_1"))
        return embedding_output_1


class Embedding63(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding63.weight_1",
            forge.Parameter(*(32064, 3072), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding63.weight_1"))
        return embedding_output_1


class Embedding64(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding64.weight_1",
            forge.Parameter(*(100352, 5120), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding64.weight_1"))
        return embedding_output_1


class Embedding65(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding65.weight_1",
            forge.Parameter(*(151936, 1024), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding65.weight_1"))
        return embedding_output_1


class Embedding66(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding66.weight_1",
            forge.Parameter(*(152064, 3584), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding66.weight_1"))
        return embedding_output_1


class Embedding67(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding67.weight_1",
            forge.Parameter(*(151936, 1536), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding67.weight_1"))
        return embedding_output_1


class Embedding68(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding68.weight_1",
            forge.Parameter(*(151936, 896), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding68.weight_1"))
        return embedding_output_1


class Embedding69(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding69.weight_1",
            forge.Parameter(*(151936, 2048), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding69.weight_1"))
        return embedding_output_1


class Embedding70(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding70.weight_1",
            forge.Parameter(*(250002, 768), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding70.weight_1"))
        return embedding_output_1


class Embedding71(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding71.weight_1",
            forge.Parameter(*(1, 768), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding71.weight_1"))
        return embedding_output_1


class Embedding72(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding72.weight_1",
            forge.Parameter(*(514, 768), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding72.weight_1"))
        return embedding_output_1


class Embedding73(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding73.weight_1",
            forge.Parameter(*(50265, 768), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding73.weight_1"))
        return embedding_output_1


class Embedding74(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding74.weight_1",
            forge.Parameter(*(30528, 768), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding74.weight_1"))
        return embedding_output_1


class Embedding75(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding75.weight_1",
            forge.Parameter(*(32128, 1024), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding75.weight_1"))
        return embedding_output_1


class Embedding76(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding76.weight_1",
            forge.Parameter(*(32128, 512), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding76.weight_1"))
        return embedding_output_1


class Embedding77(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding77.weight_1",
            forge.Parameter(*(256008, 1024), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding77.weight_1"))
        return embedding_output_1


class Embedding78(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding78.weight_1",
            forge.Parameter(*(256008, 2048), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding78.weight_1"))
        return embedding_output_1


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (
        Embedding0,
        [((1, 384), torch.int64)],
        {
            "model_names": [
                "onnx_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
            ],
            "pcc": 0.99,
            "max_int": 30521,
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
        Embedding2,
        [((1, 6), torch.int64)],
        {
            "model_names": [
                "onnx_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf",
                "pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf",
            ],
            "pcc": 0.99,
            "max_int": 31999,
        },
    ),
    (
        Embedding3,
        [((1, 6), torch.int64)],
        {
            "model_names": [
                "onnx_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf",
                "pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf",
            ],
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
        Embedding4,
        [((1, 128), torch.int64)],
        {
            "model_names": [
                "onnx_bert_bert_base_uncased_mlm_hf",
                "pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf",
                "pt_bert_bert_base_uncased_mlm_hf",
                "pt_distilbert_distilbert_base_uncased_mlm_hf",
                "pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader",
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
                "pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf",
            ],
            "pcc": 0.99,
            "max_int": 1,
        },
    ),
    (
        Embedding1,
        [((1, 128), torch.int64), ((512, 768), torch.float32)],
        {"model_names": ["onnx_bert_bert_base_uncased_mlm_hf"], "pcc": 0.99, "max_int": 511},
    ),
    (
        Embedding5,
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
        Embedding6,
        [((1, 8), torch.int64)],
        {
            "model_names": ["pd_blip_salesforce_blip_image_captioning_base_img_enc_padlenlp"],
            "pcc": 0.99,
            "max_int": 30523,
        },
    ),
    (
        Embedding7,
        [((1, 8), torch.int64)],
        {
            "model_names": [
                "pd_blip_salesforce_blip_image_captioning_base_img_enc_padlenlp",
                "pd_bert_bert_base_uncased_seq_cls_padlenlp",
            ],
            "pcc": 0.99,
            "max_int": 511,
        },
    ),
    (
        Embedding8,
        [((1, 8), torch.int64)],
        {
            "model_names": ["pd_chineseclip_ofa_sys_chinese_clip_vit_base_patch16_img_enc_padlenlp"],
            "pcc": 0.99,
            "max_int": 21127,
        },
    ),
    (
        Embedding1,
        [((1, 8), torch.int64), ((512, 768), torch.float32)],
        {
            "model_names": ["pd_chineseclip_ofa_sys_chinese_clip_vit_base_patch16_img_enc_padlenlp"],
            "pcc": 0.99,
            "max_int": 511,
        },
    ),
    (
        Embedding1,
        [((1, 8), torch.int64), ((2, 768), torch.float32)],
        {
            "model_names": ["pd_chineseclip_ofa_sys_chinese_clip_vit_base_patch16_img_enc_padlenlp"],
            "pcc": 0.99,
            "max_int": 1,
        },
    ),
    (
        Embedding9,
        [((1, 11), torch.int64)],
        {"model_names": ["pd_albert_chinese_tiny_mlm_padlenlp"], "pcc": 0.99, "max_int": 21127},
    ),
    (
        Embedding10,
        [((1, 11), torch.int64)],
        {"model_names": ["pd_albert_chinese_tiny_mlm_padlenlp"], "pcc": 0.99, "max_int": 1},
    ),
    (
        Embedding11,
        [((1, 11), torch.int64)],
        {"model_names": ["pd_albert_chinese_tiny_mlm_padlenlp"], "pcc": 0.99, "max_int": 511},
    ),
    (
        Embedding4,
        [((1, 9), torch.int64)],
        {
            "model_names": ["pd_bert_bert_base_uncased_qa_padlenlp", "pd_bert_bert_base_uncased_mlm_padlenlp"],
            "pcc": 0.99,
            "max_int": 30521,
        },
    ),
    (
        Embedding7,
        [((1, 9), torch.int64)],
        {
            "model_names": [
                "pd_bert_bert_base_uncased_qa_padlenlp",
                "pd_bert_bert_base_uncased_mlm_padlenlp",
                "pd_bert_chinese_roberta_base_mlm_padlenlp",
                "pd_roberta_rbt4_ch_seq_cls_padlenlp",
            ],
            "pcc": 0.99,
            "max_int": 511,
        },
    ),
    (
        Embedding3,
        [((1, 9), torch.int64)],
        {
            "model_names": [
                "pd_bert_bert_base_uncased_qa_padlenlp",
                "pd_bert_bert_base_uncased_mlm_padlenlp",
                "pd_bert_chinese_roberta_base_mlm_padlenlp",
                "pd_ernie_1_0_qa_padlenlp",
                "pd_ernie_1_0_seq_cls_padlenlp",
                "pd_ernie_1_0_mlm_padlenlp",
            ],
            "pcc": 0.99,
            "max_int": 1,
        },
    ),
    (
        Embedding2,
        [((1, 15), torch.int64)],
        {"model_names": ["pd_bert_bert_base_japanese_seq_cls_padlenlp"], "pcc": 0.99, "max_int": 31999},
    ),
    (
        Embedding7,
        [((1, 15), torch.int64)],
        {"model_names": ["pd_bert_bert_base_japanese_seq_cls_padlenlp"], "pcc": 0.99, "max_int": 511},
    ),
    (
        Embedding3,
        [((1, 15), torch.int64)],
        {"model_names": ["pd_bert_bert_base_japanese_seq_cls_padlenlp"], "pcc": 0.99, "max_int": 1},
    ),
    (
        Embedding8,
        [((1, 11), torch.int64)],
        {
            "model_names": [
                "pd_bert_chinese_roberta_base_seq_cls_padlenlp",
                "pd_bert_chinese_roberta_base_qa_padlenlp",
            ],
            "pcc": 0.99,
            "max_int": 21127,
        },
    ),
    (
        Embedding7,
        [((1, 11), torch.int64)],
        {
            "model_names": [
                "pd_bert_chinese_roberta_base_seq_cls_padlenlp",
                "pd_bert_chinese_roberta_base_qa_padlenlp",
                "pd_roberta_rbt4_ch_clm_padlenlp",
            ],
            "pcc": 0.99,
            "max_int": 511,
        },
    ),
    (
        Embedding3,
        [((1, 11), torch.int64)],
        {
            "model_names": [
                "pd_bert_chinese_roberta_base_seq_cls_padlenlp",
                "pd_bert_chinese_roberta_base_qa_padlenlp",
            ],
            "pcc": 0.99,
            "max_int": 1,
        },
    ),
    (
        Embedding8,
        [((1, 9), torch.int64)],
        {"model_names": ["pd_bert_chinese_roberta_base_mlm_padlenlp"], "pcc": 0.99, "max_int": 21127},
    ),
    (
        Embedding4,
        [((1, 8), torch.int64)],
        {"model_names": ["pd_bert_bert_base_uncased_seq_cls_padlenlp"], "pcc": 0.99, "max_int": 30521},
    ),
    (
        Embedding3,
        [((1, 8), torch.int64)],
        {"model_names": ["pd_bert_bert_base_uncased_seq_cls_padlenlp"], "pcc": 0.99, "max_int": 1},
    ),
    (
        Embedding2,
        [((1, 14), torch.int64)],
        {"model_names": ["pd_bert_bert_base_japanese_qa_padlenlp"], "pcc": 0.99, "max_int": 31999},
    ),
    (
        Embedding7,
        [((1, 14), torch.int64)],
        {"model_names": ["pd_bert_bert_base_japanese_qa_padlenlp"], "pcc": 0.99, "max_int": 511},
    ),
    (
        Embedding3,
        [((1, 14), torch.int64)],
        {"model_names": ["pd_bert_bert_base_japanese_qa_padlenlp"], "pcc": 0.99, "max_int": 1},
    ),
    (
        Embedding2,
        [((1, 10), torch.int64)],
        {"model_names": ["pd_bert_bert_base_japanese_mlm_padlenlp"], "pcc": 0.99, "max_int": 31999},
    ),
    (
        Embedding7,
        [((1, 10), torch.int64)],
        {"model_names": ["pd_bert_bert_base_japanese_mlm_padlenlp"], "pcc": 0.99, "max_int": 511},
    ),
    (
        Embedding3,
        [((1, 10), torch.int64)],
        {"model_names": ["pd_bert_bert_base_japanese_mlm_padlenlp"], "pcc": 0.99, "max_int": 1},
    ),
    (
        Embedding12,
        [((1, 9), torch.int64)],
        {
            "model_names": ["pd_ernie_1_0_qa_padlenlp", "pd_ernie_1_0_seq_cls_padlenlp", "pd_ernie_1_0_mlm_padlenlp"],
            "pcc": 0.99,
            "max_int": 17999,
        },
    ),
    (
        Embedding13,
        [((1, 9), torch.int64)],
        {
            "model_names": ["pd_ernie_1_0_qa_padlenlp", "pd_ernie_1_0_seq_cls_padlenlp", "pd_ernie_1_0_mlm_padlenlp"],
            "pcc": 0.99,
            "max_int": 512,
        },
    ),
    (
        Embedding12,
        [((1, 12), torch.int64)],
        {"model_names": ["ErnieEmbeddings", "Ernie"], "pcc": 0.99, "max_int": 17999},
    ),
    (
        Embedding1,
        [((1, 12), torch.int64), ((513, 768), torch.float32)],
        {"model_names": ["ErnieEmbeddings", "Ernie"], "pcc": 0.99, "max_int": 512},
    ),
    (
        Embedding1,
        [((1, 12), torch.int64), ((2, 768), torch.float32)],
        {"model_names": ["ErnieEmbeddings", "Ernie"], "pcc": 0.99, "max_int": 1},
    ),
    (Embedding12, [((1, 12), torch.int32)], {"model_names": ["ErnieModel"], "pcc": 0.99, "max_int": 17999}),
    (Embedding13, [((1, 12), torch.int32)], {"model_names": ["ErnieModel"], "pcc": 0.99, "max_int": 512}),
    (Embedding3, [((1, 12), torch.int32)], {"model_names": ["ErnieModel"], "pcc": 0.99, "max_int": 1}),
    (
        Embedding14,
        [((1, 11), torch.int64)],
        {"model_names": ["pd_roberta_rbt4_ch_clm_padlenlp"], "pcc": 0.99, "max_int": 21127},
    ),
    (
        Embedding1,
        [((1, 11), torch.int64), ((2, 768), torch.float32)],
        {"model_names": ["pd_roberta_rbt4_ch_clm_padlenlp"], "pcc": 0.99, "max_int": 1},
    ),
    (
        Embedding14,
        [((1, 9), torch.int64)],
        {"model_names": ["pd_roberta_rbt4_ch_seq_cls_padlenlp"], "pcc": 0.99, "max_int": 21127},
    ),
    (
        Embedding1,
        [((1, 9), torch.int64), ((2, 768), torch.float32)],
        {"model_names": ["pd_roberta_rbt4_ch_seq_cls_padlenlp"], "pcc": 0.99, "max_int": 1},
    ),
    (
        Embedding15,
        [((2, 1), torch.int64)],
        {"model_names": ["pt_stereo_facebook_musicgen_large_music_generation_hf"], "pcc": 0.99, "max_int": 2048},
    ),
    (
        Embedding16,
        [((2, 13), torch.int64)],
        {
            "model_names": [
                "pt_stereo_facebook_musicgen_large_music_generation_hf",
                "pt_stereo_facebook_musicgen_medium_music_generation_hf",
            ],
            "pcc": 0.99,
            "max_int": 32127,
        },
    ),
    pytest.param(
        (
            Embedding1,
            [((13, 13), torch.int32), ((32, 12), torch.float32)],
            {
                "model_names": [
                    "pt_stereo_facebook_musicgen_large_music_generation_hf",
                    "pt_stereo_facebook_musicgen_medium_music_generation_hf",
                ],
                "pcc": 0.99,
                "max_int": 31,
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Embedding17,
        [((2, 1), torch.int64)],
        {"model_names": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"], "pcc": 0.99, "max_int": 2048},
    ),
    (
        Embedding18,
        [((1, 1), torch.int64)],
        {"model_names": ["pt_whisper_openai_whisper_base_speech_recognition_hf"], "pcc": 0.99, "max_int": 51864},
    ),
    (
        Embedding19,
        [((1, 1), torch.int64)],
        {"model_names": ["pt_whisper_openai_whisper_large_speech_recognition_hf"], "pcc": 0.99, "max_int": 51864},
    ),
    (
        Embedding20,
        [((1, 1), torch.int64)],
        {"model_names": ["pt_whisper_openai_whisper_small_speech_recognition_hf"], "pcc": 0.99, "max_int": 51864},
    ),
    (
        Embedding21,
        [((1, 1), torch.int64)],
        {"model_names": ["pt_whisper_openai_whisper_medium_speech_recognition_hf"], "pcc": 0.99, "max_int": 51864},
    ),
    (
        Embedding22,
        [((1, 1), torch.int64)],
        {"model_names": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"], "pcc": 0.99, "max_int": 51864},
    ),
    (
        Embedding23,
        [((1, 2), torch.int64)],
        {
            "model_names": ["pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf"],
            "pcc": 0.99,
            "max_int": 51865,
        },
    ),
    (
        Embedding24,
        [((2, 7), torch.int64)],
        {"model_names": ["pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text"], "pcc": 0.99, "max_int": 49407},
    ),
    (
        Embedding25,
        [((1, 7), torch.int64)],
        {"model_names": ["pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text"], "pcc": 0.99, "max_int": 76},
    ),
    (
        Embedding26,
        [((1, 588), torch.int64)],
        {"model_names": ["pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf"], "pcc": 0.99, "max_int": 32255},
    ),
    (
        Embedding27,
        [((1, 39), torch.int64)],
        {"model_names": ["pt_deepseek_deepseek_math_7b_instruct_qa_hf"], "pcc": 0.99, "max_int": 102399},
    ),
    (
        Embedding1,
        [((1, 577), torch.int64), ((577, 1024), torch.float32)],
        {"model_names": ["pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf"], "pcc": 0.99, "max_int": 576},
    ),
    (
        Embedding28,
        [((1, 596), torch.int64)],
        {"model_names": ["pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf"], "pcc": 0.99, "max_int": 32063},
    ),
    (
        Embedding29,
        [((1, 128), torch.int64)],
        {
            "model_names": [
                "pt_albert_base_v1_token_cls_hf",
                "pt_albert_large_v2_mlm_hf",
                "pt_albert_xlarge_v2_token_cls_hf",
                "pt_albert_large_v2_token_cls_hf",
                "pt_albert_large_v1_token_cls_hf",
                "pt_albert_base_v2_mlm_hf",
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_albert_xxlarge_v2_token_cls_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
                "pt_albert_xxlarge_v1_mlm_hf",
                "pt_albert_xlarge_v2_mlm_hf",
                "pt_albert_xlarge_v1_token_cls_hf",
                "pt_albert_large_v1_mlm_hf",
                "pt_albert_base_v2_token_cls_hf",
                "pt_albert_xlarge_v1_mlm_hf",
                "pt_albert_base_v1_mlm_hf",
            ],
            "pcc": 0.99,
            "max_int": 29999,
        },
    ),
    (
        Embedding10,
        [((1, 128), torch.int64)],
        {
            "model_names": [
                "pt_albert_base_v1_token_cls_hf",
                "pt_albert_large_v2_mlm_hf",
                "pt_albert_xlarge_v2_token_cls_hf",
                "pt_albert_large_v2_token_cls_hf",
                "pt_albert_large_v1_token_cls_hf",
                "pt_albert_base_v2_mlm_hf",
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_albert_xxlarge_v2_token_cls_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
                "pt_albert_xxlarge_v1_mlm_hf",
                "pt_albert_xlarge_v2_mlm_hf",
                "pt_albert_xlarge_v1_token_cls_hf",
                "pt_albert_large_v1_mlm_hf",
                "pt_albert_base_v2_token_cls_hf",
                "pt_albert_xlarge_v1_mlm_hf",
                "pt_albert_base_v1_mlm_hf",
            ],
            "pcc": 0.99,
            "max_int": 1,
        },
    ),
    (
        Embedding11,
        [((1, 128), torch.int64)],
        {
            "model_names": [
                "pt_albert_base_v1_token_cls_hf",
                "pt_albert_large_v2_mlm_hf",
                "pt_albert_xlarge_v2_token_cls_hf",
                "pt_albert_large_v2_token_cls_hf",
                "pt_albert_large_v1_token_cls_hf",
                "pt_albert_base_v2_mlm_hf",
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_albert_xxlarge_v2_token_cls_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
                "pt_albert_xxlarge_v1_mlm_hf",
                "pt_albert_xlarge_v2_mlm_hf",
                "pt_albert_xlarge_v1_token_cls_hf",
                "pt_albert_large_v1_mlm_hf",
                "pt_albert_base_v2_token_cls_hf",
                "pt_albert_xlarge_v1_mlm_hf",
                "pt_albert_base_v1_mlm_hf",
            ],
            "pcc": 0.99,
            "max_int": 511,
        },
    ),
    (
        Embedding29,
        [((1, 9), torch.int64)],
        {"model_names": ["pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf"], "pcc": 0.99, "max_int": 29999},
    ),
    (
        Embedding10,
        [((1, 9), torch.int64)],
        {"model_names": ["pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf"], "pcc": 0.99, "max_int": 1},
    ),
    (
        Embedding11,
        [((1, 9), torch.int64)],
        {"model_names": ["pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf"], "pcc": 0.99, "max_int": 511},
    ),
    (
        Embedding29,
        [((1, 14), torch.int64)],
        {"model_names": ["pt_albert_twmkn9_albert_base_v2_squad2_qa_hf"], "pcc": 0.99, "max_int": 29999},
    ),
    (
        Embedding10,
        [((1, 14), torch.int64)],
        {"model_names": ["pt_albert_twmkn9_albert_base_v2_squad2_qa_hf"], "pcc": 0.99, "max_int": 1},
    ),
    (
        Embedding11,
        [((1, 14), torch.int64)],
        {"model_names": ["pt_albert_twmkn9_albert_base_v2_squad2_qa_hf"], "pcc": 0.99, "max_int": 511},
    ),
    (
        Embedding30,
        [((1, 256), torch.int64)],
        {"model_names": ["pt_bart_facebook_bart_large_mnli_seq_cls_hf"], "pcc": 0.99, "max_int": 50264},
    ),
    (
        Embedding1,
        [((1, 256), torch.int64), ((1026, 1024), torch.float32)],
        {"model_names": ["pt_bart_facebook_bart_large_mnli_seq_cls_hf"], "pcc": 0.99, "max_int": 1025},
    ),
    (
        Embedding3,
        [((1, 128), torch.int64)],
        {
            "model_names": [
                "pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf",
                "pt_bert_bert_base_uncased_mlm_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder",
            ],
            "pcc": 0.99,
            "max_int": 1,
        },
    ),
    (
        Embedding7,
        [((1, 128), torch.int64)],
        {
            "model_names": [
                "pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf",
                "pt_bert_bert_base_uncased_mlm_hf",
                "pt_distilbert_distilbert_base_uncased_mlm_hf",
                "pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf",
                "pt_distilbert_distilbert_base_multilingual_cased_mlm_hf",
                "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder",
                "pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf",
            ],
            "pcc": 0.99,
            "max_int": 511,
        },
    ),
    (
        Embedding31,
        [((1, 384), torch.int64)],
        {
            "model_names": ["pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf"],
            "pcc": 0.99,
            "max_int": 28995,
        },
    ),
    (
        Embedding32,
        [((1, 384), torch.int64)],
        {
            "model_names": [
                "pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf",
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
            ],
            "pcc": 0.99,
            "max_int": 1,
        },
    ),
    (
        Embedding33,
        [((1, 384), torch.int64)],
        {
            "model_names": [
                "pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf",
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
            ],
            "pcc": 0.99,
            "max_int": 511,
        },
    ),
    (
        Embedding7,
        [((1, 6), torch.int64)],
        {
            "model_names": ["pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf"],
            "pcc": 0.99,
            "max_int": 511,
        },
    ),
    (
        Embedding31,
        [((1, 128), torch.int64)],
        {
            "model_names": ["pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf"],
            "pcc": 0.99,
            "max_int": 28995,
        },
    ),
    (
        Embedding32,
        [((1, 128), torch.int64)],
        {
            "model_names": ["pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf"],
            "pcc": 0.99,
            "max_int": 1,
        },
    ),
    (
        Embedding33,
        [((1, 128), torch.int64)],
        {
            "model_names": ["pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf"],
            "pcc": 0.99,
            "max_int": 511,
        },
    ),
    (
        Embedding34,
        [((1, 32), torch.int64)],
        {"model_names": ["pt_bloom_bigscience_bloom_1b1_clm_hf"], "pcc": 0.99, "max_int": 250879},
    ),
    (
        Embedding35,
        [((1, 256), torch.int64)],
        {
            "model_names": [
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
                "pt_codegen_salesforce_codegen_350m_mono_clm_hf",
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
            ],
            "pcc": 0.99,
            "max_int": 51199,
        },
    ),
    (
        Embedding36,
        [((1, 128), torch.int64)],
        {
            "model_names": [
                "pt_distilbert_distilbert_base_multilingual_cased_mlm_hf",
                "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
            ],
            "pcc": 0.99,
            "max_int": 119546,
        },
    ),
    (
        Embedding37,
        [((1, 384), torch.int64)],
        {"model_names": ["pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf"], "pcc": 0.99, "max_int": 28995},
    ),
    (
        Embedding7,
        [((1, 384), torch.int64)],
        {"model_names": ["pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf"], "pcc": 0.99, "max_int": 511},
    ),
    (
        Embedding38,
        [((1, 522), torch.int64)],
        {
            "model_names": ["pt_falcon3_tiiuae_falcon3_7b_base_clm_hf", "pt_falcon3_tiiuae_falcon3_3b_base_clm_hf"],
            "pcc": 0.99,
            "max_int": 131071,
        },
    ),
    (
        Embedding39,
        [((1, 522), torch.int64)],
        {"model_names": ["pt_falcon3_tiiuae_falcon3_1b_base_clm_hf"], "pcc": 0.99, "max_int": 131071},
    ),
    (
        Embedding40,
        [((1, 6), torch.int64)],
        {"model_names": ["pt_falcon_tiiuae_falcon_7b_instruct_clm_hf"], "pcc": 0.99, "max_int": 65023},
    ),
    (
        Embedding41,
        [((1, 207), torch.int64)],
        {"model_names": ["pt_gemma_google_gemma_2_2b_it_qa_hf"], "pcc": 0.99, "max_int": 255999},
    ),
    (
        Embedding42,
        [((1, 207), torch.int64)],
        {"model_names": ["pt_gemma_google_gemma_2_9b_it_qa_hf"], "pcc": 0.99, "max_int": 255999},
    ),
    (
        Embedding43,
        [((1, 7), torch.int64)],
        {"model_names": ["pt_gemma_google_gemma_2b_text_gen_hf"], "pcc": 0.99, "max_int": 255999},
    ),
    (
        Embedding44,
        [((1, 107), torch.int64)],
        {"model_names": ["pt_gemma_google_gemma_1_1_7b_it_qa_hf"], "pcc": 0.99, "max_int": 255999},
    ),
    (
        Embedding43,
        [((1, 107), torch.int64)],
        {"model_names": ["pt_gemma_google_gemma_1_1_2b_it_qa_hf"], "pcc": 0.99, "max_int": 255999},
    ),
    (
        Embedding45,
        [((1, 256), torch.int64)],
        {
            "model_names": ["pt_gpt2_gpt2_text_gen_hf", "pt_gptneo_eleutherai_gpt_neo_125m_clm_hf"],
            "pcc": 0.99,
            "max_int": 50256,
        },
    ),
    (
        Embedding1,
        [((1, 256), torch.int64), ((1024, 768), torch.float32)],
        {"model_names": ["pt_gpt2_gpt2_text_gen_hf"], "pcc": 0.99, "max_int": 1023},
    ),
    (
        Embedding45,
        [((1, 7), torch.int64)],
        {
            "model_names": [
                "pt_gpt2_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
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
                "pt_gpt2_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
            ],
            "pcc": 0.99,
            "max_int": 1023,
        },
    ),
    (
        Embedding46,
        [((1, 32), torch.int64)],
        {"model_names": ["pt_gptneo_eleutherai_gpt_neo_2_7b_seq_cls_hf"], "pcc": 0.99, "max_int": 50256},
    ),
    (
        Embedding1,
        [((1, 32), torch.int64), ((2048, 2560), torch.float32)],
        {"model_names": ["pt_gptneo_eleutherai_gpt_neo_2_7b_seq_cls_hf"], "pcc": 0.99, "max_int": 2047},
    ),
    (
        Embedding1,
        [((1, 256), torch.int64), ((2048, 768), torch.float32)],
        {"model_names": ["pt_gptneo_eleutherai_gpt_neo_125m_clm_hf"], "pcc": 0.99, "max_int": 2047},
    ),
    (
        Embedding47,
        [((1, 32), torch.int64)],
        {"model_names": ["pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf"], "pcc": 0.99, "max_int": 50256},
    ),
    (
        Embedding1,
        [((1, 32), torch.int64), ((2048, 2048), torch.float32)],
        {"model_names": ["pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf"], "pcc": 0.99, "max_int": 2047},
    ),
    (
        Embedding45,
        [((1, 32), torch.int64)],
        {"model_names": ["pt_gptneo_eleutherai_gpt_neo_125m_seq_cls_hf"], "pcc": 0.99, "max_int": 50256},
    ),
    (
        Embedding1,
        [((1, 32), torch.int64), ((2048, 768), torch.float32)],
        {"model_names": ["pt_gptneo_eleutherai_gpt_neo_125m_seq_cls_hf"], "pcc": 0.99, "max_int": 2047},
    ),
    (
        Embedding47,
        [((1, 256), torch.int64)],
        {"model_names": ["pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf"], "pcc": 0.99, "max_int": 50256},
    ),
    (
        Embedding1,
        [((1, 256), torch.int64), ((2048, 2048), torch.float32)],
        {"model_names": ["pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf"], "pcc": 0.99, "max_int": 2047},
    ),
    (
        Embedding46,
        [((1, 256), torch.int64)],
        {"model_names": ["pt_gptneo_eleutherai_gpt_neo_2_7b_clm_hf"], "pcc": 0.99, "max_int": 50256},
    ),
    (
        Embedding1,
        [((1, 256), torch.int64), ((2048, 2560), torch.float32)],
        {"model_names": ["pt_gptneo_eleutherai_gpt_neo_2_7b_clm_hf"], "pcc": 0.99, "max_int": 2047},
    ),
    (
        Embedding48,
        [((1, 32), torch.int64)],
        {"model_names": ["pt_llama3_huggyllama_llama_7b_clm_hf"], "pcc": 0.99, "max_int": 31999},
    ),
    (
        Embedding49,
        [((1, 4), torch.int64)],
        {
            "model_names": [
                "pt_llama3_meta_llama_llama_3_2_3b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_3b_instruct_seq_cls_hf",
            ],
            "pcc": 0.99,
            "max_int": 128255,
        },
    ),
    (
        Embedding48,
        [((1, 4), torch.int64)],
        {"model_names": ["pt_llama3_huggyllama_llama_7b_seq_cls_hf"], "pcc": 0.99, "max_int": 31999},
    ),
    (
        Embedding50,
        [((1, 4), torch.int64)],
        {
            "model_names": [
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_seq_cls_hf",
            ],
            "pcc": 0.99,
            "max_int": 128255,
        },
    ),
    (
        Embedding51,
        [((1, 256), torch.int64)],
        {
            "model_names": [
                "pt_llama3_meta_llama_meta_llama_3_8b_clm_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_clm_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_clm_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_clm_hf",
            ],
            "pcc": 0.99,
            "max_int": 128255,
        },
    ),
    (
        Embedding51,
        [((1, 4), torch.int64)],
        {
            "model_names": [
                "pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf",
            ],
            "pcc": 0.99,
            "max_int": 128255,
        },
    ),
    (
        Embedding49,
        [((1, 256), torch.int64)],
        {"model_names": ["pt_llama3_meta_llama_llama_3_2_3b_instruct_clm_hf"], "pcc": 0.99, "max_int": 128255},
    ),
    (
        Embedding50,
        [((1, 256), torch.int64)],
        {
            "model_names": [
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_clm_hf",
            ],
            "pcc": 0.99,
            "max_int": 128255,
        },
    ),
    (
        Embedding49,
        [((1, 32), torch.int64)],
        {"model_names": ["pt_llama3_meta_llama_llama_3_2_3b_clm_hf"], "pcc": 0.99, "max_int": 128255},
    ),
    (
        Embedding52,
        [((1, 8), torch.int64)],
        {"model_names": ["pt_ministral_mistralai_ministral_8b_instruct_2410_clm_hf"], "pcc": 0.99, "max_int": 131071},
    ),
    (
        Embedding53,
        [((1, 135), torch.int64)],
        {"model_names": ["pt_mistral_mistralai_mistral_7b_instruct_v0_3_clm_hf"], "pcc": 0.99, "max_int": 32767},
    ),
    (
        Embedding48,
        [((1, 128), torch.int64)],
        {"model_names": ["pt_mistral_mistralai_mistral_7b_v0_1_clm_hf"], "pcc": 0.99, "max_int": 31999},
    ),
    (
        Embedding54,
        [((1, 32), torch.int64)],
        {
            "model_names": ["pt_opt_facebook_opt_125m_qa_hf", "pt_opt_facebook_opt_125m_seq_cls_hf"],
            "pcc": 0.99,
            "max_int": 50271,
        },
    ),
    (
        Embedding55,
        [((1, 32), torch.int64)],
        {
            "model_names": ["pt_opt_facebook_opt_125m_qa_hf", "pt_opt_facebook_opt_125m_seq_cls_hf"],
            "pcc": 0.99,
            "max_int": 2049,
        },
    ),
    (
        Embedding56,
        [((1, 32), torch.int64)],
        {
            "model_names": ["pt_opt_facebook_opt_350m_seq_cls_hf", "pt_opt_facebook_opt_350m_qa_hf"],
            "pcc": 0.99,
            "max_int": 50271,
        },
    ),
    (
        Embedding57,
        [((1, 32), torch.int64)],
        {
            "model_names": ["pt_opt_facebook_opt_350m_seq_cls_hf", "pt_opt_facebook_opt_350m_qa_hf"],
            "pcc": 0.99,
            "max_int": 2049,
        },
    ),
    (
        Embedding58,
        [((1, 32), torch.int64)],
        {
            "model_names": ["pt_opt_facebook_opt_1_3b_seq_cls_hf", "pt_opt_facebook_opt_1_3b_qa_hf"],
            "pcc": 0.99,
            "max_int": 50271,
        },
    ),
    (
        Embedding59,
        [((1, 32), torch.int64)],
        {
            "model_names": ["pt_opt_facebook_opt_1_3b_seq_cls_hf", "pt_opt_facebook_opt_1_3b_qa_hf"],
            "pcc": 0.99,
            "max_int": 2049,
        },
    ),
    (
        Embedding56,
        [((1, 256), torch.int64)],
        {"model_names": ["pt_opt_facebook_opt_350m_clm_hf"], "pcc": 0.99, "max_int": 50271},
    ),
    (
        Embedding57,
        [((1, 256), torch.int64)],
        {"model_names": ["pt_opt_facebook_opt_350m_clm_hf"], "pcc": 0.99, "max_int": 2049},
    ),
    (
        Embedding54,
        [((1, 256), torch.int64)],
        {"model_names": ["pt_opt_facebook_opt_125m_clm_hf"], "pcc": 0.99, "max_int": 50271},
    ),
    (
        Embedding55,
        [((1, 256), torch.int64)],
        {"model_names": ["pt_opt_facebook_opt_125m_clm_hf"], "pcc": 0.99, "max_int": 2049},
    ),
    (
        Embedding58,
        [((1, 256), torch.int64)],
        {"model_names": ["pt_opt_facebook_opt_1_3b_clm_hf"], "pcc": 0.99, "max_int": 50271},
    ),
    (
        Embedding59,
        [((1, 256), torch.int64)],
        {"model_names": ["pt_opt_facebook_opt_1_3b_clm_hf"], "pcc": 0.99, "max_int": 2049},
    ),
    (
        Embedding60,
        [((1, 2048), torch.int64)],
        {"model_names": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"], "pcc": 0.99, "max_int": 261},
    ),
    (
        Embedding1,
        [((2048,), torch.int64), ((2048, 768), torch.float32)],
        {"model_names": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"], "pcc": 0.99, "max_int": 2047},
    ),
    (
        Embedding61,
        [((1, 256), torch.int64)],
        {
            "model_names": ["pt_phi1_5_microsoft_phi_1_5_seq_cls_hf", "pt_phi1_microsoft_phi_1_seq_cls_hf"],
            "pcc": 0.99,
            "max_int": 51199,
        },
    ),
    (
        Embedding61,
        [((1, 7), torch.int64)],
        {
            "model_names": ["pt_phi1_5_microsoft_phi_1_5_clm_hf", "pt_phi1_microsoft_phi_1_clm_hf"],
            "pcc": 0.99,
            "max_int": 51199,
        },
    ),
    (
        Embedding61,
        [((1, 12), torch.int64)],
        {
            "model_names": ["pt_phi1_5_microsoft_phi_1_5_token_cls_hf", "pt_phi1_microsoft_phi_1_token_cls_hf"],
            "pcc": 0.99,
            "max_int": 51199,
        },
    ),
    (
        Embedding62,
        [((1, 11), torch.int64)],
        {
            "model_names": ["pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf", "pt_phi2_microsoft_phi_2_seq_cls_hf"],
            "pcc": 0.99,
            "max_int": 51199,
        },
    ),
    (
        Embedding62,
        [((1, 12), torch.int64)],
        {
            "model_names": ["pt_phi2_microsoft_phi_2_pytdml_token_cls_hf", "pt_phi2_microsoft_phi_2_token_cls_hf"],
            "pcc": 0.99,
            "max_int": 51199,
        },
    ),
    (
        Embedding62,
        [((1, 256), torch.int64)],
        {
            "model_names": ["pt_phi2_microsoft_phi_2_clm_hf", "pt_phi2_microsoft_phi_2_pytdml_clm_hf"],
            "pcc": 0.99,
            "max_int": 51199,
        },
    ),
    (
        Embedding63,
        [((1, 256), torch.int64)],
        {
            "model_names": [
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf",
                "pt_phi3_5_microsoft_phi_3_5_mini_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "max_int": 32063,
        },
    ),
    (
        Embedding63,
        [((1, 5), torch.int64)],
        {"model_names": ["pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf"], "pcc": 0.99, "max_int": 32063},
    ),
    (
        Embedding63,
        [((1, 13), torch.int64)],
        {"model_names": ["pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf"], "pcc": 0.99, "max_int": 32063},
    ),
    (
        Embedding64,
        [((1, 256), torch.int64)],
        {"model_names": ["pt_phi4_microsoft_phi_4_seq_cls_hf"], "pcc": 0.99, "max_int": 100351},
    ),
    (
        Embedding64,
        [((1, 12), torch.int64)],
        {"model_names": ["pt_phi4_microsoft_phi_4_token_cls_hf"], "pcc": 0.99, "max_int": 100351},
    ),
    (
        Embedding64,
        [((1, 6), torch.int64)],
        {"model_names": ["pt_phi4_microsoft_phi_4_clm_hf"], "pcc": 0.99, "max_int": 100351},
    ),
    (
        Embedding65,
        [((1, 29), torch.int64)],
        {"model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_chat_clm_hf"], "pcc": 0.99, "max_int": 151935},
    ),
    (
        Embedding65,
        [((1, 6), torch.int64)],
        {"model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"], "pcc": 0.99, "max_int": 151935},
    ),
    (
        Embedding66,
        [((1, 35), torch.int64)],
        {
            "model_names": [
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_instruct_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_clm_hf",
            ],
            "pcc": 0.99,
            "max_int": 152063,
        },
    ),
    (
        Embedding67,
        [((1, 35), torch.int64)],
        {
            "model_names": [
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf",
            ],
            "pcc": 0.99,
            "max_int": 151935,
        },
    ),
    (
        Embedding68,
        [((1, 35), torch.int64)],
        {"model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf"], "pcc": 0.99, "max_int": 151935},
    ),
    (
        Embedding69,
        [((1, 35), torch.int64)],
        {
            "model_names": [
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "max_int": 151935,
        },
    ),
    (
        Embedding66,
        [((1, 13), torch.int64)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_7b_token_cls_hf"], "pcc": 0.99, "max_int": 152063},
    ),
    (
        Embedding69,
        [((1, 39), torch.int64)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm_hf"], "pcc": 0.99, "max_int": 151935},
    ),
    (
        Embedding69,
        [((1, 29), torch.int64)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_3b_clm_hf"], "pcc": 0.99, "max_int": 151935},
    ),
    (
        Embedding66,
        [((1, 29), torch.int64)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_7b_clm_hf"], "pcc": 0.99, "max_int": 152063},
    ),
    (
        Embedding67,
        [((1, 39), torch.int64)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf"], "pcc": 0.99, "max_int": 151935},
    ),
    (
        Embedding68,
        [((1, 39), torch.int64)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf"], "pcc": 0.99, "max_int": 151935},
    ),
    (
        Embedding68,
        [((1, 29), torch.int64)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"], "pcc": 0.99, "max_int": 151935},
    ),
    (
        Embedding67,
        [((1, 29), torch.int64)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99, "max_int": 151935},
    ),
    (
        Embedding66,
        [((1, 39), torch.int64)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf"], "pcc": 0.99, "max_int": 152063},
    ),
    (
        Embedding70,
        [((1, 128), torch.int64)],
        {"model_names": ["pt_roberta_xlm_roberta_base_mlm_hf"], "pcc": 0.99, "max_int": 250001},
    ),
    (
        Embedding71,
        [((1, 128), torch.int64)],
        {
            "model_names": [
                "pt_roberta_xlm_roberta_base_mlm_hf",
                "pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf",
            ],
            "pcc": 0.99,
            "max_int": 0,
        },
    ),
    (
        Embedding72,
        [((1, 128), torch.int64)],
        {
            "model_names": [
                "pt_roberta_xlm_roberta_base_mlm_hf",
                "pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf",
            ],
            "pcc": 0.99,
            "max_int": 513,
        },
    ),
    (
        Embedding73,
        [((1, 128), torch.int64)],
        {
            "model_names": ["pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf"],
            "pcc": 0.99,
            "max_int": 50264,
        },
    ),
    (
        Embedding74,
        [((1, 128), torch.int64)],
        {"model_names": ["pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf"], "pcc": 0.99, "max_int": 30527},
    ),
    (
        Embedding75,
        [((1, 1), torch.int64)],
        {
            "model_names": ["pt_t5_google_flan_t5_large_text_gen_hf", "pt_t5_t5_large_text_gen_hf"],
            "pcc": 0.99,
            "max_int": 32127,
        },
    ),
    (
        Embedding1,
        [((1, 1), torch.int32), ((32, 16), torch.float32)],
        {
            "model_names": ["pt_t5_google_flan_t5_large_text_gen_hf", "pt_t5_t5_large_text_gen_hf"],
            "pcc": 0.99,
            "max_int": 31,
        },
    ),
    (
        Embedding75,
        [((1, 61), torch.int64)],
        {
            "model_names": ["pt_t5_google_flan_t5_large_text_gen_hf", "pt_t5_t5_large_text_gen_hf"],
            "pcc": 0.99,
            "max_int": 32127,
        },
    ),
    pytest.param(
        (
            Embedding1,
            [((61, 61), torch.int32), ((32, 16), torch.float32)],
            {
                "model_names": ["pt_t5_google_flan_t5_large_text_gen_hf", "pt_t5_t5_large_text_gen_hf"],
                "pcc": 0.99,
                "max_int": 31,
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Embedding76,
        [((1, 1), torch.int64)],
        {
            "model_names": ["pt_t5_t5_small_text_gen_hf", "pt_t5_google_flan_t5_small_text_gen_hf"],
            "pcc": 0.99,
            "max_int": 32127,
        },
    ),
    (
        Embedding1,
        [((1, 1), torch.int32), ((32, 8), torch.float32)],
        {"model_names": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99, "max_int": 31},
    ),
    (
        Embedding76,
        [((1, 61), torch.int64)],
        {
            "model_names": ["pt_t5_t5_small_text_gen_hf", "pt_t5_google_flan_t5_small_text_gen_hf"],
            "pcc": 0.99,
            "max_int": 32127,
        },
    ),
    pytest.param(
        (
            Embedding1,
            [((61, 61), torch.int32), ((32, 8), torch.float32)],
            {"model_names": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99, "max_int": 31},
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Embedding16,
        [((1, 1), torch.int64)],
        {
            "model_names": ["pt_t5_t5_base_text_gen_hf", "pt_t5_google_flan_t5_base_text_gen_hf"],
            "pcc": 0.99,
            "max_int": 32127,
        },
    ),
    (
        Embedding1,
        [((1, 1), torch.int32), ((32, 12), torch.float32)],
        {
            "model_names": ["pt_t5_t5_base_text_gen_hf", "pt_t5_google_flan_t5_base_text_gen_hf"],
            "pcc": 0.99,
            "max_int": 31,
        },
    ),
    (
        Embedding16,
        [((1, 61), torch.int64)],
        {
            "model_names": ["pt_t5_t5_base_text_gen_hf", "pt_t5_google_flan_t5_base_text_gen_hf"],
            "pcc": 0.99,
            "max_int": 32127,
        },
    ),
    pytest.param(
        (
            Embedding1,
            [((61, 61), torch.int32), ((32, 12), torch.float32)],
            {
                "model_names": ["pt_t5_t5_base_text_gen_hf", "pt_t5_google_flan_t5_base_text_gen_hf"],
                "pcc": 0.99,
                "max_int": 31,
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Embedding1,
        [((1, 1), torch.int32), ((32, 6), torch.float32)],
        {"model_names": ["pt_t5_google_flan_t5_small_text_gen_hf"], "pcc": 0.99, "max_int": 31},
    ),
    pytest.param(
        (
            Embedding1,
            [((61, 61), torch.int32), ((32, 6), torch.float32)],
            {"model_names": ["pt_t5_google_flan_t5_small_text_gen_hf"], "pcc": 0.99, "max_int": 31},
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Embedding77,
        [((1, 256), torch.int64)],
        {"model_names": ["pt_xglm_facebook_xglm_564m_clm_hf"], "pcc": 0.99, "max_int": 256007},
    ),
    (
        Embedding78,
        [((1, 256), torch.int64)],
        {"model_names": ["pt_xglm_facebook_xglm_1_7b_clm_hf"], "pcc": 0.99, "max_int": 256007},
    ),
]


@pytest.mark.nightly_models_ops
@pytest.mark.parametrize("forge_module_and_shapes_dtypes", forge_modules_and_shapes_dtypes_list, ids=ids_func)
def test_module(forge_module_and_shapes_dtypes, forge_property_recorder):

    forge_property_recorder.enable_single_op_details_recording()
    forge_property_recorder.record_forge_op_name("Embedding")

    forge_module, operand_shapes_dtypes, metadata = forge_module_and_shapes_dtypes

    pcc = metadata.pop("pcc")
    max_int = metadata.pop("max_int")

    for metadata_name, metadata_value in metadata.items():
        if metadata_name == "model_names":
            forge_property_recorder.record_op_model_names(metadata_value)
        elif metadata_name == "args":
            forge_property_recorder.record_forge_op_args(metadata_value)
        else:
            logger.warning(
                "No utility function available in forge property handler to record %s property", metadata_name
            )

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

    forge_property_recorder.record_single_op_operands_info(framework_model, inputs)

    compiled_model = compile(framework_model, sample_inputs=inputs, forge_property_handler=forge_property_recorder)

    verify(
        inputs,
        framework_model,
        compiled_model,
        VerifyConfig(value_checker=AutomaticValueChecker(pcc=pcc)),
        forge_property_handler=forge_property_recorder,
    )
