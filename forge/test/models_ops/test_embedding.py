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
            forge.Parameter(*(2049, 2048), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding0.weight_1"))
        return embedding_output_1


class Embedding1(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding1.weight_1",
            forge.Parameter(*(32128, 768), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding1.weight_1"))
        return embedding_output_1


class Embedding2(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, embedding_input_0, embedding_input_1):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, embedding_input_1)
        return embedding_output_1


class Embedding3(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding3.weight_1",
            forge.Parameter(*(2049, 1536), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding3.weight_1"))
        return embedding_output_1


class Embedding4(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding4.weight_1",
            forge.Parameter(*(2049, 1024), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding4.weight_1"))
        return embedding_output_1


class Embedding5(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding5.weight_1",
            forge.Parameter(*(51865, 1024), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding5.weight_1"))
        return embedding_output_1


class Embedding6(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding6.weight_1",
            forge.Parameter(*(51865, 1280), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding6.weight_1"))
        return embedding_output_1


class Embedding7(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding7.weight_1",
            forge.Parameter(*(51865, 384), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding7.weight_1"))
        return embedding_output_1


class Embedding8(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding8.weight_1",
            forge.Parameter(*(51865, 512), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding8.weight_1"))
        return embedding_output_1


class Embedding9(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding9.weight_1",
            forge.Parameter(*(51865, 768), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding9.weight_1"))
        return embedding_output_1


class Embedding10(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding10.weight_1",
            forge.Parameter(*(51866, 1280), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding10.weight_1"))
        return embedding_output_1


class Embedding11(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding11.weight_1",
            forge.Parameter(*(49408, 512), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding11.weight_1"))
        return embedding_output_1


class Embedding12(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding12.weight_1",
            forge.Parameter(*(77, 512), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding12.weight_1"))
        return embedding_output_1


class Embedding13(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding13.weight_1",
            forge.Parameter(*(102400, 4096), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding13.weight_1"))
        return embedding_output_1


class Embedding14(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding14.weight_1",
            forge.Parameter(*(30000, 128), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding14.weight_1"))
        return embedding_output_1


class Embedding15(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding15.weight_1",
            forge.Parameter(*(2, 128), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding15.weight_1"))
        return embedding_output_1


class Embedding16(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding16.weight_1",
            forge.Parameter(*(512, 128), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding16.weight_1"))
        return embedding_output_1


class Embedding17(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding17.weight_1",
            forge.Parameter(*(50265, 1024), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding17.weight_1"))
        return embedding_output_1


class Embedding18(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding18.weight_1",
            forge.Parameter(*(30522, 768), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding18.weight_1"))
        return embedding_output_1


class Embedding19(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding19.weight_1",
            forge.Parameter(*(2, 768), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding19.weight_1"))
        return embedding_output_1


class Embedding20(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding20.weight_1",
            forge.Parameter(*(512, 768), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding20.weight_1"))
        return embedding_output_1


class Embedding21(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding21.weight_1",
            forge.Parameter(*(28996, 1024), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding21.weight_1"))
        return embedding_output_1


class Embedding22(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding22.weight_1",
            forge.Parameter(*(2, 1024), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding22.weight_1"))
        return embedding_output_1


class Embedding23(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding23.weight_1",
            forge.Parameter(*(512, 1024), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding23.weight_1"))
        return embedding_output_1


class Embedding24(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding24.weight_1",
            forge.Parameter(*(51200, 1024), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding24.weight_1"))
        return embedding_output_1


class Embedding25(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding25.weight_1",
            forge.Parameter(*(119547, 768), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding25.weight_1"))
        return embedding_output_1


class Embedding26(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding26.weight_1",
            forge.Parameter(*(28996, 768), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding26.weight_1"))
        return embedding_output_1


class Embedding27(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding27.weight_1",
            forge.Parameter(*(65024, 4544), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding27.weight_1"))
        return embedding_output_1


class Embedding28(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding28.weight_1",
            forge.Parameter(*(131072, 3072), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding28.weight_1"))
        return embedding_output_1


class Embedding29(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding29.weight_1",
            forge.Parameter(*(131072, 2048), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding29.weight_1"))
        return embedding_output_1


class Embedding30(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding30.weight_1",
            forge.Parameter(*(256000, 2048), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding30.weight_1"))
        return embedding_output_1


class Embedding31(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding31.weight_1",
            forge.Parameter(*(50257, 768), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding31.weight_1"))
        return embedding_output_1


class Embedding32(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding32.weight_1",
            forge.Parameter(*(50257, 2560), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding32.weight_1"))
        return embedding_output_1


class Embedding33(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding33.weight_1",
            forge.Parameter(*(50257, 2048), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding33.weight_1"))
        return embedding_output_1


class Embedding34(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding34.weight_1",
            forge.Parameter(*(128256, 2048), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding34.weight_1"))
        return embedding_output_1


class Embedding35(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding35.weight_1",
            forge.Parameter(*(128256, 4096), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding35.weight_1"))
        return embedding_output_1


class Embedding36(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("embedding36_const_1", shape=(32000, 4096), dtype=torch.float32)

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_constant("embedding36_const_1"))
        return embedding_output_1


class Embedding37(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding37.weight_1",
            forge.Parameter(*(50272, 2048), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding37.weight_1"))
        return embedding_output_1


class Embedding38(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding38.weight_1",
            forge.Parameter(*(2050, 2048), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding38.weight_1"))
        return embedding_output_1


class Embedding39(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding39.weight_1",
            forge.Parameter(*(50272, 512), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding39.weight_1"))
        return embedding_output_1


class Embedding40(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding40.weight_1",
            forge.Parameter(*(2050, 1024), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding40.weight_1"))
        return embedding_output_1


class Embedding41(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding41.weight_1",
            forge.Parameter(*(50272, 768), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding41.weight_1"))
        return embedding_output_1


class Embedding42(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding42.weight_1",
            forge.Parameter(*(2050, 768), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding42.weight_1"))
        return embedding_output_1


class Embedding43(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding43.weight_1",
            forge.Parameter(*(51200, 2560), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding43.weight_1"))
        return embedding_output_1


class Embedding44(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding44.weight_1",
            forge.Parameter(*(32064, 3072), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding44.weight_1"))
        return embedding_output_1


class Embedding45(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding45.weight_1",
            forge.Parameter(*(151936, 1024), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding45.weight_1"))
        return embedding_output_1


class Embedding46(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding46.weight_1",
            forge.Parameter(*(152064, 3584), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding46.weight_1"))
        return embedding_output_1


class Embedding47(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding47.weight_1",
            forge.Parameter(*(151936, 1536), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding47.weight_1"))
        return embedding_output_1


class Embedding48(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding48.weight_1",
            forge.Parameter(*(151936, 2048), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding48.weight_1"))
        return embedding_output_1


class Embedding49(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding49.weight_1",
            forge.Parameter(*(151936, 896), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding49.weight_1"))
        return embedding_output_1


class Embedding50(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding50.weight_1",
            forge.Parameter(*(250002, 768), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding50.weight_1"))
        return embedding_output_1


class Embedding51(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding51.weight_1",
            forge.Parameter(*(1, 768), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding51.weight_1"))
        return embedding_output_1


class Embedding52(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding52.weight_1",
            forge.Parameter(*(514, 768), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding52.weight_1"))
        return embedding_output_1


class Embedding53(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding53.weight_1",
            forge.Parameter(*(50265, 768), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding53.weight_1"))
        return embedding_output_1


class Embedding54(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding54.weight_1",
            forge.Parameter(*(30528, 768), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding54.weight_1"))
        return embedding_output_1


class Embedding55(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding55.weight_1",
            forge.Parameter(*(32128, 1024), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding55.weight_1"))
        return embedding_output_1


class Embedding56(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding56.weight_1",
            forge.Parameter(*(32128, 512), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding56.weight_1"))
        return embedding_output_1


class Embedding57(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding57.weight_1",
            forge.Parameter(*(256008, 2048), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding57.weight_1"))
        return embedding_output_1


class Embedding58(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding58.weight_1",
            forge.Parameter(*(256008, 1024), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding58.weight_1"))
        return embedding_output_1


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (
        Embedding0,
        [((2, 1), torch.int64)],
        {"model_name": ["pt_stereo_facebook_musicgen_large_music_generation_hf"], "pcc": 0.99, "max_int": 2048},
    ),
    (
        Embedding1,
        [((2, 13), torch.int64)],
        {
            "model_name": [
                "pt_stereo_facebook_musicgen_large_music_generation_hf",
                "pt_stereo_facebook_musicgen_medium_music_generation_hf",
                "pt_stereo_facebook_musicgen_small_music_generation_hf",
            ],
            "pcc": 0.99,
            "max_int": 32127,
        },
    ),
    pytest.param(
        (
            Embedding2,
            [((13, 13), torch.int32), ((32, 12), torch.float32)],
            {
                "model_name": [
                    "pt_stereo_facebook_musicgen_large_music_generation_hf",
                    "pt_stereo_facebook_musicgen_medium_music_generation_hf",
                    "pt_stereo_facebook_musicgen_small_music_generation_hf",
                ],
                "pcc": 0.99,
                "max_int": 31,
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Embedding3,
        [((2, 1), torch.int64)],
        {"model_name": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"], "pcc": 0.99, "max_int": 2048},
    ),
    (
        Embedding4,
        [((2, 1), torch.int64)],
        {"model_name": ["pt_stereo_facebook_musicgen_small_music_generation_hf"], "pcc": 0.99, "max_int": 2048},
    ),
    (
        Embedding5,
        [((1, 1), torch.int64)],
        {"model_name": ["pt_whisper_openai_whisper_medium_speech_recognition_hf"], "pcc": 0.99, "max_int": 51864},
    ),
    (
        Embedding6,
        [((1, 1), torch.int64)],
        {"model_name": ["pt_whisper_openai_whisper_large_speech_recognition_hf"], "pcc": 0.99, "max_int": 51864},
    ),
    (
        Embedding7,
        [((1, 1), torch.int64)],
        {"model_name": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"], "pcc": 0.99, "max_int": 51864},
    ),
    (
        Embedding8,
        [((1, 1), torch.int64)],
        {"model_name": ["pt_whisper_openai_whisper_base_speech_recognition_hf"], "pcc": 0.99, "max_int": 51864},
    ),
    (
        Embedding9,
        [((1, 1), torch.int64)],
        {"model_name": ["pt_whisper_openai_whisper_small_speech_recognition_hf"], "pcc": 0.99, "max_int": 51864},
    ),
    (
        Embedding10,
        [((1, 2), torch.int32)],
        {"model_name": ["pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf"], "pcc": 0.99, "max_int": 51865},
    ),
    (
        Embedding11,
        [((2, 7), torch.int64)],
        {"model_name": ["pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text"], "pcc": 0.99, "max_int": 49407},
    ),
    (
        Embedding12,
        [((1, 7), torch.int64)],
        {"model_name": ["pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text"], "pcc": 0.99, "max_int": 76},
    ),
    (
        Embedding13,
        [((1, 39), torch.int64)],
        {"model_name": ["pt_deepseek_deepseek_math_7b_instruct_qa_hf"], "pcc": 0.99, "max_int": 102399},
    ),
    (
        Embedding13,
        [((1, 39), torch.int32)],
        {"model_name": ["DeepSeekWrapper_decoder"], "pcc": 0.99, "max_int": 102399},
    ),
    (
        Embedding14,
        [((1, 128), torch.int64)],
        {
            "model_name": [
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_albert_base_v2_token_cls_hf",
                "pt_albert_xxlarge_v2_token_cls_hf",
                "pt_albert_large_v1_token_cls_hf",
                "pt_albert_large_v2_token_cls_hf",
                "pt_albert_base_v1_token_cls_hf",
                "pt_albert_base_v1_mlm_hf",
                "pt_albert_xlarge_v2_token_cls_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
                "pt_albert_large_v2_mlm_hf",
                "pt_albert_base_v2_mlm_hf",
                "pt_albert_xlarge_v1_token_cls_hf",
                "pt_albert_xlarge_v1_mlm_hf",
                "pt_albert_large_v1_mlm_hf",
                "pt_albert_xxlarge_v1_mlm_hf",
                "pt_albert_xlarge_v2_mlm_hf",
            ],
            "pcc": 0.99,
            "max_int": 29999,
        },
    ),
    (
        Embedding15,
        [((1, 128), torch.int64)],
        {
            "model_name": [
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_albert_base_v2_token_cls_hf",
                "pt_albert_xxlarge_v2_token_cls_hf",
                "pt_albert_large_v1_token_cls_hf",
                "pt_albert_large_v2_token_cls_hf",
                "pt_albert_base_v1_token_cls_hf",
                "pt_albert_base_v1_mlm_hf",
                "pt_albert_xlarge_v2_token_cls_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
                "pt_albert_large_v2_mlm_hf",
                "pt_albert_base_v2_mlm_hf",
                "pt_albert_xlarge_v1_token_cls_hf",
                "pt_albert_xlarge_v1_mlm_hf",
                "pt_albert_large_v1_mlm_hf",
                "pt_albert_xxlarge_v1_mlm_hf",
                "pt_albert_xlarge_v2_mlm_hf",
            ],
            "pcc": 0.99,
            "max_int": 1,
        },
    ),
    (
        Embedding16,
        [((1, 128), torch.int64)],
        {
            "model_name": [
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_albert_base_v2_token_cls_hf",
                "pt_albert_xxlarge_v2_token_cls_hf",
                "pt_albert_large_v1_token_cls_hf",
                "pt_albert_large_v2_token_cls_hf",
                "pt_albert_base_v1_token_cls_hf",
                "pt_albert_base_v1_mlm_hf",
                "pt_albert_xlarge_v2_token_cls_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
                "pt_albert_large_v2_mlm_hf",
                "pt_albert_base_v2_mlm_hf",
                "pt_albert_xlarge_v1_token_cls_hf",
                "pt_albert_xlarge_v1_mlm_hf",
                "pt_albert_large_v1_mlm_hf",
                "pt_albert_xxlarge_v1_mlm_hf",
                "pt_albert_xlarge_v2_mlm_hf",
            ],
            "pcc": 0.99,
            "max_int": 511,
        },
    ),
    (
        Embedding17,
        [((1, 256), torch.int64)],
        {"model_name": ["pt_bart_facebook_bart_large_mnli_seq_cls_hf"], "pcc": 0.99, "max_int": 50264},
    ),
    (
        Embedding2,
        [((1, 256), torch.int64), ((1026, 1024), torch.float32)],
        {"model_name": ["pt_bart_facebook_bart_large_mnli_seq_cls_hf"], "pcc": 0.99, "max_int": 1025},
    ),
    (
        Embedding18,
        [((1, 128), torch.int64)],
        {
            "model_name": [
                "pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf",
                "pt_bert_bert_base_uncased_mlm_hf",
                "pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf",
                "pt_distilbert_distilbert_base_uncased_mlm_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader",
            ],
            "pcc": 0.99,
            "max_int": 30521,
        },
    ),
    (
        Embedding19,
        [((1, 128), torch.int64)],
        {
            "model_name": [
                "pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf",
                "pt_bert_bert_base_uncased_mlm_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader",
            ],
            "pcc": 0.99,
            "max_int": 1,
        },
    ),
    (
        Embedding20,
        [((1, 128), torch.int64)],
        {
            "model_name": [
                "pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf",
                "pt_bert_bert_base_uncased_mlm_hf",
                "pt_distilbert_distilbert_base_multilingual_cased_mlm_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
                "pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf",
                "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "pt_distilbert_distilbert_base_uncased_mlm_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader",
                "pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf",
            ],
            "pcc": 0.99,
            "max_int": 511,
        },
    ),
    (
        Embedding21,
        [((1, 384), torch.int64)],
        {
            "model_name": ["pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf"],
            "pcc": 0.99,
            "max_int": 28995,
        },
    ),
    (
        Embedding22,
        [((1, 384), torch.int64)],
        {
            "model_name": ["pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf"],
            "pcc": 0.99,
            "max_int": 1,
        },
    ),
    (
        Embedding23,
        [((1, 384), torch.int64)],
        {
            "model_name": ["pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf"],
            "pcc": 0.99,
            "max_int": 511,
        },
    ),
    (
        Embedding21,
        [((1, 128), torch.int64)],
        {
            "model_name": ["pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf"],
            "pcc": 0.99,
            "max_int": 28995,
        },
    ),
    (
        Embedding22,
        [((1, 128), torch.int64)],
        {
            "model_name": ["pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf"],
            "pcc": 0.99,
            "max_int": 1,
        },
    ),
    (
        Embedding23,
        [((1, 128), torch.int64)],
        {
            "model_name": ["pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf"],
            "pcc": 0.99,
            "max_int": 511,
        },
    ),
    (
        Embedding24,
        [((1, 256), torch.int32)],
        {
            "model_name": [
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_mono_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
            ],
            "pcc": 0.99,
            "max_int": 51199,
        },
    ),
    (
        Embedding25,
        [((1, 128), torch.int64)],
        {
            "model_name": [
                "pt_distilbert_distilbert_base_multilingual_cased_mlm_hf",
                "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
            ],
            "pcc": 0.99,
            "max_int": 119546,
        },
    ),
    (
        Embedding26,
        [((1, 384), torch.int64)],
        {"model_name": ["pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf"], "pcc": 0.99, "max_int": 28995},
    ),
    (
        Embedding20,
        [((1, 384), torch.int64)],
        {"model_name": ["pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf"], "pcc": 0.99, "max_int": 511},
    ),
    (
        Embedding26,
        [((1, 128), torch.int64)],
        {"model_name": ["pt_distilbert_distilbert_base_cased_mlm_hf"], "pcc": 0.99, "max_int": 28995},
    ),
    (
        Embedding27,
        [((1, 6), torch.int64)],
        {"model_name": ["pt_falcon_tiiuae_falcon_7b_instruct_clm_hf"], "pcc": 0.99, "max_int": 65023},
    ),
    (
        Embedding28,
        [((1, 10), torch.int64)],
        {
            "model_name": ["pt_falcon3_tiiuae_falcon3_3b_base_clm_hf", "pt_falcon3_tiiuae_falcon3_7b_base_clm_hf"],
            "pcc": 0.99,
            "max_int": 131071,
        },
    ),
    (
        Embedding29,
        [((1, 10), torch.int64)],
        {"model_name": ["pt_falcon3_tiiuae_falcon3_1b_base_clm_hf"], "pcc": 0.99, "max_int": 131071},
    ),
    (
        Embedding30,
        [((1, 7), torch.int64)],
        {"model_name": ["pt_gemma_google_gemma_2b_text_gen_hf"], "pcc": 0.99, "max_int": 255999},
    ),
    (
        Embedding31,
        [((1, 256), torch.int64)],
        {
            "model_name": ["pt_gpt2_gpt2_text_gen_hf", "pt_gptneo_eleutherai_gpt_neo_125m_clm_hf"],
            "pcc": 0.99,
            "max_int": 50256,
        },
    ),
    (
        Embedding2,
        [((1, 256), torch.int64), ((1024, 768), torch.float32)],
        {"model_name": ["pt_gpt2_gpt2_text_gen_hf"], "pcc": 0.99, "max_int": 1023},
    ),
    (
        Embedding32,
        [((1, 256), torch.int64)],
        {"model_name": ["pt_gptneo_eleutherai_gpt_neo_2_7b_clm_hf"], "pcc": 0.99, "max_int": 50256},
    ),
    (
        Embedding2,
        [((1, 256), torch.int64), ((2048, 2560), torch.float32)],
        {"model_name": ["pt_gptneo_eleutherai_gpt_neo_2_7b_clm_hf"], "pcc": 0.99, "max_int": 2047},
    ),
    (
        Embedding33,
        [((1, 256), torch.int64)],
        {"model_name": ["pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf"], "pcc": 0.99, "max_int": 50256},
    ),
    (
        Embedding2,
        [((1, 256), torch.int64), ((2048, 2048), torch.float32)],
        {"model_name": ["pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf"], "pcc": 0.99, "max_int": 2047},
    ),
    (
        Embedding31,
        [((1, 32), torch.int64)],
        {"model_name": ["pt_gptneo_eleutherai_gpt_neo_125m_seq_cls_hf"], "pcc": 0.99, "max_int": 50256},
    ),
    (
        Embedding2,
        [((1, 32), torch.int64), ((2048, 768), torch.float32)],
        {"model_name": ["pt_gptneo_eleutherai_gpt_neo_125m_seq_cls_hf"], "pcc": 0.99, "max_int": 2047},
    ),
    (
        Embedding32,
        [((1, 32), torch.int64)],
        {"model_name": ["pt_gptneo_eleutherai_gpt_neo_2_7b_seq_cls_hf"], "pcc": 0.99, "max_int": 50256},
    ),
    (
        Embedding2,
        [((1, 32), torch.int64), ((2048, 2560), torch.float32)],
        {"model_name": ["pt_gptneo_eleutherai_gpt_neo_2_7b_seq_cls_hf"], "pcc": 0.99, "max_int": 2047},
    ),
    (
        Embedding2,
        [((1, 256), torch.int64), ((2048, 768), torch.float32)],
        {"model_name": ["pt_gptneo_eleutherai_gpt_neo_125m_clm_hf"], "pcc": 0.99, "max_int": 2047},
    ),
    (
        Embedding33,
        [((1, 32), torch.int64)],
        {"model_name": ["pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf"], "pcc": 0.99, "max_int": 50256},
    ),
    (
        Embedding2,
        [((1, 32), torch.int64), ((2048, 2048), torch.float32)],
        {"model_name": ["pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf"], "pcc": 0.99, "max_int": 2047},
    ),
    (
        Embedding34,
        [((1, 4), torch.int64)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_seq_cls_hf",
            ],
            "pcc": 0.99,
            "max_int": 128255,
        },
    ),
    (
        Embedding34,
        [((1, 256), torch.int32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_clm_hf",
            ],
            "pcc": 0.99,
            "max_int": 128255,
        },
    ),
    (
        Embedding35,
        [((1, 4), torch.int64)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf",
            ],
            "pcc": 0.99,
            "max_int": 128255,
        },
    ),
    (
        Embedding36,
        [((1, 128), torch.int64)],
        {"model_name": ["pt_mistral_mistralai_mistral_7b_v0_1_clm_hf"], "pcc": 0.99, "max_int": 31999},
    ),
    (
        Embedding31,
        [((1, 7), torch.int64)],
        {"model_name": ["pt_nanogpt_financialsupport_nanogpt_text_gen_hf"], "pcc": 0.99, "max_int": 50256},
    ),
    (
        Embedding2,
        [((1, 7), torch.int64), ((1024, 768), torch.float32)],
        {"model_name": ["pt_nanogpt_financialsupport_nanogpt_text_gen_hf"], "pcc": 0.99, "max_int": 1023},
    ),
    (
        Embedding37,
        [((1, 256), torch.int64)],
        {"model_name": ["pt_opt_facebook_opt_1_3b_clm_hf"], "pcc": 0.99, "max_int": 50271},
    ),
    (
        Embedding38,
        [((1, 256), torch.int64)],
        {"model_name": ["pt_opt_facebook_opt_1_3b_clm_hf"], "pcc": 0.99, "max_int": 2049},
    ),
    (
        Embedding37,
        [((1, 32), torch.int64)],
        {
            "model_name": ["pt_opt_facebook_opt_1_3b_seq_cls_hf", "pt_opt_facebook_opt_1_3b_qa_hf"],
            "pcc": 0.99,
            "max_int": 50271,
        },
    ),
    (
        Embedding38,
        [((1, 32), torch.int64)],
        {
            "model_name": ["pt_opt_facebook_opt_1_3b_seq_cls_hf", "pt_opt_facebook_opt_1_3b_qa_hf"],
            "pcc": 0.99,
            "max_int": 2049,
        },
    ),
    (
        Embedding39,
        [((1, 32), torch.int64)],
        {
            "model_name": ["pt_opt_facebook_opt_350m_qa_hf", "pt_opt_facebook_opt_350m_seq_cls_hf"],
            "pcc": 0.99,
            "max_int": 50271,
        },
    ),
    (
        Embedding40,
        [((1, 32), torch.int64)],
        {
            "model_name": ["pt_opt_facebook_opt_350m_qa_hf", "pt_opt_facebook_opt_350m_seq_cls_hf"],
            "pcc": 0.99,
            "max_int": 2049,
        },
    ),
    (
        Embedding41,
        [((1, 32), torch.int64)],
        {
            "model_name": ["pt_opt_facebook_opt_125m_seq_cls_hf", "pt_opt_facebook_opt_125m_qa_hf"],
            "pcc": 0.99,
            "max_int": 50271,
        },
    ),
    (
        Embedding42,
        [((1, 32), torch.int64)],
        {
            "model_name": ["pt_opt_facebook_opt_125m_seq_cls_hf", "pt_opt_facebook_opt_125m_qa_hf"],
            "pcc": 0.99,
            "max_int": 2049,
        },
    ),
    (
        Embedding41,
        [((1, 256), torch.int64)],
        {"model_name": ["pt_opt_facebook_opt_125m_clm_hf"], "pcc": 0.99, "max_int": 50271},
    ),
    (
        Embedding42,
        [((1, 256), torch.int64)],
        {"model_name": ["pt_opt_facebook_opt_125m_clm_hf"], "pcc": 0.99, "max_int": 2049},
    ),
    (
        Embedding39,
        [((1, 256), torch.int64)],
        {"model_name": ["pt_opt_facebook_opt_350m_clm_hf"], "pcc": 0.99, "max_int": 50271},
    ),
    (
        Embedding40,
        [((1, 256), torch.int64)],
        {"model_name": ["pt_opt_facebook_opt_350m_clm_hf"], "pcc": 0.99, "max_int": 2049},
    ),
    (
        Embedding43,
        [((1, 12), torch.int64)],
        {
            "model_name": ["pt_phi2_microsoft_phi_2_pytdml_token_cls_hf", "pt_phi2_microsoft_phi_2_token_cls_hf"],
            "pcc": 0.99,
            "max_int": 51199,
        },
    ),
    (
        Embedding43,
        [((1, 256), torch.int32)],
        {
            "model_name": ["pt_phi2_microsoft_phi_2_clm_hf", "pt_phi2_microsoft_phi_2_pytdml_clm_hf"],
            "pcc": 0.99,
            "max_int": 51199,
        },
    ),
    (
        Embedding43,
        [((1, 11), torch.int64)],
        {
            "model_name": ["pt_phi2_microsoft_phi_2_seq_cls_hf", "pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf"],
            "pcc": 0.99,
            "max_int": 51199,
        },
    ),
    (
        Embedding44,
        [((1, 256), torch.int64)],
        {"model_name": ["pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf"], "pcc": 0.99, "max_int": 32063},
    ),
    (
        Embedding44,
        [((1, 13), torch.int64)],
        {"model_name": ["pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf"], "pcc": 0.99, "max_int": 32063},
    ),
    (
        Embedding44,
        [((1, 5), torch.int64)],
        {"model_name": ["pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf"], "pcc": 0.99, "max_int": 32063},
    ),
    (
        Embedding45,
        [((1, 6), torch.int64)],
        {"model_name": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"], "pcc": 0.99, "max_int": 151935},
    ),
    (
        Embedding45,
        [((1, 29), torch.int64)],
        {"model_name": ["pt_qwen1_5_qwen_qwen1_5_0_5b_chat_clm_hf"], "pcc": 0.99, "max_int": 151935},
    ),
    (
        Embedding46,
        [((1, 35), torch.int64)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "max_int": 152063,
        },
    ),
    (
        Embedding47,
        [((1, 35), torch.int64)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "max_int": 151935,
        },
    ),
    (
        Embedding48,
        [((1, 35), torch.int64)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "max_int": 151935,
        },
    ),
    (
        Embedding49,
        [((1, 35), torch.int64)],
        {"model_name": ["pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf"], "pcc": 0.99, "max_int": 151935},
    ),
    (
        Embedding47,
        [((1, 29), torch.int64)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99, "max_int": 151935},
    ),
    (
        Embedding47,
        [((1, 39), torch.int64)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf"], "pcc": 0.99, "max_int": 151935},
    ),
    (
        Embedding46,
        [((1, 39), torch.int64)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf"], "pcc": 0.99, "max_int": 152063},
    ),
    (
        Embedding46,
        [((1, 29), torch.int64)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_7b_clm_hf"], "pcc": 0.99, "max_int": 152063},
    ),
    (
        Embedding48,
        [((1, 29), torch.int64)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_3b_clm_hf"], "pcc": 0.99, "max_int": 151935},
    ),
    (
        Embedding48,
        [((1, 39), torch.int64)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm_hf"], "pcc": 0.99, "max_int": 151935},
    ),
    (
        Embedding49,
        [((1, 29), torch.int64)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"], "pcc": 0.99, "max_int": 151935},
    ),
    (
        Embedding49,
        [((1, 39), torch.int64)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf"], "pcc": 0.99, "max_int": 151935},
    ),
    (
        Embedding50,
        [((1, 128), torch.int64)],
        {"model_name": ["pt_roberta_xlm_roberta_base_mlm_hf"], "pcc": 0.99, "max_int": 250001},
    ),
    (
        Embedding51,
        [((1, 128), torch.int64)],
        {
            "model_name": [
                "pt_roberta_xlm_roberta_base_mlm_hf",
                "pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf",
            ],
            "pcc": 0.99,
            "max_int": 0,
        },
    ),
    (
        Embedding52,
        [((1, 128), torch.int64)],
        {
            "model_name": [
                "pt_roberta_xlm_roberta_base_mlm_hf",
                "pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf",
            ],
            "pcc": 0.99,
            "max_int": 513,
        },
    ),
    (
        Embedding53,
        [((1, 128), torch.int64)],
        {
            "model_name": ["pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf"],
            "pcc": 0.99,
            "max_int": 50264,
        },
    ),
    (
        Embedding54,
        [((1, 128), torch.int64)],
        {"model_name": ["pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf"], "pcc": 0.99, "max_int": 30527},
    ),
    (
        Embedding2,
        [((1, 128), torch.int64), ((2, 768), torch.float32)],
        {"model_name": ["pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf"], "pcc": 0.99, "max_int": 1},
    ),
    (
        Embedding55,
        [((1, 1), torch.int64)],
        {
            "model_name": ["pt_t5_google_flan_t5_large_text_gen_hf", "pt_t5_t5_large_text_gen_hf"],
            "pcc": 0.99,
            "max_int": 32127,
        },
    ),
    (
        Embedding2,
        [((1, 1), torch.int32), ((32, 16), torch.float32)],
        {
            "model_name": ["pt_t5_google_flan_t5_large_text_gen_hf", "pt_t5_t5_large_text_gen_hf"],
            "pcc": 0.99,
            "max_int": 31,
        },
    ),
    (
        Embedding55,
        [((1, 61), torch.int64)],
        {
            "model_name": ["pt_t5_google_flan_t5_large_text_gen_hf", "pt_t5_t5_large_text_gen_hf"],
            "pcc": 0.99,
            "max_int": 32127,
        },
    ),
    pytest.param(
        (
            Embedding2,
            [((61, 61), torch.int32), ((32, 16), torch.float32)],
            {
                "model_name": ["pt_t5_google_flan_t5_large_text_gen_hf", "pt_t5_t5_large_text_gen_hf"],
                "pcc": 0.99,
                "max_int": 31,
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Embedding56,
        [((1, 1), torch.int64)],
        {
            "model_name": ["pt_t5_t5_small_text_gen_hf", "pt_t5_google_flan_t5_small_text_gen_hf"],
            "pcc": 0.99,
            "max_int": 32127,
        },
    ),
    (
        Embedding2,
        [((1, 1), torch.int32), ((32, 8), torch.float32)],
        {"model_name": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99, "max_int": 31},
    ),
    (
        Embedding56,
        [((1, 61), torch.int64)],
        {
            "model_name": ["pt_t5_t5_small_text_gen_hf", "pt_t5_google_flan_t5_small_text_gen_hf"],
            "pcc": 0.99,
            "max_int": 32127,
        },
    ),
    pytest.param(
        (
            Embedding2,
            [((61, 61), torch.int32), ((32, 8), torch.float32)],
            {"model_name": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99, "max_int": 31},
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Embedding2,
        [((1, 1), torch.int32), ((32, 6), torch.float32)],
        {"model_name": ["pt_t5_google_flan_t5_small_text_gen_hf"], "pcc": 0.99, "max_int": 31},
    ),
    pytest.param(
        (
            Embedding2,
            [((61, 61), torch.int32), ((32, 6), torch.float32)],
            {"model_name": ["pt_t5_google_flan_t5_small_text_gen_hf"], "pcc": 0.99, "max_int": 31},
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Embedding1,
        [((1, 1), torch.int64)],
        {
            "model_name": ["pt_t5_t5_base_text_gen_hf", "pt_t5_google_flan_t5_base_text_gen_hf"],
            "pcc": 0.99,
            "max_int": 32127,
        },
    ),
    (
        Embedding2,
        [((1, 1), torch.int32), ((32, 12), torch.float32)],
        {
            "model_name": ["pt_t5_t5_base_text_gen_hf", "pt_t5_google_flan_t5_base_text_gen_hf"],
            "pcc": 0.99,
            "max_int": 31,
        },
    ),
    (
        Embedding1,
        [((1, 61), torch.int64)],
        {
            "model_name": ["pt_t5_t5_base_text_gen_hf", "pt_t5_google_flan_t5_base_text_gen_hf"],
            "pcc": 0.99,
            "max_int": 32127,
        },
    ),
    pytest.param(
        (
            Embedding2,
            [((61, 61), torch.int32), ((32, 12), torch.float32)],
            {
                "model_name": ["pt_t5_t5_base_text_gen_hf", "pt_t5_google_flan_t5_base_text_gen_hf"],
                "pcc": 0.99,
                "max_int": 31,
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Embedding57,
        [((1, 256), torch.int64)],
        {"model_name": ["pt_xglm_facebook_xglm_1_7b_clm_hf"], "pcc": 0.99, "max_int": 256007},
    ),
    (
        Embedding58,
        [((1, 256), torch.int64)],
        {"model_name": ["pt_xglm_facebook_xglm_564m_clm_hf"], "pcc": 0.99, "max_int": 256007},
    ),
]


@pytest.mark.nightly_models_ops
@pytest.mark.parametrize("forge_module_and_shapes_dtypes", forge_modules_and_shapes_dtypes_list, ids=ids_func)
def test_module(forge_module_and_shapes_dtypes, forge_property_recorder):
    forge_property_recorder.record_op_name("Embedding")

    forge_module, operand_shapes_dtypes, metadata = forge_module_and_shapes_dtypes

    pcc = metadata.pop("pcc")
    max_int = metadata.pop("max_int")

    for metadata_name, metadata_value in metadata.items():
        forge_property_recorder("tags." + str(metadata_name), metadata_value)

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
