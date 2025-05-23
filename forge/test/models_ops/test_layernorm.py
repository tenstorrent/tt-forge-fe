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


class Layernorm0(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("layernorm0_const_1", shape=(1024,), dtype=torch.float32)
        self.add_constant("layernorm0_const_2", shape=(1024,), dtype=torch.float32)

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_constant("layernorm0_const_1"),
            self.get_constant("layernorm0_const_2"),
            dim=-1,
            epsilon=0.0,
        )
        return layernorm_output_1


class Layernorm1(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("layernorm1_const_1", shape=(768,), dtype=torch.float32)
        self.add_constant("layernorm1_const_2", shape=(768,), dtype=torch.float32)

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_constant("layernorm1_const_1"),
            self.get_constant("layernorm1_const_2"),
            dim=-1,
            epsilon=0.0,
        )
        return layernorm_output_1


class Layernorm2(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("layernorm2_const_1", shape=(384,), dtype=torch.float32)
        self.add_constant("layernorm2_const_2", shape=(384,), dtype=torch.float32)

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_constant("layernorm2_const_1"),
            self.get_constant("layernorm2_const_2"),
            dim=-1,
            epsilon=0.0,
        )
        return layernorm_output_1


class Layernorm3(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("layernorm3_const_1", shape=(256,), dtype=torch.float32)
        self.add_constant("layernorm3_const_2", shape=(256,), dtype=torch.float32)

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_constant("layernorm3_const_1"),
            self.get_constant("layernorm3_const_2"),
            dim=-1,
            epsilon=1e-05,
        )
        return layernorm_output_1


class Layernorm4(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("layernorm4_const_1", shape=(64,), dtype=torch.float32)
        self.add_constant("layernorm4_const_2", shape=(64,), dtype=torch.float32)

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_constant("layernorm4_const_1"),
            self.get_constant("layernorm4_const_2"),
            dim=-1,
            epsilon=1e-05,
        )
        return layernorm_output_1


class Layernorm5(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("layernorm5_const_1", shape=(128,), dtype=torch.float32)
        self.add_constant("layernorm5_const_2", shape=(128,), dtype=torch.float32)

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_constant("layernorm5_const_1"),
            self.get_constant("layernorm5_const_2"),
            dim=-1,
            epsilon=1e-05,
        )
        return layernorm_output_1


class Layernorm6(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("layernorm6_const_1", shape=(320,), dtype=torch.float32)
        self.add_constant("layernorm6_const_2", shape=(320,), dtype=torch.float32)

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_constant("layernorm6_const_1"),
            self.get_constant("layernorm6_const_2"),
            dim=-1,
            epsilon=1e-05,
        )
        return layernorm_output_1


class Layernorm7(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("layernorm7_const_1", shape=(512,), dtype=torch.float32)
        self.add_constant("layernorm7_const_2", shape=(512,), dtype=torch.float32)

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_constant("layernorm7_const_1"),
            self.get_constant("layernorm7_const_2"),
            dim=-1,
            epsilon=1e-05,
        )
        return layernorm_output_1


class Layernorm8(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("layernorm8_const_1", shape=(32,), dtype=torch.float32)
        self.add_constant("layernorm8_const_2", shape=(32,), dtype=torch.float32)

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_constant("layernorm8_const_1"),
            self.get_constant("layernorm8_const_2"),
            dim=-1,
            epsilon=1e-05,
        )
        return layernorm_output_1


class Layernorm9(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("layernorm9_const_1", shape=(160,), dtype=torch.float32)
        self.add_constant("layernorm9_const_2", shape=(160,), dtype=torch.float32)

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_constant("layernorm9_const_1"),
            self.get_constant("layernorm9_const_2"),
            dim=-1,
            epsilon=1e-05,
        )
        return layernorm_output_1


class Layernorm10(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("layernorm10_const_1", shape=(96,), dtype=torch.float32)
        self.add_constant("layernorm10_const_2", shape=(96,), dtype=torch.float32)

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_constant("layernorm10_const_1"),
            self.get_constant("layernorm10_const_2"),
            dim=-1,
            epsilon=1e-05,
        )
        return layernorm_output_1


class Layernorm11(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("layernorm11_const_1", shape=(192,), dtype=torch.float32)
        self.add_constant("layernorm11_const_2", shape=(192,), dtype=torch.float32)

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_constant("layernorm11_const_1"),
            self.get_constant("layernorm11_const_2"),
            dim=-1,
            epsilon=1e-05,
        )
        return layernorm_output_1


class Layernorm12(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("layernorm12_const_1", shape=(384,), dtype=torch.float32)
        self.add_constant("layernorm12_const_2", shape=(384,), dtype=torch.float32)

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_constant("layernorm12_const_1"),
            self.get_constant("layernorm12_const_2"),
            dim=-1,
            epsilon=1e-05,
        )
        return layernorm_output_1


class Layernorm13(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("layernorm13_const_1", shape=(768,), dtype=torch.float32)
        self.add_constant("layernorm13_const_2", shape=(768,), dtype=torch.float32)

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_constant("layernorm13_const_1"),
            self.get_constant("layernorm13_const_2"),
            dim=-1,
            epsilon=1e-05,
        )
        return layernorm_output_1


class Layernorm14(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "layernorm14.weight_1",
            forge.Parameter(*(1024,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )
        self.add_parameter(
            "layernorm14.weight_2",
            forge.Parameter(*(1024,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_parameter("layernorm14.weight_1"),
            self.get_parameter("layernorm14.weight_2"),
            dim=-1,
            epsilon=1e-05,
        )
        return layernorm_output_1


class Layernorm15(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "layernorm15.weight_1",
            forge.Parameter(*(2048,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )
        self.add_parameter(
            "layernorm15.weight_2",
            forge.Parameter(*(2048,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_parameter("layernorm15.weight_1"),
            self.get_parameter("layernorm15.weight_2"),
            dim=-1,
            epsilon=1e-05,
        )
        return layernorm_output_1


class Layernorm16(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "layernorm16.weight_1",
            forge.Parameter(*(1536,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )
        self.add_parameter(
            "layernorm16.weight_2",
            forge.Parameter(*(1536,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_parameter("layernorm16.weight_1"),
            self.get_parameter("layernorm16.weight_2"),
            dim=-1,
            epsilon=1e-05,
        )
        return layernorm_output_1


class Layernorm17(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "layernorm17.weight_1",
            forge.Parameter(*(512,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )
        self.add_parameter(
            "layernorm17.weight_2",
            forge.Parameter(*(512,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_parameter("layernorm17.weight_1"),
            self.get_parameter("layernorm17.weight_2"),
            dim=-1,
            epsilon=1e-05,
        )
        return layernorm_output_1


class Layernorm18(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "layernorm18.weight_1",
            forge.Parameter(*(384,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )
        self.add_parameter(
            "layernorm18.weight_2",
            forge.Parameter(*(384,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_parameter("layernorm18.weight_1"),
            self.get_parameter("layernorm18.weight_2"),
            dim=-1,
            epsilon=1e-05,
        )
        return layernorm_output_1


class Layernorm19(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "layernorm19.weight_1",
            forge.Parameter(*(768,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )
        self.add_parameter(
            "layernorm19.weight_2",
            forge.Parameter(*(768,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_parameter("layernorm19.weight_1"),
            self.get_parameter("layernorm19.weight_2"),
            dim=-1,
            epsilon=1e-05,
        )
        return layernorm_output_1


class Layernorm20(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "layernorm20.weight_1",
            forge.Parameter(*(1280,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )
        self.add_parameter(
            "layernorm20.weight_2",
            forge.Parameter(*(1280,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_parameter("layernorm20.weight_1"),
            self.get_parameter("layernorm20.weight_2"),
            dim=-1,
            epsilon=1e-05,
        )
        return layernorm_output_1


class Layernorm21(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("layernorm21_const_1", shape=(2432,), dtype=torch.float32)
        self.add_constant("layernorm21_const_2", shape=(2432,), dtype=torch.float32)

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_constant("layernorm21_const_1"),
            self.get_constant("layernorm21_const_2"),
            dim=-1,
            epsilon=1e-06,
        )
        return layernorm_output_1


class Layernorm22(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("layernorm22_const_1", shape=(1536,), dtype=torch.float32)
        self.add_constant("layernorm22_const_2", shape=(1536,), dtype=torch.float32)

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_constant("layernorm22_const_1"),
            self.get_constant("layernorm22_const_2"),
            dim=-1,
            epsilon=1e-06,
        )
        return layernorm_output_1


class Layernorm23(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "layernorm23.weight_1",
            forge.Parameter(*(768,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )
        self.add_parameter(
            "layernorm23.weight_2",
            forge.Parameter(*(768,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_parameter("layernorm23.weight_1"),
            self.get_parameter("layernorm23.weight_2"),
            dim=-1,
            epsilon=0.0,
        )
        return layernorm_output_1


class Layernorm24(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "layernorm24.weight_1",
            forge.Parameter(*(128,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )
        self.add_parameter(
            "layernorm24.weight_2",
            forge.Parameter(*(128,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_parameter("layernorm24.weight_1"),
            self.get_parameter("layernorm24.weight_2"),
            dim=-1,
            epsilon=0.0,
        )
        return layernorm_output_1


class Layernorm25(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "layernorm25.weight_1",
            forge.Parameter(*(4096,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )
        self.add_parameter(
            "layernorm25.weight_2",
            forge.Parameter(*(4096,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_parameter("layernorm25.weight_1"),
            self.get_parameter("layernorm25.weight_2"),
            dim=-1,
            epsilon=0.0,
        )
        return layernorm_output_1


class Layernorm26(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "layernorm26.weight_1",
            forge.Parameter(*(2048,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )
        self.add_parameter(
            "layernorm26.weight_2",
            forge.Parameter(*(2048,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_parameter("layernorm26.weight_1"),
            self.get_parameter("layernorm26.weight_2"),
            dim=-1,
            epsilon=0.0,
        )
        return layernorm_output_1


class Layernorm27(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "layernorm27.weight_1",
            forge.Parameter(*(1024,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )
        self.add_parameter(
            "layernorm27.weight_2",
            forge.Parameter(*(1024,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_parameter("layernorm27.weight_1"),
            self.get_parameter("layernorm27.weight_2"),
            dim=-1,
            epsilon=0.0,
        )
        return layernorm_output_1


class Layernorm28(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "layernorm28.weight_1",
            forge.Parameter(*(4544,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )
        self.add_parameter(
            "layernorm28.weight_2",
            forge.Parameter(*(4544,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_parameter("layernorm28.weight_1"),
            self.get_parameter("layernorm28.weight_2"),
            dim=-1,
            epsilon=1e-05,
        )
        return layernorm_output_1


class Layernorm29(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "layernorm29.weight_1",
            forge.Parameter(*(4096,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )
        self.add_parameter(
            "layernorm29.weight_2",
            forge.Parameter(*(4096,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_parameter("layernorm29.weight_1"),
            self.get_parameter("layernorm29.weight_2"),
            dim=-1,
            epsilon=1e-05,
        )
        return layernorm_output_1


class Layernorm30(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "layernorm30.weight_1",
            forge.Parameter(*(64,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )
        self.add_parameter(
            "layernorm30.weight_2",
            forge.Parameter(*(64,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_parameter("layernorm30.weight_1"),
            self.get_parameter("layernorm30.weight_2"),
            dim=-1,
            epsilon=1e-05,
        )
        return layernorm_output_1


class Layernorm31(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "layernorm31.weight_1",
            forge.Parameter(*(2560,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )
        self.add_parameter(
            "layernorm31.weight_2",
            forge.Parameter(*(2560,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_parameter("layernorm31.weight_1"),
            self.get_parameter("layernorm31.weight_2"),
            dim=-1,
            epsilon=1e-05,
        )
        return layernorm_output_1


class Layernorm32(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "layernorm32.weight_1",
            forge.Parameter(*(261,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )
        self.add_parameter(
            "layernorm32.weight_2",
            forge.Parameter(*(261,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_parameter("layernorm32.weight_1"),
            self.get_parameter("layernorm32.weight_2"),
            dim=-1,
            epsilon=1e-05,
        )
        return layernorm_output_1


class Layernorm33(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "layernorm33.weight_1",
            forge.Parameter(*(322,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )
        self.add_parameter(
            "layernorm33.weight_2",
            forge.Parameter(*(322,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_parameter("layernorm33.weight_1"),
            self.get_parameter("layernorm33.weight_2"),
            dim=-1,
            epsilon=1e-05,
        )
        return layernorm_output_1


class Layernorm34(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "layernorm34.weight_1",
            forge.Parameter(*(1024,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )
        self.add_parameter(
            "layernorm34.weight_2",
            forge.Parameter(*(1024,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_parameter("layernorm34.weight_1"),
            self.get_parameter("layernorm34.weight_2"),
            dim=-1,
            epsilon=1e-06,
        )
        return layernorm_output_1


class Layernorm35(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "layernorm35.weight_1",
            forge.Parameter(*(768,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )
        self.add_parameter(
            "layernorm35.weight_2",
            forge.Parameter(*(768,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_parameter("layernorm35.weight_1"),
            self.get_parameter("layernorm35.weight_2"),
            dim=-1,
            epsilon=1e-06,
        )
        return layernorm_output_1


class Layernorm36(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "layernorm36.weight_1",
            forge.Parameter(*(192,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )
        self.add_parameter(
            "layernorm36.weight_2",
            forge.Parameter(*(192,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_parameter("layernorm36.weight_1"),
            self.get_parameter("layernorm36.weight_2"),
            dim=-1,
            epsilon=0.0,
        )
        return layernorm_output_1


class Layernorm37(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "layernorm37.weight_1",
            forge.Parameter(*(384,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )
        self.add_parameter(
            "layernorm37.weight_2",
            forge.Parameter(*(384,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_parameter("layernorm37.weight_1"),
            self.get_parameter("layernorm37.weight_2"),
            dim=-1,
            epsilon=0.0,
        )
        return layernorm_output_1


class Layernorm38(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "layernorm38.weight_1",
            forge.Parameter(*(256,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )
        self.add_parameter(
            "layernorm38.weight_2",
            forge.Parameter(*(256,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_parameter("layernorm38.weight_1"),
            self.get_parameter("layernorm38.weight_2"),
            dim=-1,
            epsilon=1e-05,
        )
        return layernorm_output_1


class Layernorm39(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "layernorm39.weight_1",
            forge.Parameter(*(128,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )
        self.add_parameter(
            "layernorm39.weight_2",
            forge.Parameter(*(128,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_parameter("layernorm39.weight_1"),
            self.get_parameter("layernorm39.weight_2"),
            dim=-1,
            epsilon=1e-05,
        )
        return layernorm_output_1


class Layernorm40(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "layernorm40.weight_1",
            forge.Parameter(*(320,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )
        self.add_parameter(
            "layernorm40.weight_2",
            forge.Parameter(*(320,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_parameter("layernorm40.weight_1"),
            self.get_parameter("layernorm40.weight_2"),
            dim=-1,
            epsilon=1e-05,
        )
        return layernorm_output_1


class Layernorm41(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "layernorm41.weight_1",
            forge.Parameter(*(512,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )
        self.add_parameter(
            "layernorm41.weight_2",
            forge.Parameter(*(512,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_parameter("layernorm41.weight_1"),
            self.get_parameter("layernorm41.weight_2"),
            dim=-1,
            epsilon=1e-06,
        )
        return layernorm_output_1


class Layernorm42(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "layernorm42.weight_1",
            forge.Parameter(*(32,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )
        self.add_parameter(
            "layernorm42.weight_2",
            forge.Parameter(*(32,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_parameter("layernorm42.weight_1"),
            self.get_parameter("layernorm42.weight_2"),
            dim=-1,
            epsilon=1e-05,
        )
        return layernorm_output_1


class Layernorm43(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "layernorm43.weight_1",
            forge.Parameter(*(160,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )
        self.add_parameter(
            "layernorm43.weight_2",
            forge.Parameter(*(160,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_parameter("layernorm43.weight_1"),
            self.get_parameter("layernorm43.weight_2"),
            dim=-1,
            epsilon=1e-05,
        )
        return layernorm_output_1


class Layernorm44(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "layernorm44.weight_1",
            forge.Parameter(*(96,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )
        self.add_parameter(
            "layernorm44.weight_2",
            forge.Parameter(*(96,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_parameter("layernorm44.weight_1"),
            self.get_parameter("layernorm44.weight_2"),
            dim=-1,
            epsilon=1e-05,
        )
        return layernorm_output_1


class Layernorm45(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "layernorm45.weight_1",
            forge.Parameter(*(192,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )
        self.add_parameter(
            "layernorm45.weight_2",
            forge.Parameter(*(192,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_parameter("layernorm45.weight_1"),
            self.get_parameter("layernorm45.weight_2"),
            dim=-1,
            epsilon=1e-05,
        )
        return layernorm_output_1


class Layernorm46(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "layernorm46.weight_1",
            forge.Parameter(*(1280,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )
        self.add_parameter(
            "layernorm46.weight_2",
            forge.Parameter(*(1280,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_parameter("layernorm46.weight_1"),
            self.get_parameter("layernorm46.weight_2"),
            dim=-1,
            epsilon=1e-06,
        )
        return layernorm_output_1


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (
        Layernorm0,
        [((1, 384, 1024), torch.float32)],
        {
            "model_names": ["onnx_bert_phiyodr_bert_large_finetuned_squad2_qa_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "0.0"},
        },
    ),
    (
        Layernorm1,
        [((1, 128, 768), torch.float32)],
        {"model_names": ["onnx_bert_bert_base_uncased_mlm_hf"], "pcc": 0.99, "args": {"dim": "-1", "epsilon": "0.0"}},
    ),
    (
        Layernorm1,
        [((1, 6, 768), torch.float32)],
        {
            "model_names": ["onnx_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "0.0"},
        },
    ),
    (
        Layernorm2,
        [((1, 13, 384), torch.float32)],
        {
            "model_names": ["onnx_minilm_sentence_transformers_all_minilm_l6_v2_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "0.0"},
        },
    ),
    (
        Layernorm3,
        [((1, 100, 256), torch.float32)],
        {
            "model_names": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm3,
        [((1, 280, 256), torch.float32)],
        {
            "model_names": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm4,
        [((1, 16384, 64), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_mit_b3_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm4,
        [((1, 256, 64), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_mit_b3_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm5,
        [((1, 4096, 128), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_mit_b3_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm5,
        [((1, 256, 128), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_mit_b3_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm6,
        [((1, 1024, 320), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_mit_b3_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm6,
        [((1, 256, 320), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_mit_b3_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm7,
        [((1, 256, 512), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_mit_b3_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm8,
        [((1, 16384, 32), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm8,
        [((1, 256, 32), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm4,
        [((1, 4096, 64), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm9,
        [((1, 1024, 160), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm9,
        [((1, 256, 160), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm3,
        [((1, 256, 256), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm10,
        [((1, 4096, 96), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm11,
        [((1, 1024, 192), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm12,
        [((1, 256, 384), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm13,
        [((1, 64, 768), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm0,
        [((1, 197, 1024), torch.float32)],
        {
            "model_names": ["onnx_vit_base_google_vit_large_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "0.0"},
        },
    ),
    (
        Layernorm1,
        [((1, 197, 768), torch.float32)],
        {
            "model_names": ["onnx_vit_base_google_vit_base_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "0.0"},
        },
    ),
    pytest.param(
        (
            Layernorm14,
            [((2, 1, 1024), torch.float32)],
            {
                "model_names": ["pt_stereo_facebook_musicgen_small_music_generation_hf"],
                "pcc": 0.99,
                "args": {"dim": "-1", "epsilon": "1e-05"},
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Layernorm15,
            [((2, 1, 2048), torch.float32)],
            {
                "model_names": ["pt_stereo_facebook_musicgen_large_music_generation_hf"],
                "pcc": 0.99,
                "args": {"dim": "-1", "epsilon": "1e-05"},
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Layernorm16,
            [((2, 1, 1536), torch.float32)],
            {
                "model_names": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"],
                "pcc": 0.99,
                "args": {"dim": "-1", "epsilon": "1e-05"},
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Layernorm17,
        [((1, 1, 512), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_base_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm17,
        [((1, 1500, 512), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_base_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm18,
        [((1, 1, 384), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm18,
        [((1, 1500, 384), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm19,
        [((1, 1, 768), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm19,
        [((1, 1500, 768), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm20,
        [((1, 1, 1280), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_large_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm20,
        [((1, 1500, 1280), torch.float32)],
        {
            "model_names": [
                "pt_whisper_openai_whisper_large_speech_recognition_hf",
                "pt_whisper_openai_whisper_large_v3_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm14,
        [((1, 1, 1024), torch.float32)],
        {
            "model_names": [
                "pt_whisper_openai_whisper_medium_speech_recognition_hf",
                "pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm14,
        [((1, 1500, 1024), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_medium_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm20,
        [((1, 2, 1280), torch.float32)],
        {
            "model_names": [
                "pt_whisper_openai_whisper_large_v3_clm_hf",
                "pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm17,
        [((2, 7, 512), torch.float32)],
        {
            "model_names": ["pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm14,
        [((1, 577, 1024), torch.float32)],
        {
            "model_names": ["pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm21,
        [((2, 4096, 2432), torch.float32)],
        {
            "model_names": [
                "pt_stable_diffusion_stable_diffusion_3_5_large_cond_gen_hf",
                "pt_stable_diffusion_stable_diffusion_3_5_large_turbo_cond_gen_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-06"},
        },
    ),
    (
        Layernorm21,
        [((2, 333, 2432), torch.float32)],
        {
            "model_names": [
                "pt_stable_diffusion_stable_diffusion_3_5_large_cond_gen_hf",
                "pt_stable_diffusion_stable_diffusion_3_5_large_turbo_cond_gen_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-06"},
        },
    ),
    (
        Layernorm22,
        [((2, 4096, 1536), torch.float32)],
        {
            "model_names": ["pt_stable_diffusion_stable_diffusion_3_5_medium_cond_gen_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-06"},
        },
    ),
    (
        Layernorm22,
        [((2, 333, 1536), torch.float32)],
        {
            "model_names": ["pt_stable_diffusion_stable_diffusion_3_5_medium_cond_gen_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-06"},
        },
    ),
    (
        Layernorm23,
        [((1, 201, 768), torch.float32)],
        {
            "model_names": ["pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "0.0"},
        },
    ),
    (
        Layernorm16,
        [((1, 1536), torch.float32)],
        {
            "model_names": ["pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm23,
        [((1, 204, 768), torch.float32)],
        {"model_names": ["pt_vilt_dandelin_vilt_b32_mlm_mlm_hf"], "pcc": 0.99, "args": {"dim": "-1", "epsilon": "0.0"}},
    ),
    (
        Layernorm24,
        [((1, 128, 128), torch.float32)],
        {
            "model_names": [
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_albert_base_v1_token_cls_hf",
                "pt_albert_xxlarge_v2_token_cls_hf",
                "pt_albert_base_v2_token_cls_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
                "pt_albert_xlarge_v2_token_cls_hf",
                "pt_albert_xxlarge_v1_mlm_hf",
                "pt_albert_large_v2_mlm_hf",
                "pt_albert_xlarge_v1_mlm_hf",
                "pt_albert_base_v2_mlm_hf",
                "pt_albert_large_v2_token_cls_hf",
                "pt_albert_large_v1_token_cls_hf",
                "pt_albert_large_v1_mlm_hf",
                "pt_albert_base_v1_mlm_hf",
                "pt_albert_xlarge_v1_token_cls_hf",
                "pt_albert_xlarge_v2_mlm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "0.0"},
        },
    ),
    (
        Layernorm25,
        [((1, 128, 4096), torch.float32)],
        {
            "model_names": [
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_albert_xxlarge_v2_token_cls_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
                "pt_albert_xxlarge_v1_mlm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "0.0"},
        },
    ),
    (
        Layernorm23,
        [((1, 128, 768), torch.float32)],
        {
            "model_names": [
                "pt_albert_base_v1_token_cls_hf",
                "pt_albert_base_v2_token_cls_hf",
                "pt_albert_base_v2_mlm_hf",
                "pt_albert_base_v1_mlm_hf",
                "pt_bert_bert_base_uncased_mlm_hf",
                "pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf",
                "pt_distilbert_distilbert_base_uncased_mlm_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
                "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf",
                "pt_distilbert_distilbert_base_multilingual_cased_mlm_hf",
                "pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder",
                "pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "0.0"},
        },
    ),
    (
        Layernorm19,
        [((1, 128, 768), torch.float32)],
        {
            "model_names": [
                "pt_roberta_xlm_roberta_base_mlm_hf",
                "pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm24,
        [((1, 9, 128), torch.float32)],
        {
            "model_names": ["pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "0.0"},
        },
    ),
    (
        Layernorm23,
        [((1, 9, 768), torch.float32)],
        {
            "model_names": ["pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "0.0"},
        },
    ),
    (
        Layernorm26,
        [((1, 128, 2048), torch.float32)],
        {
            "model_names": [
                "pt_albert_xlarge_v2_token_cls_hf",
                "pt_albert_xlarge_v1_mlm_hf",
                "pt_albert_xlarge_v1_token_cls_hf",
                "pt_albert_xlarge_v2_mlm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "0.0"},
        },
    ),
    (
        Layernorm27,
        [((1, 128, 1024), torch.float32)],
        {
            "model_names": [
                "pt_albert_large_v2_mlm_hf",
                "pt_albert_large_v2_token_cls_hf",
                "pt_albert_large_v1_token_cls_hf",
                "pt_albert_large_v1_mlm_hf",
                "pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "0.0"},
        },
    ),
    (
        Layernorm24,
        [((1, 14, 128), torch.float32)],
        {
            "model_names": ["pt_albert_twmkn9_albert_base_v2_squad2_qa_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "0.0"},
        },
    ),
    (
        Layernorm23,
        [((1, 14, 768), torch.float32)],
        {
            "model_names": ["pt_albert_twmkn9_albert_base_v2_squad2_qa_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "0.0"},
        },
    ),
    (
        Layernorm14,
        [((1, 256, 1024), torch.float32)],
        {
            "model_names": [
                "pt_bart_facebook_bart_large_mnli_seq_cls_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
                "pt_codegen_salesforce_codegen_350m_mono_clm_hf",
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_opt_facebook_opt_350m_clm_hf",
                "pt_xglm_facebook_xglm_564m_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm27,
        [((1, 384, 1024), torch.float32)],
        {
            "model_names": [
                "pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf",
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "0.0"},
        },
    ),
    (
        Layernorm23,
        [((1, 6, 768), torch.float32)],
        {
            "model_names": ["pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "0.0"},
        },
    ),
    (
        Layernorm16,
        [((1, 32, 1536), torch.float32)],
        {
            "model_names": ["pt_bloom_bigscience_bloom_1b1_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm23,
        [((1, 384, 768), torch.float32)],
        {
            "model_names": ["pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "0.0"},
        },
    ),
    (
        Layernorm28,
        [((1, 6, 4544), torch.float32)],
        {
            "model_names": ["pt_falcon_tiiuae_falcon_7b_instruct_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm29,
        [((1, 334, 4096), torch.float32)],
        {"model_names": ["pt_fuyu_adept_fuyu_8b_qa_hf"], "pcc": 0.99, "args": {"dim": "-1", "epsilon": "1e-05"}},
    ),
    (
        Layernorm30,
        [((1, 334, 64, 64), torch.float32)],
        {"model_names": ["pt_fuyu_adept_fuyu_8b_qa_hf"], "pcc": 0.99, "args": {"dim": "-1", "epsilon": "1e-05"}},
    ),
    (
        Layernorm19,
        [((1, 7, 768), torch.float32)],
        {
            "model_names": [
                "pt_gpt2_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm19,
        [((1, 256, 768), torch.float32)],
        {
            "model_names": [
                "pt_gpt2_gpt2_text_gen_hf",
                "pt_gptneo_eleutherai_gpt_neo_125m_clm_hf",
                "pt_opt_facebook_opt_125m_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm31,
        [((1, 256, 2560), torch.float32)],
        {
            "model_names": [
                "pt_gptneo_eleutherai_gpt_neo_2_7b_clm_hf",
                "pt_phi2_microsoft_phi_2_clm_hf",
                "pt_phi2_microsoft_phi_2_pytdml_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm15,
        [((1, 256, 2048), torch.float32)],
        {
            "model_names": [
                "pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf",
                "pt_opt_facebook_opt_1_3b_clm_hf",
                "pt_phi1_5_microsoft_phi_1_5_seq_cls_hf",
                "pt_phi1_microsoft_phi_1_seq_cls_hf",
                "pt_xglm_facebook_xglm_1_7b_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm15,
        [((1, 32, 2048), torch.float32)],
        {
            "model_names": [
                "pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf",
                "pt_opt_facebook_opt_1_3b_qa_hf",
                "pt_opt_facebook_opt_1_3b_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm19,
        [((1, 32, 768), torch.float32)],
        {
            "model_names": [
                "pt_gptneo_eleutherai_gpt_neo_125m_seq_cls_hf",
                "pt_opt_facebook_opt_125m_qa_hf",
                "pt_opt_facebook_opt_125m_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm31,
        [((1, 32, 2560), torch.float32)],
        {
            "model_names": ["pt_gptneo_eleutherai_gpt_neo_2_7b_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm19,
        [((32, 768), torch.float32)],
        {
            "model_names": ["pt_opt_facebook_opt_125m_qa_hf", "pt_opt_facebook_opt_125m_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm14,
        [((1, 32, 1024), torch.float32)],
        {
            "model_names": ["pt_opt_facebook_opt_350m_qa_hf", "pt_opt_facebook_opt_350m_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm15,
        [((32, 2048), torch.float32)],
        {
            "model_names": ["pt_opt_facebook_opt_1_3b_qa_hf", "pt_opt_facebook_opt_1_3b_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm15,
        [((256, 2048), torch.float32)],
        {"model_names": ["pt_opt_facebook_opt_1_3b_clm_hf"], "pcc": 0.99, "args": {"dim": "-1", "epsilon": "1e-05"}},
    ),
    (
        Layernorm19,
        [((256, 768), torch.float32)],
        {"model_names": ["pt_opt_facebook_opt_125m_clm_hf"], "pcc": 0.99, "args": {"dim": "-1", "epsilon": "1e-05"}},
    ),
    (
        Layernorm14,
        [((1, 512, 1024), torch.float32)],
        {
            "model_names": [
                "pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm32,
        [((1, 50176, 261), torch.float32)],
        {
            "model_names": ["pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm19,
        [((1, 2048, 768), torch.float32)],
        {
            "model_names": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm20,
        [((1, 256, 1280), torch.float32)],
        {
            "model_names": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm17,
        [((1, 50176, 512), torch.float32)],
        {
            "model_names": ["pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm33,
        [((1, 3025, 322), torch.float32)],
        {
            "model_names": ["pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm15,
        [((1, 7, 2048), torch.float32)],
        {
            "model_names": ["pt_phi1_5_microsoft_phi_1_5_clm_hf", "pt_phi1_microsoft_phi_1_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm15,
        [((1, 12, 2048), torch.float32)],
        {
            "model_names": ["pt_phi1_5_microsoft_phi_1_5_token_cls_hf", "pt_phi1_microsoft_phi_1_token_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm31,
        [((1, 12, 2560), torch.float32)],
        {
            "model_names": ["pt_phi2_microsoft_phi_2_pytdml_token_cls_hf", "pt_phi2_microsoft_phi_2_token_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm31,
        [((1, 11, 2560), torch.float32)],
        {
            "model_names": ["pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf", "pt_phi2_microsoft_phi_2_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm27,
        [((1, 197, 1024), torch.float32)],
        {
            "model_names": [
                "pt_beit_microsoft_beit_large_patch16_224_img_cls_hf",
                "pt_vit_google_vit_large_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "0.0"},
        },
    ),
    (
        Layernorm34,
        [((1, 197, 1024), torch.float32)],
        {
            "model_names": ["pt_vit_vit_l_16_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-06"},
        },
    ),
    (
        Layernorm27,
        [((1, 1024), torch.float32)],
        {
            "model_names": ["pt_beit_microsoft_beit_large_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "0.0"},
        },
    ),
    (
        Layernorm23,
        [((1, 197, 768), torch.float32)],
        {
            "model_names": [
                "pt_beit_microsoft_beit_base_patch16_224_img_cls_hf",
                "pt_deit_facebook_deit_base_patch16_224_img_cls_hf",
                "pt_deit_facebook_deit_base_distilled_patch16_224_img_cls_hf",
                "pt_vit_google_vit_base_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "0.0"},
        },
    ),
    (
        Layernorm35,
        [((1, 197, 768), torch.float32)],
        {
            "model_names": ["pt_vit_vit_b_16_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-06"},
        },
    ),
    (
        Layernorm23,
        [((1, 768), torch.float32)],
        {
            "model_names": ["pt_beit_microsoft_beit_base_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "0.0"},
        },
    ),
    (
        Layernorm36,
        [((1, 197, 192), torch.float32)],
        {
            "model_names": ["pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "0.0"},
        },
    ),
    (
        Layernorm37,
        [((1, 197, 384), torch.float32)],
        {
            "model_names": ["pt_deit_facebook_deit_small_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "0.0"},
        },
    ),
    (
        Layernorm38,
        [((1, 100, 256), torch.float32)],
        {
            "model_names": [
                "pt_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
                "pt_detr_facebook_detr_resnet_50_obj_det_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm38,
        [((1, 280, 256), torch.float32)],
        {
            "model_names": [
                "pt_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
                "pt_detr_facebook_detr_resnet_50_obj_det_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm30,
        [((1, 19200, 64), torch.float32)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm30,
        [((1, 300, 64), torch.float32)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm39,
        [((1, 4800, 128), torch.float32)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm39,
        [((1, 300, 128), torch.float32)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm40,
        [((1, 1200, 320), torch.float32)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm40,
        [((1, 300, 320), torch.float32)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm17,
        [((1, 300, 512), torch.float32)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm19,
        [((1, 257, 768), torch.float32)],
        {
            "model_names": ["pt_mgp_alibaba_damo_mgp_str_base_scene_text_recognition_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm19,
        [((1, 27, 768), torch.float32)],
        {
            "model_names": ["pt_mgp_alibaba_damo_mgp_str_base_scene_text_recognition_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm35,
        [((1, 49, 768), torch.float32)],
        {
            "model_names": ["pt_mlp_mixer_mixer_b32_224_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-06"},
        },
    ),
    (
        Layernorm19,
        [((1, 49, 768), torch.float32)],
        {
            "model_names": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm35,
        [((1, 196, 768), torch.float32)],
        {
            "model_names": [
                "pt_mlp_mixer_mixer_b16_224_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_goog_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_miil_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_miil_in21k_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-06"},
        },
    ),
    (
        Layernorm19,
        [((1, 196, 768), torch.float32)],
        {
            "model_names": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm41,
        [((1, 49, 512), torch.float32)],
        {
            "model_names": ["pt_mlp_mixer_mixer_s32_224_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-06"},
        },
    ),
    (
        Layernorm34,
        [((1, 49, 1024), torch.float32)],
        {
            "model_names": ["pt_mlp_mixer_mixer_l32_224_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-06"},
        },
    ),
    (
        Layernorm41,
        [((1, 196, 512), torch.float32)],
        {
            "model_names": ["pt_mlp_mixer_mixer_s16_224_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-06"},
        },
    ),
    (
        Layernorm17,
        [((1, 256, 512), torch.float32)],
        {
            "model_names": [
                "pt_mlp_mixer_base_img_cls_github",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm34,
        [((1, 196, 1024), torch.float32)],
        {
            "model_names": ["pt_mlp_mixer_mixer_l16_224_img_cls_timm", "pt_mlp_mixer_mixer_l16_224_in21k_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-06"},
        },
    ),
    (
        Layernorm42,
        [((1, 16384, 32), torch.float32)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm42,
        [((1, 256, 32), torch.float32)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm30,
        [((1, 4096, 64), torch.float32)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm30,
        [((1, 256, 64), torch.float32)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm43,
        [((1, 1024, 160), torch.float32)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm43,
        [((1, 256, 160), torch.float32)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm38,
        [((1, 256, 256), torch.float32)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm30,
        [((1, 16384, 64), torch.float32)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm39,
        [((1, 4096, 128), torch.float32)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm39,
        [((1, 256, 128), torch.float32)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm40,
        [((1, 1024, 320), torch.float32)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm40,
        [((1, 256, 320), torch.float32)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm44,
        [((1, 64, 64, 96), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm45,
        [((1, 32, 32, 192), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm18,
        [((1, 16, 16, 384), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm19,
        [((1, 8, 8, 768), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm44,
        [((1, 56, 56, 96), torch.float32)],
        {
            "model_names": ["pt_swin_swin_t_img_cls_torchvision", "pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm18,
        [((1, 28, 28, 384), torch.float32)],
        {
            "model_names": ["pt_swin_swin_t_img_cls_torchvision", "pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm45,
        [((1, 28, 28, 192), torch.float32)],
        {
            "model_names": ["pt_swin_swin_t_img_cls_torchvision", "pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm19,
        [((1, 14, 14, 768), torch.float32)],
        {
            "model_names": ["pt_swin_swin_t_img_cls_torchvision", "pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm18,
        [((1, 14, 14, 384), torch.float32)],
        {
            "model_names": ["pt_swin_swin_t_img_cls_torchvision", "pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm16,
        [((1, 7, 7, 1536), torch.float32)],
        {
            "model_names": ["pt_swin_swin_t_img_cls_torchvision", "pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm19,
        [((1, 7, 7, 768), torch.float32)],
        {
            "model_names": ["pt_swin_swin_t_img_cls_torchvision", "pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm44,
        [((1, 3136, 96), torch.float32)],
        {
            "model_names": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm18,
        [((1, 784, 384), torch.float32)],
        {
            "model_names": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm45,
        [((1, 784, 192), torch.float32)],
        {
            "model_names": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm18,
        [((1, 196, 384), torch.float32)],
        {
            "model_names": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm16,
        [((1, 49, 1536), torch.float32)],
        {
            "model_names": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm39,
        [((1, 64, 64, 128), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm38,
        [((1, 32, 32, 256), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm17,
        [((1, 16, 16, 512), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm14,
        [((1, 8, 8, 1024), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm39,
        [((1, 56, 56, 128), torch.float32)],
        {"model_names": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "args": {"dim": "-1", "epsilon": "1e-05"}},
    ),
    (
        Layernorm17,
        [((1, 28, 28, 512), torch.float32)],
        {"model_names": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "args": {"dim": "-1", "epsilon": "1e-05"}},
    ),
    (
        Layernorm38,
        [((1, 28, 28, 256), torch.float32)],
        {"model_names": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "args": {"dim": "-1", "epsilon": "1e-05"}},
    ),
    (
        Layernorm14,
        [((1, 14, 14, 1024), torch.float32)],
        {"model_names": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "args": {"dim": "-1", "epsilon": "1e-05"}},
    ),
    (
        Layernorm17,
        [((1, 14, 14, 512), torch.float32)],
        {"model_names": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "args": {"dim": "-1", "epsilon": "1e-05"}},
    ),
    (
        Layernorm15,
        [((1, 7, 7, 2048), torch.float32)],
        {"model_names": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "args": {"dim": "-1", "epsilon": "1e-05"}},
    ),
    (
        Layernorm14,
        [((1, 7, 7, 1024), torch.float32)],
        {"model_names": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "args": {"dim": "-1", "epsilon": "1e-05"}},
    ),
    (
        Layernorm35,
        [((1, 50, 768), torch.float32)],
        {
            "model_names": ["pt_vit_vit_b_32_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-06"},
        },
    ),
    (
        Layernorm34,
        [((1, 50, 1024), torch.float32)],
        {
            "model_names": ["pt_vit_vit_l_32_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-06"},
        },
    ),
    (
        Layernorm46,
        [((1, 1370, 1280), torch.float32)],
        {
            "model_names": ["pt_vit_vit_h_14_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-06"},
        },
    ),
    (
        Layernorm36,
        [((1, 1445, 192), torch.float32)],
        {
            "model_names": ["pt_yolos_hustvl_yolos_tiny_obj_det_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "0.0"},
        },
    ),
]


@pytest.mark.nightly_models_ops
@pytest.mark.parametrize("forge_module_and_shapes_dtypes", forge_modules_and_shapes_dtypes_list, ids=ids_func)
def test_module(forge_module_and_shapes_dtypes):

    record_forge_op_name("Layernorm")

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

    verify(
        inputs,
        framework_model,
        compiled_model,
        VerifyConfig(value_checker=AutomaticValueChecker(pcc=pcc)),
    )
