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
        self.add_parameter(
            "layernorm0.weight_1",
            forge.Parameter(*(128,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )
        self.add_parameter(
            "layernorm0.weight_2",
            forge.Parameter(*(128,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_parameter("layernorm0.weight_1"),
            self.get_parameter("layernorm0.weight_2"),
            dim=-1,
            epsilon=0.0,
        )
        return layernorm_output_1


class Layernorm1(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "layernorm1.weight_1",
            forge.Parameter(*(1024,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )
        self.add_parameter(
            "layernorm1.weight_2",
            forge.Parameter(*(1024,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_parameter("layernorm1.weight_1"),
            self.get_parameter("layernorm1.weight_2"),
            dim=-1,
            epsilon=0.0,
        )
        return layernorm_output_1


class Layernorm2(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "layernorm2.weight_1",
            forge.Parameter(*(768,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )
        self.add_parameter(
            "layernorm2.weight_2",
            forge.Parameter(*(768,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_parameter("layernorm2.weight_1"),
            self.get_parameter("layernorm2.weight_2"),
            dim=-1,
            epsilon=0.0,
        )
        return layernorm_output_1


class Layernorm3(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "layernorm3.weight_1",
            forge.Parameter(*(512,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )
        self.add_parameter(
            "layernorm3.weight_2",
            forge.Parameter(*(512,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_parameter("layernorm3.weight_1"),
            self.get_parameter("layernorm3.weight_2"),
            dim=-1,
            epsilon=1e-05,
        )
        return layernorm_output_1


class Layernorm4(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "layernorm4.weight_1",
            forge.Parameter(*(512,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )
        self.add_parameter(
            "layernorm4.weight_2",
            forge.Parameter(*(512,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_parameter("layernorm4.weight_1"),
            self.get_parameter("layernorm4.weight_2"),
            dim=-1,
            epsilon=1e-06,
        )
        return layernorm_output_1


class Layernorm5(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "layernorm5.weight_1",
            forge.Parameter(*(2048,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )
        self.add_parameter(
            "layernorm5.weight_2",
            forge.Parameter(*(2048,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_parameter("layernorm5.weight_1"),
            self.get_parameter("layernorm5.weight_2"),
            dim=-1,
            epsilon=1e-05,
        )
        return layernorm_output_1


class Layernorm6(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "layernorm6.weight_1",
            forge.Parameter(*(32,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )
        self.add_parameter(
            "layernorm6.weight_2",
            forge.Parameter(*(32,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_parameter("layernorm6.weight_1"),
            self.get_parameter("layernorm6.weight_2"),
            dim=-1,
            epsilon=1e-05,
        )
        return layernorm_output_1


class Layernorm7(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "layernorm7.weight_1",
            forge.Parameter(*(64,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )
        self.add_parameter(
            "layernorm7.weight_2",
            forge.Parameter(*(64,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_parameter("layernorm7.weight_1"),
            self.get_parameter("layernorm7.weight_2"),
            dim=-1,
            epsilon=1e-05,
        )
        return layernorm_output_1


class Layernorm8(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "layernorm8.weight_1",
            forge.Parameter(*(160,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )
        self.add_parameter(
            "layernorm8.weight_2",
            forge.Parameter(*(160,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_parameter("layernorm8.weight_1"),
            self.get_parameter("layernorm8.weight_2"),
            dim=-1,
            epsilon=1e-05,
        )
        return layernorm_output_1


class Layernorm9(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "layernorm9.weight_1",
            forge.Parameter(*(256,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )
        self.add_parameter(
            "layernorm9.weight_2",
            forge.Parameter(*(256,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_parameter("layernorm9.weight_1"),
            self.get_parameter("layernorm9.weight_2"),
            dim=-1,
            epsilon=1e-05,
        )
        return layernorm_output_1


class Layernorm10(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "layernorm10.weight_1",
            forge.Parameter(*(96,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )
        self.add_parameter(
            "layernorm10.weight_2",
            forge.Parameter(*(96,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_parameter("layernorm10.weight_1"),
            self.get_parameter("layernorm10.weight_2"),
            dim=-1,
            epsilon=1e-05,
        )
        return layernorm_output_1


class Layernorm11(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "layernorm11.weight_1",
            forge.Parameter(*(384,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )
        self.add_parameter(
            "layernorm11.weight_2",
            forge.Parameter(*(384,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_parameter("layernorm11.weight_1"),
            self.get_parameter("layernorm11.weight_2"),
            dim=-1,
            epsilon=1e-05,
        )
        return layernorm_output_1


class Layernorm12(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "layernorm12.weight_1",
            forge.Parameter(*(192,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )
        self.add_parameter(
            "layernorm12.weight_2",
            forge.Parameter(*(192,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_parameter("layernorm12.weight_1"),
            self.get_parameter("layernorm12.weight_2"),
            dim=-1,
            epsilon=1e-05,
        )
        return layernorm_output_1


class Layernorm13(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "layernorm13.weight_1",
            forge.Parameter(*(768,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )
        self.add_parameter(
            "layernorm13.weight_2",
            forge.Parameter(*(768,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_parameter("layernorm13.weight_1"),
            self.get_parameter("layernorm13.weight_2"),
            dim=-1,
            epsilon=1e-05,
        )
        return layernorm_output_1


class Layernorm14(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "layernorm14.weight_1",
            forge.Parameter(*(1536,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )
        self.add_parameter(
            "layernorm14.weight_2",
            forge.Parameter(*(1536,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
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
            forge.Parameter(*(768,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )
        self.add_parameter(
            "layernorm15.weight_2",
            forge.Parameter(*(768,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_parameter("layernorm15.weight_1"),
            self.get_parameter("layernorm15.weight_2"),
            dim=-1,
            epsilon=0.0,
        )
        return layernorm_output_1


class Layernorm16(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "layernorm16.weight_1",
            forge.Parameter(*(768,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )
        self.add_parameter(
            "layernorm16.weight_2",
            forge.Parameter(*(768,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_parameter("layernorm16.weight_1"),
            self.get_parameter("layernorm16.weight_2"),
            dim=-1,
            epsilon=1e-06,
        )
        return layernorm_output_1


class Layernorm17(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "layernorm17.weight_1",
            forge.Parameter(*(1024,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )
        self.add_parameter(
            "layernorm17.weight_2",
            forge.Parameter(*(1024,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_parameter("layernorm17.weight_1"),
            self.get_parameter("layernorm17.weight_2"),
            dim=-1,
            epsilon=1e-06,
        )
        return layernorm_output_1


class Layernorm18(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("layernorm18_const_1", shape=(64,), dtype=torch.float32)
        self.add_constant("layernorm18_const_2", shape=(64,), dtype=torch.float32)

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_constant("layernorm18_const_1"),
            self.get_constant("layernorm18_const_2"),
            dim=-1,
            epsilon=1e-05,
        )
        return layernorm_output_1


class Layernorm19(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("layernorm19_const_1", shape=(128,), dtype=torch.float32)
        self.add_constant("layernorm19_const_2", shape=(128,), dtype=torch.float32)

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_constant("layernorm19_const_1"),
            self.get_constant("layernorm19_const_2"),
            dim=-1,
            epsilon=1e-05,
        )
        return layernorm_output_1


class Layernorm20(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("layernorm20_const_1", shape=(320,), dtype=torch.float32)
        self.add_constant("layernorm20_const_2", shape=(320,), dtype=torch.float32)

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_constant("layernorm20_const_1"),
            self.get_constant("layernorm20_const_2"),
            dim=-1,
            epsilon=1e-05,
        )
        return layernorm_output_1


class Layernorm21(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("layernorm21_const_1", shape=(512,), dtype=torch.float32)
        self.add_constant("layernorm21_const_2", shape=(512,), dtype=torch.float32)

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_constant("layernorm21_const_1"),
            self.get_constant("layernorm21_const_2"),
            dim=-1,
            epsilon=1e-05,
        )
        return layernorm_output_1


class Layernorm22(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "layernorm22.weight_1",
            forge.Parameter(*(1024,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )
        self.add_parameter(
            "layernorm22.weight_2",
            forge.Parameter(*(1024,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_parameter("layernorm22.weight_1"),
            self.get_parameter("layernorm22.weight_2"),
            dim=-1,
            epsilon=0.0,
        )
        return layernorm_output_1


class Layernorm23(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "layernorm23.weight_1",
            forge.Parameter(*(2560,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )
        self.add_parameter(
            "layernorm23.weight_2",
            forge.Parameter(*(2560,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_parameter("layernorm23.weight_1"),
            self.get_parameter("layernorm23.weight_2"),
            dim=-1,
            epsilon=1e-05,
        )
        return layernorm_output_1


class Layernorm24(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "layernorm24.weight_1",
            forge.Parameter(*(96,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )
        self.add_parameter(
            "layernorm24.weight_2",
            forge.Parameter(*(96,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_parameter("layernorm24.weight_1"),
            self.get_parameter("layernorm24.weight_2"),
            dim=-1,
            epsilon=1e-05,
        )
        return layernorm_output_1


class Layernorm25(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "layernorm25.weight_1",
            forge.Parameter(*(192,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )
        self.add_parameter(
            "layernorm25.weight_2",
            forge.Parameter(*(192,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_parameter("layernorm25.weight_1"),
            self.get_parameter("layernorm25.weight_2"),
            dim=-1,
            epsilon=1e-05,
        )
        return layernorm_output_1


class Layernorm26(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "layernorm26.weight_1",
            forge.Parameter(*(384,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )
        self.add_parameter(
            "layernorm26.weight_2",
            forge.Parameter(*(384,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_parameter("layernorm26.weight_1"),
            self.get_parameter("layernorm26.weight_2"),
            dim=-1,
            epsilon=1e-05,
        )
        return layernorm_output_1


class Layernorm27(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "layernorm27.weight_1",
            forge.Parameter(*(768,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )
        self.add_parameter(
            "layernorm27.weight_2",
            forge.Parameter(*(768,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_parameter("layernorm27.weight_1"),
            self.get_parameter("layernorm27.weight_2"),
            dim=-1,
            epsilon=1e-05,
        )
        return layernorm_output_1


class Layernorm28(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "layernorm28.weight_1",
            forge.Parameter(*(2048,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )
        self.add_parameter(
            "layernorm28.weight_2",
            forge.Parameter(*(2048,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_parameter("layernorm28.weight_1"),
            self.get_parameter("layernorm28.weight_2"),
            dim=-1,
            epsilon=0.0,
        )
        return layernorm_output_1


class Layernorm29(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "layernorm29.weight_1",
            forge.Parameter(*(192,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )
        self.add_parameter(
            "layernorm29.weight_2",
            forge.Parameter(*(192,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_parameter("layernorm29.weight_1"),
            self.get_parameter("layernorm29.weight_2"),
            dim=-1,
            epsilon=0.0,
        )
        return layernorm_output_1


class Layernorm30(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "layernorm30.weight_1",
            forge.Parameter(*(128,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )
        self.add_parameter(
            "layernorm30.weight_2",
            forge.Parameter(*(128,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
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
            forge.Parameter(*(320,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )
        self.add_parameter(
            "layernorm31.weight_2",
            forge.Parameter(*(320,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
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
            forge.Parameter(*(1024,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )
        self.add_parameter(
            "layernorm32.weight_2",
            forge.Parameter(*(1024,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
        self.add_constant("layernorm33_const_1", shape=(192,), dtype=torch.float32)
        self.add_constant("layernorm33_const_2", shape=(192,), dtype=torch.float32)

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_constant("layernorm33_const_1"),
            self.get_constant("layernorm33_const_2"),
            dim=-1,
            epsilon=0.0,
        )
        return layernorm_output_1


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (
        Layernorm0,
        [((1, 128, 128), torch.float32)],
        {
            "model_names": ["pt_albert_large_v2_token_cls_hf", "pt_albert_xlarge_v2_token_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "0.0"},
        },
    ),
    (
        Layernorm1,
        [((1, 128, 1024), torch.float32)],
        {
            "model_names": [
                "pt_albert_large_v2_token_cls_hf",
                "pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "0.0"},
        },
    ),
    (
        Layernorm2,
        [((1, 128, 768), torch.float32)],
        {
            "model_names": [
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "0.0"},
        },
    ),
    (
        Layernorm3,
        [((1, 256, 512), torch.bfloat16)],
        {
            "model_names": [
                "pt_mlp_mixer_base_img_cls_github",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm4,
        [((1, 196, 512), torch.bfloat16)],
        {
            "model_names": ["pt_mlp_mixer_mixer_s16_224_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "epsilon": "1e-06"},
        },
    ),
    (
        Layernorm5,
        [((1, 256, 2048), torch.float32)],
        {"model_names": ["pt_phi1_microsoft_phi_1_clm_hf"], "pcc": 0.99, "args": {"dim": "-1", "epsilon": "1e-05"}},
    ),
    (
        Layernorm5,
        [((1, 5, 2048), torch.float32)],
        {
            "model_names": ["pt_phi_1_5_microsoft_phi_1_5_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm6,
        [((1, 16384, 32), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm6,
        [((1, 256, 32), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm7,
        [((1, 4096, 64), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm7,
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
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm8,
        [((1, 1024, 160), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm8,
        [((1, 256, 160), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm9,
        [((1, 256, 256), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm10,
        [((1, 56, 56, 96), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm11,
        [((1, 28, 28, 384), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm12,
        [((1, 28, 28, 192), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm13,
        [((1, 14, 14, 768), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm11,
        [((1, 14, 14, 384), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm14,
        [((1, 7, 7, 1536), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm13,
        [((1, 7, 7, 768), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm15,
        [((1, 197, 768), torch.bfloat16)],
        {
            "model_names": [
                "pt_vit_google_vit_base_patch16_224_img_cls_hf",
                "pt_beit_microsoft_beit_base_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "epsilon": "0.0"},
        },
    ),
    (
        Layernorm16,
        [((1, 197, 768), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_b_16_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "epsilon": "1e-06"},
        },
    ),
    (
        Layernorm17,
        [((1, 50, 1024), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_l_32_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "epsilon": "1e-06"},
        },
    ),
    (
        Layernorm18,
        [((1, 16384, 64), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm18,
        [((1, 256, 64), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm19,
        [((1, 4096, 128), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm19,
        [((1, 256, 128), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm20,
        [((1, 1024, 320), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm20,
        [((1, 256, 320), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm21,
        [((1, 256, 512), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm22,
        [((1, 197, 1024), torch.bfloat16)],
        {
            "model_names": ["pt_beit_microsoft_beit_large_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "epsilon": "0.0"},
        },
    ),
    (
        Layernorm17,
        [((1, 197, 1024), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_l_16_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "epsilon": "1e-06"},
        },
    ),
    (
        Layernorm22,
        [((1, 1024), torch.bfloat16)],
        {
            "model_names": ["pt_beit_microsoft_beit_large_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "epsilon": "0.0"},
        },
    ),
    (
        Layernorm16,
        [((1, 196, 768), torch.bfloat16)],
        {
            "model_names": [
                "pt_mlp_mixer_mixer_b16_224_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_miil_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "epsilon": "1e-06"},
        },
    ),
    (
        Layernorm4,
        [((1, 49, 512), torch.bfloat16)],
        {
            "model_names": ["pt_mlp_mixer_mixer_s32_224_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "epsilon": "1e-06"},
        },
    ),
    (
        Layernorm23,
        [((1, 11, 2560), torch.float32)],
        {
            "model_names": ["pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm24,
        [((1, 64, 64, 96), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm25,
        [((1, 32, 32, 192), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm26,
        [((1, 16, 16, 384), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm27,
        [((1, 8, 8, 768), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm28,
        [((1, 128, 2048), torch.float32)],
        {"model_names": ["pt_albert_xlarge_v2_token_cls_hf"], "pcc": 0.99, "args": {"dim": "-1", "epsilon": "0.0"}},
    ),
    (
        Layernorm15,
        [((1, 768), torch.bfloat16)],
        {
            "model_names": ["pt_beit_microsoft_beit_base_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "epsilon": "0.0"},
        },
    ),
    (
        Layernorm29,
        [((1, 197, 192), torch.bfloat16)],
        {
            "model_names": ["pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "epsilon": "0.0"},
        },
    ),
    (
        Layernorm17,
        [((1, 196, 1024), torch.bfloat16)],
        {
            "model_names": ["pt_mlp_mixer_mixer_l16_224_in21k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "epsilon": "1e-06"},
        },
    ),
    (
        Layernorm7,
        [((1, 16384, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm30,
        [((1, 4096, 128), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm30,
        [((1, 256, 128), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm31,
        [((1, 1024, 320), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm31,
        [((1, 256, 320), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm32,
        [((1, 256, 1024), torch.float32)],
        {
            "model_names": ["pt_xglm_facebook_xglm_564m_clm_hf", "pt_codegen_salesforce_codegen_350m_nl_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm33,
        [((1, 197, 192), torch.float32)],
        {
            "model_names": ["onnx_deit_facebook_deit_tiny_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "0.0"},
        },
    ),
    (
        Layernorm27,
        [((1, 256, 768), torch.float32)],
        {
            "model_names": ["pt_gpt_gpt2_text_gen_hf", "pt_opt_facebook_opt_125m_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm27,
        [((256, 768), torch.float32)],
        {"model_names": ["pt_opt_facebook_opt_125m_clm_hf"], "pcc": 0.99, "args": {"dim": "-1", "epsilon": "1e-05"}},
    ),
    (
        Layernorm27,
        [((1, 32, 768), torch.float32)],
        {
            "model_names": ["pt_opt_facebook_opt_125m_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm27,
        [((32, 768), torch.float32)],
        {
            "model_names": ["pt_opt_facebook_opt_125m_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm5,
        [((1, 12, 2048), torch.float32)],
        {
            "model_names": ["pt_phi1_microsoft_phi_1_token_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
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
