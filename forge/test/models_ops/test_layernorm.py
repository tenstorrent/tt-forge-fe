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
from forge.verify.value_checkers import AutomaticValueChecker
from forge.verify.config import VerifyConfig
import pytest


class Layernorm0(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "layernorm0.weight_1",
            forge.Parameter(*(1024,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )
        self.add_parameter(
            "layernorm0.weight_2",
            forge.Parameter(*(1024,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_parameter("layernorm0.weight_1"),
            self.get_parameter("layernorm0.weight_2"),
            dim=-1,
            epsilon=1e-05,
        )
        return layernorm_output_1


class Layernorm1(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "layernorm1.weight_1",
            forge.Parameter(*(2048,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )
        self.add_parameter(
            "layernorm1.weight_2",
            forge.Parameter(*(2048,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_parameter("layernorm1.weight_1"),
            self.get_parameter("layernorm1.weight_2"),
            dim=-1,
            epsilon=1e-05,
        )
        return layernorm_output_1


class Layernorm2(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "layernorm2.weight_1",
            forge.Parameter(*(1536,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )
        self.add_parameter(
            "layernorm2.weight_2",
            forge.Parameter(*(1536,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_parameter("layernorm2.weight_1"),
            self.get_parameter("layernorm2.weight_2"),
            dim=-1,
            epsilon=1e-05,
        )
        return layernorm_output_1


class Layernorm3(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "layernorm3.weight_1",
            forge.Parameter(*(512,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )
        self.add_parameter(
            "layernorm3.weight_2",
            forge.Parameter(*(512,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(1280,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )
        self.add_parameter(
            "layernorm4.weight_2",
            forge.Parameter(*(1280,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_parameter("layernorm4.weight_1"),
            self.get_parameter("layernorm4.weight_2"),
            dim=-1,
            epsilon=1e-05,
        )
        return layernorm_output_1


class Layernorm5(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "layernorm5.weight_1",
            forge.Parameter(*(768,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )
        self.add_parameter(
            "layernorm5.weight_2",
            forge.Parameter(*(768,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(384,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )
        self.add_parameter(
            "layernorm6.weight_2",
            forge.Parameter(*(384,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(768,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )
        self.add_parameter(
            "layernorm7.weight_2",
            forge.Parameter(*(768,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_parameter("layernorm7.weight_1"),
            self.get_parameter("layernorm7.weight_2"),
            dim=-1,
            epsilon=0.0,
        )
        return layernorm_output_1


class Layernorm8(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "layernorm8.weight_1",
            forge.Parameter(*(128,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )
        self.add_parameter(
            "layernorm8.weight_2",
            forge.Parameter(*(128,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_parameter("layernorm8.weight_1"),
            self.get_parameter("layernorm8.weight_2"),
            dim=-1,
            epsilon=0.0,
        )
        return layernorm_output_1


class Layernorm9(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "layernorm9.weight_1",
            forge.Parameter(*(2048,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )
        self.add_parameter(
            "layernorm9.weight_2",
            forge.Parameter(*(2048,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_parameter("layernorm9.weight_1"),
            self.get_parameter("layernorm9.weight_2"),
            dim=-1,
            epsilon=0.0,
        )
        return layernorm_output_1


class Layernorm10(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "layernorm10.weight_1",
            forge.Parameter(*(1024,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )
        self.add_parameter(
            "layernorm10.weight_2",
            forge.Parameter(*(1024,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_parameter("layernorm10.weight_1"),
            self.get_parameter("layernorm10.weight_2"),
            dim=-1,
            epsilon=0.0,
        )
        return layernorm_output_1


class Layernorm11(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "layernorm11.weight_1",
            forge.Parameter(*(4096,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )
        self.add_parameter(
            "layernorm11.weight_2",
            forge.Parameter(*(4096,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_parameter("layernorm11.weight_1"),
            self.get_parameter("layernorm11.weight_2"),
            dim=-1,
            epsilon=0.0,
        )
        return layernorm_output_1


class Layernorm12(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "layernorm12.weight_1",
            forge.Parameter(*(4544,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )
        self.add_parameter(
            "layernorm12.weight_2",
            forge.Parameter(*(4544,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(4096,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )
        self.add_parameter(
            "layernorm13.weight_2",
            forge.Parameter(*(4096,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(64,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )
        self.add_parameter(
            "layernorm14.weight_2",
            forge.Parameter(*(64,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(2560,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )
        self.add_parameter(
            "layernorm15.weight_2",
            forge.Parameter(*(2560,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(192,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )
        self.add_parameter(
            "layernorm16.weight_2",
            forge.Parameter(*(192,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_parameter("layernorm16.weight_1"),
            self.get_parameter("layernorm16.weight_2"),
            dim=-1,
            epsilon=0.0,
        )
        return layernorm_output_1


class Layernorm17(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "layernorm17.weight_1",
            forge.Parameter(*(384,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )
        self.add_parameter(
            "layernorm17.weight_2",
            forge.Parameter(*(384,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_parameter("layernorm17.weight_1"),
            self.get_parameter("layernorm17.weight_2"),
            dim=-1,
            epsilon=0.0,
        )
        return layernorm_output_1


class Layernorm18(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "layernorm18.weight_1",
            forge.Parameter(*(768,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )
        self.add_parameter(
            "layernorm18.weight_2",
            forge.Parameter(*(768,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_parameter("layernorm18.weight_1"),
            self.get_parameter("layernorm18.weight_2"),
            dim=-1,
            epsilon=1e-06,
        )
        return layernorm_output_1


class Layernorm19(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "layernorm19.weight_1",
            forge.Parameter(*(1024,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )
        self.add_parameter(
            "layernorm19.weight_2",
            forge.Parameter(*(1024,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_parameter("layernorm19.weight_1"),
            self.get_parameter("layernorm19.weight_2"),
            dim=-1,
            epsilon=1e-06,
        )
        return layernorm_output_1


class Layernorm20(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "layernorm20.weight_1",
            forge.Parameter(*(512,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )
        self.add_parameter(
            "layernorm20.weight_2",
            forge.Parameter(*(512,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_parameter("layernorm20.weight_1"),
            self.get_parameter("layernorm20.weight_2"),
            dim=-1,
            epsilon=1e-06,
        )
        return layernorm_output_1


class Layernorm21(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "layernorm21.weight_1",
            forge.Parameter(*(322,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )
        self.add_parameter(
            "layernorm21.weight_2",
            forge.Parameter(*(322,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_parameter("layernorm21.weight_1"),
            self.get_parameter("layernorm21.weight_2"),
            dim=-1,
            epsilon=1e-05,
        )
        return layernorm_output_1


class Layernorm22(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "layernorm22.weight_1",
            forge.Parameter(*(261,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )
        self.add_parameter(
            "layernorm22.weight_2",
            forge.Parameter(*(261,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_parameter("layernorm22.weight_1"),
            self.get_parameter("layernorm22.weight_2"),
            dim=-1,
            epsilon=1e-05,
        )
        return layernorm_output_1


class Layernorm23(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "layernorm23.weight_1",
            forge.Parameter(*(32,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )
        self.add_parameter(
            "layernorm23.weight_2",
            forge.Parameter(*(32,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(160,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )
        self.add_parameter(
            "layernorm24.weight_2",
            forge.Parameter(*(160,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(256,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )
        self.add_parameter(
            "layernorm25.weight_2",
            forge.Parameter(*(256,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(128,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )
        self.add_parameter(
            "layernorm26.weight_2",
            forge.Parameter(*(128,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(320,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )
        self.add_parameter(
            "layernorm27.weight_2",
            forge.Parameter(*(320,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (Layernorm0, [((2, 1, 1024), torch.float32)]),
    (Layernorm1, [((2, 1, 2048), torch.float32)]),
    (Layernorm2, [((2, 1, 1536), torch.float32)]),
    (Layernorm0, [((1, 2, 1024), torch.float32)]),
    (Layernorm3, [((1, 2, 512), torch.float32)]),
    (Layernorm4, [((1, 2, 1280), torch.float32)]),
    (Layernorm5, [((1, 2, 768), torch.float32)]),
    (Layernorm6, [((1, 2, 384), torch.float32)]),
    (Layernorm3, [((2, 7, 512), torch.float32)]),
    (Layernorm7, [((1, 204, 768), torch.float32)]),
    (Layernorm7, [((1, 11, 768), torch.float32)]),
    (Layernorm7, [((1, 201, 768), torch.float32)]),
    (Layernorm2, [((1, 1536), torch.float32)]),
    (Layernorm8, [((1, 128, 128), torch.float32)]),
    (Layernorm9, [((1, 128, 2048), torch.float32)]),
    (Layernorm10, [((1, 128, 1024), torch.float32)]),
    (Layernorm11, [((1, 128, 4096), torch.float32)]),
    (Layernorm7, [((1, 128, 768), torch.float32)]),
    (Layernorm5, [((1, 128, 768), torch.float32)]),
    (Layernorm0, [((1, 256, 1024), torch.float32)]),
    (Layernorm10, [((1, 384, 1024), torch.float32)]),
    (Layernorm7, [((1, 384, 768), torch.float32)]),
    (Layernorm12, [((1, 6, 4544), torch.float32)]),
    (Layernorm13, [((1, 334, 4096), torch.float32)]),
    (Layernorm14, [((1, 334, 64, 64), torch.float32)]),
    (Layernorm5, [((1, 256, 768), torch.float32)]),
    (Layernorm5, [((1, 32, 768), torch.float32)]),
    (Layernorm1, [((1, 32, 2048), torch.float32)]),
    (Layernorm1, [((1, 256, 2048), torch.float32)]),
    (Layernorm15, [((1, 256, 2560), torch.float32)]),
    (Layernorm15, [((1, 32, 2560), torch.float32)]),
    (Layernorm1, [((32, 2048), torch.float32)]),
    (Layernorm1, [((256, 2048), torch.float32)]),
    (Layernorm0, [((1, 32, 1024), torch.float32)]),
    (Layernorm5, [((32, 768), torch.float32)]),
    (Layernorm5, [((256, 768), torch.float32)]),
    (Layernorm15, [((1, 12, 2560), torch.float32)]),
    (Layernorm15, [((1, 11, 2560), torch.float32)]),
    (Layernorm7, [((1, 197, 768), torch.float32)]),
    (Layernorm16, [((1, 197, 192), torch.float32)]),
    (Layernorm17, [((1, 197, 384), torch.float32)]),
    (Layernorm18, [((1, 49, 768), torch.float32)]),
    (Layernorm19, [((1, 49, 1024), torch.float32)]),
    (Layernorm18, [((1, 196, 768), torch.float32)]),
    (Layernorm20, [((1, 49, 512), torch.float32)]),
    (Layernorm20, [((1, 196, 512), torch.float32)]),
    (Layernorm19, [((1, 196, 1024), torch.float32)]),
    (Layernorm0, [((1, 1, 1024), torch.float32)]),
    (Layernorm0, [((1, 512, 1024), torch.float32)]),
    (Layernorm21, [((1, 3025, 322), torch.float32)]),
    (Layernorm3, [((1, 50176, 512), torch.float32)]),
    (Layernorm22, [((1, 50176, 261), torch.float32)]),
    (Layernorm23, [((1, 16384, 32), torch.float32)]),
    (Layernorm23, [((1, 256, 32), torch.float32)]),
    (Layernorm14, [((1, 4096, 64), torch.float32)]),
    (Layernorm14, [((1, 256, 64), torch.float32)]),
    (Layernorm24, [((1, 1024, 160), torch.float32)]),
    (Layernorm24, [((1, 256, 160), torch.float32)]),
    (Layernorm25, [((1, 256, 256), torch.float32)]),
    (Layernorm14, [((1, 16384, 64), torch.float32)]),
    (Layernorm26, [((1, 4096, 128), torch.float32)]),
    (Layernorm26, [((1, 256, 128), torch.float32)]),
    (Layernorm27, [((1, 1024, 320), torch.float32)]),
    (Layernorm27, [((1, 256, 320), torch.float32)]),
    (Layernorm3, [((1, 256, 512), torch.float32)]),
    (Layernorm10, [((1, 197, 1024), torch.float32)]),
]


@pytest.mark.push
@pytest.mark.parametrize("forge_module_and_shapes_dtypes", forge_modules_and_shapes_dtypes_list, ids=ids_func)
def test_module(forge_module_and_shapes_dtypes):

    forge_module, operand_shapes_dtypes = forge_module_and_shapes_dtypes

    inputs = [
        Tensor.create_from_shape(operand_shape, operand_dtype) for operand_shape, operand_dtype in operand_shapes_dtypes
    ]

    framework_model = forge_module(forge_module.__name__)
    framework_model.process_framework_parameters()

    for name, parameter in framework_model._parameters.items():
        parameter_tensor = Tensor.create_torch_tensor(
            shape=parameter.shape.get_pytorch_shape(), dtype=parameter.pt_data_format
        )
        framework_model.set_parameter(name, parameter_tensor)

    for name, constant in framework_model._constants.items():
        constant_tensor = Tensor.create_torch_tensor(
            shape=constant.shape.get_pytorch_shape(), dtype=constant.pt_data_format
        )
        framework_model.set_constant(name, constant_tensor)

    compiled_model = compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)
