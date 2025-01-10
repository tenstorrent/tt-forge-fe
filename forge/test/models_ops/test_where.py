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


class Where0(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("where0_const_1", shape=(1,), dtype=torch.float32)
        self.add_constant("where0_const_2", shape=(1,), dtype=torch.float32)

    def forward(self, where_input_0):
        where_output_1 = forge.op.Where(
            "", where_input_0, self.get_constant("where0_const_1"), self.get_constant("where0_const_2")
        )
        return where_output_1


class Where1(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("where1_const_2", shape=(1, 256, 10, 32), dtype=torch.float32)

    def forward(self, where_input_0, where_input_1):
        where_output_1 = forge.op.Where("", where_input_0, where_input_1, self.get_constant("where1_const_2"))
        return where_output_1


class Where2(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, where_input_0, where_input_1, where_input_2):
        where_output_1 = forge.op.Where("", where_input_0, where_input_1, where_input_2)
        return where_output_1


class Where3(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("where3_const_2", shape=(1, 256, 20, 64), dtype=torch.float32)

    def forward(self, where_input_0, where_input_1):
        where_output_1 = forge.op.Where("", where_input_0, where_input_1, self.get_constant("where3_const_2"))
        return where_output_1


class Where4(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("where4_const_2", shape=(1, 128, 20, 64), dtype=torch.float32)

    def forward(self, where_input_0, where_input_1):
        where_output_1 = forge.op.Where("", where_input_0, where_input_1, self.get_constant("where4_const_2"))
        return where_output_1


class Where5(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("where5_const_2", shape=(1, 128, 40, 128), dtype=torch.float32)

    def forward(self, where_input_0, where_input_1):
        where_output_1 = forge.op.Where("", where_input_0, where_input_1, self.get_constant("where5_const_2"))
        return where_output_1


class Where6(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("where6_const_2", shape=(1, 64, 40, 128), dtype=torch.float32)

    def forward(self, where_input_0, where_input_1):
        where_output_1 = forge.op.Where("", where_input_0, where_input_1, self.get_constant("where6_const_2"))
        return where_output_1


class Where7(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("where7_const_2", shape=(1, 64, 80, 256), dtype=torch.float32)

    def forward(self, where_input_0, where_input_1):
        where_output_1 = forge.op.Where("", where_input_0, where_input_1, self.get_constant("where7_const_2"))
        return where_output_1


class Where8(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("where8_const_2", shape=(1, 32, 80, 256), dtype=torch.float32)

    def forward(self, where_input_0, where_input_1):
        where_output_1 = forge.op.Where("", where_input_0, where_input_1, self.get_constant("where8_const_2"))
        return where_output_1


class Where9(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("where9_const_2", shape=(1, 32, 160, 512), dtype=torch.float32)

    def forward(self, where_input_0, where_input_1):
        where_output_1 = forge.op.Where("", where_input_0, where_input_1, self.get_constant("where9_const_2"))
        return where_output_1


class Where10(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("where10_const_2", shape=(1, 16, 160, 512), dtype=torch.float32)

    def forward(self, where_input_0, where_input_1):
        where_output_1 = forge.op.Where("", where_input_0, where_input_1, self.get_constant("where10_const_2"))
        return where_output_1


class Where11(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("where11_const_2", shape=(1, 16, 320, 1024), dtype=torch.float32)

    def forward(self, where_input_0, where_input_1):
        where_output_1 = forge.op.Where("", where_input_0, where_input_1, self.get_constant("where11_const_2"))
        return where_output_1


class Where12(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("where12_const_2", shape=(1, 256, 6, 20), dtype=torch.float32)

    def forward(self, where_input_0, where_input_1):
        where_output_1 = forge.op.Where("", where_input_0, where_input_1, self.get_constant("where12_const_2"))
        return where_output_1


class Where13(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("where13_const_2", shape=(1, 256, 12, 40), dtype=torch.float32)

    def forward(self, where_input_0, where_input_1):
        where_output_1 = forge.op.Where("", where_input_0, where_input_1, self.get_constant("where13_const_2"))
        return where_output_1


class Where14(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("where14_const_2", shape=(1, 128, 12, 40), dtype=torch.float32)

    def forward(self, where_input_0, where_input_1):
        where_output_1 = forge.op.Where("", where_input_0, where_input_1, self.get_constant("where14_const_2"))
        return where_output_1


class Where15(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("where15_const_2", shape=(1, 128, 24, 80), dtype=torch.float32)

    def forward(self, where_input_0, where_input_1):
        where_output_1 = forge.op.Where("", where_input_0, where_input_1, self.get_constant("where15_const_2"))
        return where_output_1


class Where16(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("where16_const_2", shape=(1, 64, 24, 80), dtype=torch.float32)

    def forward(self, where_input_0, where_input_1):
        where_output_1 = forge.op.Where("", where_input_0, where_input_1, self.get_constant("where16_const_2"))
        return where_output_1


class Where17(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("where17_const_2", shape=(1, 64, 48, 160), dtype=torch.float32)

    def forward(self, where_input_0, where_input_1):
        where_output_1 = forge.op.Where("", where_input_0, where_input_1, self.get_constant("where17_const_2"))
        return where_output_1


class Where18(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("where18_const_2", shape=(1, 32, 48, 160), dtype=torch.float32)

    def forward(self, where_input_0, where_input_1):
        where_output_1 = forge.op.Where("", where_input_0, where_input_1, self.get_constant("where18_const_2"))
        return where_output_1


class Where19(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("where19_const_2", shape=(1, 32, 96, 320), dtype=torch.float32)

    def forward(self, where_input_0, where_input_1):
        where_output_1 = forge.op.Where("", where_input_0, where_input_1, self.get_constant("where19_const_2"))
        return where_output_1


class Where20(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("where20_const_2", shape=(1, 16, 96, 320), dtype=torch.float32)

    def forward(self, where_input_0, where_input_1):
        where_output_1 = forge.op.Where("", where_input_0, where_input_1, self.get_constant("where20_const_2"))
        return where_output_1


class Where21(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("where21_const_2", shape=(1, 16, 192, 640), dtype=torch.float32)

    def forward(self, where_input_0, where_input_1):
        where_output_1 = forge.op.Where("", where_input_0, where_input_1, self.get_constant("where21_const_2"))
        return where_output_1


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (Where0, [((1, 1, 256, 256), torch.float32)]),
    (Where0, [((1, 1, 32, 32), torch.float32)]),
    (Where1, [((1, 256, 10, 32), torch.float32), ((1, 256, 10, 32), torch.float32)]),
    (Where2, [((1, 256, 10, 32), torch.float32), ((1, 256, 10, 32), torch.float32), ((1, 256, 10, 32), torch.float32)]),
    (Where3, [((1, 256, 20, 64), torch.float32), ((1, 256, 20, 64), torch.float32)]),
    (Where2, [((1, 256, 20, 64), torch.float32), ((1, 256, 20, 64), torch.float32), ((1, 256, 20, 64), torch.float32)]),
    (Where4, [((1, 128, 20, 64), torch.float32), ((1, 128, 20, 64), torch.float32)]),
    (Where2, [((1, 128, 20, 64), torch.float32), ((1, 128, 20, 64), torch.float32), ((1, 128, 20, 64), torch.float32)]),
    (Where5, [((1, 128, 40, 128), torch.float32), ((1, 128, 40, 128), torch.float32)]),
    (
        Where2,
        [((1, 128, 40, 128), torch.float32), ((1, 128, 40, 128), torch.float32), ((1, 128, 40, 128), torch.float32)],
    ),
    (Where6, [((1, 64, 40, 128), torch.float32), ((1, 64, 40, 128), torch.float32)]),
    (Where2, [((1, 64, 40, 128), torch.float32), ((1, 64, 40, 128), torch.float32), ((1, 64, 40, 128), torch.float32)]),
    (Where7, [((1, 64, 80, 256), torch.float32), ((1, 64, 80, 256), torch.float32)]),
    (Where2, [((1, 64, 80, 256), torch.float32), ((1, 64, 80, 256), torch.float32), ((1, 64, 80, 256), torch.float32)]),
    (Where8, [((1, 32, 80, 256), torch.float32), ((1, 32, 80, 256), torch.float32)]),
    (Where2, [((1, 32, 80, 256), torch.float32), ((1, 32, 80, 256), torch.float32), ((1, 32, 80, 256), torch.float32)]),
    (Where9, [((1, 32, 160, 512), torch.float32), ((1, 32, 160, 512), torch.float32)]),
    (
        Where2,
        [((1, 32, 160, 512), torch.float32), ((1, 32, 160, 512), torch.float32), ((1, 32, 160, 512), torch.float32)],
    ),
    (Where10, [((1, 16, 160, 512), torch.float32), ((1, 16, 160, 512), torch.float32)]),
    (
        Where2,
        [((1, 16, 160, 512), torch.float32), ((1, 16, 160, 512), torch.float32), ((1, 16, 160, 512), torch.float32)],
    ),
    (Where11, [((1, 16, 320, 1024), torch.float32), ((1, 16, 320, 1024), torch.float32)]),
    (
        Where2,
        [((1, 16, 320, 1024), torch.float32), ((1, 16, 320, 1024), torch.float32), ((1, 16, 320, 1024), torch.float32)],
    ),
    (Where12, [((1, 256, 6, 20), torch.float32), ((1, 256, 6, 20), torch.float32)]),
    (Where2, [((1, 256, 6, 20), torch.float32), ((1, 256, 6, 20), torch.float32), ((1, 256, 6, 20), torch.float32)]),
    (Where13, [((1, 256, 12, 40), torch.float32), ((1, 256, 12, 40), torch.float32)]),
    (Where2, [((1, 256, 12, 40), torch.float32), ((1, 256, 12, 40), torch.float32), ((1, 256, 12, 40), torch.float32)]),
    (Where14, [((1, 128, 12, 40), torch.float32), ((1, 128, 12, 40), torch.float32)]),
    (Where2, [((1, 128, 12, 40), torch.float32), ((1, 128, 12, 40), torch.float32), ((1, 128, 12, 40), torch.float32)]),
    (Where15, [((1, 128, 24, 80), torch.float32), ((1, 128, 24, 80), torch.float32)]),
    (Where2, [((1, 128, 24, 80), torch.float32), ((1, 128, 24, 80), torch.float32), ((1, 128, 24, 80), torch.float32)]),
    (Where16, [((1, 64, 24, 80), torch.float32), ((1, 64, 24, 80), torch.float32)]),
    (Where2, [((1, 64, 24, 80), torch.float32), ((1, 64, 24, 80), torch.float32), ((1, 64, 24, 80), torch.float32)]),
    (Where17, [((1, 64, 48, 160), torch.float32), ((1, 64, 48, 160), torch.float32)]),
    (Where2, [((1, 64, 48, 160), torch.float32), ((1, 64, 48, 160), torch.float32), ((1, 64, 48, 160), torch.float32)]),
    (Where18, [((1, 32, 48, 160), torch.float32), ((1, 32, 48, 160), torch.float32)]),
    (Where2, [((1, 32, 48, 160), torch.float32), ((1, 32, 48, 160), torch.float32), ((1, 32, 48, 160), torch.float32)]),
    (Where19, [((1, 32, 96, 320), torch.float32), ((1, 32, 96, 320), torch.float32)]),
    (Where2, [((1, 32, 96, 320), torch.float32), ((1, 32, 96, 320), torch.float32), ((1, 32, 96, 320), torch.float32)]),
    (Where20, [((1, 16, 96, 320), torch.float32), ((1, 16, 96, 320), torch.float32)]),
    (Where2, [((1, 16, 96, 320), torch.float32), ((1, 16, 96, 320), torch.float32), ((1, 16, 96, 320), torch.float32)]),
    (Where21, [((1, 16, 192, 640), torch.float32), ((1, 16, 192, 640), torch.float32)]),
    (
        Where2,
        [((1, 16, 192, 640), torch.float32), ((1, 16, 192, 640), torch.float32), ((1, 16, 192, 640), torch.float32)],
    ),
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
