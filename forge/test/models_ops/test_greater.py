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


class Greater0(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("greater0_const_1", shape=(1,), dtype=torch.float32)

    def forward(self, greater_input_0):
        greater_output_1 = forge.op.Greater("", greater_input_0, self.get_constant("greater0_const_1"))
        return greater_output_1


class Greater1(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("greater1_const_1", shape=(1, 256, 10, 32), dtype=torch.float32)

    def forward(self, greater_input_0):
        greater_output_1 = forge.op.Greater("", greater_input_0, self.get_constant("greater1_const_1"))
        return greater_output_1


class Greater2(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("greater2_const_1", shape=(1, 256, 20, 64), dtype=torch.float32)

    def forward(self, greater_input_0):
        greater_output_1 = forge.op.Greater("", greater_input_0, self.get_constant("greater2_const_1"))
        return greater_output_1


class Greater3(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("greater3_const_1", shape=(1, 128, 20, 64), dtype=torch.float32)

    def forward(self, greater_input_0):
        greater_output_1 = forge.op.Greater("", greater_input_0, self.get_constant("greater3_const_1"))
        return greater_output_1


class Greater4(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("greater4_const_1", shape=(1, 128, 40, 128), dtype=torch.float32)

    def forward(self, greater_input_0):
        greater_output_1 = forge.op.Greater("", greater_input_0, self.get_constant("greater4_const_1"))
        return greater_output_1


class Greater5(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("greater5_const_1", shape=(1, 64, 40, 128), dtype=torch.float32)

    def forward(self, greater_input_0):
        greater_output_1 = forge.op.Greater("", greater_input_0, self.get_constant("greater5_const_1"))
        return greater_output_1


class Greater6(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("greater6_const_1", shape=(1, 64, 80, 256), dtype=torch.float32)

    def forward(self, greater_input_0):
        greater_output_1 = forge.op.Greater("", greater_input_0, self.get_constant("greater6_const_1"))
        return greater_output_1


class Greater7(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("greater7_const_1", shape=(1, 32, 80, 256), dtype=torch.float32)

    def forward(self, greater_input_0):
        greater_output_1 = forge.op.Greater("", greater_input_0, self.get_constant("greater7_const_1"))
        return greater_output_1


class Greater8(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("greater8_const_1", shape=(1, 32, 160, 512), dtype=torch.float32)

    def forward(self, greater_input_0):
        greater_output_1 = forge.op.Greater("", greater_input_0, self.get_constant("greater8_const_1"))
        return greater_output_1


class Greater9(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("greater9_const_1", shape=(1, 16, 160, 512), dtype=torch.float32)

    def forward(self, greater_input_0):
        greater_output_1 = forge.op.Greater("", greater_input_0, self.get_constant("greater9_const_1"))
        return greater_output_1


class Greater10(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("greater10_const_1", shape=(1, 16, 320, 1024), dtype=torch.float32)

    def forward(self, greater_input_0):
        greater_output_1 = forge.op.Greater("", greater_input_0, self.get_constant("greater10_const_1"))
        return greater_output_1


class Greater11(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("greater11_const_1", shape=(1, 256, 6, 20), dtype=torch.float32)

    def forward(self, greater_input_0):
        greater_output_1 = forge.op.Greater("", greater_input_0, self.get_constant("greater11_const_1"))
        return greater_output_1


class Greater12(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("greater12_const_1", shape=(1, 256, 12, 40), dtype=torch.float32)

    def forward(self, greater_input_0):
        greater_output_1 = forge.op.Greater("", greater_input_0, self.get_constant("greater12_const_1"))
        return greater_output_1


class Greater13(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("greater13_const_1", shape=(1, 128, 12, 40), dtype=torch.float32)

    def forward(self, greater_input_0):
        greater_output_1 = forge.op.Greater("", greater_input_0, self.get_constant("greater13_const_1"))
        return greater_output_1


class Greater14(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("greater14_const_1", shape=(1, 128, 24, 80), dtype=torch.float32)

    def forward(self, greater_input_0):
        greater_output_1 = forge.op.Greater("", greater_input_0, self.get_constant("greater14_const_1"))
        return greater_output_1


class Greater15(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("greater15_const_1", shape=(1, 64, 24, 80), dtype=torch.float32)

    def forward(self, greater_input_0):
        greater_output_1 = forge.op.Greater("", greater_input_0, self.get_constant("greater15_const_1"))
        return greater_output_1


class Greater16(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("greater16_const_1", shape=(1, 64, 48, 160), dtype=torch.float32)

    def forward(self, greater_input_0):
        greater_output_1 = forge.op.Greater("", greater_input_0, self.get_constant("greater16_const_1"))
        return greater_output_1


class Greater17(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("greater17_const_1", shape=(1, 32, 48, 160), dtype=torch.float32)

    def forward(self, greater_input_0):
        greater_output_1 = forge.op.Greater("", greater_input_0, self.get_constant("greater17_const_1"))
        return greater_output_1


class Greater18(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("greater18_const_1", shape=(1, 32, 96, 320), dtype=torch.float32)

    def forward(self, greater_input_0):
        greater_output_1 = forge.op.Greater("", greater_input_0, self.get_constant("greater18_const_1"))
        return greater_output_1


class Greater19(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("greater19_const_1", shape=(1, 16, 96, 320), dtype=torch.float32)

    def forward(self, greater_input_0):
        greater_output_1 = forge.op.Greater("", greater_input_0, self.get_constant("greater19_const_1"))
        return greater_output_1


class Greater20(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("greater20_const_1", shape=(1, 16, 192, 640), dtype=torch.float32)

    def forward(self, greater_input_0):
        greater_output_1 = forge.op.Greater("", greater_input_0, self.get_constant("greater20_const_1"))
        return greater_output_1


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (Greater0, [((2, 1, 1, 13), torch.float32)]),
    (Greater0, [((2, 1, 7, 7), torch.float32)]),
    (Greater0, [((1, 1, 256, 256), torch.float32)]),
    (Greater0, [((1, 12, 128, 128), torch.float32)]),
    (Greater0, [((1, 12, 384, 384), torch.float32)]),
    (Greater0, [((1, 1, 32, 32), torch.float32)]),
    (Greater1, [((1, 256, 10, 32), torch.float32)]),
    (Greater2, [((1, 256, 20, 64), torch.float32)]),
    (Greater3, [((1, 128, 20, 64), torch.float32)]),
    (Greater4, [((1, 128, 40, 128), torch.float32)]),
    (Greater5, [((1, 64, 40, 128), torch.float32)]),
    (Greater6, [((1, 64, 80, 256), torch.float32)]),
    (Greater7, [((1, 32, 80, 256), torch.float32)]),
    (Greater8, [((1, 32, 160, 512), torch.float32)]),
    (Greater9, [((1, 16, 160, 512), torch.float32)]),
    (Greater10, [((1, 16, 320, 1024), torch.float32)]),
    (Greater11, [((1, 256, 6, 20), torch.float32)]),
    (Greater12, [((1, 256, 12, 40), torch.float32)]),
    (Greater13, [((1, 128, 12, 40), torch.float32)]),
    (Greater14, [((1, 128, 24, 80), torch.float32)]),
    (Greater15, [((1, 64, 24, 80), torch.float32)]),
    (Greater16, [((1, 64, 48, 160), torch.float32)]),
    (Greater17, [((1, 32, 48, 160), torch.float32)]),
    (Greater18, [((1, 32, 96, 320), torch.float32)]),
    (Greater19, [((1, 16, 96, 320), torch.float32)]),
    (Greater20, [((1, 16, 192, 640), torch.float32)]),
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
