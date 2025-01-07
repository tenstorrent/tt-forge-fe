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


class Reshape0(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(128, 2048))
        return reshape_output_1


class Reshape1(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 16, 128))
        return reshape_output_1


class Reshape2(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 2048))
        return reshape_output_1


class Reshape3(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 128, 128))
        return reshape_output_1


class Reshape4(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 128, 128))
        return reshape_output_1


class Reshape5(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 2048, 1))
        return reshape_output_1


class Reshape6(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(128, 1024))
        return reshape_output_1


class Reshape7(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 16, 64))
        return reshape_output_1


class Reshape8(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 1024))
        return reshape_output_1


class Reshape9(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 128, 64))
        return reshape_output_1


class Reshape10(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 64, 128))
        return reshape_output_1


class Reshape11(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 128, 64))
        return reshape_output_1


class Reshape12(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 1024, 1))
        return reshape_output_1


class Reshape13(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(128, 4096))
        return reshape_output_1


class Reshape14(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 64, 64))
        return reshape_output_1


class Reshape15(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 4096))
        return reshape_output_1


class Reshape16(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 128, 64))
        return reshape_output_1


class Reshape17(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 64, 128, 128))
        return reshape_output_1


class Reshape18(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 128, 128))
        return reshape_output_1


class Reshape19(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 64, 128))
        return reshape_output_1


class Reshape20(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 64, 128, 64))
        return reshape_output_1


class Reshape21(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 4096, 1))
        return reshape_output_1


class Reshape22(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(128, 768))
        return reshape_output_1


class Reshape23(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 12, 64))
        return reshape_output_1


class Reshape24(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 768))
        return reshape_output_1


class Reshape25(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 128, 64))
        return reshape_output_1


class Reshape26(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 128, 128))
        return reshape_output_1


class Reshape27(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 128, 128))
        return reshape_output_1


class Reshape28(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 64, 128))
        return reshape_output_1


class Reshape29(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 128, 64))
        return reshape_output_1


class Reshape30(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 768, 1))
        return reshape_output_1


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (Reshape0, [((1, 128, 2048), torch.float32)]),
    (Reshape1, [((1, 128, 2048), torch.float32)]),
    (Reshape2, [((128, 2048), torch.float32)]),
    (Reshape3, [((1, 16, 128, 128), torch.float32)]),
    (Reshape4, [((16, 128, 128), torch.float32)]),
    (Reshape5, [((1, 128, 16, 128), torch.float32)]),
    (Reshape6, [((1, 128, 1024), torch.float32)]),
    (Reshape7, [((1, 128, 1024), torch.float32)]),
    (Reshape8, [((128, 1024), torch.float32)]),
    (Reshape9, [((1, 16, 128, 64), torch.float32)]),
    (Reshape10, [((1, 16, 64, 128), torch.float32)]),
    (Reshape11, [((16, 128, 64), torch.float32)]),
    (Reshape12, [((1, 128, 16, 64), torch.float32)]),
    (Reshape13, [((1, 128, 4096), torch.float32)]),
    (Reshape14, [((1, 128, 4096), torch.float32)]),
    (Reshape15, [((128, 4096), torch.float32)]),
    (Reshape16, [((1, 64, 128, 64), torch.float32)]),
    (Reshape17, [((64, 128, 128), torch.float32)]),
    (Reshape18, [((1, 64, 128, 128), torch.float32)]),
    (Reshape19, [((1, 64, 64, 128), torch.float32)]),
    (Reshape20, [((64, 128, 64), torch.float32)]),
    (Reshape21, [((1, 128, 64, 64), torch.float32)]),
    (Reshape22, [((1, 128, 768), torch.float32)]),
    (Reshape23, [((1, 128, 768), torch.float32)]),
    (Reshape24, [((128, 768), torch.float32)]),
    (Reshape25, [((1, 12, 128, 64), torch.float32)]),
    (Reshape26, [((12, 128, 128), torch.float32)]),
    (Reshape27, [((1, 12, 128, 128), torch.float32)]),
    (Reshape28, [((1, 12, 64, 128), torch.float32)]),
    (Reshape29, [((12, 128, 64), torch.float32)]),
    (Reshape30, [((1, 128, 12, 64), torch.float32)]),
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
