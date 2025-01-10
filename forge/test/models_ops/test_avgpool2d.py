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


class Avgpool2D0(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, avgpool2d_input_0):
        avgpool2d_output_1 = forge.op.AvgPool2d(
            "",
            avgpool2d_input_0,
            kernel_size=[1, 1],
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            ceil_mode=False,
            count_include_pad=True,
            channel_last=0,
        )
        return avgpool2d_output_1


class Avgpool2D1(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, avgpool2d_input_0):
        avgpool2d_output_1 = forge.op.AvgPool2d(
            "",
            avgpool2d_input_0,
            kernel_size=[2, 2],
            stride=[2, 2],
            padding=[0, 0, 0, 0],
            ceil_mode=False,
            count_include_pad=True,
            channel_last=0,
        )
        return avgpool2d_output_1


class Avgpool2D2(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, avgpool2d_input_0):
        avgpool2d_output_1 = forge.op.AvgPool2d(
            "",
            avgpool2d_input_0,
            kernel_size=[56, 56],
            stride=[56, 56],
            padding=[0, 0, 0, 0],
            ceil_mode=False,
            count_include_pad=True,
            channel_last=0,
        )
        return avgpool2d_output_1


class Avgpool2D3(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, avgpool2d_input_0):
        avgpool2d_output_1 = forge.op.AvgPool2d(
            "",
            avgpool2d_input_0,
            kernel_size=[7, 7],
            stride=[7, 7],
            padding=[0, 0, 0, 0],
            ceil_mode=False,
            count_include_pad=True,
            channel_last=0,
        )
        return avgpool2d_output_1


class Avgpool2D4(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, avgpool2d_input_0):
        avgpool2d_output_1 = forge.op.AvgPool2d(
            "",
            avgpool2d_input_0,
            kernel_size=[14, 14],
            stride=[14, 14],
            padding=[0, 0, 0, 0],
            ceil_mode=False,
            count_include_pad=True,
            channel_last=0,
        )
        return avgpool2d_output_1


class Avgpool2D5(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, avgpool2d_input_0):
        avgpool2d_output_1 = forge.op.AvgPool2d(
            "",
            avgpool2d_input_0,
            kernel_size=[7, 7],
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            ceil_mode=False,
            count_include_pad=True,
            channel_last=0,
        )
        return avgpool2d_output_1


class Avgpool2D6(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, avgpool2d_input_0):
        avgpool2d_output_1 = forge.op.AvgPool2d(
            "",
            avgpool2d_input_0,
            kernel_size=[112, 112],
            stride=[112, 112],
            padding=[0, 0, 0, 0],
            ceil_mode=False,
            count_include_pad=True,
            channel_last=0,
        )
        return avgpool2d_output_1


class Avgpool2D7(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, avgpool2d_input_0):
        avgpool2d_output_1 = forge.op.AvgPool2d(
            "",
            avgpool2d_input_0,
            kernel_size=[28, 28],
            stride=[28, 28],
            padding=[0, 0, 0, 0],
            ceil_mode=False,
            count_include_pad=True,
            channel_last=0,
        )
        return avgpool2d_output_1


class Avgpool2D8(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, avgpool2d_input_0):
        avgpool2d_output_1 = forge.op.AvgPool2d(
            "",
            avgpool2d_input_0,
            kernel_size=[10, 10],
            stride=[10, 10],
            padding=[0, 0, 0, 0],
            ceil_mode=False,
            count_include_pad=True,
            channel_last=0,
        )
        return avgpool2d_output_1


class Avgpool2D9(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, avgpool2d_input_0):
        avgpool2d_output_1 = forge.op.AvgPool2d(
            "",
            avgpool2d_input_0,
            kernel_size=[3, 3],
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            ceil_mode=False,
            count_include_pad=False,
            channel_last=0,
        )
        return avgpool2d_output_1


class Avgpool2D10(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, avgpool2d_input_0):
        avgpool2d_output_1 = forge.op.AvgPool2d(
            "",
            avgpool2d_input_0,
            kernel_size=[8, 8],
            stride=[8, 8],
            padding=[0, 0, 0, 0],
            ceil_mode=False,
            count_include_pad=True,
            channel_last=0,
        )
        return avgpool2d_output_1


class Avgpool2D11(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, avgpool2d_input_0):
        avgpool2d_output_1 = forge.op.AvgPool2d(
            "",
            avgpool2d_input_0,
            kernel_size=[8, 8],
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            ceil_mode=False,
            count_include_pad=True,
            channel_last=0,
        )
        return avgpool2d_output_1


class Avgpool2D12(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, avgpool2d_input_0):
        avgpool2d_output_1 = forge.op.AvgPool2d(
            "",
            avgpool2d_input_0,
            kernel_size=[6, 6],
            stride=[6, 6],
            padding=[0, 0, 0, 0],
            ceil_mode=False,
            count_include_pad=True,
            channel_last=0,
        )
        return avgpool2d_output_1


class Avgpool2D13(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, avgpool2d_input_0):
        avgpool2d_output_1 = forge.op.AvgPool2d(
            "",
            avgpool2d_input_0,
            kernel_size=[5, 5],
            stride=[5, 5],
            padding=[0, 0, 0, 0],
            ceil_mode=False,
            count_include_pad=True,
            channel_last=0,
        )
        return avgpool2d_output_1


class Avgpool2D14(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, avgpool2d_input_0):
        avgpool2d_output_1 = forge.op.AvgPool2d(
            "",
            avgpool2d_input_0,
            kernel_size=[3, 3],
            stride=[3, 3],
            padding=[0, 0, 0, 0],
            ceil_mode=False,
            count_include_pad=True,
            channel_last=0,
        )
        return avgpool2d_output_1


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (Avgpool2D0, [((1, 256, 6, 6), torch.float32)]),
    (Avgpool2D1, [((1, 128, 56, 56), torch.float32)]),
    (Avgpool2D2, [((1, 128, 56, 56), torch.float32)]),
    (Avgpool2D1, [((1, 256, 28, 28), torch.float32)]),
    (Avgpool2D1, [((1, 896, 14, 14), torch.float32)]),
    (Avgpool2D3, [((1, 1920, 7, 7), torch.float32)]),
    (Avgpool2D1, [((1, 512, 14, 14), torch.float32)]),
    (Avgpool2D4, [((1, 512, 14, 14), torch.float32)]),
    (Avgpool2D3, [((1, 1024, 7, 7), torch.float32)]),
    (Avgpool2D5, [((1, 1024, 7, 7), torch.float32)]),
    (Avgpool2D1, [((1, 192, 56, 56), torch.float32)]),
    (Avgpool2D2, [((1, 192, 56, 56), torch.float32)]),
    (Avgpool2D1, [((1, 384, 28, 28), torch.float32)]),
    (Avgpool2D1, [((1, 1056, 14, 14), torch.float32)]),
    (Avgpool2D3, [((1, 2208, 7, 7), torch.float32)]),
    (Avgpool2D1, [((1, 640, 14, 14), torch.float32)]),
    (Avgpool2D3, [((1, 1664, 7, 7), torch.float32)]),
    (Avgpool2D3, [((1, 256, 7, 7), torch.float32)]),
    (Avgpool2D3, [((1, 512, 7, 7), torch.float32)]),
    (Avgpool2D0, [((1, 512, 7, 7), torch.float32)]),
    (Avgpool2D5, [((1, 512, 7, 7), torch.float32)]),
    (Avgpool2D6, [((1, 48, 112, 112), torch.float32)]),
    (Avgpool2D6, [((1, 24, 112, 112), torch.float32)]),
    (Avgpool2D2, [((1, 144, 56, 56), torch.float32)]),
    (Avgpool2D7, [((1, 192, 28, 28), torch.float32)]),
    (Avgpool2D7, [((1, 336, 28, 28), torch.float32)]),
    (Avgpool2D4, [((1, 336, 14, 14), torch.float32)]),
    (Avgpool2D4, [((1, 672, 14, 14), torch.float32)]),
    (Avgpool2D4, [((1, 960, 14, 14), torch.float32)]),
    (Avgpool2D3, [((1, 960, 7, 7), torch.float32)]),
    (Avgpool2D3, [((1, 1632, 7, 7), torch.float32)]),
    (Avgpool2D3, [((1, 2688, 7, 7), torch.float32)]),
    (Avgpool2D3, [((1, 1792, 7, 7), torch.float32)]),
    (Avgpool2D8, [((1, 1792, 10, 10), torch.float32)]),
    (Avgpool2D6, [((1, 32, 112, 112), torch.float32)]),
    (Avgpool2D2, [((1, 96, 56, 56), torch.float32)]),
    (Avgpool2D7, [((1, 144, 28, 28), torch.float32)]),
    (Avgpool2D7, [((1, 240, 28, 28), torch.float32)]),
    (Avgpool2D4, [((1, 240, 14, 14), torch.float32)]),
    (Avgpool2D4, [((1, 480, 14, 14), torch.float32)]),
    (Avgpool2D3, [((1, 672, 7, 7), torch.float32)]),
    (Avgpool2D3, [((1, 1152, 7, 7), torch.float32)]),
    (Avgpool2D3, [((1, 1280, 7, 7), torch.float32)]),
    (Avgpool2D5, [((1, 2048, 7, 7), torch.float32)]),
    (Avgpool2D3, [((1, 2048, 7, 7), torch.float32)]),
    (Avgpool2D9, [((1, 384, 35, 35), torch.float32)]),
    (Avgpool2D9, [((1, 1024, 17, 17), torch.float32)]),
    (Avgpool2D9, [((1, 1536, 8, 8), torch.float32)]),
    (Avgpool2D10, [((1, 1536, 8, 8), torch.float32)]),
    (Avgpool2D11, [((1, 1536, 8, 8), torch.float32)]),
    (Avgpool2D12, [((1, 768, 6, 6), torch.float32)]),
    (Avgpool2D1, [((1, 1024, 2, 2), torch.float32)]),
    (Avgpool2D13, [((1, 1280, 5, 5), torch.float32)]),
    (Avgpool2D14, [((1, 1280, 3, 3), torch.float32)]),
    (Avgpool2D7, [((1, 320, 28, 28), torch.float32)]),
    (Avgpool2D3, [((1, 576, 7, 7), torch.float32)]),
    (Avgpool2D7, [((1, 72, 28, 28), torch.float32)]),
    (Avgpool2D7, [((1, 120, 28, 28), torch.float32)]),
    (Avgpool2D2, [((1, 16, 56, 56), torch.float32)]),
    (Avgpool2D4, [((1, 96, 14, 14), torch.float32)]),
    (Avgpool2D4, [((1, 120, 14, 14), torch.float32)]),
    (Avgpool2D4, [((1, 144, 14, 14), torch.float32)]),
    (Avgpool2D3, [((1, 288, 7, 7), torch.float32)]),
    (Avgpool2D3, [((1, 1088, 7, 7), torch.float32)]),
    (Avgpool2D0, [((1, 4096, 1, 1), torch.float32)]),
    (Avgpool2D8, [((1, 2048, 10, 10), torch.float32)]),
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
