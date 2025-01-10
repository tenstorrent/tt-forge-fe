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


class Repeatinterleave0(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, repeatinterleave_input_0):
        repeatinterleave_output_1 = forge.op.RepeatInterleave("", repeatinterleave_input_0, repeats=1, dim=1)
        return repeatinterleave_output_1


class Repeatinterleave1(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, repeatinterleave_input_0):
        repeatinterleave_output_1 = forge.op.RepeatInterleave("", repeatinterleave_input_0, repeats=1, dim=2)
        return repeatinterleave_output_1


class Repeatinterleave2(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, repeatinterleave_input_0):
        repeatinterleave_output_1 = forge.op.RepeatInterleave("", repeatinterleave_input_0, repeats=7, dim=2)
        return repeatinterleave_output_1


class Repeatinterleave3(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, repeatinterleave_input_0):
        repeatinterleave_output_1 = forge.op.RepeatInterleave("", repeatinterleave_input_0, repeats=1, dim=0)
        return repeatinterleave_output_1


class Repeatinterleave4(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, repeatinterleave_input_0):
        repeatinterleave_output_1 = forge.op.RepeatInterleave("", repeatinterleave_input_0, repeats=256, dim=2)
        return repeatinterleave_output_1


class Repeatinterleave5(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, repeatinterleave_input_0):
        repeatinterleave_output_1 = forge.op.RepeatInterleave("", repeatinterleave_input_0, repeats=8, dim=2)
        return repeatinterleave_output_1


class Repeatinterleave6(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, repeatinterleave_input_0):
        repeatinterleave_output_1 = forge.op.RepeatInterleave("", repeatinterleave_input_0, repeats=4, dim=2)
        return repeatinterleave_output_1


class Repeatinterleave7(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, repeatinterleave_input_0):
        repeatinterleave_output_1 = forge.op.RepeatInterleave("", repeatinterleave_input_0, repeats=32, dim=2)
        return repeatinterleave_output_1


class Repeatinterleave8(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, repeatinterleave_input_0):
        repeatinterleave_output_1 = forge.op.RepeatInterleave("", repeatinterleave_input_0, repeats=6, dim=2)
        return repeatinterleave_output_1


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (Repeatinterleave0, [((2, 1, 1, 13), torch.float32)]),
    (Repeatinterleave1, [((2, 1, 1, 13), torch.float32)]),
    (Repeatinterleave0, [((2, 1, 1, 7), torch.float32)]),
    (Repeatinterleave2, [((2, 1, 1, 7), torch.float32)]),
    (Repeatinterleave3, [((1, 128), torch.float32)]),
    (Repeatinterleave3, [((1, 1, 1, 256), torch.float32)]),
    (Repeatinterleave0, [((1, 1, 1, 256), torch.float32)]),
    (Repeatinterleave4, [((1, 1, 1, 256), torch.float32)]),
    (Repeatinterleave3, [((1, 384), torch.float32)]),
    (Repeatinterleave3, [((1, 32, 1), torch.float32)]),
    (Repeatinterleave1, [((1, 32, 1), torch.float32)]),
    (Repeatinterleave3, [((1, 16, 1), torch.float32)]),
    (Repeatinterleave1, [((1, 16, 1), torch.float32)]),
    (Repeatinterleave3, [((1, 128, 1), torch.float32)]),
    (Repeatinterleave1, [((1, 128, 1), torch.float32)]),
    (Repeatinterleave3, [((1, 1, 1, 7, 256), torch.float32)]),
    (Repeatinterleave0, [((1, 1, 1, 7, 256), torch.float32)]),
    (Repeatinterleave5, [((1, 1, 1, 7, 256), torch.float32)]),
    (Repeatinterleave3, [((1, 64, 1), torch.float32)]),
    (Repeatinterleave1, [((1, 64, 1), torch.float32)]),
    (Repeatinterleave3, [((1, 8, 1, 256, 128), torch.float32)]),
    (Repeatinterleave6, [((1, 8, 1, 256, 128), torch.float32)]),
    (Repeatinterleave3, [((1, 8, 1, 4, 64), torch.float32)]),
    (Repeatinterleave6, [((1, 8, 1, 4, 64), torch.float32)]),
    (Repeatinterleave3, [((1, 8, 1, 4, 128), torch.float32)]),
    (Repeatinterleave6, [((1, 8, 1, 4, 128), torch.float32)]),
    (Repeatinterleave3, [((1, 8, 1, 256, 64), torch.float32)]),
    (Repeatinterleave6, [((1, 8, 1, 256, 64), torch.float32)]),
    (Repeatinterleave3, [((1, 8, 1, 128, 128), torch.float32)]),
    (Repeatinterleave6, [((1, 8, 1, 128, 128), torch.float32)]),
    (Repeatinterleave3, [((1, 1, 1, 32), torch.float32)]),
    (Repeatinterleave0, [((1, 1, 1, 32), torch.float32)]),
    (Repeatinterleave7, [((1, 1, 1, 32), torch.float32)]),
    (Repeatinterleave3, [((1, 2, 1, 35, 128), torch.float32)]),
    (Repeatinterleave8, [((1, 2, 1, 35, 128), torch.float32)]),
    (Repeatinterleave5, [((1, 2, 1, 35, 128), torch.float32)]),
    (Repeatinterleave3, [((1, 4, 1, 35, 128), torch.float32)]),
    (Repeatinterleave2, [((1, 4, 1, 35, 128), torch.float32)]),
    (Repeatinterleave3, [((1, 2, 1, 35, 64), torch.float32)]),
    (Repeatinterleave2, [((1, 2, 1, 35, 64), torch.float32)]),
    (Repeatinterleave3, [((1, 2, 1, 29, 64), torch.float32)]),
    (Repeatinterleave2, [((1, 2, 1, 29, 64), torch.float32)]),
    (Repeatinterleave3, [((1, 2, 1, 39, 128), torch.float32)]),
    (Repeatinterleave8, [((1, 2, 1, 39, 128), torch.float32)]),
    (Repeatinterleave5, [((1, 2, 1, 39, 128), torch.float32)]),
    (Repeatinterleave3, [((1, 2, 1, 29, 128), torch.float32)]),
    (Repeatinterleave8, [((1, 2, 1, 29, 128), torch.float32)]),
    (Repeatinterleave5, [((1, 2, 1, 29, 128), torch.float32)]),
    (Repeatinterleave3, [((1, 2, 1, 39, 64), torch.float32)]),
    (Repeatinterleave2, [((1, 2, 1, 39, 64), torch.float32)]),
    (Repeatinterleave3, [((1, 4, 1, 39, 128), torch.float32)]),
    (Repeatinterleave2, [((1, 4, 1, 39, 128), torch.float32)]),
    (Repeatinterleave3, [((1, 4, 1, 29, 128), torch.float32)]),
    (Repeatinterleave2, [((1, 4, 1, 29, 128), torch.float32)]),
    (Repeatinterleave3, [((1, 1, 768), torch.float32)]),
    (Repeatinterleave3, [((1, 1, 192), torch.float32)]),
    (Repeatinterleave3, [((1, 1, 384), torch.float32)]),
    (Repeatinterleave3, [((1, 1, 1024), torch.float32)]),
    (Repeatinterleave3, [((1, 512, 1024), torch.float32)]),
    (Repeatinterleave3, [((1, 50176, 256), torch.float32)]),
    (Repeatinterleave3, [((1, 1, 1024), torch.float32)]),
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
