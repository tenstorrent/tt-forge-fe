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


class Clip0(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, clip_input_0):
        clip_output_1 = forge.op.Clip("", clip_input_0, min=0.0, max=1.0)
        return clip_output_1


class Clip1(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, clip_input_0):
        clip_output_1 = forge.op.Clip("", clip_input_0, min=0.0, max=6.0)
        return clip_output_1


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (Clip0, [((2, 1, 1, 13), torch.float32)]),
    (Clip0, [((2, 1, 7, 7), torch.float32)]),
    (Clip0, [((1, 1, 256, 256), torch.float32)]),
    (Clip0, [((1, 12, 128, 128), torch.float32)]),
    (Clip0, [((1, 12, 384, 384), torch.float32)]),
    (Clip0, [((1, 1, 32, 32), torch.float32)]),
    (Clip1, [((1, 72, 1, 1), torch.float32)]),
    (Clip1, [((1, 120, 1, 1), torch.float32)]),
    (Clip1, [((1, 480, 1, 1), torch.float32)]),
    (Clip1, [((1, 672, 1, 1), torch.float32)]),
    (Clip1, [((1, 960, 1, 1), torch.float32)]),
    (Clip1, [((1, 24, 96, 96), torch.float32)]),
    (Clip1, [((1, 48, 96, 96), torch.float32)]),
    (Clip1, [((1, 48, 48, 48), torch.float32)]),
    (Clip1, [((1, 96, 48, 48), torch.float32)]),
    (Clip1, [((1, 96, 24, 24), torch.float32)]),
    (Clip1, [((1, 192, 24, 24), torch.float32)]),
    (Clip1, [((1, 192, 12, 12), torch.float32)]),
    (Clip1, [((1, 384, 12, 12), torch.float32)]),
    (Clip1, [((1, 384, 6, 6), torch.float32)]),
    (Clip1, [((1, 768, 6, 6), torch.float32)]),
    (Clip1, [((1, 32, 112, 112), torch.float32)]),
    (Clip1, [((1, 64, 112, 112), torch.float32)]),
    (Clip1, [((1, 64, 56, 56), torch.float32)]),
    (Clip1, [((1, 128, 56, 56), torch.float32)]),
    (Clip1, [((1, 128, 28, 28), torch.float32)]),
    (Clip1, [((1, 256, 28, 28), torch.float32)]),
    (Clip1, [((1, 256, 14, 14), torch.float32)]),
    (Clip1, [((1, 512, 14, 14), torch.float32)]),
    (Clip1, [((1, 512, 7, 7), torch.float32)]),
    (Clip1, [((1, 1024, 7, 7), torch.float32)]),
    (Clip1, [((1, 96, 112, 112), torch.float32)]),
    (Clip1, [((1, 96, 56, 56), torch.float32)]),
    (Clip1, [((1, 144, 56, 56), torch.float32)]),
    (Clip1, [((1, 144, 28, 28), torch.float32)]),
    (Clip1, [((1, 192, 28, 28), torch.float32)]),
    (Clip1, [((1, 192, 14, 14), torch.float32)]),
    (Clip1, [((1, 384, 14, 14), torch.float32)]),
    (Clip1, [((1, 576, 14, 14), torch.float32)]),
    (Clip1, [((1, 576, 7, 7), torch.float32)]),
    (Clip1, [((1, 960, 7, 7), torch.float32)]),
    (Clip1, [((1, 1280, 7, 7), torch.float32)]),
    (Clip1, [((1, 24, 80, 80), torch.float32)]),
    (Clip1, [((1, 96, 80, 80), torch.float32)]),
    (Clip1, [((1, 96, 40, 40), torch.float32)]),
    (Clip1, [((1, 144, 40, 40), torch.float32)]),
    (Clip1, [((1, 144, 20, 20), torch.float32)]),
    (Clip1, [((1, 144, 10, 10), torch.float32)]),
    (Clip1, [((1, 288, 10, 10), torch.float32)]),
    (Clip1, [((1, 432, 10, 10), torch.float32)]),
    (Clip1, [((1, 432, 5, 5), torch.float32)]),
    (Clip1, [((1, 720, 5, 5), torch.float32)]),
    (Clip1, [((1, 1280, 5, 5), torch.float32)]),
    (Clip1, [((1, 16, 48, 48), torch.float32)]),
    (Clip1, [((1, 48, 24, 24), torch.float32)]),
    (Clip1, [((1, 48, 12, 12), torch.float32)]),
    (Clip1, [((1, 96, 12, 12), torch.float32)]),
    (Clip1, [((1, 96, 6, 6), torch.float32)]),
    (Clip1, [((1, 144, 6, 6), torch.float32)]),
    (Clip1, [((1, 192, 6, 6), torch.float32)]),
    (Clip1, [((1, 192, 3, 3), torch.float32)]),
    (Clip1, [((1, 336, 3, 3), torch.float32)]),
    (Clip1, [((1, 1280, 3, 3), torch.float32)]),
    (Clip1, [((1, 384, 28, 28), torch.float32)]),
    (Clip1, [((1, 576, 28, 28), torch.float32)]),
    (Clip1, [((1, 960, 28, 28), torch.float32)]),
    (Clip1, [((1, 16, 112, 112), torch.float32)]),
    (Clip1, [((1, 16, 1, 1), torch.float32)]),
    (Clip1, [((1, 96, 28, 28), torch.float32)]),
    (Clip1, [((1, 96, 14, 14), torch.float32)]),
    (Clip1, [((1, 96, 1, 1), torch.float32)]),
    (Clip1, [((1, 240, 14, 14), torch.float32)]),
    (Clip1, [((1, 240, 1, 1), torch.float32)]),
    (Clip1, [((1, 120, 14, 14), torch.float32)]),
    (Clip1, [((1, 144, 14, 14), torch.float32)]),
    (Clip1, [((1, 144, 1, 1), torch.float32)]),
    (Clip1, [((1, 288, 14, 14), torch.float32)]),
    (Clip1, [((1, 288, 7, 7), torch.float32)]),
    (Clip1, [((1, 288, 1, 1), torch.float32)]),
    (Clip1, [((1, 576, 1, 1), torch.float32)]),
    (Clip1, [((1, 1024, 1, 1), torch.float32)]),
    (Clip1, [((1, 240, 28, 28), torch.float32)]),
    (Clip1, [((1, 200, 14, 14), torch.float32)]),
    (Clip1, [((1, 184, 14, 14), torch.float32)]),
    (Clip1, [((1, 480, 14, 14), torch.float32)]),
    (Clip1, [((1, 672, 14, 14), torch.float32)]),
    (Clip1, [((1, 672, 7, 7), torch.float32)]),
    (Clip1, [((1, 1280, 1, 1), torch.float32)]),
    (Clip1, [((1, 1280), torch.float32)]),
    (Clip1, [((1, 1024), torch.float32)]),
    (Clip1, [((1, 256, 1, 1), torch.float32)]),
    (Clip1, [((1, 512, 1, 1), torch.float32)]),
    (Clip1, [((1, 768, 1, 1), torch.float32)]),
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
