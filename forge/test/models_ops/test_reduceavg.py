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


class Reduceavg0(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reduceavg_input_0):
        reduceavg_output_1 = forge.op.ReduceAvg("", reduceavg_input_0, dim=-1, keep_dim=True)
        return reduceavg_output_1


class Reduceavg1(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reduceavg_input_0):
        reduceavg_output_1 = forge.op.ReduceAvg("", reduceavg_input_0, dim=-2, keep_dim=True)
        return reduceavg_output_1


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (Reduceavg0, [((2, 13, 768), torch.float32)]),
    (Reduceavg0, [((1, 7, 2048), torch.float32)]),
    (Reduceavg0, [((1, 256, 4096), torch.float32)]),
    (Reduceavg0, [((1, 4, 2048), torch.float32)]),
    (Reduceavg0, [((1, 4, 4096), torch.float32)]),
    (Reduceavg0, [((1, 256, 2048), torch.float32)]),
    (Reduceavg0, [((1, 128, 4096), torch.float32)]),
    (Reduceavg0, [((1, 6, 1024), torch.float32)]),
    (Reduceavg0, [((1, 29, 1024), torch.float32)]),
    (Reduceavg0, [((1, 35, 1536), torch.float32)]),
    (Reduceavg0, [((1, 35, 3584), torch.float32)]),
    (Reduceavg0, [((1, 35, 2048), torch.float32)]),
    (Reduceavg0, [((1, 35, 896), torch.float32)]),
    (Reduceavg0, [((1, 29, 896), torch.float32)]),
    (Reduceavg0, [((1, 39, 1536), torch.float32)]),
    (Reduceavg0, [((1, 29, 1536), torch.float32)]),
    (Reduceavg0, [((1, 29, 2048), torch.float32)]),
    (Reduceavg0, [((1, 39, 2048), torch.float32)]),
    (Reduceavg0, [((1, 39, 896), torch.float32)]),
    (Reduceavg0, [((1, 39, 3584), torch.float32)]),
    (Reduceavg0, [((1, 29, 3584), torch.float32)]),
    (Reduceavg0, [((1, 1, 512), torch.float32)]),
    (Reduceavg0, [((1, 1, 768), torch.float32)]),
    (Reduceavg0, [((1, 1, 1024), torch.float32)]),
    (Reduceavg1, [((1, 48, 160, 160), torch.float32)]),
    (Reduceavg0, [((1, 48, 1, 160), torch.float32)]),
    (Reduceavg1, [((1, 24, 160, 160), torch.float32)]),
    (Reduceavg0, [((1, 24, 1, 160), torch.float32)]),
    (Reduceavg1, [((1, 144, 80, 80), torch.float32)]),
    (Reduceavg0, [((1, 144, 1, 80), torch.float32)]),
    (Reduceavg1, [((1, 192, 80, 80), torch.float32)]),
    (Reduceavg0, [((1, 192, 1, 80), torch.float32)]),
    (Reduceavg1, [((1, 192, 40, 40), torch.float32)]),
    (Reduceavg0, [((1, 192, 1, 40), torch.float32)]),
    (Reduceavg1, [((1, 336, 40, 40), torch.float32)]),
    (Reduceavg0, [((1, 336, 1, 40), torch.float32)]),
    (Reduceavg1, [((1, 336, 20, 20), torch.float32)]),
    (Reduceavg0, [((1, 336, 1, 20), torch.float32)]),
    (Reduceavg1, [((1, 672, 20, 20), torch.float32)]),
    (Reduceavg0, [((1, 672, 1, 20), torch.float32)]),
    (Reduceavg1, [((1, 960, 20, 20), torch.float32)]),
    (Reduceavg0, [((1, 960, 1, 20), torch.float32)]),
    (Reduceavg1, [((1, 960, 10, 10), torch.float32)]),
    (Reduceavg0, [((1, 960, 1, 10), torch.float32)]),
    (Reduceavg1, [((1, 1632, 10, 10), torch.float32)]),
    (Reduceavg0, [((1, 1632, 1, 10), torch.float32)]),
    (Reduceavg1, [((1, 2688, 10, 10), torch.float32)]),
    (Reduceavg0, [((1, 2688, 1, 10), torch.float32)]),
    (Reduceavg1, [((1, 32, 112, 112), torch.float32)]),
    (Reduceavg0, [((1, 32, 1, 112), torch.float32)]),
    (Reduceavg1, [((1, 96, 56, 56), torch.float32)]),
    (Reduceavg0, [((1, 96, 1, 56), torch.float32)]),
    (Reduceavg1, [((1, 144, 56, 56), torch.float32)]),
    (Reduceavg0, [((1, 144, 1, 56), torch.float32)]),
    (Reduceavg1, [((1, 144, 28, 28), torch.float32)]),
    (Reduceavg0, [((1, 144, 1, 28), torch.float32)]),
    (Reduceavg1, [((1, 240, 28, 28), torch.float32)]),
    (Reduceavg0, [((1, 240, 1, 28), torch.float32)]),
    (Reduceavg1, [((1, 240, 14, 14), torch.float32)]),
    (Reduceavg0, [((1, 240, 1, 14), torch.float32)]),
    (Reduceavg1, [((1, 480, 14, 14), torch.float32)]),
    (Reduceavg0, [((1, 480, 1, 14), torch.float32)]),
    (Reduceavg1, [((1, 672, 14, 14), torch.float32)]),
    (Reduceavg0, [((1, 672, 1, 14), torch.float32)]),
    (Reduceavg1, [((1, 672, 7, 7), torch.float32)]),
    (Reduceavg0, [((1, 672, 1, 7), torch.float32)]),
    (Reduceavg1, [((1, 1152, 7, 7), torch.float32)]),
    (Reduceavg0, [((1, 1152, 1, 7), torch.float32)]),
    (Reduceavg1, [((1, 72, 28, 28), torch.float32)]),
    (Reduceavg0, [((1, 72, 1, 28), torch.float32)]),
    (Reduceavg1, [((1, 120, 28, 28), torch.float32)]),
    (Reduceavg0, [((1, 120, 1, 28), torch.float32)]),
    (Reduceavg1, [((1, 960, 7, 7), torch.float32)]),
    (Reduceavg0, [((1, 960, 1, 7), torch.float32)]),
    (Reduceavg1, [((1, 49, 768), torch.float32)]),
    (Reduceavg1, [((1, 49, 1024), torch.float32)]),
    (Reduceavg1, [((1, 196, 768), torch.float32)]),
    (Reduceavg1, [((1, 49, 512), torch.float32)]),
    (Reduceavg1, [((1, 196, 512), torch.float32)]),
    (Reduceavg1, [((1, 196, 1024), torch.float32)]),
    (Reduceavg1, [((1, 16, 56, 56), torch.float32)]),
    (Reduceavg0, [((1, 16, 1, 56), torch.float32)]),
    (Reduceavg1, [((1, 96, 14, 14), torch.float32)]),
    (Reduceavg0, [((1, 96, 1, 14), torch.float32)]),
    (Reduceavg1, [((1, 120, 14, 14), torch.float32)]),
    (Reduceavg0, [((1, 120, 1, 14), torch.float32)]),
    (Reduceavg1, [((1, 144, 14, 14), torch.float32)]),
    (Reduceavg0, [((1, 144, 1, 14), torch.float32)]),
    (Reduceavg1, [((1, 288, 7, 7), torch.float32)]),
    (Reduceavg0, [((1, 288, 1, 7), torch.float32)]),
    (Reduceavg1, [((1, 576, 7, 7), torch.float32)]),
    (Reduceavg0, [((1, 576, 1, 7), torch.float32)]),
    (Reduceavg1, [((1, 256, 512), torch.float32)]),
    (Reduceavg1, [((1, 256, 256), torch.float32)]),
    (Reduceavg1, [((1, 256, 56, 56), torch.float32)]),
    (Reduceavg0, [((1, 256, 1, 56), torch.float32)]),
    (Reduceavg1, [((1, 512, 28, 28), torch.float32)]),
    (Reduceavg0, [((1, 512, 1, 28), torch.float32)]),
    (Reduceavg1, [((1, 768, 14, 14), torch.float32)]),
    (Reduceavg0, [((1, 768, 1, 14), torch.float32)]),
    (Reduceavg1, [((1, 1024, 7, 7), torch.float32)]),
    (Reduceavg0, [((1, 1024, 1, 7), torch.float32)]),
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
