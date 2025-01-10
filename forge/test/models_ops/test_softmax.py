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


class Softmax0(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, softmax_input_0):
        softmax_output_1 = forge.op.Softmax("", softmax_input_0, dim=-1)
        return softmax_output_1


class Softmax1(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, softmax_input_0):
        softmax_output_1 = forge.op.Softmax("", softmax_input_0, dim=1)
        return softmax_output_1


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (Softmax0, [((32, 1, 1), torch.float32)]),
    (Softmax0, [((2, 12, 13, 13), torch.float32)]),
    (Softmax0, [((32, 1, 13), torch.float32)]),
    (Softmax0, [((64, 1, 1), torch.float32)]),
    (Softmax0, [((64, 1, 13), torch.float32)]),
    (Softmax0, [((48, 1, 1), torch.float32)]),
    (Softmax0, [((48, 1, 13), torch.float32)]),
    (Softmax0, [((1, 16, 2, 2), torch.float32)]),
    (Softmax0, [((1, 16, 2, 1500), torch.float32)]),
    (Softmax0, [((1, 8, 2, 2), torch.float32)]),
    (Softmax0, [((1, 8, 2, 1500), torch.float32)]),
    (Softmax0, [((1, 20, 2, 2), torch.float32)]),
    (Softmax0, [((1, 20, 2, 1500), torch.float32)]),
    (Softmax0, [((1, 12, 2, 2), torch.float32)]),
    (Softmax0, [((1, 12, 2, 1500), torch.float32)]),
    (Softmax0, [((1, 6, 2, 2), torch.float32)]),
    (Softmax0, [((1, 6, 2, 1500), torch.float32)]),
    (Softmax0, [((16, 7, 7), torch.float32)]),
    (Softmax0, [((1, 12, 204, 204), torch.float32)]),
    (Softmax0, [((1, 12, 201, 201), torch.float32)]),
    (Softmax0, [((1, 16, 128, 128), torch.float32)]),
    (Softmax0, [((1, 64, 128, 128), torch.float32)]),
    (Softmax0, [((1, 12, 128, 128), torch.float32)]),
    (Softmax0, [((16, 256, 256), torch.float32)]),
    (Softmax0, [((1, 16, 384, 384), torch.float32)]),
    (Softmax0, [((1, 16, 256, 256), torch.float32)]),
    (Softmax0, [((1, 12, 384, 384), torch.float32)]),
    (Softmax0, [((1, 71, 6, 6), torch.float32)]),
    (Softmax0, [((1, 64, 334, 334), torch.float32)]),
    (Softmax0, [((1, 8, 7, 7), torch.float32)]),
    (Softmax0, [((1, 12, 256, 256), torch.float32)]),
    (Softmax0, [((1, 12, 32, 32), torch.float32)]),
    (Softmax0, [((1, 16, 32, 32), torch.float32)]),
    (Softmax0, [((1, 20, 256, 256), torch.float32)]),
    (Softmax0, [((1, 20, 32, 32), torch.float32)]),
    (Softmax0, [((1, 32, 256, 256), torch.float32)]),
    (Softmax0, [((1, 32, 4, 4), torch.float32)]),
    (Softmax0, [((1, 32, 128, 128), torch.float32)]),
    (Softmax0, [((32, 32, 32), torch.float32)]),
    (Softmax0, [((32, 256, 256), torch.float32)]),
    (Softmax0, [((16, 32, 32), torch.float32)]),
    (Softmax0, [((12, 32, 32), torch.float32)]),
    (Softmax0, [((12, 256, 256), torch.float32)]),
    (Softmax0, [((1, 32, 12, 12), torch.float32)]),
    (Softmax0, [((1, 32, 11, 11), torch.float32)]),
    (Softmax0, [((1, 16, 6, 6), torch.float32)]),
    (Softmax0, [((1, 16, 29, 29), torch.float32)]),
    (Softmax0, [((1, 12, 35, 35), torch.float32)]),
    (Softmax0, [((1, 28, 35, 35), torch.float32)]),
    (Softmax0, [((1, 16, 35, 35), torch.float32)]),
    (Softmax0, [((1, 14, 35, 35), torch.float32)]),
    (Softmax0, [((1, 14, 29, 29), torch.float32)]),
    (Softmax0, [((1, 12, 39, 39), torch.float32)]),
    (Softmax0, [((1, 12, 29, 29), torch.float32)]),
    (Softmax0, [((1, 16, 39, 39), torch.float32)]),
    (Softmax0, [((1, 14, 39, 39), torch.float32)]),
    (Softmax0, [((1, 28, 39, 39), torch.float32)]),
    (Softmax0, [((1, 28, 29, 29), torch.float32)]),
    (Softmax0, [((1, 6, 1, 1), torch.float32)]),
    (Softmax0, [((1, 12, 1, 1), torch.float32)]),
    (Softmax0, [((1, 12, 1, 256), torch.float32)]),
    (Softmax0, [((1, 16, 1, 1), torch.float32)]),
    (Softmax0, [((1, 16, 1, 256), torch.float32)]),
    (Softmax0, [((1, 8, 1, 1), torch.float32)]),
    (Softmax0, [((1, 12, 197, 197), torch.float32)]),
    (Softmax0, [((1, 3, 197, 197), torch.float32)]),
    (Softmax0, [((1, 6, 197, 197), torch.float32)]),
    (Softmax0, [((1, 1, 512, 3025), torch.float32)]),
    (Softmax0, [((1, 8, 512, 512), torch.float32)]),
    (Softmax0, [((1, 1, 1, 512), torch.float32)]),
    (Softmax0, [((1, 1, 512, 50176), torch.float32)]),
    (Softmax0, [((1, 1, 16384, 256), torch.float32)]),
    (Softmax0, [((1, 2, 4096, 256), torch.float32)]),
    (Softmax0, [((1, 5, 1024, 256), torch.float32)]),
    (Softmax0, [((1, 8, 256, 256), torch.float32)]),
    (Softmax0, [((1, 16, 197, 197), torch.float32)]),
    (Softmax1, [((1, 17, 4, 4480), torch.float32)]),
    (Softmax1, [((1, 17, 4, 1120), torch.float32)]),
    (Softmax1, [((1, 17, 4, 280), torch.float32)]),
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
