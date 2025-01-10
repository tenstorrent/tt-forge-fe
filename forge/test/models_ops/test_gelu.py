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


class Gelu0(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, gelu_input_0):
        gelu_output_1 = forge.op.Gelu("", gelu_input_0, approximate="none")
        return gelu_output_1


class Gelu1(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, gelu_input_0):
        gelu_output_1 = forge.op.Gelu("", gelu_input_0, approximate="tanh")
        return gelu_output_1


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (Gelu0, [((2, 1, 4096), torch.float32)]),
    (Gelu0, [((2, 1, 8192), torch.float32)]),
    (Gelu0, [((2, 1, 6144), torch.float32)]),
    (Gelu0, [((1, 2, 4096), torch.float32)]),
    (Gelu0, [((1, 2, 2048), torch.float32)]),
    (Gelu0, [((1, 2, 5120), torch.float32)]),
    (Gelu0, [((1, 2, 3072), torch.float32)]),
    (Gelu0, [((1, 2, 1536), torch.float32)]),
    (Gelu0, [((1, 204, 3072), torch.float32)]),
    (Gelu0, [((1, 11, 768), torch.float32)]),
    (Gelu0, [((1, 201, 3072), torch.float32)]),
    (Gelu0, [((1, 1536), torch.float32)]),
    (Gelu1, [((1, 128, 8192), torch.float32)]),
    (Gelu0, [((1, 128, 8192), torch.float32)]),
    (Gelu1, [((1, 128, 4096), torch.float32)]),
    (Gelu0, [((1, 128, 4096), torch.float32)]),
    (Gelu1, [((1, 128, 16384), torch.float32)]),
    (Gelu0, [((1, 128, 16384), torch.float32)]),
    (Gelu1, [((1, 128, 128), torch.float32)]),
    (Gelu0, [((1, 128, 128), torch.float32)]),
    (Gelu0, [((1, 128, 3072), torch.float32)]),
    (Gelu1, [((1, 128, 3072), torch.float32)]),
    (Gelu0, [((1, 256, 4096), torch.float32)]),
    (Gelu1, [((1, 256, 4096), torch.float32)]),
    (Gelu0, [((1, 384, 4096), torch.float32)]),
    (Gelu0, [((1, 128, 768), torch.float32)]),
    (Gelu0, [((1, 384, 3072), torch.float32)]),
    (Gelu0, [((1, 6, 18176), torch.float32)]),
    (Gelu1, [((1, 7, 16384), torch.float32)]),
    (Gelu1, [((1, 256, 3072), torch.float32)]),
    (Gelu1, [((1, 32, 3072), torch.float32)]),
    (Gelu1, [((1, 32, 8192), torch.float32)]),
    (Gelu1, [((1, 256, 8192), torch.float32)]),
    (Gelu0, [((1, 256, 8192), torch.float32)]),
    (Gelu1, [((1, 256, 10240), torch.float32)]),
    (Gelu1, [((1, 32, 10240), torch.float32)]),
    (Gelu1, [((1, 12, 10240), torch.float32)]),
    (Gelu1, [((1, 11, 10240), torch.float32)]),
    (Gelu0, [((1, 3072, 128), torch.float32)]),
    (Gelu1, [((1, 1, 1024), torch.float32)]),
    (Gelu0, [((1, 1, 1024), torch.float32)]),
    (Gelu1, [((1, 1, 2048), torch.float32)]),
    (Gelu0, [((1, 197, 3072), torch.float32)]),
    (Gelu0, [((1, 197, 768), torch.float32)]),
    (Gelu0, [((1, 197, 1536), torch.float32)]),
    (Gelu0, [((1, 768, 384), torch.float32)]),
    (Gelu0, [((1, 49, 3072), torch.float32)]),
    (Gelu0, [((1, 1024, 512), torch.float32)]),
    (Gelu0, [((1, 49, 4096), torch.float32)]),
    (Gelu0, [((1, 196, 3072), torch.float32)]),
    (Gelu0, [((1, 512, 256), torch.float32)]),
    (Gelu0, [((1, 49, 2048), torch.float32)]),
    (Gelu0, [((1, 196, 2048), torch.float32)]),
    (Gelu0, [((1, 196, 4096), torch.float32)]),
    (Gelu0, [((1, 512, 1024), torch.float32)]),
    (Gelu0, [((1, 16384, 128), torch.float32)]),
    (Gelu0, [((1, 4096, 256), torch.float32)]),
    (Gelu0, [((1, 1024, 640), torch.float32)]),
    (Gelu0, [((1, 256, 1024), torch.float32)]),
    (Gelu0, [((1, 16384, 256), torch.float32)]),
    (Gelu0, [((1, 4096, 512), torch.float32)]),
    (Gelu0, [((1, 1024, 1280), torch.float32)]),
    (Gelu0, [((1, 256, 2048), torch.float32)]),
    (Gelu0, [((1, 197, 4096), torch.float32)]),
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
