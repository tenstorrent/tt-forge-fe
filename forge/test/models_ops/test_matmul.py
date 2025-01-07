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


class Matmul0(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, matmul_input_0, matmul_input_1):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, matmul_input_1)
        return matmul_output_1


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (Matmul0, [((1, 128, 128), torch.float32), ((128, 2048), torch.float32)]),
    (Matmul0, [((128, 2048), torch.float32), ((2048, 2048), torch.float32)]),
    (Matmul0, [((16, 128, 128), torch.float32), ((16, 128, 128), torch.float32)]),
    (Matmul0, [((1, 128, 2048), torch.float32), ((2048, 2048), torch.float32)]),
    (Matmul0, [((1, 128, 2048), torch.float32), ((2048, 8192), torch.float32)]),
    (Matmul0, [((1, 128, 8192), torch.float32), ((8192, 2048), torch.float32)]),
    (Matmul0, [((1, 128, 2048), torch.float32), ((2048, 2), torch.float32)]),
    (Matmul0, [((1, 128, 128), torch.float32), ((128, 1024), torch.float32)]),
    (Matmul0, [((128, 1024), torch.float32), ((1024, 1024), torch.float32)]),
    (Matmul0, [((16, 128, 64), torch.float32), ((16, 64, 128), torch.float32)]),
    (Matmul0, [((16, 128, 128), torch.float32), ((16, 128, 64), torch.float32)]),
    (Matmul0, [((1, 128, 1024), torch.float32), ((1024, 1024), torch.float32)]),
    (Matmul0, [((1, 128, 1024), torch.float32), ((1024, 4096), torch.float32)]),
    (Matmul0, [((1, 128, 4096), torch.float32), ((4096, 1024), torch.float32)]),
    (Matmul0, [((1, 128, 1024), torch.float32), ((1024, 2), torch.float32)]),
    (Matmul0, [((1, 128, 128), torch.float32), ((128, 4096), torch.float32)]),
    (Matmul0, [((128, 4096), torch.float32), ((4096, 4096), torch.float32)]),
    (Matmul0, [((64, 128, 64), torch.float32), ((64, 64, 128), torch.float32)]),
    (Matmul0, [((64, 128, 128), torch.float32), ((64, 128, 64), torch.float32)]),
    (Matmul0, [((1, 128, 4096), torch.float32), ((4096, 4096), torch.float32)]),
    (Matmul0, [((1, 128, 4096), torch.float32), ((4096, 16384), torch.float32)]),
    (Matmul0, [((1, 128, 16384), torch.float32), ((16384, 4096), torch.float32)]),
    (Matmul0, [((1, 128, 4096), torch.float32), ((4096, 128), torch.float32)]),
    (Matmul0, [((1, 128, 128), torch.float32), ((128, 30000), torch.float32)]),
    (Matmul0, [((1, 128, 4096), torch.float32), ((4096, 2), torch.float32)]),
    (Matmul0, [((1, 128, 2048), torch.float32), ((2048, 128), torch.float32)]),
    (Matmul0, [((1, 128, 1024), torch.float32), ((1024, 128), torch.float32)]),
    (Matmul0, [((1, 128, 128), torch.float32), ((128, 768), torch.float32)]),
    (Matmul0, [((128, 768), torch.float32), ((768, 768), torch.float32)]),
    (Matmul0, [((12, 128, 64), torch.float32), ((12, 64, 128), torch.float32)]),
    (Matmul0, [((12, 128, 128), torch.float32), ((12, 128, 64), torch.float32)]),
    (Matmul0, [((1, 128, 768), torch.float32), ((768, 768), torch.float32)]),
    (Matmul0, [((1, 128, 768), torch.float32), ((768, 3072), torch.float32)]),
    (Matmul0, [((1, 128, 3072), torch.float32), ((3072, 768), torch.float32)]),
    (Matmul0, [((1, 128, 768), torch.float32), ((768, 2), torch.float32)]),
    (Matmul0, [((1, 128, 768), torch.float32), ((768, 128), torch.float32)]),
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
