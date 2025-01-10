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


class Identity0(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, identity_input_0):
        identity_output_1 = forge.op.Identity("", identity_input_0)
        return identity_output_1


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (Identity0, [((2, 1, 1024), torch.float32)]),
    (Identity0, [((32, 1, 1), torch.float32)]),
    (Identity0, [((2, 13, 768), torch.float32)]),
    (Identity0, [((2, 12, 13, 13), torch.float32)]),
    (Identity0, [((2, 13, 3072), torch.float32)]),
    (Identity0, [((32, 1, 13), torch.float32)]),
    (Identity0, [((2, 1, 4096), torch.float32)]),
    (Identity0, [((2, 1, 2048), torch.float32)]),
    (Identity0, [((64, 1, 1), torch.float32)]),
    (Identity0, [((64, 1, 13), torch.float32)]),
    (Identity0, [((2, 1, 8192), torch.float32)]),
    (Identity0, [((2, 1, 1536), torch.float32)]),
    (Identity0, [((48, 1, 1), torch.float32)]),
    (Identity0, [((48, 1, 13), torch.float32)]),
    (Identity0, [((2, 1, 6144), torch.float32)]),
    (Identity0, [((1, 2, 1024), torch.float32)]),
    (Identity0, [((1, 16, 2, 2), torch.float32)]),
    (Identity0, [((1, 16, 2, 1500), torch.float32)]),
    (Identity0, [((1, 2, 4096), torch.float32)]),
    (Identity0, [((1, 2, 512), torch.float32)]),
    (Identity0, [((1, 8, 2, 2), torch.float32)]),
    (Identity0, [((1, 8, 2, 1500), torch.float32)]),
    (Identity0, [((1, 2, 2048), torch.float32)]),
    (Identity0, [((1, 2, 1280), torch.float32)]),
    (Identity0, [((1, 20, 2, 2), torch.float32)]),
    (Identity0, [((1, 20, 2, 1500), torch.float32)]),
    (Identity0, [((1, 2, 5120), torch.float32)]),
    (Identity0, [((1, 2, 768), torch.float32)]),
    (Identity0, [((1, 12, 2, 2), torch.float32)]),
    (Identity0, [((1, 12, 2, 1500), torch.float32)]),
    (Identity0, [((1, 2, 3072), torch.float32)]),
    (Identity0, [((1, 2, 384), torch.float32)]),
    (Identity0, [((1, 6, 2, 2), torch.float32)]),
    (Identity0, [((1, 6, 2, 1500), torch.float32)]),
    (Identity0, [((1, 2, 1536), torch.float32)]),
    (Identity0, [((16, 7, 7), torch.float32)]),
    (Identity0, [((1, 12, 204, 204), torch.float32)]),
    (Identity0, [((1, 204, 768), torch.float32)]),
    (Identity0, [((1, 12, 201, 201), torch.float32)]),
    (Identity0, [((1, 201, 768), torch.float32)]),
    (Identity0, [((1, 128, 128), torch.float32)]),
    (Identity0, [((1, 16, 128, 128), torch.float32)]),
    (Identity0, [((1, 128, 2048), torch.float32)]),
    (Identity0, [((1, 128, 1024), torch.float32)]),
    (Identity0, [((1, 64, 128, 128), torch.float32)]),
    (Identity0, [((1, 128, 4096), torch.float32)]),
    (Identity0, [((1, 12, 128, 128), torch.float32)]),
    (Identity0, [((1, 128, 768), torch.float32)]),
    (Identity0, [((1, 256, 1024), torch.float32)]),
    (Identity0, [((16, 256, 256), torch.float32)]),
    (Identity0, [((1, 256, 4096), torch.float32)]),
    (Identity0, [((1, 384, 1024), torch.float32)]),
    (Identity0, [((1, 16, 384, 384), torch.float32)]),
    (Identity0, [((1, 16, 256, 256), torch.float32)]),
    (Identity0, [((1, 768), torch.float32)]),
    (Identity0, [((1, 384, 768), torch.float32)]),
    (Identity0, [((1, 12, 384, 384), torch.float32)]),
    (Identity0, [((1, 71, 6, 6), torch.float32)]),
    (Identity0, [((1, 6, 4544), torch.float32)]),
    (Identity0, [((1, 64, 334, 334), torch.float32)]),
    (Identity0, [((1, 334, 4096), torch.float32)]),
    (Identity0, [((1, 8, 7, 7), torch.float32)]),
    (Identity0, [((1, 256, 768), torch.float32)]),
    (Identity0, [((1, 12, 256, 256), torch.float32)]),
    (Identity0, [((1, 32, 768), torch.float32)]),
    (Identity0, [((1, 12, 32, 32), torch.float32)]),
    (Identity0, [((1, 32, 2048), torch.float32)]),
    (Identity0, [((1, 16, 32, 32), torch.float32)]),
    (Identity0, [((1, 256, 2048), torch.float32)]),
    (Identity0, [((1, 256, 2560), torch.float32)]),
    (Identity0, [((1, 20, 256, 256), torch.float32)]),
    (Identity0, [((1, 32, 2560), torch.float32)]),
    (Identity0, [((1, 20, 32, 32), torch.float32)]),
    (Identity0, [((1, 32, 256, 256), torch.float32)]),
    (Identity0, [((1, 32, 4, 4), torch.float32)]),
    (Identity0, [((1, 32, 128, 128), torch.float32)]),
    (Identity0, [((32, 32, 32), torch.float32)]),
    (Identity0, [((32, 2048), torch.float32)]),
    (Identity0, [((32, 256, 256), torch.float32)]),
    (Identity0, [((256, 2048), torch.float32)]),
    (Identity0, [((16, 32, 32), torch.float32)]),
    (Identity0, [((1, 32, 1024), torch.float32)]),
    (Identity0, [((32, 1024), torch.float32)]),
    (Identity0, [((12, 32, 32), torch.float32)]),
    (Identity0, [((32, 768), torch.float32)]),
    (Identity0, [((12, 256, 256), torch.float32)]),
    (Identity0, [((256, 768), torch.float32)]),
    (Identity0, [((256, 1024), torch.float32)]),
    (Identity0, [((1, 12, 2560), torch.float32)]),
    (Identity0, [((1, 32, 12, 12), torch.float32)]),
    (Identity0, [((1, 11, 2560), torch.float32)]),
    (Identity0, [((1, 32, 11, 11), torch.float32)]),
    (Identity0, [((1, 16, 6, 6), torch.float32)]),
    (Identity0, [((1, 16, 29, 29), torch.float32)]),
    (Identity0, [((1, 12, 35, 35), torch.float32)]),
    (Identity0, [((1, 28, 35, 35), torch.float32)]),
    (Identity0, [((1, 16, 35, 35), torch.float32)]),
    (Identity0, [((1, 14, 35, 35), torch.float32)]),
    (Identity0, [((1, 14, 29, 29), torch.float32)]),
    (Identity0, [((1, 12, 39, 39), torch.float32)]),
    (Identity0, [((1, 12, 29, 29), torch.float32)]),
    (Identity0, [((1, 16, 39, 39), torch.float32)]),
    (Identity0, [((1, 14, 39, 39), torch.float32)]),
    (Identity0, [((1, 28, 39, 39), torch.float32)]),
    (Identity0, [((1, 28, 29, 29), torch.float32)]),
    (Identity0, [((1, 768, 128), torch.float32)]),
    (Identity0, [((1, 1, 512), torch.float32)]),
    (Identity0, [((1, 6, 1, 1), torch.float32)]),
    (Identity0, [((1, 1, 1024), torch.float32)]),
    (Identity0, [((1, 1, 768), torch.float32)]),
    (Identity0, [((1, 12, 1, 1), torch.float32)]),
    (Identity0, [((1, 12, 1, 256), torch.float32)]),
    (Identity0, [((1, 1, 3072), torch.float32)]),
    (Identity0, [((1, 16, 1, 1), torch.float32)]),
    (Identity0, [((1, 16, 1, 256), torch.float32)]),
    (Identity0, [((1, 1, 4096), torch.float32)]),
    (Identity0, [((1, 1, 2048), torch.float32)]),
    (Identity0, [((1, 8, 1, 1), torch.float32)]),
    (Identity0, [((1, 256, 8192), torch.float32)]),
    (Identity0, [((1, 9216), torch.float32)]),
    (Identity0, [((1, 4096), torch.float32)]),
    (Identity0, [((1, 197, 768), torch.float32)]),
    (Identity0, [((1, 12, 197, 197), torch.float32)]),
    (Identity0, [((1, 197, 192), torch.float32)]),
    (Identity0, [((1, 3, 197, 197), torch.float32)]),
    (Identity0, [((1, 197, 384), torch.float32)]),
    (Identity0, [((1, 6, 197, 197), torch.float32)]),
    (Identity0, [((1, 1792), torch.float32)]),
    (Identity0, [((1, 1280), torch.float32)]),
    (Identity0, [((1, 1024), torch.float32)]),
    (Identity0, [((1, 768, 384), torch.float32)]),
    (Identity0, [((1, 768, 49), torch.float32)]),
    (Identity0, [((1, 49, 3072), torch.float32)]),
    (Identity0, [((1, 49, 768), torch.float32)]),
    (Identity0, [((1, 1024, 512), torch.float32)]),
    (Identity0, [((1, 1024, 49), torch.float32)]),
    (Identity0, [((1, 49, 4096), torch.float32)]),
    (Identity0, [((1, 49, 1024), torch.float32)]),
    (Identity0, [((1, 768, 196), torch.float32)]),
    (Identity0, [((1, 196, 3072), torch.float32)]),
    (Identity0, [((1, 196, 768), torch.float32)]),
    (Identity0, [((1, 512, 256), torch.float32)]),
    (Identity0, [((1, 512, 49), torch.float32)]),
    (Identity0, [((1, 49, 2048), torch.float32)]),
    (Identity0, [((1, 49, 512), torch.float32)]),
    (Identity0, [((1, 512, 196), torch.float32)]),
    (Identity0, [((1, 196, 2048), torch.float32)]),
    (Identity0, [((1, 196, 512), torch.float32)]),
    (Identity0, [((1, 1024, 196), torch.float32)]),
    (Identity0, [((1, 196, 4096), torch.float32)]),
    (Identity0, [((1, 196, 1024), torch.float32)]),
    (Identity0, [((1, 256, 28, 28), torch.float32)]),
    (Identity0, [((1, 1, 512, 3025), torch.float32)]),
    (Identity0, [((1, 8, 512, 512), torch.float32)]),
    (Identity0, [((1, 1, 1, 512), torch.float32)]),
    (Identity0, [((1, 1, 512, 50176), torch.float32)]),
    (Identity0, [((1, 1, 16384, 256), torch.float32)]),
    (Identity0, [((1, 16384, 32), torch.float32)]),
    (Identity0, [((1, 16384, 128), torch.float32)]),
    (Identity0, [((1, 2, 4096, 256), torch.float32)]),
    (Identity0, [((1, 4096, 64), torch.float32)]),
    (Identity0, [((1, 4096, 256), torch.float32)]),
    (Identity0, [((1, 5, 1024, 256), torch.float32)]),
    (Identity0, [((1, 1024, 160), torch.float32)]),
    (Identity0, [((1, 1024, 640), torch.float32)]),
    (Identity0, [((1, 8, 256, 256), torch.float32)]),
    (Identity0, [((1, 256, 256), torch.float32)]),
    (Identity0, [((1, 256, 128, 128), torch.float32)]),
    (Identity0, [((1, 16384, 64), torch.float32)]),
    (Identity0, [((1, 16384, 256), torch.float32)]),
    (Identity0, [((1, 4096, 128), torch.float32)]),
    (Identity0, [((1, 4096, 512), torch.float32)]),
    (Identity0, [((1, 1024, 320), torch.float32)]),
    (Identity0, [((1, 1024, 1280), torch.float32)]),
    (Identity0, [((1, 256, 512), torch.float32)]),
    (Identity0, [((1, 768, 128, 128), torch.float32)]),
    (Identity0, [((1, 4096, 1, 1), torch.float32)]),
    (Identity0, [((1, 197, 1024), torch.float32)]),
    (Identity0, [((1, 16, 197, 197), torch.float32)]),
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
