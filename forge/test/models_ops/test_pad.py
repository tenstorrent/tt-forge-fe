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


class Pad0(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, pad_input_0):
        pad_output_1 = forge.op.Pad("", pad_input_0, pad=(1, 1, 1, 1), mode="reflect", channel_last=False)
        return pad_output_1


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (Pad0, [((1, 512, 10, 32), torch.float32)]),
    (Pad0, [((1, 512, 20, 64), torch.float32)]),
    (Pad0, [((1, 256, 20, 64), torch.float32)]),
    (Pad0, [((1, 256, 40, 128), torch.float32)]),
    (Pad0, [((1, 128, 40, 128), torch.float32)]),
    (Pad0, [((1, 128, 80, 256), torch.float32)]),
    (Pad0, [((1, 64, 80, 256), torch.float32)]),
    (Pad0, [((1, 96, 160, 512), torch.float32)]),
    (Pad0, [((1, 32, 160, 512), torch.float32)]),
    (Pad0, [((1, 16, 320, 1024), torch.float32)]),
    (Pad0, [((1, 512, 6, 20), torch.float32)]),
    (Pad0, [((1, 512, 12, 40), torch.float32)]),
    (Pad0, [((1, 256, 12, 40), torch.float32)]),
    (Pad0, [((1, 256, 24, 80), torch.float32)]),
    (Pad0, [((1, 128, 24, 80), torch.float32)]),
    (Pad0, [((1, 128, 48, 160), torch.float32)]),
    (Pad0, [((1, 64, 48, 160), torch.float32)]),
    (Pad0, [((1, 96, 96, 320), torch.float32)]),
    (Pad0, [((1, 32, 96, 320), torch.float32)]),
    (Pad0, [((1, 16, 192, 640), torch.float32)]),
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
