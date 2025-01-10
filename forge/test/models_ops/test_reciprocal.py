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


class Reciprocal0(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reciprocal_input_0):
        reciprocal_output_1 = forge.op.Reciprocal("", reciprocal_input_0)
        return reciprocal_output_1


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (Reciprocal0, [((2, 13, 1), torch.float32)]),
    (Reciprocal0, [((1, 7, 1), torch.float32)]),
    (Reciprocal0, [((1, 256, 1), torch.float32)]),
    (Reciprocal0, [((1, 4, 1), torch.float32)]),
    (Reciprocal0, [((1, 128, 1), torch.float32)]),
    (Reciprocal0, [((1, 6, 1), torch.float32)]),
    (Reciprocal0, [((1, 29, 1), torch.float32)]),
    (Reciprocal0, [((1, 35, 1), torch.float32)]),
    (Reciprocal0, [((1, 39, 1), torch.float32)]),
    (Reciprocal0, [((1, 1, 1), torch.float32)]),
    (Reciprocal0, [((64,), torch.float32)]),
    (Reciprocal0, [((128,), torch.float32)]),
    (Reciprocal0, [((96,), torch.float32)]),
    (Reciprocal0, [((160,), torch.float32)]),
    (Reciprocal0, [((192,), torch.float32)]),
    (Reciprocal0, [((224,), torch.float32)]),
    (Reciprocal0, [((256,), torch.float32)]),
    (Reciprocal0, [((288,), torch.float32)]),
    (Reciprocal0, [((320,), torch.float32)]),
    (Reciprocal0, [((352,), torch.float32)]),
    (Reciprocal0, [((384,), torch.float32)]),
    (Reciprocal0, [((416,), torch.float32)]),
    (Reciprocal0, [((448,), torch.float32)]),
    (Reciprocal0, [((480,), torch.float32)]),
    (Reciprocal0, [((512,), torch.float32)]),
    (Reciprocal0, [((544,), torch.float32)]),
    (Reciprocal0, [((576,), torch.float32)]),
    (Reciprocal0, [((608,), torch.float32)]),
    (Reciprocal0, [((640,), torch.float32)]),
    (Reciprocal0, [((672,), torch.float32)]),
    (Reciprocal0, [((704,), torch.float32)]),
    (Reciprocal0, [((736,), torch.float32)]),
    (Reciprocal0, [((768,), torch.float32)]),
    (Reciprocal0, [((800,), torch.float32)]),
    (Reciprocal0, [((832,), torch.float32)]),
    (Reciprocal0, [((864,), torch.float32)]),
    (Reciprocal0, [((896,), torch.float32)]),
    (Reciprocal0, [((928,), torch.float32)]),
    (Reciprocal0, [((960,), torch.float32)]),
    (Reciprocal0, [((992,), torch.float32)]),
    (Reciprocal0, [((1024,), torch.float32)]),
    (Reciprocal0, [((1056,), torch.float32)]),
    (Reciprocal0, [((1088,), torch.float32)]),
    (Reciprocal0, [((1120,), torch.float32)]),
    (Reciprocal0, [((1152,), torch.float32)]),
    (Reciprocal0, [((1184,), torch.float32)]),
    (Reciprocal0, [((1216,), torch.float32)]),
    (Reciprocal0, [((1248,), torch.float32)]),
    (Reciprocal0, [((1280,), torch.float32)]),
    (Reciprocal0, [((1312,), torch.float32)]),
    (Reciprocal0, [((1344,), torch.float32)]),
    (Reciprocal0, [((1376,), torch.float32)]),
    (Reciprocal0, [((1408,), torch.float32)]),
    (Reciprocal0, [((1440,), torch.float32)]),
    (Reciprocal0, [((1472,), torch.float32)]),
    (Reciprocal0, [((1504,), torch.float32)]),
    (Reciprocal0, [((1536,), torch.float32)]),
    (Reciprocal0, [((1568,), torch.float32)]),
    (Reciprocal0, [((1600,), torch.float32)]),
    (Reciprocal0, [((1632,), torch.float32)]),
    (Reciprocal0, [((1664,), torch.float32)]),
    (Reciprocal0, [((1696,), torch.float32)]),
    (Reciprocal0, [((1728,), torch.float32)]),
    (Reciprocal0, [((1760,), torch.float32)]),
    (Reciprocal0, [((1792,), torch.float32)]),
    (Reciprocal0, [((1824,), torch.float32)]),
    (Reciprocal0, [((1856,), torch.float32)]),
    (Reciprocal0, [((1888,), torch.float32)]),
    (Reciprocal0, [((1920,), torch.float32)]),
    (Reciprocal0, [((144,), torch.float32)]),
    (Reciprocal0, [((240,), torch.float32)]),
    (Reciprocal0, [((336,), torch.float32)]),
    (Reciprocal0, [((432,), torch.float32)]),
    (Reciprocal0, [((528,), torch.float32)]),
    (Reciprocal0, [((624,), torch.float32)]),
    (Reciprocal0, [((720,), torch.float32)]),
    (Reciprocal0, [((816,), torch.float32)]),
    (Reciprocal0, [((912,), torch.float32)]),
    (Reciprocal0, [((1008,), torch.float32)]),
    (Reciprocal0, [((1104,), torch.float32)]),
    (Reciprocal0, [((1200,), torch.float32)]),
    (Reciprocal0, [((1296,), torch.float32)]),
    (Reciprocal0, [((1392,), torch.float32)]),
    (Reciprocal0, [((1488,), torch.float32)]),
    (Reciprocal0, [((1584,), torch.float32)]),
    (Reciprocal0, [((1680,), torch.float32)]),
    (Reciprocal0, [((1776,), torch.float32)]),
    (Reciprocal0, [((1872,), torch.float32)]),
    (Reciprocal0, [((1968,), torch.float32)]),
    (Reciprocal0, [((2016,), torch.float32)]),
    (Reciprocal0, [((2064,), torch.float32)]),
    (Reciprocal0, [((2112,), torch.float32)]),
    (Reciprocal0, [((2160,), torch.float32)]),
    (Reciprocal0, [((2208,), torch.float32)]),
    (Reciprocal0, [((16,), torch.float32)]),
    (Reciprocal0, [((32,), torch.float32)]),
    (Reciprocal0, [((2048,), torch.float32)]),
    (Reciprocal0, [((48,), torch.float32)]),
    (Reciprocal0, [((24,), torch.float32)]),
    (Reciprocal0, [((56,), torch.float32)]),
    (Reciprocal0, [((112,), torch.float32)]),
    (Reciprocal0, [((272,), torch.float32)]),
    (Reciprocal0, [((2688,), torch.float32)]),
    (Reciprocal0, [((40,), torch.float32)]),
    (Reciprocal0, [((80,), torch.float32)]),
    (Reciprocal0, [((8,), torch.float32)]),
    (Reciprocal0, [((12,), torch.float32)]),
    (Reciprocal0, [((36,), torch.float32)]),
    (Reciprocal0, [((72,), torch.float32)]),
    (Reciprocal0, [((20,), torch.float32)]),
    (Reciprocal0, [((60,), torch.float32)]),
    (Reciprocal0, [((120,), torch.float32)]),
    (Reciprocal0, [((100,), torch.float32)]),
    (Reciprocal0, [((92,), torch.float32)]),
    (Reciprocal0, [((208,), torch.float32)]),
    (Reciprocal0, [((18,), torch.float32)]),
    (Reciprocal0, [((44,), torch.float32)]),
    (Reciprocal0, [((88,), torch.float32)]),
    (Reciprocal0, [((176,), torch.float32)]),
    (Reciprocal0, [((30,), torch.float32)]),
    (Reciprocal0, [((200,), torch.float32)]),
    (Reciprocal0, [((184,), torch.float32)]),
    (Reciprocal0, [((728,), torch.float32)]),
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
