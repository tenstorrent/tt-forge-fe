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


class Squeeze0(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, squeeze_input_0):
        squeeze_output_1 = forge.op.Squeeze("", squeeze_input_0, dim=-1)
        return squeeze_output_1


class Squeeze1(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, squeeze_input_0):
        squeeze_output_1 = forge.op.Squeeze("", squeeze_input_0, dim=-2)
        return squeeze_output_1


class Squeeze2(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, squeeze_input_0):
        squeeze_output_1 = forge.op.Squeeze("", squeeze_input_0, dim=-4)
        return squeeze_output_1


class Squeeze3(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, squeeze_input_0):
        squeeze_output_1 = forge.op.Squeeze("", squeeze_input_0, dim=-3)
        return squeeze_output_1


class Squeeze4(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, squeeze_input_0):
        squeeze_output_1 = forge.op.Squeeze("", squeeze_input_0, dim=1)
        return squeeze_output_1


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (Squeeze0, [((1, 128, 2048, 1), torch.float32)]),
    (Squeeze0, [((1, 128, 1024, 1), torch.float32)]),
    (Squeeze0, [((1, 128, 4096, 1), torch.float32)]),
    (Squeeze0, [((1, 128, 768, 1), torch.float32)]),
    (Squeeze0, [((1, 384, 1), torch.float32)]),
    (Squeeze0, [((1, 256, 16, 32, 1), torch.float32)]),
    (Squeeze0, [((1, 128, 1), torch.float32)]),
    (Squeeze0, [((1, 32, 1), torch.float32)]),
    (Squeeze0, [((1, 1), torch.int32)]),
    (Squeeze1, [((1, 768, 1, 128), torch.float32)]),
    (Squeeze1, [((1, 3072, 1, 128), torch.float32)]),
    (Squeeze2, [((1, 1, 1024, 1), torch.float32)]),
    (Squeeze3, [((1, 1024, 1), torch.float32)]),
    (Squeeze0, [((1, 1024, 1), torch.float32)]),
    (Squeeze2, [((1, 1, 1024, 72), torch.float32)]),
    (Squeeze3, [((1, 1024, 72), torch.float32)]),
    (Squeeze1, [((1, 9216, 1, 1), torch.float32)]),
    (Squeeze0, [((1, 9216, 1), torch.float32)]),
    (Squeeze0, [((1, 768, 196, 1), torch.float32)]),
    (Squeeze0, [((1, 192, 196, 1), torch.float32)]),
    (Squeeze0, [((1, 384, 196, 1), torch.float32)]),
    (Squeeze1, [((1, 1920, 1, 1), torch.float32)]),
    (Squeeze0, [((1, 1920, 1), torch.float32)]),
    (Squeeze1, [((1, 1024, 1, 1), torch.float32)]),
    (Squeeze1, [((1, 2208, 1, 1), torch.float32)]),
    (Squeeze0, [((1, 2208, 1), torch.float32)]),
    (Squeeze1, [((1, 1664, 1, 1), torch.float32)]),
    (Squeeze0, [((1, 1664, 1), torch.float32)]),
    (Squeeze1, [((1, 1792, 1, 1), torch.float32)]),
    (Squeeze0, [((1, 1792, 1), torch.float32)]),
    (Squeeze1, [((1, 1280, 1, 1), torch.float32)]),
    (Squeeze0, [((1, 1280, 1), torch.float32)]),
    (Squeeze1, [((1, 2048, 1, 1), torch.float32)]),
    (Squeeze0, [((1, 2048, 1), torch.float32)]),
    (Squeeze1, [((1, 1536, 1, 1), torch.float32)]),
    (Squeeze0, [((1, 1536, 1), torch.float32)]),
    (Squeeze0, [((1, 768, 49, 1), torch.float32)]),
    (Squeeze4, [((1, 1, 768), torch.float32)]),
    (Squeeze0, [((1, 1024, 49, 1), torch.float32)]),
    (Squeeze4, [((1, 1, 1024), torch.float32)]),
    (Squeeze0, [((1, 512, 49, 1), torch.float32)]),
    (Squeeze4, [((1, 1, 512), torch.float32)]),
    (Squeeze0, [((1, 512, 196, 1), torch.float32)]),
    (Squeeze0, [((1, 1024, 196, 1), torch.float32)]),
    (Squeeze1, [((1, 768, 1, 1), torch.float32)]),
    (Squeeze0, [((1, 768, 1), torch.float32)]),
    (Squeeze1, [((1, 960, 1, 1), torch.float32)]),
    (Squeeze0, [((1, 960, 1), torch.float32)]),
    (Squeeze1, [((1, 576, 1, 1), torch.float32)]),
    (Squeeze0, [((1, 576, 1), torch.float32)]),
    (Squeeze1, [((1, 1, 1, 1024), torch.float32)]),
    (Squeeze1, [((1, 512, 1, 322), torch.float32)]),
    (Squeeze1, [((1, 3025, 1, 322), torch.float32)]),
    (Squeeze1, [((1, 512, 1, 1024), torch.float32)]),
    (Squeeze1, [((1, 512, 1, 512), torch.float32)]),
    (Squeeze1, [((1, 50176, 1, 512), torch.float32)]),
    (Squeeze1, [((1, 512, 1, 261), torch.float32)]),
    (Squeeze1, [((1, 50176, 1, 261), torch.float32)]),
    (Squeeze1, [((1, 1088, 1, 1), torch.float32)]),
    (Squeeze0, [((1, 1088, 1), torch.float32)]),
    (Squeeze0, [((1, 32, 16384, 1), torch.float32)]),
    (Squeeze1, [((1, 16384, 1, 32), torch.float32)]),
    (Squeeze1, [((1, 256, 1, 32), torch.float32)]),
    (Squeeze0, [((1, 128, 16384, 1), torch.float32)]),
    (Squeeze0, [((1, 64, 4096, 1), torch.float32)]),
    (Squeeze0, [((1, 256, 4096, 1), torch.float32)]),
    (Squeeze0, [((1, 160, 1024, 1), torch.float32)]),
    (Squeeze0, [((1, 640, 1024, 1), torch.float32)]),
    (Squeeze0, [((1, 256, 256, 1), torch.float32)]),
    (Squeeze0, [((1, 1024, 256, 1), torch.float32)]),
    (Squeeze0, [((1, 64, 16384, 1), torch.float32)]),
    (Squeeze1, [((1, 16384, 1, 64), torch.float32)]),
    (Squeeze1, [((1, 256, 1, 64), torch.float32)]),
    (Squeeze0, [((1, 256, 16384, 1), torch.float32)]),
    (Squeeze0, [((1, 512, 4096, 1), torch.float32)]),
    (Squeeze0, [((1, 320, 1024, 1), torch.float32)]),
    (Squeeze0, [((1, 1280, 1024, 1), torch.float32)]),
    (Squeeze0, [((1, 512, 256, 1), torch.float32)]),
    (Squeeze0, [((1, 2048, 256, 1), torch.float32)]),
    (Squeeze4, [((1, 1, 256), torch.float32)]),
    (Squeeze1, [((1, 25088, 1, 1), torch.float32)]),
    (Squeeze0, [((1, 25088, 1), torch.float32)]),
    (Squeeze1, [((1, 4096, 1, 1), torch.float32)]),
    (Squeeze0, [((1, 4096, 1), torch.float32)]),
    (Squeeze0, [((1, 85, 6400, 1), torch.float32)]),
    (Squeeze0, [((1, 85, 1600, 1), torch.float32)]),
    (Squeeze0, [((1, 85, 400, 1), torch.float32)]),
    (Squeeze0, [((1, 85, 2704, 1), torch.float32)]),
    (Squeeze0, [((1, 85, 676, 1), torch.float32)]),
    (Squeeze0, [((1, 85, 169, 1), torch.float32)]),
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
