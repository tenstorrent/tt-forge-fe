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


class Cast0(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, cast_input_0):
        cast_output_1 = forge.op.Cast("", cast_input_0, dtype=torch.bfloat16)
        return cast_output_1


class Cast1(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, cast_input_0):
        cast_output_1 = forge.op.Cast("", cast_input_0, dtype=torch.float32)
        return cast_output_1


class Cast2(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, cast_input_0):
        cast_output_1 = forge.op.Cast("", cast_input_0, dtype=torch.bool)
        return cast_output_1


class Cast3(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, cast_input_0):
        cast_output_1 = forge.op.Cast("", cast_input_0, dtype=torch.int32)
        return cast_output_1


class Cast4(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, cast_input_0):
        cast_output_1 = forge.op.Cast("", cast_input_0, dtype=torch.int64)
        return cast_output_1


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (Cast0, [((2049, 1024), torch.float32)]),
    (Cast1, [((2, 1, 1024), torch.bfloat16)]),
    (Cast0, [((32128, 768), torch.float32)]),
    (Cast1, [((2, 13, 768), torch.bfloat16)]),
    (Cast0, [((32, 12), torch.float32)]),
    (Cast1, [((13, 13, 12), torch.bfloat16)]),
    (Cast1, [((2, 1, 1, 13), torch.float32)]),
    (Cast1, [((2, 13, 1), torch.float32)]),
    (Cast1, [((2, 1, 1, 13), torch.float32)]),
    (Cast0, [((2049, 2048), torch.float32)]),
    (Cast1, [((2, 1, 2048), torch.bfloat16)]),
    (Cast0, [((2049, 1536), torch.float32)]),
    (Cast1, [((2, 1, 1536), torch.bfloat16)]),
    (Cast0, [((51865, 1024), torch.float32)]),
    (Cast1, [((1, 2, 1024), torch.bfloat16)]),
    (Cast0, [((51865, 512), torch.float32)]),
    (Cast1, [((1, 2, 512), torch.bfloat16)]),
    (Cast0, [((51865, 1280), torch.float32)]),
    (Cast1, [((1, 2, 1280), torch.bfloat16)]),
    (Cast0, [((51865, 768), torch.float32)]),
    (Cast1, [((1, 2, 768), torch.bfloat16)]),
    (Cast0, [((51865, 384), torch.float32)]),
    (Cast1, [((1, 2, 384), torch.bfloat16)]),
    (Cast0, [((51866, 1280), torch.float32)]),
    (Cast0, [((49408, 512), torch.float32)]),
    (Cast1, [((2, 7, 512), torch.bfloat16)]),
    (Cast0, [((77, 512), torch.float32)]),
    (Cast1, [((1, 7, 512), torch.bfloat16)]),
    (Cast1, [((2, 1, 7, 7), torch.float32)]),
    (Cast1, [((2, 1, 7, 7), torch.float32)]),
    (Cast0, [((30000, 128), torch.float32)]),
    (Cast1, [((1, 128, 128), torch.bfloat16)]),
    (Cast0, [((2, 128), torch.float32)]),
    (Cast0, [((512, 128), torch.float32)]),
    (Cast1, [((1, 1, 1, 128), torch.float32)]),
    (Cast0, [((50265, 1024), torch.float32)]),
    (Cast1, [((1, 256, 1024), torch.bfloat16)]),
    (Cast0, [((1026, 1024), torch.float32)]),
    (Cast1, [((1, 1, 256, 256), torch.float32)]),
    (Cast1, [((1, 1, 256, 256), torch.float32)]),
    (Cast2, [((1, 1, 256, 256), torch.float32)]),
    (Cast3, [((1, 1, 256, 256), torch.float32)]),
    (Cast0, [((28996, 1024), torch.float32)]),
    (Cast1, [((1, 384, 1024), torch.bfloat16)]),
    (Cast0, [((2, 1024), torch.float32)]),
    (Cast0, [((512, 1024), torch.float32)]),
    (Cast1, [((1, 128, 1024), torch.bfloat16)]),
    (Cast0, [((30522, 768), torch.float32)]),
    (Cast1, [((1, 128, 768), torch.bfloat16)]),
    (Cast0, [((2, 768), torch.float32)]),
    (Cast0, [((512, 768), torch.float32)]),
    (Cast0, [((51200, 1024), torch.float32)]),
    (Cast2, [((1, 128), torch.float32)]),
    (Cast3, [((1, 128), torch.float32)]),
    (Cast2, [((1, 128), torch.int32)]),
    (Cast4, [((1, 128), torch.int32)]),
    (Cast1, [((1, 12, 128, 128), torch.float32)]),
    (Cast0, [((119547, 768), torch.float32)]),
    (Cast0, [((28996, 768), torch.float32)]),
    (Cast1, [((1, 384, 768), torch.bfloat16)]),
    (Cast2, [((1, 384), torch.float32)]),
    (Cast3, [((1, 384), torch.float32)]),
    (Cast2, [((1, 384), torch.int32)]),
    (Cast1, [((1, 12, 384, 384), torch.float32)]),
    (Cast0, [((65024, 4544), torch.float32)]),
    (Cast1, [((1, 6, 4544), torch.bfloat16)]),
    (Cast0, [((256000, 2048), torch.float32)]),
    (Cast1, [((1, 7, 2048), torch.bfloat16)]),
    (Cast0, [((50257, 768), torch.float32)]),
    (Cast1, [((1, 256, 768), torch.bfloat16)]),
    (Cast0, [((1024, 768), torch.float32)]),
    (Cast1, [((1, 32, 768), torch.bfloat16)]),
    (Cast0, [((2048, 768), torch.float32)]),
    (Cast2, [((1, 1, 32, 32), torch.float32)]),
    (Cast1, [((1, 1, 32, 32), torch.float32)]),
    (Cast0, [((50257, 2048), torch.float32)]),
    (Cast1, [((1, 32, 2048), torch.bfloat16)]),
    (Cast0, [((2048, 2048), torch.float32)]),
    (Cast1, [((1, 256, 2048), torch.bfloat16)]),
    (Cast0, [((50257, 2560), torch.float32)]),
    (Cast1, [((1, 256, 2560), torch.bfloat16)]),
    (Cast0, [((2048, 2560), torch.float32)]),
    (Cast1, [((1, 32, 2560), torch.bfloat16)]),
    (Cast0, [((128256, 4096), torch.float32)]),
    (Cast1, [((1, 256, 4096), torch.bfloat16)]),
    (Cast2, [((1, 1, 256, 256), torch.float32)]),
    (Cast2, [((1, 1, 256, 256), torch.int32)]),
    (Cast0, [((128256, 2048), torch.float32)]),
    (Cast1, [((1, 4, 2048), torch.bfloat16)]),
    (Cast1, [((1, 4, 4096), torch.bfloat16)]),
    (Cast0, [((32000, 4096), torch.float32)]),
    (Cast1, [((1, 128, 4096), torch.bfloat16)]),
    (Cast0, [((50272, 2048), torch.float32)]),
    (Cast0, [((2050, 2048), torch.float32)]),
    (Cast1, [((1, 1, 32, 32), torch.float32)]),
    (Cast0, [((50272, 512), torch.float32)]),
    (Cast1, [((1, 32, 512), torch.bfloat16)]),
    (Cast0, [((2050, 1024), torch.float32)]),
    (Cast1, [((1, 32, 1024), torch.bfloat16)]),
    (Cast0, [((50272, 768), torch.float32)]),
    (Cast0, [((2050, 768), torch.float32)]),
    (Cast3, [((1, 32), torch.float32)]),
    (Cast4, [((1,), torch.int32)]),
    (Cast1, [((1, 256, 512), torch.bfloat16)]),
    (Cast0, [((51200, 2560), torch.float32)]),
    (Cast1, [((1, 12, 2560), torch.bfloat16)]),
    (Cast1, [((1, 11, 2560), torch.bfloat16)]),
    (Cast0, [((151936, 1024), torch.float32)]),
    (Cast1, [((1, 6, 1024), torch.bfloat16)]),
    (Cast1, [((1, 29, 1024), torch.bfloat16)]),
    (Cast0, [((151936, 1536), torch.float32)]),
    (Cast1, [((1, 35, 1536), torch.bfloat16)]),
    (Cast0, [((152064, 3584), torch.float32)]),
    (Cast1, [((1, 35, 3584), torch.bfloat16)]),
    (Cast0, [((151936, 2048), torch.float32)]),
    (Cast1, [((1, 35, 2048), torch.bfloat16)]),
    (Cast0, [((151936, 896), torch.float32)]),
    (Cast1, [((1, 35, 896), torch.bfloat16)]),
    (Cast1, [((1, 29, 896), torch.bfloat16)]),
    (Cast1, [((1, 39, 1536), torch.bfloat16)]),
    (Cast1, [((1, 29, 1536), torch.bfloat16)]),
    (Cast1, [((1, 29, 2048), torch.bfloat16)]),
    (Cast1, [((1, 39, 2048), torch.bfloat16)]),
    (Cast1, [((1, 39, 896), torch.bfloat16)]),
    (Cast1, [((1, 39, 3584), torch.bfloat16)]),
    (Cast1, [((1, 29, 3584), torch.bfloat16)]),
    (Cast0, [((50265, 768), torch.float32)]),
    (Cast0, [((1, 768), torch.float32)]),
    (Cast0, [((514, 768), torch.float32)]),
    (Cast0, [((250002, 768), torch.float32)]),
    (Cast0, [((30528, 768), torch.float32)]),
    (Cast0, [((32128, 512), torch.float32)]),
    (Cast1, [((1, 1, 512), torch.bfloat16)]),
    (Cast0, [((32, 6), torch.float32)]),
    (Cast1, [((1, 1, 6), torch.bfloat16)]),
    (Cast1, [((1, 1, 768), torch.bfloat16)]),
    (Cast1, [((1, 1, 12), torch.bfloat16)]),
    (Cast0, [((32128, 1024), torch.float32)]),
    (Cast1, [((1, 1, 1024), torch.bfloat16)]),
    (Cast0, [((32, 16), torch.float32)]),
    (Cast1, [((1, 1, 16), torch.bfloat16)]),
    (Cast0, [((32, 8), torch.float32)]),
    (Cast1, [((1, 1, 8), torch.bfloat16)]),
    (Cast0, [((256008, 2048), torch.float32)]),
    (Cast0, [((256008, 1024), torch.float32)]),
    (Cast2, [((1, 16, 320, 1024), torch.float32)]),
    (Cast2, [((1, 16, 192, 640), torch.float32)]),
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
