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


class Embedding0(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, embedding_input_0, embedding_input_1):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, embedding_input_1)
        return embedding_output_1


class Embedding1(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("embedding1_const_0", shape=(13, 13), dtype=torch.int32)

    def forward(self, embedding_input_1):
        embedding_output_1 = forge.op.Embedding("", self.get_constant("embedding1_const_0"), embedding_input_1)
        return embedding_output_1


class Embedding2(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("embedding2_const_0", shape=(1, 256), dtype=torch.int64)

    def forward(self, embedding_input_1):
        embedding_output_1 = forge.op.Embedding("", self.get_constant("embedding2_const_0"), embedding_input_1)
        return embedding_output_1


class Embedding3(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("embedding3_const_0", shape=(1, 32), dtype=torch.int64)

    def forward(self, embedding_input_1):
        embedding_output_1 = forge.op.Embedding("", self.get_constant("embedding3_const_0"), embedding_input_1)
        return embedding_output_1


class Embedding4(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("embedding4_const_0", shape=(1, 128), dtype=torch.int64)

    def forward(self, embedding_input_1):
        embedding_output_1 = forge.op.Embedding("", self.get_constant("embedding4_const_0"), embedding_input_1)
        return embedding_output_1


class Embedding5(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("embedding5_const_0", shape=(1, 1), dtype=torch.int32)

    def forward(self, embedding_input_1):
        embedding_output_1 = forge.op.Embedding("", self.get_constant("embedding5_const_0"), embedding_input_1)
        return embedding_output_1


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (Embedding0, [((2, 1), torch.float32), ((2049, 1024), torch.bfloat16)]),
    (Embedding0, [((2, 13), torch.float32), ((32128, 768), torch.bfloat16)]),
    (Embedding1, [((32, 12), torch.bfloat16)]),
    (Embedding0, [((2, 1), torch.float32), ((2049, 2048), torch.bfloat16)]),
    (Embedding0, [((2, 1), torch.float32), ((2049, 1536), torch.bfloat16)]),
    (Embedding0, [((1, 2), torch.int32), ((51865, 1024), torch.bfloat16)]),
    (Embedding0, [((1, 2), torch.int32), ((51865, 512), torch.bfloat16)]),
    (Embedding0, [((1, 2), torch.int32), ((51865, 1280), torch.bfloat16)]),
    (Embedding0, [((1, 2), torch.int32), ((51865, 768), torch.bfloat16)]),
    (Embedding0, [((1, 2), torch.int32), ((51865, 384), torch.bfloat16)]),
    (Embedding0, [((1, 2), torch.int32), ((51866, 1280), torch.bfloat16)]),
    (Embedding0, [((2, 7), torch.float32), ((49408, 512), torch.bfloat16)]),
    (Embedding0, [((1, 7), torch.float32), ((77, 512), torch.bfloat16)]),
    (Embedding0, [((1, 128), torch.float32), ((30000, 128), torch.bfloat16)]),
    (Embedding0, [((1, 128), torch.float32), ((2, 128), torch.bfloat16)]),
    (Embedding0, [((1, 128), torch.float32), ((512, 128), torch.bfloat16)]),
    (Embedding0, [((1, 256), torch.float32), ((50265, 1024), torch.bfloat16)]),
    (Embedding2, [((1026, 1024), torch.bfloat16)]),
    (Embedding0, [((1, 384), torch.float32), ((28996, 1024), torch.bfloat16)]),
    (Embedding0, [((1, 384), torch.float32), ((2, 1024), torch.bfloat16)]),
    (Embedding0, [((1, 384), torch.float32), ((512, 1024), torch.bfloat16)]),
    (Embedding0, [((1, 128), torch.float32), ((28996, 1024), torch.bfloat16)]),
    (Embedding0, [((1, 128), torch.float32), ((2, 1024), torch.bfloat16)]),
    (Embedding0, [((1, 128), torch.float32), ((512, 1024), torch.bfloat16)]),
    (Embedding0, [((1, 128), torch.float32), ((30522, 768), torch.bfloat16)]),
    (Embedding0, [((1, 128), torch.float32), ((2, 768), torch.bfloat16)]),
    (Embedding0, [((1, 128), torch.float32), ((512, 768), torch.bfloat16)]),
    (Embedding0, [((1, 256), torch.int32), ((51200, 1024), torch.bfloat16)]),
    (Embedding0, [((1, 128), torch.float32), ((119547, 768), torch.bfloat16)]),
    (Embedding0, [((1, 384), torch.float32), ((28996, 768), torch.bfloat16)]),
    (Embedding0, [((1, 384), torch.float32), ((512, 768), torch.bfloat16)]),
    (Embedding0, [((1, 6), torch.float32), ((65024, 4544), torch.bfloat16)]),
    (Embedding0, [((1, 7), torch.float32), ((256000, 2048), torch.bfloat16)]),
    (Embedding0, [((1, 256), torch.float32), ((50257, 768), torch.bfloat16)]),
    (Embedding2, [((1024, 768), torch.bfloat16)]),
    (Embedding0, [((1, 32), torch.float32), ((50257, 768), torch.bfloat16)]),
    (Embedding3, [((2048, 768), torch.bfloat16)]),
    (Embedding0, [((1, 32), torch.float32), ((50257, 2048), torch.bfloat16)]),
    (Embedding3, [((2048, 2048), torch.bfloat16)]),
    (Embedding2, [((2048, 768), torch.bfloat16)]),
    (Embedding0, [((1, 256), torch.float32), ((50257, 2048), torch.bfloat16)]),
    (Embedding2, [((2048, 2048), torch.bfloat16)]),
    (Embedding0, [((1, 256), torch.float32), ((50257, 2560), torch.bfloat16)]),
    (Embedding2, [((2048, 2560), torch.bfloat16)]),
    (Embedding0, [((1, 32), torch.float32), ((50257, 2560), torch.bfloat16)]),
    (Embedding3, [((2048, 2560), torch.bfloat16)]),
    (Embedding0, [((1, 256), torch.int32), ((128256, 4096), torch.bfloat16)]),
    (Embedding0, [((1, 4), torch.float32), ((128256, 2048), torch.bfloat16)]),
    (Embedding0, [((1, 4), torch.float32), ((128256, 4096), torch.bfloat16)]),
    (Embedding0, [((1, 256), torch.int32), ((128256, 2048), torch.bfloat16)]),
    (Embedding0, [((1, 128), torch.float32), ((32000, 4096), torch.bfloat16)]),
    (Embedding0, [((1, 32), torch.float32), ((50272, 2048), torch.bfloat16)]),
    (Embedding0, [((1, 32), torch.float32), ((2050, 2048), torch.bfloat16)]),
    (Embedding0, [((1, 256), torch.float32), ((50272, 2048), torch.bfloat16)]),
    (Embedding0, [((1, 256), torch.float32), ((2050, 2048), torch.bfloat16)]),
    (Embedding0, [((1, 32), torch.float32), ((50272, 512), torch.bfloat16)]),
    (Embedding0, [((1, 32), torch.float32), ((2050, 1024), torch.bfloat16)]),
    (Embedding0, [((1, 32), torch.float32), ((50272, 768), torch.bfloat16)]),
    (Embedding0, [((1, 32), torch.float32), ((2050, 768), torch.bfloat16)]),
    (Embedding0, [((1, 256), torch.float32), ((50272, 768), torch.bfloat16)]),
    (Embedding0, [((1, 256), torch.float32), ((2050, 768), torch.bfloat16)]),
    (Embedding0, [((1, 256), torch.float32), ((50272, 512), torch.bfloat16)]),
    (Embedding0, [((1, 256), torch.float32), ((2050, 1024), torch.bfloat16)]),
    (Embedding0, [((1, 256), torch.int32), ((51200, 2560), torch.bfloat16)]),
    (Embedding0, [((1, 12), torch.float32), ((51200, 2560), torch.bfloat16)]),
    (Embedding0, [((1, 11), torch.float32), ((51200, 2560), torch.bfloat16)]),
    (Embedding0, [((1, 6), torch.float32), ((151936, 1024), torch.bfloat16)]),
    (Embedding0, [((1, 29), torch.float32), ((151936, 1024), torch.bfloat16)]),
    (Embedding0, [((1, 35), torch.float32), ((151936, 1536), torch.bfloat16)]),
    (Embedding0, [((1, 35), torch.float32), ((152064, 3584), torch.bfloat16)]),
    (Embedding0, [((1, 35), torch.float32), ((151936, 2048), torch.bfloat16)]),
    (Embedding0, [((1, 35), torch.float32), ((151936, 896), torch.bfloat16)]),
    (Embedding0, [((1, 29), torch.float32), ((151936, 896), torch.bfloat16)]),
    (Embedding0, [((1, 39), torch.float32), ((151936, 1536), torch.bfloat16)]),
    (Embedding0, [((1, 29), torch.float32), ((151936, 1536), torch.bfloat16)]),
    (Embedding0, [((1, 29), torch.float32), ((151936, 2048), torch.bfloat16)]),
    (Embedding0, [((1, 39), torch.float32), ((151936, 2048), torch.bfloat16)]),
    (Embedding0, [((1, 39), torch.float32), ((151936, 896), torch.bfloat16)]),
    (Embedding0, [((1, 39), torch.float32), ((152064, 3584), torch.bfloat16)]),
    (Embedding0, [((1, 29), torch.float32), ((152064, 3584), torch.bfloat16)]),
    (Embedding0, [((1, 128), torch.float32), ((50265, 768), torch.bfloat16)]),
    (Embedding0, [((1, 128), torch.float32), ((1, 768), torch.bfloat16)]),
    (Embedding0, [((1, 128), torch.float32), ((514, 768), torch.bfloat16)]),
    (Embedding0, [((1, 128), torch.float32), ((250002, 768), torch.bfloat16)]),
    (Embedding0, [((1, 128), torch.float32), ((30528, 768), torch.bfloat16)]),
    (Embedding4, [((2, 768), torch.bfloat16)]),
    (Embedding0, [((1, 1), torch.int32), ((32128, 512), torch.bfloat16)]),
    (Embedding5, [((32, 6), torch.bfloat16)]),
    (Embedding0, [((1, 1), torch.int32), ((32128, 768), torch.bfloat16)]),
    (Embedding5, [((32, 12), torch.bfloat16)]),
    (Embedding0, [((1, 1), torch.int32), ((32128, 1024), torch.bfloat16)]),
    (Embedding5, [((32, 16), torch.bfloat16)]),
    (Embedding5, [((32, 8), torch.bfloat16)]),
    (Embedding0, [((1, 256), torch.float32), ((256008, 2048), torch.bfloat16)]),
    (Embedding0, [((1, 256), torch.float32), ((256008, 1024), torch.bfloat16)]),
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
