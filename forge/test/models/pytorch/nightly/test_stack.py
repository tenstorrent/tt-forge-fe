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
from forge.verify.config import VerifyConfig
import pytest


class Stack0(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, stack_input_0, stack_input_1, stack_input_2, stack_input_3):
        stack_output_1 = forge.op.Stack("", stack_input_0, stack_input_1, stack_input_2, stack_input_3, axis=-3)
        return stack_output_1


class Stack1(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, stack_input_0, stack_input_1):
        stack_output_1 = forge.op.Stack("", stack_input_0, stack_input_1, axis=-1)
        return stack_output_1


def ids_func(param):
    forge_module, shapes_dtypes, _ = param
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (
        Stack0,
        [
            ((2, 1, 2048), torch.float32),
            ((2, 1, 2048), torch.float32),
            ((2, 1, 2048), torch.float32),
            ((2, 1, 2048), torch.float32),
        ],
        {"model_name": ["pt_musicgen_large", "pt_musicgen_small", "pt_musicgen_medium"]},
    ),
    (
        Stack1,
        [((1, 256, 16, 16), torch.float32), ((1, 256, 16, 16), torch.float32)],
        {"model_name": ["pt_codegen_350M_mono"]},
    ),
]


@pytest.mark.push
@pytest.mark.parametrize("forge_module_and_shapes_dtypes", forge_modules_and_shapes_dtypes_list, ids=ids_func)
def test_module(forge_module_and_shapes_dtypes, record_property):
    record_property("frontend", "tt-forge-fe")

    forge_module, operand_shapes_dtypes, metadata = forge_module_and_shapes_dtypes

    for metadata_name, metadata_value in metadata.items():
        record_property(metadata_name, metadata_value)

    inputs = [
        Tensor.create_from_shape(operand_shape, operand_dtype) for operand_shape, operand_dtype in operand_shapes_dtypes
    ]

    framework_model = forge_module(forge_module.__name__)
    framework_model.process_framework_parameters()

    compiled_model = compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)
