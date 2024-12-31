# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
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


class Reshape0(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(128, 768))
        return reshape_output_1


class Reshape1(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 12, 64))
        return reshape_output_1


class Reshape2(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 768))
        return reshape_output_1


class Reshape3(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 128, 64))
        return reshape_output_1


class Reshape4(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 128, 128))
        return reshape_output_1


class Reshape5(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 128, 128))
        return reshape_output_1


class Reshape6(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 64, 128))
        return reshape_output_1


class Reshape7(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 128, 64))
        return reshape_output_1


class Reshape8(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 768, 1))
        return reshape_output_1


def ids_func(param):
    forge_module, shapes_dtypes, _ = param
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (
        Reshape0,
        [((1, 128, 768), torch.float32)],
        {"model_name": ["albert_pt_albert_base_v1_masked_lm", "albert_pt_albert_base_v2_masked_lm"]},
    ),
    (
        Reshape1,
        [((1, 128, 768), torch.float32)],
        {"model_name": ["albert_pt_albert_base_v1_masked_lm", "albert_pt_albert_base_v2_masked_lm"]},
    ),
    (
        Reshape2,
        [((128, 768), torch.float32)],
        {"model_name": ["albert_pt_albert_base_v1_masked_lm", "albert_pt_albert_base_v2_masked_lm"]},
    ),
    (
        Reshape3,
        [((1, 12, 128, 64), torch.float32)],
        {"model_name": ["albert_pt_albert_base_v1_masked_lm", "albert_pt_albert_base_v2_masked_lm"]},
    ),
    (
        Reshape4,
        [((12, 128, 128), torch.float32)],
        {"model_name": ["albert_pt_albert_base_v1_masked_lm", "albert_pt_albert_base_v2_masked_lm"]},
    ),
    (
        Reshape5,
        [((1, 12, 128, 128), torch.float32)],
        {"model_name": ["albert_pt_albert_base_v1_masked_lm", "albert_pt_albert_base_v2_masked_lm"]},
    ),
    (
        Reshape6,
        [((1, 12, 64, 128), torch.float32)],
        {"model_name": ["albert_pt_albert_base_v1_masked_lm", "albert_pt_albert_base_v2_masked_lm"]},
    ),
    (
        Reshape7,
        [((12, 128, 64), torch.float32)],
        {"model_name": ["albert_pt_albert_base_v1_masked_lm", "albert_pt_albert_base_v2_masked_lm"]},
    ),
    (
        Reshape8,
        [((1, 128, 12, 64), torch.float32)],
        {"model_name": ["albert_pt_albert_base_v1_masked_lm", "albert_pt_albert_base_v2_masked_lm"]},
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
