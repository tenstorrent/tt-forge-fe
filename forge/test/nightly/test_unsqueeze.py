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


class Unsqueeze0(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, unsqueeze_input_0):
        unsqueeze_output_1 = forge.op.Unsqueeze("", unsqueeze_input_0, dim=1)
        return unsqueeze_output_1


class Unsqueeze1(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, unsqueeze_input_0):
        unsqueeze_output_1 = forge.op.Unsqueeze("", unsqueeze_input_0, dim=2)
        return unsqueeze_output_1


def ids_func(param):
    forge_module, shapes_dtypes, _ = param
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (
        Unsqueeze0,
        [((1, 128), torch.float32)],
        {"model_name": ["albert_pt_albert_base_v1_masked_lm", "albert_pt_albert_base_v2_masked_lm"]},
    ),
    (
        Unsqueeze1,
        [((1, 1, 128), torch.float32)],
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
