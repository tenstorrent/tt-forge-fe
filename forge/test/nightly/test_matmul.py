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


class Matmul0(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, matmul_input_0, matmul_input_1):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, matmul_input_1)
        return matmul_output_1


def ids_func(param):
    forge_module, shapes_dtypes, _ = param
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (
        Matmul0,
        [((1, 128, 128), torch.float32), ((128, 768), torch.float32)],
        {"model_name": ["albert_pt_albert_base_v1_masked_lm", "albert_pt_albert_base_v2_masked_lm"]},
    ),
    (
        Matmul0,
        [((128, 768), torch.float32), ((768, 768), torch.float32)],
        {"model_name": ["albert_pt_albert_base_v1_masked_lm", "albert_pt_albert_base_v2_masked_lm"]},
    ),
    (
        Matmul0,
        [((12, 128, 64), torch.float32), ((12, 64, 128), torch.float32)],
        {"model_name": ["albert_pt_albert_base_v1_masked_lm", "albert_pt_albert_base_v2_masked_lm"]},
    ),
    (
        Matmul0,
        [((12, 128, 128), torch.float32), ((12, 128, 64), torch.float32)],
        {"model_name": ["albert_pt_albert_base_v1_masked_lm", "albert_pt_albert_base_v2_masked_lm"]},
    ),
    (
        Matmul0,
        [((1, 128, 768), torch.float32), ((768, 768), torch.float32)],
        {"model_name": ["albert_pt_albert_base_v1_masked_lm", "albert_pt_albert_base_v2_masked_lm"]},
    ),
    (
        Matmul0,
        [((1, 128, 768), torch.float32), ((768, 3072), torch.float32)],
        {"model_name": ["albert_pt_albert_base_v1_masked_lm", "albert_pt_albert_base_v2_masked_lm"]},
    ),
    (
        Matmul0,
        [((1, 128, 3072), torch.float32), ((3072, 768), torch.float32)],
        {"model_name": ["albert_pt_albert_base_v1_masked_lm", "albert_pt_albert_base_v2_masked_lm"]},
    ),
    (
        Matmul0,
        [((1, 128, 768), torch.float32), ((768, 128), torch.float32)],
        {"model_name": ["albert_pt_albert_base_v1_masked_lm", "albert_pt_albert_base_v2_masked_lm"]},
    ),
    (
        Matmul0,
        [((1, 128, 128), torch.float32), ((128, 30000), torch.float32)],
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
