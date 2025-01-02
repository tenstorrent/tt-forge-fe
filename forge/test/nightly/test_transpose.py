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


class Transpose0(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, transpose_input_0):
        transpose_output_1 = forge.op.Transpose("", transpose_input_0, dim0=-2, dim1=-1)
        return transpose_output_1


class Transpose1(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, transpose_input_0):
        transpose_output_1 = forge.op.Transpose("", transpose_input_0, dim0=-3, dim1=-2)
        return transpose_output_1


def ids_func(param):
    forge_module, shapes_dtypes, _ = param
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (
        Transpose0,
        [((2048, 128), torch.float32)],
        {
            "model_name": [
                "pt_albert_xlarge_v1_token_cls",
                "pt_albert_xlarge_v1_masked_lm",
                "pt_albert_xlarge_v2_masked_lm",
                "pt_albert_xlarge_v2_token_cls",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Transpose0,
        [((2048, 2048), torch.float32)],
        {
            "model_name": [
                "pt_albert_xlarge_v1_token_cls",
                "pt_albert_xlarge_v1_masked_lm",
                "pt_albert_xlarge_v2_masked_lm",
                "pt_albert_xlarge_v2_token_cls",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Transpose1,
        [((1, 128, 16, 128), torch.float32)],
        {
            "model_name": [
                "pt_albert_xlarge_v1_token_cls",
                "pt_albert_xlarge_v1_masked_lm",
                "pt_albert_xlarge_v2_masked_lm",
                "pt_albert_xlarge_v2_token_cls",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Transpose0,
        [((16, 128, 128), torch.float32)],
        {
            "model_name": [
                "pt_albert_xlarge_v1_token_cls",
                "pt_albert_xlarge_v1_masked_lm",
                "pt_albert_xlarge_v2_masked_lm",
                "pt_albert_xlarge_v2_token_cls",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Transpose0,
        [((1, 16, 128, 128), torch.float32)],
        {
            "model_name": [
                "pt_albert_xlarge_v1_token_cls",
                "pt_albert_xlarge_v1_masked_lm",
                "pt_albert_xlarge_v2_masked_lm",
                "pt_albert_xlarge_v2_token_cls",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Transpose1,
        [((1, 16, 128, 128), torch.float32)],
        {
            "model_name": [
                "pt_albert_xlarge_v1_token_cls",
                "pt_albert_xlarge_v1_masked_lm",
                "pt_albert_xlarge_v2_masked_lm",
                "pt_albert_xlarge_v2_token_cls",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Transpose0,
        [((8192, 2048), torch.float32)],
        {
            "model_name": [
                "pt_albert_xlarge_v1_token_cls",
                "pt_albert_xlarge_v1_masked_lm",
                "pt_albert_xlarge_v2_masked_lm",
                "pt_albert_xlarge_v2_token_cls",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Transpose0,
        [((2048, 8192), torch.float32)],
        {
            "model_name": [
                "pt_albert_xlarge_v1_token_cls",
                "pt_albert_xlarge_v1_masked_lm",
                "pt_albert_xlarge_v2_masked_lm",
                "pt_albert_xlarge_v2_token_cls",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Transpose0,
        [((2, 2048), torch.float32)],
        {"model_name": ["pt_albert_xlarge_v1_token_cls", "pt_albert_xlarge_v2_token_cls"], "pcc": 0.99},
    ),
    (
        Transpose0,
        [((768, 128), torch.float32)],
        {
            "model_name": [
                "pt_albert_base_v1_masked_lm",
                "pt_albert_base_v1_token_cls",
                "pt_albert_base_v2_masked_lm",
                "pt_albert_base_v2_token_cls",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Transpose0,
        [((768, 768), torch.float32)],
        {
            "model_name": [
                "pt_albert_base_v1_masked_lm",
                "pt_albert_base_v1_token_cls",
                "pt_albert_base_v2_masked_lm",
                "pt_albert_base_v2_token_cls",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Transpose1,
        [((1, 128, 12, 64), torch.float32)],
        {
            "model_name": [
                "pt_albert_base_v1_masked_lm",
                "pt_albert_base_v1_token_cls",
                "pt_albert_base_v2_masked_lm",
                "pt_albert_base_v2_token_cls",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Transpose0,
        [((12, 128, 64), torch.float32)],
        {
            "model_name": [
                "pt_albert_base_v1_masked_lm",
                "pt_albert_base_v1_token_cls",
                "pt_albert_base_v2_masked_lm",
                "pt_albert_base_v2_token_cls",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Transpose0,
        [((1, 12, 128, 64), torch.float32)],
        {
            "model_name": [
                "pt_albert_base_v1_masked_lm",
                "pt_albert_base_v1_token_cls",
                "pt_albert_base_v2_masked_lm",
                "pt_albert_base_v2_token_cls",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Transpose1,
        [((1, 12, 128, 64), torch.float32)],
        {
            "model_name": [
                "pt_albert_base_v1_masked_lm",
                "pt_albert_base_v1_token_cls",
                "pt_albert_base_v2_masked_lm",
                "pt_albert_base_v2_token_cls",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Transpose0,
        [((12, 64, 128), torch.float32)],
        {
            "model_name": [
                "pt_albert_base_v1_masked_lm",
                "pt_albert_base_v1_token_cls",
                "pt_albert_base_v2_masked_lm",
                "pt_albert_base_v2_token_cls",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Transpose0,
        [((3072, 768), torch.float32)],
        {
            "model_name": [
                "pt_albert_base_v1_masked_lm",
                "pt_albert_base_v1_token_cls",
                "pt_albert_base_v2_masked_lm",
                "pt_albert_base_v2_token_cls",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Transpose0,
        [((768, 3072), torch.float32)],
        {
            "model_name": [
                "pt_albert_base_v1_masked_lm",
                "pt_albert_base_v1_token_cls",
                "pt_albert_base_v2_masked_lm",
                "pt_albert_base_v2_token_cls",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Transpose0,
        [((128, 768), torch.float32)],
        {"model_name": ["pt_albert_base_v1_masked_lm", "pt_albert_base_v2_masked_lm"], "pcc": 0.99},
    ),
    (
        Transpose0,
        [((30000, 128), torch.float32)],
        {
            "model_name": [
                "pt_albert_base_v1_masked_lm",
                "pt_albert_xxlarge_v2_masked_lm",
                "pt_albert_xlarge_v1_masked_lm",
                "pt_albert_large_v2_masked_lm",
                "pt_albert_xlarge_v2_masked_lm",
                "pt_albert_large_v1_masked_lm",
                "pt_albert_base_v2_masked_lm",
                "pt_albert_xxlarge_v1_masked_lm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Transpose0,
        [((4096, 128), torch.float32)],
        {
            "model_name": [
                "pt_albert_xxlarge_v2_masked_lm",
                "pt_albert_xxlarge_v1_token_cls",
                "pt_albert_xxlarge_v2_token_cls",
                "pt_albert_xxlarge_v1_masked_lm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Transpose0,
        [((4096, 4096), torch.float32)],
        {
            "model_name": [
                "pt_albert_xxlarge_v2_masked_lm",
                "pt_albert_xxlarge_v1_token_cls",
                "pt_albert_xxlarge_v2_token_cls",
                "pt_albert_xxlarge_v1_masked_lm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Transpose1,
        [((1, 128, 64, 64), torch.float32)],
        {
            "model_name": [
                "pt_albert_xxlarge_v2_masked_lm",
                "pt_albert_xxlarge_v1_token_cls",
                "pt_albert_xxlarge_v2_token_cls",
                "pt_albert_xxlarge_v1_masked_lm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Transpose0,
        [((64, 128, 64), torch.float32)],
        {
            "model_name": [
                "pt_albert_xxlarge_v2_masked_lm",
                "pt_albert_xxlarge_v1_token_cls",
                "pt_albert_xxlarge_v2_token_cls",
                "pt_albert_xxlarge_v1_masked_lm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Transpose0,
        [((1, 64, 128, 64), torch.float32)],
        {
            "model_name": [
                "pt_albert_xxlarge_v2_masked_lm",
                "pt_albert_xxlarge_v1_token_cls",
                "pt_albert_xxlarge_v2_token_cls",
                "pt_albert_xxlarge_v1_masked_lm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Transpose1,
        [((1, 64, 128, 64), torch.float32)],
        {
            "model_name": [
                "pt_albert_xxlarge_v2_masked_lm",
                "pt_albert_xxlarge_v1_token_cls",
                "pt_albert_xxlarge_v2_token_cls",
                "pt_albert_xxlarge_v1_masked_lm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Transpose0,
        [((64, 64, 128), torch.float32)],
        {
            "model_name": [
                "pt_albert_xxlarge_v2_masked_lm",
                "pt_albert_xxlarge_v1_token_cls",
                "pt_albert_xxlarge_v2_token_cls",
                "pt_albert_xxlarge_v1_masked_lm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Transpose0,
        [((16384, 4096), torch.float32)],
        {
            "model_name": [
                "pt_albert_xxlarge_v2_masked_lm",
                "pt_albert_xxlarge_v1_token_cls",
                "pt_albert_xxlarge_v2_token_cls",
                "pt_albert_xxlarge_v1_masked_lm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Transpose0,
        [((4096, 16384), torch.float32)],
        {
            "model_name": [
                "pt_albert_xxlarge_v2_masked_lm",
                "pt_albert_xxlarge_v1_token_cls",
                "pt_albert_xxlarge_v2_token_cls",
                "pt_albert_xxlarge_v1_masked_lm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Transpose0,
        [((128, 4096), torch.float32)],
        {"model_name": ["pt_albert_xxlarge_v2_masked_lm", "pt_albert_xxlarge_v1_masked_lm"], "pcc": 0.99},
    ),
    (
        Transpose0,
        [((128, 2048), torch.float32)],
        {"model_name": ["pt_albert_xlarge_v1_masked_lm", "pt_albert_xlarge_v2_masked_lm"], "pcc": 0.99},
    ),
    (
        Transpose0,
        [((1024, 128), torch.float32)],
        {
            "model_name": [
                "pt_albert_large_v2_masked_lm",
                "pt_albert_large_v1_token_cls",
                "pt_albert_large_v1_masked_lm",
                "pt_albert_large_v2_token_cls",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Transpose0,
        [((1024, 1024), torch.float32)],
        {
            "model_name": [
                "pt_albert_large_v2_masked_lm",
                "pt_albert_large_v1_token_cls",
                "pt_albert_large_v1_masked_lm",
                "pt_albert_large_v2_token_cls",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Transpose1,
        [((1, 128, 16, 64), torch.float32)],
        {
            "model_name": [
                "pt_albert_large_v2_masked_lm",
                "pt_albert_large_v1_token_cls",
                "pt_albert_large_v1_masked_lm",
                "pt_albert_large_v2_token_cls",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Transpose0,
        [((16, 128, 64), torch.float32)],
        {
            "model_name": [
                "pt_albert_large_v2_masked_lm",
                "pt_albert_large_v1_token_cls",
                "pt_albert_large_v1_masked_lm",
                "pt_albert_large_v2_token_cls",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Transpose0,
        [((1, 16, 128, 64), torch.float32)],
        {
            "model_name": [
                "pt_albert_large_v2_masked_lm",
                "pt_albert_large_v1_token_cls",
                "pt_albert_large_v1_masked_lm",
                "pt_albert_large_v2_token_cls",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Transpose1,
        [((1, 16, 128, 64), torch.float32)],
        {
            "model_name": [
                "pt_albert_large_v2_masked_lm",
                "pt_albert_large_v1_token_cls",
                "pt_albert_large_v1_masked_lm",
                "pt_albert_large_v2_token_cls",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Transpose0,
        [((16, 64, 128), torch.float32)],
        {
            "model_name": [
                "pt_albert_large_v2_masked_lm",
                "pt_albert_large_v1_token_cls",
                "pt_albert_large_v1_masked_lm",
                "pt_albert_large_v2_token_cls",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Transpose0,
        [((4096, 1024), torch.float32)],
        {
            "model_name": [
                "pt_albert_large_v2_masked_lm",
                "pt_albert_large_v1_token_cls",
                "pt_albert_large_v1_masked_lm",
                "pt_albert_large_v2_token_cls",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Transpose0,
        [((1024, 4096), torch.float32)],
        {
            "model_name": [
                "pt_albert_large_v2_masked_lm",
                "pt_albert_large_v1_token_cls",
                "pt_albert_large_v1_masked_lm",
                "pt_albert_large_v2_token_cls",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Transpose0,
        [((128, 1024), torch.float32)],
        {"model_name": ["pt_albert_large_v2_masked_lm", "pt_albert_large_v1_masked_lm"], "pcc": 0.99},
    ),
    (
        Transpose0,
        [((2, 4096), torch.float32)],
        {"model_name": ["pt_albert_xxlarge_v1_token_cls", "pt_albert_xxlarge_v2_token_cls"], "pcc": 0.99},
    ),
    (
        Transpose0,
        [((2, 1024), torch.float32)],
        {"model_name": ["pt_albert_large_v1_token_cls", "pt_albert_large_v2_token_cls"], "pcc": 0.99},
    ),
    (
        Transpose0,
        [((2, 768), torch.float32)],
        {"model_name": ["pt_albert_base_v1_token_cls", "pt_albert_base_v2_token_cls"], "pcc": 0.99},
    ),
]


@pytest.mark.push
@pytest.mark.parametrize("forge_module_and_shapes_dtypes", forge_modules_and_shapes_dtypes_list, ids=ids_func)
def test_module(forge_module_and_shapes_dtypes, record_property):
    record_property("frontend", "tt-forge-fe")

    forge_module, operand_shapes_dtypes, metadata = forge_module_and_shapes_dtypes
    pcc = metadata.pop("pcc")

    for metadata_name, metadata_value in metadata.items():
        record_property(metadata_name, metadata_value)

    inputs = [
        Tensor.create_from_shape(operand_shape, operand_dtype) for operand_shape, operand_dtype in operand_shapes_dtypes
    ]

    framework_model = forge_module(forge_module.__name__)
    framework_model.process_framework_parameters()

    compiled_model = compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model, VerifyConfig(value_checker=AutomaticValueChecker(pcc=pcc)))
