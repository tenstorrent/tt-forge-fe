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


class Broadcast0(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, broadcast_input_0):
        broadcast_output_1 = forge.op.Broadcast("", broadcast_input_0, dim=-3, shape=12)
        return broadcast_output_1


class Broadcast1(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, broadcast_input_0):
        broadcast_output_1 = forge.op.Broadcast("", broadcast_input_0, dim=-4, shape=1)
        return broadcast_output_1


class Broadcast2(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, broadcast_input_0):
        broadcast_output_1 = forge.op.Broadcast("", broadcast_input_0, dim=-2, shape=128)
        return broadcast_output_1


class Broadcast3(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, broadcast_input_0):
        broadcast_output_1 = forge.op.Broadcast("", broadcast_input_0, dim=-2, shape=384)
        return broadcast_output_1


class Broadcast4(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, broadcast_input_0):
        broadcast_output_1 = forge.op.Broadcast("", broadcast_input_0, dim=-1, shape=32)
        return broadcast_output_1


def ids_func(param):
    forge_module, shapes_dtypes, _ = param
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (
        Broadcast0,
        [((1, 1, 1, 128), torch.float32)],
        {
            "model_name": [
                "pt_distilbert_masked_lm",
                "pt_distilbert_sequence_classification",
                "pt_distilbert_token_classification",
            ]
        },
    ),
    (
        Broadcast1,
        [((1, 1, 1, 128), torch.float32)],
        {
            "model_name": [
                "pt_distilbert_masked_lm",
                "pt_distilbert_sequence_classification",
                "pt_distilbert_token_classification",
            ]
        },
    ),
    (
        Broadcast2,
        [((1, 12, 1, 128), torch.float32)],
        {
            "model_name": [
                "pt_distilbert_masked_lm",
                "pt_distilbert_sequence_classification",
                "pt_distilbert_token_classification",
            ]
        },
    ),
    (Broadcast0, [((1, 1, 1, 384), torch.float32)], {"model_name": ["pt_distilbert_question_answering"]}),
    (Broadcast1, [((1, 1, 1, 384), torch.float32)], {"model_name": ["pt_distilbert_question_answering"]}),
    (Broadcast3, [((1, 12, 1, 384), torch.float32)], {"model_name": ["pt_distilbert_question_answering"]}),
    (Broadcast4, [((64, 3, 64, 1), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Broadcast4, [((16, 6, 64, 1), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Broadcast4, [((4, 12, 64, 1), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Broadcast4, [((1, 24, 64, 1), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
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
