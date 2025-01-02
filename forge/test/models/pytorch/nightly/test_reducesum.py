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


class Reducesum0(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reducesum_input_0):
        reducesum_output_1 = forge.op.ReduceSum("", reducesum_input_0, dim=-1, keep_dim=True)
        return reducesum_output_1


def ids_func(param):
    forge_module, shapes_dtypes, _ = param
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (Reducesum0, [((64, 3, 64, 32), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Reducesum0, [((16, 6, 64, 32), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Reducesum0, [((4, 12, 64, 32), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Reducesum0, [((1, 24, 64, 32), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
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
