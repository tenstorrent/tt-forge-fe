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


class Where0(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("where0_const_1", shape=(1,), dtype=torch.float32, use_random_value=True)
        self.add_constant("where0_const_2", shape=(1,), dtype=torch.float32, use_random_value=True)

    def forward(self, where_input_0):
        where_output_1 = forge.op.Where(
            "", where_input_0, self.get_constant("where0_const_1"), self.get_constant("where0_const_2")
        )
        return where_output_1


def ids_func(param):
    forge_module, shapes_dtypes, _ = param
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (
        Where0,
        [((1, 1, 256, 256), torch.float32)],
        {
            "model_name": [
                "pt_gpt2_generation",
                "pt_gpt_neo_2_7B_causal_lm",
                "pt_gpt_neo_1_3B_causal_lm",
                "pt_gpt_neo_125M_causal_lm",
            ]
        },
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
