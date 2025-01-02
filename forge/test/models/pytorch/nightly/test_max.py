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


class Max0(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("max0_const_1", shape=(1,), dtype=torch.float32, use_random_value=True)

    def forward(self, max_input_0):
        max_output_1 = forge.op.Max("", max_input_0, self.get_constant("max0_const_1"))
        return max_output_1


def ids_func(param):
    forge_module, shapes_dtypes, _ = param
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (Max0, [((1, 12, 32, 32), torch.float32)], {"model_name": ["pt_opt_125m_seq_cls", "pt_opt_125m_qa"]}),
    (Max0, [((1, 32, 32, 32), torch.float32)], {"model_name": ["pt_opt_1_3b_seq_cls", "pt_opt_1_3b_qa"]}),
    (Max0, [((1, 16, 32, 32), torch.float32)], {"model_name": ["pt_opt_350m_qa", "pt_opt_350m_seq_cls"]}),
    (
        Max0,
        [((1, 16, 256, 256), torch.float32)],
        {"model_name": ["pt_opt_350m_causal_lm", "pt_xglm_1_7B", "pt_xglm_564M"]},
    ),
    (Max0, [((1, 12, 256, 256), torch.float32)], {"model_name": ["pt_opt_125m_causal_lm"]}),
    (Max0, [((1, 32, 256, 256), torch.float32)], {"model_name": ["pt_opt_1_3b_causal_lm"]}),
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
