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


class Sine0(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, sine_input_0):
        sine_output_1 = forge.op.Sine("", sine_input_0)
        return sine_output_1


def ids_func(param):
    forge_module, shapes_dtypes, _ = param
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (Sine0, [((1, 6, 64), torch.float32)], {"model_name": ["pt_falcon", "pt_qwen_causal_lm"]}),
    (Sine0, [((1, 334, 32), torch.float32)], {"model_name": ["pt_fuyu_8b"]}),
    (Sine0, [((1, 7, 256), torch.float32)], {"model_name": ["pt_gemma_2b"]}),
    (
        Sine0,
        [((1, 256, 128), torch.float32)],
        {
            "model_name": [
                "pt_Llama_3_1_8B_Instruct_causal_lm",
                "pt_Meta_Llama_3_8B_causal_lm",
                "pt_Meta_Llama_3_8B_Instruct_causal_lm",
                "pt_Llama_3_1_8B_causal_lm",
            ]
        },
    ),
    (
        Sine0,
        [((1, 256, 64), torch.float32)],
        {"model_name": ["pt_Llama_3_2_1B_Instruct_causal_lm", "pt_Llama_3_2_1B_causal_lm"]},
    ),
    (
        Sine0,
        [((1, 4, 128), torch.float32)],
        {
            "model_name": [
                "pt_Llama_3_1_8B_Instruct_seq_cls",
                "pt_Meta_Llama_3_8B_seq_cls",
                "pt_Llama_3_1_8B_seq_cls",
                "pt_Meta_Llama_3_8B_Instruct_seq_cls",
            ]
        },
    ),
    (
        Sine0,
        [((1, 4, 64), torch.float32)],
        {"model_name": ["pt_Llama_3_2_1B_Instruct_seq_cls", "pt_Llama_3_2_1B_seq_cls"]},
    ),
    (Sine0, [((1, 128, 128), torch.float32)], {"model_name": ["pt_Mistral_7B_v0_1"]}),
    (Sine0, [((1, 12, 32), torch.float32)], {"model_name": ["pt_phi_2_pytdml_token_cls", "pt_phi_2_token_cls"]}),
    (Sine0, [((1, 256, 32), torch.float32)], {"model_name": ["pt_phi_2_causal_lm", "pt_phi_2_pytdml_causal_lm"]}),
    (Sine0, [((1, 11, 32), torch.float32)], {"model_name": ["pt_phi_2_seq_cls", "pt_phi_2_pytdml_seq_cls"]}),
    (Sine0, [((1, 29, 64), torch.float32)], {"model_name": ["pt_qwen_chat", "pt_Qwen_Qwen2_5_0_5B"]}),
    (
        Sine0,
        [((1, 35, 128), torch.float32)],
        {
            "model_name": [
                "pt_Qwen_Qwen2_5_Coder_1_5B_Instruct",
                "pt_Qwen_Qwen2_5_Coder_3B",
                "pt_Qwen_Qwen2_5_Coder_7B",
                "pt_Qwen_Qwen2_5_Coder_1_5B",
                "pt_Qwen_Qwen2_5_Coder_3B_Instruct",
                "pt_Qwen_Qwen2_5_Coder_7B_Instruct",
            ]
        },
    ),
    (Sine0, [((1, 35, 64), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_Coder_0_5B"]}),
    (
        Sine0,
        [((1, 39, 128), torch.float32)],
        {"model_name": ["pt_Qwen_Qwen2_5_1_5B_Instruct", "pt_Qwen_Qwen2_5_3B_Instruct", "pt_Qwen_Qwen2_5_7B_Instruct"]},
    ),
    (
        Sine0,
        [((1, 29, 128), torch.float32)],
        {"model_name": ["pt_Qwen_Qwen2_5_1_5B", "pt_Qwen_Qwen2_5_7B", "pt_Qwen_Qwen2_5_3B"]},
    ),
    (Sine0, [((1, 39, 64), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_0_5B_Instruct"]}),
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
