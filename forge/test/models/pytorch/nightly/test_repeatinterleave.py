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


class Repeatinterleave0(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, repeatinterleave_input_0):
        repeatinterleave_output_1 = forge.op.RepeatInterleave("", repeatinterleave_input_0, repeats=1, dim=1)
        return repeatinterleave_output_1


class Repeatinterleave1(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, repeatinterleave_input_0):
        repeatinterleave_output_1 = forge.op.RepeatInterleave("", repeatinterleave_input_0, repeats=1, dim=2)
        return repeatinterleave_output_1


class Repeatinterleave2(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, repeatinterleave_input_0):
        repeatinterleave_output_1 = forge.op.RepeatInterleave("", repeatinterleave_input_0, repeats=7, dim=2)
        return repeatinterleave_output_1


class Repeatinterleave3(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, repeatinterleave_input_0):
        repeatinterleave_output_1 = forge.op.RepeatInterleave("", repeatinterleave_input_0, repeats=1, dim=0)
        return repeatinterleave_output_1


class Repeatinterleave4(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, repeatinterleave_input_0):
        repeatinterleave_output_1 = forge.op.RepeatInterleave("", repeatinterleave_input_0, repeats=256, dim=2)
        return repeatinterleave_output_1


class Repeatinterleave5(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, repeatinterleave_input_0):
        repeatinterleave_output_1 = forge.op.RepeatInterleave("", repeatinterleave_input_0, repeats=8, dim=2)
        return repeatinterleave_output_1


class Repeatinterleave6(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, repeatinterleave_input_0):
        repeatinterleave_output_1 = forge.op.RepeatInterleave("", repeatinterleave_input_0, repeats=4, dim=2)
        return repeatinterleave_output_1


class Repeatinterleave7(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, repeatinterleave_input_0):
        repeatinterleave_output_1 = forge.op.RepeatInterleave("", repeatinterleave_input_0, repeats=32, dim=2)
        return repeatinterleave_output_1


class Repeatinterleave8(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, repeatinterleave_input_0):
        repeatinterleave_output_1 = forge.op.RepeatInterleave("", repeatinterleave_input_0, repeats=6, dim=2)
        return repeatinterleave_output_1


def ids_func(param):
    forge_module, shapes_dtypes, _ = param
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (
        Repeatinterleave0,
        [((2, 1, 1, 13), torch.float32)],
        {"model_name": ["pt_musicgen_large", "pt_musicgen_small", "pt_musicgen_medium"]},
    ),
    (
        Repeatinterleave1,
        [((2, 1, 1, 13), torch.float32)],
        {"model_name": ["pt_musicgen_large", "pt_musicgen_small", "pt_musicgen_medium"]},
    ),
    (Repeatinterleave0, [((2, 1, 1, 7), torch.float32)], {"model_name": ["pt_clip_vit_base_patch32_text"]}),
    (Repeatinterleave2, [((2, 1, 1, 7), torch.float32)], {"model_name": ["pt_clip_vit_base_patch32_text"]}),
    (
        Repeatinterleave3,
        [((1, 128), torch.float32)],
        {
            "model_name": [
                "pt_albert_base_v2_masked_lm",
                "pt_albert_xlarge_v1_token_cls",
                "pt_albert_large_v1_masked_lm",
                "pt_albert_xxlarge_v1_token_cls",
                "pt_albert_large_v2_token_cls",
                "pt_albert_large_v2_masked_lm",
                "pt_albert_xlarge_v2_token_cls",
                "pt_albert_base_v1_masked_lm",
                "pt_albert_xxlarge_v2_masked_lm",
                "pt_albert_xlarge_v1_masked_lm",
                "pt_albert_xxlarge_v1_masked_lm",
                "pt_albert_large_v1_token_cls",
                "pt_albert_base_v1_token_cls",
                "pt_albert_xlarge_v2_masked_lm",
                "pt_albert_xxlarge_v2_token_cls",
                "pt_albert_base_v2_token_cls",
                "pt_bert_masked_lm",
                "pt_bert_sequence_classification",
                "pt_dpr_reader_single_nq_base",
                "pt_dpr_reader_multiset_base",
                "pt_roberta_masked_lm",
                "pt_roberta_sentiment",
            ]
        },
    ),
    (
        Repeatinterleave3,
        [((1, 1, 1, 256), torch.float32)],
        {
            "model_name": [
                "pt_bart",
                "pt_opt_350m_causal_lm",
                "pt_opt_125m_causal_lm",
                "pt_opt_1_3b_causal_lm",
                "pt_xglm_1_7B",
                "pt_xglm_564M",
            ]
        },
    ),
    (
        Repeatinterleave0,
        [((1, 1, 1, 256), torch.float32)],
        {
            "model_name": [
                "pt_bart",
                "pt_opt_350m_causal_lm",
                "pt_opt_125m_causal_lm",
                "pt_opt_1_3b_causal_lm",
                "pt_xglm_1_7B",
                "pt_xglm_564M",
            ]
        },
    ),
    (
        Repeatinterleave4,
        [((1, 1, 1, 256), torch.float32)],
        {
            "model_name": [
                "pt_bart",
                "pt_opt_350m_causal_lm",
                "pt_opt_125m_causal_lm",
                "pt_opt_1_3b_causal_lm",
                "pt_xglm_1_7B",
                "pt_xglm_564M",
            ]
        },
    ),
    (Repeatinterleave3, [((1, 384), torch.float32)], {"model_name": ["pt_bert_qa"]}),
    (
        Repeatinterleave3,
        [((1, 32, 1), torch.float32)],
        {
            "model_name": [
                "pt_falcon",
                "pt_Llama_3_2_1B_Instruct_causal_lm",
                "pt_Llama_3_2_1B_causal_lm",
                "pt_Llama_3_2_1B_Instruct_seq_cls",
                "pt_Llama_3_2_1B_seq_cls",
                "pt_qwen_chat",
                "pt_qwen_causal_lm",
                "pt_Qwen_Qwen2_5_Coder_0_5B",
                "pt_Qwen_Qwen2_5_0_5B_Instruct",
                "pt_Qwen_Qwen2_5_0_5B",
            ]
        },
    ),
    (
        Repeatinterleave1,
        [((1, 32, 1), torch.float32)],
        {
            "model_name": [
                "pt_falcon",
                "pt_Llama_3_2_1B_Instruct_causal_lm",
                "pt_Llama_3_2_1B_causal_lm",
                "pt_Llama_3_2_1B_Instruct_seq_cls",
                "pt_Llama_3_2_1B_seq_cls",
                "pt_qwen_chat",
                "pt_qwen_causal_lm",
                "pt_Qwen_Qwen2_5_Coder_0_5B",
                "pt_Qwen_Qwen2_5_0_5B_Instruct",
                "pt_Qwen_Qwen2_5_0_5B",
            ]
        },
    ),
    (
        Repeatinterleave3,
        [((1, 16, 1), torch.float32)],
        {
            "model_name": [
                "pt_fuyu_8b",
                "pt_phi_2_pytdml_token_cls",
                "pt_phi_2_causal_lm",
                "pt_phi_2_seq_cls",
                "pt_phi_2_token_cls",
                "pt_phi_2_pytdml_seq_cls",
                "pt_phi_2_pytdml_causal_lm",
            ]
        },
    ),
    (
        Repeatinterleave1,
        [((1, 16, 1), torch.float32)],
        {
            "model_name": [
                "pt_fuyu_8b",
                "pt_phi_2_pytdml_token_cls",
                "pt_phi_2_causal_lm",
                "pt_phi_2_seq_cls",
                "pt_phi_2_token_cls",
                "pt_phi_2_pytdml_seq_cls",
                "pt_phi_2_pytdml_causal_lm",
            ]
        },
    ),
    (Repeatinterleave3, [((1, 128, 1), torch.float32)], {"model_name": ["pt_gemma_2b"]}),
    (Repeatinterleave1, [((1, 128, 1), torch.float32)], {"model_name": ["pt_gemma_2b"]}),
    (Repeatinterleave3, [((1, 1, 1, 7, 256), torch.float32)], {"model_name": ["pt_gemma_2b"]}),
    (Repeatinterleave0, [((1, 1, 1, 7, 256), torch.float32)], {"model_name": ["pt_gemma_2b"]}),
    (Repeatinterleave5, [((1, 1, 1, 7, 256), torch.float32)], {"model_name": ["pt_gemma_2b"]}),
    (
        Repeatinterleave3,
        [((1, 64, 1), torch.float32)],
        {
            "model_name": [
                "pt_Llama_3_1_8B_Instruct_causal_lm",
                "pt_Meta_Llama_3_8B_causal_lm",
                "pt_Meta_Llama_3_8B_Instruct_causal_lm",
                "pt_Llama_3_1_8B_Instruct_seq_cls",
                "pt_Meta_Llama_3_8B_seq_cls",
                "pt_Llama_3_1_8B_seq_cls",
                "pt_Llama_3_1_8B_causal_lm",
                "pt_Meta_Llama_3_8B_Instruct_seq_cls",
                "pt_Mistral_7B_v0_1",
                "pt_Qwen_Qwen2_5_Coder_1_5B_Instruct",
                "pt_Qwen_Qwen2_5_Coder_3B",
                "pt_Qwen_Qwen2_5_Coder_7B",
                "pt_Qwen_Qwen2_5_Coder_1_5B",
                "pt_Qwen_Qwen2_5_Coder_3B_Instruct",
                "pt_Qwen_Qwen2_5_Coder_7B_Instruct",
                "pt_Qwen_Qwen2_5_1_5B_Instruct",
                "pt_Qwen_Qwen2_5_1_5B",
                "pt_Qwen_Qwen2_5_7B",
                "pt_Qwen_Qwen2_5_3B_Instruct",
                "pt_Qwen_Qwen2_5_7B_Instruct",
                "pt_Qwen_Qwen2_5_3B",
            ]
        },
    ),
    (
        Repeatinterleave1,
        [((1, 64, 1), torch.float32)],
        {
            "model_name": [
                "pt_Llama_3_1_8B_Instruct_causal_lm",
                "pt_Meta_Llama_3_8B_causal_lm",
                "pt_Meta_Llama_3_8B_Instruct_causal_lm",
                "pt_Llama_3_1_8B_Instruct_seq_cls",
                "pt_Meta_Llama_3_8B_seq_cls",
                "pt_Llama_3_1_8B_seq_cls",
                "pt_Llama_3_1_8B_causal_lm",
                "pt_Meta_Llama_3_8B_Instruct_seq_cls",
                "pt_Mistral_7B_v0_1",
                "pt_Qwen_Qwen2_5_Coder_1_5B_Instruct",
                "pt_Qwen_Qwen2_5_Coder_3B",
                "pt_Qwen_Qwen2_5_Coder_7B",
                "pt_Qwen_Qwen2_5_Coder_1_5B",
                "pt_Qwen_Qwen2_5_Coder_3B_Instruct",
                "pt_Qwen_Qwen2_5_Coder_7B_Instruct",
                "pt_Qwen_Qwen2_5_1_5B_Instruct",
                "pt_Qwen_Qwen2_5_1_5B",
                "pt_Qwen_Qwen2_5_7B",
                "pt_Qwen_Qwen2_5_3B_Instruct",
                "pt_Qwen_Qwen2_5_7B_Instruct",
                "pt_Qwen_Qwen2_5_3B",
            ]
        },
    ),
    (
        Repeatinterleave3,
        [((1, 8, 1, 256, 128), torch.float32)],
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
        Repeatinterleave6,
        [((1, 8, 1, 256, 128), torch.float32)],
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
        Repeatinterleave3,
        [((1, 8, 1, 256, 64), torch.float32)],
        {"model_name": ["pt_Llama_3_2_1B_Instruct_causal_lm", "pt_Llama_3_2_1B_causal_lm"]},
    ),
    (
        Repeatinterleave6,
        [((1, 8, 1, 256, 64), torch.float32)],
        {"model_name": ["pt_Llama_3_2_1B_Instruct_causal_lm", "pt_Llama_3_2_1B_causal_lm"]},
    ),
    (
        Repeatinterleave3,
        [((1, 8, 1, 4, 128), torch.float32)],
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
        Repeatinterleave6,
        [((1, 8, 1, 4, 128), torch.float32)],
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
        Repeatinterleave3,
        [((1, 8, 1, 4, 64), torch.float32)],
        {"model_name": ["pt_Llama_3_2_1B_Instruct_seq_cls", "pt_Llama_3_2_1B_seq_cls"]},
    ),
    (
        Repeatinterleave6,
        [((1, 8, 1, 4, 64), torch.float32)],
        {"model_name": ["pt_Llama_3_2_1B_Instruct_seq_cls", "pt_Llama_3_2_1B_seq_cls"]},
    ),
    (Repeatinterleave3, [((1, 8, 1, 128, 128), torch.float32)], {"model_name": ["pt_Mistral_7B_v0_1"]}),
    (Repeatinterleave6, [((1, 8, 1, 128, 128), torch.float32)], {"model_name": ["pt_Mistral_7B_v0_1"]}),
    (
        Repeatinterleave3,
        [((1, 1, 1, 32), torch.float32)],
        {
            "model_name": [
                "pt_opt_125m_seq_cls",
                "pt_opt_1_3b_seq_cls",
                "pt_opt_1_3b_qa",
                "pt_opt_350m_qa",
                "pt_opt_125m_qa",
                "pt_opt_350m_seq_cls",
            ]
        },
    ),
    (
        Repeatinterleave0,
        [((1, 1, 1, 32), torch.float32)],
        {
            "model_name": [
                "pt_opt_125m_seq_cls",
                "pt_opt_1_3b_seq_cls",
                "pt_opt_1_3b_qa",
                "pt_opt_350m_qa",
                "pt_opt_125m_qa",
                "pt_opt_350m_seq_cls",
            ]
        },
    ),
    (
        Repeatinterleave7,
        [((1, 1, 1, 32), torch.float32)],
        {
            "model_name": [
                "pt_opt_125m_seq_cls",
                "pt_opt_1_3b_seq_cls",
                "pt_opt_1_3b_qa",
                "pt_opt_350m_qa",
                "pt_opt_125m_qa",
                "pt_opt_350m_seq_cls",
            ]
        },
    ),
    (
        Repeatinterleave3,
        [((1, 2, 1, 35, 128), torch.float32)],
        {
            "model_name": [
                "pt_Qwen_Qwen2_5_Coder_1_5B_Instruct",
                "pt_Qwen_Qwen2_5_Coder_3B",
                "pt_Qwen_Qwen2_5_Coder_1_5B",
                "pt_Qwen_Qwen2_5_Coder_3B_Instruct",
            ]
        },
    ),
    (
        Repeatinterleave8,
        [((1, 2, 1, 35, 128), torch.float32)],
        {"model_name": ["pt_Qwen_Qwen2_5_Coder_1_5B_Instruct", "pt_Qwen_Qwen2_5_Coder_1_5B"]},
    ),
    (
        Repeatinterleave5,
        [((1, 2, 1, 35, 128), torch.float32)],
        {"model_name": ["pt_Qwen_Qwen2_5_Coder_3B", "pt_Qwen_Qwen2_5_Coder_3B_Instruct"]},
    ),
    (
        Repeatinterleave3,
        [((1, 4, 1, 35, 128), torch.float32)],
        {"model_name": ["pt_Qwen_Qwen2_5_Coder_7B", "pt_Qwen_Qwen2_5_Coder_7B_Instruct"]},
    ),
    (
        Repeatinterleave2,
        [((1, 4, 1, 35, 128), torch.float32)],
        {"model_name": ["pt_Qwen_Qwen2_5_Coder_7B", "pt_Qwen_Qwen2_5_Coder_7B_Instruct"]},
    ),
    (Repeatinterleave3, [((1, 2, 1, 35, 64), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_Coder_0_5B"]}),
    (Repeatinterleave2, [((1, 2, 1, 35, 64), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_Coder_0_5B"]}),
    (
        Repeatinterleave3,
        [((1, 2, 1, 39, 128), torch.float32)],
        {"model_name": ["pt_Qwen_Qwen2_5_1_5B_Instruct", "pt_Qwen_Qwen2_5_3B_Instruct"]},
    ),
    (Repeatinterleave8, [((1, 2, 1, 39, 128), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_1_5B_Instruct"]}),
    (Repeatinterleave5, [((1, 2, 1, 39, 128), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_3B_Instruct"]}),
    (
        Repeatinterleave3,
        [((1, 2, 1, 29, 128), torch.float32)],
        {"model_name": ["pt_Qwen_Qwen2_5_1_5B", "pt_Qwen_Qwen2_5_3B"]},
    ),
    (Repeatinterleave8, [((1, 2, 1, 29, 128), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_1_5B"]}),
    (Repeatinterleave5, [((1, 2, 1, 29, 128), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_3B"]}),
    (Repeatinterleave3, [((1, 4, 1, 29, 128), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_7B"]}),
    (Repeatinterleave2, [((1, 4, 1, 29, 128), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_7B"]}),
    (Repeatinterleave3, [((1, 2, 1, 39, 64), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_0_5B_Instruct"]}),
    (Repeatinterleave2, [((1, 2, 1, 39, 64), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_0_5B_Instruct"]}),
    (Repeatinterleave3, [((1, 2, 1, 29, 64), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_0_5B"]}),
    (Repeatinterleave2, [((1, 2, 1, 29, 64), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_0_5B"]}),
    (Repeatinterleave3, [((1, 4, 1, 39, 128), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_7B_Instruct"]}),
    (Repeatinterleave2, [((1, 4, 1, 39, 128), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_7B_Instruct"]}),
    (Repeatinterleave3, [((1, 1, 384), torch.float32)], {"model_name": ["pt_deit_small_patch16_224"]}),
    (
        Repeatinterleave3,
        [((1, 1, 768), torch.float32)],
        {"model_name": ["pt_deit_base_patch16_224", "pt_deit_base_distilled_patch16_224", "pt_vit_base_patch16_224"]},
    ),
    (Repeatinterleave3, [((1, 1, 192), torch.float32)], {"model_name": ["pt_deit_tiny_patch16_224"]}),
    (
        Repeatinterleave3,
        [((1, 1, 1024), torch.float32)],
        {"model_name": ["pt_vision_perceiver_learned", "pt_vision_perceiver_conv", "pt_vision_perceiver_fourier"]},
    ),
    (
        Repeatinterleave3,
        [((1, 512, 1024), torch.float32)],
        {"model_name": ["pt_vision_perceiver_learned", "pt_vision_perceiver_conv", "pt_vision_perceiver_fourier"]},
    ),
    (Repeatinterleave3, [((1, 50176, 256), torch.float32)], {"model_name": ["pt_vision_perceiver_learned"]}),
    (Repeatinterleave3, [((1, 1, 1024), torch.float32)], {"model_name": ["pt_vit_large_patch16_224"]}),
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
