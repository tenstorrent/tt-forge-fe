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


class Embedding0(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, embedding_input_0, embedding_input_1):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, embedding_input_1)
        return embedding_output_1


class Embedding1(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("embedding1_const_0", shape=(13, 13), dtype=torch.int32, use_random_value=True)

    def forward(self, embedding_input_1):
        embedding_output_1 = forge.op.Embedding("", self.get_constant("embedding1_const_0"), embedding_input_1)
        return embedding_output_1


class Embedding2(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("embedding2_const_0", shape=(1, 256), dtype=torch.int64, use_random_value=True)

    def forward(self, embedding_input_1):
        embedding_output_1 = forge.op.Embedding("", self.get_constant("embedding2_const_0"), embedding_input_1)
        return embedding_output_1


class Embedding3(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("embedding3_const_0", shape=(1, 128), dtype=torch.int64, use_random_value=True)

    def forward(self, embedding_input_1):
        embedding_output_1 = forge.op.Embedding("", self.get_constant("embedding3_const_0"), embedding_input_1)
        return embedding_output_1


class Embedding4(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("embedding4_const_0", shape=(1, 1), dtype=torch.int32, use_random_value=True)

    def forward(self, embedding_input_1):
        embedding_output_1 = forge.op.Embedding("", self.get_constant("embedding4_const_0"), embedding_input_1)
        return embedding_output_1


def ids_func(param):
    forge_module, shapes_dtypes, _ = param
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (Embedding0, [((2, 1), torch.float32), ((2049, 2048), torch.bfloat16)], {"model_name": ["pt_musicgen_large"]}),
    (
        Embedding0,
        [((2, 13), torch.float32), ((32128, 768), torch.bfloat16)],
        {"model_name": ["pt_musicgen_large", "pt_musicgen_small", "pt_musicgen_medium"]},
    ),
    (
        Embedding1,
        [((32, 12), torch.bfloat16)],
        {"model_name": ["pt_musicgen_large", "pt_musicgen_small", "pt_musicgen_medium"]},
    ),
    (Embedding0, [((2, 1), torch.float32), ((2049, 1024), torch.bfloat16)], {"model_name": ["pt_musicgen_small"]}),
    (Embedding0, [((2, 1), torch.float32), ((2049, 1536), torch.bfloat16)], {"model_name": ["pt_musicgen_medium"]}),
    (Embedding0, [((1, 2), torch.int32), ((51865, 768), torch.bfloat16)], {"model_name": ["pt_whisper_small"]}),
    (Embedding0, [((1, 2), torch.int32), ((51865, 1280), torch.bfloat16)], {"model_name": ["pt_whisper_large"]}),
    (Embedding0, [((1, 2), torch.int32), ((51865, 1024), torch.bfloat16)], {"model_name": ["pt_whisper_medium"]}),
    (Embedding0, [((1, 2), torch.int32), ((51865, 384), torch.bfloat16)], {"model_name": ["pt_whisper_tiny"]}),
    (Embedding0, [((1, 2), torch.int32), ((51865, 512), torch.bfloat16)], {"model_name": ["pt_whisper_base"]}),
    (
        Embedding0,
        [((1, 2), torch.int32), ((51866, 1280), torch.bfloat16)],
        {"model_name": ["pt_whisper_large_v3_turbo"]},
    ),
    (
        Embedding0,
        [((2, 7), torch.float32), ((49408, 512), torch.bfloat16)],
        {"model_name": ["pt_clip_vit_base_patch32_text"]},
    ),
    (
        Embedding0,
        [((1, 7), torch.float32), ((77, 512), torch.bfloat16)],
        {"model_name": ["pt_clip_vit_base_patch32_text"]},
    ),
    (
        Embedding0,
        [((1, 128), torch.float32), ((30000, 128), torch.bfloat16)],
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
            ]
        },
    ),
    (
        Embedding0,
        [((1, 128), torch.float32), ((2, 128), torch.bfloat16)],
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
            ]
        },
    ),
    (
        Embedding0,
        [((1, 128), torch.float32), ((512, 128), torch.bfloat16)],
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
            ]
        },
    ),
    (Embedding0, [((1, 256), torch.float32), ((50265, 1024), torch.bfloat16)], {"model_name": ["pt_bart"]}),
    (Embedding2, [((1026, 1024), torch.bfloat16)], {"model_name": ["pt_bart"]}),
    (
        Embedding0,
        [((1, 128), torch.float32), ((30522, 768), torch.bfloat16)],
        {
            "model_name": [
                "pt_bert_masked_lm",
                "pt_distilbert_sequence_classification",
                "pt_dpr_ctx_encoder_single_nq_base",
                "pt_dpr_reader_single_nq_base",
                "pt_dpr_reader_multiset_base",
                "pt_dpr_question_encoder_single_nq_base",
                "pt_dpr_ctx_encoder_multiset_base",
                "pt_dpr_question_encoder_multiset_base",
            ]
        },
    ),
    (
        Embedding0,
        [((1, 128), torch.float32), ((2, 768), torch.bfloat16)],
        {
            "model_name": [
                "pt_bert_masked_lm",
                "pt_dpr_ctx_encoder_single_nq_base",
                "pt_dpr_reader_single_nq_base",
                "pt_dpr_reader_multiset_base",
                "pt_dpr_question_encoder_single_nq_base",
                "pt_dpr_ctx_encoder_multiset_base",
                "pt_dpr_question_encoder_multiset_base",
            ]
        },
    ),
    (
        Embedding0,
        [((1, 128), torch.float32), ((512, 768), torch.bfloat16)],
        {
            "model_name": [
                "pt_bert_masked_lm",
                "pt_distilbert_masked_lm",
                "pt_distilbert_sequence_classification",
                "pt_distilbert_token_classification",
                "pt_dpr_ctx_encoder_single_nq_base",
                "pt_dpr_reader_single_nq_base",
                "pt_dpr_reader_multiset_base",
                "pt_dpr_question_encoder_single_nq_base",
                "pt_dpr_ctx_encoder_multiset_base",
                "pt_dpr_question_encoder_multiset_base",
                "pt_squeezebert",
            ]
        },
    ),
    (Embedding0, [((1, 384), torch.float32), ((28996, 1024), torch.bfloat16)], {"model_name": ["pt_bert_qa"]}),
    (Embedding0, [((1, 384), torch.float32), ((2, 1024), torch.bfloat16)], {"model_name": ["pt_bert_qa"]}),
    (Embedding0, [((1, 384), torch.float32), ((512, 1024), torch.bfloat16)], {"model_name": ["pt_bert_qa"]}),
    (
        Embedding0,
        [((1, 128), torch.float32), ((28996, 1024), torch.bfloat16)],
        {"model_name": ["pt_bert_sequence_classification"]},
    ),
    (
        Embedding0,
        [((1, 128), torch.float32), ((2, 1024), torch.bfloat16)],
        {"model_name": ["pt_bert_sequence_classification"]},
    ),
    (
        Embedding0,
        [((1, 128), torch.float32), ((512, 1024), torch.bfloat16)],
        {"model_name": ["pt_bert_sequence_classification"]},
    ),
    (Embedding0, [((1, 256), torch.int32), ((51200, 1024), torch.bfloat16)], {"model_name": ["pt_codegen_350M_mono"]}),
    (
        Embedding0,
        [((1, 128), torch.float32), ((119547, 768), torch.bfloat16)],
        {"model_name": ["pt_distilbert_masked_lm", "pt_distilbert_token_classification"]},
    ),
    (
        Embedding0,
        [((1, 384), torch.float32), ((28996, 768), torch.bfloat16)],
        {"model_name": ["pt_distilbert_question_answering"]},
    ),
    (
        Embedding0,
        [((1, 384), torch.float32), ((512, 768), torch.bfloat16)],
        {"model_name": ["pt_distilbert_question_answering"]},
    ),
    (Embedding0, [((1, 6), torch.float32), ((65024, 4544), torch.bfloat16)], {"model_name": ["pt_falcon"]}),
    (Embedding0, [((1, 7), torch.float32), ((256000, 2048), torch.bfloat16)], {"model_name": ["pt_gemma_2b"]}),
    (
        Embedding0,
        [((1, 256), torch.float32), ((50257, 768), torch.bfloat16)],
        {"model_name": ["pt_gpt2_generation", "pt_gpt_neo_125M_causal_lm"]},
    ),
    (Embedding2, [((1024, 768), torch.bfloat16)], {"model_name": ["pt_gpt2_generation"]}),
    (
        Embedding0,
        [((1, 256), torch.float32), ((50257, 2560), torch.bfloat16)],
        {"model_name": ["pt_gpt_neo_2_7B_causal_lm"]},
    ),
    (Embedding2, [((2048, 2560), torch.bfloat16)], {"model_name": ["pt_gpt_neo_2_7B_causal_lm"]}),
    (
        Embedding0,
        [((1, 256), torch.float32), ((50257, 2048), torch.bfloat16)],
        {"model_name": ["pt_gpt_neo_1_3B_causal_lm"]},
    ),
    (Embedding2, [((2048, 2048), torch.bfloat16)], {"model_name": ["pt_gpt_neo_1_3B_causal_lm"]}),
    (Embedding2, [((2048, 768), torch.bfloat16)], {"model_name": ["pt_gpt_neo_125M_causal_lm"]}),
    (
        Embedding0,
        [((1, 256), torch.int32), ((128256, 4096), torch.bfloat16)],
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
        Embedding0,
        [((1, 256), torch.int32), ((128256, 2048), torch.bfloat16)],
        {"model_name": ["pt_Llama_3_2_1B_Instruct_causal_lm", "pt_Llama_3_2_1B_causal_lm"]},
    ),
    (
        Embedding0,
        [((1, 4), torch.float32), ((128256, 4096), torch.bfloat16)],
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
        Embedding0,
        [((1, 4), torch.float32), ((128256, 2048), torch.bfloat16)],
        {"model_name": ["pt_Llama_3_2_1B_Instruct_seq_cls", "pt_Llama_3_2_1B_seq_cls"]},
    ),
    (Embedding0, [((1, 128), torch.float32), ((32000, 4096), torch.bfloat16)], {"model_name": ["pt_Mistral_7B_v0_1"]}),
    (
        Embedding0,
        [((1, 32), torch.float32), ((50272, 768), torch.bfloat16)],
        {"model_name": ["pt_opt_125m_seq_cls", "pt_opt_125m_qa"]},
    ),
    (
        Embedding0,
        [((1, 32), torch.float32), ((2050, 768), torch.bfloat16)],
        {"model_name": ["pt_opt_125m_seq_cls", "pt_opt_125m_qa"]},
    ),
    (
        Embedding0,
        [((1, 32), torch.float32), ((50272, 2048), torch.bfloat16)],
        {"model_name": ["pt_opt_1_3b_seq_cls", "pt_opt_1_3b_qa"]},
    ),
    (
        Embedding0,
        [((1, 32), torch.float32), ((2050, 2048), torch.bfloat16)],
        {"model_name": ["pt_opt_1_3b_seq_cls", "pt_opt_1_3b_qa"]},
    ),
    (
        Embedding0,
        [((1, 32), torch.float32), ((50272, 512), torch.bfloat16)],
        {"model_name": ["pt_opt_350m_qa", "pt_opt_350m_seq_cls"]},
    ),
    (
        Embedding0,
        [((1, 32), torch.float32), ((2050, 1024), torch.bfloat16)],
        {"model_name": ["pt_opt_350m_qa", "pt_opt_350m_seq_cls"]},
    ),
    (
        Embedding0,
        [((1, 256), torch.float32), ((50272, 512), torch.bfloat16)],
        {"model_name": ["pt_opt_350m_causal_lm"]},
    ),
    (
        Embedding0,
        [((1, 256), torch.float32), ((2050, 1024), torch.bfloat16)],
        {"model_name": ["pt_opt_350m_causal_lm"]},
    ),
    (
        Embedding0,
        [((1, 256), torch.float32), ((50272, 768), torch.bfloat16)],
        {"model_name": ["pt_opt_125m_causal_lm"]},
    ),
    (Embedding0, [((1, 256), torch.float32), ((2050, 768), torch.bfloat16)], {"model_name": ["pt_opt_125m_causal_lm"]}),
    (
        Embedding0,
        [((1, 256), torch.float32), ((50272, 2048), torch.bfloat16)],
        {"model_name": ["pt_opt_1_3b_causal_lm"]},
    ),
    (
        Embedding0,
        [((1, 256), torch.float32), ((2050, 2048), torch.bfloat16)],
        {"model_name": ["pt_opt_1_3b_causal_lm"]},
    ),
    (
        Embedding0,
        [((1, 12), torch.float32), ((51200, 2560), torch.bfloat16)],
        {"model_name": ["pt_phi_2_pytdml_token_cls", "pt_phi_2_token_cls"]},
    ),
    (
        Embedding0,
        [((1, 256), torch.int32), ((51200, 2560), torch.bfloat16)],
        {"model_name": ["pt_phi_2_causal_lm", "pt_phi_2_pytdml_causal_lm"]},
    ),
    (
        Embedding0,
        [((1, 11), torch.float32), ((51200, 2560), torch.bfloat16)],
        {"model_name": ["pt_phi_2_seq_cls", "pt_phi_2_pytdml_seq_cls"]},
    ),
    (Embedding0, [((1, 29), torch.float32), ((151936, 1024), torch.bfloat16)], {"model_name": ["pt_qwen_chat"]}),
    (Embedding0, [((1, 6), torch.float32), ((151936, 1024), torch.bfloat16)], {"model_name": ["pt_qwen_causal_lm"]}),
    (
        Embedding0,
        [((1, 35), torch.float32), ((151936, 1536), torch.bfloat16)],
        {"model_name": ["pt_Qwen_Qwen2_5_Coder_1_5B_Instruct", "pt_Qwen_Qwen2_5_Coder_1_5B"]},
    ),
    (
        Embedding0,
        [((1, 35), torch.float32), ((151936, 2048), torch.bfloat16)],
        {"model_name": ["pt_Qwen_Qwen2_5_Coder_3B", "pt_Qwen_Qwen2_5_Coder_3B_Instruct"]},
    ),
    (
        Embedding0,
        [((1, 35), torch.float32), ((152064, 3584), torch.bfloat16)],
        {"model_name": ["pt_Qwen_Qwen2_5_Coder_7B", "pt_Qwen_Qwen2_5_Coder_7B_Instruct"]},
    ),
    (
        Embedding0,
        [((1, 35), torch.float32), ((151936, 896), torch.bfloat16)],
        {"model_name": ["pt_Qwen_Qwen2_5_Coder_0_5B"]},
    ),
    (
        Embedding0,
        [((1, 39), torch.float32), ((151936, 1536), torch.bfloat16)],
        {"model_name": ["pt_Qwen_Qwen2_5_1_5B_Instruct"]},
    ),
    (
        Embedding0,
        [((1, 29), torch.float32), ((151936, 1536), torch.bfloat16)],
        {"model_name": ["pt_Qwen_Qwen2_5_1_5B"]},
    ),
    (Embedding0, [((1, 29), torch.float32), ((152064, 3584), torch.bfloat16)], {"model_name": ["pt_Qwen_Qwen2_5_7B"]}),
    (
        Embedding0,
        [((1, 39), torch.float32), ((151936, 896), torch.bfloat16)],
        {"model_name": ["pt_Qwen_Qwen2_5_0_5B_Instruct"]},
    ),
    (Embedding0, [((1, 29), torch.float32), ((151936, 896), torch.bfloat16)], {"model_name": ["pt_Qwen_Qwen2_5_0_5B"]}),
    (
        Embedding0,
        [((1, 39), torch.float32), ((151936, 2048), torch.bfloat16)],
        {"model_name": ["pt_Qwen_Qwen2_5_3B_Instruct"]},
    ),
    (
        Embedding0,
        [((1, 39), torch.float32), ((152064, 3584), torch.bfloat16)],
        {"model_name": ["pt_Qwen_Qwen2_5_7B_Instruct"]},
    ),
    (Embedding0, [((1, 29), torch.float32), ((151936, 2048), torch.bfloat16)], {"model_name": ["pt_Qwen_Qwen2_5_3B"]}),
    (
        Embedding0,
        [((1, 128), torch.float32), ((250002, 768), torch.bfloat16)],
        {"model_name": ["pt_roberta_masked_lm"]},
    ),
    (
        Embedding0,
        [((1, 128), torch.float32), ((1, 768), torch.bfloat16)],
        {"model_name": ["pt_roberta_masked_lm", "pt_roberta_sentiment"]},
    ),
    (
        Embedding0,
        [((1, 128), torch.float32), ((514, 768), torch.bfloat16)],
        {"model_name": ["pt_roberta_masked_lm", "pt_roberta_sentiment"]},
    ),
    (Embedding0, [((1, 128), torch.float32), ((50265, 768), torch.bfloat16)], {"model_name": ["pt_roberta_sentiment"]}),
    (Embedding0, [((1, 128), torch.float32), ((30528, 768), torch.bfloat16)], {"model_name": ["pt_squeezebert"]}),
    (Embedding3, [((2, 768), torch.bfloat16)], {"model_name": ["pt_squeezebert"]}),
    (Embedding0, [((1, 1), torch.int32), ((32128, 1024), torch.bfloat16)], {"model_name": ["pt_t5_large"]}),
    (Embedding4, [((32, 16), torch.bfloat16)], {"model_name": ["pt_t5_large"]}),
    (
        Embedding0,
        [((1, 1), torch.int32), ((32128, 768), torch.bfloat16)],
        {"model_name": ["pt_t5_base", "pt_google_flan_t5_base"]},
    ),
    (Embedding4, [((32, 12), torch.bfloat16)], {"model_name": ["pt_t5_base", "pt_google_flan_t5_base"]}),
    (
        Embedding0,
        [((1, 1), torch.int32), ((32128, 512), torch.bfloat16)],
        {"model_name": ["pt_t5_small", "pt_google_flan_t5_small"]},
    ),
    (Embedding4, [((32, 8), torch.bfloat16)], {"model_name": ["pt_t5_small"]}),
    (Embedding4, [((32, 6), torch.bfloat16)], {"model_name": ["pt_google_flan_t5_small"]}),
    (Embedding0, [((1, 256), torch.float32), ((256008, 2048), torch.bfloat16)], {"model_name": ["pt_xglm_1_7B"]}),
    (Embedding0, [((1, 256), torch.float32), ((256008, 1024), torch.bfloat16)], {"model_name": ["pt_xglm_564M"]}),
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
