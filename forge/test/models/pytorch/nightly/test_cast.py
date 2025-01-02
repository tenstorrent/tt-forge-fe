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


class Cast0(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, cast_input_0):
        cast_output_1 = forge.op.Cast("", cast_input_0, dtype=torch.bfloat16)
        return cast_output_1


class Cast1(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, cast_input_0):
        cast_output_1 = forge.op.Cast("", cast_input_0, dtype=torch.float32)
        return cast_output_1


class Cast2(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, cast_input_0):
        cast_output_1 = forge.op.Cast("", cast_input_0, dtype=torch.bool)
        return cast_output_1


class Cast3(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, cast_input_0):
        cast_output_1 = forge.op.Cast("", cast_input_0, dtype=torch.int32)
        return cast_output_1


class Cast4(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, cast_input_0):
        cast_output_1 = forge.op.Cast("", cast_input_0, dtype=torch.int64)
        return cast_output_1


def ids_func(param):
    forge_module, shapes_dtypes, _ = param
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (Cast0, [((2049, 2048), torch.float32)], {"model_name": ["pt_musicgen_large"]}),
    (Cast1, [((2, 1, 2048), torch.bfloat16)], {"model_name": ["pt_musicgen_large"]}),
    (
        Cast0,
        [((32128, 768), torch.float32)],
        {
            "model_name": [
                "pt_musicgen_large",
                "pt_musicgen_small",
                "pt_musicgen_medium",
                "pt_t5_base",
                "pt_google_flan_t5_base",
            ]
        },
    ),
    (
        Cast1,
        [((2, 13, 768), torch.bfloat16)],
        {"model_name": ["pt_musicgen_large", "pt_musicgen_small", "pt_musicgen_medium"]},
    ),
    (
        Cast0,
        [((32, 12), torch.float32)],
        {
            "model_name": [
                "pt_musicgen_large",
                "pt_musicgen_small",
                "pt_musicgen_medium",
                "pt_t5_base",
                "pt_google_flan_t5_base",
            ]
        },
    ),
    (
        Cast1,
        [((13, 13, 12), torch.bfloat16)],
        {"model_name": ["pt_musicgen_large", "pt_musicgen_small", "pt_musicgen_medium"]},
    ),
    (
        Cast1,
        [((2, 1, 1, 13), torch.float32)],
        {"model_name": ["pt_musicgen_large", "pt_musicgen_small", "pt_musicgen_medium"]},
    ),
    (
        Cast1,
        [((2, 13, 1), torch.float32)],
        {"model_name": ["pt_musicgen_large", "pt_musicgen_small", "pt_musicgen_medium"]},
    ),
    (
        Cast1,
        [((2, 1, 1, 13), torch.float32)],
        {"model_name": ["pt_musicgen_large", "pt_musicgen_small", "pt_musicgen_medium"]},
    ),
    (Cast0, [((2049, 1024), torch.float32)], {"model_name": ["pt_musicgen_small"]}),
    (Cast1, [((2, 1, 1024), torch.bfloat16)], {"model_name": ["pt_musicgen_small"]}),
    (Cast0, [((2049, 1536), torch.float32)], {"model_name": ["pt_musicgen_medium"]}),
    (Cast1, [((2, 1, 1536), torch.bfloat16)], {"model_name": ["pt_musicgen_medium"]}),
    (Cast0, [((51865, 768), torch.float32)], {"model_name": ["pt_whisper_small"]}),
    (Cast1, [((1, 2, 768), torch.bfloat16)], {"model_name": ["pt_whisper_small"]}),
    (Cast0, [((51865, 1280), torch.float32)], {"model_name": ["pt_whisper_large"]}),
    (Cast1, [((1, 2, 1280), torch.bfloat16)], {"model_name": ["pt_whisper_large", "pt_whisper_large_v3_turbo"]}),
    (Cast0, [((51865, 1024), torch.float32)], {"model_name": ["pt_whisper_medium"]}),
    (Cast1, [((1, 2, 1024), torch.bfloat16)], {"model_name": ["pt_whisper_medium"]}),
    (Cast0, [((51865, 384), torch.float32)], {"model_name": ["pt_whisper_tiny"]}),
    (Cast1, [((1, 2, 384), torch.bfloat16)], {"model_name": ["pt_whisper_tiny"]}),
    (Cast0, [((51865, 512), torch.float32)], {"model_name": ["pt_whisper_base"]}),
    (Cast1, [((1, 2, 512), torch.bfloat16)], {"model_name": ["pt_whisper_base"]}),
    (Cast0, [((51866, 1280), torch.float32)], {"model_name": ["pt_whisper_large_v3_turbo"]}),
    (Cast0, [((49408, 512), torch.float32)], {"model_name": ["pt_clip_vit_base_patch32_text"]}),
    (Cast1, [((2, 7, 512), torch.bfloat16)], {"model_name": ["pt_clip_vit_base_patch32_text"]}),
    (Cast0, [((77, 512), torch.float32)], {"model_name": ["pt_clip_vit_base_patch32_text"]}),
    (Cast1, [((1, 7, 512), torch.bfloat16)], {"model_name": ["pt_clip_vit_base_patch32_text"]}),
    (Cast1, [((2, 1, 7, 7), torch.float32)], {"model_name": ["pt_clip_vit_base_patch32_text"]}),
    (Cast1, [((2, 1, 7, 7), torch.float32)], {"model_name": ["pt_clip_vit_base_patch32_text"]}),
    (
        Cast0,
        [((30000, 128), torch.float32)],
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
        Cast1,
        [((1, 128, 128), torch.bfloat16)],
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
        Cast0,
        [((2, 128), torch.float32)],
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
        Cast0,
        [((512, 128), torch.float32)],
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
        Cast1,
        [((1, 1, 1, 128), torch.float32)],
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
                "pt_dpr_ctx_encoder_single_nq_base",
                "pt_dpr_reader_single_nq_base",
                "pt_dpr_reader_multiset_base",
                "pt_dpr_question_encoder_single_nq_base",
                "pt_dpr_ctx_encoder_multiset_base",
                "pt_dpr_question_encoder_multiset_base",
                "pt_roberta_masked_lm",
            ]
        },
    ),
    (Cast0, [((50265, 1024), torch.float32)], {"model_name": ["pt_bart"]}),
    (
        Cast1,
        [((1, 256, 1024), torch.bfloat16)],
        {"model_name": ["pt_bart", "pt_codegen_350M_mono", "pt_opt_350m_causal_lm", "pt_xglm_564M"]},
    ),
    (Cast0, [((1026, 1024), torch.float32)], {"model_name": ["pt_bart"]}),
    (
        Cast1,
        [((1, 1, 256, 256), torch.float32)],
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
        Cast1,
        [((1, 1, 256, 256), torch.float32)],
        {
            "model_name": [
                "pt_bart",
                "pt_Llama_3_1_8B_Instruct_causal_lm",
                "pt_Meta_Llama_3_8B_causal_lm",
                "pt_Llama_3_2_1B_Instruct_causal_lm",
                "pt_Meta_Llama_3_8B_Instruct_causal_lm",
                "pt_Llama_3_2_1B_causal_lm",
                "pt_Llama_3_1_8B_causal_lm",
                "pt_opt_350m_causal_lm",
                "pt_opt_125m_causal_lm",
                "pt_opt_1_3b_causal_lm",
                "pt_xglm_1_7B",
                "pt_xglm_564M",
            ]
        },
    ),
    (
        Cast2,
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
    (
        Cast3,
        [((1, 1, 256, 256), torch.float32)],
        {
            "model_name": [
                "pt_Llama_3_1_8B_Instruct_causal_lm",
                "pt_Meta_Llama_3_8B_causal_lm",
                "pt_Llama_3_2_1B_Instruct_causal_lm",
                "pt_Meta_Llama_3_8B_Instruct_causal_lm",
                "pt_Llama_3_2_1B_causal_lm",
                "pt_Llama_3_1_8B_causal_lm",
            ]
        },
    ),
    (
        Cast0,
        [((30522, 768), torch.float32)],
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
        Cast1,
        [((1, 128, 768), torch.bfloat16)],
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
                "pt_roberta_masked_lm",
                "pt_roberta_sentiment",
                "pt_squeezebert",
            ]
        },
    ),
    (
        Cast0,
        [((2, 768), torch.float32)],
        {
            "model_name": [
                "pt_bert_masked_lm",
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
    (
        Cast0,
        [((512, 768), torch.float32)],
        {
            "model_name": [
                "pt_bert_masked_lm",
                "pt_distilbert_masked_lm",
                "pt_distilbert_sequence_classification",
                "pt_distilbert_question_answering",
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
    (Cast0, [((28996, 1024), torch.float32)], {"model_name": ["pt_bert_qa", "pt_bert_sequence_classification"]}),
    (Cast1, [((1, 384, 1024), torch.bfloat16)], {"model_name": ["pt_bert_qa"]}),
    (Cast0, [((2, 1024), torch.float32)], {"model_name": ["pt_bert_qa", "pt_bert_sequence_classification"]}),
    (Cast0, [((512, 1024), torch.float32)], {"model_name": ["pt_bert_qa", "pt_bert_sequence_classification"]}),
    (Cast1, [((1, 128, 1024), torch.bfloat16)], {"model_name": ["pt_bert_sequence_classification"]}),
    (Cast0, [((51200, 1024), torch.float32)], {"model_name": ["pt_codegen_350M_mono"]}),
    (
        Cast0,
        [((119547, 768), torch.float32)],
        {"model_name": ["pt_distilbert_masked_lm", "pt_distilbert_token_classification"]},
    ),
    (
        Cast2,
        [((1, 128), torch.float32)],
        {
            "model_name": [
                "pt_distilbert_masked_lm",
                "pt_distilbert_sequence_classification",
                "pt_distilbert_token_classification",
            ]
        },
    ),
    (
        Cast3,
        [((1, 128), torch.float32)],
        {
            "model_name": [
                "pt_distilbert_masked_lm",
                "pt_distilbert_sequence_classification",
                "pt_distilbert_token_classification",
                "pt_roberta_masked_lm",
                "pt_roberta_sentiment",
            ]
        },
    ),
    (
        Cast2,
        [((1, 128), torch.int32)],
        {
            "model_name": [
                "pt_distilbert_masked_lm",
                "pt_distilbert_sequence_classification",
                "pt_distilbert_token_classification",
            ]
        },
    ),
    (Cast4, [((1, 128), torch.int32)], {"model_name": ["pt_roberta_masked_lm", "pt_roberta_sentiment"]}),
    (
        Cast1,
        [((1, 12, 128, 128), torch.float32)],
        {
            "model_name": [
                "pt_distilbert_masked_lm",
                "pt_distilbert_sequence_classification",
                "pt_distilbert_token_classification",
            ]
        },
    ),
    (Cast0, [((28996, 768), torch.float32)], {"model_name": ["pt_distilbert_question_answering"]}),
    (Cast1, [((1, 384, 768), torch.bfloat16)], {"model_name": ["pt_distilbert_question_answering"]}),
    (Cast2, [((1, 384), torch.float32)], {"model_name": ["pt_distilbert_question_answering"]}),
    (Cast3, [((1, 384), torch.float32)], {"model_name": ["pt_distilbert_question_answering"]}),
    (Cast2, [((1, 384), torch.int32)], {"model_name": ["pt_distilbert_question_answering"]}),
    (Cast1, [((1, 12, 384, 384), torch.float32)], {"model_name": ["pt_distilbert_question_answering"]}),
    (Cast0, [((65024, 4544), torch.float32)], {"model_name": ["pt_falcon"]}),
    (Cast1, [((1, 6, 4544), torch.bfloat16)], {"model_name": ["pt_falcon"]}),
    (Cast0, [((256000, 2048), torch.float32)], {"model_name": ["pt_gemma_2b"]}),
    (Cast1, [((1, 7, 2048), torch.bfloat16)], {"model_name": ["pt_gemma_2b"]}),
    (Cast0, [((50257, 768), torch.float32)], {"model_name": ["pt_gpt2_generation", "pt_gpt_neo_125M_causal_lm"]}),
    (
        Cast1,
        [((1, 256, 768), torch.bfloat16)],
        {"model_name": ["pt_gpt2_generation", "pt_gpt_neo_125M_causal_lm", "pt_opt_125m_causal_lm"]},
    ),
    (Cast0, [((1024, 768), torch.float32)], {"model_name": ["pt_gpt2_generation"]}),
    (Cast0, [((50257, 2560), torch.float32)], {"model_name": ["pt_gpt_neo_2_7B_causal_lm"]}),
    (
        Cast1,
        [((1, 256, 2560), torch.bfloat16)],
        {"model_name": ["pt_gpt_neo_2_7B_causal_lm", "pt_phi_2_causal_lm", "pt_phi_2_pytdml_causal_lm"]},
    ),
    (Cast0, [((2048, 2560), torch.float32)], {"model_name": ["pt_gpt_neo_2_7B_causal_lm"]}),
    (Cast0, [((50257, 2048), torch.float32)], {"model_name": ["pt_gpt_neo_1_3B_causal_lm"]}),
    (
        Cast1,
        [((1, 256, 2048), torch.bfloat16)],
        {
            "model_name": [
                "pt_gpt_neo_1_3B_causal_lm",
                "pt_Llama_3_2_1B_Instruct_causal_lm",
                "pt_Llama_3_2_1B_causal_lm",
                "pt_opt_1_3b_causal_lm",
                "pt_xglm_1_7B",
            ]
        },
    ),
    (Cast0, [((2048, 2048), torch.float32)], {"model_name": ["pt_gpt_neo_1_3B_causal_lm"]}),
    (Cast0, [((2048, 768), torch.float32)], {"model_name": ["pt_gpt_neo_125M_causal_lm"]}),
    (
        Cast0,
        [((128256, 4096), torch.float32)],
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
            ]
        },
    ),
    (
        Cast1,
        [((1, 256, 4096), torch.bfloat16)],
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
        Cast2,
        [((1, 1, 256, 256), torch.float32)],
        {
            "model_name": [
                "pt_Llama_3_1_8B_Instruct_causal_lm",
                "pt_Meta_Llama_3_8B_causal_lm",
                "pt_Llama_3_2_1B_Instruct_causal_lm",
                "pt_Meta_Llama_3_8B_Instruct_causal_lm",
                "pt_Llama_3_2_1B_causal_lm",
                "pt_Llama_3_1_8B_causal_lm",
            ]
        },
    ),
    (
        Cast2,
        [((1, 1, 256, 256), torch.int32)],
        {
            "model_name": [
                "pt_Llama_3_1_8B_Instruct_causal_lm",
                "pt_Meta_Llama_3_8B_causal_lm",
                "pt_Llama_3_2_1B_Instruct_causal_lm",
                "pt_Meta_Llama_3_8B_Instruct_causal_lm",
                "pt_Llama_3_2_1B_causal_lm",
                "pt_Llama_3_1_8B_causal_lm",
            ]
        },
    ),
    (
        Cast0,
        [((128256, 2048), torch.float32)],
        {
            "model_name": [
                "pt_Llama_3_2_1B_Instruct_causal_lm",
                "pt_Llama_3_2_1B_causal_lm",
                "pt_Llama_3_2_1B_Instruct_seq_cls",
                "pt_Llama_3_2_1B_seq_cls",
            ]
        },
    ),
    (
        Cast1,
        [((1, 4, 4096), torch.bfloat16)],
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
        Cast1,
        [((1, 4, 2048), torch.bfloat16)],
        {"model_name": ["pt_Llama_3_2_1B_Instruct_seq_cls", "pt_Llama_3_2_1B_seq_cls"]},
    ),
    (Cast0, [((32000, 4096), torch.float32)], {"model_name": ["pt_Mistral_7B_v0_1"]}),
    (Cast1, [((1, 128, 4096), torch.bfloat16)], {"model_name": ["pt_Mistral_7B_v0_1"]}),
    (
        Cast0,
        [((50272, 768), torch.float32)],
        {"model_name": ["pt_opt_125m_seq_cls", "pt_opt_125m_qa", "pt_opt_125m_causal_lm"]},
    ),
    (Cast1, [((1, 32, 768), torch.bfloat16)], {"model_name": ["pt_opt_125m_seq_cls", "pt_opt_125m_qa"]}),
    (
        Cast0,
        [((2050, 768), torch.float32)],
        {"model_name": ["pt_opt_125m_seq_cls", "pt_opt_125m_qa", "pt_opt_125m_causal_lm"]},
    ),
    (
        Cast1,
        [((1, 1, 32, 32), torch.float32)],
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
        Cast1,
        [((1, 1, 32, 32), torch.float32)],
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
        Cast3,
        [((1, 32), torch.float32)],
        {"model_name": ["pt_opt_125m_seq_cls", "pt_opt_1_3b_seq_cls", "pt_opt_350m_seq_cls"]},
    ),
    (
        Cast4,
        [((1,), torch.int32)],
        {"model_name": ["pt_opt_125m_seq_cls", "pt_opt_1_3b_seq_cls", "pt_opt_350m_seq_cls"]},
    ),
    (
        Cast0,
        [((50272, 2048), torch.float32)],
        {"model_name": ["pt_opt_1_3b_seq_cls", "pt_opt_1_3b_qa", "pt_opt_1_3b_causal_lm"]},
    ),
    (Cast1, [((1, 32, 2048), torch.bfloat16)], {"model_name": ["pt_opt_1_3b_seq_cls", "pt_opt_1_3b_qa"]}),
    (
        Cast0,
        [((2050, 2048), torch.float32)],
        {"model_name": ["pt_opt_1_3b_seq_cls", "pt_opt_1_3b_qa", "pt_opt_1_3b_causal_lm"]},
    ),
    (
        Cast0,
        [((50272, 512), torch.float32)],
        {"model_name": ["pt_opt_350m_qa", "pt_opt_350m_causal_lm", "pt_opt_350m_seq_cls"]},
    ),
    (Cast1, [((1, 32, 512), torch.bfloat16)], {"model_name": ["pt_opt_350m_qa", "pt_opt_350m_seq_cls"]}),
    (
        Cast0,
        [((2050, 1024), torch.float32)],
        {"model_name": ["pt_opt_350m_qa", "pt_opt_350m_causal_lm", "pt_opt_350m_seq_cls"]},
    ),
    (Cast1, [((1, 32, 1024), torch.bfloat16)], {"model_name": ["pt_opt_350m_qa", "pt_opt_350m_seq_cls"]}),
    (Cast1, [((1, 256, 512), torch.bfloat16)], {"model_name": ["pt_opt_350m_causal_lm"]}),
    (
        Cast0,
        [((51200, 2560), torch.float32)],
        {
            "model_name": [
                "pt_phi_2_pytdml_token_cls",
                "pt_phi_2_causal_lm",
                "pt_phi_2_seq_cls",
                "pt_phi_2_token_cls",
                "pt_phi_2_pytdml_seq_cls",
                "pt_phi_2_pytdml_causal_lm",
            ]
        },
    ),
    (Cast1, [((1, 12, 2560), torch.bfloat16)], {"model_name": ["pt_phi_2_pytdml_token_cls", "pt_phi_2_token_cls"]}),
    (Cast1, [((1, 11, 2560), torch.bfloat16)], {"model_name": ["pt_phi_2_seq_cls", "pt_phi_2_pytdml_seq_cls"]}),
    (Cast0, [((151936, 1024), torch.float32)], {"model_name": ["pt_qwen_chat", "pt_qwen_causal_lm"]}),
    (Cast1, [((1, 29, 1024), torch.bfloat16)], {"model_name": ["pt_qwen_chat"]}),
    (Cast1, [((1, 6, 1024), torch.bfloat16)], {"model_name": ["pt_qwen_causal_lm"]}),
    (
        Cast0,
        [((151936, 1536), torch.float32)],
        {
            "model_name": [
                "pt_Qwen_Qwen2_5_Coder_1_5B_Instruct",
                "pt_Qwen_Qwen2_5_Coder_1_5B",
                "pt_Qwen_Qwen2_5_1_5B_Instruct",
                "pt_Qwen_Qwen2_5_1_5B",
            ]
        },
    ),
    (
        Cast1,
        [((1, 35, 1536), torch.bfloat16)],
        {"model_name": ["pt_Qwen_Qwen2_5_Coder_1_5B_Instruct", "pt_Qwen_Qwen2_5_Coder_1_5B"]},
    ),
    (
        Cast0,
        [((151936, 2048), torch.float32)],
        {
            "model_name": [
                "pt_Qwen_Qwen2_5_Coder_3B",
                "pt_Qwen_Qwen2_5_Coder_3B_Instruct",
                "pt_Qwen_Qwen2_5_3B_Instruct",
                "pt_Qwen_Qwen2_5_3B",
            ]
        },
    ),
    (
        Cast1,
        [((1, 35, 2048), torch.bfloat16)],
        {"model_name": ["pt_Qwen_Qwen2_5_Coder_3B", "pt_Qwen_Qwen2_5_Coder_3B_Instruct"]},
    ),
    (
        Cast0,
        [((152064, 3584), torch.float32)],
        {
            "model_name": [
                "pt_Qwen_Qwen2_5_Coder_7B",
                "pt_Qwen_Qwen2_5_Coder_7B_Instruct",
                "pt_Qwen_Qwen2_5_7B",
                "pt_Qwen_Qwen2_5_7B_Instruct",
            ]
        },
    ),
    (
        Cast1,
        [((1, 35, 3584), torch.bfloat16)],
        {"model_name": ["pt_Qwen_Qwen2_5_Coder_7B", "pt_Qwen_Qwen2_5_Coder_7B_Instruct"]},
    ),
    (
        Cast0,
        [((151936, 896), torch.float32)],
        {"model_name": ["pt_Qwen_Qwen2_5_Coder_0_5B", "pt_Qwen_Qwen2_5_0_5B_Instruct", "pt_Qwen_Qwen2_5_0_5B"]},
    ),
    (Cast1, [((1, 35, 896), torch.bfloat16)], {"model_name": ["pt_Qwen_Qwen2_5_Coder_0_5B"]}),
    (Cast1, [((1, 39, 1536), torch.bfloat16)], {"model_name": ["pt_Qwen_Qwen2_5_1_5B_Instruct"]}),
    (Cast1, [((1, 29, 1536), torch.bfloat16)], {"model_name": ["pt_Qwen_Qwen2_5_1_5B"]}),
    (Cast1, [((1, 29, 3584), torch.bfloat16)], {"model_name": ["pt_Qwen_Qwen2_5_7B"]}),
    (Cast1, [((1, 39, 896), torch.bfloat16)], {"model_name": ["pt_Qwen_Qwen2_5_0_5B_Instruct"]}),
    (Cast1, [((1, 29, 896), torch.bfloat16)], {"model_name": ["pt_Qwen_Qwen2_5_0_5B"]}),
    (Cast1, [((1, 39, 2048), torch.bfloat16)], {"model_name": ["pt_Qwen_Qwen2_5_3B_Instruct"]}),
    (Cast1, [((1, 39, 3584), torch.bfloat16)], {"model_name": ["pt_Qwen_Qwen2_5_7B_Instruct"]}),
    (Cast1, [((1, 29, 2048), torch.bfloat16)], {"model_name": ["pt_Qwen_Qwen2_5_3B"]}),
    (Cast0, [((250002, 768), torch.float32)], {"model_name": ["pt_roberta_masked_lm"]}),
    (Cast0, [((1, 768), torch.float32)], {"model_name": ["pt_roberta_masked_lm", "pt_roberta_sentiment"]}),
    (Cast0, [((514, 768), torch.float32)], {"model_name": ["pt_roberta_masked_lm", "pt_roberta_sentiment"]}),
    (Cast0, [((50265, 768), torch.float32)], {"model_name": ["pt_roberta_sentiment"]}),
    (Cast0, [((30528, 768), torch.float32)], {"model_name": ["pt_squeezebert"]}),
    (Cast0, [((32128, 1024), torch.float32)], {"model_name": ["pt_t5_large"]}),
    (Cast1, [((1, 1, 1024), torch.bfloat16)], {"model_name": ["pt_t5_large"]}),
    (Cast0, [((32, 16), torch.float32)], {"model_name": ["pt_t5_large"]}),
    (Cast1, [((1, 1, 16), torch.bfloat16)], {"model_name": ["pt_t5_large"]}),
    (Cast1, [((1, 1, 768), torch.bfloat16)], {"model_name": ["pt_t5_base", "pt_google_flan_t5_base"]}),
    (Cast1, [((1, 1, 12), torch.bfloat16)], {"model_name": ["pt_t5_base", "pt_google_flan_t5_base"]}),
    (Cast0, [((32128, 512), torch.float32)], {"model_name": ["pt_t5_small", "pt_google_flan_t5_small"]}),
    (Cast1, [((1, 1, 512), torch.bfloat16)], {"model_name": ["pt_t5_small", "pt_google_flan_t5_small"]}),
    (Cast0, [((32, 8), torch.float32)], {"model_name": ["pt_t5_small"]}),
    (Cast1, [((1, 1, 8), torch.bfloat16)], {"model_name": ["pt_t5_small"]}),
    (Cast0, [((32, 6), torch.float32)], {"model_name": ["pt_google_flan_t5_small"]}),
    (Cast1, [((1, 1, 6), torch.bfloat16)], {"model_name": ["pt_google_flan_t5_small"]}),
    (Cast0, [((256008, 2048), torch.float32)], {"model_name": ["pt_xglm_1_7B"]}),
    (Cast0, [((256008, 1024), torch.float32)], {"model_name": ["pt_xglm_564M"]}),
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
