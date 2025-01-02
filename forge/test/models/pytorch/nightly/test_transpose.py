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


class Transpose2(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, transpose_input_0):
        transpose_output_1 = forge.op.Transpose("", transpose_input_0, dim0=-3, dim1=-1)
        return transpose_output_1


class Transpose3(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, transpose_input_0):
        transpose_output_1 = forge.op.Transpose("", transpose_input_0, dim0=-4, dim1=-3)
        return transpose_output_1


def ids_func(param):
    forge_module, shapes_dtypes, _ = param
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (
        Transpose0,
        [((2048, 2048), torch.float32)],
        {
            "model_name": [
                "pt_musicgen_large",
                "pt_albert_xlarge_v1_token_cls",
                "pt_albert_xlarge_v2_token_cls",
                "pt_albert_xlarge_v1_masked_lm",
                "pt_albert_xlarge_v2_masked_lm",
                "pt_gemma_2b",
                "pt_gpt_neo_1_3B_causal_lm",
                "pt_Llama_3_2_1B_Instruct_causal_lm",
                "pt_Llama_3_2_1B_causal_lm",
                "pt_Llama_3_2_1B_Instruct_seq_cls",
                "pt_Llama_3_2_1B_seq_cls",
                "pt_opt_1_3b_seq_cls",
                "pt_opt_1_3b_qa",
                "pt_opt_1_3b_causal_lm",
                "pt_Qwen_Qwen2_5_Coder_3B",
                "pt_Qwen_Qwen2_5_Coder_3B_Instruct",
                "pt_Qwen_Qwen2_5_3B_Instruct",
                "pt_Qwen_Qwen2_5_3B",
                "pt_xglm_1_7B",
                "nbeats_seasonality",
            ]
        },
    ),
    (Transpose1, [((2, 1, 32, 64), torch.float32)], {"model_name": ["pt_musicgen_large"]}),
    (Transpose0, [((64, 1, 64), torch.float32)], {"model_name": ["pt_musicgen_large"]}),
    (Transpose0, [((64, 64, 1), torch.float32)], {"model_name": ["pt_musicgen_large"]}),
    (Transpose1, [((2, 32, 1, 64), torch.float32)], {"model_name": ["pt_musicgen_large"]}),
    (
        Transpose0,
        [((768, 768), torch.float32)],
        {
            "model_name": [
                "pt_musicgen_large",
                "pt_musicgen_small",
                "pt_musicgen_medium",
                "pt_whisper_small",
                "pt_ViLt_maskedlm",
                "pt_ViLt_question_answering",
                "pt_albert_base_v2_masked_lm",
                "pt_albert_base_v1_masked_lm",
                "pt_albert_base_v1_token_cls",
                "pt_albert_base_v2_token_cls",
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
                "pt_gpt_neo_125M_causal_lm",
                "pt_opt_125m_seq_cls",
                "pt_opt_125m_qa",
                "pt_opt_125m_causal_lm",
                "pt_roberta_masked_lm",
                "pt_roberta_sentiment",
                "pt_squeezebert",
                "pt_t5_base",
                "pt_google_flan_t5_base",
                "pt_deit_base_patch16_224",
                "pt_deit_base_distilled_patch16_224",
                "pt_swinv2_tiny_patch4_window8_256",
                "pt_vit_base_patch16_224",
            ]
        },
    ),
    (
        Transpose1,
        [((2, 13, 12, 64), torch.float32)],
        {"model_name": ["pt_musicgen_large", "pt_musicgen_small", "pt_musicgen_medium"]},
    ),
    (
        Transpose0,
        [((24, 13, 64), torch.float32)],
        {"model_name": ["pt_musicgen_large", "pt_musicgen_small", "pt_musicgen_medium"]},
    ),
    (
        Transpose2,
        [((13, 13, 12), torch.float32)],
        {"model_name": ["pt_musicgen_large", "pt_musicgen_small", "pt_musicgen_medium"]},
    ),
    (
        Transpose0,
        [((12, 13, 13), torch.float32)],
        {"model_name": ["pt_musicgen_large", "pt_musicgen_small", "pt_musicgen_medium"]},
    ),
    (
        Transpose0,
        [((2, 12, 13, 64), torch.float32)],
        {"model_name": ["pt_musicgen_large", "pt_musicgen_small", "pt_musicgen_medium"]},
    ),
    (
        Transpose1,
        [((2, 12, 13, 64), torch.float32)],
        {"model_name": ["pt_musicgen_large", "pt_musicgen_small", "pt_musicgen_medium"]},
    ),
    (
        Transpose0,
        [((24, 64, 13), torch.float32)],
        {"model_name": ["pt_musicgen_large", "pt_musicgen_small", "pt_musicgen_medium"]},
    ),
    (
        Transpose0,
        [((3072, 768), torch.float32)],
        {
            "model_name": [
                "pt_musicgen_large",
                "pt_musicgen_small",
                "pt_musicgen_medium",
                "pt_whisper_small",
                "pt_ViLt_maskedlm",
                "pt_ViLt_question_answering",
                "pt_albert_base_v2_masked_lm",
                "pt_albert_base_v1_masked_lm",
                "pt_albert_base_v1_token_cls",
                "pt_albert_base_v2_token_cls",
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
                "pt_gpt_neo_125M_causal_lm",
                "pt_opt_125m_seq_cls",
                "pt_opt_125m_qa",
                "pt_opt_125m_causal_lm",
                "pt_roberta_masked_lm",
                "pt_roberta_sentiment",
                "pt_t5_base",
                "pt_deit_base_patch16_224",
                "pt_deit_base_distilled_patch16_224",
                "pt_mixer_b16_224_miil_in21k",
                "pt_mixer_b16_224",
                "pt_mixer_b32_224",
                "pt_mixer_b16_224_in21k",
                "pt_mixer_b16_224_miil",
                "pt_swinv2_tiny_patch4_window8_256",
                "pt_vit_base_patch16_224",
            ]
        },
    ),
    (
        Transpose0,
        [((768, 3072), torch.float32)],
        {
            "model_name": [
                "pt_musicgen_large",
                "pt_musicgen_small",
                "pt_musicgen_medium",
                "pt_whisper_small",
                "pt_ViLt_maskedlm",
                "pt_ViLt_question_answering",
                "pt_albert_base_v2_masked_lm",
                "pt_albert_base_v1_masked_lm",
                "pt_albert_base_v1_token_cls",
                "pt_albert_base_v2_token_cls",
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
                "pt_gpt_neo_125M_causal_lm",
                "pt_opt_125m_seq_cls",
                "pt_opt_125m_qa",
                "pt_opt_125m_causal_lm",
                "pt_roberta_masked_lm",
                "pt_roberta_sentiment",
                "pt_t5_base",
                "pt_deit_base_patch16_224",
                "pt_deit_base_distilled_patch16_224",
                "pt_mixer_b16_224_miil_in21k",
                "pt_mixer_b16_224",
                "pt_mixer_b32_224",
                "pt_mixer_b16_224_in21k",
                "pt_mixer_b16_224_miil",
                "pt_swinv2_tiny_patch4_window8_256",
                "pt_vit_base_patch16_224",
            ]
        },
    ),
    (Transpose0, [((2048, 768), torch.float32)], {"model_name": ["pt_musicgen_large", "pt_google_flan_t5_base"]}),
    (Transpose1, [((2, 13, 32, 64), torch.float32)], {"model_name": ["pt_musicgen_large"]}),
    (Transpose0, [((64, 13, 64), torch.float32)], {"model_name": ["pt_musicgen_large"]}),
    (Transpose0, [((64, 64, 13), torch.float32)], {"model_name": ["pt_musicgen_large"]}),
    (
        Transpose0,
        [((8192, 2048), torch.float32)],
        {
            "model_name": [
                "pt_musicgen_large",
                "pt_albert_xlarge_v1_token_cls",
                "pt_albert_xlarge_v2_token_cls",
                "pt_albert_xlarge_v1_masked_lm",
                "pt_albert_xlarge_v2_masked_lm",
                "pt_gpt_neo_1_3B_causal_lm",
                "pt_Llama_3_2_1B_Instruct_causal_lm",
                "pt_Llama_3_2_1B_causal_lm",
                "pt_Llama_3_2_1B_Instruct_seq_cls",
                "pt_Llama_3_2_1B_seq_cls",
                "pt_opt_1_3b_seq_cls",
                "pt_opt_1_3b_qa",
                "pt_opt_1_3b_causal_lm",
                "pt_xglm_1_7B",
            ]
        },
    ),
    (
        Transpose0,
        [((2048, 8192), torch.float32)],
        {
            "model_name": [
                "pt_musicgen_large",
                "pt_albert_xlarge_v1_token_cls",
                "pt_albert_xlarge_v2_token_cls",
                "pt_albert_xlarge_v1_masked_lm",
                "pt_albert_xlarge_v2_masked_lm",
                "pt_gpt_neo_1_3B_causal_lm",
                "pt_Llama_3_2_1B_Instruct_causal_lm",
                "pt_Llama_3_2_1B_causal_lm",
                "pt_Llama_3_2_1B_Instruct_seq_cls",
                "pt_Llama_3_2_1B_seq_cls",
                "pt_opt_1_3b_seq_cls",
                "pt_opt_1_3b_qa",
                "pt_opt_1_3b_causal_lm",
                "pt_xglm_1_7B",
            ]
        },
    ),
    (
        Transpose0,
        [((1024, 1024), torch.float32)],
        {
            "model_name": [
                "pt_musicgen_small",
                "pt_whisper_medium",
                "pt_albert_large_v1_masked_lm",
                "pt_albert_large_v2_token_cls",
                "pt_albert_large_v2_masked_lm",
                "pt_albert_large_v1_token_cls",
                "pt_bart",
                "pt_bert_qa",
                "pt_bert_sequence_classification",
                "pt_codegen_350M_mono",
                "pt_opt_350m_qa",
                "pt_opt_350m_causal_lm",
                "pt_opt_350m_seq_cls",
                "pt_qwen_chat",
                "pt_qwen_causal_lm",
                "pt_t5_large",
                "pt_xglm_564M",
                "pt_vision_perceiver_learned",
                "pt_vision_perceiver_conv",
                "pt_vision_perceiver_fourier",
                "pt_vit_large_patch16_224",
            ]
        },
    ),
    (Transpose1, [((2, 1, 16, 64), torch.float32)], {"model_name": ["pt_musicgen_small"]}),
    (Transpose0, [((32, 1, 64), torch.float32)], {"model_name": ["pt_musicgen_small"]}),
    (Transpose0, [((32, 64, 1), torch.float32)], {"model_name": ["pt_musicgen_small"]}),
    (Transpose1, [((2, 16, 1, 64), torch.float32)], {"model_name": ["pt_musicgen_small"]}),
    (Transpose0, [((1024, 768), torch.float32)], {"model_name": ["pt_musicgen_small"]}),
    (Transpose1, [((2, 13, 16, 64), torch.float32)], {"model_name": ["pt_musicgen_small"]}),
    (Transpose0, [((32, 13, 64), torch.float32)], {"model_name": ["pt_musicgen_small"]}),
    (Transpose0, [((32, 64, 13), torch.float32)], {"model_name": ["pt_musicgen_small"]}),
    (
        Transpose0,
        [((4096, 1024), torch.float32)],
        {
            "model_name": [
                "pt_musicgen_small",
                "pt_whisper_medium",
                "pt_albert_large_v1_masked_lm",
                "pt_albert_large_v2_token_cls",
                "pt_albert_large_v2_masked_lm",
                "pt_albert_large_v1_token_cls",
                "pt_bart",
                "pt_bert_qa",
                "pt_bert_sequence_classification",
                "pt_codegen_350M_mono",
                "pt_opt_350m_qa",
                "pt_opt_350m_causal_lm",
                "pt_opt_350m_seq_cls",
                "pt_t5_large",
                "pt_xglm_564M",
                "pt_mixer_l32_224",
                "pt_mixer_l16_224",
                "pt_mixer_l16_224_in21k",
                "pt_vit_large_patch16_224",
            ]
        },
    ),
    (
        Transpose0,
        [((1024, 4096), torch.float32)],
        {
            "model_name": [
                "pt_musicgen_small",
                "pt_whisper_medium",
                "pt_albert_large_v1_masked_lm",
                "pt_albert_large_v2_token_cls",
                "pt_albert_large_v2_masked_lm",
                "pt_albert_large_v1_token_cls",
                "pt_bart",
                "pt_bert_qa",
                "pt_bert_sequence_classification",
                "pt_codegen_350M_mono",
                "pt_Llama_3_1_8B_Instruct_causal_lm",
                "pt_Meta_Llama_3_8B_causal_lm",
                "pt_Meta_Llama_3_8B_Instruct_causal_lm",
                "pt_Llama_3_1_8B_Instruct_seq_cls",
                "pt_Meta_Llama_3_8B_seq_cls",
                "pt_Llama_3_1_8B_seq_cls",
                "pt_Llama_3_1_8B_causal_lm",
                "pt_Meta_Llama_3_8B_Instruct_seq_cls",
                "pt_opt_350m_qa",
                "pt_opt_350m_causal_lm",
                "pt_opt_350m_seq_cls",
                "pt_t5_large",
                "pt_xglm_564M",
                "pt_mixer_l32_224",
                "pt_mixer_l16_224",
                "pt_mixer_l16_224_in21k",
                "pt_vit_large_patch16_224",
            ]
        },
    ),
    (Transpose0, [((2048, 1024), torch.float32)], {"model_name": ["pt_musicgen_small"]}),
    (
        Transpose0,
        [((1536, 1536), torch.float32)],
        {
            "model_name": [
                "pt_musicgen_medium",
                "pt_Qwen_Qwen2_5_Coder_1_5B_Instruct",
                "pt_Qwen_Qwen2_5_Coder_1_5B",
                "pt_Qwen_Qwen2_5_1_5B_Instruct",
                "pt_Qwen_Qwen2_5_1_5B",
            ]
        },
    ),
    (Transpose1, [((2, 1, 24, 64), torch.float32)], {"model_name": ["pt_musicgen_medium"]}),
    (Transpose0, [((48, 1, 64), torch.float32)], {"model_name": ["pt_musicgen_medium"]}),
    (Transpose0, [((48, 64, 1), torch.float32)], {"model_name": ["pt_musicgen_medium"]}),
    (Transpose1, [((2, 24, 1, 64), torch.float32)], {"model_name": ["pt_musicgen_medium"]}),
    (Transpose0, [((1536, 768), torch.float32)], {"model_name": ["pt_musicgen_medium", "pt_ViLt_question_answering"]}),
    (Transpose1, [((2, 13, 24, 64), torch.float32)], {"model_name": ["pt_musicgen_medium"]}),
    (Transpose0, [((48, 13, 64), torch.float32)], {"model_name": ["pt_musicgen_medium"]}),
    (Transpose0, [((48, 64, 13), torch.float32)], {"model_name": ["pt_musicgen_medium"]}),
    (Transpose0, [((6144, 1536), torch.float32)], {"model_name": ["pt_musicgen_medium"]}),
    (Transpose0, [((1536, 6144), torch.float32)], {"model_name": ["pt_musicgen_medium"]}),
    (Transpose0, [((2048, 1536), torch.float32)], {"model_name": ["pt_musicgen_medium"]}),
    (Transpose1, [((1, 2, 12, 64), torch.float32)], {"model_name": ["pt_whisper_small"]}),
    (Transpose0, [((12, 2, 64), torch.float32)], {"model_name": ["pt_whisper_small"]}),
    (Transpose0, [((1, 12, 2, 64), torch.float32)], {"model_name": ["pt_whisper_small"]}),
    (Transpose1, [((1, 12, 2, 64), torch.float32)], {"model_name": ["pt_whisper_small"]}),
    (Transpose0, [((12, 64, 2), torch.float32)], {"model_name": ["pt_whisper_small"]}),
    (Transpose1, [((1, 1500, 12, 64), torch.float32)], {"model_name": ["pt_whisper_small"]}),
    (Transpose0, [((12, 1500, 64), torch.float32)], {"model_name": ["pt_whisper_small"]}),
    (Transpose0, [((1, 12, 1500, 64), torch.float32)], {"model_name": ["pt_whisper_small"]}),
    (Transpose0, [((12, 64, 1500), torch.float32)], {"model_name": ["pt_whisper_small"]}),
    (Transpose0, [((51865, 768), torch.float32)], {"model_name": ["pt_whisper_small"]}),
    (Transpose0, [((1280, 1280), torch.float32)], {"model_name": ["pt_whisper_large", "pt_whisper_large_v3_turbo"]}),
    (Transpose1, [((1, 2, 20, 64), torch.float32)], {"model_name": ["pt_whisper_large", "pt_whisper_large_v3_turbo"]}),
    (Transpose0, [((20, 2, 64), torch.float32)], {"model_name": ["pt_whisper_large", "pt_whisper_large_v3_turbo"]}),
    (Transpose0, [((1, 20, 2, 64), torch.float32)], {"model_name": ["pt_whisper_large", "pt_whisper_large_v3_turbo"]}),
    (Transpose1, [((1, 20, 2, 64), torch.float32)], {"model_name": ["pt_whisper_large", "pt_whisper_large_v3_turbo"]}),
    (Transpose0, [((20, 64, 2), torch.float32)], {"model_name": ["pt_whisper_large", "pt_whisper_large_v3_turbo"]}),
    (
        Transpose1,
        [((1, 1500, 20, 64), torch.float32)],
        {"model_name": ["pt_whisper_large", "pt_whisper_large_v3_turbo"]},
    ),
    (Transpose0, [((20, 1500, 64), torch.float32)], {"model_name": ["pt_whisper_large", "pt_whisper_large_v3_turbo"]}),
    (
        Transpose0,
        [((1, 20, 1500, 64), torch.float32)],
        {"model_name": ["pt_whisper_large", "pt_whisper_large_v3_turbo"]},
    ),
    (Transpose0, [((20, 64, 1500), torch.float32)], {"model_name": ["pt_whisper_large", "pt_whisper_large_v3_turbo"]}),
    (Transpose0, [((5120, 1280), torch.float32)], {"model_name": ["pt_whisper_large", "pt_whisper_large_v3_turbo"]}),
    (Transpose0, [((1280, 5120), torch.float32)], {"model_name": ["pt_whisper_large", "pt_whisper_large_v3_turbo"]}),
    (Transpose0, [((51865, 1280), torch.float32)], {"model_name": ["pt_whisper_large"]}),
    (Transpose1, [((1, 2, 16, 64), torch.float32)], {"model_name": ["pt_whisper_medium"]}),
    (Transpose0, [((16, 2, 64), torch.float32)], {"model_name": ["pt_whisper_medium"]}),
    (Transpose0, [((1, 16, 2, 64), torch.float32)], {"model_name": ["pt_whisper_medium"]}),
    (Transpose1, [((1, 16, 2, 64), torch.float32)], {"model_name": ["pt_whisper_medium"]}),
    (Transpose0, [((16, 64, 2), torch.float32)], {"model_name": ["pt_whisper_medium"]}),
    (Transpose1, [((1, 1500, 16, 64), torch.float32)], {"model_name": ["pt_whisper_medium"]}),
    (Transpose0, [((16, 1500, 64), torch.float32)], {"model_name": ["pt_whisper_medium"]}),
    (Transpose0, [((1, 16, 1500, 64), torch.float32)], {"model_name": ["pt_whisper_medium"]}),
    (Transpose0, [((16, 64, 1500), torch.float32)], {"model_name": ["pt_whisper_medium"]}),
    (Transpose0, [((51865, 1024), torch.float32)], {"model_name": ["pt_whisper_medium"]}),
    (
        Transpose0,
        [((384, 384), torch.float32)],
        {"model_name": ["pt_whisper_tiny", "pt_deit_small_patch16_224", "pt_swinv2_tiny_patch4_window8_256"]},
    ),
    (Transpose1, [((1, 2, 6, 64), torch.float32)], {"model_name": ["pt_whisper_tiny"]}),
    (Transpose0, [((6, 2, 64), torch.float32)], {"model_name": ["pt_whisper_tiny"]}),
    (Transpose0, [((1, 6, 2, 64), torch.float32)], {"model_name": ["pt_whisper_tiny"]}),
    (Transpose1, [((1, 6, 2, 64), torch.float32)], {"model_name": ["pt_whisper_tiny"]}),
    (Transpose0, [((6, 64, 2), torch.float32)], {"model_name": ["pt_whisper_tiny"]}),
    (Transpose1, [((1, 1500, 6, 64), torch.float32)], {"model_name": ["pt_whisper_tiny"]}),
    (Transpose0, [((6, 1500, 64), torch.float32)], {"model_name": ["pt_whisper_tiny"]}),
    (Transpose0, [((1, 6, 1500, 64), torch.float32)], {"model_name": ["pt_whisper_tiny"]}),
    (Transpose0, [((6, 64, 1500), torch.float32)], {"model_name": ["pt_whisper_tiny"]}),
    (
        Transpose0,
        [((1536, 384), torch.float32)],
        {"model_name": ["pt_whisper_tiny", "pt_deit_small_patch16_224", "pt_swinv2_tiny_patch4_window8_256"]},
    ),
    (
        Transpose0,
        [((384, 1536), torch.float32)],
        {"model_name": ["pt_whisper_tiny", "pt_deit_small_patch16_224", "pt_swinv2_tiny_patch4_window8_256"]},
    ),
    (Transpose0, [((51865, 384), torch.float32)], {"model_name": ["pt_whisper_tiny"]}),
    (
        Transpose0,
        [((512, 512), torch.float32)],
        {
            "model_name": [
                "pt_whisper_base",
                "pt_clip_vit_base_patch32_text",
                "pt_t5_small",
                "nbeats_generic",
                "pt_vision_perceiver_learned",
                "pt_mit_b4",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (Transpose1, [((1, 2, 8, 64), torch.float32)], {"model_name": ["pt_whisper_base"]}),
    (Transpose0, [((8, 2, 64), torch.float32)], {"model_name": ["pt_whisper_base"]}),
    (Transpose0, [((1, 8, 2, 64), torch.float32)], {"model_name": ["pt_whisper_base"]}),
    (Transpose1, [((1, 8, 2, 64), torch.float32)], {"model_name": ["pt_whisper_base"]}),
    (Transpose0, [((8, 64, 2), torch.float32)], {"model_name": ["pt_whisper_base"]}),
    (Transpose1, [((1, 1500, 8, 64), torch.float32)], {"model_name": ["pt_whisper_base"]}),
    (Transpose0, [((8, 1500, 64), torch.float32)], {"model_name": ["pt_whisper_base"]}),
    (Transpose0, [((1, 8, 1500, 64), torch.float32)], {"model_name": ["pt_whisper_base"]}),
    (Transpose0, [((8, 64, 1500), torch.float32)], {"model_name": ["pt_whisper_base"]}),
    (
        Transpose0,
        [((2048, 512), torch.float32)],
        {
            "model_name": [
                "pt_whisper_base",
                "pt_clip_vit_base_patch32_text",
                "pt_t5_small",
                "pt_mixer_s32_224",
                "pt_mixer_s16_224",
                "pt_mit_b4",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Transpose0,
        [((512, 2048), torch.float32)],
        {
            "model_name": [
                "pt_whisper_base",
                "pt_clip_vit_base_patch32_text",
                "pt_Llama_3_2_1B_Instruct_causal_lm",
                "pt_Llama_3_2_1B_causal_lm",
                "pt_Llama_3_2_1B_Instruct_seq_cls",
                "pt_Llama_3_2_1B_seq_cls",
                "pt_t5_small",
                "pt_mixer_s32_224",
                "pt_mixer_s16_224",
                "pt_mit_b4",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (Transpose0, [((51865, 512), torch.float32)], {"model_name": ["pt_whisper_base"]}),
    (Transpose0, [((51866, 1280), torch.float32)], {"model_name": ["pt_whisper_large_v3_turbo"]}),
    (Transpose1, [((2, 7, 8, 64), torch.float32)], {"model_name": ["pt_clip_vit_base_patch32_text"]}),
    (Transpose0, [((16, 7, 64), torch.float32)], {"model_name": ["pt_clip_vit_base_patch32_text"]}),
    (Transpose0, [((16, 64, 7), torch.float32)], {"model_name": ["pt_clip_vit_base_patch32_text"]}),
    (Transpose1, [((2, 8, 7, 64), torch.float32)], {"model_name": ["pt_clip_vit_base_patch32_text"]}),
    (Transpose1, [((1, 204, 12, 64), torch.float32)], {"model_name": ["pt_ViLt_maskedlm"]}),
    (Transpose0, [((12, 204, 64), torch.float32)], {"model_name": ["pt_ViLt_maskedlm"]}),
    (Transpose0, [((1, 12, 204, 64), torch.float32)], {"model_name": ["pt_ViLt_maskedlm"]}),
    (Transpose1, [((1, 12, 204, 64), torch.float32)], {"model_name": ["pt_ViLt_maskedlm"]}),
    (Transpose0, [((12, 64, 204), torch.float32)], {"model_name": ["pt_ViLt_maskedlm"]}),
    (Transpose0, [((30522, 768), torch.float32)], {"model_name": ["pt_ViLt_maskedlm", "pt_bert_masked_lm"]}),
    (Transpose1, [((1, 201, 12, 64), torch.float32)], {"model_name": ["pt_ViLt_question_answering"]}),
    (Transpose0, [((12, 201, 64), torch.float32)], {"model_name": ["pt_ViLt_question_answering"]}),
    (Transpose0, [((1, 12, 201, 64), torch.float32)], {"model_name": ["pt_ViLt_question_answering"]}),
    (Transpose1, [((1, 12, 201, 64), torch.float32)], {"model_name": ["pt_ViLt_question_answering"]}),
    (Transpose0, [((12, 64, 201), torch.float32)], {"model_name": ["pt_ViLt_question_answering"]}),
    (Transpose0, [((3129, 1536), torch.float32)], {"model_name": ["pt_ViLt_question_answering"]}),
    (
        Transpose0,
        [((768, 128), torch.float32)],
        {
            "model_name": [
                "pt_albert_base_v2_masked_lm",
                "pt_albert_base_v1_masked_lm",
                "pt_albert_base_v1_token_cls",
                "pt_albert_base_v2_token_cls",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
            ]
        },
    ),
    (
        Transpose1,
        [((1, 128, 12, 64), torch.float32)],
        {
            "model_name": [
                "pt_albert_base_v2_masked_lm",
                "pt_albert_base_v1_masked_lm",
                "pt_albert_base_v1_token_cls",
                "pt_albert_base_v2_token_cls",
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
            ]
        },
    ),
    (
        Transpose0,
        [((12, 128, 64), torch.float32)],
        {
            "model_name": [
                "pt_albert_base_v2_masked_lm",
                "pt_albert_base_v1_masked_lm",
                "pt_albert_base_v1_token_cls",
                "pt_albert_base_v2_token_cls",
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
        Transpose0,
        [((1, 12, 128, 64), torch.float32)],
        {
            "model_name": [
                "pt_albert_base_v2_masked_lm",
                "pt_albert_base_v1_masked_lm",
                "pt_albert_base_v1_token_cls",
                "pt_albert_base_v2_token_cls",
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
        Transpose1,
        [((1, 12, 128, 64), torch.float32)],
        {
            "model_name": [
                "pt_albert_base_v2_masked_lm",
                "pt_albert_base_v1_masked_lm",
                "pt_albert_base_v1_token_cls",
                "pt_albert_base_v2_token_cls",
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
            ]
        },
    ),
    (
        Transpose0,
        [((12, 64, 128), torch.float32)],
        {
            "model_name": [
                "pt_albert_base_v2_masked_lm",
                "pt_albert_base_v1_masked_lm",
                "pt_albert_base_v1_token_cls",
                "pt_albert_base_v2_token_cls",
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
        Transpose0,
        [((128, 768), torch.float32)],
        {"model_name": ["pt_albert_base_v2_masked_lm", "pt_albert_base_v1_masked_lm"]},
    ),
    (
        Transpose0,
        [((30000, 128), torch.float32)],
        {
            "model_name": [
                "pt_albert_base_v2_masked_lm",
                "pt_albert_large_v1_masked_lm",
                "pt_albert_large_v2_masked_lm",
                "pt_albert_base_v1_masked_lm",
                "pt_albert_xxlarge_v2_masked_lm",
                "pt_albert_xlarge_v1_masked_lm",
                "pt_albert_xxlarge_v1_masked_lm",
                "pt_albert_xlarge_v2_masked_lm",
            ]
        },
    ),
    (
        Transpose0,
        [((2048, 128), torch.float32)],
        {
            "model_name": [
                "pt_albert_xlarge_v1_token_cls",
                "pt_albert_xlarge_v2_token_cls",
                "pt_albert_xlarge_v1_masked_lm",
                "pt_albert_xlarge_v2_masked_lm",
            ]
        },
    ),
    (
        Transpose1,
        [((1, 128, 16, 128), torch.float32)],
        {
            "model_name": [
                "pt_albert_xlarge_v1_token_cls",
                "pt_albert_xlarge_v2_token_cls",
                "pt_albert_xlarge_v1_masked_lm",
                "pt_albert_xlarge_v2_masked_lm",
            ]
        },
    ),
    (
        Transpose0,
        [((16, 128, 128), torch.float32)],
        {
            "model_name": [
                "pt_albert_xlarge_v1_token_cls",
                "pt_albert_xlarge_v2_token_cls",
                "pt_albert_xlarge_v1_masked_lm",
                "pt_albert_xlarge_v2_masked_lm",
            ]
        },
    ),
    (
        Transpose0,
        [((1, 16, 128, 128), torch.float32)],
        {
            "model_name": [
                "pt_albert_xlarge_v1_token_cls",
                "pt_albert_xlarge_v2_token_cls",
                "pt_albert_xlarge_v1_masked_lm",
                "pt_albert_xlarge_v2_masked_lm",
            ]
        },
    ),
    (
        Transpose1,
        [((1, 16, 128, 128), torch.float32)],
        {
            "model_name": [
                "pt_albert_xlarge_v1_token_cls",
                "pt_albert_xlarge_v2_token_cls",
                "pt_albert_xlarge_v1_masked_lm",
                "pt_albert_xlarge_v2_masked_lm",
            ]
        },
    ),
    (
        Transpose0,
        [((2, 2048), torch.float32)],
        {
            "model_name": [
                "pt_albert_xlarge_v1_token_cls",
                "pt_albert_xlarge_v2_token_cls",
                "pt_Llama_3_2_1B_Instruct_seq_cls",
                "pt_Llama_3_2_1B_seq_cls",
                "pt_opt_1_3b_seq_cls",
            ]
        },
    ),
    (
        Transpose0,
        [((1024, 128), torch.float32)],
        {
            "model_name": [
                "pt_albert_large_v1_masked_lm",
                "pt_albert_large_v2_token_cls",
                "pt_albert_large_v2_masked_lm",
                "pt_albert_large_v1_token_cls",
            ]
        },
    ),
    (
        Transpose1,
        [((1, 128, 16, 64), torch.float32)],
        {
            "model_name": [
                "pt_albert_large_v1_masked_lm",
                "pt_albert_large_v2_token_cls",
                "pt_albert_large_v2_masked_lm",
                "pt_albert_large_v1_token_cls",
                "pt_bert_sequence_classification",
            ]
        },
    ),
    (
        Transpose0,
        [((16, 128, 64), torch.float32)],
        {
            "model_name": [
                "pt_albert_large_v1_masked_lm",
                "pt_albert_large_v2_token_cls",
                "pt_albert_large_v2_masked_lm",
                "pt_albert_large_v1_token_cls",
                "pt_bert_sequence_classification",
            ]
        },
    ),
    (
        Transpose0,
        [((1, 16, 128, 64), torch.float32)],
        {
            "model_name": [
                "pt_albert_large_v1_masked_lm",
                "pt_albert_large_v2_token_cls",
                "pt_albert_large_v2_masked_lm",
                "pt_albert_large_v1_token_cls",
                "pt_bert_sequence_classification",
            ]
        },
    ),
    (
        Transpose1,
        [((1, 16, 128, 64), torch.float32)],
        {
            "model_name": [
                "pt_albert_large_v1_masked_lm",
                "pt_albert_large_v2_token_cls",
                "pt_albert_large_v2_masked_lm",
                "pt_albert_large_v1_token_cls",
                "pt_bert_sequence_classification",
            ]
        },
    ),
    (
        Transpose0,
        [((16, 64, 128), torch.float32)],
        {
            "model_name": [
                "pt_albert_large_v1_masked_lm",
                "pt_albert_large_v2_token_cls",
                "pt_albert_large_v2_masked_lm",
                "pt_albert_large_v1_token_cls",
                "pt_bert_sequence_classification",
            ]
        },
    ),
    (
        Transpose0,
        [((128, 1024), torch.float32)],
        {"model_name": ["pt_albert_large_v1_masked_lm", "pt_albert_large_v2_masked_lm"]},
    ),
    (
        Transpose0,
        [((4096, 128), torch.float32)],
        {
            "model_name": [
                "pt_albert_xxlarge_v1_token_cls",
                "pt_albert_xxlarge_v2_masked_lm",
                "pt_albert_xxlarge_v1_masked_lm",
                "pt_albert_xxlarge_v2_token_cls",
            ]
        },
    ),
    (
        Transpose0,
        [((4096, 4096), torch.float32)],
        {
            "model_name": [
                "pt_albert_xxlarge_v1_token_cls",
                "pt_albert_xxlarge_v2_masked_lm",
                "pt_albert_xxlarge_v1_masked_lm",
                "pt_albert_xxlarge_v2_token_cls",
                "pt_fuyu_8b",
                "pt_Llama_3_1_8B_Instruct_causal_lm",
                "pt_Meta_Llama_3_8B_causal_lm",
                "pt_Meta_Llama_3_8B_Instruct_causal_lm",
                "pt_Llama_3_1_8B_Instruct_seq_cls",
                "pt_Meta_Llama_3_8B_seq_cls",
                "pt_Llama_3_1_8B_seq_cls",
                "pt_Llama_3_1_8B_causal_lm",
                "pt_Meta_Llama_3_8B_Instruct_seq_cls",
                "pt_alexnet_torchhub",
                "pt_vgg13_osmr",
                "pt_bn_vgg19_osmr",
                "pt_bn_vgg19b_osmr",
                "pt_vgg16_osmr",
                "pt_vgg19_osmr",
                "pt_vgg_19_hf",
                "pt_vgg_bn19_torchhub",
                "pt_vgg11_osmr",
            ]
        },
    ),
    (
        Transpose1,
        [((1, 128, 64, 64), torch.float32)],
        {
            "model_name": [
                "pt_albert_xxlarge_v1_token_cls",
                "pt_albert_xxlarge_v2_masked_lm",
                "pt_albert_xxlarge_v1_masked_lm",
                "pt_albert_xxlarge_v2_token_cls",
            ]
        },
    ),
    (
        Transpose0,
        [((1, 128, 64, 64), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Transpose0,
        [((64, 128, 64), torch.float32)],
        {
            "model_name": [
                "pt_albert_xxlarge_v1_token_cls",
                "pt_albert_xxlarge_v2_masked_lm",
                "pt_albert_xxlarge_v1_masked_lm",
                "pt_albert_xxlarge_v2_token_cls",
            ]
        },
    ),
    (
        Transpose0,
        [((1, 64, 128, 64), torch.float32)],
        {
            "model_name": [
                "pt_albert_xxlarge_v1_token_cls",
                "pt_albert_xxlarge_v2_masked_lm",
                "pt_albert_xxlarge_v1_masked_lm",
                "pt_albert_xxlarge_v2_token_cls",
            ]
        },
    ),
    (
        Transpose1,
        [((1, 64, 128, 64), torch.float32)],
        {
            "model_name": [
                "pt_albert_xxlarge_v1_token_cls",
                "pt_albert_xxlarge_v2_masked_lm",
                "pt_albert_xxlarge_v1_masked_lm",
                "pt_albert_xxlarge_v2_token_cls",
            ]
        },
    ),
    (
        Transpose0,
        [((64, 64, 128), torch.float32)],
        {
            "model_name": [
                "pt_albert_xxlarge_v1_token_cls",
                "pt_albert_xxlarge_v2_masked_lm",
                "pt_albert_xxlarge_v1_masked_lm",
                "pt_albert_xxlarge_v2_token_cls",
            ]
        },
    ),
    (
        Transpose0,
        [((16384, 4096), torch.float32)],
        {
            "model_name": [
                "pt_albert_xxlarge_v1_token_cls",
                "pt_albert_xxlarge_v2_masked_lm",
                "pt_albert_xxlarge_v1_masked_lm",
                "pt_albert_xxlarge_v2_token_cls",
                "pt_fuyu_8b",
            ]
        },
    ),
    (
        Transpose0,
        [((4096, 16384), torch.float32)],
        {
            "model_name": [
                "pt_albert_xxlarge_v1_token_cls",
                "pt_albert_xxlarge_v2_masked_lm",
                "pt_albert_xxlarge_v1_masked_lm",
                "pt_albert_xxlarge_v2_token_cls",
                "pt_fuyu_8b",
            ]
        },
    ),
    (
        Transpose0,
        [((2, 4096), torch.float32)],
        {
            "model_name": [
                "pt_albert_xxlarge_v1_token_cls",
                "pt_albert_xxlarge_v2_token_cls",
                "pt_Llama_3_1_8B_Instruct_seq_cls",
                "pt_Meta_Llama_3_8B_seq_cls",
                "pt_Llama_3_1_8B_seq_cls",
                "pt_Meta_Llama_3_8B_Instruct_seq_cls",
            ]
        },
    ),
    (
        Transpose0,
        [((2, 1024), torch.float32)],
        {"model_name": ["pt_albert_large_v2_token_cls", "pt_albert_large_v1_token_cls"]},
    ),
    (
        Transpose0,
        [((128, 4096), torch.float32)],
        {"model_name": ["pt_albert_xxlarge_v2_masked_lm", "pt_albert_xxlarge_v1_masked_lm"]},
    ),
    (
        Transpose0,
        [((128, 2048), torch.float32)],
        {"model_name": ["pt_albert_xlarge_v1_masked_lm", "pt_albert_xlarge_v2_masked_lm"]},
    ),
    (
        Transpose0,
        [((2, 768), torch.float32)],
        {
            "model_name": [
                "pt_albert_base_v1_token_cls",
                "pt_albert_base_v2_token_cls",
                "pt_distilbert_sequence_classification",
                "pt_opt_125m_seq_cls",
            ]
        },
    ),
    (
        Transpose1,
        [((1, 256, 16, 64), torch.float32)],
        {"model_name": ["pt_bart", "pt_codegen_350M_mono", "pt_opt_350m_causal_lm", "pt_t5_large", "pt_xglm_564M"]},
    ),
    (
        Transpose0,
        [((16, 256, 64), torch.float32)],
        {"model_name": ["pt_bart", "pt_codegen_350M_mono", "pt_opt_350m_causal_lm", "pt_t5_large", "pt_xglm_564M"]},
    ),
    (
        Transpose0,
        [((16, 64, 256), torch.float32)],
        {"model_name": ["pt_bart", "pt_codegen_350M_mono", "pt_opt_350m_causal_lm", "pt_t5_large", "pt_xglm_564M"]},
    ),
    (
        Transpose1,
        [((1, 16, 256, 64), torch.float32)],
        {"model_name": ["pt_bart", "pt_codegen_350M_mono", "pt_opt_350m_causal_lm", "pt_xglm_564M"]},
    ),
    (Transpose0, [((1, 16, 256, 64), torch.float32)], {"model_name": ["pt_codegen_350M_mono", "pt_t5_large"]}),
    (Transpose1, [((1, 384, 16, 64), torch.float32)], {"model_name": ["pt_bert_qa"]}),
    (Transpose0, [((16, 384, 64), torch.float32)], {"model_name": ["pt_bert_qa"]}),
    (Transpose0, [((1, 16, 384, 64), torch.float32)], {"model_name": ["pt_bert_qa"]}),
    (Transpose1, [((1, 16, 384, 64), torch.float32)], {"model_name": ["pt_bert_qa"]}),
    (Transpose0, [((16, 64, 384), torch.float32)], {"model_name": ["pt_bert_qa"]}),
    (Transpose0, [((1, 1024), torch.float32)], {"model_name": ["pt_bert_qa"]}),
    (
        Transpose0,
        [((9, 1024), torch.float32)],
        {"model_name": ["pt_bert_sequence_classification", "pt_mobilenet_v1_basic"]},
    ),
    (Transpose0, [((1024, 1024), torch.float32)], {"model_name": ["pt_codegen_350M_mono"]}),
    (Transpose0, [((51200, 1024), torch.float32)], {"model_name": ["pt_codegen_350M_mono"]}),
    (Transpose0, [((119547, 768), torch.float32)], {"model_name": ["pt_distilbert_masked_lm"]}),
    (Transpose1, [((1, 384, 12, 64), torch.float32)], {"model_name": ["pt_distilbert_question_answering"]}),
    (Transpose0, [((12, 384, 64), torch.float32)], {"model_name": ["pt_distilbert_question_answering"]}),
    (Transpose0, [((1, 12, 384, 64), torch.float32)], {"model_name": ["pt_distilbert_question_answering"]}),
    (Transpose1, [((1, 12, 384, 64), torch.float32)], {"model_name": ["pt_distilbert_question_answering"]}),
    (Transpose0, [((12, 64, 384), torch.float32)], {"model_name": ["pt_distilbert_question_answering"]}),
    (
        Transpose0,
        [((1, 768), torch.float32)],
        {
            "model_name": [
                "pt_distilbert_question_answering",
                "pt_dpr_reader_single_nq_base",
                "pt_dpr_reader_multiset_base",
                "pt_opt_125m_qa",
            ]
        },
    ),
    (Transpose0, [((9, 768), torch.float32)], {"model_name": ["pt_distilbert_token_classification"]}),
    (
        Transpose0,
        [((1, 768), torch.float32)],
        {"model_name": ["pt_dpr_reader_single_nq_base", "pt_dpr_reader_multiset_base"]},
    ),
    (Transpose0, [((18176, 4544), torch.float32)], {"model_name": ["pt_falcon"]}),
    (Transpose0, [((4544, 18176), torch.float32)], {"model_name": ["pt_falcon"]}),
    (Transpose0, [((4672, 4544), torch.float32)], {"model_name": ["pt_falcon"]}),
    (Transpose1, [((1, 6, 71, 64), torch.float32)], {"model_name": ["pt_falcon"]}),
    (Transpose0, [((1, 32, 6), torch.float32)], {"model_name": ["pt_falcon", "pt_qwen_causal_lm"]}),
    (Transpose1, [((1, 6, 1, 64), torch.float32)], {"model_name": ["pt_falcon", "pt_google_flan_t5_small"]}),
    (Transpose0, [((1, 6, 1, 64), torch.float32)], {"model_name": ["pt_google_flan_t5_small"]}),
    (Transpose0, [((1, 6, 64), torch.float32)], {"model_name": ["pt_falcon"]}),
    (Transpose1, [((1, 71, 6, 64), torch.float32)], {"model_name": ["pt_falcon"]}),
    (Transpose0, [((4544, 4544), torch.float32)], {"model_name": ["pt_falcon"]}),
    (Transpose0, [((65024, 4544), torch.float32)], {"model_name": ["pt_falcon"]}),
    (Transpose0, [((12288, 4096), torch.float32)], {"model_name": ["pt_fuyu_8b"]}),
    (Transpose1, [((1, 334, 64, 64), torch.float32)], {"model_name": ["pt_fuyu_8b"]}),
    (Transpose0, [((1, 16, 334), torch.float32)], {"model_name": ["pt_fuyu_8b"]}),
    (Transpose0, [((64, 334, 64), torch.float32)], {"model_name": ["pt_fuyu_8b"]}),
    (Transpose0, [((1, 64, 334, 64), torch.float32)], {"model_name": ["pt_fuyu_8b"]}),
    (Transpose1, [((1, 64, 334, 64), torch.float32)], {"model_name": ["pt_fuyu_8b"]}),
    (Transpose0, [((64, 64, 334), torch.float32)], {"model_name": ["pt_fuyu_8b"]}),
    (Transpose1, [((1, 7, 8, 256), torch.float32)], {"model_name": ["pt_gemma_2b"]}),
    (Transpose0, [((1, 128, 7), torch.float32)], {"model_name": ["pt_gemma_2b"]}),
    (
        Transpose0,
        [((256, 2048), torch.float32)],
        {
            "model_name": [
                "pt_gemma_2b",
                "pt_Qwen_Qwen2_5_Coder_3B",
                "pt_Qwen_Qwen2_5_Coder_3B_Instruct",
                "pt_Qwen_Qwen2_5_3B_Instruct",
                "pt_Qwen_Qwen2_5_3B",
            ]
        },
    ),
    (Transpose1, [((1, 7, 1, 256), torch.float32)], {"model_name": ["pt_gemma_2b"]}),
    (Transpose0, [((8, 7, 256), torch.float32)], {"model_name": ["pt_gemma_2b"]}),
    (Transpose0, [((1, 8, 7, 256), torch.float32)], {"model_name": ["pt_gemma_2b"]}),
    (Transpose1, [((1, 8, 7, 256), torch.float32)], {"model_name": ["pt_gemma_2b"]}),
    (Transpose0, [((8, 256, 7), torch.float32)], {"model_name": ["pt_gemma_2b"]}),
    (Transpose0, [((16384, 2048), torch.float32)], {"model_name": ["pt_gemma_2b"]}),
    (Transpose0, [((2048, 16384), torch.float32)], {"model_name": ["pt_gemma_2b"]}),
    (Transpose0, [((256000, 2048), torch.float32)], {"model_name": ["pt_gemma_2b"]}),
    (Transpose0, [((768, 2304), torch.float32)], {"model_name": ["pt_gpt2_generation"]}),
    (Transpose0, [((768, 768), torch.float32)], {"model_name": ["pt_gpt2_generation"]}),
    (
        Transpose1,
        [((1, 256, 12, 64), torch.float32)],
        {
            "model_name": [
                "pt_gpt2_generation",
                "pt_gpt_neo_125M_causal_lm",
                "pt_opt_125m_causal_lm",
                "pt_t5_base",
                "pt_google_flan_t5_base",
            ]
        },
    ),
    (
        Transpose0,
        [((12, 256, 64), torch.float32)],
        {
            "model_name": [
                "pt_gpt2_generation",
                "pt_gpt_neo_125M_causal_lm",
                "pt_opt_125m_causal_lm",
                "pt_t5_base",
                "pt_google_flan_t5_base",
            ]
        },
    ),
    (
        Transpose0,
        [((1, 12, 256, 64), torch.float32)],
        {"model_name": ["pt_gpt2_generation", "pt_gpt_neo_125M_causal_lm", "pt_t5_base", "pt_google_flan_t5_base"]},
    ),
    (
        Transpose1,
        [((1, 12, 256, 64), torch.float32)],
        {"model_name": ["pt_gpt2_generation", "pt_gpt_neo_125M_causal_lm", "pt_opt_125m_causal_lm"]},
    ),
    (
        Transpose0,
        [((12, 64, 256), torch.float32)],
        {
            "model_name": [
                "pt_gpt2_generation",
                "pt_gpt_neo_125M_causal_lm",
                "pt_opt_125m_causal_lm",
                "pt_t5_base",
                "pt_google_flan_t5_base",
            ]
        },
    ),
    (Transpose0, [((50257, 768), torch.float32)], {"model_name": ["pt_gpt2_generation", "pt_gpt_neo_125M_causal_lm"]}),
    (
        Transpose0,
        [((2560, 2560), torch.float32)],
        {
            "model_name": [
                "pt_gpt_neo_2_7B_causal_lm",
                "pt_phi_2_pytdml_token_cls",
                "pt_phi_2_causal_lm",
                "pt_phi_2_seq_cls",
                "pt_phi_2_token_cls",
                "pt_phi_2_pytdml_seq_cls",
                "pt_phi_2_pytdml_causal_lm",
            ]
        },
    ),
    (Transpose1, [((1, 256, 20, 128), torch.float32)], {"model_name": ["pt_gpt_neo_2_7B_causal_lm"]}),
    (Transpose0, [((20, 256, 128), torch.float32)], {"model_name": ["pt_gpt_neo_2_7B_causal_lm"]}),
    (Transpose0, [((1, 20, 256, 128), torch.float32)], {"model_name": ["pt_gpt_neo_2_7B_causal_lm"]}),
    (Transpose1, [((1, 20, 256, 128), torch.float32)], {"model_name": ["pt_gpt_neo_2_7B_causal_lm"]}),
    (Transpose0, [((20, 128, 256), torch.float32)], {"model_name": ["pt_gpt_neo_2_7B_causal_lm"]}),
    (
        Transpose0,
        [((10240, 2560), torch.float32)],
        {
            "model_name": [
                "pt_gpt_neo_2_7B_causal_lm",
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
        Transpose0,
        [((2560, 10240), torch.float32)],
        {
            "model_name": [
                "pt_gpt_neo_2_7B_causal_lm",
                "pt_phi_2_pytdml_token_cls",
                "pt_phi_2_causal_lm",
                "pt_phi_2_seq_cls",
                "pt_phi_2_token_cls",
                "pt_phi_2_pytdml_seq_cls",
                "pt_phi_2_pytdml_causal_lm",
            ]
        },
    ),
    (Transpose0, [((50257, 2560), torch.float32)], {"model_name": ["pt_gpt_neo_2_7B_causal_lm"]}),
    (Transpose1, [((1, 256, 16, 128), torch.float32)], {"model_name": ["pt_gpt_neo_1_3B_causal_lm", "pt_xglm_1_7B"]}),
    (Transpose0, [((16, 256, 128), torch.float32)], {"model_name": ["pt_gpt_neo_1_3B_causal_lm", "pt_xglm_1_7B"]}),
    (Transpose0, [((1, 16, 256, 128), torch.float32)], {"model_name": ["pt_gpt_neo_1_3B_causal_lm"]}),
    (Transpose1, [((1, 16, 256, 128), torch.float32)], {"model_name": ["pt_gpt_neo_1_3B_causal_lm", "pt_xglm_1_7B"]}),
    (Transpose0, [((16, 128, 256), torch.float32)], {"model_name": ["pt_gpt_neo_1_3B_causal_lm", "pt_xglm_1_7B"]}),
    (Transpose0, [((50257, 2048), torch.float32)], {"model_name": ["pt_gpt_neo_1_3B_causal_lm"]}),
    (
        Transpose1,
        [((1, 256, 32, 128), torch.float32)],
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
        Transpose0,
        [((1, 64, 256), torch.float32)],
        {
            "model_name": [
                "pt_Llama_3_1_8B_Instruct_causal_lm",
                "pt_Meta_Llama_3_8B_causal_lm",
                "pt_Meta_Llama_3_8B_Instruct_causal_lm",
                "pt_Llama_3_1_8B_causal_lm",
                "pt_mit_b4",
                "pt_segformer_b0_finetuned_ade_512_512",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_mit_b0",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Transpose1,
        [((1, 256, 8, 128), torch.float32)],
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
        Transpose0,
        [((32, 256, 128), torch.float32)],
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
        Transpose0,
        [((1, 32, 256, 128), torch.float32)],
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
        Transpose1,
        [((1, 32, 256, 128), torch.float32)],
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
        Transpose0,
        [((32, 128, 256), torch.float32)],
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
        Transpose0,
        [((14336, 4096), torch.float32)],
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
        Transpose0,
        [((4096, 14336), torch.float32)],
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
        Transpose0,
        [((128256, 4096), torch.float32)],
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
        Transpose1,
        [((1, 256, 32, 64), torch.float32)],
        {"model_name": ["pt_Llama_3_2_1B_Instruct_causal_lm", "pt_Llama_3_2_1B_causal_lm", "pt_opt_1_3b_causal_lm"]},
    ),
    (
        Transpose0,
        [((1, 32, 256), torch.float32)],
        {
            "model_name": [
                "pt_Llama_3_2_1B_Instruct_causal_lm",
                "pt_Llama_3_2_1B_causal_lm",
                "pt_segformer_b0_finetuned_ade_512_512",
                "pt_mit_b0",
            ]
        },
    ),
    (
        Transpose1,
        [((1, 256, 8, 64), torch.float32)],
        {
            "model_name": [
                "pt_Llama_3_2_1B_Instruct_causal_lm",
                "pt_Llama_3_2_1B_causal_lm",
                "pt_mit_b4",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Transpose0,
        [((32, 256, 64), torch.float32)],
        {"model_name": ["pt_Llama_3_2_1B_Instruct_causal_lm", "pt_Llama_3_2_1B_causal_lm", "pt_opt_1_3b_causal_lm"]},
    ),
    (
        Transpose0,
        [((1, 32, 256, 64), torch.float32)],
        {"model_name": ["pt_Llama_3_2_1B_Instruct_causal_lm", "pt_Llama_3_2_1B_causal_lm"]},
    ),
    (
        Transpose1,
        [((1, 32, 256, 64), torch.float32)],
        {"model_name": ["pt_Llama_3_2_1B_Instruct_causal_lm", "pt_Llama_3_2_1B_causal_lm", "pt_opt_1_3b_causal_lm"]},
    ),
    (
        Transpose0,
        [((32, 64, 256), torch.float32)],
        {"model_name": ["pt_Llama_3_2_1B_Instruct_causal_lm", "pt_Llama_3_2_1B_causal_lm", "pt_opt_1_3b_causal_lm"]},
    ),
    (
        Transpose0,
        [((128256, 2048), torch.float32)],
        {"model_name": ["pt_Llama_3_2_1B_Instruct_causal_lm", "pt_Llama_3_2_1B_causal_lm"]},
    ),
    (
        Transpose1,
        [((1, 4, 32, 128), torch.float32)],
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
        Transpose0,
        [((1, 64, 4), torch.float32)],
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
        Transpose1,
        [((1, 4, 8, 128), torch.float32)],
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
        Transpose0,
        [((32, 4, 128), torch.float32)],
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
        Transpose0,
        [((1, 32, 4, 128), torch.float32)],
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
        Transpose1,
        [((1, 32, 4, 128), torch.float32)],
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
        Transpose0,
        [((32, 128, 4), torch.float32)],
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
        Transpose1,
        [((1, 4, 32, 64), torch.float32)],
        {"model_name": ["pt_Llama_3_2_1B_Instruct_seq_cls", "pt_Llama_3_2_1B_seq_cls"]},
    ),
    (
        Transpose0,
        [((1, 32, 4), torch.float32)],
        {"model_name": ["pt_Llama_3_2_1B_Instruct_seq_cls", "pt_Llama_3_2_1B_seq_cls"]},
    ),
    (
        Transpose1,
        [((1, 4, 8, 64), torch.float32)],
        {"model_name": ["pt_Llama_3_2_1B_Instruct_seq_cls", "pt_Llama_3_2_1B_seq_cls"]},
    ),
    (
        Transpose0,
        [((32, 4, 64), torch.float32)],
        {"model_name": ["pt_Llama_3_2_1B_Instruct_seq_cls", "pt_Llama_3_2_1B_seq_cls"]},
    ),
    (
        Transpose0,
        [((1, 32, 4, 64), torch.float32)],
        {"model_name": ["pt_Llama_3_2_1B_Instruct_seq_cls", "pt_Llama_3_2_1B_seq_cls"]},
    ),
    (
        Transpose1,
        [((1, 32, 4, 64), torch.float32)],
        {"model_name": ["pt_Llama_3_2_1B_Instruct_seq_cls", "pt_Llama_3_2_1B_seq_cls"]},
    ),
    (
        Transpose0,
        [((32, 64, 4), torch.float32)],
        {"model_name": ["pt_Llama_3_2_1B_Instruct_seq_cls", "pt_Llama_3_2_1B_seq_cls"]},
    ),
    (
        Transpose0,
        [((4096, 4096), torch.float32)],
        {
            "model_name": [
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_rcnn",
            ]
        },
    ),
    (Transpose1, [((1, 128, 32, 128), torch.float32)], {"model_name": ["pt_Mistral_7B_v0_1"]}),
    (Transpose0, [((1, 64, 128), torch.float32)], {"model_name": ["pt_Mistral_7B_v0_1"]}),
    (
        Transpose0,
        [((1024, 4096), torch.float32)],
        {
            "model_name": [
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
            ]
        },
    ),
    (Transpose1, [((1, 128, 8, 128), torch.float32)], {"model_name": ["pt_Mistral_7B_v0_1"]}),
    (Transpose0, [((32, 128, 128), torch.float32)], {"model_name": ["pt_Mistral_7B_v0_1"]}),
    (
        Transpose0,
        [((1, 32, 128, 128), torch.float32)],
        {"model_name": ["pt_Mistral_7B_v0_1", "pt_segformer_b0_finetuned_ade_512_512", "pt_mit_b0"]},
    ),
    (Transpose1, [((1, 32, 128, 128), torch.float32)], {"model_name": ["pt_Mistral_7B_v0_1"]}),
    (
        Transpose0,
        [((14336, 4096), torch.float32)],
        {
            "model_name": [
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
            ]
        },
    ),
    (
        Transpose0,
        [((4096, 14336), torch.float32)],
        {
            "model_name": [
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
                "pt_Mistral_7B_v0_1",
            ]
        },
    ),
    (Transpose0, [((32000, 4096), torch.float32)], {"model_name": ["pt_Mistral_7B_v0_1"]}),
    (Transpose1, [((1, 32, 12, 64), torch.float32)], {"model_name": ["pt_opt_125m_seq_cls", "pt_opt_125m_qa"]}),
    (Transpose0, [((12, 32, 64), torch.float32)], {"model_name": ["pt_opt_125m_seq_cls", "pt_opt_125m_qa"]}),
    (Transpose0, [((12, 64, 32), torch.float32)], {"model_name": ["pt_opt_125m_seq_cls", "pt_opt_125m_qa"]}),
    (Transpose1, [((1, 12, 32, 64), torch.float32)], {"model_name": ["pt_opt_125m_seq_cls", "pt_opt_125m_qa"]}),
    (Transpose1, [((1, 32, 32, 64), torch.float32)], {"model_name": ["pt_opt_1_3b_seq_cls", "pt_opt_1_3b_qa"]}),
    (Transpose0, [((32, 32, 64), torch.float32)], {"model_name": ["pt_opt_1_3b_seq_cls", "pt_opt_1_3b_qa"]}),
    (Transpose0, [((32, 64, 32), torch.float32)], {"model_name": ["pt_opt_1_3b_seq_cls", "pt_opt_1_3b_qa"]}),
    (Transpose0, [((1, 2048), torch.float32)], {"model_name": ["pt_opt_1_3b_qa"]}),
    (
        Transpose0,
        [((1024, 512), torch.float32)],
        {
            "model_name": [
                "pt_opt_350m_qa",
                "pt_opt_350m_causal_lm",
                "pt_opt_350m_seq_cls",
                "pt_google_flan_t5_small",
                "pt_vision_perceiver_learned",
            ]
        },
    ),
    (Transpose1, [((1, 32, 16, 64), torch.float32)], {"model_name": ["pt_opt_350m_qa", "pt_opt_350m_seq_cls"]}),
    (Transpose0, [((16, 32, 64), torch.float32)], {"model_name": ["pt_opt_350m_qa", "pt_opt_350m_seq_cls"]}),
    (Transpose0, [((16, 64, 32), torch.float32)], {"model_name": ["pt_opt_350m_qa", "pt_opt_350m_seq_cls"]}),
    (Transpose1, [((1, 16, 32, 64), torch.float32)], {"model_name": ["pt_opt_350m_qa", "pt_opt_350m_seq_cls"]}),
    (
        Transpose0,
        [((512, 1024), torch.float32)],
        {
            "model_name": [
                "pt_opt_350m_qa",
                "pt_opt_350m_causal_lm",
                "pt_opt_350m_seq_cls",
                "pt_google_flan_t5_small",
                "pt_vision_perceiver_learned",
            ]
        },
    ),
    (Transpose0, [((1, 512), torch.float32)], {"model_name": ["pt_opt_350m_qa"]}),
    (Transpose0, [((50272, 512), torch.float32)], {"model_name": ["pt_opt_350m_causal_lm"]}),
    (Transpose0, [((50272, 768), torch.float32)], {"model_name": ["pt_opt_125m_causal_lm"]}),
    (Transpose0, [((50272, 2048), torch.float32)], {"model_name": ["pt_opt_1_3b_causal_lm"]}),
    (Transpose0, [((2, 512), torch.float32)], {"model_name": ["pt_opt_350m_seq_cls"]}),
    (
        Transpose1,
        [((1, 12, 32, 80), torch.float32)],
        {"model_name": ["pt_phi_2_pytdml_token_cls", "pt_phi_2_token_cls"]},
    ),
    (Transpose0, [((1, 16, 12), torch.float32)], {"model_name": ["pt_phi_2_pytdml_token_cls", "pt_phi_2_token_cls"]}),
    (Transpose0, [((32, 12, 80), torch.float32)], {"model_name": ["pt_phi_2_pytdml_token_cls", "pt_phi_2_token_cls"]}),
    (
        Transpose0,
        [((1, 32, 12, 80), torch.float32)],
        {"model_name": ["pt_phi_2_pytdml_token_cls", "pt_phi_2_token_cls"]},
    ),
    (
        Transpose1,
        [((1, 32, 12, 80), torch.float32)],
        {"model_name": ["pt_phi_2_pytdml_token_cls", "pt_phi_2_token_cls"]},
    ),
    (Transpose0, [((32, 80, 12), torch.float32)], {"model_name": ["pt_phi_2_pytdml_token_cls", "pt_phi_2_token_cls"]}),
    (
        Transpose0,
        [((2, 2560), torch.float32)],
        {
            "model_name": [
                "pt_phi_2_pytdml_token_cls",
                "pt_phi_2_seq_cls",
                "pt_phi_2_token_cls",
                "pt_phi_2_pytdml_seq_cls",
            ]
        },
    ),
    (
        Transpose1,
        [((1, 256, 32, 80), torch.float32)],
        {"model_name": ["pt_phi_2_causal_lm", "pt_phi_2_pytdml_causal_lm"]},
    ),
    (Transpose0, [((1, 16, 256), torch.float32)], {"model_name": ["pt_phi_2_causal_lm", "pt_phi_2_pytdml_causal_lm"]}),
    (Transpose0, [((32, 256, 80), torch.float32)], {"model_name": ["pt_phi_2_causal_lm", "pt_phi_2_pytdml_causal_lm"]}),
    (
        Transpose0,
        [((1, 32, 256, 80), torch.float32)],
        {"model_name": ["pt_phi_2_causal_lm", "pt_phi_2_pytdml_causal_lm"]},
    ),
    (
        Transpose1,
        [((1, 32, 256, 80), torch.float32)],
        {"model_name": ["pt_phi_2_causal_lm", "pt_phi_2_pytdml_causal_lm"]},
    ),
    (Transpose0, [((32, 80, 256), torch.float32)], {"model_name": ["pt_phi_2_causal_lm", "pt_phi_2_pytdml_causal_lm"]}),
    (Transpose0, [((51200, 2560), torch.float32)], {"model_name": ["pt_phi_2_causal_lm", "pt_phi_2_pytdml_causal_lm"]}),
    (Transpose1, [((1, 11, 32, 80), torch.float32)], {"model_name": ["pt_phi_2_seq_cls", "pt_phi_2_pytdml_seq_cls"]}),
    (Transpose0, [((1, 16, 11), torch.float32)], {"model_name": ["pt_phi_2_seq_cls", "pt_phi_2_pytdml_seq_cls"]}),
    (Transpose0, [((32, 11, 80), torch.float32)], {"model_name": ["pt_phi_2_seq_cls", "pt_phi_2_pytdml_seq_cls"]}),
    (Transpose0, [((1, 32, 11, 80), torch.float32)], {"model_name": ["pt_phi_2_seq_cls", "pt_phi_2_pytdml_seq_cls"]}),
    (Transpose1, [((1, 32, 11, 80), torch.float32)], {"model_name": ["pt_phi_2_seq_cls", "pt_phi_2_pytdml_seq_cls"]}),
    (Transpose0, [((32, 80, 11), torch.float32)], {"model_name": ["pt_phi_2_seq_cls", "pt_phi_2_pytdml_seq_cls"]}),
    (Transpose1, [((1, 29, 16, 64), torch.float32)], {"model_name": ["pt_qwen_chat"]}),
    (Transpose0, [((1, 32, 29), torch.float32)], {"model_name": ["pt_qwen_chat", "pt_Qwen_Qwen2_5_0_5B"]}),
    (Transpose0, [((16, 29, 64), torch.float32)], {"model_name": ["pt_qwen_chat"]}),
    (Transpose0, [((1, 16, 29, 64), torch.float32)], {"model_name": ["pt_qwen_chat"]}),
    (Transpose1, [((1, 16, 29, 64), torch.float32)], {"model_name": ["pt_qwen_chat"]}),
    (Transpose0, [((16, 64, 29), torch.float32)], {"model_name": ["pt_qwen_chat"]}),
    (Transpose0, [((2816, 1024), torch.float32)], {"model_name": ["pt_qwen_chat", "pt_qwen_causal_lm"]}),
    (Transpose0, [((1024, 2816), torch.float32)], {"model_name": ["pt_qwen_chat", "pt_qwen_causal_lm"]}),
    (Transpose0, [((151936, 1024), torch.float32)], {"model_name": ["pt_qwen_chat", "pt_qwen_causal_lm"]}),
    (Transpose1, [((1, 6, 16, 64), torch.float32)], {"model_name": ["pt_qwen_causal_lm"]}),
    (Transpose0, [((16, 6, 64), torch.float32)], {"model_name": ["pt_qwen_causal_lm"]}),
    (Transpose0, [((1, 16, 6, 64), torch.float32)], {"model_name": ["pt_qwen_causal_lm"]}),
    (Transpose1, [((1, 16, 6, 64), torch.float32)], {"model_name": ["pt_qwen_causal_lm"]}),
    (Transpose0, [((16, 64, 6), torch.float32)], {"model_name": ["pt_qwen_causal_lm"]}),
    (
        Transpose1,
        [((1, 35, 12, 128), torch.float32)],
        {"model_name": ["pt_Qwen_Qwen2_5_Coder_1_5B_Instruct", "pt_Qwen_Qwen2_5_Coder_1_5B"]},
    ),
    (
        Transpose0,
        [((1, 64, 35), torch.float32)],
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
    (
        Transpose0,
        [((256, 1536), torch.float32)],
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
        Transpose1,
        [((1, 35, 2, 128), torch.float32)],
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
        Transpose0,
        [((12, 35, 128), torch.float32)],
        {"model_name": ["pt_Qwen_Qwen2_5_Coder_1_5B_Instruct", "pt_Qwen_Qwen2_5_Coder_1_5B"]},
    ),
    (
        Transpose0,
        [((1, 12, 35, 128), torch.float32)],
        {"model_name": ["pt_Qwen_Qwen2_5_Coder_1_5B_Instruct", "pt_Qwen_Qwen2_5_Coder_1_5B"]},
    ),
    (
        Transpose1,
        [((1, 12, 35, 128), torch.float32)],
        {"model_name": ["pt_Qwen_Qwen2_5_Coder_1_5B_Instruct", "pt_Qwen_Qwen2_5_Coder_1_5B"]},
    ),
    (
        Transpose0,
        [((12, 128, 35), torch.float32)],
        {"model_name": ["pt_Qwen_Qwen2_5_Coder_1_5B_Instruct", "pt_Qwen_Qwen2_5_Coder_1_5B"]},
    ),
    (
        Transpose0,
        [((8960, 1536), torch.float32)],
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
        Transpose0,
        [((1536, 8960), torch.float32)],
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
        Transpose0,
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
        Transpose1,
        [((1, 35, 16, 128), torch.float32)],
        {"model_name": ["pt_Qwen_Qwen2_5_Coder_3B", "pt_Qwen_Qwen2_5_Coder_3B_Instruct"]},
    ),
    (
        Transpose0,
        [((16, 35, 128), torch.float32)],
        {"model_name": ["pt_Qwen_Qwen2_5_Coder_3B", "pt_Qwen_Qwen2_5_Coder_3B_Instruct"]},
    ),
    (
        Transpose0,
        [((1, 16, 35, 128), torch.float32)],
        {"model_name": ["pt_Qwen_Qwen2_5_Coder_3B", "pt_Qwen_Qwen2_5_Coder_3B_Instruct"]},
    ),
    (
        Transpose1,
        [((1, 16, 35, 128), torch.float32)],
        {"model_name": ["pt_Qwen_Qwen2_5_Coder_3B", "pt_Qwen_Qwen2_5_Coder_3B_Instruct"]},
    ),
    (
        Transpose0,
        [((16, 128, 35), torch.float32)],
        {"model_name": ["pt_Qwen_Qwen2_5_Coder_3B", "pt_Qwen_Qwen2_5_Coder_3B_Instruct"]},
    ),
    (
        Transpose0,
        [((11008, 2048), torch.float32)],
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
        Transpose0,
        [((2048, 11008), torch.float32)],
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
        Transpose0,
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
        Transpose0,
        [((3584, 3584), torch.float32)],
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
        Transpose1,
        [((1, 35, 28, 128), torch.float32)],
        {"model_name": ["pt_Qwen_Qwen2_5_Coder_7B", "pt_Qwen_Qwen2_5_Coder_7B_Instruct"]},
    ),
    (
        Transpose0,
        [((512, 3584), torch.float32)],
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
        Transpose1,
        [((1, 35, 4, 128), torch.float32)],
        {"model_name": ["pt_Qwen_Qwen2_5_Coder_7B", "pt_Qwen_Qwen2_5_Coder_7B_Instruct"]},
    ),
    (
        Transpose0,
        [((28, 35, 128), torch.float32)],
        {"model_name": ["pt_Qwen_Qwen2_5_Coder_7B", "pt_Qwen_Qwen2_5_Coder_7B_Instruct"]},
    ),
    (
        Transpose0,
        [((1, 28, 35, 128), torch.float32)],
        {"model_name": ["pt_Qwen_Qwen2_5_Coder_7B", "pt_Qwen_Qwen2_5_Coder_7B_Instruct"]},
    ),
    (
        Transpose1,
        [((1, 28, 35, 128), torch.float32)],
        {"model_name": ["pt_Qwen_Qwen2_5_Coder_7B", "pt_Qwen_Qwen2_5_Coder_7B_Instruct"]},
    ),
    (
        Transpose0,
        [((28, 128, 35), torch.float32)],
        {"model_name": ["pt_Qwen_Qwen2_5_Coder_7B", "pt_Qwen_Qwen2_5_Coder_7B_Instruct"]},
    ),
    (
        Transpose0,
        [((18944, 3584), torch.float32)],
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
        Transpose0,
        [((3584, 18944), torch.float32)],
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
        Transpose0,
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
        Transpose0,
        [((896, 896), torch.float32)],
        {"model_name": ["pt_Qwen_Qwen2_5_Coder_0_5B", "pt_Qwen_Qwen2_5_0_5B_Instruct", "pt_Qwen_Qwen2_5_0_5B"]},
    ),
    (Transpose1, [((1, 35, 14, 64), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_Coder_0_5B"]}),
    (Transpose0, [((1, 32, 35), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_Coder_0_5B"]}),
    (
        Transpose0,
        [((128, 896), torch.float32)],
        {"model_name": ["pt_Qwen_Qwen2_5_Coder_0_5B", "pt_Qwen_Qwen2_5_0_5B_Instruct", "pt_Qwen_Qwen2_5_0_5B"]},
    ),
    (Transpose1, [((1, 35, 2, 64), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_Coder_0_5B"]}),
    (Transpose0, [((14, 35, 64), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_Coder_0_5B"]}),
    (Transpose0, [((1, 14, 35, 64), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_Coder_0_5B"]}),
    (Transpose1, [((1, 14, 35, 64), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_Coder_0_5B"]}),
    (Transpose0, [((14, 64, 35), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_Coder_0_5B"]}),
    (
        Transpose0,
        [((4864, 896), torch.float32)],
        {"model_name": ["pt_Qwen_Qwen2_5_Coder_0_5B", "pt_Qwen_Qwen2_5_0_5B_Instruct", "pt_Qwen_Qwen2_5_0_5B"]},
    ),
    (
        Transpose0,
        [((896, 4864), torch.float32)],
        {"model_name": ["pt_Qwen_Qwen2_5_Coder_0_5B", "pt_Qwen_Qwen2_5_0_5B_Instruct", "pt_Qwen_Qwen2_5_0_5B"]},
    ),
    (
        Transpose0,
        [((151936, 896), torch.float32)],
        {"model_name": ["pt_Qwen_Qwen2_5_Coder_0_5B", "pt_Qwen_Qwen2_5_0_5B_Instruct", "pt_Qwen_Qwen2_5_0_5B"]},
    ),
    (Transpose1, [((1, 39, 12, 128), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_1_5B_Instruct"]}),
    (
        Transpose0,
        [((1, 64, 39), torch.float32)],
        {"model_name": ["pt_Qwen_Qwen2_5_1_5B_Instruct", "pt_Qwen_Qwen2_5_3B_Instruct", "pt_Qwen_Qwen2_5_7B_Instruct"]},
    ),
    (
        Transpose1,
        [((1, 39, 2, 128), torch.float32)],
        {"model_name": ["pt_Qwen_Qwen2_5_1_5B_Instruct", "pt_Qwen_Qwen2_5_3B_Instruct"]},
    ),
    (Transpose0, [((12, 39, 128), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_1_5B_Instruct"]}),
    (Transpose0, [((1, 12, 39, 128), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_1_5B_Instruct"]}),
    (Transpose1, [((1, 12, 39, 128), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_1_5B_Instruct"]}),
    (Transpose0, [((12, 128, 39), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_1_5B_Instruct"]}),
    (Transpose1, [((1, 29, 12, 128), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_1_5B"]}),
    (
        Transpose0,
        [((1, 64, 29), torch.float32)],
        {"model_name": ["pt_Qwen_Qwen2_5_1_5B", "pt_Qwen_Qwen2_5_7B", "pt_Qwen_Qwen2_5_3B"]},
    ),
    (Transpose1, [((1, 29, 2, 128), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_1_5B", "pt_Qwen_Qwen2_5_3B"]}),
    (Transpose0, [((12, 29, 128), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_1_5B"]}),
    (Transpose0, [((1, 12, 29, 128), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_1_5B"]}),
    (Transpose1, [((1, 12, 29, 128), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_1_5B"]}),
    (Transpose0, [((12, 128, 29), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_1_5B"]}),
    (Transpose1, [((1, 29, 28, 128), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_7B"]}),
    (Transpose1, [((1, 29, 4, 128), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_7B"]}),
    (Transpose0, [((28, 29, 128), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_7B"]}),
    (Transpose0, [((1, 28, 29, 128), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_7B"]}),
    (Transpose1, [((1, 28, 29, 128), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_7B"]}),
    (Transpose0, [((28, 128, 29), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_7B"]}),
    (Transpose1, [((1, 39, 14, 64), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_0_5B_Instruct"]}),
    (Transpose0, [((1, 32, 39), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_0_5B_Instruct"]}),
    (Transpose1, [((1, 39, 2, 64), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_0_5B_Instruct"]}),
    (Transpose0, [((14, 39, 64), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_0_5B_Instruct"]}),
    (Transpose0, [((1, 14, 39, 64), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_0_5B_Instruct"]}),
    (Transpose1, [((1, 14, 39, 64), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_0_5B_Instruct"]}),
    (Transpose0, [((14, 64, 39), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_0_5B_Instruct"]}),
    (Transpose1, [((1, 29, 14, 64), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_0_5B"]}),
    (Transpose1, [((1, 29, 2, 64), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_0_5B"]}),
    (Transpose0, [((14, 29, 64), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_0_5B"]}),
    (Transpose0, [((1, 14, 29, 64), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_0_5B"]}),
    (Transpose1, [((1, 14, 29, 64), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_0_5B"]}),
    (Transpose0, [((14, 64, 29), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_0_5B"]}),
    (Transpose1, [((1, 39, 16, 128), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_3B_Instruct"]}),
    (Transpose0, [((16, 39, 128), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_3B_Instruct"]}),
    (Transpose0, [((1, 16, 39, 128), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_3B_Instruct"]}),
    (Transpose1, [((1, 16, 39, 128), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_3B_Instruct"]}),
    (Transpose0, [((16, 128, 39), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_3B_Instruct"]}),
    (Transpose1, [((1, 39, 28, 128), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_7B_Instruct"]}),
    (Transpose1, [((1, 39, 4, 128), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_7B_Instruct"]}),
    (Transpose0, [((28, 39, 128), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_7B_Instruct"]}),
    (Transpose0, [((1, 28, 39, 128), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_7B_Instruct"]}),
    (Transpose1, [((1, 28, 39, 128), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_7B_Instruct"]}),
    (Transpose0, [((28, 128, 39), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_7B_Instruct"]}),
    (Transpose1, [((1, 29, 16, 128), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_3B"]}),
    (Transpose0, [((16, 29, 128), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_3B"]}),
    (Transpose0, [((1, 16, 29, 128), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_3B"]}),
    (Transpose1, [((1, 16, 29, 128), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_3B"]}),
    (Transpose0, [((16, 128, 29), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_3B"]}),
    (Transpose0, [((250002, 768), torch.float32)], {"model_name": ["pt_roberta_masked_lm"]}),
    (Transpose0, [((3, 768), torch.float32)], {"model_name": ["pt_roberta_sentiment", "pt_squeezebert"]}),
    (Transpose0, [((1, 128, 768), torch.float32)], {"model_name": ["pt_squeezebert"]}),
    (Transpose0, [((1, 12, 64, 128), torch.float32)], {"model_name": ["pt_squeezebert"]}),
    (Transpose0, [((1, 768, 128), torch.float32)], {"model_name": ["pt_squeezebert"]}),
    (Transpose1, [((1, 1, 16, 64), torch.float32)], {"model_name": ["pt_t5_large"]}),
    (Transpose0, [((16, 1, 64), torch.float32)], {"model_name": ["pt_t5_large"]}),
    (Transpose2, [((1, 1, 16), torch.float32)], {"model_name": ["pt_t5_large"]}),
    (Transpose0, [((16, 1, 1), torch.float32)], {"model_name": ["pt_t5_large"]}),
    (Transpose0, [((1, 16, 1, 64), torch.float32)], {"model_name": ["pt_t5_large"]}),
    (Transpose1, [((1, 16, 1, 64), torch.float32)], {"model_name": ["pt_t5_large"]}),
    (Transpose0, [((16, 64, 1), torch.float32)], {"model_name": ["pt_t5_large"]}),
    (Transpose0, [((32128, 1024), torch.float32)], {"model_name": ["pt_t5_large"]}),
    (Transpose1, [((1, 1, 12, 64), torch.float32)], {"model_name": ["pt_t5_base", "pt_google_flan_t5_base"]}),
    (Transpose0, [((12, 1, 64), torch.float32)], {"model_name": ["pt_t5_base", "pt_google_flan_t5_base"]}),
    (Transpose2, [((1, 1, 12), torch.float32)], {"model_name": ["pt_t5_base", "pt_google_flan_t5_base"]}),
    (Transpose0, [((12, 1, 1), torch.float32)], {"model_name": ["pt_t5_base", "pt_google_flan_t5_base"]}),
    (Transpose0, [((1, 12, 1, 64), torch.float32)], {"model_name": ["pt_t5_base", "pt_google_flan_t5_base"]}),
    (Transpose1, [((1, 12, 1, 64), torch.float32)], {"model_name": ["pt_t5_base", "pt_google_flan_t5_base"]}),
    (Transpose0, [((12, 64, 1), torch.float32)], {"model_name": ["pt_t5_base", "pt_google_flan_t5_base"]}),
    (Transpose0, [((32128, 768), torch.float32)], {"model_name": ["pt_t5_base", "pt_google_flan_t5_base"]}),
    (Transpose0, [((768, 2048), torch.float32)], {"model_name": ["pt_google_flan_t5_base"]}),
    (Transpose1, [((1, 1, 8, 64), torch.float32)], {"model_name": ["pt_t5_small"]}),
    (Transpose0, [((8, 1, 64), torch.float32)], {"model_name": ["pt_t5_small"]}),
    (Transpose2, [((1, 1, 8), torch.float32)], {"model_name": ["pt_t5_small"]}),
    (Transpose0, [((8, 1, 1), torch.float32)], {"model_name": ["pt_t5_small"]}),
    (Transpose0, [((1, 8, 1, 64), torch.float32)], {"model_name": ["pt_t5_small"]}),
    (Transpose1, [((1, 8, 1, 64), torch.float32)], {"model_name": ["pt_t5_small"]}),
    (Transpose0, [((8, 64, 1), torch.float32)], {"model_name": ["pt_t5_small"]}),
    (Transpose0, [((32128, 512), torch.float32)], {"model_name": ["pt_t5_small", "pt_google_flan_t5_small"]}),
    (Transpose0, [((384, 512), torch.float32)], {"model_name": ["pt_google_flan_t5_small"]}),
    (Transpose1, [((1, 1, 6, 64), torch.float32)], {"model_name": ["pt_google_flan_t5_small"]}),
    (Transpose0, [((6, 1, 64), torch.float32)], {"model_name": ["pt_google_flan_t5_small"]}),
    (Transpose2, [((1, 1, 6), torch.float32)], {"model_name": ["pt_google_flan_t5_small"]}),
    (Transpose0, [((6, 1, 1), torch.float32)], {"model_name": ["pt_google_flan_t5_small"]}),
    (Transpose0, [((6, 64, 1), torch.float32)], {"model_name": ["pt_google_flan_t5_small"]}),
    (Transpose0, [((512, 384), torch.float32)], {"model_name": ["pt_google_flan_t5_small"]}),
    (Transpose0, [((256008, 2048), torch.float32)], {"model_name": ["pt_xglm_1_7B"]}),
    (Transpose0, [((256008, 1024), torch.float32)], {"model_name": ["pt_xglm_564M"]}),
    (
        Transpose3,
        [((1, 1, 1024, 72), torch.float32)],
        {"model_name": ["nbeats_generic", "nbeats_trend", "nbeats_seasonality"]},
    ),
    (Transpose0, [((512, 72), torch.float32)], {"model_name": ["nbeats_generic"]}),
    (Transpose0, [((96, 512), torch.float32)], {"model_name": ["nbeats_generic"]}),
    (Transpose0, [((256, 72), torch.float32)], {"model_name": ["nbeats_trend"]}),
    (
        Transpose0,
        [((256, 256), torch.float32)],
        {
            "model_name": [
                "nbeats_trend",
                "pt_vision_perceiver_learned",
                "pt_segformer_b0_finetuned_ade_512_512",
                "pt_mit_b0",
            ]
        },
    ),
    (Transpose0, [((8, 256), torch.float32)], {"model_name": ["nbeats_trend"]}),
    (Transpose0, [((2048, 72), torch.float32)], {"model_name": ["nbeats_seasonality"]}),
    (Transpose0, [((48, 2048), torch.float32)], {"model_name": ["nbeats_seasonality"]}),
    (Transpose0, [((4096, 9216), torch.float32)], {"model_name": ["pt_alexnet_torchhub"]}),
    (
        Transpose0,
        [((1000, 4096), torch.float32)],
        {
            "model_name": [
                "pt_alexnet_torchhub",
                "pt_vgg13_osmr",
                "pt_bn_vgg19_osmr",
                "pt_bn_vgg19b_osmr",
                "pt_vgg16_osmr",
                "pt_vgg19_osmr",
                "pt_vgg_19_hf",
                "pt_vgg_bn19_torchhub",
                "pt_vgg11_osmr",
                "pt_vgg19_bn_timm",
            ]
        },
    ),
    (Transpose0, [((128, 784), torch.float32)], {"model_name": ["pt_linear_ae"]}),
    (Transpose0, [((64, 128), torch.float32)], {"model_name": ["pt_linear_ae"]}),
    (Transpose0, [((12, 64), torch.float32)], {"model_name": ["pt_linear_ae"]}),
    (Transpose0, [((3, 12), torch.float32)], {"model_name": ["pt_linear_ae"]}),
    (Transpose0, [((12, 3), torch.float32)], {"model_name": ["pt_linear_ae"]}),
    (Transpose0, [((64, 12), torch.float32)], {"model_name": ["pt_linear_ae"]}),
    (Transpose0, [((128, 64), torch.float32)], {"model_name": ["pt_linear_ae"]}),
    (Transpose0, [((784, 128), torch.float32)], {"model_name": ["pt_linear_ae"]}),
    (Transpose0, [((1, 384, 196), torch.float32)], {"model_name": ["pt_deit_small_patch16_224"]}),
    (Transpose1, [((1, 197, 6, 64), torch.float32)], {"model_name": ["pt_deit_small_patch16_224"]}),
    (Transpose0, [((6, 197, 64), torch.float32)], {"model_name": ["pt_deit_small_patch16_224"]}),
    (Transpose0, [((1, 6, 197, 64), torch.float32)], {"model_name": ["pt_deit_small_patch16_224"]}),
    (Transpose1, [((1, 6, 197, 64), torch.float32)], {"model_name": ["pt_deit_small_patch16_224"]}),
    (Transpose0, [((6, 64, 197), torch.float32)], {"model_name": ["pt_deit_small_patch16_224"]}),
    (Transpose0, [((1000, 384), torch.float32)], {"model_name": ["pt_deit_small_patch16_224"]}),
    (
        Transpose0,
        [((1, 768, 196), torch.float32)],
        {
            "model_name": [
                "pt_deit_base_patch16_224",
                "pt_deit_base_distilled_patch16_224",
                "pt_mixer_b16_224_miil_in21k",
                "pt_mixer_b16_224",
                "pt_mixer_b16_224_in21k",
                "pt_mixer_b16_224_miil",
                "pt_vit_base_patch16_224",
            ]
        },
    ),
    (
        Transpose1,
        [((1, 197, 12, 64), torch.float32)],
        {"model_name": ["pt_deit_base_patch16_224", "pt_deit_base_distilled_patch16_224", "pt_vit_base_patch16_224"]},
    ),
    (
        Transpose0,
        [((12, 197, 64), torch.float32)],
        {"model_name": ["pt_deit_base_patch16_224", "pt_deit_base_distilled_patch16_224", "pt_vit_base_patch16_224"]},
    ),
    (
        Transpose0,
        [((1, 12, 197, 64), torch.float32)],
        {"model_name": ["pt_deit_base_patch16_224", "pt_deit_base_distilled_patch16_224", "pt_vit_base_patch16_224"]},
    ),
    (
        Transpose1,
        [((1, 12, 197, 64), torch.float32)],
        {"model_name": ["pt_deit_base_patch16_224", "pt_deit_base_distilled_patch16_224", "pt_vit_base_patch16_224"]},
    ),
    (
        Transpose0,
        [((12, 64, 197), torch.float32)],
        {"model_name": ["pt_deit_base_patch16_224", "pt_deit_base_distilled_patch16_224", "pt_vit_base_patch16_224"]},
    ),
    (
        Transpose0,
        [((1000, 768), torch.float32)],
        {
            "model_name": [
                "pt_deit_base_patch16_224",
                "pt_deit_base_distilled_patch16_224",
                "pt_mixer_b16_224",
                "pt_mixer_b32_224",
                "pt_mixer_b16_224_miil",
                "pt_vit_base_patch16_224",
            ]
        },
    ),
    (Transpose0, [((1, 192, 196), torch.float32)], {"model_name": ["pt_deit_tiny_patch16_224"]}),
    (
        Transpose0,
        [((192, 192), torch.float32)],
        {"model_name": ["pt_deit_tiny_patch16_224", "pt_swinv2_tiny_patch4_window8_256"]},
    ),
    (Transpose1, [((1, 197, 3, 64), torch.float32)], {"model_name": ["pt_deit_tiny_patch16_224"]}),
    (Transpose0, [((3, 197, 64), torch.float32)], {"model_name": ["pt_deit_tiny_patch16_224"]}),
    (Transpose0, [((1, 3, 197, 64), torch.float32)], {"model_name": ["pt_deit_tiny_patch16_224"]}),
    (Transpose1, [((1, 3, 197, 64), torch.float32)], {"model_name": ["pt_deit_tiny_patch16_224"]}),
    (Transpose0, [((3, 64, 197), torch.float32)], {"model_name": ["pt_deit_tiny_patch16_224"]}),
    (
        Transpose0,
        [((768, 192), torch.float32)],
        {"model_name": ["pt_deit_tiny_patch16_224", "pt_swinv2_tiny_patch4_window8_256"]},
    ),
    (
        Transpose0,
        [((192, 768), torch.float32)],
        {"model_name": ["pt_deit_tiny_patch16_224", "pt_swinv2_tiny_patch4_window8_256"]},
    ),
    (Transpose0, [((1000, 192), torch.float32)], {"model_name": ["pt_deit_tiny_patch16_224"]}),
    (
        Transpose0,
        [((1000, 1024), torch.float32)],
        {
            "model_name": [
                "pt_densenet121",
                "pt_googlenet",
                "pt_mixer_l32_224",
                "pt_mixer_l16_224",
                "pt_mobilenet_v3_small",
                "pt_mobilenetv3_small_100",
                "pt_vision_perceiver_learned",
                "pt_vision_perceiver_conv",
                "pt_vision_perceiver_fourier",
                "pt_vit_large_patch16_224",
                "pt_ese_vovnet19b_dw",
                "pt_ese_vovnet39b",
                "pt_vovnet39",
                "vovnet_57_stigma_pt",
                "pt_ese_vovnet99b",
                "pt_vovnet_39_stigma",
                "pt_vovnet57",
            ]
        },
    ),
    (Transpose0, [((1000, 2208), torch.float32)], {"model_name": ["pt_densenet_161"]}),
    (Transpose0, [((1000, 1664), torch.float32)], {"model_name": ["pt_densenet_169"]}),
    (Transpose0, [((1000, 1920), torch.float32)], {"model_name": ["pt_densenet_201"]}),
    (
        Transpose0,
        [((1000, 1792), torch.float32)],
        {"model_name": ["pt_efficientnet_b4_timm", "pt_efficientnet_b4_torchvision"]},
    ),
    (
        Transpose0,
        [((1000, 1280), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_b0_torchvision",
                "pt_efficientnet_b0_timm",
                "pt_ghostnet_100",
                "mobilenetv2_basic",
                "mobilenetv2_timm",
                "pt_mobilenet_v3_large",
                "pt_mobilenetv3_large_100",
            ]
        },
    ),
    (
        Transpose0,
        [((1000, 2048), torch.float32)],
        {
            "model_name": [
                "pt_hrnet_timm_hrnet_w18",
                "pt_hrnet_osmr_hrnet_w18_small_v2",
                "pt_hrnet_osmr_hrnetv2_w64",
                "pt_hrnet_timm_hrnet_w30",
                "pt_hrnet_timm_hrnet_w32",
                "pt_hrnet_osmr_hrnetv2_w40",
                "pt_hrnet_timm_hrnet_w48",
                "pt_hrnet_osmr_hrnetv2_w18",
                "pt_hrnet_osmr_hrnetv2_w32",
                "pt_hrnet_timm_hrnet_w40",
                "pt_hrnet_osmr_hrnetv2_w30",
                "pt_hrnet_timm_hrnet_w44",
                "pt_hrnet_timm_hrnet_w18_small",
                "pt_hrnet_timm_hrnet_w64",
                "pt_hrnet_timm_hrnet_w18_small_v2",
                "pt_hrnet_osmr_hrnetv2_w44",
                "pt_hrnet_osmr_hrnetv2_w48",
                "pt_hrnet_osmr_hrnet_w18_small_v1",
                "pt_resnet50_timm",
                "pt_resnet50",
                "pt_resnext50_torchhub",
                "pt_resnext101_torchhub",
                "pt_resnext14_osmr",
                "pt_resnext26_osmr",
                "pt_resnext101_fb_wsl",
                "pt_resnext101_osmr",
                "pt_resnext50_osmr",
                "pt_wide_resnet101_2_timm",
                "pt_wide_resnet50_2_hub",
                "pt_wide_resnet50_2_timm",
                "pt_wide_resnet101_2_hub",
                "pt_xception65_timm",
                "pt_xception71_timm",
                "pt_xception41_timm",
                "pt_xception_timm",
            ]
        },
    ),
    (Transpose0, [((1000, 1536), torch.float32)], {"model_name": ["pt_timm_inception_v4", "pt_osmr_inception_v4"]}),
    (Transpose0, [((1, 1024, 49), torch.float32)], {"model_name": ["pt_mixer_l32_224"]}),
    (Transpose0, [((1, 49, 1024), torch.float32)], {"model_name": ["pt_mixer_l32_224"]}),
    (Transpose0, [((512, 49), torch.float32)], {"model_name": ["pt_mixer_l32_224"]}),
    (Transpose0, [((49, 512), torch.float32)], {"model_name": ["pt_mixer_l32_224"]}),
    (
        Transpose0,
        [((1, 1024, 196), torch.float32)],
        {"model_name": ["pt_mixer_l16_224", "pt_mixer_l16_224_in21k", "pt_vit_large_patch16_224"]},
    ),
    (Transpose0, [((1, 196, 1024), torch.float32)], {"model_name": ["pt_mixer_l16_224", "pt_mixer_l16_224_in21k"]}),
    (Transpose0, [((512, 196), torch.float32)], {"model_name": ["pt_mixer_l16_224", "pt_mixer_l16_224_in21k"]}),
    (Transpose0, [((196, 512), torch.float32)], {"model_name": ["pt_mixer_l16_224", "pt_mixer_l16_224_in21k"]}),
    (Transpose0, [((1, 512, 49), torch.float32)], {"model_name": ["pt_mixer_s32_224"]}),
    (Transpose0, [((1, 49, 512), torch.float32)], {"model_name": ["pt_mixer_s32_224"]}),
    (Transpose0, [((256, 49), torch.float32)], {"model_name": ["pt_mixer_s32_224"]}),
    (Transpose0, [((49, 256), torch.float32)], {"model_name": ["pt_mixer_s32_224"]}),
    (
        Transpose0,
        [((1000, 512), torch.float32)],
        {
            "model_name": [
                "pt_mixer_s32_224",
                "pt_mixer_s16_224",
                "pt_mit_b4",
                "pt_mit_b1",
                "pt_mit_b2",
                "pt_mit_b3",
                "pt_mit_b5",
                "pt_vovnet27s",
            ]
        },
    ),
    (
        Transpose0,
        [((1, 196, 768), torch.float32)],
        {
            "model_name": [
                "pt_mixer_b16_224_miil_in21k",
                "pt_mixer_b16_224",
                "pt_mixer_b16_224_in21k",
                "pt_mixer_b16_224_miil",
            ]
        },
    ),
    (
        Transpose0,
        [((384, 196), torch.float32)],
        {
            "model_name": [
                "pt_mixer_b16_224_miil_in21k",
                "pt_mixer_b16_224",
                "pt_mixer_b16_224_in21k",
                "pt_mixer_b16_224_miil",
            ]
        },
    ),
    (
        Transpose0,
        [((196, 384), torch.float32)],
        {
            "model_name": [
                "pt_mixer_b16_224_miil_in21k",
                "pt_mixer_b16_224",
                "pt_mixer_b16_224_in21k",
                "pt_mixer_b16_224_miil",
            ]
        },
    ),
    (Transpose0, [((11221, 768), torch.float32)], {"model_name": ["pt_mixer_b16_224_miil_in21k"]}),
    (Transpose0, [((1, 512, 196), torch.float32)], {"model_name": ["pt_mixer_s16_224"]}),
    (Transpose0, [((1, 196, 512), torch.float32)], {"model_name": ["pt_mixer_s16_224"]}),
    (Transpose0, [((256, 196), torch.float32)], {"model_name": ["pt_mixer_s16_224"]}),
    (Transpose0, [((196, 256), torch.float32)], {"model_name": ["pt_mixer_s16_224"]}),
    (Transpose0, [((21843, 1024), torch.float32)], {"model_name": ["pt_mixer_l16_224_in21k"]}),
    (Transpose0, [((1, 768, 49), torch.float32)], {"model_name": ["pt_mixer_b32_224"]}),
    (Transpose0, [((1, 49, 768), torch.float32)], {"model_name": ["pt_mixer_b32_224"]}),
    (Transpose0, [((384, 49), torch.float32)], {"model_name": ["pt_mixer_b32_224"]}),
    (Transpose0, [((49, 384), torch.float32)], {"model_name": ["pt_mixer_b32_224"]}),
    (Transpose0, [((21843, 768), torch.float32)], {"model_name": ["pt_mixer_b16_224_in21k"]}),
    (Transpose0, [((1001, 1024), torch.float32)], {"model_name": ["pt_mobilenet_v1_224"]}),
    (Transpose0, [((1001, 768), torch.float32)], {"model_name": ["pt_mobilenet_v1_192"]}),
    (
        Transpose0,
        [((1001, 1280), torch.float32)],
        {"model_name": ["mobilenetv2_160", "mobilenetv2_96", "mobilenetv2_224"]},
    ),
    (Transpose0, [((1024, 576), torch.float32)], {"model_name": ["pt_mobilenet_v3_small"]}),
    (Transpose0, [((1280, 960), torch.float32)], {"model_name": ["pt_mobilenet_v3_large"]}),
    (Transpose1, [((1, 256, 224, 224), torch.float32)], {"model_name": ["pt_vision_perceiver_learned"]}),
    (Transpose0, [((1, 224, 256, 224), torch.float32)], {"model_name": ["pt_vision_perceiver_learned"]}),
    (Transpose0, [((1, 50176, 512), torch.float32)], {"model_name": ["pt_vision_perceiver_learned"]}),
    (Transpose1, [((1, 50176, 1, 512), torch.float32)], {"model_name": ["pt_vision_perceiver_learned"]}),
    (Transpose0, [((1, 1, 50176, 512), torch.float32)], {"model_name": ["pt_vision_perceiver_learned"]}),
    (Transpose0, [((1, 512, 50176), torch.float32)], {"model_name": ["pt_vision_perceiver_learned"]}),
    (
        Transpose1,
        [((1, 512, 8, 128), torch.float32)],
        {"model_name": ["pt_vision_perceiver_learned", "pt_vision_perceiver_conv", "pt_vision_perceiver_fourier"]},
    ),
    (
        Transpose0,
        [((8, 512, 128), torch.float32)],
        {"model_name": ["pt_vision_perceiver_learned", "pt_vision_perceiver_conv", "pt_vision_perceiver_fourier"]},
    ),
    (
        Transpose0,
        [((1, 8, 512, 128), torch.float32)],
        {"model_name": ["pt_vision_perceiver_learned", "pt_vision_perceiver_conv", "pt_vision_perceiver_fourier"]},
    ),
    (
        Transpose1,
        [((1, 8, 512, 128), torch.float32)],
        {"model_name": ["pt_vision_perceiver_learned", "pt_vision_perceiver_conv", "pt_vision_perceiver_fourier"]},
    ),
    (
        Transpose0,
        [((8, 128, 512), torch.float32)],
        {"model_name": ["pt_vision_perceiver_learned", "pt_vision_perceiver_conv", "pt_vision_perceiver_fourier"]},
    ),
    (
        Transpose0,
        [((1, 512, 1024), torch.float32)],
        {"model_name": ["pt_vision_perceiver_learned", "pt_vision_perceiver_conv", "pt_vision_perceiver_fourier"]},
    ),
    (
        Transpose1,
        [((1, 512, 1, 1024), torch.float32)],
        {"model_name": ["pt_vision_perceiver_learned", "pt_vision_perceiver_conv", "pt_vision_perceiver_fourier"]},
    ),
    (
        Transpose0,
        [((1, 1, 512, 1024), torch.float32)],
        {"model_name": ["pt_vision_perceiver_learned", "pt_vision_perceiver_conv", "pt_vision_perceiver_fourier"]},
    ),
    (
        Transpose0,
        [((1, 1024, 512), torch.float32)],
        {"model_name": ["pt_vision_perceiver_learned", "pt_vision_perceiver_conv", "pt_vision_perceiver_fourier"]},
    ),
    (Transpose0, [((322, 1024), torch.float32)], {"model_name": ["pt_vision_perceiver_conv"]}),
    (Transpose1, [((1, 64, 55, 55), torch.float32)], {"model_name": ["pt_vision_perceiver_conv"]}),
    (Transpose0, [((1, 55, 64, 55), torch.float32)], {"model_name": ["pt_vision_perceiver_conv"]}),
    (Transpose0, [((322, 322), torch.float32)], {"model_name": ["pt_vision_perceiver_conv"]}),
    (Transpose0, [((1, 3025, 322), torch.float32)], {"model_name": ["pt_vision_perceiver_conv"]}),
    (Transpose1, [((1, 3025, 1, 322), torch.float32)], {"model_name": ["pt_vision_perceiver_conv"]}),
    (Transpose0, [((1, 1, 3025, 322), torch.float32)], {"model_name": ["pt_vision_perceiver_conv"]}),
    (Transpose0, [((1, 322, 3025), torch.float32)], {"model_name": ["pt_vision_perceiver_conv"]}),
    (Transpose0, [((1024, 322), torch.float32)], {"model_name": ["pt_vision_perceiver_conv"]}),
    (Transpose0, [((261, 1024), torch.float32)], {"model_name": ["pt_vision_perceiver_fourier"]}),
    (Transpose1, [((1, 3, 224, 224), torch.float32)], {"model_name": ["pt_vision_perceiver_fourier"]}),
    (Transpose0, [((1, 224, 3, 224), torch.float32)], {"model_name": ["pt_vision_perceiver_fourier"]}),
    (Transpose0, [((261, 261), torch.float32)], {"model_name": ["pt_vision_perceiver_fourier"]}),
    (Transpose0, [((1, 50176, 261), torch.float32)], {"model_name": ["pt_vision_perceiver_fourier"]}),
    (Transpose1, [((1, 50176, 1, 261), torch.float32)], {"model_name": ["pt_vision_perceiver_fourier"]}),
    (Transpose0, [((1, 1, 50176, 261), torch.float32)], {"model_name": ["pt_vision_perceiver_fourier"]}),
    (Transpose0, [((1, 261, 50176), torch.float32)], {"model_name": ["pt_vision_perceiver_fourier"]}),
    (Transpose0, [((1024, 261), torch.float32)], {"model_name": ["pt_vision_perceiver_fourier"]}),
    (Transpose0, [((4096, 9216), torch.float32)], {"model_name": ["pt_rcnn"]}),
    (Transpose0, [((2, 4096), torch.float32)], {"model_name": ["pt_rcnn"]}),
    (Transpose0, [((1000, 1088), torch.float32)], {"model_name": ["pt_regnet_y_040"]}),
    (
        Transpose0,
        [((1, 64, 16384), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Transpose0,
        [((64, 64), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_segformer_b0_finetuned_ade_512_512",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_mit_b0",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Transpose0,
        [((1, 16384, 64), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Transpose0,
        [((1, 256, 64), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Transpose1,
        [((1, 256, 1, 64), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Transpose0,
        [((1, 1, 256, 64), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Transpose0,
        [((256, 64), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_segformer_b0_finetuned_ade_512_512",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_mit_b0",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Transpose0,
        [((1, 16384, 256), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_segformer_b0_finetuned_ade_512_512",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Transpose0,
        [((1, 256, 16384), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Transpose0,
        [((64, 256), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_segformer_b0_finetuned_ade_512_512",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_mit_b0",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Transpose2,
        [((1, 128, 128, 64), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Transpose0,
        [((1, 64, 128, 128), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Transpose0,
        [((1, 128, 4096), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Transpose0,
        [((128, 128), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Transpose1,
        [((1, 4096, 2, 64), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Transpose0,
        [((1, 4096, 128), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Transpose0,
        [((1, 128, 256), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Transpose1,
        [((1, 256, 2, 64), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Transpose0,
        [((2, 256, 64), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Transpose0,
        [((1, 2, 256, 64), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Transpose0,
        [((2, 64, 256), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Transpose1,
        [((1, 2, 4096, 64), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Transpose0,
        [((512, 128), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Transpose0,
        [((1, 4096, 512), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Transpose0,
        [((1, 512, 4096), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Transpose0,
        [((128, 512), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Transpose2,
        [((1, 64, 64, 128), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Transpose0,
        [((1, 320, 1024), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Transpose0,
        [((320, 320), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Transpose1,
        [((1, 1024, 5, 64), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Transpose0,
        [((1, 1024, 320), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Transpose0,
        [((1, 320, 256), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Transpose1,
        [((1, 256, 5, 64), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Transpose0,
        [((5, 256, 64), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Transpose0,
        [((1, 5, 256, 64), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Transpose0,
        [((5, 64, 256), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Transpose1,
        [((1, 5, 1024, 64), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Transpose0,
        [((1280, 320), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Transpose0,
        [((1, 1024, 1280), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Transpose0,
        [((1, 1280, 1024), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Transpose0,
        [((320, 1280), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Transpose2,
        [((1, 32, 32, 320), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Transpose0,
        [((1, 320, 32, 32), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Transpose0,
        [((1, 512, 256), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Transpose0,
        [((8, 256, 64), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Transpose0,
        [((1, 8, 256, 64), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Transpose1,
        [((1, 8, 256, 64), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Transpose0,
        [((8, 64, 256), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Transpose0,
        [((1, 256, 2048), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Transpose0,
        [((1, 2048, 256), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Transpose0,
        [((1, 32, 16384), torch.float32)],
        {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_mit_b0"]},
    ),
    (Transpose0, [((32, 32), torch.float32)], {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_mit_b0"]}),
    (
        Transpose0,
        [((1, 16384, 32), torch.float32)],
        {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_mit_b0"]},
    ),
    (
        Transpose0,
        [((1, 256, 32), torch.float32)],
        {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_mit_b0"]},
    ),
    (
        Transpose1,
        [((1, 256, 1, 32), torch.float32)],
        {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_mit_b0"]},
    ),
    (
        Transpose0,
        [((1, 1, 256, 32), torch.float32)],
        {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_mit_b0"]},
    ),
    (Transpose0, [((128, 32), torch.float32)], {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_mit_b0"]}),
    (
        Transpose0,
        [((1, 16384, 128), torch.float32)],
        {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_mit_b0"]},
    ),
    (
        Transpose0,
        [((1, 128, 16384), torch.float32)],
        {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_mit_b0"]},
    ),
    (Transpose0, [((32, 128), torch.float32)], {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_mit_b0"]}),
    (
        Transpose2,
        [((1, 128, 128, 32), torch.float32)],
        {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_mit_b0"]},
    ),
    (
        Transpose0,
        [((1, 64, 4096), torch.float32)],
        {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_mit_b0"]},
    ),
    (
        Transpose1,
        [((1, 4096, 2, 32), torch.float32)],
        {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_mit_b0"]},
    ),
    (
        Transpose0,
        [((1, 4096, 64), torch.float32)],
        {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_mit_b0"]},
    ),
    (
        Transpose1,
        [((1, 256, 2, 32), torch.float32)],
        {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_mit_b0"]},
    ),
    (
        Transpose0,
        [((2, 256, 32), torch.float32)],
        {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_mit_b0"]},
    ),
    (
        Transpose0,
        [((1, 2, 256, 32), torch.float32)],
        {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_mit_b0"]},
    ),
    (
        Transpose0,
        [((2, 32, 256), torch.float32)],
        {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_mit_b0"]},
    ),
    (
        Transpose1,
        [((1, 2, 4096, 32), torch.float32)],
        {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_mit_b0"]},
    ),
    (
        Transpose0,
        [((1, 4096, 256), torch.float32)],
        {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_mit_b0", "pt_segformer_b1_finetuned_ade_512_512"]},
    ),
    (
        Transpose0,
        [((1, 256, 4096), torch.float32)],
        {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_mit_b0"]},
    ),
    (
        Transpose2,
        [((1, 64, 64, 64), torch.float32)],
        {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_mit_b0"]},
    ),
    (
        Transpose0,
        [((1, 64, 64, 64), torch.float32)],
        {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_mit_b0"]},
    ),
    (
        Transpose0,
        [((1, 160, 1024), torch.float32)],
        {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_mit_b0"]},
    ),
    (Transpose0, [((160, 160), torch.float32)], {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_mit_b0"]}),
    (
        Transpose1,
        [((1, 1024, 5, 32), torch.float32)],
        {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_mit_b0"]},
    ),
    (
        Transpose0,
        [((1, 1024, 160), torch.float32)],
        {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_mit_b0"]},
    ),
    (
        Transpose0,
        [((1, 160, 256), torch.float32)],
        {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_mit_b0"]},
    ),
    (
        Transpose1,
        [((1, 256, 5, 32), torch.float32)],
        {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_mit_b0"]},
    ),
    (
        Transpose0,
        [((5, 256, 32), torch.float32)],
        {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_mit_b0"]},
    ),
    (
        Transpose0,
        [((1, 5, 256, 32), torch.float32)],
        {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_mit_b0"]},
    ),
    (
        Transpose0,
        [((5, 32, 256), torch.float32)],
        {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_mit_b0"]},
    ),
    (
        Transpose1,
        [((1, 5, 1024, 32), torch.float32)],
        {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_mit_b0"]},
    ),
    (Transpose0, [((640, 160), torch.float32)], {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_mit_b0"]}),
    (
        Transpose0,
        [((1, 1024, 640), torch.float32)],
        {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_mit_b0"]},
    ),
    (
        Transpose0,
        [((1, 640, 1024), torch.float32)],
        {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_mit_b0"]},
    ),
    (Transpose0, [((160, 640), torch.float32)], {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_mit_b0"]}),
    (
        Transpose2,
        [((1, 32, 32, 160), torch.float32)],
        {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_mit_b0"]},
    ),
    (
        Transpose0,
        [((1, 160, 32, 32), torch.float32)],
        {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_mit_b0"]},
    ),
    (
        Transpose0,
        [((1, 256, 256), torch.float32)],
        {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_mit_b0", "pt_segformer_b1_finetuned_ade_512_512"]},
    ),
    (
        Transpose1,
        [((1, 256, 8, 32), torch.float32)],
        {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_mit_b0"]},
    ),
    (
        Transpose0,
        [((8, 256, 32), torch.float32)],
        {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_mit_b0"]},
    ),
    (
        Transpose0,
        [((1, 8, 256, 32), torch.float32)],
        {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_mit_b0"]},
    ),
    (
        Transpose1,
        [((1, 8, 256, 32), torch.float32)],
        {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_mit_b0"]},
    ),
    (
        Transpose0,
        [((8, 32, 256), torch.float32)],
        {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_mit_b0"]},
    ),
    (
        Transpose0,
        [((1024, 256), torch.float32)],
        {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_mit_b0"]},
    ),
    (
        Transpose0,
        [((1, 256, 1024), torch.float32)],
        {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_mit_b0"]},
    ),
    (
        Transpose0,
        [((1, 1024, 256), torch.float32)],
        {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_mit_b0", "pt_segformer_b1_finetuned_ade_512_512"]},
    ),
    (
        Transpose0,
        [((256, 1024), torch.float32)],
        {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_mit_b0"]},
    ),
    (Transpose2, [((1, 16, 16, 256), torch.float32)], {"model_name": ["pt_segformer_b0_finetuned_ade_512_512"]}),
    (Transpose0, [((1, 256, 16, 16), torch.float32)], {"model_name": ["pt_segformer_b0_finetuned_ade_512_512"]}),
    (Transpose0, [((256, 160), torch.float32)], {"model_name": ["pt_segformer_b0_finetuned_ade_512_512"]}),
    (Transpose0, [((256, 32), torch.float32)], {"model_name": ["pt_segformer_b0_finetuned_ade_512_512"]}),
    (
        Transpose2,
        [((1, 16, 16, 512), torch.float32)],
        {
            "model_name": [
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
            ]
        },
    ),
    (
        Transpose0,
        [((1, 512, 16, 16), torch.float32)],
        {
            "model_name": [
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
            ]
        },
    ),
    (
        Transpose0,
        [((768, 512), torch.float32)],
        {
            "model_name": [
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
            ]
        },
    ),
    (
        Transpose0,
        [((1, 256, 768), torch.float32)],
        {
            "model_name": [
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
            ]
        },
    ),
    (
        Transpose0,
        [((768, 320), torch.float32)],
        {
            "model_name": [
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
            ]
        },
    ),
    (
        Transpose0,
        [((1, 1024, 768), torch.float32)],
        {
            "model_name": [
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
            ]
        },
    ),
    (
        Transpose0,
        [((1, 4096, 768), torch.float32)],
        {
            "model_name": [
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
            ]
        },
    ),
    (
        Transpose0,
        [((768, 64), torch.float32)],
        {
            "model_name": [
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
            ]
        },
    ),
    (
        Transpose0,
        [((1, 16384, 768), torch.float32)],
        {
            "model_name": [
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
            ]
        },
    ),
    (Transpose0, [((1000, 256), torch.float32)], {"model_name": ["pt_mit_b0"]}),
    (Transpose0, [((256, 512), torch.float32)], {"model_name": ["pt_segformer_b1_finetuned_ade_512_512"]}),
    (Transpose0, [((256, 320), torch.float32)], {"model_name": ["pt_segformer_b1_finetuned_ade_512_512"]}),
    (Transpose0, [((256, 128), torch.float32)], {"model_name": ["pt_segformer_b1_finetuned_ade_512_512"]}),
    (Transpose0, [((1, 96, 4096), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Transpose3, [((1, 8, 8, 8, 8, 96), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Transpose0, [((96, 96), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Transpose1, [((64, 64, 3, 32), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Transpose0, [((192, 64, 32), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Transpose0, [((512, 2), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Transpose0, [((3, 512), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Transpose2, [((64, 64, 3), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Transpose0, [((3, 64, 64), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Transpose0, [((64, 3, 64, 32), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Transpose1, [((64, 3, 64, 32), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Transpose0, [((192, 32, 64), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Transpose0, [((384, 96), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Transpose0, [((96, 384), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Transpose0, [((192, 384), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Transpose3, [((1, 4, 8, 4, 8, 192), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Transpose1, [((16, 64, 6, 32), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Transpose0, [((96, 64, 32), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Transpose0, [((6, 512), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Transpose2, [((64, 64, 6), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Transpose0, [((6, 64, 64), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Transpose0, [((16, 6, 64, 32), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Transpose1, [((16, 6, 64, 32), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Transpose0, [((96, 32, 64), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Transpose3, [((1, 4, 4, 8, 8, 192), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Transpose0, [((384, 768), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Transpose3, [((1, 2, 8, 2, 8, 384), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Transpose1, [((4, 64, 12, 32), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Transpose0, [((48, 64, 32), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Transpose0, [((12, 512), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Transpose2, [((64, 64, 12), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Transpose0, [((12, 64, 64), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Transpose0, [((4, 12, 64, 32), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Transpose1, [((4, 12, 64, 32), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Transpose0, [((48, 32, 64), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Transpose3, [((1, 2, 2, 8, 8, 384), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Transpose0, [((768, 1536), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Transpose1, [((1, 64, 24, 32), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Transpose0, [((24, 64, 32), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Transpose0, [((24, 512), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Transpose2, [((64, 64, 24), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Transpose0, [((24, 64, 64), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Transpose0, [((1, 24, 64, 32), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Transpose1, [((1, 24, 64, 32), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Transpose0, [((24, 32, 64), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Transpose0, [((1, 64, 768), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (
        Transpose0,
        [((4096, 25088), torch.float32)],
        {
            "model_name": [
                "pt_vgg13_osmr",
                "pt_bn_vgg19_osmr",
                "pt_bn_vgg19b_osmr",
                "pt_vgg16_osmr",
                "pt_vgg19_osmr",
                "pt_vgg_19_hf",
                "pt_vgg_bn19_torchhub",
                "pt_vgg11_osmr",
            ]
        },
    ),
    (Transpose1, [((1, 197, 16, 64), torch.float32)], {"model_name": ["pt_vit_large_patch16_224"]}),
    (Transpose0, [((16, 197, 64), torch.float32)], {"model_name": ["pt_vit_large_patch16_224"]}),
    (Transpose0, [((1, 16, 197, 64), torch.float32)], {"model_name": ["pt_vit_large_patch16_224"]}),
    (Transpose1, [((1, 16, 197, 64), torch.float32)], {"model_name": ["pt_vit_large_patch16_224"]}),
    (Transpose0, [((16, 64, 197), torch.float32)], {"model_name": ["pt_vit_large_patch16_224"]}),
    (
        Transpose0,
        [((1, 3, 85, 6400), torch.float32)],
        {
            "model_name": [
                "pt_yolov5x_640x640",
                "pt_yolov5s_1280x1280",
                "pt_yolov5s_640x640",
                "pt_yolov5m_640x640",
                "pt_yolov5l_640x640",
                "pt_yolov5n_640x640",
            ]
        },
    ),
    (
        Transpose0,
        [((1, 3, 85, 1600), torch.float32)],
        {
            "model_name": [
                "pt_yolov5x_640x640",
                "pt_yolov5l_320x320",
                "pt_yolov5s_1280x1280",
                "pt_yolov5m_320x320",
                "pt_yolov5s_640x640",
                "pt_yolov5x_320x320",
                "pt_yolov5s_320x320",
                "pt_yolov5m_640x640",
                "pt_yolov5l_640x640",
                "pt_yolov5n_640x640",
                "pt_yolov5n_320x320",
            ]
        },
    ),
    (
        Transpose0,
        [((1, 3, 85, 400), torch.float32)],
        {
            "model_name": [
                "pt_yolov5x_640x640",
                "pt_yolov5l_320x320",
                "pt_yolov5m_320x320",
                "pt_yolov5s_640x640",
                "pt_yolov5x_320x320",
                "pt_yolov5s_320x320",
                "pt_yolov5m_640x640",
                "pt_yolov5l_640x640",
                "pt_yolov5n_640x640",
                "pt_yolov5n_320x320",
            ]
        },
    ),
    (
        Transpose0,
        [((1, 3, 85, 100), torch.float32)],
        {
            "model_name": [
                "pt_yolov5l_320x320",
                "pt_yolov5m_320x320",
                "pt_yolov5x_320x320",
                "pt_yolov5s_320x320",
                "pt_yolov5n_320x320",
            ]
        },
    ),
    (Transpose0, [((1, 3, 85, 25600), torch.float32)], {"model_name": ["pt_yolov5s_1280x1280"]}),
    (
        Transpose0,
        [((1, 3, 85, 3600), torch.float32)],
        {
            "model_name": [
                "pt_yolov5m_480x480",
                "pt_yolov5n_480x480",
                "pt_yolov5l_480x480",
                "pt_yolov5x_480x480",
                "pt_yolov5s_480x480",
            ]
        },
    ),
    (
        Transpose0,
        [((1, 3, 85, 900), torch.float32)],
        {
            "model_name": [
                "pt_yolov5m_480x480",
                "pt_yolov5n_480x480",
                "pt_yolov5l_480x480",
                "pt_yolov5x_480x480",
                "pt_yolov5s_480x480",
            ]
        },
    ),
    (
        Transpose0,
        [((1, 3, 85, 225), torch.float32)],
        {
            "model_name": [
                "pt_yolov5m_480x480",
                "pt_yolov5n_480x480",
                "pt_yolov5l_480x480",
                "pt_yolov5x_480x480",
                "pt_yolov5s_480x480",
            ]
        },
    ),
    (Transpose1, [((1, 4, 17, 4480), torch.float32)], {"model_name": ["pt_yolov6m", "pt_yolov6l"]}),
    (Transpose1, [((1, 4, 17, 1120), torch.float32)], {"model_name": ["pt_yolov6m", "pt_yolov6l"]}),
    (Transpose1, [((1, 4, 17, 280), torch.float32)], {"model_name": ["pt_yolov6m", "pt_yolov6l"]}),
    (
        Transpose0,
        [((1, 4, 5880), torch.float32)],
        {"model_name": ["pt_yolov6m", "pt_yolov6n", "pt_yolov6l", "pt_yolov6s"]},
    ),
    (
        Transpose0,
        [((1, 80, 5880), torch.float32)],
        {"model_name": ["pt_yolov6m", "pt_yolov6n", "pt_yolov6l", "pt_yolov6s"]},
    ),
    (
        Transpose0,
        [((1, 85, 8400), torch.float32)],
        {"model_name": ["pt_yolox_m", "pt_yolox_s", "pt_yolox_darknet", "pt_yolox_x", "pt_yolox_l"]},
    ),
    (Transpose0, [((1, 85, 3549), torch.float32)], {"model_name": ["pt_yolox_nano", "pt_yolox_tiny"]}),
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
