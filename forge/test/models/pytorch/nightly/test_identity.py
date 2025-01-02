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


class Identity0(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, identity_input_0):
        identity_output_1 = forge.op.Identity("", identity_input_0)
        return identity_output_1


def ids_func(param):
    forge_module, shapes_dtypes, _ = param
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (Identity0, [((2, 1, 2048), torch.float32)], {"model_name": ["pt_musicgen_large"]}),
    (Identity0, [((64, 1, 1), torch.float32)], {"model_name": ["pt_musicgen_large"]}),
    (
        Identity0,
        [((2, 13, 768), torch.float32)],
        {"model_name": ["pt_musicgen_large", "pt_musicgen_small", "pt_musicgen_medium"]},
    ),
    (
        Identity0,
        [((2, 12, 13, 13), torch.float32)],
        {"model_name": ["pt_musicgen_large", "pt_musicgen_small", "pt_musicgen_medium"]},
    ),
    (
        Identity0,
        [((2, 13, 3072), torch.float32)],
        {"model_name": ["pt_musicgen_large", "pt_musicgen_small", "pt_musicgen_medium"]},
    ),
    (Identity0, [((64, 1, 13), torch.float32)], {"model_name": ["pt_musicgen_large"]}),
    (Identity0, [((2, 1, 8192), torch.float32)], {"model_name": ["pt_musicgen_large"]}),
    (Identity0, [((2, 1, 1024), torch.float32)], {"model_name": ["pt_musicgen_small"]}),
    (Identity0, [((32, 1, 1), torch.float32)], {"model_name": ["pt_musicgen_small"]}),
    (Identity0, [((32, 1, 13), torch.float32)], {"model_name": ["pt_musicgen_small"]}),
    (Identity0, [((2, 1, 4096), torch.float32)], {"model_name": ["pt_musicgen_small"]}),
    (Identity0, [((2, 1, 1536), torch.float32)], {"model_name": ["pt_musicgen_medium"]}),
    (Identity0, [((48, 1, 1), torch.float32)], {"model_name": ["pt_musicgen_medium"]}),
    (Identity0, [((48, 1, 13), torch.float32)], {"model_name": ["pt_musicgen_medium"]}),
    (Identity0, [((2, 1, 6144), torch.float32)], {"model_name": ["pt_musicgen_medium"]}),
    (Identity0, [((1, 2, 768), torch.float32)], {"model_name": ["pt_whisper_small"]}),
    (Identity0, [((1, 12, 2, 2), torch.float32)], {"model_name": ["pt_whisper_small"]}),
    (Identity0, [((1, 12, 2, 1500), torch.float32)], {"model_name": ["pt_whisper_small"]}),
    (Identity0, [((1, 2, 3072), torch.float32)], {"model_name": ["pt_whisper_small"]}),
    (Identity0, [((1, 2, 1280), torch.float32)], {"model_name": ["pt_whisper_large", "pt_whisper_large_v3_turbo"]}),
    (Identity0, [((1, 20, 2, 2), torch.float32)], {"model_name": ["pt_whisper_large", "pt_whisper_large_v3_turbo"]}),
    (Identity0, [((1, 20, 2, 1500), torch.float32)], {"model_name": ["pt_whisper_large", "pt_whisper_large_v3_turbo"]}),
    (Identity0, [((1, 2, 5120), torch.float32)], {"model_name": ["pt_whisper_large", "pt_whisper_large_v3_turbo"]}),
    (Identity0, [((1, 2, 1024), torch.float32)], {"model_name": ["pt_whisper_medium"]}),
    (Identity0, [((1, 16, 2, 2), torch.float32)], {"model_name": ["pt_whisper_medium"]}),
    (Identity0, [((1, 16, 2, 1500), torch.float32)], {"model_name": ["pt_whisper_medium"]}),
    (Identity0, [((1, 2, 4096), torch.float32)], {"model_name": ["pt_whisper_medium"]}),
    (Identity0, [((1, 2, 384), torch.float32)], {"model_name": ["pt_whisper_tiny"]}),
    (Identity0, [((1, 6, 2, 2), torch.float32)], {"model_name": ["pt_whisper_tiny"]}),
    (Identity0, [((1, 6, 2, 1500), torch.float32)], {"model_name": ["pt_whisper_tiny"]}),
    (Identity0, [((1, 2, 1536), torch.float32)], {"model_name": ["pt_whisper_tiny"]}),
    (Identity0, [((1, 2, 512), torch.float32)], {"model_name": ["pt_whisper_base"]}),
    (Identity0, [((1, 8, 2, 2), torch.float32)], {"model_name": ["pt_whisper_base"]}),
    (Identity0, [((1, 8, 2, 1500), torch.float32)], {"model_name": ["pt_whisper_base"]}),
    (Identity0, [((1, 2, 2048), torch.float32)], {"model_name": ["pt_whisper_base"]}),
    (Identity0, [((16, 7, 7), torch.float32)], {"model_name": ["pt_clip_vit_base_patch32_text"]}),
    (Identity0, [((1, 12, 204, 204), torch.float32)], {"model_name": ["pt_ViLt_maskedlm"]}),
    (Identity0, [((1, 204, 768), torch.float32)], {"model_name": ["pt_ViLt_maskedlm"]}),
    (Identity0, [((1, 12, 201, 201), torch.float32)], {"model_name": ["pt_ViLt_question_answering"]}),
    (Identity0, [((1, 201, 768), torch.float32)], {"model_name": ["pt_ViLt_question_answering"]}),
    (
        Identity0,
        [((1, 128, 128), torch.float32)],
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
        Identity0,
        [((1, 12, 128, 128), torch.float32)],
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
        Identity0,
        [((1, 128, 768), torch.float32)],
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
        Identity0,
        [((1, 16, 128, 128), torch.float32)],
        {
            "model_name": [
                "pt_albert_xlarge_v1_token_cls",
                "pt_albert_large_v1_masked_lm",
                "pt_albert_large_v2_token_cls",
                "pt_albert_large_v2_masked_lm",
                "pt_albert_xlarge_v2_token_cls",
                "pt_albert_xlarge_v1_masked_lm",
                "pt_albert_large_v1_token_cls",
                "pt_albert_xlarge_v2_masked_lm",
                "pt_bert_sequence_classification",
            ]
        },
    ),
    (
        Identity0,
        [((1, 128, 2048), torch.float32)],
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
        Identity0,
        [((1, 128, 1024), torch.float32)],
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
        Identity0,
        [((1, 64, 128, 128), torch.float32)],
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
        Identity0,
        [((1, 128, 4096), torch.float32)],
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
        Identity0,
        [((1, 256, 1024), torch.float32)],
        {
            "model_name": [
                "pt_bart",
                "pt_codegen_350M_mono",
                "pt_opt_350m_causal_lm",
                "pt_t5_large",
                "pt_xglm_564M",
                "pt_segformer_b0_finetuned_ade_512_512",
                "pt_mit_b0",
            ]
        },
    ),
    (
        Identity0,
        [((16, 256, 256), torch.float32)],
        {"model_name": ["pt_bart", "pt_opt_350m_causal_lm", "pt_xglm_1_7B", "pt_xglm_564M"]},
    ),
    (Identity0, [((1, 256, 4096), torch.float32)], {"model_name": ["pt_bart", "pt_xglm_564M"]}),
    (Identity0, [((1, 384, 1024), torch.float32)], {"model_name": ["pt_bert_qa"]}),
    (Identity0, [((1, 16, 384, 384), torch.float32)], {"model_name": ["pt_bert_qa"]}),
    (
        Identity0,
        [((1, 16, 256, 256), torch.float32)],
        {"model_name": ["pt_codegen_350M_mono", "pt_gpt_neo_1_3B_causal_lm"]},
    ),
    (
        Identity0,
        [((1, 768), torch.float32)],
        {
            "model_name": [
                "pt_distilbert_sequence_classification",
                "pt_roberta_sentiment",
                "pt_squeezebert",
                "pt_mobilenet_v1_192",
            ]
        },
    ),
    (Identity0, [((1, 384, 768), torch.float32)], {"model_name": ["pt_distilbert_question_answering"]}),
    (Identity0, [((1, 12, 384, 384), torch.float32)], {"model_name": ["pt_distilbert_question_answering"]}),
    (Identity0, [((1, 71, 6, 6), torch.float32)], {"model_name": ["pt_falcon"]}),
    (Identity0, [((1, 6, 4544), torch.float32)], {"model_name": ["pt_falcon"]}),
    (Identity0, [((1, 64, 334, 334), torch.float32)], {"model_name": ["pt_fuyu_8b"]}),
    (Identity0, [((1, 334, 4096), torch.float32)], {"model_name": ["pt_fuyu_8b"]}),
    (Identity0, [((1, 8, 7, 7), torch.float32)], {"model_name": ["pt_gemma_2b"]}),
    (
        Identity0,
        [((1, 256, 768), torch.float32)],
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
        Identity0,
        [((1, 12, 256, 256), torch.float32)],
        {"model_name": ["pt_gpt2_generation", "pt_gpt_neo_125M_causal_lm"]},
    ),
    (
        Identity0,
        [((1, 256, 2560), torch.float32)],
        {"model_name": ["pt_gpt_neo_2_7B_causal_lm", "pt_phi_2_causal_lm", "pt_phi_2_pytdml_causal_lm"]},
    ),
    (Identity0, [((1, 20, 256, 256), torch.float32)], {"model_name": ["pt_gpt_neo_2_7B_causal_lm"]}),
    (
        Identity0,
        [((1, 256, 2048), torch.float32)],
        {
            "model_name": [
                "pt_gpt_neo_1_3B_causal_lm",
                "pt_opt_1_3b_causal_lm",
                "pt_xglm_1_7B",
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
        Identity0,
        [((1, 32, 256, 256), torch.float32)],
        {
            "model_name": [
                "pt_Llama_3_1_8B_Instruct_causal_lm",
                "pt_Meta_Llama_3_8B_causal_lm",
                "pt_Llama_3_2_1B_Instruct_causal_lm",
                "pt_Meta_Llama_3_8B_Instruct_causal_lm",
                "pt_Llama_3_2_1B_causal_lm",
                "pt_Llama_3_1_8B_causal_lm",
                "pt_phi_2_causal_lm",
                "pt_phi_2_pytdml_causal_lm",
            ]
        },
    ),
    (
        Identity0,
        [((1, 32, 4, 4), torch.float32)],
        {
            "model_name": [
                "pt_Llama_3_1_8B_Instruct_seq_cls",
                "pt_Meta_Llama_3_8B_seq_cls",
                "pt_Llama_3_2_1B_Instruct_seq_cls",
                "pt_Llama_3_2_1B_seq_cls",
                "pt_Llama_3_1_8B_seq_cls",
                "pt_Meta_Llama_3_8B_Instruct_seq_cls",
            ]
        },
    ),
    (Identity0, [((1, 32, 128, 128), torch.float32)], {"model_name": ["pt_Mistral_7B_v0_1"]}),
    (Identity0, [((12, 32, 32), torch.float32)], {"model_name": ["pt_opt_125m_seq_cls", "pt_opt_125m_qa"]}),
    (Identity0, [((1, 32, 768), torch.float32)], {"model_name": ["pt_opt_125m_seq_cls", "pt_opt_125m_qa"]}),
    (Identity0, [((32, 768), torch.float32)], {"model_name": ["pt_opt_125m_seq_cls", "pt_opt_125m_qa"]}),
    (Identity0, [((32, 32, 32), torch.float32)], {"model_name": ["pt_opt_1_3b_seq_cls", "pt_opt_1_3b_qa"]}),
    (Identity0, [((1, 32, 2048), torch.float32)], {"model_name": ["pt_opt_1_3b_seq_cls", "pt_opt_1_3b_qa"]}),
    (Identity0, [((32, 2048), torch.float32)], {"model_name": ["pt_opt_1_3b_seq_cls", "pt_opt_1_3b_qa"]}),
    (Identity0, [((16, 32, 32), torch.float32)], {"model_name": ["pt_opt_350m_qa", "pt_opt_350m_seq_cls"]}),
    (Identity0, [((1, 32, 1024), torch.float32)], {"model_name": ["pt_opt_350m_qa", "pt_opt_350m_seq_cls"]}),
    (Identity0, [((32, 1024), torch.float32)], {"model_name": ["pt_opt_350m_qa", "pt_opt_350m_seq_cls"]}),
    (Identity0, [((256, 1024), torch.float32)], {"model_name": ["pt_opt_350m_causal_lm"]}),
    (Identity0, [((12, 256, 256), torch.float32)], {"model_name": ["pt_opt_125m_causal_lm"]}),
    (Identity0, [((256, 768), torch.float32)], {"model_name": ["pt_opt_125m_causal_lm"]}),
    (Identity0, [((32, 256, 256), torch.float32)], {"model_name": ["pt_opt_1_3b_causal_lm"]}),
    (Identity0, [((256, 2048), torch.float32)], {"model_name": ["pt_opt_1_3b_causal_lm"]}),
    (Identity0, [((1, 12, 2560), torch.float32)], {"model_name": ["pt_phi_2_pytdml_token_cls", "pt_phi_2_token_cls"]}),
    (
        Identity0,
        [((1, 32, 12, 12), torch.float32)],
        {"model_name": ["pt_phi_2_pytdml_token_cls", "pt_phi_2_token_cls"]},
    ),
    (Identity0, [((1, 11, 2560), torch.float32)], {"model_name": ["pt_phi_2_seq_cls", "pt_phi_2_pytdml_seq_cls"]}),
    (Identity0, [((1, 32, 11, 11), torch.float32)], {"model_name": ["pt_phi_2_seq_cls", "pt_phi_2_pytdml_seq_cls"]}),
    (Identity0, [((1, 16, 29, 29), torch.float32)], {"model_name": ["pt_qwen_chat", "pt_Qwen_Qwen2_5_3B"]}),
    (Identity0, [((1, 16, 6, 6), torch.float32)], {"model_name": ["pt_qwen_causal_lm"]}),
    (
        Identity0,
        [((1, 12, 35, 35), torch.float32)],
        {"model_name": ["pt_Qwen_Qwen2_5_Coder_1_5B_Instruct", "pt_Qwen_Qwen2_5_Coder_1_5B"]},
    ),
    (
        Identity0,
        [((1, 16, 35, 35), torch.float32)],
        {"model_name": ["pt_Qwen_Qwen2_5_Coder_3B", "pt_Qwen_Qwen2_5_Coder_3B_Instruct"]},
    ),
    (
        Identity0,
        [((1, 28, 35, 35), torch.float32)],
        {"model_name": ["pt_Qwen_Qwen2_5_Coder_7B", "pt_Qwen_Qwen2_5_Coder_7B_Instruct"]},
    ),
    (Identity0, [((1, 14, 35, 35), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_Coder_0_5B"]}),
    (Identity0, [((1, 12, 39, 39), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_1_5B_Instruct"]}),
    (Identity0, [((1, 12, 29, 29), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_1_5B"]}),
    (Identity0, [((1, 28, 29, 29), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_7B"]}),
    (Identity0, [((1, 14, 39, 39), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_0_5B_Instruct"]}),
    (Identity0, [((1, 14, 29, 29), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_0_5B"]}),
    (Identity0, [((1, 16, 39, 39), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_3B_Instruct"]}),
    (Identity0, [((1, 28, 39, 39), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_7B_Instruct"]}),
    (Identity0, [((1, 768, 128), torch.float32)], {"model_name": ["pt_squeezebert"]}),
    (Identity0, [((1, 1, 1024), torch.float32)], {"model_name": ["pt_t5_large", "pt_google_flan_t5_small"]}),
    (Identity0, [((1, 16, 1, 1), torch.float32)], {"model_name": ["pt_t5_large"]}),
    (Identity0, [((1, 16, 1, 256), torch.float32)], {"model_name": ["pt_t5_large"]}),
    (Identity0, [((1, 1, 4096), torch.float32)], {"model_name": ["pt_t5_large"]}),
    (Identity0, [((1, 1, 768), torch.float32)], {"model_name": ["pt_t5_base", "pt_google_flan_t5_base"]}),
    (Identity0, [((1, 12, 1, 1), torch.float32)], {"model_name": ["pt_t5_base", "pt_google_flan_t5_base"]}),
    (Identity0, [((1, 12, 1, 256), torch.float32)], {"model_name": ["pt_t5_base", "pt_google_flan_t5_base"]}),
    (Identity0, [((1, 1, 3072), torch.float32)], {"model_name": ["pt_t5_base"]}),
    (Identity0, [((1, 1, 2048), torch.float32)], {"model_name": ["pt_google_flan_t5_base", "pt_t5_small"]}),
    (Identity0, [((1, 1, 512), torch.float32)], {"model_name": ["pt_t5_small", "pt_google_flan_t5_small"]}),
    (Identity0, [((1, 8, 1, 1), torch.float32)], {"model_name": ["pt_t5_small"]}),
    (Identity0, [((1, 6, 1, 1), torch.float32)], {"model_name": ["pt_google_flan_t5_small"]}),
    (Identity0, [((1, 256, 8192), torch.float32)], {"model_name": ["pt_xglm_1_7B"]}),
    (Identity0, [((1, 9216), torch.float32)], {"model_name": ["pt_alexnet_torchhub", "pt_rcnn"]}),
    (
        Identity0,
        [((1, 4096), torch.float32)],
        {
            "model_name": [
                "pt_alexnet_torchhub",
                "pt_rcnn",
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
    (Identity0, [((1, 197, 384), torch.float32)], {"model_name": ["pt_deit_small_patch16_224"]}),
    (Identity0, [((1, 6, 197, 197), torch.float32)], {"model_name": ["pt_deit_small_patch16_224"]}),
    (
        Identity0,
        [((1, 197, 768), torch.float32)],
        {"model_name": ["pt_deit_base_patch16_224", "pt_deit_base_distilled_patch16_224", "pt_vit_base_patch16_224"]},
    ),
    (
        Identity0,
        [((1, 12, 197, 197), torch.float32)],
        {"model_name": ["pt_deit_base_patch16_224", "pt_deit_base_distilled_patch16_224", "pt_vit_base_patch16_224"]},
    ),
    (Identity0, [((1, 197, 192), torch.float32)], {"model_name": ["pt_deit_tiny_patch16_224"]}),
    (Identity0, [((1, 3, 197, 197), torch.float32)], {"model_name": ["pt_deit_tiny_patch16_224"]}),
    (Identity0, [((1, 1792), torch.float32)], {"model_name": ["pt_efficientnet_b4_torchvision"]}),
    (
        Identity0,
        [((1, 1280), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_b0_torchvision",
                "pt_ghostnet_100",
                "mobilenetv2_basic",
                "mobilenetv2_160",
                "mobilenetv2_96",
                "mobilenetv2_224",
                "pt_mobilenet_v3_large",
            ]
        },
    ),
    (
        Identity0,
        [((1, 1024), torch.float32)],
        {"model_name": ["pt_googlenet", "pt_mobilenet_v1_224", "pt_mobilenet_v3_small"]},
    ),
    (
        Identity0,
        [((1, 1024, 512), torch.float32)],
        {"model_name": ["pt_mixer_l32_224", "pt_mixer_l16_224", "pt_mixer_l16_224_in21k"]},
    ),
    (Identity0, [((1, 1024, 49), torch.float32)], {"model_name": ["pt_mixer_l32_224"]}),
    (Identity0, [((1, 49, 4096), torch.float32)], {"model_name": ["pt_mixer_l32_224"]}),
    (Identity0, [((1, 49, 1024), torch.float32)], {"model_name": ["pt_mixer_l32_224"]}),
    (Identity0, [((1, 1024, 196), torch.float32)], {"model_name": ["pt_mixer_l16_224", "pt_mixer_l16_224_in21k"]}),
    (Identity0, [((1, 196, 4096), torch.float32)], {"model_name": ["pt_mixer_l16_224", "pt_mixer_l16_224_in21k"]}),
    (Identity0, [((1, 196, 1024), torch.float32)], {"model_name": ["pt_mixer_l16_224", "pt_mixer_l16_224_in21k"]}),
    (Identity0, [((1, 512, 256), torch.float32)], {"model_name": ["pt_mixer_s32_224", "pt_mixer_s16_224"]}),
    (Identity0, [((1, 512, 49), torch.float32)], {"model_name": ["pt_mixer_s32_224"]}),
    (Identity0, [((1, 49, 2048), torch.float32)], {"model_name": ["pt_mixer_s32_224"]}),
    (Identity0, [((1, 49, 512), torch.float32)], {"model_name": ["pt_mixer_s32_224"]}),
    (
        Identity0,
        [((1, 768, 384), torch.float32)],
        {
            "model_name": [
                "pt_mixer_b16_224_miil_in21k",
                "pt_mixer_b16_224",
                "pt_mixer_b32_224",
                "pt_mixer_b16_224_in21k",
                "pt_mixer_b16_224_miil",
            ]
        },
    ),
    (
        Identity0,
        [((1, 768, 196), torch.float32)],
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
        Identity0,
        [((1, 196, 3072), torch.float32)],
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
        Identity0,
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
    (Identity0, [((1, 512, 196), torch.float32)], {"model_name": ["pt_mixer_s16_224"]}),
    (Identity0, [((1, 196, 2048), torch.float32)], {"model_name": ["pt_mixer_s16_224"]}),
    (Identity0, [((1, 196, 512), torch.float32)], {"model_name": ["pt_mixer_s16_224"]}),
    (Identity0, [((1, 768, 49), torch.float32)], {"model_name": ["pt_mixer_b32_224"]}),
    (Identity0, [((1, 49, 3072), torch.float32)], {"model_name": ["pt_mixer_b32_224"]}),
    (Identity0, [((1, 49, 768), torch.float32)], {"model_name": ["pt_mixer_b32_224"]}),
    (Identity0, [((1, 256, 28, 28), torch.float32)], {"model_name": ["mobilenetv2_deeplabv3"]}),
    (
        Identity0,
        [((1, 1, 512, 50176), torch.float32)],
        {"model_name": ["pt_vision_perceiver_learned", "pt_vision_perceiver_fourier"]},
    ),
    (
        Identity0,
        [((1, 8, 512, 512), torch.float32)],
        {"model_name": ["pt_vision_perceiver_learned", "pt_vision_perceiver_conv", "pt_vision_perceiver_fourier"]},
    ),
    (
        Identity0,
        [((1, 1, 1, 512), torch.float32)],
        {"model_name": ["pt_vision_perceiver_learned", "pt_vision_perceiver_conv", "pt_vision_perceiver_fourier"]},
    ),
    (Identity0, [((1, 1, 512, 3025), torch.float32)], {"model_name": ["pt_vision_perceiver_conv"]}),
    (
        Identity0,
        [((1, 1, 16384, 256), torch.float32)],
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
        Identity0,
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
        Identity0,
        [((1, 16384, 256), torch.float32)],
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
        Identity0,
        [((1, 2, 4096, 256), torch.float32)],
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
        Identity0,
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
        Identity0,
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
        Identity0,
        [((1, 5, 1024, 256), torch.float32)],
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
        Identity0,
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
        Identity0,
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
        Identity0,
        [((1, 8, 256, 256), torch.float32)],
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
        Identity0,
        [((1, 256, 512), torch.float32)],
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
        Identity0,
        [((1, 16384, 32), torch.float32)],
        {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_mit_b0"]},
    ),
    (
        Identity0,
        [((1, 16384, 128), torch.float32)],
        {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_mit_b0"]},
    ),
    (
        Identity0,
        [((1, 4096, 64), torch.float32)],
        {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_mit_b0"]},
    ),
    (
        Identity0,
        [((1, 4096, 256), torch.float32)],
        {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_mit_b0"]},
    ),
    (
        Identity0,
        [((1, 1024, 160), torch.float32)],
        {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_mit_b0"]},
    ),
    (
        Identity0,
        [((1, 1024, 640), torch.float32)],
        {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_mit_b0"]},
    ),
    (
        Identity0,
        [((1, 256, 256), torch.float32)],
        {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_mit_b0"]},
    ),
    (
        Identity0,
        [((1, 256, 128, 128), torch.float32)],
        {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_segformer_b1_finetuned_ade_512_512"]},
    ),
    (
        Identity0,
        [((1, 768, 128, 128), torch.float32)],
        {
            "model_name": [
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
            ]
        },
    ),
    (Identity0, [((1, 4096, 96), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Identity0, [((64, 3, 64, 64), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Identity0, [((64, 64, 96), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Identity0, [((16, 6, 64, 64), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Identity0, [((16, 64, 192), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Identity0, [((1, 1024, 192), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Identity0, [((4, 12, 64, 64), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Identity0, [((4, 64, 384), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Identity0, [((1, 256, 384), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Identity0, [((1, 24, 64, 64), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Identity0, [((1, 64, 768), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Identity0, [((1, 4096, 1, 1), torch.float32)], {"model_name": ["pt_vgg19_bn_timm"]}),
    (Identity0, [((1, 197, 1024), torch.float32)], {"model_name": ["pt_vit_large_patch16_224"]}),
    (Identity0, [((1, 16, 197, 197), torch.float32)], {"model_name": ["pt_vit_large_patch16_224"]}),
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
