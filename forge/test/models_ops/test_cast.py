# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import forge
import forge.op
from forge import ForgeModule

from loguru import logger
import torch

from forge import Tensor, compile
from forge.verify.verify import verify
from forge.verify.value_checkers import AutomaticValueChecker
from forge.verify.config import VerifyConfig
from forge.forge_property_utils import (
    record_forge_op_name,
    record_op_model_names,
    record_forge_op_args,
    record_single_op_operands_info,
)
import pytest


class Cast0(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, cast_input_0):
        cast_output_1 = forge.op.Cast("", cast_input_0, dtype=torch.float32)
        return cast_output_1


class Cast1(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, cast_input_0):
        cast_output_1 = forge.op.Cast("", cast_input_0, dtype=torch.bool)
        return cast_output_1


class Cast2(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, cast_input_0):
        cast_output_1 = forge.op.Cast("", cast_input_0, dtype=torch.int32)
        return cast_output_1


class Cast3(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, cast_input_0):
        cast_output_1 = forge.op.Cast("", cast_input_0, dtype=torch.int64)
        return cast_output_1


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (
        Cast0,
        [((1, 1, 128, 128), torch.int64)],
        {
            "model_names": [
                "onnx_albert_xxlarge_v1_mlm_hf",
                "pt_albert_base_v1_mlm_hf",
                "pt_albert_base_v1_token_cls_hf",
                "onnx_albert_xlarge_v2_mlm_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
                "pt_albert_xxlarge_v2_token_cls_hf",
                "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "onnx_albert_large_v1_mlm_hf",
                "pt_albert_xlarge_v1_mlm_hf",
                "pt_albert_xlarge_v1_token_cls_hf",
                "onnx_albert_large_v2_mlm_hf",
                "pt_albert_xlarge_v2_mlm_hf",
                "pt_albert_xlarge_v2_token_cls_hf",
                "pt_bert_bert_base_uncased_mlm_hf",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "onnx_albert_base_v2_mlm_hf",
                "onnx_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "pt_albert_large_v2_mlm_hf",
                "pt_albert_large_v2_token_cls_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
                "pt_albert_base_v2_mlm_hf",
                "pt_albert_base_v2_token_cls_hf",
                "pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder",
                "pt_qwen_v3_0_6b_clm_hf",
                "onnx_albert_base_v1_mlm_hf",
                "pt_albert_large_v1_mlm_hf",
                "pt_albert_large_v1_token_cls_hf",
                "pt_distilbert_distilbert_base_uncased_mlm_hf",
                "pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder",
                "pt_qwen_v3_1_7b_clm_hf",
                "onnx_albert_xxlarge_v2_mlm_hf",
                "pt_distilbert_distilbert_base_multilingual_cased_mlm_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder",
                "onnx_albert_xlarge_v1_mlm_hf",
                "pt_albert_xxlarge_v1_mlm_hf",
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_qwen_v3_4b_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dtype": "torch.float32"},
        },
    ),
    (
        Cast1,
        [((1, 1, 128, 128), torch.float32)],
        {
            "model_names": [
                "onnx_albert_xxlarge_v1_mlm_hf",
                "onnx_albert_xlarge_v2_mlm_hf",
                "onnx_albert_large_v1_mlm_hf",
                "onnx_albert_large_v2_mlm_hf",
                "onnx_albert_base_v2_mlm_hf",
                "onnx_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "onnx_albert_base_v1_mlm_hf",
                "onnx_albert_xxlarge_v2_mlm_hf",
                "onnx_albert_xlarge_v1_mlm_hf",
            ],
            "pcc": 0.99,
            "args": {"dtype": "torch.bool"},
        },
    ),
    (
        Cast0,
        [((1, 1, 128, 128), torch.bool)],
        {
            "model_names": [
                "onnx_albert_xxlarge_v1_mlm_hf",
                "pt_albert_base_v1_mlm_hf",
                "pt_albert_base_v1_token_cls_hf",
                "onnx_albert_xlarge_v2_mlm_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
                "pt_albert_xxlarge_v2_token_cls_hf",
                "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "onnx_albert_large_v1_mlm_hf",
                "pt_albert_xlarge_v1_mlm_hf",
                "pt_albert_xlarge_v1_token_cls_hf",
                "onnx_albert_large_v2_mlm_hf",
                "pt_albert_xlarge_v2_mlm_hf",
                "pt_albert_xlarge_v2_token_cls_hf",
                "pt_bert_bert_base_uncased_mlm_hf",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "onnx_albert_base_v2_mlm_hf",
                "onnx_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "pt_albert_large_v2_mlm_hf",
                "pt_albert_large_v2_token_cls_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
                "pt_albert_base_v2_mlm_hf",
                "pt_albert_base_v2_token_cls_hf",
                "pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder",
                "pt_qwen_v3_0_6b_clm_hf",
                "onnx_albert_base_v1_mlm_hf",
                "pt_albert_large_v1_mlm_hf",
                "pt_albert_large_v1_token_cls_hf",
                "pt_distilbert_distilbert_base_uncased_mlm_hf",
                "pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder",
                "pt_qwen_v3_1_7b_clm_hf",
                "onnx_albert_xxlarge_v2_mlm_hf",
                "pt_distilbert_distilbert_base_multilingual_cased_mlm_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder",
                "onnx_albert_xlarge_v1_mlm_hf",
                "pt_albert_xxlarge_v1_mlm_hf",
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_qwen_v3_4b_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dtype": "torch.float32"},
        },
    ),
    (
        Cast2,
        [((1, 7), torch.bool)],
        {
            "model_names": [
                "onnx_gpt_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
                "pt_gpt_gpt2_sequence_classification_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dtype": "torch.int32"},
        },
    ),
    (
        Cast3,
        [((1, 7), torch.int32)],
        {
            "model_names": ["onnx_gpt_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dtype": "torch.int64"},
        },
    ),
    (
        Cast3,
        [((1,), torch.int32)],
        {
            "model_names": ["onnx_gpt_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dtype": "torch.int64"},
        },
    ),
    (
        Cast1,
        [((1,), torch.bool)],
        {
            "model_names": ["onnx_gpt_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dtype": "torch.bool"},
        },
    ),
    (
        Cast0,
        [((1, 1, 14, 14), torch.int64)],
        {"model_names": ["pt_albert_squad2_qa_hf"], "pcc": 0.99, "args": {"dtype": "torch.float32"}},
    ),
    (
        Cast0,
        [((1, 1, 14, 14), torch.bool)],
        {"model_names": ["pt_albert_squad2_qa_hf"], "pcc": 0.99, "args": {"dtype": "torch.float32"}},
    ),
    (
        Cast0,
        [((1, 1, 256, 256), torch.int64)],
        {
            "model_names": [
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_phi_1_5_microsoft_phi_1_5_clm_hf",
                "pt_opt_facebook_opt_125m_clm_hf",
                "pt_bart_large_seq_cls_hf",
                "pt_opt_facebook_opt_1_3b_clm_hf",
                "pt_xglm_xglm_1_7b_clm_hf",
                "pt_gptneo_gpt_neo_125m_clm_hf",
                "pt_gptneo_gpt_neo_1_3b_clm_hf",
                "pt_phi1_microsoft_phi_1_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
                "pt_opt_facebook_opt_350m_clm_hf",
                "pt_xglm_xglm_564m_clm_hf",
                "pt_llama3_llama_3_2_1b_clm_hf",
                "pt_llama3_llama_3_2_1b_instruct_clm_hf",
                "pt_gptneo_gpt_neo_2_7b_clm_hf",
                "pt_llama3_huggyllama_7b_clm_hf",
                "pt_llama3_llama_3_1_8b_clm_hf",
                "pt_llama3_llama_3_1_8b_instruct_clm_hf",
                "pt_llama3_llama_3_2_3b_clm_hf",
                "pt_llama3_llama_3_2_3b_instruct_clm_hf",
                "pt_llama3_llama_3_8b_clm_hf",
                "pt_llama3_llama_3_8b_instruct_clm_hf",
                "pt_phi2_microsoft_phi_2_pytdml_clm_hf",
                "pt_phi3_microsoft_phi_3_mini_128k_instruct_clm_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dtype": "torch.float32"},
        },
    ),
    (
        Cast1,
        [((1, 1, 256, 256), torch.float32)],
        {
            "model_names": [
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dtype": "torch.bool"},
        },
    ),
    (
        Cast0,
        [((1, 1, 256, 256), torch.bool)],
        {
            "model_names": [
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_phi_1_5_microsoft_phi_1_5_clm_hf",
                "pt_opt_facebook_opt_125m_clm_hf",
                "pt_bart_large_seq_cls_hf",
                "pt_opt_facebook_opt_1_3b_clm_hf",
                "pt_xglm_xglm_1_7b_clm_hf",
                "pt_gptneo_gpt_neo_125m_clm_hf",
                "pt_gptneo_gpt_neo_1_3b_clm_hf",
                "pt_phi1_microsoft_phi_1_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
                "pt_opt_facebook_opt_350m_clm_hf",
                "pt_xglm_xglm_564m_clm_hf",
                "pt_llama3_llama_3_2_1b_clm_hf",
                "pt_llama3_llama_3_2_1b_instruct_clm_hf",
                "pt_gptneo_gpt_neo_2_7b_clm_hf",
                "pt_llama3_huggyllama_7b_clm_hf",
                "pt_llama3_llama_3_1_8b_clm_hf",
                "pt_llama3_llama_3_1_8b_instruct_clm_hf",
                "pt_llama3_llama_3_2_3b_clm_hf",
                "pt_llama3_llama_3_2_3b_instruct_clm_hf",
                "pt_llama3_llama_3_8b_clm_hf",
                "pt_llama3_llama_3_8b_instruct_clm_hf",
                "pt_phi2_microsoft_phi_2_pytdml_clm_hf",
                "pt_phi3_microsoft_phi_3_mini_128k_instruct_clm_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dtype": "torch.float32"},
        },
    ),
    (
        Cast1,
        [((1, 1, 256, 256), torch.bool)],
        {
            "model_names": [
                "pt_gptneo_gpt_neo_125m_clm_hf",
                "pt_gptneo_gpt_neo_1_3b_clm_hf",
                "pt_gptneo_gpt_neo_2_7b_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dtype": "torch.bool"},
        },
    ),
    (
        Cast0,
        [((1, 1, 32, 32), torch.int64)],
        {
            "model_names": [
                "pt_opt_facebook_opt_125m_seq_cls_hf",
                "pt_opt_facebook_opt_350m_qa_hf",
                "pt_opt_facebook_opt_1_3b_qa_hf",
                "pt_opt_facebook_opt_125m_qa_hf",
                "pt_opt_facebook_opt_1_3b_seq_cls_hf",
                "pt_opt_facebook_opt_350m_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dtype": "torch.float32"},
        },
    ),
    (
        Cast0,
        [((1, 1, 32, 32), torch.bool)],
        {
            "model_names": [
                "pt_opt_facebook_opt_125m_seq_cls_hf",
                "pt_opt_facebook_opt_350m_qa_hf",
                "pt_bloom_default_clm_hf",
                "pt_opt_facebook_opt_1_3b_qa_hf",
                "pt_opt_facebook_opt_125m_qa_hf",
                "pt_opt_facebook_opt_1_3b_seq_cls_hf",
                "pt_opt_facebook_opt_350m_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dtype": "torch.float32"},
        },
    ),
    (
        Cast2,
        [((1, 1, 32, 32), torch.bool)],
        {"model_names": ["pt_bloom_default_clm_hf"], "pcc": 0.99, "args": {"dtype": "torch.int32"}},
    ),
    (
        Cast0,
        [((1, 1, 1, 2048), torch.int64)],
        {
            "model_names": [
                "pt_perceiverio_deepmind_language_perceiver_mlm_hf",
                "onnx_perceiverio_deepmind_language_perceiver_mlm_hf",
            ],
            "pcc": 0.99,
            "args": {"dtype": "torch.float32"},
        },
    ),
    (
        Cast0,
        [((1, 1, 35, 35), torch.int64)],
        {
            "model_names": ["pt_qwen_coder_0_5b_clm_hf", "pt_qwen_coder_1_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"dtype": "torch.float32"},
        },
    ),
    (
        Cast0,
        [((1, 1, 35, 35), torch.bool)],
        {
            "model_names": ["pt_qwen_coder_0_5b_clm_hf", "pt_qwen_coder_1_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"dtype": "torch.float32"},
        },
    ),
    (
        Cast0,
        [((1, 1, 1, 25), torch.int64)],
        {
            "model_names": [
                "pt_stereo_medium_music_generation_hf",
                "pt_stereo_small_music_generation_hf",
                "pt_stereo_large_music_generation_hf",
            ],
            "pcc": 0.99,
            "args": {"dtype": "torch.float32"},
        },
    ),
    (
        Cast0,
        [((1, 25, 1), torch.int64)],
        {
            "model_names": [
                "pt_stereo_medium_music_generation_hf",
                "pt_stereo_small_music_generation_hf",
                "pt_stereo_large_music_generation_hf",
            ],
            "pcc": 0.99,
            "args": {"dtype": "torch.float32"},
        },
    ),
    (
        Cast0,
        [((1, 1, 1, 25), torch.bool)],
        {
            "model_names": [
                "pt_stereo_medium_music_generation_hf",
                "pt_stereo_small_music_generation_hf",
                "pt_stereo_large_music_generation_hf",
            ],
            "pcc": 0.99,
            "args": {"dtype": "torch.float32"},
        },
    ),
    (
        Cast0,
        [((1, 1, 1, 14), torch.int64)],
        {"model_names": ["pd_bert_bert_base_japanese_qa_padlenlp"], "pcc": 0.99, "args": {"dtype": "torch.float32"}},
    ),
    (
        Cast0,
        [((1, 1, 1, 9), torch.int64)],
        {
            "model_names": [
                "pd_bert_bert_base_uncased_qa_padlenlp",
                "pd_ernie_1_0_mlm_padlenlp",
                "pd_ernie_1_0_qa_padlenlp",
                "pd_bert_chinese_roberta_base_mlm_padlenlp",
                "pd_bert_bert_base_uncased_mlm_padlenlp",
                "pd_ernie_1_0_seq_cls_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"dtype": "torch.float32"},
        },
    ),
    (
        Cast0,
        [((1, 1, 7, 7), torch.int64)],
        {
            "model_names": [
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
                "onnx_nanogpt_financialsupport_nanogpt_text_gen_hf",
            ],
            "pcc": 0.99,
            "args": {"dtype": "torch.float32"},
        },
    ),
    (
        Cast0,
        [((1, 1, 7, 7), torch.bool)],
        {
            "model_names": [
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
                "onnx_nanogpt_financialsupport_nanogpt_text_gen_hf",
            ],
            "pcc": 0.99,
            "args": {"dtype": "torch.float32"},
        },
    ),
    (
        Cast2,
        [((1, 128), torch.bool)],
        {
            "model_names": [
                "pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf",
                "onnx_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dtype": "torch.int32"},
        },
    ),
    (
        Cast3,
        [((1, 128), torch.int32)],
        {
            "model_names": [
                "pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf",
                "onnx_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dtype": "torch.int64"},
        },
    ),
    (
        Cast2,
        [((), torch.float32)],
        {
            "model_names": ["pt_surya_ocr_ocr_detection_optical_character_recognition_github"],
            "pcc": 0.99,
            "args": {"dtype": "torch.int32"},
        },
    ),
    (
        Cast0,
        [((1, 1, 5, 5), torch.int64)],
        {
            "model_names": [
                "pt_codegen_salesforce_codegen_350m_mono_clm_hf",
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dtype": "torch.float32"},
        },
    ),
    (
        Cast0,
        [((1, 1, 5, 5), torch.bool)],
        {
            "model_names": [
                "pt_codegen_salesforce_codegen_350m_mono_clm_hf",
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dtype": "torch.float32"},
        },
    ),
    (
        Cast1,
        [((1, 1, 5, 5), torch.bool)],
        {
            "model_names": [
                "pt_gptneo_gpt_neo_1_3b_seq_cls_hf",
                "pt_gptneo_gpt_neo_125m_seq_cls_hf",
                "pt_gptneo_gpt_neo_2_7b_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dtype": "torch.bool"},
        },
    ),
    (
        Cast0,
        [((1, 1, 384, 384), torch.int64)],
        {
            "model_names": [
                "pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf",
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                "pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf",
            ],
            "pcc": 0.99,
            "args": {"dtype": "torch.float32"},
        },
    ),
    (
        Cast0,
        [((1, 1, 384, 384), torch.bool)],
        {
            "model_names": [
                "pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf",
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                "pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf",
            ],
            "pcc": 0.99,
            "args": {"dtype": "torch.float32"},
        },
    ),
    (
        Cast0,
        [((1, 1, 29, 29), torch.int64)],
        {
            "model_names": ["pt_qwen_v2_1_5b_clm_hf", "pt_qwen1_5_0_5b_chat_clm_hf"],
            "pcc": 0.99,
            "args": {"dtype": "torch.float32"},
        },
    ),
    (
        Cast0,
        [((1, 1, 29, 29), torch.bool)],
        {
            "model_names": ["pt_qwen_v2_1_5b_clm_hf", "pt_qwen1_5_0_5b_chat_clm_hf"],
            "pcc": 0.99,
            "args": {"dtype": "torch.float32"},
        },
    ),
    (
        Cast0,
        [((1, 1, 1, 11), torch.int64)],
        {
            "model_names": [
                "pd_albert_chinese_tiny_mlm_padlenlp",
                "pd_bert_chinese_roberta_base_qa_padlenlp",
                "pd_bert_chinese_roberta_base_seq_cls_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"dtype": "torch.float32"},
        },
    ),
    (
        Cast0,
        [((1, 1, 1, 15), torch.int64)],
        {
            "model_names": ["pd_bert_bert_base_japanese_seq_cls_padlenlp"],
            "pcc": 0.99,
            "args": {"dtype": "torch.float32"},
        },
    ),
    (
        Cast0,
        [((1, 1, 1, 8), torch.int64)],
        {
            "model_names": ["pd_bert_bert_base_uncased_seq_cls_padlenlp"],
            "pcc": 0.99,
            "args": {"dtype": "torch.float32"},
        },
    ),
    (
        Cast0,
        [((1, 1, 25, 25), torch.int64)],
        {
            "model_names": [
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader",
            ],
            "pcc": 0.99,
            "args": {"dtype": "torch.float32"},
        },
    ),
    (
        Cast0,
        [((1, 1, 25, 25), torch.bool)],
        {
            "model_names": [
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader",
            ],
            "pcc": 0.99,
            "args": {"dtype": "torch.float32"},
        },
    ),
    (
        Cast3,
        [((1, 11), torch.bool)],
        {"model_names": ["pd_roberta_rbt4_ch_clm_padlenlp"], "pcc": 0.99, "args": {"dtype": "torch.int64"}},
    ),
    (
        Cast2,
        [((1, 11), torch.bool)],
        {"model_names": ["pd_roberta_rbt4_ch_clm_padlenlp"], "pcc": 0.99, "args": {"dtype": "torch.int32"}},
    ),
    (
        Cast0,
        [((1, 11), torch.bool)],
        {"model_names": ["pd_roberta_rbt4_ch_clm_padlenlp"], "pcc": 0.99, "args": {"dtype": "torch.float32"}},
    ),
    (
        Cast1,
        [((1, 11), torch.int64)],
        {"model_names": ["pd_roberta_rbt4_ch_clm_padlenlp"], "pcc": 0.99, "args": {"dtype": "torch.bool"}},
    ),
    (
        Cast1,
        [((1, 11), torch.int32)],
        {"model_names": ["pd_roberta_rbt4_ch_clm_padlenlp"], "pcc": 0.99, "args": {"dtype": "torch.bool"}},
    ),
    (
        Cast0,
        [((1, 1, 6, 6), torch.int64)],
        {
            "model_names": [
                "pt_qwen1_5_0_5b_clm_hf",
                "onnx_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf",
            ],
            "pcc": 0.99,
            "args": {"dtype": "torch.float32"},
        },
    ),
    (
        Cast0,
        [((1, 1, 6, 6), torch.bool)],
        {
            "model_names": [
                "pt_qwen1_5_0_5b_clm_hf",
                "onnx_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf",
            ],
            "pcc": 0.99,
            "args": {"dtype": "torch.float32"},
        },
    ),
    (
        Cast0,
        [((1, 1, 16, 16), torch.int64)],
        {
            "model_names": ["pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf"],
            "pcc": 0.99,
            "args": {"dtype": "torch.float32"},
        },
    ),
    (
        Cast0,
        [((1, 1, 16, 16), torch.bool)],
        {
            "model_names": ["pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf"],
            "pcc": 0.99,
            "args": {"dtype": "torch.float32"},
        },
    ),
    (
        Cast0,
        [((1, 1, 32), torch.int64)],
        {"model_names": ["pt_bloom_default_clm_hf"], "pcc": 0.99, "args": {"dtype": "torch.float32"}},
    ),
    (
        Cast0,
        [((1, 1, 1, 32), torch.int64)],
        {"model_names": ["pt_bloom_default_clm_hf"], "pcc": 0.99, "args": {"dtype": "torch.float32"}},
    ),
    (
        Cast1,
        [((1, 1, 32, 32), torch.float32)],
        {"model_names": ["pt_bloom_default_clm_hf"], "pcc": 0.99, "args": {"dtype": "torch.bool"}},
    ),
    (
        Cast1,
        [((1, 1, 32, 32), torch.int32)],
        {"model_names": ["pt_bloom_default_clm_hf"], "pcc": 0.99, "args": {"dtype": "torch.bool"}},
    ),
    (
        Cast2,
        [((1, 10), torch.bool)],
        {"model_names": ["pt_roberta_xlm_base_mlm_hf"], "pcc": 0.99, "args": {"dtype": "torch.int32"}},
    ),
    (
        Cast3,
        [((1, 10), torch.int32)],
        {"model_names": ["pt_roberta_xlm_base_mlm_hf"], "pcc": 0.99, "args": {"dtype": "torch.int64"}},
    ),
    (
        Cast0,
        [((1, 1, 10, 10), torch.int64)],
        {"model_names": ["pt_roberta_xlm_base_mlm_hf"], "pcc": 0.99, "args": {"dtype": "torch.float32"}},
    ),
    (
        Cast0,
        [((1, 1, 10, 10), torch.bool)],
        {"model_names": ["pt_roberta_xlm_base_mlm_hf"], "pcc": 0.99, "args": {"dtype": "torch.float32"}},
    ),
    (
        Cast0,
        [((2, 1, 7, 7), torch.int64)],
        {
            "model_names": [
                "pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text",
                "onnx_clip_openai_clip_vit_base_patch32_text_gen_hf_text",
            ],
            "pcc": 0.99,
            "args": {"dtype": "torch.float32"},
        },
    ),
    (
        Cast0,
        [((2, 1, 7, 7), torch.bool)],
        {
            "model_names": [
                "pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text",
                "onnx_clip_openai_clip_vit_base_patch32_text_gen_hf_text",
            ],
            "pcc": 0.99,
            "args": {"dtype": "torch.float32"},
        },
    ),
    (
        Cast1,
        [((1, 1, 7, 7), torch.float32)],
        {
            "model_names": ["onnx_nanogpt_financialsupport_nanogpt_text_gen_hf"],
            "pcc": 0.99,
            "args": {"dtype": "torch.bool"},
        },
    ),
    (
        Cast0,
        [((1, 1, 1, 10), torch.int64)],
        {"model_names": ["pd_bert_bert_base_japanese_mlm_padlenlp"], "pcc": 0.99, "args": {"dtype": "torch.float32"}},
    ),
    (
        Cast0,
        [((1, 1, 9, 9), torch.int64)],
        {"model_names": ["pt_albert_imdb_seq_cls_hf"], "pcc": 0.99, "args": {"dtype": "torch.float32"}},
    ),
    (
        Cast0,
        [((1, 1, 9, 9), torch.bool)],
        {"model_names": ["pt_albert_imdb_seq_cls_hf"], "pcc": 0.99, "args": {"dtype": "torch.float32"}},
    ),
    (
        Cast1,
        [((1, 1, 6, 6), torch.float32)],
        {
            "model_names": ["onnx_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf"],
            "pcc": 0.99,
            "args": {"dtype": "torch.bool"},
        },
    ),
    (
        Cast1,
        [((2, 1, 7, 7), torch.float32)],
        {
            "model_names": ["onnx_clip_openai_clip_vit_base_patch32_text_gen_hf_text"],
            "pcc": 0.99,
            "args": {"dtype": "torch.bool"},
        },
    ),
    (
        Cast0,
        [((2, 1, 1, 4), torch.int64)],
        {
            "model_names": ["pd_blip_salesforce_blip_image_captioning_base_img_captioning_padlenlp"],
            "pcc": 0.99,
            "args": {"dtype": "torch.float32"},
        },
    ),
    (
        Cast0,
        [((1, 1, 39, 39), torch.int64)],
        {"model_names": ["pt_qwen_v2_0_5b_instruct_clm_hf"], "pcc": 0.99, "args": {"dtype": "torch.float32"}},
    ),
    (
        Cast0,
        [((1, 1, 39, 39), torch.bool)],
        {"model_names": ["pt_qwen_v2_0_5b_instruct_clm_hf"], "pcc": 0.99, "args": {"dtype": "torch.float32"}},
    ),
    (
        Cast0,
        [((1, 596, 4096), torch.bool)],
        {"model_names": ["pt_llava_1_5_7b_cond_gen_hf"], "pcc": 0.99, "args": {"dtype": "torch.float32"}},
    ),
    (
        Cast1,
        [((2441216,), torch.float32)],
        {"model_names": ["pt_llava_1_5_7b_cond_gen_hf"], "pcc": 0.99, "args": {"dtype": "torch.bool"}},
    ),
    (
        Cast2,
        [((2441216,), torch.float32)],
        {"model_names": ["pt_llava_1_5_7b_cond_gen_hf"], "pcc": 0.99, "args": {"dtype": "torch.int32"}},
    ),
    (
        Cast3,
        [((1, 9), torch.bool)],
        {"model_names": ["pd_roberta_rbt4_ch_seq_cls_padlenlp"], "pcc": 0.99, "args": {"dtype": "torch.int64"}},
    ),
    (
        Cast2,
        [((1, 9), torch.bool)],
        {"model_names": ["pd_roberta_rbt4_ch_seq_cls_padlenlp"], "pcc": 0.99, "args": {"dtype": "torch.int32"}},
    ),
    (
        Cast0,
        [((1, 9), torch.bool)],
        {"model_names": ["pd_roberta_rbt4_ch_seq_cls_padlenlp"], "pcc": 0.99, "args": {"dtype": "torch.float32"}},
    ),
    (
        Cast1,
        [((1, 9), torch.int64)],
        {"model_names": ["pd_roberta_rbt4_ch_seq_cls_padlenlp"], "pcc": 0.99, "args": {"dtype": "torch.bool"}},
    ),
    (
        Cast1,
        [((1, 9), torch.int32)],
        {"model_names": ["pd_roberta_rbt4_ch_seq_cls_padlenlp"], "pcc": 0.99, "args": {"dtype": "torch.bool"}},
    ),
    (
        Cast1,
        [((1, 256, 6, 20), torch.bool)],
        {
            "model_names": [
                "pt_monodepth2_mono_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dtype": "torch.bool"},
        },
    ),
    (
        Cast1,
        [((1, 256, 12, 40), torch.bool)],
        {
            "model_names": [
                "pt_monodepth2_mono_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dtype": "torch.bool"},
        },
    ),
    (
        Cast1,
        [((1, 128, 12, 40), torch.bool)],
        {
            "model_names": [
                "pt_monodepth2_mono_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dtype": "torch.bool"},
        },
    ),
    (
        Cast1,
        [((1, 128, 24, 80), torch.bool)],
        {
            "model_names": [
                "pt_monodepth2_mono_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dtype": "torch.bool"},
        },
    ),
    (
        Cast1,
        [((1, 64, 24, 80), torch.bool)],
        {
            "model_names": [
                "pt_monodepth2_mono_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dtype": "torch.bool"},
        },
    ),
    (
        Cast1,
        [((1, 64, 48, 160), torch.bool)],
        {
            "model_names": [
                "pt_monodepth2_mono_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dtype": "torch.bool"},
        },
    ),
    (
        Cast1,
        [((1, 32, 48, 160), torch.bool)],
        {
            "model_names": [
                "pt_monodepth2_mono_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dtype": "torch.bool"},
        },
    ),
    (
        Cast1,
        [((1, 32, 96, 320), torch.bool)],
        {
            "model_names": [
                "pt_monodepth2_mono_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dtype": "torch.bool"},
        },
    ),
    (
        Cast1,
        [((1, 16, 96, 320), torch.bool)],
        {
            "model_names": [
                "pt_monodepth2_mono_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dtype": "torch.bool"},
        },
    ),
    (
        Cast1,
        [((1, 16, 192, 640), torch.bool)],
        {
            "model_names": [
                "pt_monodepth2_mono_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dtype": "torch.bool"},
        },
    ),
    (
        Cast0,
        [((1, 1, 850, 850), torch.bool)],
        {
            "model_names": ["pt_detr_resnet_50_obj_det_hf", "pt_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"dtype": "torch.float32"},
        },
    ),
    (
        Cast0,
        [((1, 1, 100, 850), torch.bool)],
        {
            "model_names": ["pt_detr_resnet_50_obj_det_hf", "pt_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"dtype": "torch.float32"},
        },
    ),
    (
        Cast1,
        [((1, 25, 34), torch.bfloat16)],
        {
            "model_names": ["pt_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dtype": "torch.bool"},
        },
    ),
    (
        Cast0,
        [((1, 1, 1, 25, 34), torch.bool)],
        {"model_names": ["pt_detr_resnet_50_panoptic_sem_seg_hf"], "pcc": 0.99, "args": {"dtype": "torch.float32"}},
    ),
    (
        Cast0,
        [((1, 1, 356, 356), torch.int64)],
        {
            "model_names": ["pt_gemma_google_gemma_1_1_2b_it_qa_hf", "pt_gemma_google_gemma_1_1_7b_it_qa_hf"],
            "pcc": 0.99,
            "args": {"dtype": "torch.float32"},
        },
    ),
    (
        Cast0,
        [((1, 1, 356, 356), torch.bool)],
        {
            "model_names": ["pt_gemma_google_gemma_1_1_2b_it_qa_hf", "pt_gemma_google_gemma_1_1_7b_it_qa_hf"],
            "pcc": 0.99,
            "args": {"dtype": "torch.float32"},
        },
    ),
    (
        Cast0,
        [((1, 1, 512, 512), torch.int64)],
        {"model_names": ["pt_gemma_google_gemma_2b_text_gen_hf"], "pcc": 0.99, "args": {"dtype": "torch.float32"}},
    ),
    (
        Cast0,
        [((1, 1, 512, 512), torch.bool)],
        {"model_names": ["pt_gemma_google_gemma_2b_text_gen_hf"], "pcc": 0.99, "args": {"dtype": "torch.float32"}},
    ),
    (
        Cast2,
        [((1, 4), torch.bool)],
        {"model_names": ["pt_llama3_huggyllama_7b_seq_cls_hf"], "pcc": 0.99, "args": {"dtype": "torch.int32"}},
    ),
    (
        Cast2,
        [((1, 5), torch.bool)],
        {"model_names": ["pt_phi4_microsoft_phi_4_seq_cls_hf"], "pcc": 0.99, "args": {"dtype": "torch.int32"}},
    ),
    (
        Cast3,
        [((1, 1, 256), torch.bool)],
        {"model_names": ["pt_speecht5_tts_tts_text_to_speech_hf"], "pcc": 0.99, "args": {"dtype": "torch.int64"}},
    ),
    (
        Cast1,
        [((1, 1, 256), torch.bool)],
        {"model_names": ["pt_speecht5_tts_tts_text_to_speech_hf"], "pcc": 0.99, "args": {"dtype": "torch.bool"}},
    ),
    (
        Cast0,
        [((1, 1, 24, 24), torch.int64)],
        {"model_names": ["pt_speecht5_tts_tts_text_to_speech_hf"], "pcc": 0.99, "args": {"dtype": "torch.float32"}},
    ),
    (
        Cast0,
        [((1, 1, 24, 24), torch.bool)],
        {"model_names": ["pt_speecht5_tts_tts_text_to_speech_hf"], "pcc": 0.99, "args": {"dtype": "torch.float32"}},
    ),
    (
        Cast0,
        [((1, 1, 1, 24), torch.int64)],
        {"model_names": ["pt_speecht5_tts_tts_text_to_speech_hf"], "pcc": 0.99, "args": {"dtype": "torch.float32"}},
    ),
    (
        Cast0,
        [((1, 1, 1, 24), torch.bool)],
        {"model_names": ["pt_speecht5_tts_tts_text_to_speech_hf"], "pcc": 0.99, "args": {"dtype": "torch.float32"}},
    ),
]


@pytest.mark.nightly_models_ops
@pytest.mark.parametrize("forge_module_and_shapes_dtypes", forge_modules_and_shapes_dtypes_list, ids=ids_func)
@pytest.mark.parametrize("training_test", [False, True], ids=["inference", "training"])
def test_module(forge_module_and_shapes_dtypes, training_test):

    record_forge_op_name("Cast")

    forge_module, operand_shapes_dtypes, metadata = forge_module_and_shapes_dtypes

    pcc = metadata.get("pcc")

    for metadata_name, metadata_value in metadata.items():
        if metadata_name in ["pcc"]:
            continue
        elif metadata_name == "model_names":
            record_op_model_names(metadata_value)
        elif metadata_name == "args":
            record_forge_op_args(metadata_value)
        else:
            logger.warning(
                "No utility function available in forge property handler to record %s property", metadata_name
            )

    max_int = 1000
    inputs = [
        Tensor.create_from_shape(operand_shape, operand_dtype, max_int=max_int, requires_grad=training_test)
        for operand_shape, operand_dtype in operand_shapes_dtypes
    ]

    framework_model = forge_module(forge_module.__name__)

    for name, parameter in framework_model._parameters.items():
        parameter_tensor = Tensor.create_torch_tensor(
            shape=parameter.shape.get_pytorch_shape(),
            dtype=parameter.pt_data_format,
            max_int=max_int,
            requires_grad=training_test,
        )
        framework_model.set_parameter(name, parameter_tensor)

    for name, constant in framework_model._constants.items():
        constant_tensor = Tensor.create_torch_tensor(
            shape=constant.shape.get_pytorch_shape(),
            dtype=constant.pt_data_format,
            max_int=max_int,
            requires_grad=training_test,
        )
        framework_model.set_constant(name, constant_tensor)

    record_single_op_operands_info(framework_model, inputs)

    compiler_cfg = forge.config.CompilerConfig()
    if "default_df_override" in metadata.keys():
        compiler_cfg.default_df_override = forge.DataFormat.from_json(metadata["default_df_override"])

    compiled_model = compile(framework_model, sample_inputs=inputs, compiler_cfg=compiler_cfg, training=training_test)

    verify(
        inputs,
        framework_model,
        compiled_model,
        with_backward=training_test,
        verify_cfg=VerifyConfig(value_checker=AutomaticValueChecker(pcc=pcc)),
    )
