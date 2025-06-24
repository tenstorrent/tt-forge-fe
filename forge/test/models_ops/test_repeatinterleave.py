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


class Repeatinterleave0(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, repeatinterleave_input_0):
        repeatinterleave_output_1 = forge.op.RepeatInterleave("", repeatinterleave_input_0, repeats=1, dim=0)
        return repeatinterleave_output_1


class Repeatinterleave1(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, repeatinterleave_input_0):
        repeatinterleave_output_1 = forge.op.RepeatInterleave("", repeatinterleave_input_0, repeats=1, dim=1)
        return repeatinterleave_output_1


class Repeatinterleave2(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, repeatinterleave_input_0):
        repeatinterleave_output_1 = forge.op.RepeatInterleave("", repeatinterleave_input_0, repeats=128, dim=2)
        return repeatinterleave_output_1


class Repeatinterleave3(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, repeatinterleave_input_0):
        repeatinterleave_output_1 = forge.op.RepeatInterleave("", repeatinterleave_input_0, repeats=32, dim=2)
        return repeatinterleave_output_1


class Repeatinterleave4(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, repeatinterleave_input_0):
        repeatinterleave_output_1 = forge.op.RepeatInterleave("", repeatinterleave_input_0, repeats=1, dim=2)
        return repeatinterleave_output_1


class Repeatinterleave5(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, repeatinterleave_input_0):
        repeatinterleave_output_1 = forge.op.RepeatInterleave("", repeatinterleave_input_0, repeats=256, dim=2)
        return repeatinterleave_output_1


class Repeatinterleave6(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, repeatinterleave_input_0):
        repeatinterleave_output_1 = forge.op.RepeatInterleave("", repeatinterleave_input_0, repeats=7, dim=2)
        return repeatinterleave_output_1


class Repeatinterleave7(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, repeatinterleave_input_0):
        repeatinterleave_output_1 = forge.op.RepeatInterleave("", repeatinterleave_input_0, repeats=6, dim=2)
        return repeatinterleave_output_1


class Repeatinterleave8(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, repeatinterleave_input_0):
        repeatinterleave_output_1 = forge.op.RepeatInterleave("", repeatinterleave_input_0, repeats=8, dim=2)
        return repeatinterleave_output_1


class Repeatinterleave9(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, repeatinterleave_input_0):
        repeatinterleave_output_1 = forge.op.RepeatInterleave("", repeatinterleave_input_0, repeats=850, dim=2)
        return repeatinterleave_output_1


class Repeatinterleave10(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, repeatinterleave_input_0):
        repeatinterleave_output_1 = forge.op.RepeatInterleave("", repeatinterleave_input_0, repeats=100, dim=2)
        return repeatinterleave_output_1


class Repeatinterleave11(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, repeatinterleave_input_0):
        repeatinterleave_output_1 = forge.op.RepeatInterleave("", repeatinterleave_input_0, repeats=14, dim=2)
        return repeatinterleave_output_1


class Repeatinterleave12(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, repeatinterleave_input_0):
        repeatinterleave_output_1 = forge.op.RepeatInterleave("", repeatinterleave_input_0, repeats=4, dim=2)
        return repeatinterleave_output_1


class Repeatinterleave13(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, repeatinterleave_input_0):
        repeatinterleave_output_1 = forge.op.RepeatInterleave("", repeatinterleave_input_0, repeats=9, dim=2)
        return repeatinterleave_output_1


class Repeatinterleave14(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, repeatinterleave_input_0):
        repeatinterleave_output_1 = forge.op.RepeatInterleave("", repeatinterleave_input_0, repeats=384, dim=2)
        return repeatinterleave_output_1


class Repeatinterleave15(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, repeatinterleave_input_0):
        repeatinterleave_output_1 = forge.op.RepeatInterleave("", repeatinterleave_input_0, repeats=16, dim=2)
        return repeatinterleave_output_1


class Repeatinterleave16(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, repeatinterleave_input_0):
        repeatinterleave_output_1 = forge.op.RepeatInterleave("", repeatinterleave_input_0, repeats=2, dim=2)
        return repeatinterleave_output_1


class Repeatinterleave17(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, repeatinterleave_input_0):
        repeatinterleave_output_1 = forge.op.RepeatInterleave("", repeatinterleave_input_0, repeats=3, dim=2)
        return repeatinterleave_output_1


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (
        Repeatinterleave0,
        [((1, 128), torch.int64)],
        {
            "model_names": [
                "pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf",
                "pt_albert_xlarge_v2_token_cls_hf",
                "pt_albert_xxlarge_v2_token_cls_hf",
                "pt_albert_base_v2_mlm_hf",
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
                "pt_albert_xlarge_v1_mlm_hf",
                "pt_albert_xlarge_v2_mlm_hf",
                "pt_albert_xxlarge_v1_mlm_hf",
                "pt_bert_bert_base_uncased_mlm_hf",
                "pt_albert_large_v1_mlm_hf",
                "pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader",
                "pt_albert_base_v1_token_cls_hf",
                "pt_albert_base_v2_token_cls_hf",
                "pt_albert_large_v1_token_cls_hf",
                "pt_albert_large_v2_mlm_hf",
                "pt_albert_large_v2_token_cls_hf",
                "pt_albert_xlarge_v1_token_cls_hf",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_albert_base_v1_mlm_hf",
                "pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf",
                "pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf",
                "pt_roberta_xlm_roberta_base_mlm_hf",
            ],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave0,
        [((1, 1, 1, 128), torch.int64)],
        {
            "model_names": [
                "pt_distilbert_distilbert_base_cased_mlm_hf",
                "pt_albert_xlarge_v2_token_cls_hf",
                "pt_albert_xxlarge_v2_token_cls_hf",
                "pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder",
                "pt_albert_base_v2_mlm_hf",
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
                "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf",
                "pt_albert_xlarge_v1_mlm_hf",
                "pt_albert_xlarge_v2_mlm_hf",
                "pt_albert_xxlarge_v1_mlm_hf",
                "pt_bert_bert_base_uncased_mlm_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder",
                "pt_albert_large_v1_mlm_hf",
                "pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader",
                "pt_albert_base_v1_token_cls_hf",
                "pt_albert_base_v2_token_cls_hf",
                "pt_albert_large_v1_token_cls_hf",
                "pt_albert_large_v2_mlm_hf",
                "pt_albert_large_v2_token_cls_hf",
                "pt_albert_xlarge_v1_token_cls_hf",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_albert_base_v1_mlm_hf",
                "pt_distilbert_distilbert_base_uncased_mlm_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_roberta_xlm_roberta_base_mlm_hf",
                "pt_distilbert_distilbert_base_multilingual_cased_mlm_hf",
            ],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave1,
        [((1, 1, 1, 128), torch.int64)],
        {
            "model_names": [
                "pt_distilbert_distilbert_base_cased_mlm_hf",
                "pt_albert_xlarge_v2_token_cls_hf",
                "pt_albert_xxlarge_v2_token_cls_hf",
                "pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder",
                "pt_albert_base_v2_mlm_hf",
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
                "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf",
                "pt_albert_xlarge_v1_mlm_hf",
                "pt_albert_xlarge_v2_mlm_hf",
                "pt_albert_xxlarge_v1_mlm_hf",
                "pt_bert_bert_base_uncased_mlm_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder",
                "pt_albert_large_v1_mlm_hf",
                "pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader",
                "pt_albert_base_v1_token_cls_hf",
                "pt_albert_base_v2_token_cls_hf",
                "pt_albert_large_v1_token_cls_hf",
                "pt_albert_large_v2_mlm_hf",
                "pt_albert_large_v2_token_cls_hf",
                "pt_albert_xlarge_v1_token_cls_hf",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_albert_base_v1_mlm_hf",
                "pt_distilbert_distilbert_base_uncased_mlm_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_roberta_xlm_roberta_base_mlm_hf",
                "pt_distilbert_distilbert_base_multilingual_cased_mlm_hf",
            ],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "1"},
        },
    ),
    (
        Repeatinterleave2,
        [((1, 1, 1, 128), torch.int64)],
        {
            "model_names": [
                "pt_distilbert_distilbert_base_cased_mlm_hf",
                "pt_albert_xlarge_v2_token_cls_hf",
                "pt_albert_xxlarge_v2_token_cls_hf",
                "pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder",
                "pt_albert_base_v2_mlm_hf",
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
                "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf",
                "pt_albert_xlarge_v1_mlm_hf",
                "pt_albert_xlarge_v2_mlm_hf",
                "pt_albert_xxlarge_v1_mlm_hf",
                "pt_bert_bert_base_uncased_mlm_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder",
                "pt_albert_large_v1_mlm_hf",
                "pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader",
                "pt_albert_base_v1_token_cls_hf",
                "pt_albert_base_v2_token_cls_hf",
                "pt_albert_large_v1_token_cls_hf",
                "pt_albert_large_v2_mlm_hf",
                "pt_albert_large_v2_token_cls_hf",
                "pt_albert_xlarge_v1_token_cls_hf",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_albert_base_v1_mlm_hf",
                "pt_distilbert_distilbert_base_uncased_mlm_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_roberta_xlm_roberta_base_mlm_hf",
                "pt_distilbert_distilbert_base_multilingual_cased_mlm_hf",
            ],
            "pcc": 0.99,
            "args": {"repeats": "128", "dim": "2"},
        },
    ),
    (
        Repeatinterleave0,
        [((1, 1, 1, 32), torch.int64)],
        {
            "model_names": [
                "pt_opt_facebook_opt_125m_qa_hf",
                "pt_opt_facebook_opt_350m_qa_hf",
                "pt_opt_facebook_opt_125m_seq_cls_hf",
                "pt_opt_facebook_opt_350m_seq_cls_hf",
                "pt_opt_facebook_opt_1_3b_qa_hf",
                "pt_opt_facebook_opt_1_3b_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave1,
        [((1, 1, 1, 32), torch.int64)],
        {
            "model_names": [
                "pt_opt_facebook_opt_125m_qa_hf",
                "pt_opt_facebook_opt_350m_qa_hf",
                "pt_opt_facebook_opt_125m_seq_cls_hf",
                "pt_opt_facebook_opt_350m_seq_cls_hf",
                "pt_opt_facebook_opt_1_3b_qa_hf",
                "pt_opt_facebook_opt_1_3b_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "1"},
        },
    ),
    (
        Repeatinterleave3,
        [((1, 1, 1, 32), torch.int64)],
        {
            "model_names": [
                "pt_opt_facebook_opt_125m_qa_hf",
                "pt_opt_facebook_opt_350m_qa_hf",
                "pt_opt_facebook_opt_125m_seq_cls_hf",
                "pt_opt_facebook_opt_350m_seq_cls_hf",
                "pt_opt_facebook_opt_1_3b_qa_hf",
                "pt_opt_facebook_opt_1_3b_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"repeats": "32", "dim": "2"},
        },
    ),
    (
        Repeatinterleave0,
        [((1, 16, 1), torch.float32)],
        {
            "model_names": [
                "pt_phi1_microsoft_phi_1_clm_hf",
                "pt_phi2_microsoft_phi_2_clm_hf",
                "pt_phi1_microsoft_phi_1_token_cls_hf",
                "pt_phi_1_5_microsoft_phi_1_5_token_cls_hf",
                "pt_phi_1_5_microsoft_phi_1_5_clm_hf",
                "pt_phi1_microsoft_phi_1_seq_cls_hf",
                "pt_fuyu_adept_fuyu_8b_qa_hf",
                "pt_phi2_microsoft_phi_2_pytdml_clm_hf",
                "pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf",
                "pt_phi2_microsoft_phi_2_pytdml_token_cls_hf",
                "pt_phi2_microsoft_phi_2_seq_cls_hf",
                "pt_phi2_microsoft_phi_2_token_cls_hf",
                "pt_phi_1_5_microsoft_phi_1_5_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave4,
        [((1, 16, 1), torch.float32)],
        {
            "model_names": [
                "pt_phi1_microsoft_phi_1_clm_hf",
                "pt_phi2_microsoft_phi_2_clm_hf",
                "pt_phi1_microsoft_phi_1_token_cls_hf",
                "pt_phi_1_5_microsoft_phi_1_5_token_cls_hf",
                "pt_phi_1_5_microsoft_phi_1_5_clm_hf",
                "pt_phi1_microsoft_phi_1_seq_cls_hf",
                "pt_fuyu_adept_fuyu_8b_qa_hf",
                "pt_phi2_microsoft_phi_2_pytdml_clm_hf",
                "pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf",
                "pt_phi2_microsoft_phi_2_pytdml_token_cls_hf",
                "pt_phi2_microsoft_phi_2_seq_cls_hf",
                "pt_phi2_microsoft_phi_2_token_cls_hf",
                "pt_phi_1_5_microsoft_phi_1_5_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "2"},
        },
    ),
    (
        Repeatinterleave0,
        [((1, 32, 1), torch.float32)],
        {
            "model_names": [
                "pt_qwen1_5_qwen_qwen1_5_0_5b_chat_clm_hf",
                "pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf",
                "pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf",
                "pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf",
                "pt_falcon_tiiuae_falcon_7b_instruct_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave4,
        [((1, 32, 1), torch.float32)],
        {
            "model_names": [
                "pt_qwen1_5_qwen_qwen1_5_0_5b_chat_clm_hf",
                "pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf",
                "pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf",
                "pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf",
                "pt_falcon_tiiuae_falcon_7b_instruct_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "2"},
        },
    ),
    (
        Repeatinterleave0,
        [((1, 64, 1), torch.float32)],
        {
            "model_names": [
                "pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf",
                "pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf",
                "pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf",
                "pt_cogito_deepcogito_cogito_v1_preview_llama_3b_text_gen_hf",
                "pt_deepseek_deepseek_math_7b_instruct_qa_hf",
                "pt_llama3_huggyllama_llama_7b_clm_hf",
                "pt_llama3_huggyllama_llama_7b_seq_cls_hf",
                "pt_phi4_microsoft_phi_4_seq_cls_hf",
                "pt_phi4_microsoft_phi_4_token_cls_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_instruct_clm_hf",
                "pt_qwen_v2_qwen_qwen2_7b_token_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_clm_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_3b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_3b_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf",
                "pt_ministral_ministral_ministral_3b_instruct_clm_hf",
                "pt_ministral_mistralai_ministral_8b_instruct_2410_clm_hf",
                "pt_mistral_mistralai_mistral_7b_instruct_v0_3_clm_hf",
                "pt_mistral_mistralai_mistral_7b_v0_1_clm_hf",
                "pt_phi4_microsoft_phi_4_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_3b_instruct_clm_hf",
                "pt_qwen_v2_qwen_qwen2_5_3b_clm_hf",
                "pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm_hf",
                "pt_qwen_v2_qwen_qwen2_5_7b_clm_hf",
                "pt_qwen_v2_qwen_qwen2_5_7b_instruct_1m_clm_hf",
                "pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave4,
        [((1, 64, 1), torch.float32)],
        {
            "model_names": [
                "pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf",
                "pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf",
                "pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf",
                "pt_cogito_deepcogito_cogito_v1_preview_llama_3b_text_gen_hf",
                "pt_deepseek_deepseek_math_7b_instruct_qa_hf",
                "pt_llama3_huggyllama_llama_7b_clm_hf",
                "pt_llama3_huggyllama_llama_7b_seq_cls_hf",
                "pt_phi4_microsoft_phi_4_seq_cls_hf",
                "pt_phi4_microsoft_phi_4_token_cls_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_instruct_clm_hf",
                "pt_qwen_v2_qwen_qwen2_7b_token_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_clm_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_3b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_3b_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf",
                "pt_ministral_ministral_ministral_3b_instruct_clm_hf",
                "pt_ministral_mistralai_ministral_8b_instruct_2410_clm_hf",
                "pt_mistral_mistralai_mistral_7b_instruct_v0_3_clm_hf",
                "pt_mistral_mistralai_mistral_7b_v0_1_clm_hf",
                "pt_phi4_microsoft_phi_4_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_3b_instruct_clm_hf",
                "pt_qwen_v2_qwen_qwen2_5_3b_clm_hf",
                "pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm_hf",
                "pt_qwen_v2_qwen_qwen2_5_7b_clm_hf",
                "pt_qwen_v2_qwen_qwen2_5_7b_instruct_1m_clm_hf",
                "pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "2"},
        },
    ),
    (
        Repeatinterleave0,
        [((1, 1, 1, 256), torch.float32)],
        {"model_names": ["pt_gpt_gpt2_text_gen_hf"], "pcc": 0.99, "args": {"repeats": "1", "dim": "0"}},
    ),
    (
        Repeatinterleave1,
        [((1, 1, 1, 256), torch.float32)],
        {"model_names": ["pt_gpt_gpt2_text_gen_hf"], "pcc": 0.99, "args": {"repeats": "1", "dim": "1"}},
    ),
    (
        Repeatinterleave5,
        [((1, 1, 1, 256), torch.float32)],
        {"model_names": ["pt_gpt_gpt2_text_gen_hf"], "pcc": 0.99, "args": {"repeats": "256", "dim": "2"}},
    ),
    (
        Repeatinterleave0,
        [((1, 1, 1, 7), torch.int64)],
        {
            "model_names": [
                "pt_gpt_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
            ],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave1,
        [((1, 1, 1, 7), torch.int64)],
        {
            "model_names": [
                "pt_gpt_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
            ],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "1"},
        },
    ),
    (
        Repeatinterleave6,
        [((1, 1, 1, 7), torch.int64)],
        {
            "model_names": [
                "pt_gpt_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
            ],
            "pcc": 0.99,
            "args": {"repeats": "7", "dim": "2"},
        },
    ),
    (
        Repeatinterleave0,
        [((1, 2, 1, 39, 64), torch.float32)],
        {
            "model_names": ["pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf"],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave6,
        [((1, 2, 1, 39, 64), torch.float32)],
        {
            "model_names": ["pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf"],
            "pcc": 0.99,
            "args": {"repeats": "7", "dim": "2"},
        },
    ),
    (
        Repeatinterleave0,
        [((1, 2, 1, 29, 128), torch.float32)],
        {
            "model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf", "pt_qwen_v2_qwen_qwen2_5_3b_clm_hf"],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave7,
        [((1, 2, 1, 29, 128), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99, "args": {"repeats": "6", "dim": "2"}},
    ),
    (
        Repeatinterleave8,
        [((1, 2, 1, 29, 128), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_3b_clm_hf"], "pcc": 0.99, "args": {"repeats": "8", "dim": "2"}},
    ),
    (
        Repeatinterleave0,
        [((1, 2, 1, 39, 128), torch.float32)],
        {
            "model_names": [
                "pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf",
                "pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave7,
        [((1, 2, 1, 39, 128), torch.float32)],
        {
            "model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf"],
            "pcc": 0.99,
            "args": {"repeats": "6", "dim": "2"},
        },
    ),
    (
        Repeatinterleave8,
        [((1, 2, 1, 39, 128), torch.float32)],
        {
            "model_names": ["pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm_hf"],
            "pcc": 0.99,
            "args": {"repeats": "8", "dim": "2"},
        },
    ),
    (
        Repeatinterleave0,
        [((1, 1, 1024), torch.bfloat16)],
        {
            "model_names": [
                "pt_vit_vit_l_32_img_cls_torchvision",
                "pt_beit_microsoft_beit_large_patch16_224_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf",
                "pt_vit_vit_l_16_img_cls_torchvision",
                "pt_vit_google_vit_large_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave0,
        [((1, 1, 768), torch.bfloat16)],
        {
            "model_names": [
                "pt_beit_microsoft_beit_base_patch16_224_img_cls_hf",
                "pt_mgp_alibaba_damo_mgp_str_base_scene_text_recognition_hf",
                "pt_deit_facebook_deit_base_distilled_patch16_224_img_cls_hf",
                "pt_vit_vit_b_32_img_cls_torchvision",
                "pt_vit_vit_b_16_img_cls_torchvision",
                "pt_deit_facebook_deit_base_patch16_224_img_cls_hf",
                "pt_vit_google_vit_base_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave0,
        [((1, 1, 384), torch.bfloat16)],
        {
            "model_names": ["pt_deit_facebook_deit_small_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave0,
        [((1, 1, 1, 850), torch.bfloat16)],
        {
            "model_names": [
                "pt_detr_facebook_detr_resnet_50_obj_det_hf",
                "pt_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave1,
        [((1, 1, 1, 850), torch.bfloat16)],
        {
            "model_names": [
                "pt_detr_facebook_detr_resnet_50_obj_det_hf",
                "pt_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "1"},
        },
    ),
    (
        Repeatinterleave9,
        [((1, 1, 1, 850), torch.bfloat16)],
        {
            "model_names": [
                "pt_detr_facebook_detr_resnet_50_obj_det_hf",
                "pt_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"repeats": "850", "dim": "2"},
        },
    ),
    (
        Repeatinterleave10,
        [((1, 1, 1, 850), torch.bfloat16)],
        {
            "model_names": [
                "pt_detr_facebook_detr_resnet_50_obj_det_hf",
                "pt_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"repeats": "100", "dim": "2"},
        },
    ),
    (
        Repeatinterleave0,
        [((1, 512, 1024), torch.bfloat16)],
        {
            "model_names": [
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave0,
        [((1, 50176, 256), torch.bfloat16)],
        {
            "model_names": ["pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf"],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave0,
        [((1, 2, 1, 35, 128), torch.float32)],
        {
            "model_names": [
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave7,
        [((1, 2, 1, 35, 128), torch.float32)],
        {
            "model_names": [
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"repeats": "6", "dim": "2"},
        },
    ),
    (
        Repeatinterleave8,
        [((1, 2, 1, 35, 128), torch.float32)],
        {
            "model_names": [
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"repeats": "8", "dim": "2"},
        },
    ),
    (
        Repeatinterleave0,
        [((1, 1, 1, 256), torch.int64)],
        {
            "model_names": [
                "pt_xglm_facebook_xglm_564m_clm_hf",
                "pt_bart_facebook_bart_large_mnli_seq_cls_hf",
                "pt_opt_facebook_opt_1_3b_clm_hf",
                "pt_xglm_facebook_xglm_1_7b_clm_hf",
                "pt_opt_facebook_opt_125m_clm_hf",
                "pt_phi3_5_microsoft_phi_3_5_mini_instruct_clm_hf",
                "pt_opt_facebook_opt_350m_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave1,
        [((1, 1, 1, 256), torch.int64)],
        {
            "model_names": [
                "pt_xglm_facebook_xglm_564m_clm_hf",
                "pt_bart_facebook_bart_large_mnli_seq_cls_hf",
                "pt_opt_facebook_opt_1_3b_clm_hf",
                "pt_xglm_facebook_xglm_1_7b_clm_hf",
                "pt_opt_facebook_opt_125m_clm_hf",
                "pt_phi3_5_microsoft_phi_3_5_mini_instruct_clm_hf",
                "pt_opt_facebook_opt_350m_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "1"},
        },
    ),
    (
        Repeatinterleave5,
        [((1, 1, 1, 256), torch.int64)],
        {
            "model_names": [
                "pt_xglm_facebook_xglm_564m_clm_hf",
                "pt_bart_facebook_bart_large_mnli_seq_cls_hf",
                "pt_opt_facebook_opt_1_3b_clm_hf",
                "pt_xglm_facebook_xglm_1_7b_clm_hf",
                "pt_opt_facebook_opt_125m_clm_hf",
                "pt_phi3_5_microsoft_phi_3_5_mini_instruct_clm_hf",
                "pt_opt_facebook_opt_350m_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"repeats": "256", "dim": "2"},
        },
    ),
    (
        Repeatinterleave0,
        [((1, 14), torch.int64)],
        {
            "model_names": ["pt_albert_twmkn9_albert_base_v2_squad2_qa_hf"],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave0,
        [((1, 1, 1, 14), torch.int64)],
        {
            "model_names": ["pt_albert_twmkn9_albert_base_v2_squad2_qa_hf"],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave1,
        [((1, 1, 1, 14), torch.int64)],
        {
            "model_names": ["pt_albert_twmkn9_albert_base_v2_squad2_qa_hf"],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "1"},
        },
    ),
    (
        Repeatinterleave11,
        [((1, 1, 1, 14), torch.int64)],
        {
            "model_names": ["pt_albert_twmkn9_albert_base_v2_squad2_qa_hf"],
            "pcc": 0.99,
            "args": {"repeats": "14", "dim": "2"},
        },
    ),
    (
        Repeatinterleave1,
        [((2, 1, 1, 7), torch.int64)],
        {
            "model_names": ["pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text"],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "1"},
        },
    ),
    (
        Repeatinterleave6,
        [((2, 1, 1, 7), torch.int64)],
        {
            "model_names": ["pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text"],
            "pcc": 0.99,
            "args": {"repeats": "7", "dim": "2"},
        },
    ),
    (
        Repeatinterleave0,
        [((1, 1, 192), torch.bfloat16)],
        {
            "model_names": [
                "pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf",
                "pt_yolos_hustvl_yolos_tiny_obj_det_hf",
            ],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave0,
        [((1, 8, 1, 4, 64), torch.float32)],
        {
            "model_names": [
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave12,
        [((1, 8, 1, 4, 64), torch.float32)],
        {
            "model_names": [
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"repeats": "4", "dim": "2"},
        },
    ),
    (
        Repeatinterleave0,
        [((1, 2, 1, 29, 64), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"], "pcc": 0.99, "args": {"repeats": "1", "dim": "0"}},
    ),
    (
        Repeatinterleave6,
        [((1, 2, 1, 29, 64), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"], "pcc": 0.99, "args": {"repeats": "7", "dim": "2"}},
    ),
    (
        Repeatinterleave1,
        [((2, 1, 1, 13), torch.int64)],
        {
            "model_names": [
                "pt_stereo_facebook_musicgen_small_music_generation_hf",
                "pt_stereo_facebook_musicgen_large_music_generation_hf",
                "pt_stereo_facebook_musicgen_medium_music_generation_hf",
            ],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "1"},
        },
    ),
    (
        Repeatinterleave4,
        [((2, 1, 1, 13), torch.int64)],
        {
            "model_names": [
                "pt_stereo_facebook_musicgen_small_music_generation_hf",
                "pt_stereo_facebook_musicgen_large_music_generation_hf",
                "pt_stereo_facebook_musicgen_medium_music_generation_hf",
            ],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "2"},
        },
    ),
    (
        Repeatinterleave0,
        [((1, 9), torch.int64)],
        {
            "model_names": ["pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave0,
        [((1, 1, 1, 9), torch.int64)],
        {
            "model_names": ["pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave1,
        [((1, 1, 1, 9), torch.int64)],
        {
            "model_names": ["pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "1"},
        },
    ),
    (
        Repeatinterleave13,
        [((1, 1, 1, 9), torch.int64)],
        {
            "model_names": ["pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"repeats": "9", "dim": "2"},
        },
    ),
    (
        Repeatinterleave0,
        [((1, 384), torch.int64)],
        {
            "model_names": [
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                "pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf",
            ],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave0,
        [((1, 1, 1, 384), torch.int64)],
        {
            "model_names": [
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                "pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf",
                "pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf",
            ],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave1,
        [((1, 1, 1, 384), torch.int64)],
        {
            "model_names": [
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                "pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf",
                "pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf",
            ],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "1"},
        },
    ),
    (
        Repeatinterleave14,
        [((1, 1, 1, 384), torch.int64)],
        {
            "model_names": [
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                "pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf",
                "pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf",
            ],
            "pcc": 0.99,
            "args": {"repeats": "384", "dim": "2"},
        },
    ),
    (
        Repeatinterleave0,
        [((1, 8, 1, 256, 64), torch.float32)],
        {
            "model_names": [
                "pt_llama3_meta_llama_llama_3_2_1b_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave12,
        [((1, 8, 1, 256, 64), torch.float32)],
        {
            "model_names": [
                "pt_llama3_meta_llama_llama_3_2_1b_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"repeats": "4", "dim": "2"},
        },
    ),
    (
        Repeatinterleave0,
        [((1, 2, 1, 35, 64), torch.float32)],
        {
            "model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave6,
        [((1, 2, 1, 35, 64), torch.float32)],
        {
            "model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"repeats": "7", "dim": "2"},
        },
    ),
    (
        Repeatinterleave0,
        [((1, 16), torch.int64)],
        {
            "model_names": ["pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf"],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave0,
        [((1, 1, 1, 16), torch.int64)],
        {
            "model_names": ["pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf"],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave1,
        [((1, 1, 1, 16), torch.int64)],
        {
            "model_names": ["pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf"],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "1"},
        },
    ),
    (
        Repeatinterleave15,
        [((1, 1, 1, 16), torch.int64)],
        {
            "model_names": ["pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf"],
            "pcc": 0.99,
            "args": {"repeats": "16", "dim": "2"},
        },
    ),
    (
        Repeatinterleave0,
        [((1, 1, 1280), torch.bfloat16)],
        {"model_names": ["pt_vit_vit_h_14_img_cls_torchvision"], "pcc": 0.99, "args": {"repeats": "1", "dim": "0"}},
    ),
    (
        Repeatinterleave0,
        [((1, 100, 192), torch.bfloat16)],
        {"model_names": ["pt_yolos_hustvl_yolos_tiny_obj_det_hf"], "pcc": 0.99, "args": {"repeats": "1", "dim": "0"}},
    ),
    (
        Repeatinterleave0,
        [((1, 128, 1), torch.float32)],
        {
            "model_names": [
                "pt_falcon3_tiiuae_falcon3_1b_base_clm_hf",
                "pt_gemma_google_gemma_2_2b_it_qa_hf",
                "pt_gemma_google_gemma_1_1_2b_it_qa_hf",
                "pt_gemma_google_gemma_1_1_7b_it_qa_hf",
                "pt_falcon3_tiiuae_falcon3_3b_base_clm_hf",
                "pt_falcon3_tiiuae_falcon3_7b_base_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave4,
        [((1, 128, 1), torch.float32)],
        {
            "model_names": [
                "pt_falcon3_tiiuae_falcon3_1b_base_clm_hf",
                "pt_gemma_google_gemma_2_2b_it_qa_hf",
                "pt_gemma_google_gemma_1_1_2b_it_qa_hf",
                "pt_gemma_google_gemma_1_1_7b_it_qa_hf",
                "pt_falcon3_tiiuae_falcon3_3b_base_clm_hf",
                "pt_falcon3_tiiuae_falcon3_7b_base_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "2"},
        },
    ),
    (
        Repeatinterleave0,
        [((1, 4, 1, 522, 256), torch.float32)],
        {
            "model_names": [
                "pt_falcon3_tiiuae_falcon3_1b_base_clm_hf",
                "pt_falcon3_tiiuae_falcon3_3b_base_clm_hf",
                "pt_falcon3_tiiuae_falcon3_7b_base_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave16,
        [((1, 4, 1, 522, 256), torch.float32)],
        {
            "model_names": ["pt_falcon3_tiiuae_falcon3_1b_base_clm_hf"],
            "pcc": 0.99,
            "args": {"repeats": "2", "dim": "2"},
        },
    ),
    (
        Repeatinterleave17,
        [((1, 4, 1, 522, 256), torch.float32)],
        {
            "model_names": ["pt_falcon3_tiiuae_falcon3_3b_base_clm_hf", "pt_falcon3_tiiuae_falcon3_7b_base_clm_hf"],
            "pcc": 0.99,
            "args": {"repeats": "3", "dim": "2"},
        },
    ),
    (
        Repeatinterleave0,
        [((1, 4, 1, 207, 256), torch.float32)],
        {"model_names": ["pt_gemma_google_gemma_2_2b_it_qa_hf"], "pcc": 0.99, "args": {"repeats": "1", "dim": "0"}},
    ),
    (
        Repeatinterleave16,
        [((1, 4, 1, 207, 256), torch.float32)],
        {"model_names": ["pt_gemma_google_gemma_2_2b_it_qa_hf"], "pcc": 0.99, "args": {"repeats": "2", "dim": "2"}},
    ),
    (
        Repeatinterleave0,
        [((1, 8, 1, 44, 128), torch.float32)],
        {
            "model_names": ["pt_cogito_deepcogito_cogito_v1_preview_llama_3b_text_gen_hf"],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave17,
        [((1, 8, 1, 44, 128), torch.float32)],
        {
            "model_names": ["pt_cogito_deepcogito_cogito_v1_preview_llama_3b_text_gen_hf"],
            "pcc": 0.99,
            "args": {"repeats": "3", "dim": "2"},
        },
    ),
    (
        Repeatinterleave0,
        [((1, 48, 1), torch.float32)],
        {
            "model_names": [
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave4,
        [((1, 48, 1), torch.float32)],
        {
            "model_names": [
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "2"},
        },
    ),
    (
        Repeatinterleave0,
        [((1, 10, 1, 256, 128), torch.float32)],
        {"model_names": ["pt_phi4_microsoft_phi_4_seq_cls_hf"], "pcc": 0.99, "args": {"repeats": "1", "dim": "0"}},
    ),
    (
        Repeatinterleave12,
        [((1, 10, 1, 256, 128), torch.float32)],
        {"model_names": ["pt_phi4_microsoft_phi_4_seq_cls_hf"], "pcc": 0.99, "args": {"repeats": "4", "dim": "2"}},
    ),
    (
        Repeatinterleave0,
        [((1, 10, 1, 12, 128), torch.float32)],
        {"model_names": ["pt_phi4_microsoft_phi_4_token_cls_hf"], "pcc": 0.99, "args": {"repeats": "1", "dim": "0"}},
    ),
    (
        Repeatinterleave12,
        [((1, 10, 1, 12, 128), torch.float32)],
        {"model_names": ["pt_phi4_microsoft_phi_4_token_cls_hf"], "pcc": 0.99, "args": {"repeats": "4", "dim": "2"}},
    ),
    (
        Repeatinterleave0,
        [((1, 4, 1, 35, 128), torch.float32)],
        {
            "model_names": [
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave6,
        [((1, 4, 1, 35, 128), torch.float32)],
        {
            "model_names": [
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"repeats": "7", "dim": "2"},
        },
    ),
    (
        Repeatinterleave0,
        [((1, 4, 1, 13, 128), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_7b_token_cls_hf"], "pcc": 0.99, "args": {"repeats": "1", "dim": "0"}},
    ),
    (
        Repeatinterleave6,
        [((1, 4, 1, 13, 128), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_7b_token_cls_hf"], "pcc": 0.99, "args": {"repeats": "7", "dim": "2"}},
    ),
    (
        Repeatinterleave0,
        [((1, 1, 1, 107, 256), torch.float32)],
        {"model_names": ["pt_gemma_google_gemma_1_1_2b_it_qa_hf"], "pcc": 0.99, "args": {"repeats": "1", "dim": "0"}},
    ),
    (
        Repeatinterleave1,
        [((1, 1, 1, 107, 256), torch.float32)],
        {"model_names": ["pt_gemma_google_gemma_1_1_2b_it_qa_hf"], "pcc": 0.99, "args": {"repeats": "1", "dim": "1"}},
    ),
    (
        Repeatinterleave8,
        [((1, 1, 1, 107, 256), torch.float32)],
        {"model_names": ["pt_gemma_google_gemma_1_1_2b_it_qa_hf"], "pcc": 0.99, "args": {"repeats": "8", "dim": "2"}},
    ),
    (
        Repeatinterleave0,
        [((1, 8, 1, 256, 128), torch.float32)],
        {
            "model_names": [
                "pt_llama3_meta_llama_llama_3_1_8b_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_3b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave12,
        [((1, 8, 1, 256, 128), torch.float32)],
        {
            "model_names": ["pt_llama3_meta_llama_llama_3_1_8b_clm_hf"],
            "pcc": 0.99,
            "args": {"repeats": "4", "dim": "2"},
        },
    ),
    (
        Repeatinterleave17,
        [((1, 8, 1, 256, 128), torch.float32)],
        {
            "model_names": ["pt_llama3_meta_llama_llama_3_2_3b_instruct_clm_hf"],
            "pcc": 0.99,
            "args": {"repeats": "3", "dim": "2"},
        },
    ),
    (
        Repeatinterleave0,
        [((1, 8, 1, 4, 128), torch.float32)],
        {
            "model_names": [
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_3b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_3b_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave12,
        [((1, 8, 1, 4, 128), torch.float32)],
        {
            "model_names": [
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"repeats": "4", "dim": "2"},
        },
    ),
    (
        Repeatinterleave17,
        [((1, 8, 1, 4, 128), torch.float32)],
        {
            "model_names": [
                "pt_llama3_meta_llama_llama_3_2_3b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_3b_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"repeats": "3", "dim": "2"},
        },
    ),
    (
        Repeatinterleave0,
        [((1, 8, 1, 10, 128), torch.float32)],
        {
            "model_names": ["pt_ministral_ministral_ministral_3b_instruct_clm_hf"],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave12,
        [((1, 8, 1, 10, 128), torch.float32)],
        {
            "model_names": ["pt_ministral_ministral_ministral_3b_instruct_clm_hf"],
            "pcc": 0.99,
            "args": {"repeats": "4", "dim": "2"},
        },
    ),
    (
        Repeatinterleave0,
        [((1, 8, 1, 8, 128), torch.float32)],
        {
            "model_names": ["pt_ministral_mistralai_ministral_8b_instruct_2410_clm_hf"],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave12,
        [((1, 8, 1, 8, 128), torch.float32)],
        {
            "model_names": ["pt_ministral_mistralai_ministral_8b_instruct_2410_clm_hf"],
            "pcc": 0.99,
            "args": {"repeats": "4", "dim": "2"},
        },
    ),
    (
        Repeatinterleave0,
        [((1, 8, 1, 135, 128), torch.float32)],
        {
            "model_names": ["pt_mistral_mistralai_mistral_7b_instruct_v0_3_clm_hf"],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave12,
        [((1, 8, 1, 135, 128), torch.float32)],
        {
            "model_names": ["pt_mistral_mistralai_mistral_7b_instruct_v0_3_clm_hf"],
            "pcc": 0.99,
            "args": {"repeats": "4", "dim": "2"},
        },
    ),
    (
        Repeatinterleave0,
        [((1, 8, 1, 128, 128), torch.float32)],
        {
            "model_names": ["pt_mistral_mistralai_mistral_7b_v0_1_clm_hf"],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave12,
        [((1, 8, 1, 128, 128), torch.float32)],
        {
            "model_names": ["pt_mistral_mistralai_mistral_7b_v0_1_clm_hf"],
            "pcc": 0.99,
            "args": {"repeats": "4", "dim": "2"},
        },
    ),
    (
        Repeatinterleave0,
        [((1, 10, 1, 6, 128), torch.float32)],
        {"model_names": ["pt_phi4_microsoft_phi_4_clm_hf"], "pcc": 0.99, "args": {"repeats": "1", "dim": "0"}},
    ),
    (
        Repeatinterleave12,
        [((1, 10, 1, 6, 128), torch.float32)],
        {"model_names": ["pt_phi4_microsoft_phi_4_clm_hf"], "pcc": 0.99, "args": {"repeats": "4", "dim": "2"}},
    ),
    (
        Repeatinterleave0,
        [((1, 2048, 768), torch.float32)],
        {
            "model_names": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave0,
        [((1, 256, 1280), torch.float32)],
        {
            "model_names": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave0,
        [((1, 4, 1, 29, 128), torch.float32)],
        {
            "model_names": ["pt_qwen_v2_qwen_qwen2_5_7b_clm_hf", "pt_qwen_v2_qwen_qwen2_5_7b_instruct_1m_clm_hf"],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave6,
        [((1, 4, 1, 29, 128), torch.float32)],
        {
            "model_names": ["pt_qwen_v2_qwen_qwen2_5_7b_clm_hf", "pt_qwen_v2_qwen_qwen2_5_7b_instruct_1m_clm_hf"],
            "pcc": 0.99,
            "args": {"repeats": "7", "dim": "2"},
        },
    ),
    (
        Repeatinterleave0,
        [((1, 4, 1, 39, 128), torch.float32)],
        {
            "model_names": ["pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf"],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave6,
        [((1, 4, 1, 39, 128), torch.float32)],
        {
            "model_names": ["pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf"],
            "pcc": 0.99,
            "args": {"repeats": "7", "dim": "2"},
        },
    ),
]


@pytest.mark.nightly_models_ops
@pytest.mark.parametrize("forge_module_and_shapes_dtypes", forge_modules_and_shapes_dtypes_list, ids=ids_func)
def test_module(forge_module_and_shapes_dtypes):

    record_forge_op_name("RepeatInterleave")

    forge_module, operand_shapes_dtypes, metadata = forge_module_and_shapes_dtypes

    pcc = metadata.pop("pcc")

    for metadata_name, metadata_value in metadata.items():
        if metadata_name == "model_names":
            record_op_model_names(metadata_value)
        elif metadata_name == "args":
            record_forge_op_args(metadata_value)
        else:
            logger.warning(
                "No utility function available in forge property handler to record %s property", metadata_name
            )

    max_int = 1000
    inputs = [
        Tensor.create_from_shape(operand_shape, operand_dtype, max_int=max_int)
        for operand_shape, operand_dtype in operand_shapes_dtypes
    ]

    framework_model = forge_module(forge_module.__name__)
    framework_model.process_framework_parameters()

    for name, parameter in framework_model._parameters.items():
        parameter_tensor = Tensor.create_torch_tensor(
            shape=parameter.shape.get_pytorch_shape(), dtype=parameter.pt_data_format, max_int=max_int
        )
        framework_model.set_parameter(name, parameter_tensor)

    for name, constant in framework_model._constants.items():
        constant_tensor = Tensor.create_torch_tensor(
            shape=constant.shape.get_pytorch_shape(), dtype=constant.pt_data_format, max_int=max_int
        )
        framework_model.set_constant(name, constant_tensor)

    record_single_op_operands_info(framework_model, inputs)

    compiler_cfg = forge.config.CompilerConfig()
    if "default_df_override" in metadata.keys():
        compiler_cfg.default_df_override = forge.DataFormat.from_json(metadata["default_df_override"])

    compiled_model = compile(framework_model, sample_inputs=inputs, compiler_cfg=compiler_cfg)

    verify(inputs, framework_model, compiled_model, VerifyConfig(value_checker=AutomaticValueChecker(pcc=pcc)))
