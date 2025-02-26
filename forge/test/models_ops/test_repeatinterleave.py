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
from forge.verify.value_checkers import AutomaticValueChecker
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
        repeatinterleave_output_1 = forge.op.RepeatInterleave("", repeatinterleave_input_0, repeats=3, dim=2)
        return repeatinterleave_output_1


class Repeatinterleave6(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, repeatinterleave_input_0):
        repeatinterleave_output_1 = forge.op.RepeatInterleave("", repeatinterleave_input_0, repeats=2, dim=2)
        return repeatinterleave_output_1


class Repeatinterleave7(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, repeatinterleave_input_0):
        repeatinterleave_output_1 = forge.op.RepeatInterleave("", repeatinterleave_input_0, repeats=8, dim=2)
        return repeatinterleave_output_1


class Repeatinterleave8(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, repeatinterleave_input_0):
        repeatinterleave_output_1 = forge.op.RepeatInterleave("", repeatinterleave_input_0, repeats=4, dim=2)
        return repeatinterleave_output_1


class Repeatinterleave9(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, repeatinterleave_input_0):
        repeatinterleave_output_1 = forge.op.RepeatInterleave("", repeatinterleave_input_0, repeats=32, dim=2)
        return repeatinterleave_output_1


class Repeatinterleave10(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, repeatinterleave_input_0):
        repeatinterleave_output_1 = forge.op.RepeatInterleave("", repeatinterleave_input_0, repeats=6, dim=2)
        return repeatinterleave_output_1


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    pytest.param(
        (
            Repeatinterleave0,
            [((2, 1, 1, 13), torch.int64)],
            {
                "model_name": [
                    "pt_stereo_facebook_musicgen_large_music_generation_hf",
                    "pt_stereo_facebook_musicgen_medium_music_generation_hf",
                    "pt_stereo_facebook_musicgen_small_music_generation_hf",
                ],
                "pcc": 0.99,
                "op_params": {"repeats": "1", "dim": "1"},
            },
        ),
        marks=[
            pytest.mark.xfail(
                reason="TypeError: Dtype mismatch: framework_model.dtype=torch.int64, compiled_model.dtype=torch.int32"
            )
        ],
    ),
    pytest.param(
        (
            Repeatinterleave1,
            [((2, 1, 1, 13), torch.int64)],
            {
                "model_name": [
                    "pt_stereo_facebook_musicgen_large_music_generation_hf",
                    "pt_stereo_facebook_musicgen_medium_music_generation_hf",
                    "pt_stereo_facebook_musicgen_small_music_generation_hf",
                ],
                "pcc": 0.99,
                "op_params": {"repeats": "1", "dim": "2"},
            },
        ),
        marks=[
            pytest.mark.xfail(
                reason="TypeError: Dtype mismatch: framework_model.dtype=torch.int64, compiled_model.dtype=torch.int32"
            )
        ],
    ),
    pytest.param(
        (
            Repeatinterleave0,
            [((2, 1, 1, 7), torch.int64)],
            {
                "model_name": ["pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text"],
                "pcc": 0.99,
                "op_params": {"repeats": "1", "dim": "1"},
            },
        ),
        marks=[
            pytest.mark.xfail(
                reason="TypeError: Dtype mismatch: framework_model.dtype=torch.int64, compiled_model.dtype=torch.int32"
            )
        ],
    ),
    pytest.param(
        (
            Repeatinterleave2,
            [((2, 1, 1, 7), torch.int64)],
            {
                "model_name": ["pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text"],
                "pcc": 0.99,
                "op_params": {"repeats": "7", "dim": "2"},
            },
        ),
        marks=[
            pytest.mark.xfail(
                reason="TypeError: Dtype mismatch: framework_model.dtype=torch.int64, compiled_model.dtype=torch.int32"
            )
        ],
    ),
    (
        Repeatinterleave3,
        [((1, 64, 1), torch.float32)],
        {
            "model_name": [
                "pt_deepseek_deepseek_math_7b_instruct_qa_hf",
                "DeepSeekWrapper_decoder",
                "pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf",
                "pt_mistral_mistralai_mistral_7b_v0_1_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_instruct_clm_hf",
                "pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf",
                "pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf",
                "pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf",
                "pt_qwen_v2_qwen_qwen2_5_7b_clm_hf",
                "pt_qwen_v2_qwen_qwen2_5_3b_clm_hf",
                "pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave1,
        [((1, 64, 1), torch.float32)],
        {
            "model_name": [
                "pt_deepseek_deepseek_math_7b_instruct_qa_hf",
                "DeepSeekWrapper_decoder",
                "pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf",
                "pt_mistral_mistralai_mistral_7b_v0_1_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_instruct_clm_hf",
                "pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf",
                "pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf",
                "pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf",
                "pt_qwen_v2_qwen_qwen2_5_7b_clm_hf",
                "pt_qwen_v2_qwen_qwen2_5_3b_clm_hf",
                "pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"repeats": "1", "dim": "2"},
        },
    ),
    pytest.param(
        (
            Repeatinterleave3,
            [((1, 128), torch.int64)],
            {
                "model_name": [
                    "pt_albert_xxlarge_v1_token_cls_hf",
                    "pt_albert_base_v2_token_cls_hf",
                    "pt_albert_xxlarge_v2_token_cls_hf",
                    "pt_albert_large_v1_token_cls_hf",
                    "pt_albert_large_v2_token_cls_hf",
                    "pt_albert_base_v1_token_cls_hf",
                    "pt_albert_base_v1_mlm_hf",
                    "pt_albert_xlarge_v2_token_cls_hf",
                    "pt_albert_xxlarge_v2_mlm_hf",
                    "pt_albert_large_v2_mlm_hf",
                    "pt_albert_base_v2_mlm_hf",
                    "pt_albert_xlarge_v1_token_cls_hf",
                    "pt_albert_xlarge_v1_mlm_hf",
                    "pt_albert_large_v1_mlm_hf",
                    "pt_albert_xxlarge_v1_mlm_hf",
                    "pt_albert_xlarge_v2_mlm_hf",
                    "pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf",
                    "pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf",
                    "pt_bert_bert_base_uncased_mlm_hf",
                    "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                    "pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader",
                    "pt_roberta_xlm_roberta_base_mlm_hf",
                    "pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf",
                ],
                "pcc": 0.99,
                "op_params": {"repeats": "1", "dim": "0"},
            },
        ),
        marks=[
            pytest.mark.xfail(
                reason="TypeError: Dtype mismatch: framework_model.dtype=torch.int64, compiled_model.dtype=torch.int32"
            )
        ],
    ),
    pytest.param(
        (
            Repeatinterleave3,
            [((1, 1, 1, 256), torch.int64)],
            {
                "model_name": [
                    "pt_bart_facebook_bart_large_mnli_seq_cls_hf",
                    "pt_opt_facebook_opt_1_3b_clm_hf",
                    "pt_opt_facebook_opt_125m_clm_hf",
                    "pt_opt_facebook_opt_350m_clm_hf",
                    "pt_xglm_facebook_xglm_1_7b_clm_hf",
                    "pt_xglm_facebook_xglm_564m_clm_hf",
                ],
                "pcc": 0.99,
                "op_params": {"repeats": "1", "dim": "0"},
            },
        ),
        marks=[
            pytest.mark.xfail(
                reason="TypeError: Dtype mismatch: framework_model.dtype=torch.int64, compiled_model.dtype=torch.int32"
            )
        ],
    ),
    pytest.param(
        (
            Repeatinterleave0,
            [((1, 1, 1, 256), torch.int64)],
            {
                "model_name": [
                    "pt_bart_facebook_bart_large_mnli_seq_cls_hf",
                    "pt_opt_facebook_opt_1_3b_clm_hf",
                    "pt_opt_facebook_opt_125m_clm_hf",
                    "pt_opt_facebook_opt_350m_clm_hf",
                    "pt_xglm_facebook_xglm_1_7b_clm_hf",
                    "pt_xglm_facebook_xglm_564m_clm_hf",
                ],
                "pcc": 0.99,
                "op_params": {"repeats": "1", "dim": "1"},
            },
        ),
        marks=[
            pytest.mark.xfail(
                reason="TypeError: Dtype mismatch: framework_model.dtype=torch.int64, compiled_model.dtype=torch.int32"
            )
        ],
    ),
    pytest.param(
        (
            Repeatinterleave4,
            [((1, 1, 1, 256), torch.int64)],
            {
                "model_name": [
                    "pt_bart_facebook_bart_large_mnli_seq_cls_hf",
                    "pt_opt_facebook_opt_1_3b_clm_hf",
                    "pt_opt_facebook_opt_125m_clm_hf",
                    "pt_opt_facebook_opt_350m_clm_hf",
                    "pt_xglm_facebook_xglm_1_7b_clm_hf",
                    "pt_xglm_facebook_xglm_564m_clm_hf",
                ],
                "pcc": 0.99,
                "op_params": {"repeats": "256", "dim": "2"},
            },
        ),
        marks=[
            pytest.mark.xfail(
                reason="RuntimeError: TT_THROW @ /__w/tt-forge-fe/tt-forge-fe/third_party/tt-mlir/third_party/tt-metal/src/tt-metal/tt_metal/impl/kernels/kernel.cpp:242: tt::exception info: 1283 unique+common runtime args targeting kernel reader_concat_stick_layout_interleaved_start_id on (x=0,y=0) are too large. Max allowable is 256"
            )
        ],
    ),
    pytest.param(
        (
            Repeatinterleave3,
            [((1, 384), torch.int64)],
            {
                "model_name": ["pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf"],
                "pcc": 0.99,
                "op_params": {"repeats": "1", "dim": "0"},
            },
        ),
        marks=[
            pytest.mark.xfail(
                reason="TypeError: Dtype mismatch: framework_model.dtype=torch.int64, compiled_model.dtype=torch.int32"
            )
        ],
    ),
    (
        Repeatinterleave3,
        [((1, 32, 1), torch.float32)],
        {
            "model_name": [
                "pt_falcon_tiiuae_falcon_7b_instruct_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_seq_cls_hf",
                "pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf",
                "pt_qwen1_5_qwen_qwen1_5_0_5b_chat_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf",
                "pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf",
                "pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave1,
        [((1, 32, 1), torch.float32)],
        {
            "model_name": [
                "pt_falcon_tiiuae_falcon_7b_instruct_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_seq_cls_hf",
                "pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf",
                "pt_qwen1_5_qwen_qwen1_5_0_5b_chat_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf",
                "pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf",
                "pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"repeats": "1", "dim": "2"},
        },
    ),
    (
        Repeatinterleave3,
        [((1, 128, 1), torch.float32)],
        {
            "model_name": [
                "pt_falcon3_tiiuae_falcon3_3b_base_clm_hf",
                "pt_falcon3_tiiuae_falcon3_1b_base_clm_hf",
                "pt_falcon3_tiiuae_falcon3_7b_base_clm_hf",
                "pt_gemma_google_gemma_2b_text_gen_hf",
            ],
            "pcc": 0.99,
            "op_params": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave1,
        [((1, 128, 1), torch.float32)],
        {
            "model_name": [
                "pt_falcon3_tiiuae_falcon3_3b_base_clm_hf",
                "pt_falcon3_tiiuae_falcon3_1b_base_clm_hf",
                "pt_falcon3_tiiuae_falcon3_7b_base_clm_hf",
                "pt_gemma_google_gemma_2b_text_gen_hf",
            ],
            "pcc": 0.99,
            "op_params": {"repeats": "1", "dim": "2"},
        },
    ),
    (
        Repeatinterleave3,
        [((1, 4, 1, 10, 256), torch.float32)],
        {
            "model_name": [
                "pt_falcon3_tiiuae_falcon3_3b_base_clm_hf",
                "pt_falcon3_tiiuae_falcon3_1b_base_clm_hf",
                "pt_falcon3_tiiuae_falcon3_7b_base_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave5,
        [((1, 4, 1, 10, 256), torch.float32)],
        {
            "model_name": ["pt_falcon3_tiiuae_falcon3_3b_base_clm_hf", "pt_falcon3_tiiuae_falcon3_7b_base_clm_hf"],
            "pcc": 0.99,
            "op_params": {"repeats": "3", "dim": "2"},
        },
    ),
    (
        Repeatinterleave6,
        [((1, 4, 1, 10, 256), torch.float32)],
        {
            "model_name": ["pt_falcon3_tiiuae_falcon3_1b_base_clm_hf"],
            "pcc": 0.99,
            "op_params": {"repeats": "2", "dim": "2"},
        },
    ),
    (
        Repeatinterleave3,
        [((1, 16, 1), torch.float32)],
        {
            "model_name": [
                "pt_fuyu_adept_fuyu_8b_qa_hf",
                "pt_phi2_microsoft_phi_2_pytdml_token_cls_hf",
                "pt_phi2_microsoft_phi_2_clm_hf",
                "pt_phi2_microsoft_phi_2_token_cls_hf",
                "pt_phi2_microsoft_phi_2_seq_cls_hf",
                "pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf",
                "pt_phi2_microsoft_phi_2_pytdml_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave1,
        [((1, 16, 1), torch.float32)],
        {
            "model_name": [
                "pt_fuyu_adept_fuyu_8b_qa_hf",
                "pt_phi2_microsoft_phi_2_pytdml_token_cls_hf",
                "pt_phi2_microsoft_phi_2_clm_hf",
                "pt_phi2_microsoft_phi_2_token_cls_hf",
                "pt_phi2_microsoft_phi_2_seq_cls_hf",
                "pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf",
                "pt_phi2_microsoft_phi_2_pytdml_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"repeats": "1", "dim": "2"},
        },
    ),
    (
        Repeatinterleave3,
        [((1, 1, 1, 7, 256), torch.float32)],
        {
            "model_name": ["pt_gemma_google_gemma_2b_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave0,
        [((1, 1, 1, 7, 256), torch.float32)],
        {
            "model_name": ["pt_gemma_google_gemma_2b_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"repeats": "1", "dim": "1"},
        },
    ),
    (
        Repeatinterleave7,
        [((1, 1, 1, 7, 256), torch.float32)],
        {
            "model_name": ["pt_gemma_google_gemma_2b_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"repeats": "8", "dim": "2"},
        },
    ),
    (
        Repeatinterleave3,
        [((1, 8, 1, 4, 64), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_seq_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave8,
        [((1, 8, 1, 4, 64), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_seq_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"repeats": "4", "dim": "2"},
        },
    ),
    (
        Repeatinterleave3,
        [((1, 8, 1, 256, 64), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave8,
        [((1, 8, 1, 256, 64), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"repeats": "4", "dim": "2"},
        },
    ),
    (
        Repeatinterleave3,
        [((1, 8, 1, 4, 128), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave8,
        [((1, 8, 1, 4, 128), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"repeats": "4", "dim": "2"},
        },
    ),
    (
        Repeatinterleave3,
        [((1, 8, 1, 128, 128), torch.float32)],
        {
            "model_name": ["pt_mistral_mistralai_mistral_7b_v0_1_clm_hf"],
            "pcc": 0.99,
            "op_params": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave8,
        [((1, 8, 1, 128, 128), torch.float32)],
        {
            "model_name": ["pt_mistral_mistralai_mistral_7b_v0_1_clm_hf"],
            "pcc": 0.99,
            "op_params": {"repeats": "4", "dim": "2"},
        },
    ),
    pytest.param(
        (
            Repeatinterleave3,
            [((1, 1, 1, 32), torch.int64)],
            {
                "model_name": [
                    "pt_opt_facebook_opt_1_3b_seq_cls_hf",
                    "pt_opt_facebook_opt_1_3b_qa_hf",
                    "pt_opt_facebook_opt_350m_qa_hf",
                    "pt_opt_facebook_opt_125m_seq_cls_hf",
                    "pt_opt_facebook_opt_350m_seq_cls_hf",
                    "pt_opt_facebook_opt_125m_qa_hf",
                ],
                "pcc": 0.99,
                "op_params": {"repeats": "1", "dim": "0"},
            },
        ),
        marks=[
            pytest.mark.xfail(
                reason="TypeError: Dtype mismatch: framework_model.dtype=torch.int64, compiled_model.dtype=torch.int32"
            )
        ],
    ),
    pytest.param(
        (
            Repeatinterleave0,
            [((1, 1, 1, 32), torch.int64)],
            {
                "model_name": [
                    "pt_opt_facebook_opt_1_3b_seq_cls_hf",
                    "pt_opt_facebook_opt_1_3b_qa_hf",
                    "pt_opt_facebook_opt_350m_qa_hf",
                    "pt_opt_facebook_opt_125m_seq_cls_hf",
                    "pt_opt_facebook_opt_350m_seq_cls_hf",
                    "pt_opt_facebook_opt_125m_qa_hf",
                ],
                "pcc": 0.99,
                "op_params": {"repeats": "1", "dim": "1"},
            },
        ),
        marks=[
            pytest.mark.xfail(
                reason="TypeError: Dtype mismatch: framework_model.dtype=torch.int64, compiled_model.dtype=torch.int32"
            )
        ],
    ),
    pytest.param(
        (
            Repeatinterleave9,
            [((1, 1, 1, 32), torch.int64)],
            {
                "model_name": [
                    "pt_opt_facebook_opt_1_3b_seq_cls_hf",
                    "pt_opt_facebook_opt_1_3b_qa_hf",
                    "pt_opt_facebook_opt_350m_qa_hf",
                    "pt_opt_facebook_opt_125m_seq_cls_hf",
                    "pt_opt_facebook_opt_350m_seq_cls_hf",
                    "pt_opt_facebook_opt_125m_qa_hf",
                ],
                "pcc": 0.99,
                "op_params": {"repeats": "32", "dim": "2"},
            },
        ),
        marks=[
            pytest.mark.xfail(
                reason="TypeError: Dtype mismatch: framework_model.dtype=torch.int64, compiled_model.dtype=torch.int32"
            )
        ],
    ),
    (
        Repeatinterleave3,
        [((1, 48, 1), torch.float32)],
        {
            "model_name": [
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave1,
        [((1, 48, 1), torch.float32)],
        {
            "model_name": [
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"repeats": "1", "dim": "2"},
        },
    ),
    (
        Repeatinterleave3,
        [((1, 4, 1, 35, 128), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave2,
        [((1, 4, 1, 35, 128), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"repeats": "7", "dim": "2"},
        },
    ),
    (
        Repeatinterleave3,
        [((1, 2, 1, 35, 128), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave10,
        [((1, 2, 1, 35, 128), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"repeats": "6", "dim": "2"},
        },
    ),
    (
        Repeatinterleave7,
        [((1, 2, 1, 35, 128), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"repeats": "8", "dim": "2"},
        },
    ),
    (
        Repeatinterleave3,
        [((1, 2, 1, 35, 64), torch.float32)],
        {
            "model_name": ["pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave2,
        [((1, 2, 1, 35, 64), torch.float32)],
        {
            "model_name": ["pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"repeats": "7", "dim": "2"},
        },
    ),
    (
        Repeatinterleave3,
        [((1, 2, 1, 29, 128), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf", "pt_qwen_v2_qwen_qwen2_5_3b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave10,
        [((1, 2, 1, 29, 128), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99, "op_params": {"repeats": "6", "dim": "2"}},
    ),
    (
        Repeatinterleave7,
        [((1, 2, 1, 29, 128), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_3b_clm_hf"], "pcc": 0.99, "op_params": {"repeats": "8", "dim": "2"}},
    ),
    (
        Repeatinterleave3,
        [((1, 2, 1, 39, 128), torch.float32)],
        {
            "model_name": [
                "pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf",
                "pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave10,
        [((1, 2, 1, 39, 128), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"repeats": "6", "dim": "2"},
        },
    ),
    (
        Repeatinterleave7,
        [((1, 2, 1, 39, 128), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"repeats": "8", "dim": "2"},
        },
    ),
    (
        Repeatinterleave3,
        [((1, 4, 1, 39, 128), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave2,
        [((1, 4, 1, 39, 128), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"repeats": "7", "dim": "2"},
        },
    ),
    (
        Repeatinterleave3,
        [((1, 4, 1, 29, 128), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_7b_clm_hf"], "pcc": 0.99, "op_params": {"repeats": "1", "dim": "0"}},
    ),
    (
        Repeatinterleave2,
        [((1, 4, 1, 29, 128), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_7b_clm_hf"], "pcc": 0.99, "op_params": {"repeats": "7", "dim": "2"}},
    ),
    (
        Repeatinterleave3,
        [((1, 2, 1, 29, 64), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"], "pcc": 0.99, "op_params": {"repeats": "1", "dim": "0"}},
    ),
    (
        Repeatinterleave2,
        [((1, 2, 1, 29, 64), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"], "pcc": 0.99, "op_params": {"repeats": "7", "dim": "2"}},
    ),
    (
        Repeatinterleave3,
        [((1, 2, 1, 39, 64), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave2,
        [((1, 2, 1, 39, 64), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"repeats": "7", "dim": "2"},
        },
    ),
    (
        Repeatinterleave3,
        [((1, 1, 768), torch.float32)],
        {
            "model_name": [
                "pt_deit_facebook_deit_base_patch16_224_img_cls_hf",
                "pt_deit_facebook_deit_base_distilled_patch16_224_img_cls_hf",
                "pt_vit_google_vit_base_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave3,
        [((1, 1, 192), torch.float32)],
        {
            "model_name": ["pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave3,
        [((1, 1, 384), torch.float32)],
        {
            "model_name": ["pt_deit_facebook_deit_small_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave3,
        [((1, 1, 1024), torch.float32)],
        {
            "model_name": [
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave3,
        [((1, 512, 1024), torch.float32)],
        {
            "model_name": [
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave3,
        [((1, 50176, 256), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave3,
        [((1, 1, 1024), torch.float32)],
        {
            "model_name": ["pt_vit_google_vit_large_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"repeats": "1", "dim": "0"},
        },
    ),
]


@pytest.mark.nightly_models_ops
@pytest.mark.parametrize("forge_module_and_shapes_dtypes", forge_modules_and_shapes_dtypes_list, ids=ids_func)
def test_module(forge_module_and_shapes_dtypes, record_forge_property):
    record_forge_property("op_name", "RepeatInterleave")

    forge_module, operand_shapes_dtypes, metadata = forge_module_and_shapes_dtypes

    pcc = metadata.pop("pcc")

    for metadata_name, metadata_value in metadata.items():
        record_forge_property(metadata_name, metadata_value)

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

    compiled_model = compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model, VerifyConfig(value_checker=AutomaticValueChecker(pcc=pcc)))
